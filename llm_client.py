"""NVIDIA OpenAI-compatible LLM client with retry logic and usage tracking."""

from __future__ import annotations

import time
import logging
from typing import Any, Dict, List, Optional

from openai import OpenAI, APIError as OpenAIAPIError, APITimeoutError as OpenAITimeoutError

from config import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_MAX_OUTPUT_TOKENS,
    REQUEST_TIMEOUT_SEC,
    RETRYABLE_STATUS_CODES,
)
from exceptions import (
    APIError,
    APIRateLimitError,
    APITimeoutError,
    APIResponseError,
    AuthError,
)
from retry import retry_with_backoff
from schemas import ChatResponse, TokenUsage
from security import resolve_api_key, redact_secrets

logger = logging.getLogger("stateful_chatbot.llm_client")


class LLMClient:
    """
    Wrapper around the OpenAI Python SDK targeting the NVIDIA NIM endpoint.
    
    Features:
    - Transparent retry with exponential backoff on transient failures.
    - Usage tracking (prompt_tokens, completion_tokens, total_tokens).
    - Latency measurement per request.
    - Secret redaction in all log output.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
    ) -> None:
        self.model = model
        self.base_url = base_url

        # Cumulative session-level usage tracking
        self._total_prompt_tokens: int = 0
        self._total_completion_tokens: int = 0
        self._total_requests: int = 0

        # Initialize the client
        resolved_key = api_key or resolve_api_key(interactive_fallback=False)
        self._client = OpenAI(base_url=base_url, api_key=resolved_key)
        logger.info("LLM client initialized for model: %s", model)

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        enable_reasoning: bool = False,
    ) -> ChatResponse:
        """
        Sends a chat completion request with automatic retry on transient failures.
        
        Returns a structured ChatResponse with content, reasoning, usage, and latency.
        """
        return retry_with_backoff(
            self._execute_chat,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            enable_reasoning=enable_reasoning,
        )

    def _execute_chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        top_p: float,
        max_tokens: int,
        enable_reasoning: bool,
    ) -> ChatResponse:
        """Internal: performs the actual API call without retry wrapper."""
        start_time = time.time()

        try:
            extra_body: Optional[Dict[str, Any]] = None
            if enable_reasoning:
                extra_body = {
                    "chat_template_kwargs": {"thinking": True, "reasoning_effort": "high"}
                }

            completion = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                extra_body=extra_body,
                stream=False,
                timeout=REQUEST_TIMEOUT_SEC,
            )

            latency = time.time() - start_time
            choice = completion.choices[0]
            content = choice.message.content or ""

            # Extract reasoning if available
            reasoning = (
                getattr(choice.message, "reasoning", None)
                or getattr(choice.message, "reasoning_content", None)
            )

            # Extract usage
            usage = TokenUsage()
            if completion.usage:
                usage = TokenUsage(
                    prompt_tokens=completion.usage.prompt_tokens or 0,
                    completion_tokens=completion.usage.completion_tokens or 0,
                    total_tokens=completion.usage.total_tokens or 0,
                )

            # Update cumulative tracking
            self._total_prompt_tokens += usage.prompt_tokens
            self._total_completion_tokens += usage.completion_tokens
            self._total_requests += 1

            logger.info(
                "Chat response received in %.2fs (prompt=%d, completion=%d tokens)",
                latency, usage.prompt_tokens, usage.completion_tokens,
            )

            return ChatResponse(
                content=content,
                reasoning=reasoning,
                usage=usage,
                model=self.model,
                latency_sec=latency,
            )

        except OpenAITimeoutError as e:
            raise APITimeoutError(f"Request timed out after {REQUEST_TIMEOUT_SEC}s: {redact_secrets(str(e))}")
        except OpenAIAPIError as e:
            status = getattr(e, "status_code", None) or getattr(e, "http_status", None)
            msg = redact_secrets(str(e))

            if status == 401:
                raise AuthError(f"API authentication failed (401): {msg}")
            elif status == 429:
                raise APIRateLimitError(f"Rate limit exceeded (429): {msg}")
            elif status and status in RETRYABLE_STATUS_CODES:
                raise APIError(msg, status_code=status, retryable=True)
            else:
                raise APIResponseError(f"API error ({status}): {msg}", status_code=status)
        except Exception as e:
            raise APIResponseError(f"Unexpected error: {redact_secrets(str(e))}")

    # ──────────────────────────── Usage Stats ──────────────────────────

    def get_usage_stats(self) -> Dict[str, int]:
        """Returns cumulative token usage statistics for the session."""
        return {
            "total_prompt_tokens": self._total_prompt_tokens,
            "total_completion_tokens": self._total_completion_tokens,
            "total_tokens": self._total_prompt_tokens + self._total_completion_tokens,
            "total_requests": self._total_requests,
        }

    def update_api_key(self, new_key: str) -> None:
        """Hot-swaps the API key without requiring a full client restart."""
        self._client = OpenAI(base_url=self.base_url, api_key=new_key)
        logger.info("API key updated at runtime.")

    def configure(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        """Dynamically reconfigures base_url, model, or API key."""
        if base_url:
            self.base_url = base_url.strip()
        if model:
            self.model = model.strip()
        resolved_key = api_key or resolve_api_key(interactive_fallback=False)
        self._client = OpenAI(base_url=self.base_url, api_key=resolved_key)
        logger.info("LLMClient reconfigured: base_url=%s, model=%s", self.base_url, self.model)

    @staticmethod
    def test_connection(
        base_url: str,
        api_key: str,
        model: str,
    ) -> Dict[str, Any]:
        """Probes the API endpoint with a lightweight 1-token test prompt."""
        start = time.perf_counter()
        try:
            client = OpenAI(base_url=base_url.strip(), api_key=api_key.strip(), timeout=12.0)
            res = client.chat.completions.create(
                model=model.strip(),
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=2,
            )
            elapsed_ms = round((time.perf_counter() - start) * 1000, 1)
            return {
                "success": True,
                "latency_ms": elapsed_ms,
                "model": model,
                "message": f"Connection verified in {elapsed_ms}ms! Endpoint active.",
            }
        except Exception as e:
            elapsed_ms = round((time.perf_counter() - start) * 1000, 1)
            return {
                "success": False,
                "latency_ms": elapsed_ms,
                "error": redact_secrets(str(e)),
            }
