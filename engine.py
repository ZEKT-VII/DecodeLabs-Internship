"""Multi-turn Conversation Engine: orchestrates memory, LLM, persistence, and validation."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from config import DEFAULT_MODEL, DEFAULT_SYSTEM_PROMPT
from exceptions import ChatbotError, ContextOverflowError
from llm_client import LLMClient
from memory import MemoryManager
from persistence import SessionStore
from schemas import ChatResponse, MessageRole
from security import resolve_api_key, resolve_api_config
from token_budget import estimate_tokens
from validation import CaliperValidationGate

logger = logging.getLogger("stateful_chatbot.engine")


class ConversationEngine:
    """
    Orchestrates the stateful conversation loop:
    
    Input(M_t ∪ H_{t-1}) → Output(R_t)
    
    Where:
        M_t   = current user message
        H_t-1 = conversation history (priority-pruned)
        R_t   = assistant response
    """

    def __init__(
        self,
        model: Optional[str] = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        config = resolve_api_config()
        self.model = model or config["model"] or DEFAULT_MODEL
        self.base_url = base_url or config["base_url"]
        self._api_key = api_key or (config["api_key"] if config.get("api_key") else None)

        # Core subsystems
        self.persistence = SessionStore()
        self.memory = MemoryManager(system_prompt=system_prompt)
        self.reload_global_facts()
        self.llm = LLMClient(base_url=self.base_url, model=self.model, api_key=self._api_key)

        # Session state
        self.session_id: Optional[str] = None
        self._created_at: Optional[datetime] = None

        logger.info("ConversationEngine initialized with model=%s on base_url=%s", self.model, self.base_url)

    def configure(
        self,
        provider: str,
        base_url: str,
        model: str,
        api_key: Optional[str] = None,
    ) -> None:
        """Dynamically reconfigures API settings across subsystems."""
        self.model = model.strip()
        self.base_url = base_url.strip()
        if api_key:
            self._api_key = api_key.strip()
        self.llm.configure(base_url=self.base_url, model=self.model, api_key=self._api_key)
        logger.info("ConversationEngine reconfigured: provider=%s, model=%s", provider, self.model)

    def start_session(self) -> str:
        """Begins a new conversation session with persistence."""
        self.session_id = self.persistence.create_session(model=self.model)
        self._created_at = datetime.now(timezone.utc)
        self.memory.clear_rolling_window()
        logger.info("Session started: %s", self.session_id)
        return self.session_id

    def load_session(self, session_id: str) -> bool:
        """
        Resumes an existing persisted session from SQLite.
        Restores conversational history into the memory manager.
        """
        session = self.persistence.get_session(session_id)
        if not session:
            return False

        self.session_id = session_id
        if session.get("created_at"):
            try:
                self._created_at = datetime.fromisoformat(session["created_at"])
            except Exception:
                self._created_at = datetime.now(timezone.utc)

        # Clear and restore in-memory rolling window from SQLite history
        self.memory.clear_rolling_window()
        messages = self.persistence.get_session_messages(session_id)
        for msg in messages:
            role_val = msg["role"]
            if role_val == "user":
                self.memory.add_message(MessageRole.USER, msg["content"])
            elif role_val == "assistant":
                self.memory.add_message(MessageRole.ASSISTANT, msg["content"])

        # Restore pinned facts from session metadata
        pinned = session.get("metadata", {}).get("pinned_facts", {})
        for k, v in pinned.items():
            self.memory.add_preference(k, v)

        logger.info("Resumed session %s with %d messages and %d pinned facts loaded.", session_id, len(messages), len(pinned))
        return True

    def send_message(self, user_input: str) -> ChatResponse:
        """
        Processes a user message through the full pipeline:
        1. Validate input via CaliperValidationGate
        2. Add to memory (P4 rolling window)
        3. Build context (P1-P4 assembled)
        4. Send to LLM with retry
        5. Add assistant response to memory
        6. Persist both messages to SQLite
        7. Return structured response
        """
        if not self.session_id:
            self.start_session()

        # Step 1: Validate
        validated_input = CaliperValidationGate.validate_user_input(user_input)

        # Step 1.5: Sync global memory facts from persistence
        self.reload_global_facts()

        # Step 2: Add user message to memory
        self.memory.add_message(MessageRole.USER, validated_input)

        # Step 3: Build context
        context = self.memory.build_context()

        # Guard: ensure context is non-empty beyond system prompt
        if len(context) < 2:
            raise ContextOverflowError(
                "Context assembly failed: no messages remain after pruning."
            )

        # Step 4: Send to LLM
        try:
            response = self.llm.chat(messages=context)
        except ChatbotError:
            # On failure, remove the user message from memory to keep state consistent
            if self.memory._rolling_window and self.memory._rolling_window[-1].role == MessageRole.USER:
                self.memory._rolling_window.pop()
            raise

        # Step 5: Add assistant response to memory
        self.memory.add_message(MessageRole.ASSISTANT, response.content)

        # Step 6: Persist
        user_tokens = estimate_tokens(validated_input)
        assistant_tokens = response.usage.completion_tokens or estimate_tokens(response.content)

        self.persistence.save_message(
            session_id=self.session_id,
            role="user",
            content=validated_input,
            token_count=user_tokens,
        )
        self.persistence.save_message(
            session_id=self.session_id,
            role="assistant",
            content=response.content,
            token_count=assistant_tokens,
        )

        logger.info(
            "Turn completed: user=%d tokens, assistant=%d tokens",
            user_tokens, assistant_tokens,
        )

        return response

    def stream_message(self, user_input: str):
        """
        Processes a user message in streaming mode:
        1. Validates input via CaliperValidationGate
        2. Reloads global facts from persistence
        3. Adds user message to memory (P4)
        4. Builds prioritized P1-P4 context
        5. Calls llm.stream_chat(context) and yields tokens in real time
        6. On completion, records assistant message in memory and SQLite
        7. Yields completion metadata
        """
        if not self.session_id:
            self.start_session()

        # Step 1: Validate
        validated_input = CaliperValidationGate.validate_user_input(user_input)

        # Step 1.5: Sync global memory facts from persistence
        self.reload_global_facts()

        # Step 2: Add user message to memory
        self.memory.add_message(MessageRole.USER, validated_input)

        # Step 3: Build context
        context = self.memory.build_context()

        if len(context) < 2:
            raise ContextOverflowError("Context assembly failed: no messages remain after pruning.")

        start_time = time.time()
        accumulated_chunks: List[str] = []

        try:
            for token_chunk in self.llm.stream_chat(messages=context):
                accumulated_chunks.append(token_chunk)
                yield {"type": "token", "token": token_chunk}
        except Exception as e:
            if not accumulated_chunks and self.memory._rolling_window and self.memory._rolling_window[-1].role == MessageRole.USER:
                self.memory._rolling_window.pop()
            raise

        full_content = "".join(accumulated_chunks)
        latency = time.time() - start_time

        # Step 5: Add assistant response to memory
        self.memory.add_message(MessageRole.ASSISTANT, full_content)

        # Step 6: Persist
        user_tokens = estimate_tokens(validated_input)
        assistant_tokens = estimate_tokens(full_content)

        self.persistence.save_message(
            session_id=self.session_id,
            role="user",
            content=validated_input,
            token_count=user_tokens,
        )
        self.persistence.save_message(
            session_id=self.session_id,
            role="assistant",
            content=full_content,
            token_count=assistant_tokens,
        )

        yield {
            "type": "done",
            "content": full_content,
            "latency_sec": latency,
            "model": self.model,
            "session_id": self.session_id,
            "stats": self.get_stats(),
            "local_facts": self.get_local_facts(),
        }

    # ──────────────────────────── Session Management ──────────────────────────

    def clear_memory(self) -> None:
        """Clears in-memory conversation state (P4 rolling window) without affecting SQLite."""
        self.memory.clear_rolling_window()
        logger.info("In-memory conversation cleared for session %s", self.session_id)

    def delete_session(self) -> bool:
        """Deletes the current session from SQLite and resets memory."""
        if not self.session_id:
            return False
        deleted = self.persistence.delete_session(self.session_id)
        self.memory.reset()
        self.session_id = None
        return deleted

    def delete_all_sessions(self) -> int:
        """Purges all sessions from the database."""
        count = self.persistence.delete_all_sessions()
        self.memory.reset()
        self.session_id = None
        return count

    # ──────────────────────────── Global Memory ──────────────────────────

    def reload_global_facts(self) -> None:
        """Loads and synchronizes global memory facts from persistence."""
        global_facts = self.persistence.get_all_global_facts()
        self.memory.set_global_facts(global_facts)

    def set_global_fact(self, key: str, value: str) -> None:
        """Sets a global memory fact in persistence and active memory."""
        self.persistence.set_global_fact(key, value)
        self.reload_global_facts()

    def delete_global_fact(self, key: str) -> bool:
        """Deletes a global memory fact from persistence and active memory."""
        deleted = self.persistence.delete_global_fact(key)
        self.reload_global_facts()
        return deleted

    def get_global_facts(self) -> Dict[str, str]:
        """Returns all global memory facts."""
        return self.persistence.get_all_global_facts()

    # ──────────────────────────── Session Pinned Facts ──────────────────────────

    def remember_fact(self, key: str, value: str) -> None:
        """Pins a user preference to P3 memory and persists to session metadata."""
        self.memory.add_preference(key, value)
        if self.session_id:
            self.persistence.set_session_pinned_fact(self.session_id, key, value)

    def forget_fact(self, key: str) -> bool:
        """Removes a user preference from P3 memory and session metadata."""
        removed = self.memory.remove_preference(key)
        if self.session_id:
            self.persistence.delete_session_pinned_fact(self.session_id, key)
        return removed

    def get_local_facts(self) -> Dict[str, str]:
        """Returns all session-specific pinned facts."""
        return self.memory.get_all_preferences()

    def set_user_identity(self, key: str, value: str) -> None:
        """Pins a user identity fact to P2 memory."""
        self.memory.set_identity_fact(key, value)

    # ──────────────────────────── Statistics ──────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """Compiles session statistics from memory and LLM usage."""
        memory_stats = self.memory.get_stats()
        llm_stats = self.llm.get_usage_stats()
        return {
            "session_id": self.session_id,
            "model": self.model,
            "created_at": self._created_at.isoformat() if self._created_at else None,
            **memory_stats,
            **llm_stats,
        }

    def get_history(self) -> List[Dict[str, Any]]:
        """Retrieves persisted message history for the current session."""
        if not self.session_id:
            return []
        return self.persistence.get_session_messages(self.session_id)

    # ──────────────────────────── API Key Management ──────────────────────────

    def update_api_key(self, new_key: str) -> None:
        """Hot-swaps the API key for the current session."""
        self.llm.update_api_key(new_key)
        logger.info("API key updated for active session.")

    # ──────────────────────────── Lifecycle ──────────────────────────

    def shutdown(self) -> None:
        """Gracefully shuts down the engine and closes database connections."""
        self.persistence.close()
        logger.info("ConversationEngine shut down.")
