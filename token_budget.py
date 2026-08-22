"""Conservative token budget estimator and context headroom tracker."""

from __future__ import annotations

from typing import List, Dict

from config import (
    DEFAULT_MAX_CONTEXT_TOKENS,
    DEFAULT_RESERVED_OUTPUT_TOKENS,
    TOKEN_SAFETY_MARGIN,
)


# Conservative heuristic: 1 token ≈ 3.0 characters (deliberately conservative)
CHARS_PER_TOKEN: float = 3.0


def estimate_tokens(text: str) -> int:
    """
    Estimates token count for a string using conservative char-based heuristic.
    Formula: ceil(len(text) / CHARS_PER_TOKEN)
    """
    if not text:
        return 0
    return max(1, int(len(text) / CHARS_PER_TOKEN + 0.999))


def estimate_message_tokens(message: Dict[str, str]) -> int:
    """
    Estimates the token footprint of a single API message dict.
    Adds overhead for role label and message framing (~4 tokens per message).
    """
    content_tokens = estimate_tokens(message.get("content", ""))
    role_overhead = 4  # role label + message framing
    return content_tokens + role_overhead


def compute_context_budget(
    max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
    reserved_output_tokens: int = DEFAULT_RESERVED_OUTPUT_TOKENS,
    safety_margin: int = TOKEN_SAFETY_MARGIN,
) -> int:
    """
    Computes the available token budget for input (prompt) messages.
    Budget = max_context - reserved_output - safety_margin
    """
    budget = max_context_tokens - reserved_output_tokens - safety_margin
    return max(0, budget)


def calculate_messages_tokens(messages: List[Dict[str, str]]) -> int:
    """Totals estimated token count across a list of message dicts."""
    total = sum(estimate_message_tokens(m) for m in messages)
    # Add base overhead for the conversation framing (~3 tokens)
    return total + 3


def fits_within_budget(
    messages: List[Dict[str, str]],
    max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
    reserved_output_tokens: int = DEFAULT_RESERVED_OUTPUT_TOKENS,
) -> bool:
    """Checks whether a message list fits within the available input token budget."""
    budget = compute_context_budget(max_context_tokens, reserved_output_tokens)
    usage = calculate_messages_tokens(messages)
    return usage <= budget


def headroom_remaining(
    messages: List[Dict[str, str]],
    max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
    reserved_output_tokens: int = DEFAULT_RESERVED_OUTPUT_TOKENS,
) -> int:
    """Returns how many estimated tokens of headroom remain for additional messages."""
    budget = compute_context_budget(max_context_tokens, reserved_output_tokens)
    usage = calculate_messages_tokens(messages)
    return max(0, budget - usage)
