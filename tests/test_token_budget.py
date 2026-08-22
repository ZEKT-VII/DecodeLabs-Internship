"""Unit tests for the token budget estimator."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from token_budget import (
    estimate_tokens,
    estimate_message_tokens,
    compute_context_budget,
    calculate_messages_tokens,
    fits_within_budget,
    headroom_remaining,
    CHARS_PER_TOKEN,
)


class TestEstimateTokens:
    """Tests for character-based token estimation."""

    def test_empty_string(self):
        assert estimate_tokens("") == 0

    def test_short_string(self):
        # "hi" = 2 chars → ceil(2/3.0) = 1
        result = estimate_tokens("hi")
        assert result >= 1

    def test_known_length(self):
        # 300 chars → ceil(300/3.0) = 100 tokens
        text = "a" * 300
        result = estimate_tokens(text)
        assert result == 100

    def test_conservative_estimate(self):
        # Should always overcount rather than undercount
        text = "Hello, this is a test message."
        result = estimate_tokens(text)
        # Actual GPT tokenization would give ~7-8 tokens, ours should be higher
        assert result >= 7


class TestEstimateMessageTokens:
    """Tests for per-message token estimation with framing overhead."""

    def test_message_includes_role_overhead(self):
        msg = {"role": "user", "content": "Hello"}
        tokens = estimate_message_tokens(msg)
        content_tokens = estimate_tokens("Hello")
        assert tokens == content_tokens + 4  # 4 tokens role overhead

    def test_empty_content_message(self):
        msg = {"role": "system", "content": ""}
        tokens = estimate_message_tokens(msg)
        assert tokens == 4  # Only role overhead


class TestComputeContextBudget:
    """Tests for available token budget calculation."""

    def test_default_budget(self):
        budget = compute_context_budget(8192, 1024, 128)
        assert budget == 8192 - 1024 - 128

    def test_zero_safety_margin(self):
        budget = compute_context_budget(4096, 512, 0)
        assert budget == 3584

    def test_budget_never_negative(self):
        budget = compute_context_budget(100, 200, 50)
        assert budget == 0


class TestFitsWithinBudget:
    """Tests for budget compliance checking."""

    def test_small_context_fits(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]
        assert fits_within_budget(messages, max_context_tokens=8192, reserved_output_tokens=1024)

    def test_massive_context_fails(self):
        messages = [
            {"role": "system", "content": "x" * 50000},
        ]
        assert not fits_within_budget(messages, max_context_tokens=1000, reserved_output_tokens=200)


class TestHeadroomRemaining:
    """Tests for headroom tracking."""

    def test_headroom_positive(self):
        messages = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Hello"},
        ]
        remaining = headroom_remaining(messages, max_context_tokens=8192, reserved_output_tokens=1024)
        assert remaining > 0

    def test_headroom_zero_when_full(self):
        messages = [
            {"role": "system", "content": "x" * 30000},
        ]
        remaining = headroom_remaining(messages, max_context_tokens=1000, reserved_output_tokens=200)
        assert remaining == 0
