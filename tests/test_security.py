"""Unit tests for the security and credential management module."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from security import mask_api_key, redact_secrets


class TestMaskApiKey:
    """Tests for API key masking."""

    def test_mask_normal_key(self):
        key = "nvapi-dummytestkey00000000000000000000000000000000000000000000000000000"
        masked = mask_api_key(key)
        assert masked.startswith("nvapi-")
        assert masked.endswith("0000")
        assert "..." in masked
        # Full key should NOT be present
        assert key not in masked

    def test_mask_none(self):
        assert mask_api_key(None) == "[NO_KEY_CONFIGURED]"

    def test_mask_empty(self):
        assert mask_api_key("") == "[NO_KEY_CONFIGURED]"

    def test_mask_short_key(self):
        assert mask_api_key("short") == "******"

    def test_mask_exactly_12_chars(self):
        assert mask_api_key("123456789012") == "******"


class TestRedactSecrets:
    """Tests for secret pattern redaction in text."""

    def test_redact_nvapi_key(self):
        text = "Using key nvapi-dummytestkey00000000000000000000000000000000000000000000000000000 for auth"
        result = redact_secrets(text)
        assert "nvapi-" not in result
        assert "[REDACTED_API_KEY]" in result

    def test_redact_bearer_token(self):
        text = "Authorization: Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9"
        result = redact_secrets(text)
        assert "[REDACTED_API_KEY]" in result

    def test_redact_sk_key(self):
        text = "Key is sk-abc123def456ghi789jkl012mno345pqr678"
        result = redact_secrets(text)
        assert "sk-abc" not in result
        assert "[REDACTED_API_KEY]" in result

    def test_no_redaction_needed(self):
        text = "This is a clean message with no secrets."
        result = redact_secrets(text)
        assert result == text

    def test_empty_text(self):
        assert redact_secrets("") == ""

    def test_none_text(self):
        assert redact_secrets(None) is None
