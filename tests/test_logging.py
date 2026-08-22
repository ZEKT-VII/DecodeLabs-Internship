"""Unit tests for structured logging and SecretScrubbingFilter."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import logging
import pytest
from logging_config import SecretScrubbingFilter, setup_logger


class TestSecretScrubbingFilter:
    @pytest.fixture
    def filter_instance(self):
        return SecretScrubbingFilter()

    def test_scrubs_nvapi_key_in_message(self, filter_instance):
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Connecting using key nvapi-dummytestkey00000000000000000000000000000000000000000000000000000 to host",
            args=(),
            exc_info=None,
        )
        assert filter_instance.filter(record) is True
        assert "nvapi-" not in record.msg
        assert "[REDACTED_API_KEY]" in record.msg

    def test_scrubs_bearer_token(self, filter_instance):
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Header Authorization: Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9",
            args=(),
            exc_info=None,
        )
        assert filter_instance.filter(record) is True
        assert "[REDACTED_API_KEY]" in record.msg

    def test_scrubs_sk_key(self, filter_instance):
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Token sk-123456789012345678901234567890 passed",
            args=(),
            exc_info=None,
        )
        assert filter_instance.filter(record) is True
        assert "[REDACTED_API_KEY]" in record.msg

    def test_scrubs_args_tuple(self, filter_instance):
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Request with key %s and user %s",
            args=("nvapi-dummytestkey00000000000000000000000000000000000000000000000000000", "admin"),
            exc_info=None,
        )
        assert filter_instance.filter(record) is True
        assert "nvapi-" not in record.args[0]
        assert "[REDACTED_API_KEY]" in record.args[0]
        assert record.args[1] == "admin"

    def test_scrubs_args_dict(self, filter_instance):
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Payload info",
            args={"token": "nvapi-dummytestkey00000000000000000000000000000000000000000000000000000", "status": "ok"},
            exc_info=None,
        )
        assert filter_instance.filter(record) is True
        assert "nvapi-" not in record.args["token"]
        assert "[REDACTED_API_KEY]" in record.args["token"]
        assert record.args["status"] == "ok"

    def test_clean_message_unaltered(self, filter_instance):
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Standard informative log with no secret tokens",
            args=(),
            exc_info=None,
        )
        assert filter_instance.filter(record) is True
        assert record.msg == "Standard informative log with no secret tokens"


class TestSetupLogger:
    def test_setup_logger_adds_scrubbing_filter(self):
        custom_logger = setup_logger("test_custom_logger", level=logging.DEBUG)
        assert len(custom_logger.handlers) > 0
        has_filter = any(
            any(isinstance(f, SecretScrubbingFilter) for f in h.filters)
            for h in custom_logger.handlers
        )
        assert has_filter is True
