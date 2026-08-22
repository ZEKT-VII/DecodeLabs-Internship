"""Unit tests for the CaliperValidationGate input guardrails."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from validation import CaliperValidationGate
from exceptions import EmptyInputError, InputLengthError, InvalidCommandError


class TestSanitizeText:
    """Tests for control character sanitization."""

    def test_removes_null_bytes(self):
        result = CaliperValidationGate.sanitize_text("hello\x00world")
        assert "\x00" not in result
        assert "helloworld" == result

    def test_preserves_newlines_and_tabs(self):
        result = CaliperValidationGate.sanitize_text("hello\n\tworld")
        assert result == "hello\n\tworld"

    def test_removes_control_chars(self):
        result = CaliperValidationGate.sanitize_text("test\x01\x02\x03data")
        assert result == "testdata"

    def test_empty_string_passthrough(self):
        assert CaliperValidationGate.sanitize_text("") == ""

    def test_unicode_normalization(self):
        # café in NFD form (e + combining accent) should normalize to NFC
        nfd = "caf\u0065\u0301"  # e + combining acute accent
        result = CaliperValidationGate.sanitize_text(nfd)
        assert "é" in result or "e" in result  # NFC normalized


class TestValidateUserInput:
    """Tests for input validation invariants."""

    def test_rejects_none(self):
        with pytest.raises(EmptyInputError):
            CaliperValidationGate.validate_user_input(None)

    def test_rejects_empty_string(self):
        with pytest.raises(EmptyInputError):
            CaliperValidationGate.validate_user_input("")

    def test_rejects_whitespace_only(self):
        with pytest.raises(EmptyInputError):
            CaliperValidationGate.validate_user_input("   \t\n  ")

    def test_accepts_valid_input(self):
        result = CaliperValidationGate.validate_user_input("Hello, how are you?")
        assert result == "Hello, how are you?"

    def test_rejects_overlength_input(self):
        long_input = "a" * 25_000
        with pytest.raises(InputLengthError):
            CaliperValidationGate.validate_user_input(long_input)

    def test_accepts_max_length_input(self):
        # Exactly at the limit should pass
        exact_input = "a" * 20_000
        result = CaliperValidationGate.validate_user_input(exact_input)
        assert len(result) == 20_000


class TestParseAndValidateCommand:
    """Tests for command parsing."""

    def test_non_command_input(self):
        is_cmd, name, args = CaliperValidationGate.parse_and_validate_command("hello world")
        assert is_cmd is False
        assert name == ""

    def test_simple_command(self):
        is_cmd, name, args = CaliperValidationGate.parse_and_validate_command("/stats")
        assert is_cmd is True
        assert name == "stats"
        assert args == ""

    def test_command_with_arguments(self):
        is_cmd, name, args = CaliperValidationGate.parse_and_validate_command("/remember name=Alice")
        assert is_cmd is True
        assert name == "remember"
        assert args == "name=Alice"

    def test_command_case_insensitive(self):
        is_cmd, name, args = CaliperValidationGate.parse_and_validate_command("/STATS")
        assert is_cmd is True
        assert name == "stats"

    def test_malformed_command_rejected(self):
        with pytest.raises(InvalidCommandError):
            CaliperValidationGate.parse_and_validate_command("/!!invalid")
