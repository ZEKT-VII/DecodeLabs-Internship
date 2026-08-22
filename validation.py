"""Input guardrails and validation gatekeepers for the Stateful Chatbot Agent."""

from __future__ import annotations

import re
import unicodedata
from typing import Tuple

from config import MAX_INPUT_CHARS
from exceptions import EmptyInputError, InputLengthError, InvalidCommandError

# Filter regex for illegal control characters (preserves standard \t, \n, \r)
CONTROL_CHAR_REGEX = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")

# Valid command regex: e.g. /clear, /stats, /model <name>, /remember key=val
COMMAND_PATTERN = re.compile(r"^/([a-zA-Z0-9_-]+)(?:\s+(.*))?$")


class CaliperValidationGate:
    """
    The Structural Validation Gate:
    Enforces input sanitation, length bounds, and empty-payload assertions
    to eliminate 400 Bad Request errors and terminal crashes.
    """

    @classmethod
    def sanitize_text(cls, raw_text: str) -> str:
        """Strips harmful terminal control characters while preserving legitimate multiline input."""
        if not raw_text:
            return ""
        # Unicode normalization (NFC)
        normalized = unicodedata.normalize("NFC", raw_text)
        # Remove harmful non-printable control characters
        sanitized = CONTROL_CHAR_REGEX.sub("", normalized)
        return sanitized

    @classmethod
    def validate_user_input(cls, raw_text: str) -> str:
        """
        Validates user conversational input:
        1. Checks for null/empty/whitespace-only input.
        2. Strips harmful control characters.
        3. Enforces MAX_INPUT_CHARS bounds.
        """
        if raw_text is None:
            raise EmptyInputError("Input payload cannot be None.")

        sanitized = cls.sanitize_text(raw_text)
        stripped = sanitized.strip()

        if not stripped:
            raise EmptyInputError(
                "Empty or whitespace-only input rejected. "
                "Conversational turn requires non-empty content to prevent API 400 errors."
            )

        if len(sanitized) > MAX_INPUT_CHARS:
            raise InputLengthError(
                f"Input length ({len(sanitized):,} chars) exceeds maximum allowed limit "
                f"of {MAX_INPUT_CHARS:,} characters."
            )

        return sanitized

    @classmethod
    def parse_and_validate_command(cls, raw_text: str) -> Tuple[bool, str, str]:
        """
        Parses CLI commands (e.g. /stats, /model name).
        Returns (is_command, command_name, argument_string).
        """
        sanitized = cls.sanitize_text(raw_text).strip()
        if not sanitized.startswith("/"):
            return False, "", ""

        match = COMMAND_PATTERN.match(sanitized)
        if not match:
            raise InvalidCommandError(
                f"Malformed command syntax: '{sanitized}'. "
                "Commands must start with '/' followed by alphanumeric characters."
            )

        cmd_name = match.group(1).lower()
        cmd_args = match.group(2) or ""
        return True, cmd_name, cmd_args.strip()
