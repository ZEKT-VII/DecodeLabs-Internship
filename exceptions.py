"""Custom domain exception hierarchy for the Stateful Chatbot Agent."""

from __future__ import annotations
from typing import Optional


class ChatbotError(Exception):
    """Base class for all domain-specific chatbot exceptions."""

    def __init__(self, message: str, details: Optional[dict] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ConfigError(ChatbotError):
    """Raised when system or model configuration fails validation."""
    pass


class AuthError(ChatbotError):
    """Raised when API credentials are missing, invalid, or unauthorized."""
    pass


class ValidationError(ChatbotError):
    """Raised when user input violates guardrail invariants."""
    pass


class EmptyInputError(ValidationError):
    """Raised when user inputs empty or whitespace-only content."""
    pass


class InputLengthError(ValidationError):
    """Raised when input exceeds maximum allowed characters/tokens."""
    pass


class InvalidCommandError(ValidationError):
    """Raised when an unrecognized or malformed CLI command is invoked."""
    pass


class ContextOverflowError(ChatbotError):
    """Raised when context token budget cannot be satisfied even after pruning."""
    pass


class APIError(ChatbotError):
    """Base class for API-related communication failures."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        retryable: bool = False,
        details: Optional[dict] = None,
    ) -> None:
        super().__init__(message, details)
        self.status_code = status_code
        self.retryable = retryable


class APIRateLimitError(APIError):
    """Raised when HTTP 429 Too Many Requests is encountered."""

    def __init__(self, message: str = "API rate limit exceeded", details: Optional[dict] = None) -> None:
        super().__init__(message, status_code=429, retryable=True, details=details)


class APITimeoutError(APIError):
    """Raised when upstream API request times out."""

    def __init__(self, message: str = "API request timed out", details: Optional[dict] = None) -> None:
        super().__init__(message, status_code=408, retryable=True, details=details)


class APIResponseError(APIError):
    """Raised when the API returns an unexpected or invalid payload."""
    pass


class PersistenceError(ChatbotError):
    """Raised when SQLite database operations fail."""
    pass
