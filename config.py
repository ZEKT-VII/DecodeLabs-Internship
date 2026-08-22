"""System configuration and hyperparameter definitions for the Stateful Chatbot Agent."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Set
from dotenv import load_dotenv

# Base Directories
BASE_DIR: Path = Path(__file__).resolve().parent
_ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=_ENV_PATH)

DEFAULT_DB_PATH: Path = BASE_DIR / "chatbot_sessions.db"

# API & Model Configuration
DEFAULT_BASE_URL: str = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "meta/llama-3.1-8b-instruct")

ALLOWED_MODELS: Set[str] = {
    "deepseek-ai/deepseek-v4-flash-0731",
    "deepseek-ai/deepseek-coder-6.7b-instruct",
    "meta/llama-3.3-70b-instruct",
    "meta/llama-3.1-8b-instruct",
    "mistralai/mistral-large-2-instruct",
    "google/codegemma-1.1-7b",
}

# Generation Hyperparameters
DEFAULT_TEMPERATURE: float = 0.7
AUDIT_TEMPERATURE: float = 0.0
DEFAULT_TOP_P: float = 0.95
DEFAULT_MAX_OUTPUT_TOKENS: int = 2048

# Context Budgeting & Token Management
DEFAULT_MAX_CONTEXT_TOKENS: int = int(os.getenv("MAX_CONTEXT_TOKENS", "8192"))
DEFAULT_RESERVED_OUTPUT_TOKENS: int = int(os.getenv("RESERVED_OUTPUT_TOKENS", "1024"))
TOKEN_SAFETY_MARGIN: int = 128
MAX_INPUT_CHARS: int = 20_000

# Retry & Resilience Policies
MAX_RETRIES: int = 3
INITIAL_RETRY_DELAY_SEC: float = 1.0
BACKOFF_FACTOR: float = 2.0
MAX_RETRY_DELAY_SEC: float = 8.0
REQUEST_TIMEOUT_SEC: float = 60.0
RETRYABLE_STATUS_CODES: Set[int] = {429, 500, 502, 503, 504}

# System Prompts & Priority Layer Text
DEFAULT_SYSTEM_PROMPT: str = (
    "You are a helpful, accurate, and context-aware AI assistant. "
    "You remember information shared earlier in the conversation and maintain continuity."
)


def validate_config() -> None:
    """Validates configuration invariants upon startup."""
    if not DEFAULT_BASE_URL.startswith("https://"):
        raise ValueError(f"BASE_URL must be a secure HTTPS endpoint: {DEFAULT_BASE_URL}")
    if DEFAULT_MAX_CONTEXT_TOKENS <= 0:
        raise ValueError(f"MAX_CONTEXT_TOKENS must be positive: {DEFAULT_MAX_CONTEXT_TOKENS}")
    if DEFAULT_RESERVED_OUTPUT_TOKENS <= 0:
        raise ValueError(f"RESERVED_OUTPUT_TOKENS must be positive: {DEFAULT_RESERVED_OUTPUT_TOKENS}")
    if DEFAULT_RESERVED_OUTPUT_TOKENS >= DEFAULT_MAX_CONTEXT_TOKENS:
        raise ValueError("RESERVED_OUTPUT_TOKENS cannot exceed MAX_CONTEXT_TOKENS")
    if not (0.0 <= DEFAULT_TEMPERATURE <= 2.0):
        raise ValueError(f"TEMPERATURE must be between 0.0 and 2.0: {DEFAULT_TEMPERATURE}")
    if MAX_INPUT_CHARS <= 0:
        raise ValueError(f"MAX_INPUT_CHARS must be positive: {MAX_INPUT_CHARS}")


# Run configuration validation on import
validate_config()
