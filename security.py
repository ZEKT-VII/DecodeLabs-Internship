"""Security and credential management module with secret masking and dynamic key resolution."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

from exceptions import AuthError

# Load .env file if available
_ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH)

# In-memory runtime credential cache
_RUNTIME_KEY_OVERRIDE: Optional[str] = None

# Regex patterns for identifying potential API keys in text/logs
API_KEY_PATTERNS = [
    re.compile(r"nvapi-[A-Za-z0-9_-]{40,}"),
    re.compile(r"Bearer\s+([A-Za-z0-9_.-]{20,})"),
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
]


def mask_api_key(key: Optional[str]) -> str:
    """Masks an API key for safe display, preserving only prefix and suffix."""
    if not key or not isinstance(key, str):
        return "[NO_KEY_CONFIGURED]"
    stripped = key.strip()
    if len(stripped) <= 12:
        return "******"
    return f"{stripped[:6]}...{stripped[-4:]}"


def redact_secrets(text: str) -> str:
    """Redacts any detected API key patterns from log strings or exception texts."""
    if not text:
        return text
    scrubbed = text
    for pattern in API_KEY_PATTERNS:
        scrubbed = pattern.sub("[REDACTED_API_KEY]", scrubbed)
    return scrubbed


def resolve_api_key(interactive_fallback: bool = False) -> str:
    """
    Resolves the NVIDIA API key according to the security hierarchy:
    1. Runtime key override (in-memory custom key)
    2. os.environ['NVIDIA_API_KEY']
    3. Loaded .env file
    4. Optional interactive prompt if running in interactive mode.
    """
    global _RUNTIME_KEY_OVERRIDE

    # 1. Runtime override
    if _RUNTIME_KEY_OVERRIDE:
        return _RUNTIME_KEY_OVERRIDE

    # 2. Environment variable
    env_key = os.getenv("NVIDIA_API_KEY")
    if env_key and env_key.strip() and env_key.strip() != "your_nvidia_api_key_here":
        return env_key.strip()

    # 3. Interactive prompt if enabled
    if interactive_fallback:
        import getpass
        print("\n[!] No NVIDIA_API_KEY detected in environment or .env file.")
        user_key = getpass.getpass("Enter your NVIDIA API Key (hidden): ").strip()
        if user_key:
            _RUNTIME_KEY_OVERRIDE = user_key
            save_choice = input("Save key to local .env file? (y/N): ").strip().lower()
            if save_choice == "y":
                set_custom_api_key(user_key, save_to_env=True)
            return user_key

    raise AuthError(
        "NVIDIA_API_KEY is not configured. Please set the environment variable NVIDIA_API_KEY "
        "or define it in your local .env file."
    )


def set_custom_api_key(key: str, save_to_env: bool = False) -> None:
    """
    Sets a custom API key at runtime, optionally persisting it to the local .env file.
    """
    global _RUNTIME_KEY_OVERRIDE
    cleaned_key = key.strip()
    if not cleaned_key:
        raise AuthError("API key cannot be empty.")
    _RUNTIME_KEY_OVERRIDE = cleaned_key

    if save_to_env:
        env_path = _ENV_PATH
        env_lines = []
        key_written = False

        if env_path.exists():
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("NVIDIA_API_KEY="):
                        env_lines.append(f"NVIDIA_API_KEY={cleaned_key}\n")
                        key_written = True
                    else:
                        env_lines.append(line)

        if not key_written:
            env_lines.append(f"NVIDIA_API_KEY={cleaned_key}\n")

        with open(env_path, "w", encoding="utf-8") as f:
            f.writelines(env_lines)


def get_masked_active_key() -> str:
    """Returns the masked version of the currently resolved key."""
    try:
        key = resolve_api_key(interactive_fallback=False)
        return mask_api_key(key)
    except AuthError:
        return "[NO_KEY_CONFIGURED]"


def get_key_status() -> dict[str, str]:
    """
    Returns non-sensitive metadata about the active API key configuration.
    Safe for display, logging, and screenshots.
    """
    global _RUNTIME_KEY_OVERRIDE

    source = "None"
    status = "Missing"
    preview = "[NO_KEY_CONFIGURED]"

    if _RUNTIME_KEY_OVERRIDE:
        source = "Runtime in-memory override"
        status = "Configured"
        preview = mask_api_key(_RUNTIME_KEY_OVERRIDE)
    elif os.getenv("NVIDIA_API_KEY") and os.getenv("NVIDIA_API_KEY").strip() != "your_nvidia_api_key_here":
        source = "Environment variable (NVIDIA_API_KEY)"
        status = "Configured"
        preview = mask_api_key(os.getenv("NVIDIA_API_KEY"))
    elif _ENV_PATH.exists():
        # Check if loaded from .env
        env_key = os.getenv("NVIDIA_API_KEY")
        if env_key and env_key.strip() and env_key.strip() != "your_nvidia_api_key_here":
            source = "Local .env file"
            status = "Configured"
            preview = mask_api_key(env_key)

    is_valid_format = preview.startswith("nvapi-") or preview.startswith("sk-") or status == "Configured"

    return {
        "source": source,
        "status": status,
        "format": "Valid" if is_valid_format and status == "Configured" else "Unverified/Missing",
        "preview": preview,
    }

