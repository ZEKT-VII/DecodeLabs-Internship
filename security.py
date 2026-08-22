"""Security and credential management module with AES Fernet encryption, secret masking, and dynamic provider resolution."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, Optional
from dotenv import load_dotenv
from cryptography.fernet import Fernet

from exceptions import AuthError
from config import DEFAULT_BASE_URL, DEFAULT_MODEL

# Load .env file if available
_ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH)

# In-memory runtime credential cache
_RUNTIME_KEY_OVERRIDE: Optional[str] = None
_RUNTIME_BASE_URL_OVERRIDE: Optional[str] = None
_RUNTIME_MODEL_OVERRIDE: Optional[str] = None

# Regex patterns for identifying potential API keys in text/logs
API_KEY_PATTERNS = [
    re.compile(r"nvapi-[A-Za-z0-9_-]{40,}"),
    re.compile(r"Bearer\s+([A-Za-z0-9_.-]{20,})"),
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"AIzaSy[A-Za-z0-9_-]{33}"),
]

# Vault Key Location
_VAULT_KEY_PATH = Path(__file__).resolve().parent / ".vault.key"


def _get_or_create_vault_key() -> bytes:
    """Retrieves or generates a local machine-specific encryption key for API credentials."""
    if _VAULT_KEY_PATH.exists():
        try:
            k = _VAULT_KEY_PATH.read_bytes().strip()
            if len(k) == 44:  # Valid base64 32-byte Fernet key
                return k
        except Exception:
            pass

    new_key = Fernet.generate_key()
    try:
        _VAULT_KEY_PATH.write_bytes(new_key)
    except Exception:
        pass
    return new_key


def encrypt_credential(plain_text: str) -> str:
    """Encrypts an API credential using the local vault Fernet key."""
    if not plain_text:
        return ""
    key = _get_or_create_vault_key()
    fernet = Fernet(key)
    return fernet.encrypt(plain_text.strip().encode("utf-8")).decode("utf-8")


def decrypt_credential(cipher_text: str) -> str:
    """Decrypts an encrypted API credential from the local vault."""
    if not cipher_text:
        return ""
    key = _get_or_create_vault_key()
    fernet = Fernet(key)
    try:
        return fernet.decrypt(cipher_text.strip().encode("utf-8")).decode("utf-8")
    except Exception as e:
        raise AuthError(f"Failed to decrypt credential from local vault: {e}")


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
    Resolves the active API key according to the security hierarchy:
    1. Runtime key override (in-memory custom key)
    2. Local encrypted vault in SQLite database
    3. os.environ['NVIDIA_API_KEY'], 'OPENROUTER_API_KEY', 'GEMINI_API_KEY', or 'OPENAI_API_KEY'
    4. Loaded .env file
    5. Optional interactive prompt if running in interactive mode.
    """
    global _RUNTIME_KEY_OVERRIDE

    # 1. Runtime override
    if _RUNTIME_KEY_OVERRIDE:
        return _RUNTIME_KEY_OVERRIDE

    # 2. Local encrypted vault in SQLite
    try:
        from persistence import SessionStore
        store = SessionStore()
        enc_key = store.get_setting("encrypted_api_key")
        if enc_key:
            decrypted = decrypt_credential(enc_key)
            if decrypted and decrypted.strip() and decrypted != "your_nvidia_api_key_here":
                return decrypted.strip()
    except Exception:
        pass

    # 3. Environment variable
    for env_var in ["NVIDIA_API_KEY", "OPENROUTER_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY"]:
        env_val = os.getenv(env_var)
        if env_val and env_val.strip() and env_val.strip() != "your_nvidia_api_key_here":
            return env_val.strip()

    # 4. Interactive prompt if enabled
    if interactive_fallback:
        import getpass
        print("\n[!] No API key detected in in-app settings, environment, or .env file.")
        user_key = getpass.getpass("Enter your API Key (hidden): ").strip()
        if user_key:
            _RUNTIME_KEY_OVERRIDE = user_key
            save_choice = input("Encrypt and save key to local database? (y/N): ").strip().lower()
            if save_choice == "y":
                save_vault_api_settings(provider="custom", base_url=DEFAULT_BASE_URL, model=DEFAULT_MODEL, api_key=user_key)
            return user_key

    raise AuthError(
        "API key is not configured. Please open Settings in the web dashboard to enter your API key, "
        "or set the NVIDIA_API_KEY environment variable."
    )


def resolve_api_config() -> Dict[str, str]:
    """
    Resolves base_url, model, and active provider from vault settings, env, or defaults.
    """
    global _RUNTIME_BASE_URL_OVERRIDE, _RUNTIME_MODEL_OVERRIDE

    provider = "nvidia"
    base_url = _RUNTIME_BASE_URL_OVERRIDE or DEFAULT_BASE_URL
    model = _RUNTIME_MODEL_OVERRIDE or DEFAULT_MODEL

    try:
        from persistence import SessionStore
        store = SessionStore()
        settings = store.get_all_settings()
        if settings.get("api_provider"):
            provider = settings["api_provider"]
        if settings.get("api_base_url"):
            base_url = settings["api_base_url"]
        if settings.get("api_model"):
            model = settings["api_model"]
    except Exception:
        pass

    try:
        key = resolve_api_key(interactive_fallback=False)
    except AuthError:
        key = ""

    return {
        "provider": provider,
        "base_url": base_url,
        "model": model,
        "api_key": key,
    }


def save_vault_api_settings(
    provider: str,
    base_url: str,
    model: str,
    api_key: Optional[str] = None,
) -> None:
    """
    Encrypts and persists in-app API settings to the local SQLite database.
    """
    global _RUNTIME_KEY_OVERRIDE, _RUNTIME_BASE_URL_OVERRIDE, _RUNTIME_MODEL_OVERRIDE
    from persistence import SessionStore

    store = SessionStore()
    store.set_setting("api_provider", provider.strip().lower())
    store.set_setting("api_base_url", base_url.strip())
    store.set_setting("api_model", model.strip())

    _RUNTIME_BASE_URL_OVERRIDE = base_url.strip()
    _RUNTIME_MODEL_OVERRIDE = model.strip()

    if api_key and api_key.strip():
        encrypted = encrypt_credential(api_key.strip())
        store.set_setting("encrypted_api_key", encrypted)
        _RUNTIME_KEY_OVERRIDE = api_key.strip()


def get_vault_api_settings_summary() -> Dict[str, Any]:
    """
    Returns safe settings metadata for UI display with masked key and provider information.
    """
    config = resolve_api_config()
    masked = mask_api_key(config["api_key"]) if config["api_key"] else "[NO_KEY_CONFIGURED]"
    has_key = bool(config["api_key"])

    return {
        "provider": config["provider"],
        "base_url": config["base_url"],
        "model": config["model"],
        "is_configured": has_key,
        "masked_key": masked,
    }


def set_custom_api_key(key: str, save_to_env: bool = False) -> None:
    """
    Sets a custom API key at runtime, optionally persisting it.
    """
    global _RUNTIME_KEY_OVERRIDE
    cleaned_key = key.strip()
    if not cleaned_key:
        raise AuthError("API key cannot be empty.")
    _RUNTIME_KEY_OVERRIDE = cleaned_key

    if save_to_env:
        save_vault_api_settings("custom", DEFAULT_BASE_URL, DEFAULT_MODEL, cleaned_key)


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
    """
    summary = get_vault_api_settings_summary()
    source = "Local Encrypted Database Vault" if summary["is_configured"] else "None"

    if _RUNTIME_KEY_OVERRIDE:
        source = "Runtime in-memory override"
    elif os.getenv("NVIDIA_API_KEY") and os.getenv("NVIDIA_API_KEY").strip() != "your_nvidia_api_key_here":
        source = "Environment variable"

    return {
        "source": source,
        "provider": summary["provider"],
        "model": summary["model"],
        "status": "Configured" if summary["is_configured"] else "Missing",
        "format": "Valid" if summary["is_configured"] else "Unverified/Missing",
        "preview": summary["masked_key"],
    }
