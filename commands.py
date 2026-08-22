"""CLI Command Router: handles slash commands with strict client-side isolation."""

from __future__ import annotations

import getpass
import logging
from typing import Any, Dict, Optional, Tuple

from exceptions import InvalidCommandError
from security import (
    set_custom_api_key,
    get_masked_active_key,
    mask_api_key,
    get_key_status,
)

logger = logging.getLogger("stateful_chatbot.commands")

# Registry of valid commands and their descriptions
COMMAND_REGISTRY: Dict[str, str] = {
    "help": "Show available commands and usage.",
    "stats": "Display session statistics and token usage.",
    "history": "Show conversation history for the current session.",
    "clear": "Clear in-memory conversation state (keeps SQLite data).",
    "delete": "Delete current session from database. (Usage: /delete [--force])",
    "delete-all": "Purge ALL sessions from database. (Usage: /delete-all [--force])",
    "remember": "Pin a fact to persistent memory. Usage: /remember key=value",
    "forget": "Remove a pinned fact from memory. Usage: /forget key",
    "key-status": "Show safe, non-sensitive API key configuration status.",
    "model": "Display the currently active model.",
    "exit": "Exit the chatbot.",
    "quit": "Exit the chatbot (alias for /exit).",
}


def handle_command(
    cmd_name: str,
    cmd_args: str,
    engine: Any,
) -> Tuple[str, bool]:
    """
    Dispatches a validated command to its handler.
    
    Returns:
        (response_text, should_exit)
    """
    if cmd_name not in COMMAND_REGISTRY:
        available = ", ".join(f"/{c}" for c in sorted(COMMAND_REGISTRY))
        raise InvalidCommandError(
            f"Unknown command: /{cmd_name}\nAvailable commands: {available}"
        )

    if cmd_name == "help":
        return _cmd_help(), False

    elif cmd_name == "stats":
        return _cmd_stats(engine), False

    elif cmd_name == "history":
        return _cmd_history(engine), False

    elif cmd_name == "clear":
        return _cmd_clear(engine), False

    elif cmd_name == "delete":
        return _cmd_delete(engine, cmd_args), False

    elif cmd_name == "delete-all":
        return _cmd_delete_all(engine, cmd_args), False

    elif cmd_name == "remember":
        return _cmd_remember(engine, cmd_args), False

    elif cmd_name == "forget":
        return _cmd_forget(engine, cmd_args), False

    elif cmd_name == "key-status":
        return _cmd_key_status(), False

    elif cmd_name == "model":
        return _cmd_model(engine), False

    elif cmd_name in ("exit", "quit"):
        return "Goodbye! Session saved.", True

    return f"Command /{cmd_name} is not yet implemented.", False


def _cmd_help() -> str:
    """Formats and returns the help text."""
    lines = ["╭─ Available Commands ─╮"]
    for cmd, desc in sorted(COMMAND_REGISTRY.items()):
        lines.append(f"  /{cmd:<12} {desc}")
    lines.append("╰──────────────────────╯")
    return "\n".join(lines)


def _cmd_stats(engine: Any) -> str:
    """Returns formatted session statistics."""
    stats = engine.get_stats()
    return (
        "╭─ Session Statistics ─╮\n"
        f"  Session ID:        {stats.get('session_id', 'N/A')}\n"
        f"  Model:             {stats.get('model', 'N/A')}\n"
        f"  Active Turns:      {stats.get('active_turns', 0)}\n"
        f"  Pinned Identity:   {stats.get('pinned_identity_facts', 0)} facts\n"
        f"  Pinned Prefs:      {stats.get('pinned_preferences', 0)} items\n"
        f"  Pruned Messages:   {stats.get('pruned_messages', 0)}\n"
        f"  Context Tokens:    {stats.get('estimated_context_tokens', 0)} / {stats.get('budget_tokens', 0)}\n"
        f"  Total Prompt Tok:  {stats.get('total_prompt_tokens', 0)}\n"
        f"  Total Compl Tok:   {stats.get('total_completion_tokens', 0)}\n"
        f"  Total Requests:    {stats.get('total_requests', 0)}\n"
        f"  API Key:           {get_masked_active_key()}\n"
        "╰──────────────────────╯"
    )


def _cmd_history(engine: Any) -> str:
    """Returns formatted conversation history."""
    history = engine.get_history()
    if not history:
        return "No conversation history yet."

    lines = ["╭─ Conversation History ─╮"]
    for i, msg in enumerate(history, 1):
        role = msg["role"].upper()
        content = msg["content"]
        # Truncate very long messages for display
        if len(content) > 200:
            content = content[:200] + "..."
        lines.append(f"  [{i}] {role}: {content}")
    lines.append(f"╰─ Total: {len(history)} messages ─╯")
    return "\n".join(lines)


def _cmd_clear(engine: Any) -> str:
    """Clears in-memory state."""
    engine.clear_memory()
    return "✓ In-memory conversation cleared. Pinned facts and SQLite history preserved."


def _cmd_delete(engine: Any, args: str) -> str:
    """
    Deletes current session with safety confirmation.
    Pass --force or -y to bypass interactive prompt.
    """
    if not engine.session_id:
        return "No active session to delete."

    if "--force" not in args and "-y" not in args:
        try:
            confirm = input("⚠️  Permanently delete current session from database? [y/N]: ").strip().lower()
            if confirm not in ("y", "yes"):
                return "Operation cancelled."
        except (EOFError, KeyboardInterrupt):
            return "Operation cancelled."

    deleted = engine.delete_session()
    if deleted:
        return "✓ Current session deleted from database. Starting fresh."
    return "No active session to delete."


def _cmd_delete_all(engine: Any, args: str) -> str:
    """
    Purges ALL sessions from database with safety confirmation.
    Pass --force or -y to bypass interactive prompt.
    """
    if "--force" not in args and "-y" not in args:
        try:
            confirm = input("⚠️  WARNING: Permanently delete ALL stored sessions? [y/N]: ").strip().lower()
            if confirm not in ("y", "yes"):
                return "Operation cancelled."
        except (EOFError, KeyboardInterrupt):
            return "Operation cancelled."

    count = engine.delete_all_sessions()
    return f"✓ Purged {count} session(s) from the database."


def _cmd_remember(engine: Any, args: str) -> str:
    """Pins a user preference."""
    if not args or "=" not in args:
        return "Usage: /remember key=value\nExample: /remember language=Python"

    key, _, value = args.partition("=")
    key = key.strip()
    value = value.strip()

    if not key or not value:
        return "Both key and value are required. Usage: /remember key=value"

    engine.remember_fact(key, value)
    return f"✓ Remembered: {key} = {value}"


def _cmd_forget(engine: Any, args: str) -> str:
    """Removes a pinned fact from memory."""
    key = args.strip()
    if not key:
        return "Usage: /forget key\nExample: /forget language"

    removed = engine.forget_fact(key)
    if removed:
        return f"✓ Forgotten: {key}"
    return f"Key '{key}' was not found in pinned preferences."


def _cmd_key_status() -> str:
    """Displays safe, masked API key configuration status."""
    info = get_key_status()
    return (
        "╭─ API Key Status ─╮\n"
        f"  Source:   {info['source']}\n"
        f"  Status:   {info['status']}\n"
        f"  Format:   {info['format']}\n"
        f"  Preview:  {info['preview']}\n"
        "╰──────────────────╯"
    )


def _cmd_model(engine: Any) -> str:
    """Displays the current model."""
    return f"Active model: {engine.model}"
