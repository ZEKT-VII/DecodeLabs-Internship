"""Main entry point: Rich terminal interface and interactive REPL loop."""

from __future__ import annotations

import sys
import logging

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.markdown import Markdown
from rich.theme import Theme

from config import DEFAULT_MODEL, DEFAULT_SYSTEM_PROMPT, validate_config
from engine import ConversationEngine
from commands import handle_command, COMMAND_REGISTRY
from exceptions import (
    ChatbotError,
    EmptyInputError,
    InputLengthError,
    InvalidCommandError,
    AuthError,
)
from security import resolve_api_key, get_masked_active_key
from validation import CaliperValidationGate
from logging_config import setup_logger

# ──────────────────────────── Theme ──────────────────────────

CHATBOT_THEME = Theme({
    "user": "bold cyan",
    "assistant": "bold green",
    "system": "bold yellow",
    "error": "bold red",
    "info": "dim white",
    "command": "bold magenta",
})

console = Console(theme=CHATBOT_THEME)

BANNER = r"""
╔══════════════════════════════════════════════════════════════╗
║           🧠  Stateful Conversational Agent  🧠             ║
║              DecodeLabs · Project 1                         ║
║                                                             ║
║   Multi-turn · Token-Budgeted · Priority Memory             ║
║   Type /help for commands · /exit to quit                   ║
╚══════════════════════════════════════════════════════════════╝
"""


def main() -> None:
    """Application entry point."""
    # Setup logging
    setup_logger(level=logging.WARNING)

    # Validate configuration on startup
    try:
        validate_config()
    except ValueError as e:
        console.print(f"[error]Configuration Error: {e}[/error]")
        sys.exit(1)

    # Resolve API key (with interactive fallback)
    try:
        api_key = resolve_api_key(interactive_fallback=True)
    except AuthError as e:
        console.print(f"[error]{e.message}[/error]")
        console.print("[info]Set NVIDIA_API_KEY in your environment or .env file.[/info]")
        sys.exit(1)

    # Display banner
    console.print(BANNER, style="bold blue")
    console.print(f"[info]Model: {DEFAULT_MODEL}[/info]")
    console.print(f"[info]API Key: {get_masked_active_key()}[/info]")
    console.print()

    # Initialize conversation engine
    engine = ConversationEngine(
        model=DEFAULT_MODEL,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        api_key=api_key,
    )
    engine.start_session()
    console.print(f"[info]Session: {engine.session_id}[/info]\n")

    # ──────────────────────────── REPL Loop ──────────────────────────

    while True:
        try:
            # Prompt
            raw_input = console.input("[user]You ▸ [/user]")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[system]Session ended.[/system]")
            break

        # Skip empty lines silently
        if not raw_input or not raw_input.strip():
            continue

        # Check for commands
        try:
            is_command, cmd_name, cmd_args = CaliperValidationGate.parse_and_validate_command(raw_input)
        except InvalidCommandError as e:
            console.print(f"[error]{e.message}[/error]")
            continue

        if is_command:
            try:
                response_text, should_exit = handle_command(cmd_name, cmd_args, engine)
                console.print(f"[command]{response_text}[/command]")
                if should_exit:
                    break
            except InvalidCommandError as e:
                console.print(f"[error]{e.message}[/error]")
            except ChatbotError as e:
                console.print(f"[error]Command error: {e.message}[/error]")
            continue

        # Regular conversational turn
        try:
            with console.status("[info]Thinking...[/info]", spinner="dots"):
                response = engine.send_message(raw_input)

            # Display assistant response
            console.print()
            try:
                md = Markdown(response.content)
                console.print(Panel(md, title="🤖 Assistant", border_style="green", expand=True))
            except Exception:
                # Fallback to plain text if markdown rendering fails
                console.print(Panel(response.content, title="🤖 Assistant", border_style="green", expand=True))

            # Show latency
            console.print(
                f"[info]({response.latency_sec:.2f}s · "
                f"{response.usage.prompt_tokens}+{response.usage.completion_tokens} tokens)[/info]"
            )
            console.print()

        except EmptyInputError as e:
            console.print(f"[error]{e.message}[/error]")
        except InputLengthError as e:
            console.print(f"[error]{e.message}[/error]")
        except AuthError as e:
            console.print(f"[error]Authentication Error: {e.message}[/error]")
            console.print("[info]Check /key-status or set NVIDIA_API_KEY in your .env file.[/info]")
        except ChatbotError as e:
            console.print(f"[error]Error: {e.message}[/error]")
        except Exception as e:
            console.print(f"[error]Unexpected error: {e}[/error]")

    # ──────────────────────────── Shutdown ──────────────────────────
    engine.shutdown()
    console.print("[system]Goodbye! Your session has been saved.[/system]")


if __name__ == "__main__":
    main()
