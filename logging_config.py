"""Structured logging configuration with automated secret scrubbing and contextual tracking."""

from __future__ import annotations

import logging
import sys
from typing import Optional

from security import redact_secrets


class SecretScrubbingFilter(logging.Filter):
    """Logging filter that scrubs sensitive API keys from log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = redact_secrets(record.msg)
        if record.args:
            if isinstance(record.args, dict):
                record.args = {
                    k: (redact_secrets(v) if isinstance(v, str) else v)
                    for k, v in record.args.items()
                }
            elif isinstance(record.args, tuple):
                record.args = tuple(
                    (redact_secrets(arg) if isinstance(arg, str) else arg)
                    for arg in record.args
                )
        return True


def setup_logger(
    name: str = "stateful_chatbot",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Configures and returns a logger instance with secret redaction enabled."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console Handler (Standard Error for logs, keeping stdout clean for CLI)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(SecretScrubbingFilter())
    logger.addHandler(console_handler)

    # Optional File Handler
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.addFilter(SecretScrubbingFilter())
        logger.addHandler(file_handler)

    return logger


# Default application logger instance
logger = setup_logger()
