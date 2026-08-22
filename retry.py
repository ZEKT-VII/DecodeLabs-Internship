"""Retry handler with exponential backoff and jitter for transient API failures."""

from __future__ import annotations

import random
import time
import logging
from typing import TypeVar, Callable, Optional

from config import (
    MAX_RETRIES,
    INITIAL_RETRY_DELAY_SEC,
    BACKOFF_FACTOR,
    MAX_RETRY_DELAY_SEC,
    RETRYABLE_STATUS_CODES,
)
from exceptions import APIError, APIRateLimitError, APITimeoutError

logger = logging.getLogger("stateful_chatbot.retry")

T = TypeVar("T")


def retry_with_backoff(
    func: Callable[..., T],
    *args,
    max_retries: int = MAX_RETRIES,
    initial_delay: float = INITIAL_RETRY_DELAY_SEC,
    backoff_factor: float = BACKOFF_FACTOR,
    max_delay: float = MAX_RETRY_DELAY_SEC,
    **kwargs,
) -> T:
    """
    Executes `func` with automatic retries on transient failures.
    
    Implements exponential backoff with jitter:
        delay = min(initial_delay * backoff_factor^attempt + jitter, max_delay)
    
    Retries on:
        - APIRateLimitError (HTTP 429)
        - APITimeoutError (timeouts)
        - APIError with retryable status codes (500, 502, 503, 504)
        - ConnectionError, TimeoutError
    """
    last_exception: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except (APIRateLimitError, APITimeoutError) as e:
            last_exception = e
            if attempt == max_retries:
                logger.error("Max retries (%d) exhausted: %s", max_retries, str(e))
                raise
            delay = _calculate_delay(attempt, initial_delay, backoff_factor, max_delay)
            logger.warning(
                "Retryable error (attempt %d/%d): %s — sleeping %.2fs",
                attempt + 1, max_retries, str(e), delay,
            )
            time.sleep(delay)
        except APIError as e:
            last_exception = e
            if e.retryable and attempt < max_retries:
                delay = _calculate_delay(attempt, initial_delay, backoff_factor, max_delay)
                logger.warning(
                    "Retryable API error %s (attempt %d/%d) — sleeping %.2fs",
                    e.status_code, attempt + 1, max_retries, delay,
                )
                time.sleep(delay)
            else:
                raise
        except (ConnectionError, TimeoutError) as e:
            last_exception = e
            if attempt == max_retries:
                logger.error("Max retries (%d) exhausted: %s", max_retries, str(e))
                raise APITimeoutError(f"Connection failed after {max_retries} retries: {e}")
            delay = _calculate_delay(attempt, initial_delay, backoff_factor, max_delay)
            logger.warning(
                "Connection error (attempt %d/%d): %s — sleeping %.2fs",
                attempt + 1, max_retries, str(e), delay,
            )
            time.sleep(delay)

    # Shouldn't be reached, but type safety
    raise last_exception or APIError("Retry loop exhausted unexpectedly")


def _calculate_delay(
    attempt: int,
    initial_delay: float,
    backoff_factor: float,
    max_delay: float,
) -> float:
    """Computes delay with exponential backoff and random jitter."""
    base_delay = initial_delay * (backoff_factor ** attempt)
    jitter = random.uniform(0, base_delay * 0.5)
    return min(base_delay + jitter, max_delay)
