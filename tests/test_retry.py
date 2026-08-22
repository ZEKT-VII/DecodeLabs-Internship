"""Unit tests for the retry handler with exponential backoff."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import time
from unittest.mock import patch

from retry import retry_with_backoff, _calculate_delay
from exceptions import APIError, APIRateLimitError, APITimeoutError, APIResponseError


class TestCalculateDelay:
    """Tests for backoff delay calculation."""

    def test_first_attempt_delay(self):
        delay = _calculate_delay(attempt=0, initial_delay=1.0, backoff_factor=2.0, max_delay=8.0)
        # Base = 1.0 * 2^0 = 1.0, plus jitter [0, 0.5], so between 1.0 and 1.5
        assert 1.0 <= delay <= 1.5

    def test_second_attempt_delay(self):
        delay = _calculate_delay(attempt=1, initial_delay=1.0, backoff_factor=2.0, max_delay=8.0)
        # Base = 1.0 * 2^1 = 2.0, plus jitter [0, 1.0], so between 2.0 and 3.0
        assert 2.0 <= delay <= 3.0

    def test_delay_capped_at_max(self):
        delay = _calculate_delay(attempt=10, initial_delay=1.0, backoff_factor=2.0, max_delay=8.0)
        assert delay <= 8.0


class TestRetryWithBackoff:
    """Tests for retry mechanics."""

    def test_success_on_first_attempt(self):
        def always_succeeds():
            return "ok"
        result = retry_with_backoff(always_succeeds, max_retries=3, initial_delay=0.01)
        assert result == "ok"

    def test_retries_on_rate_limit(self):
        call_count = 0
        def fail_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise APIRateLimitError("Rate limited")
            return "ok"

        result = retry_with_backoff(fail_then_succeed, max_retries=3, initial_delay=0.01)
        assert result == "ok"
        assert call_count == 3

    def test_retries_on_timeout(self):
        call_count = 0
        def fail_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise APITimeoutError("Timed out")
            return "ok"

        result = retry_with_backoff(fail_then_succeed, max_retries=3, initial_delay=0.01)
        assert result == "ok"

    def test_exhausts_retries(self):
        def always_fails():
            raise APIRateLimitError("Always rate limited")

        with pytest.raises(APIRateLimitError):
            retry_with_backoff(always_fails, max_retries=2, initial_delay=0.01)

    def test_non_retryable_error_propagates_immediately(self):
        call_count = 0
        def fail_non_retryable():
            nonlocal call_count
            call_count += 1
            raise APIResponseError("Invalid response", status_code=400)

        with pytest.raises(APIResponseError):
            retry_with_backoff(fail_non_retryable, max_retries=3, initial_delay=0.01)
        assert call_count == 1  # Should NOT retry

    def test_retryable_api_error(self):
        call_count = 0
        def fail_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise APIError("Server error", status_code=502, retryable=True)
            return "ok"

        result = retry_with_backoff(fail_then_succeed, max_retries=3, initial_delay=0.01)
        assert result == "ok"
        assert call_count == 2

    def test_connection_error_retries(self):
        call_count = 0
        def fail_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Connection refused")
            return "ok"

        result = retry_with_backoff(fail_then_succeed, max_retries=3, initial_delay=0.01)
        assert result == "ok"
