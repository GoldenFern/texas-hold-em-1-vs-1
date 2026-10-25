"""Retry policy and shared retry budget.

The defaults mirror the retry tests in OpenCode's session runner: an
exponential schedule that starts at two seconds, caps at ten seconds, and
allows ten retries.  A single budget is shared by transparent request retries
and by continuations after a partially streamed reply, so the two paths cannot
multiply the retry allowance.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Callable, Optional

DEFAULT_MAX_RETRIES = 10
DEFAULT_BASE_DELAY = 2.0
DEFAULT_MAX_DELAY = 10.0
DEFAULT_RETRY_AFTER_CAP = 15 * 60.0
DEFAULT_MAX_TIMEOUT_RETRIES = 3
DEFAULT_JITTER = 0.2

_RETRYABLE_KINDS = {
    "rate_limit",
    "server",
    "network",
    "incomplete_stream",
    "unknown",
}
_TERMINAL_KINDS = {
    "auth",
    "quota",
    "content_policy",
    "content_filter",
    "param",
    "context_length",
    "cancelled",
    "stream_unsupported",
    "payload_too_large",
}


def is_retryable(kind: str, status: int = 0) -> bool:
    """Return whether an error kind may succeed when sent again.

    Args:
        kind: Stable error category produced by the client.
        status: HTTP status code, or zero for transport failures.

    Returns:
        True when the caller may schedule another attempt.
    """
    if kind in _TERMINAL_KINDS:
        return False
    if kind in _RETRYABLE_KINDS:
        # HTTP 408 is a deterministic provider timeout; transport timeouts use
        # status zero and are retried under the separate timeout allowance.
        return not (kind == "timeout" and status == 408)
    if kind == "timeout":
        return status == 0
    if kind == "http":
        return status == 0 or status >= 500
    return status == 0 or status >= 500


def parse_retry_after(value: object) -> float:
    """Parse a Retry-After header value into seconds.

    Args:
        value: Header value as seconds, a numeric string, or an HTTP date.

    Returns:
        Non-negative seconds, or zero when the value is absent or invalid.
    """
    if value is None:
        return 0.0
    text = str(value).strip()
    if not text:
        return 0.0
    try:
        seconds = float(text)
    except ValueError:
        seconds = -1.0
    if seconds >= 0:
        return seconds
    try:
        from email.utils import parsedate_to_datetime

        parsed = parsedate_to_datetime(text)
        if parsed is None:
            return 0.0
        return max(0.0, parsed.timestamp() - time.time())
    except (TypeError, ValueError, OverflowError):
        return 0.0


@dataclass(frozen=True)
class RetryPolicy:
    """Resolved retry schedule for one client or logical model step."""

    max_retries: int = DEFAULT_MAX_RETRIES
    base_delay: float = DEFAULT_BASE_DELAY
    max_delay: float = DEFAULT_MAX_DELAY
    retry_after_cap: float = DEFAULT_RETRY_AFTER_CAP
    max_timeout_retries: int = DEFAULT_MAX_TIMEOUT_RETRIES
    jitter: float = DEFAULT_JITTER
    deadline_seconds: float = 0.0

    @classmethod
    def from_runtime(cls, runtime: dict) -> "RetryPolicy":
        """Build a policy from ``llm_runtime`` configuration.

        Args:
            runtime: Game runtime dictionary.

        Returns:
            A policy with OpenCode-compatible defaults for missing keys.
        """

        def integer(key: str, default: int, minimum: int = 0) -> int:
            try:
                value = int(runtime.get(key, default))
            except (TypeError, ValueError):
                return default
            return max(minimum, value)

        def number(key: str, default: float, minimum: float = 0.0) -> float:
            try:
                value = float(runtime.get(key, default))
            except (TypeError, ValueError):
                return default
            return max(minimum, value)

        return cls(
            max_retries=integer("max_retries", DEFAULT_MAX_RETRIES),
            base_delay=number(
                "retry_base_delay_ms", DEFAULT_BASE_DELAY * 1000.0
            )
            / 1000.0,
            max_delay=number(
                "retry_max_delay_ms", DEFAULT_MAX_DELAY * 1000.0
            )
            / 1000.0,
            retry_after_cap=number(
                "retry_after_cap_seconds", DEFAULT_RETRY_AFTER_CAP
            ),
            max_timeout_retries=integer(
                "max_timeout_retries", DEFAULT_MAX_TIMEOUT_RETRIES
            ),
            jitter=min(1.0, number("retry_jitter", DEFAULT_JITTER)),
            deadline_seconds=number("retry_deadline_seconds", 0.0),
        )

    def nominal_delay(self, retry_index: int) -> float:
        """Return the unjittered delay before the given retry.

        Args:
            retry_index: One-based retry number, where 1 is the first retry.

        Returns:
            Delay in seconds, capped by :attr:`max_delay`.
        """
        if retry_index <= 0:
            return 0.0
        return min(self.base_delay * (2 ** (retry_index - 1)), self.max_delay)

    def delay(
        self,
        retry_index: int,
        retry_after: float = 0.0,
        rng: Optional[random.Random] = None,
    ) -> float:
        """Return the jittered delay before the given retry.

        Args:
            retry_index: One-based retry number.
            retry_after: Provider-requested delay in seconds.
            rng: Optional random source for deterministic tests.

        Returns:
            Non-negative delay in seconds.
        """
        delay = self.nominal_delay(retry_index)
        if self.jitter > 0:
            source = rng or random
            delay *= source.uniform(1.0 - self.jitter, 1.0 + self.jitter)
        if retry_after > 0:
            delay = max(delay, min(float(retry_after), self.retry_after_cap))
        return max(0.0, delay)


@dataclass(frozen=True)
class RetryDecision:
    """One decision from a :class:`RetryBudget`."""

    retry: bool
    attempt: int
    delay: float = 0.0
    reason: str = ""


class RetryBudget:
    """Shared retry counter and deadline for one logical model step."""

    def __init__(
        self,
        policy: Optional[RetryPolicy] = None,
        *,
        clock: Callable[[], float] = time.monotonic,
        rng: Optional[random.Random] = None,
        sleep: Optional[Callable[[float], None]] = None,
    ) -> None:
        self.policy = policy or RetryPolicy()
        self._clock = clock
        self._rng = rng
        self._sleep = sleep or time.sleep
        self.started_at = clock()
        self.deadline_at = (
            self.started_at + self.policy.deadline_seconds
            if self.policy.deadline_seconds > 0
            else float("inf")
        )
        self.attempts = 0
        self.retries = 0
        self.timeout_retries = 0

    @property
    def remaining_retries(self) -> int:
        """Number of retries still available."""
        return max(0, self.policy.max_retries - self.retries)

    def begin(self) -> int:
        """Record the start of one physical request and return its number."""
        self.attempts += 1
        return self.attempts

    def next_retry(
        self,
        kind: str,
        *,
        retryable: bool = True,
        retry_after: float = 0.0,
    ) -> RetryDecision:
        """Reserve the next retry and compute its delay.

        Args:
            kind: Error category or ``incomplete_stream``.
            retryable: Whether the caller considers the failure retryable.
            retry_after: Provider-requested delay in seconds.

        Returns:
            A decision whose ``attempt`` is the number of the next request.
        """
        next_attempt = self.attempts + 1
        if not retryable:
            return RetryDecision(False, next_attempt, reason=kind)
        if self.retries >= self.policy.max_retries:
            return RetryDecision(False, next_attempt, reason="retry_budget")
        now = self._clock()
        if now >= self.deadline_at:
            return RetryDecision(False, next_attempt, reason="deadline")
        is_timeout = kind in ("timeout", "transport_timeout")
        if is_timeout and self.timeout_retries >= self.policy.max_timeout_retries:
            return RetryDecision(False, next_attempt, reason="timeout_budget")
        retry_index = self.retries + 1
        delay = self.policy.delay(retry_index, retry_after, self._rng)
        if self.deadline_at != float("inf") and now + delay > self.deadline_at:
            delay = max(0.0, self.deadline_at - now)
            if delay <= 0:
                return RetryDecision(False, next_attempt, reason="deadline")
        self.retries += 1
        if is_timeout:
            self.timeout_retries += 1
        return RetryDecision(True, next_attempt, delay, kind)

    def wait(
        self,
        delay: float,
        cancel: Optional[Callable[[], bool]] = None,
    ) -> bool:
        """Sleep for ``delay`` seconds, aborting early when cancelled.

        Args:
            delay: Seconds to wait.
            cancel: Optional cancellation predicate.

        Returns:
            False when the wait was cancelled, otherwise True.
        """
        if cancel is not None and cancel():
            return False
        if delay <= 0:
            return not (cancel is not None and cancel())
        if self._sleep is not time.sleep:
            self._sleep(delay)
            return not (cancel is not None and cancel())
        end = self._clock() + delay
        while True:
            now = self._clock()
            if now >= end:
                return True
            if cancel is not None and cancel():
                return False
            time.sleep(min(0.1, max(0.0, end - now)))


__all__ = [
    "DEFAULT_BASE_DELAY",
    "DEFAULT_JITTER",
    "DEFAULT_MAX_DELAY",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_MAX_TIMEOUT_RETRIES",
    "DEFAULT_RETRY_AFTER_CAP",
    "RetryBudget",
    "RetryDecision",
    "RetryPolicy",
    "is_retryable",
    "parse_retry_after",
]
