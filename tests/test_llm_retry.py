"""Retry policy and shared budget tests."""

from __future__ import annotations

import random

import pytest

from src.llm.retry import (
    RetryBudget,
    RetryPolicy,
    is_retryable,
    parse_retry_after,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, delay: float) -> None:
        self.now += delay


def make_budget(
    *,
    max_retries: int = 10,
    max_timeout_retries: int = 3,
    deadline_seconds: float = 0.0,
) -> tuple[RetryBudget, FakeClock]:
    clock = FakeClock()
    policy = RetryPolicy(
        max_retries=max_retries,
        base_delay=2.0,
        max_delay=10.0,
        max_timeout_retries=max_timeout_retries,
        jitter=0.0,
        deadline_seconds=deadline_seconds,
    )
    return RetryBudget(policy, clock=clock, sleep=clock.sleep), clock


def test_policy_from_runtime_matches_opencode_defaults() -> None:
    policy = RetryPolicy.from_runtime({})
    assert policy.max_retries == 10
    assert policy.base_delay == 2.0
    assert policy.max_delay == 10.0
    assert policy.retry_after_cap == 900.0
    assert policy.max_timeout_retries == 3


def test_nominal_schedule_matches_opencode() -> None:
    policy = RetryPolicy()
    assert [policy.nominal_delay(index) for index in range(1, 11)] == [
        2.0,
        4.0,
        8.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
    ]


def test_jitter_stays_within_twenty_percent() -> None:
    policy = RetryPolicy()
    rng = random.Random(7)
    for index in range(1, 11):
        nominal = policy.nominal_delay(index)
        delay = policy.delay(index, rng=rng)
        assert nominal * 0.8 <= delay <= nominal * 1.2


def test_retry_after_extends_and_caps_delay() -> None:
    policy = RetryPolicy(jitter=0.0)
    assert policy.delay(1, retry_after=5.0) == 5.0
    assert policy.delay(1, retry_after=3600.0) == 900.0


def test_parse_retry_after_accepts_seconds_and_dates() -> None:
    assert parse_retry_after("5") == 5.0
    assert parse_retry_after("") == 0.0
    assert parse_retry_after("not-a-date") == 0.0
    parsed = parse_retry_after("Wed, 21 Oct 2015 07:28:00 GMT")
    assert parsed >= 0


def test_budget_allows_ten_retries_with_attempt_numbers_two_to_eleven() -> None:
    budget, clock = make_budget()
    attempts = []
    delays = []
    for _ in range(11):
        budget.begin()
        decision = budget.next_retry("server")
        if decision.retry:
            attempts.append(decision.attempt)
            delays.append(decision.delay)
            clock.sleep(decision.delay)
    assert attempts == list(range(2, 12))
    assert delays == [2.0, 4.0, 8.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
    assert budget.retries == 10
    assert budget.next_retry("server").retry is False


def test_budget_stops_after_three_transport_timeouts() -> None:
    budget, clock = make_budget()
    for _ in range(5):
        budget.begin()
        decision = budget.next_retry("timeout")
        if decision.retry:
            clock.sleep(decision.delay)
    assert budget.timeout_retries == 3
    assert budget.next_retry("timeout").retry is False


def test_budget_deadline_blocks_retry() -> None:
    budget, clock = make_budget(deadline_seconds=1.0)
    budget.begin()
    clock.sleep(2.0)
    decision = budget.next_retry("server")
    assert decision.retry is False
    assert decision.reason == "deadline"


def test_budget_cancel_stops_wait() -> None:
    budget, _clock = make_budget()
    assert budget.wait(1.0, lambda: True) is False


@pytest.mark.parametrize(
    ("kind", "status", "expected"),
    [
        ("network", 0, True),
        ("rate_limit", 429, True),
        ("server", 503, True),
        ("incomplete_stream", 0, True),
        ("auth", 401, False),
        ("context_length", 400, False),
        ("timeout", 408, False),
        ("param", 400, False),
    ],
)
def test_is_retryable_classification(kind: str, status: int, expected: bool) -> None:
    assert is_retryable(kind, status) is expected
