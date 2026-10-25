"""Unit tests for the offline LLM-vs-bot evaluation helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts.eval_llm_vs_bot import (
    HandTask,
    build_tool_recommendation,
    cache_hit_rate,
    count_auto_turns,
    estimate_hand_cost,
    hand_tool_stats,
    make_opponent,
    resolve_tool_names,
    summarize,
)
from src.agents.scripted_bot import ScriptedBot


def test_resolve_tool_names_always_keeps_apply_action() -> None:
    assert resolve_tool_names("none") == ("apply_action",)
    assert resolve_tool_names("estimate_equity") == (
        "estimate_equity",
        "apply_action",
    )
    assert resolve_tool_names("all") == (
        "estimate_hand_probabilities",
        "estimate_equity",
        "get_random_number",
        "call_ev",
        "apply_action",
    )


def test_resolve_tool_names_rejects_unknown_tools() -> None:
    with pytest.raises(ValueError, match="未知工具"):
        resolve_tool_names("estimate_equity,not_a_tool")


def test_resolve_tool_names_history_group() -> None:
    assert resolve_tool_names("history") == (
        "estimate_hand_probabilities",
        "estimate_equity",
        "get_random_number",
        "call_ev",
        "research_history",
        "apply_action",
    )
    assert resolve_tool_names("research_history") == (
        "research_history",
        "apply_action",
    )


def test_tool_recommendation_only_mentions_available_tools() -> None:
    text = build_tool_recommendation(("estimate_equity", "apply_action"))
    assert "estimate_equity" in text
    assert "call_ev" not in text
    assert "系统不会强制调用" in text
    assert build_tool_recommendation(("apply_action",)) == ""


def test_tool_recommendation_english() -> None:
    text = build_tool_recommendation(("estimate_equity", "apply_action"), "en-US")
    assert "Tool-use recommendation" in text
    assert "estimate_equity" in text
    assert "The system does not force tool calls" in text
    assert "系统不会强制调用" not in text


def test_hand_tool_stats_counts_successful_auxiliary_calls() -> None:
    events = [
        {
            "type": "tool_result",
            "name": "estimate_equity",
            "ok": True,
            "turn_id": "t1",
            "result": {"ok": True},
        },
        {
            "type": "tool_result",
            "name": "call_ev",
            "ok": True,
            "turn_id": "t2",
            "result": {"ok": True, "applicable": False},
        },
        {
            "type": "tool_result",
            "name": "apply_action",
            "ok": True,
            "turn_id": "t3",
            "result": {"ok": True},
        },
        {
            "type": "tool_result",
            "name": "estimate_equity",
            "ok": False,
            "turn_id": "t4",
            "result": {"ok": False},
        },
    ]
    stats = hand_tool_stats(events)
    assert stats["aux_tool_calls"] == 2
    assert stats["tool_turns"] == 2
    assert stats["aux_tool_errors"] == 1
    assert stats["equity_calls"] == 1
    assert stats["call_ev_calls"] == 1
    assert stats["call_ev_not_applicable"] == 1


def test_count_auto_turns_separates_api_and_invalid_reasons() -> None:
    events = [
        {
            "type": "turn_end",
            "auto": True,
            "reason": "LLM 调用失败（timeout），自动跟注。",
        },
        {
            "type": "turn_end",
            "auto": True,
            "reason": "连续三次无效动作，自动弃牌。",
        },
        {"type": "turn_end", "auto": False},
    ]
    auto_turns, invalid_auto, api_error_auto = count_auto_turns(events)
    assert auto_turns == 2
    assert invalid_auto == 1
    assert api_error_auto == 1


def test_estimate_hand_cost_uses_per_million_prices() -> None:
    args = SimpleNamespace(price_input=0.045, price_output=0.225)
    assert estimate_hand_cost(1_000_000, 1_000_000, args) == pytest.approx(0.27)
    assert estimate_hand_cost(0, 0, args) == 0.0


def test_estimate_hand_cost_charges_cached_input_at_cache_price() -> None:
    args = SimpleNamespace(
        price_input=0.045,
        price_cache_input=0.003,
        price_output=0.225,
    )
    cost = estimate_hand_cost(
        1_000_000,
        1_000_000,
        args,
        cached_tokens=400_000,
    )
    # 600k miss at 0.045 + 400k cached at 0.003 + 1M output at 0.225.
    assert cost == pytest.approx(0.2532)


def test_estimate_hand_cost_falls_back_to_input_price_without_cache_price() -> None:
    args = SimpleNamespace(
        price_input=0.045,
        price_cache_input=None,
        price_output=0.225,
    )
    assert estimate_hand_cost(
        1_000_000, 0, args, cached_tokens=1_000_000
    ) == pytest.approx(0.045)
    # Backwards compatible with legacy args that predate --price-cache-input.
    legacy_args = SimpleNamespace(price_input=0.045, price_output=0.225)
    assert estimate_hand_cost(
        1_000_000, 0, legacy_args, cached_tokens=1_000_000
    ) == pytest.approx(0.045)


def test_cache_hit_rate_handles_empty_and_bounded_input() -> None:
    assert cache_hit_rate(1_000, 250) == pytest.approx(0.25)
    assert cache_hit_rate(0, 0) == 0.0
    assert cache_hit_rate(1_000, 0) == 0.0
    assert cache_hit_rate(100, 200) == pytest.approx(1.0)


def _cache_row(
    hand_index: int,
    input_tokens: int,
    cached_tokens: int,
) -> dict:
    return {
        "hand_index": hand_index,
        "deal_index": hand_index,
        "llm_seat": 0,
        "net_llm": 0,
        "net_bot": 0,
        "pot": 20,
        "llm_actions_count": 1,
        "bot_actions_count": 1,
        "duration_sec": 1.0,
        "auto_turns": 0,
        "input_tokens": input_tokens,
        "cached_input_tokens": cached_tokens,
        "cache_miss_input_tokens": max(0, input_tokens - cached_tokens),
        "cache_hit_rate": cache_hit_rate(input_tokens, cached_tokens),
        "output_tokens": 0,
        "actions": [],
    }


def test_summarize_weights_cache_hit_rate_by_input_tokens() -> None:
    rows = [
        _cache_row(0, input_tokens=3_000, cached_tokens=3_000),
        _cache_row(1, input_tokens=1_000, cached_tokens=0),
    ]
    summary = summarize(rows, SimpleNamespace(bb=10, agent_locale="zh-CN"))
    assert summary["input_tokens"] == 4_000
    assert summary["total_cached_input_tokens"] == 3_000
    assert summary["total_cache_miss_input_tokens"] == 1_000
    # Weighted rate is 0.75; the per-hand mean would be 0.5.
    assert summary["cache_hit_rate"] == pytest.approx(0.75)
    assert summary["cache_hit_rate_mean"] == pytest.approx(0.5)


def test_summarize_counts_legacy_rows_without_cache_fields_as_misses() -> None:
    row = _cache_row(0, input_tokens=1_000, cached_tokens=0)
    for key in ("cached_input_tokens", "cache_miss_input_tokens", "cache_hit_rate"):
        row.pop(key)
    summary = summarize([row], SimpleNamespace(bb=10, agent_locale="zh-CN"))
    assert summary["total_cached_input_tokens"] == 0
    assert summary["total_cache_miss_input_tokens"] == 1_000
    assert summary["cache_hit_rate"] == 0.0


def test_make_opponent_selects_scripted_policy() -> None:
    args = SimpleNamespace(opponent_kind="random")
    task = HandTask(index=0, deal_index=0, assignment=0, seed=123)
    opponent = make_opponent(args, task)
    assert isinstance(opponent, ScriptedBot)
    assert opponent.policy == "random"
    assert opponent.seed == 123
