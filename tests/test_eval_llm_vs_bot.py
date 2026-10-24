"""Unit tests for the offline LLM-vs-bot evaluation helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts.eval_llm_vs_bot import (
    HandTask,
    build_tool_recommendation,
    count_auto_turns,
    estimate_hand_cost,
    hand_tool_stats,
    make_opponent,
    resolve_tool_names,
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


def test_make_opponent_selects_scripted_policy() -> None:
    args = SimpleNamespace(opponent_kind="random")
    task = HandTask(index=0, deal_index=0, assignment=0, seed=123)
    opponent = make_opponent(args, task)
    assert isinstance(opponent, ScriptedBot)
    assert opponent.policy == "random"
    assert opponent.seed == 123
