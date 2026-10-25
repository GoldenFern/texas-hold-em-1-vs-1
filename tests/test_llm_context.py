"""Tests for context compaction and compact prompt rendering."""

from __future__ import annotations

from dataclasses import replace

import pytest

from src.agents.base import AgentView
from src.llm.context import (
    ContextSettings,
    MatchStats,
    choose_checkpoint_cut,
    compact_history,
    compact_tool_payload,
    flatten_checkpoint_transcript,
    parse_checkpoint_range,
    parse_hand_number,
    resolve_trigger_budget,
)
from src.llm.prompt import (
    build_compact_turn_message,
    build_final_talk_summary,
    build_final_talk_summary_from_stats,
    build_hand_summary_message,
)


def make_view() -> AgentView:
    return AgentView(
        seat=0,
        name="Hero",
        hand_id=1,
        street="flop",
        hole_cards=["Ah", "Kh"],
        community=["Qh", "Jh", "2c"],
        pot=100,
        my_chips=900,
        opponent_chips=900,
        my_street_bet=0,
        opponent_street_bet=0,
        my_total_bet=50,
        opponent_total_bet=50,
        current_bet=0,
        to_call=0,
        call_amount=0,
        min_raise_to=10,
        max_raise_to=900,
        legal_actions=["fold", "check", "bet", "all_in"],
        action_history=[],
        opponent_last_action=None,
        button=0,
        small_blind=5,
        big_blind=10,
        hand_number=1,
    )

TURN = "【牌局更新 · 第 {hand} 手 · 翻牌】"
SUMMARY = "【手牌摘要 · 第 {hand} 手】你赢了；底池 30；净变化 +10；筹码 1010；公共牌 Ah Kd 7c。"


def hand_messages(hand: int, reasoning: str) -> list[dict]:
    return [
        {"role": "user", "content": TURN.format(hand=hand)},
        {
            "role": "assistant",
            "content": f"judgment {hand}",
            "reasoning_content": reasoning,
            "tool_calls": [
                {
                    "id": f"call_{hand}",
                    "type": "function",
                    "function": {"name": "apply_action", "arguments": '{"action":"check"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": f"call_{hand}", "content": '{"ok":true}'},
        {"role": "user", "content": SUMMARY.format(hand=hand)},
    ]


def test_parse_hand_number_handles_turn_and_summary() -> None:
    assert parse_hand_number(TURN.format(hand=12), "【牌局更新", "【手牌摘要") == 12
    assert parse_hand_number(SUMMARY.format(hand=7), "【牌局更新", "【手牌摘要") == 7
    assert parse_hand_number("hello", "【牌局更新", "【手牌摘要") is None
    assert (
        parse_hand_number("[Table update · Hand 9 · Flop]", "[Table update", "[Hand summary")
        == 9
    )


def test_compact_history_keeps_summaries_and_recent_hands() -> None:
    messages = [{"role": "system", "content": "rules"}]
    messages.extend(hand_messages(1, "reason one"))
    messages.extend(hand_messages(2, "reason two"))
    messages.extend(hand_messages(3, "reason three"))

    compacted, dropped = compact_history(
        messages,
        keep_recent_hands=2,
        turn_marker="【牌局更新",
        summary_marker="【手牌摘要",
    )

    contents = [str(item.get("content") or "") for item in compacted]
    assert dropped > 0
    assert TURN.format(hand=1) not in contents
    assert TURN.format(hand=2) in contents
    assert TURN.format(hand=3) in contents
    assert SUMMARY.format(hand=1) in contents
    # Retained assistant messages are untouched, including reasoning content.
    retained_assistant = [item for item in compacted if item.get("role") == "assistant"]
    assert {item["reasoning_content"] for item in retained_assistant} == {
        "reason two",
        "reason three",
    }


def test_compact_history_protects_current_hand() -> None:
    messages = [{"role": "system", "content": "rules"}]
    messages.extend(hand_messages(1, "one"))
    messages.extend(hand_messages(2, "two"))
    compacted, _ = compact_history(
        messages,
        keep_recent_hands=1,
        turn_marker="【牌局更新",
        summary_marker="【手牌摘要",
    )
    contents = [str(item.get("content") or "") for item in compacted]
    assert TURN.format(hand=2) in contents
    assert TURN.format(hand=1) not in contents
    retained_assistant = [item for item in compacted if item.get("role") == "assistant"]
    assert [item["reasoning_content"] for item in retained_assistant] == ["two"]


def test_context_settings_compact_slim_forces_both_details() -> None:
    settings = ContextSettings.from_runtime(
        {
            "context_policy": "compact_slim",
            "context_high_water_tokens": 30000,
            "context_low_water_tokens": 12000,
            "context_keep_recent_hands": 2,
        }
    )
    assert settings.compacts is True
    assert settings.state_detail == "compact"
    assert settings.tool_detail == "compact"
    assert settings.final_talk_context == "full"

    fresh = ContextSettings.from_runtime({"final_talk_context": "fresh"})
    assert fresh.final_talk_context == "fresh"


def make_result(
    hand_id: int,
    winner: int,
    net: int,
    chips: int,
    opponent_chips: int,
) -> dict:
    return {
        "hand_id": hand_id,
        "winner_seats": [winner],
        "players": [
            {"seat": 0, "net": net, "chips_after": chips},
            {"seat": 1, "net": -net, "chips_after": opponent_chips},
        ],
    }


@pytest.mark.parametrize("window", [64_000, 128_000, 1_000_000])
def test_context_settings_watermarks_stay_below_window(window: int) -> None:
    settings = ContextSettings.from_runtime(
        {
            "context_policy": "compact",
            "context_high_water_tokens": 120_000,
            "context_low_water_tokens": 40_000,
        }
    )
    clamped = settings.with_context_window(window)
    assert clamped.high_water_tokens <= window - 4096
    assert clamped.high_water_tokens >= 2_000
    assert clamped.low_water_tokens <= window - 4096
    assert clamped.low_water_tokens <= clamped.high_water_tokens
    assert clamped.low_water_tokens >= 1_000
    assert clamped.policy == settings.policy


def test_context_settings_allow_low_floor_on_tiny_window() -> None:
    settings = ContextSettings(
        high_water_tokens=120_000,
        low_water_tokens=40_000,
    )
    clamped = settings.with_context_window(8_000)
    assert clamped.high_water_tokens == 3_904
    assert clamped.low_water_tokens == 1_000


def test_match_stats_accumulates_results() -> None:
    results = [
        make_result(1, 0, 60, 1060, 940),
        make_result(2, 1, -40, 1020, 980),
    ]
    stats = MatchStats()
    for item in results:
        stats.update(item, 0)

    assert stats == MatchStats(
        hands=2,
        wins=1,
        losses=1,
        net=20,
        chips=1020,
        opponent_chips=980,
    )
    assert MatchStats.from_results(results, 0) == stats


def test_final_talk_summary_from_stats_matches_history_summary() -> None:
    results = [
        make_result(1, 0, 60, 1060, 940),
        make_result(2, 1, -40, 1020, 980),
    ]
    stats = MatchStats.from_results(results, 0)
    history_text = build_final_talk_summary(results, 0, locale="zh-CN")
    stats_text = build_final_talk_summary_from_stats(stats, locale="zh-CN")

    assert stats_text == history_text
    assert "共 2 手" in stats_text
    assert "你赢 1 手，输 1 手" in stats_text
    assert "整场净变化 +20" in stats_text
    assert "盈利" in stats_text
    assert "1020" in stats_text and "980" in stats_text

    english = build_final_talk_summary_from_stats(stats, locale="en-US")
    assert "2 hands" in english
    assert "profit" in english

    losing = MatchStats(
        hands=2,
        wins=1,
        losses=1,
        net=-20,
        chips=980,
        opponent_chips=1020,
    )
    assert "亏损" in build_final_talk_summary_from_stats(losing, locale="zh-CN")
    assert "loss" in build_final_talk_summary_from_stats(losing, locale="en-US")


def test_compact_tool_payload_removes_static_notes() -> None:
    equity = {
        "ok": True,
        "equity": 0.62,
        "win": 0.6,
        "tie": 0.04,
        "lose": 0.36,
        "opponent": "random",
        "note": "long note",
    }
    compact = compact_tool_payload("estimate_equity", equity)
    assert "note" not in compact
    assert "opponent" not in compact
    assert "lose" not in compact
    assert compact["equity"] == 0.62

    distribution = {
        "ok": True,
        "needed_board_cards": 2,
        "current_best": {"category": "high_card", "score": 123, "best_five": ["Ah"]},
        "note": "long note",
    }
    compact = compact_tool_payload("estimate_hand_probabilities", distribution)
    assert "needed_board_cards" not in compact
    assert "score" not in compact["current_best"]


def test_compact_turn_message_drops_repeated_history_fields() -> None:
    view = make_view()
    view.action_history = [
        {"seat": 0, "type": "raise", "amount": 30, "street": "flop"},
        {"seat": 1, "type": "call", "amount": 30, "street": "flop"},
    ]
    view.history_summary = [
        {
            "hand_id": 1,
            "you_won": True,
            "reason": "showdown",
            "pot": 100,
            "board": ["Ah", "Kd"],
            "your_net": 20,
            "your_chips_after": 1020,
        }
    ]
    text = build_compact_turn_message(view, locale="zh-CN")
    assert "recent_hands" not in text
    assert "match_note" not in text
    assert "actions_this_hand" not in text
    assert "recent_actions" in text
    assert '"pot":100' in text


def test_hand_summary_is_compact_and_localized() -> None:
    result = {
        "hand_id": 4,
        "winner_seats": [0],
        "pot": 120,
        "reason": "showdown",
        "board": ["Ah", "Kd", "7c", "3s", "2h"],
        "players": [
            {"seat": 0, "net": 60, "chips_after": 1060},
            {"seat": 1, "net": -60, "chips_after": 940},
        ],
        "showdown": [
            {"seat": 0, "category": "pair", "category_name": "一对"},
            {"seat": 1, "category": "high_card", "category_name": "高牌"},
        ],
    }
    text = build_hand_summary_message(result, 0, locale="zh-CN")
    assert text.startswith("【手牌摘要 · 第 4 手】")
    assert "你赢了" in text
    assert "一对" in text
    assert "高牌" in text
    assert "1060" in text


def test_checkpoint_settings_parse_and_force_compact_details() -> None:
    settings = ContextSettings.from_runtime(
        {
            "context_policy": "checkpoint",
            "context_high_water_tokens": 0,
            "context_low_water_tokens": 50_000,
            "context_keep_recent_hands": 3,
            "context_keep_recent_tokens": 20_000,
            "context_buffer_tokens": 0,
            "context_summary_max_tokens": 4_096,
            "context_summary_retries": [0.5, 0.9, 1.2, -0.1],
        }
    )
    assert settings.policy == "checkpoint"
    assert settings.high_water_tokens == 0
    assert settings.low_water_tokens == 50_000
    assert settings.keep_recent_tokens == 20_000
    assert settings.summary_max_tokens == 4_096
    assert settings.summary_retry_steps == (0.9, 0.5)
    assert settings.state_detail == "compact"
    assert settings.tool_detail == "compact"
    assert settings.uses_checkpoint is True


def test_trigger_budget_matches_opencode_window_minus_buffer() -> None:
    settings = ContextSettings.from_runtime(
        {
            "context_policy": "checkpoint",
            "context_high_water_tokens": 0,
            "context_low_water_tokens": 50_000,
        }
    )
    million = resolve_trigger_budget(settings, 1_013_760, 1_048_576)
    assert million.buffer_tokens == 157_286
    assert million.auto_ceiling_tokens == 856_474
    assert million.high_water_tokens == 856_474
    assert million.hard_water_tokens == 1_009_664
    assert million.auto_enabled is True

    capped = replace(settings, high_water_tokens=160_000)
    assert resolve_trigger_budget(capped, 1_013_760, 1_048_576).high_water_tokens == 160_000

    kimi = resolve_trigger_budget(settings, 221_184, 262_144)
    assert kimi.buffer_tokens == 39_321
    assert kimi.auto_ceiling_tokens == 181_863


def test_trigger_budget_handles_unknown_and_tiny_windows() -> None:
    settings = ContextSettings.from_runtime(
        {"context_policy": "checkpoint", "context_high_water_tokens": 0}
    )
    unknown = resolve_trigger_budget(settings, 220_000, 0)
    assert unknown.auto_enabled is False
    assert unknown.high_water_tokens == 220_000

    tiny = resolve_trigger_budget(settings, 3_000, 4_096)
    assert tiny.high_water_tokens >= 2_000
    assert tiny.hard_water_tokens >= tiny.high_water_tokens
    assert tiny.low_water_tokens >= 1_000


def test_choose_checkpoint_cut_keeps_recent_and_current_hands() -> None:
    messages = [{"role": "system", "content": "rules"}]
    for hand in range(1, 7):
        messages.extend(hand_messages(hand, f"reason {hand}"))
    messages.append({"role": "user", "content": TURN.format(hand=7)})

    cut = choose_checkpoint_cut(
        messages,
        current_hand=7,
        keep_recent_hands=3,
        keep_recent_tokens=10_000,
        turn_marker="【牌局更新",
        summary_marker="【手牌摘要",
        estimate_messages=lambda items: len(items) * 100,
    )
    assert cut is not None
    assert cut.through == 3
    assert cut.retain_from == 4
    assert TURN.format(hand=4) in str(messages[cut.index]["content"])
    assert TURN.format(hand=7) in str(messages[-1]["content"])


def test_choose_checkpoint_cut_shrinks_by_token_budget() -> None:
    messages = [{"role": "system", "content": "rules"}]
    for hand in range(1, 7):
        messages.extend(hand_messages(hand, f"reason {hand}"))
    messages.append({"role": "user", "content": TURN.format(hand=7)})

    cut = choose_checkpoint_cut(
        messages,
        current_hand=7,
        keep_recent_hands=3,
        keep_recent_tokens=900,
        turn_marker="【牌局更新",
        summary_marker="【手牌摘要",
        estimate_messages=lambda items: len(items) * 100,
    )
    assert cut is not None
    assert cut.through >= 3
    assert cut.retain_from >= 4
    # The current hand is never dropped, even when the tail is over budget.
    assert TURN.format(hand=7) in str(messages[-1]["content"])


def test_choose_checkpoint_cut_skips_the_first_hand() -> None:
    messages = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": TURN.format(hand=1)},
    ]
    cut = choose_checkpoint_cut(
        messages,
        current_hand=1,
        keep_recent_hands=3,
        keep_recent_tokens=10_000,
        turn_marker="【牌局更新",
        summary_marker="【手牌摘要",
        estimate_messages=lambda items: len(items) * 100,
    )
    assert cut is None


def test_parse_checkpoint_range_zh_and_en() -> None:
    assert parse_checkpoint_range(
        "【比赛 checkpoint · 第 1 手至第 42 手】",
        "【比赛 checkpoint",
    ) == (1, 42)
    assert parse_checkpoint_range(
        "[Match checkpoint · Hands 1-42]",
        "[Match checkpoint",
    ) == (1, 42)
    assert parse_checkpoint_range("not a checkpoint", "【比赛 checkpoint") is None
    assert parse_checkpoint_range(
        "【比赛 checkpoint · 第 9 手至第 3 手】",
        "【比赛 checkpoint",
    ) is None


def test_flatten_checkpoint_transcript_drops_oldest_and_skips_reasoning() -> None:
    messages = [{"role": "system", "content": "rules"}]
    for hand in range(1, 5):
        messages.extend(hand_messages(hand, f"SECRET_REASON_{hand}"))
    cut_index = next(
        index
        for index, message in enumerate(messages)
        if TURN.format(hand=4) in str(message.get("content") or "")
    )
    text = flatten_checkpoint_transcript(
        messages,
        cut_index,
        target_tokens=80,
        estimate_text=len,
        checkpoint_marker="【比赛 checkpoint",
    )
    assert "SECRET_REASON_4" not in text
    assert "older hands omitted" in text
    assert "第 3 手" in text or "第 2 手" in text


def test_compact_history_tool_payload_keeps_metrics_and_five_results() -> None:
    payload = {
        "ok": True,
        "scope": "both",
        "current_match": {
            "hands": 10,
            "positions": {"you": {}, "opponent": {}},
            "recent_hands": [{"hand_id": index} for index in range(10)],
        },
        "notes": ["public only"],
    }
    compact = compact_tool_payload("research_history", payload)
    assert "scope" not in compact
    assert "positions" not in compact["current_match"]
    assert len(compact["current_match"]["recent_hands"]) == 5
    assert compact["notes"] == ["public only"]
