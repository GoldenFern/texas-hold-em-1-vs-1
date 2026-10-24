"""Tests for context compaction and compact prompt rendering."""

from __future__ import annotations

from src.agents.base import AgentView
from src.llm.context import (
    ContextSettings,
    compact_history,
    compact_tool_payload,
    parse_hand_number,
)
from src.llm.prompt import (
    build_compact_turn_message,
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
