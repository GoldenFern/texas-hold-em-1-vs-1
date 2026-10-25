"""Regression tests for the localized LLM tool schemas."""

from __future__ import annotations

from src.llm.prompt import (
    AGENT_TOOL_NAMES,
    ALL_TOOL_NAMES,
    GAME_TOOL_NAMES,
    HISTORY_TOOL_NAME,
    TABLE_TALK_TOOL_NAME,
)
from src.llm.tools import build_tools, select_tools

EXPECTED_TOOL_NAMES = [
    "estimate_hand_probabilities",
    "estimate_equity",
    "get_random_number",
    "call_ev",
    "apply_action",
]


def _descriptions(locale: str) -> dict:
    return {
        tool["function"]["name"]: tool["function"]["description"]
        for tool in build_tools(locale)
    }


def test_build_tools_keeps_the_five_base_tools_in_order() -> None:
    assert [tool["function"]["name"] for tool in build_tools()] == EXPECTED_TOOL_NAMES
    for locale in ("zh-CN", "en-US"):
        assert [
            tool["function"]["name"] for tool in build_tools(locale)
        ] == EXPECTED_TOOL_NAMES
    assert list(ALL_TOOL_NAMES) == EXPECTED_TOOL_NAMES


def test_chinese_descriptions_carry_units_and_assumptions() -> None:
    text = _descriptions("zh-CN")

    estimate = text["estimate_hand_probabilities"]
    assert "百分比" in estimate
    assert "0-100" in estimate
    assert "最终" in estimate
    assert "不是胜率" in estimate

    equity = text["estimate_equity"]
    assert "0 到 1" in equity
    assert "小数" in equity
    assert "equity = 胜率 + 平局率/2" in equity
    assert "均匀随机" in equity
    assert "精确枚举" in equity
    assert "蒙特卡洛" in equity
    assert "samples" in equity
    assert "组合数" in equity

    call_ev = text["call_ev"]
    assert "静态" in call_ev
    assert "隐含赔率" in call_ev
    assert "breakeven_equity" in call_ev
    assert "effective_pot" in call_ev
    assert "refund" in call_ev
    assert "pot_to_call_ratio" in call_ev
    assert "倍数" in call_ev
    assert "3:1" in call_ev
    assert "百分比" in call_ev


def test_english_descriptions_carry_units_and_assumptions() -> None:
    text = _descriptions("en-US")

    estimate = text["estimate_hand_probabilities"]
    assert "percent" in estimate
    assert "0-100" in estimate
    assert "final" in estimate
    assert "not equity" in estimate

    equity = text["estimate_equity"]
    assert "0-1" in equity
    assert "decimals" in equity
    assert "equity = win rate + tie rate / 2" in equity
    assert "uniformly random" in equity
    assert "enumerated exactly" in equity
    assert "Monte Carlo" in equity
    assert "samples" in equity
    assert "combinations" in equity

    call_ev = text["call_ev"]
    assert "static" in call_ev
    assert "implied odds" in call_ev
    assert "breakeven_equity" in call_ev
    assert "effective_pot" in call_ev
    assert "refund" in call_ev
    assert "pot_to_call_ratio" in call_ev
    assert "multiple" in call_ev
    assert "3:1" in call_ev
    assert "percentage" in call_ev


def test_random_tool_keeps_unit_and_tiebreak_guidance() -> None:
    zh = _descriptions("zh-CN")["get_random_number"]
    assert "[0, 1)" in zh
    assert "随机" in zh

    en = _descriptions("en-US")["get_random_number"]
    assert "[0, 1)" in en
    assert "random" in en


def test_select_tools_keeps_apply_action_last() -> None:
    for locale in ("zh-CN", "en-US"):
        names = [
            tool["function"]["name"]
            for tool in select_tools(ALL_TOOL_NAMES, locale)
        ]
        assert names == EXPECTED_TOOL_NAMES
        assert names[-1] == "apply_action"
        assert TABLE_TALK_TOOL_NAME not in names


def test_agent_tools_insert_history_and_table_talk_before_apply_action() -> None:
    expected = (
        EXPECTED_TOOL_NAMES[:-1]
        + [HISTORY_TOOL_NAME, TABLE_TALK_TOOL_NAME, "apply_action"]
    )
    for locale in ("zh-CN", "en-US"):
        names = [
            tool["function"]["name"]
            for tool in select_tools(AGENT_TOOL_NAMES, locale)
        ]
        assert names == expected
        assert names[-1] == "apply_action"


def test_game_tools_add_history_without_changing_the_baseline() -> None:
    assert list(ALL_TOOL_NAMES) == EXPECTED_TOOL_NAMES
    assert HISTORY_TOOL_NAME in GAME_TOOL_NAMES
    assert HISTORY_TOOL_NAME not in ALL_TOOL_NAMES
    for locale in ("zh-CN", "en-US"):
        names = [
            tool["function"]["name"]
            for tool in select_tools(GAME_TOOL_NAMES, locale)
        ]
        assert names == EXPECTED_TOOL_NAMES[:-1] + [
            HISTORY_TOOL_NAME,
            "apply_action",
        ]
        assert TABLE_TALK_TOOL_NAME not in names
