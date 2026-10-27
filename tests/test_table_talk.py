"""Tests for table-talk normalization and runtime limits."""

from __future__ import annotations

import pytest

from src.llm.table_talk import TableTalkLimits, normalize_table_talk


def test_normalize_collapses_lines_and_control_characters() -> None:
    text = normalize_table_talk("  我这次\n不会再让你\x00便宜看牌。  ", 160)
    assert text == "我这次 不会再让你 便宜看牌。"


def test_normalize_strips_zero_width_characters() -> None:
    assert normalize_table_talk("你\u200b信\u200d吗\ufeff", 160) == "你信吗"


def test_normalize_rejects_empty_text() -> None:
    with pytest.raises(ValueError, match="不能为空"):
        normalize_table_talk(" \n\t ", 160)


def test_normalize_rejects_long_text() -> None:
    with pytest.raises(ValueError, match="不能超过 5 字"):
        normalize_table_talk("一二三四五六", 5)


def test_normalize_does_not_inspect_card_content() -> None:
    # Table talk is prompt-constrained, not content-filtered. A model may still
    # say something that looks like a real card; the server only formats it.
    assert normalize_table_talk("我拿着 A♥，你信吗？", 160) == "我拿着 A♥，你信吗？"


def test_limits_read_from_runtime_and_clamp() -> None:
    limits = TableTalkLimits.from_runtime(
        {
            "table_talk_max_chars_zh": 80,
            "table_talk_max_chars_en": 180,
            "table_talk_max_per_turn": 3,
            "table_talk_history_limit": 4,
            "table_talk_self_history_limit": 5,
            "table_talk_snapshot_limit": 20,
        }
    )
    assert limits.max_chars_zh == 80
    assert limits.max_chars_en == 180
    assert limits.chars_for("zh-CN") == 80
    assert limits.chars_for("en-US") == 180
    assert limits.max_per_turn == 3
    assert limits.history_limit == 4
    assert limits.self_history_limit == 5
    assert limits.snapshot_limit == 20

    legacy = TableTalkLimits.from_runtime({"table_talk_max_chars": 120})
    assert legacy.max_chars_zh == 120
    assert legacy.max_chars_en == 120

    fallback = TableTalkLimits.from_runtime(
        {
            "table_talk_max_chars_zh": "many",
            "table_talk_max_chars_en": -5,
            "table_talk_max_per_turn": -3,
        }
    )
    assert fallback.max_chars_zh == 80
    assert fallback.max_chars_en == 160
    assert fallback.max_per_turn == 1
    assert fallback.self_history_limit == 3
