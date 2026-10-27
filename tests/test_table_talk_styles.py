"""Tests for the table-talk style registry and prompt injection."""

from __future__ import annotations

import pytest

from src.llm.prompt import ALL_TOOL_NAMES, build_system_prompt
from src.llm.table_talk_styles import (
    DEFAULT_STYLE_ID,
    STYLE_ORDER,
    TABLE_TALK_STYLES,
    list_styles,
    resolve_style,
)


def test_registry_has_expected_defaults_and_order() -> None:
    assert DEFAULT_STYLE_ID == "gentle"
    assert list(STYLE_ORDER) == [
        "gentle",
        "tieba",
        "tieba_l3",
        "sarcastic",
        "newbie",
        "veteran",
        "chuuni",
    ]
    assert TABLE_TALK_STYLES["gentle"].default is True
    assert [item["id"] for item in list_styles()] == list(STYLE_ORDER)
    assert sum(1 for item in list_styles() if item["default"]) == 1
    assert TABLE_TALK_STYLES["tieba_l3"].intensity == 3


def test_every_style_has_bilingual_metadata_and_prompt() -> None:
    for style in TABLE_TALK_STYLES.values():
        assert style.name and style.name_en
        assert style.description and style.description_en
        assert style.sample and style.sample_en
        assert style.prompt and style.prompt_en
        assert style.final_line and style.final_line_en
        assert 0 <= style.intensity <= 3


def test_resolve_style_is_case_insensitive_and_rejects_unknown() -> None:
    assert resolve_style("").id == DEFAULT_STYLE_ID
    assert resolve_style("TIEBA").id == "tieba"
    with pytest.raises(ValueError, match="未知的桌边喊话风格"):
        resolve_style("nope", "zh-CN")
    with pytest.raises(ValueError, match="Unknown table-talk style"):
        resolve_style("nope", "en-US")


def test_style_card_is_inserted_only_when_table_talk_is_enabled() -> None:
    base = build_system_prompt(ALL_TOOL_NAMES)
    assert "【说话风格】" not in base

    for style in TABLE_TALK_STYLES.values():
        prompt = build_system_prompt(
            ("apply_action",),
            table_talk_mode="inline",
            table_talk_style=style.id,
        )
        assert "【说话风格】" in prompt
        assert style.prompt.splitlines()[0] in prompt

    english = build_system_prompt(
        ("apply_action",),
        table_talk_mode="inline",
        table_talk_style="chuuni",
        locale="en-US",
    )
    assert "[Speaking style]" in english
    assert "full-blown chuunibyou" in english
