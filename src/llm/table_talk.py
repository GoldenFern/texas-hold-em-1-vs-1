"""Table-talk text normalization and runtime limits."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from src.i18n import normalize_locale, tr


_ZERO_WIDTH = re.compile(r"[\u200b-\u200d\u2060\ufeff]")
_BIDI = re.compile(r"[\u202a-\u202e\u2066-\u2069]")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class TableTalkLimits:
    """Per-match table-talk limits derived from ``config/game.json``."""

    max_chars_zh: int = 80
    max_chars_en: int = 160
    max_per_turn: int = 1
    history_limit: int = 6
    snapshot_limit: int = 40
    self_history_limit: int = 3

    @classmethod
    def from_runtime(cls, runtime: dict | None) -> "TableTalkLimits":
        """Read limits from the LLM runtime config, falling back to defaults."""
        runtime = runtime or {}

        def value(
            key: str,
            default: int,
            upper: int,
            minimum: int = 1,
        ) -> int:
            try:
                parsed = int(runtime.get(key, default))
            except (TypeError, ValueError):
                return default
            if parsed < minimum:
                return default
            return min(upper, parsed)

        legacy = runtime.get("table_talk_max_chars")
        legacy_default = None
        if legacy is not None:
            try:
                legacy_default = max(1, min(1000, int(legacy)))
            except (TypeError, ValueError):
                legacy_default = None
        return cls(
            max_chars_zh=value(
                "table_talk_max_chars_zh",
                legacy_default if legacy_default is not None else 80,
                1000,
            ),
            max_chars_en=value(
                "table_talk_max_chars_en",
                legacy_default if legacy_default is not None else 160,
                1000,
            ),
            max_per_turn=value("table_talk_max_per_turn", 1, 20),
            history_limit=value("table_talk_history_limit", 6, 30),
            snapshot_limit=value("table_talk_snapshot_limit", 40, 300),
            self_history_limit=value(
                "table_talk_self_history_limit",
                3,
                10,
                minimum=0,
            ),
        )

    def chars_for(self, locale: str = "zh-CN") -> int:
        """Return the character cap for the requested prompt language."""
        normalized = normalize_locale(locale, "zh-CN") or "zh-CN"
        return self.max_chars_en if normalized == "en-US" else self.max_chars_zh


def normalize_table_talk(text: str, max_chars: int = 160, locale: str = "zh-CN") -> str:
    """Normalize one table-talk message into a single safe line.

    Args:
        text: Raw text produced by the model.
        max_chars: Maximum number of characters allowed after normalization.
        locale: Locale used for error messages.

    Returns:
        A single-line string without control or zero-width characters.

    Raises:
        ValueError: If the text is empty or longer than ``max_chars``.
    """
    if not isinstance(text, str):
        raise ValueError(tr(locale, "table_talk.not_text"))
    normalized = unicodedata.normalize("NFC", text)
    normalized = _ZERO_WIDTH.sub("", normalized)
    normalized = _BIDI.sub("", normalized)
    normalized = _CONTROL.sub(" ", normalized)
    normalized = _WHITESPACE.sub(" ", normalized).strip()
    if not normalized:
        raise ValueError(tr(locale, "table_talk.empty"))
    if len(normalized) > max_chars:
        raise ValueError(tr(locale, "table_talk.too_long", max=max_chars))
    return normalized


__all__ = ["TableTalkLimits", "normalize_table_talk"]
