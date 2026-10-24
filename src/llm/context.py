"""Context window policies for the LLM poker agent.

The legacy policy trims the oldest complete decision turns whenever the
estimated prompt exceeds a hard budget.  That keeps the prompt under the
provider limit, but it changes the prefix of almost every later request and
throws away the current hand's earlier streets.

The compact policy keeps a compact one-line summary for every completed hand
and retains full detail only for the most recent hands.  Old hand blocks are
removed as whole messages, so retained assistant messages keep their original
``reasoning_content``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Sequence

_ZH_HAND_RE = re.compile(r"第\s*(\d+)\s*手")
_EN_HAND_RE = re.compile(r"Hand\s*(\d+)", re.IGNORECASE)

_CONTEXT_POLICIES = ("legacy", "full", "compact", "compact_slim")
_DETAIL_LEVELS = ("full", "compact")


@dataclass(frozen=True)
class ContextSettings:
    """Resolved context policy for one agent."""

    policy: str = "legacy"
    high_water_tokens: int = 120_000
    low_water_tokens: int = 40_000
    keep_recent_hands: int = 3
    state_detail: str = "full"
    tool_detail: str = "full"
    final_talk_context: str = "full"

    @property
    def compacts(self) -> bool:
        return self.policy in ("compact", "compact_slim")

    @classmethod
    def from_runtime(cls, runtime: dict) -> "ContextSettings":
        policy = str(runtime.get("context_policy", "legacy")).strip().lower()
        if policy not in _CONTEXT_POLICIES:
            policy = "legacy"
        state_detail = str(runtime.get("state_detail", "full")).strip().lower()
        if state_detail not in _DETAIL_LEVELS:
            state_detail = "full"
        tool_detail = str(runtime.get("tool_detail", "full")).strip().lower()
        if tool_detail not in _DETAIL_LEVELS:
            tool_detail = "full"
        if policy == "compact_slim":
            state_detail = "compact"
            tool_detail = "compact"
        final_talk_context = str(
            runtime.get("final_talk_context", "full")
        ).strip().lower()
        if final_talk_context not in ("full", "fresh"):
            final_talk_context = "full"

        def positive_int(key: str, default: int) -> int:
            try:
                value = int(runtime.get(key, default))
            except (TypeError, ValueError):
                return default
            return value if value > 0 else default

        high = positive_int("context_high_water_tokens", 120_000)
        low = positive_int("context_low_water_tokens", 40_000)
        if low > high:
            low = high
        keep = positive_int("context_keep_recent_hands", 3)
        keep = max(1, min(keep, 20))
        return cls(
            policy=policy,
            high_water_tokens=max(2_000, high),
            low_water_tokens=max(1_000, low),
            keep_recent_hands=keep,
            state_detail=state_detail,
            tool_detail=tool_detail,
            final_talk_context=final_talk_context,
        )


def parse_hand_number(
    content: str,
    turn_marker: str,
    summary_marker: str,
) -> Optional[int]:
    """Return the hand number of a turn update or hand summary message."""
    if not content:
        return None
    if not (content.startswith(turn_marker) or content.startswith(summary_marker)):
        return None
    match = _ZH_HAND_RE.search(content) or _EN_HAND_RE.search(content)
    if not match:
        return None
    try:
        return int(match.group(1))
    except (TypeError, ValueError):
        return None


def is_summary_message(content: str, summary_marker: str) -> bool:
    return bool(content) and content.startswith(summary_marker)


def compact_history(
    messages: Sequence[dict],
    keep_recent_hands: int,
    turn_marker: str,
    summary_marker: str,
) -> tuple[list[dict], int]:
    """Drop old hand detail blocks while keeping their summary messages.

    Args:
        messages: Full chronological message list, system prompt first.
        keep_recent_hands: Number of most recent hands to keep in full.
        turn_marker: Localized prefix of a turn update.
        summary_marker: Localized prefix of a hand summary.

    Returns:
        A new message list and the number of dropped messages.  The current
        hand is always retained.  Retained messages are not modified, so
        assistant reasoning content stays intact.
    """
    if len(messages) <= 1:
        return list(messages), 0

    current_hand = 0
    for message in messages:
        if message.get("role") != "user":
            continue
        content = str(message.get("content") or "")
        hand = parse_hand_number(content, turn_marker, summary_marker)
        if hand is not None:
            current_hand = max(current_hand, hand)
    if current_hand <= keep_recent_hands:
        return list(messages), 0

    cutoff = current_hand - keep_recent_hands + 1
    kept: list[dict] = [messages[0]]
    dropped = 0
    skip_old_detail = False
    for message in messages[1:]:
        content = str(message.get("content") or "")
        hand = parse_hand_number(content, turn_marker, summary_marker)
        if hand is not None:
            if hand < cutoff:
                if is_summary_message(content, summary_marker):
                    kept.append(message)
                    skip_old_detail = False
                else:
                    skip_old_detail = True
                continue
            skip_old_detail = False
        if skip_old_detail:
            dropped += 1
            continue
        kept.append(message)
    return kept, dropped


def compact_tool_payload(tool_name: str, payload: dict) -> dict:
    """Remove repeated static notes and fields derivable from table state."""
    compact = dict(payload)
    compact.pop("note", None)
    if tool_name == "estimate_hand_probabilities":
        compact.pop("needed_board_cards", None)
        current = compact.get("current_best")
        if isinstance(current, dict):
            current = dict(current)
            current.pop("score", None)
            compact["current_best"] = current
    elif tool_name == "estimate_equity":
        compact.pop("opponent", None)
        compact.pop("lose", None)
    elif tool_name == "call_ev":
        compact.pop("ev_fold", None)
        compact.pop("pot", None)
        compact.pop("to_call", None)
        compact.pop("call_amount", None)
    elif tool_name == "apply_action":
        compact.pop("label", None)
        compact.pop("message", None)
    return compact


__all__ = [
    "ContextSettings",
    "compact_history",
    "compact_tool_payload",
    "is_summary_message",
    "parse_hand_number",
]
