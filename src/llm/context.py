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
from dataclasses import dataclass, replace
from typing import Callable, Mapping, Optional, Sequence

_ZH_HAND_RE = re.compile(r"第\s*(\d+)\s*手")
_EN_HAND_RE = re.compile(r"Hand\s*(\d+)", re.IGNORECASE)
_ZH_CHECKPOINT_RE = re.compile(r"第\s*(\d+)\s*手\s*至\s*第\s*(\d+)\s*手")
_EN_CHECKPOINT_RE = re.compile(
    r"Hands?\s*(\d+)\s*[-–]\s*(\d+)", re.IGNORECASE
)

_CONTEXT_POLICIES = ("legacy", "full", "compact", "compact_slim", "checkpoint")
_DETAIL_LEVELS = ("full", "compact")


@dataclass(frozen=True)
class ContextSettings:
    """Resolved context policy for one agent."""

    policy: str = "legacy"
    high_water_tokens: int = 120_000
    low_water_tokens: int = 40_000
    keep_recent_hands: int = 3
    keep_recent_tokens: int = 20_000
    buffer_tokens: int = 0
    summary_max_tokens: int = 4_096
    summary_retry_steps: tuple[float, ...] = (0.85, 0.7, 0.5, 0.35)
    state_detail: str = "full"
    tool_detail: str = "full"
    final_talk_context: str = "full"

    @property
    def compacts(self) -> bool:
        return self.policy in ("compact", "compact_slim", "checkpoint")

    @property
    def uses_checkpoint(self) -> bool:
        return self.policy == "checkpoint"

    def with_context_window(
        self,
        max_context_tokens: int,
        margin: int = 4096,
    ) -> "ContextSettings":
        """Clamp both watermarks below the effective model context budget.

        Args:
            max_context_tokens: Prompt budget already reduced by the reserved
                output tokens.
            margin: Headroom kept free below the prompt budget.

        Returns:
            A copy with watermarks that cannot exceed the budget.  The high
            watermark is at least 2000 tokens and the low watermark is at
            least 1000 tokens; when the high watermark is tiny, the low
            watermark is allowed to sit at 1000 instead of going negative.
        """
        high = max(2_000, min(self.high_water_tokens, max_context_tokens - margin))
        low = max(1_000, min(self.low_water_tokens, high - margin))
        return replace(
            self,
            high_water_tokens=high,
            low_water_tokens=min(low, high),
        )

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
        if policy in ("compact_slim", "checkpoint"):
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

        def non_negative_int(key: str, default: int) -> int:
            try:
                value = int(runtime.get(key, default))
            except (TypeError, ValueError):
                return default
            return value if value >= 0 else default

        def retry_steps(value: object) -> tuple[float, ...]:
            if isinstance(value, (list, tuple)):
                parsed: list[float] = []
                for item in value:
                    try:
                        step = float(item)
                    except (TypeError, ValueError):
                        continue
                    if 0.0 < step < 1.0:
                        parsed.append(step)
                if parsed:
                    parsed.sort(reverse=True)
                    return tuple(parsed)
            return (0.85, 0.7, 0.5, 0.35)

        raw_high = non_negative_int("context_high_water_tokens", 0)
        if policy == "checkpoint":
            high = raw_high
            low_default = 50_000
        else:
            high = raw_high if raw_high > 0 else 120_000
            low_default = 40_000
        low = positive_int("context_low_water_tokens", low_default)
        if high > 0 and low > high:
            low = high
        keep = positive_int("context_keep_recent_hands", 3)
        keep = max(1, min(keep, 20))
        keep_tokens = positive_int("context_keep_recent_tokens", 20_000)
        buffer_tokens = non_negative_int("context_buffer_tokens", 0)
        summary_max = positive_int("context_summary_max_tokens", 4_096)
        return cls(
            policy=policy,
            high_water_tokens=max(2_000, high) if high > 0 else 0,
            low_water_tokens=max(1_000, low),
            keep_recent_hands=keep,
            keep_recent_tokens=max(1_000, keep_tokens),
            buffer_tokens=buffer_tokens,
            summary_max_tokens=max(512, summary_max),
            summary_retry_steps=retry_steps(
                runtime.get("context_summary_retries")
            ),
            state_detail=state_detail,
            tool_detail=tool_detail,
            final_talk_context=final_talk_context,
        )


@dataclass(frozen=True)
class TriggerBudget:
    """Resolved checkpoint trigger thresholds for one model."""

    high_water_tokens: int
    hard_water_tokens: int
    low_water_tokens: int
    buffer_tokens: int
    auto_ceiling_tokens: int
    auto_enabled: bool


def resolve_trigger_budget(
    settings: ContextSettings,
    prompt_budget: int,
    context_window: int,
) -> TriggerBudget:
    """Resolve the OpenCode-style window-minus-buffer trigger.

    Args:
        settings: Context settings with the optional buffer and high-water cap.
        prompt_budget: Prompt budget after reserving output tokens.
        context_window: Raw model context window, or 0 when unknown.

    Returns:
        Trigger thresholds.  A positive ``high_water_tokens`` in the settings
        acts as a cost cap; zero means use the automatic ceiling.
    """
    prompt_budget = max(2_000, int(prompt_budget or 0))
    window = max(0, int(context_window or 0))
    auto_enabled = window > 0
    if window > 0:
        if settings.buffer_tokens > 0:
            buffer = int(settings.buffer_tokens)
        else:
            share = int(window * 0.15)
            buffer = max(share, 16_000 if window >= 32_000 else 0)
    else:
        buffer = 0
    buffer = min(max(0, buffer), max(0, prompt_budget - 2_000))
    auto_ceiling = max(2_000, prompt_budget - buffer)
    configured = int(settings.high_water_tokens or 0)
    high = auto_ceiling if configured <= 0 else min(configured, auto_ceiling)
    high = max(2_000, high)
    hard = max(high, prompt_budget - 4_096)
    low = max(1_000, min(int(settings.low_water_tokens or 0), high))
    return TriggerBudget(
        high_water_tokens=high,
        hard_water_tokens=hard,
        low_water_tokens=low,
        buffer_tokens=buffer,
        auto_ceiling_tokens=auto_ceiling,
        auto_enabled=auto_enabled,
    )


@dataclass(frozen=True)
class CheckpointCut:
    """Where to split the history for a checkpoint summary."""

    index: int
    retain_from: int
    through: int


def find_hand_starts(
    messages: Sequence[dict],
    turn_marker: str,
    summary_marker: str,
) -> dict[int, int]:
    """Return the first message index for each hand's table update."""
    starts: dict[int, int] = {}
    for index, message in enumerate(messages):
        if message.get("role") != "user":
            continue
        content = str(message.get("content") or "")
        if not content.startswith(turn_marker):
            continue
        hand = parse_hand_number(content, turn_marker, summary_marker)
        if hand is not None:
            starts.setdefault(hand, index)
    return starts


def choose_checkpoint_cut(
    messages: Sequence[dict],
    current_hand: int,
    keep_recent_hands: int,
    keep_recent_tokens: int,
    turn_marker: str,
    summary_marker: str,
    estimate_messages: Callable[[Sequence[dict]], int],
) -> Optional[CheckpointCut]:
    """Choose the retained tail and the summary cutoff.

    The current hand is never summarized.  Older completed hands are dropped
    from the tail oldest-first until the tail fits the token budget; the
    checkpoint still covers every completed hand before the tail.

    Args:
        messages: Full chronological message list, system prompt first.
        current_hand: Hand currently being played, or the next hand number.
        keep_recent_hands: Minimum number of completed hands to retain.
        keep_recent_tokens: Approximate token cap for the retained tail.
        turn_marker: Localized prefix of a table update.
        summary_marker: Localized prefix of a hand summary.
        estimate_messages: Token estimator for a message slice.

    Returns:
        A split, or ``None`` when there is nothing older to summarize.
    """
    current_hand = max(1, int(current_hand))
    last_completed = current_hand - 1
    if last_completed < 1:
        return None
    starts = find_hand_starts(messages, turn_marker, summary_marker)
    if not starts:
        return None
    candidate = max(1, last_completed - max(1, int(keep_recent_hands)) + 1)
    available = sorted(starts)
    if candidate not in starts:
        later = [hand for hand in available if hand >= candidate]
        if not later:
            return None
        candidate = later[0]
    cut_index = starts[candidate]
    while True:
        tail = messages[cut_index:]
        if estimate_messages(tail) <= max(1_000, int(keep_recent_tokens)):
            break
        if candidate >= current_hand:
            break
        later = [hand for hand in available if hand > candidate]
        if not later:
            break
        candidate = later[0]
        cut_index = starts[candidate]
    through = candidate - 1
    if through < 1:
        return None
    return CheckpointCut(index=cut_index, retain_from=candidate, through=through)


def parse_checkpoint_range(
    content: str,
    checkpoint_marker: str,
) -> Optional[tuple[int, int]]:
    """Return the inclusive hand range stored in a checkpoint header."""
    if not content or not str(content).startswith(checkpoint_marker):
        return None
    text = str(content)
    match = _ZH_CHECKPOINT_RE.search(text) or _EN_CHECKPOINT_RE.search(text)
    if not match:
        return None
    try:
        start = int(match.group(1))
        end = int(match.group(2))
    except (TypeError, ValueError):
        return None
    if start < 1 or end < start:
        return None
    return start, end


def checkpoint_through_hand(content: str, checkpoint_marker: str) -> Optional[int]:
    """Return the last hand covered by a checkpoint message."""
    parsed = parse_checkpoint_range(content, checkpoint_marker)
    return parsed[1] if parsed else None


def is_checkpoint_message(content: str, checkpoint_marker: str) -> bool:
    """Return whether a user message is a checkpoint."""
    return bool(content) and str(content).startswith(checkpoint_marker)


def checkpoint_message_indices(
    messages: Sequence[dict],
    checkpoint_marker: str,
) -> list[int]:
    """Return indices of all checkpoint messages."""
    return [
        index
        for index, message in enumerate(messages)
        if message.get("role") == "user"
        and is_checkpoint_message(str(message.get("content") or ""), checkpoint_marker)
    ]


def _clip_text(value: object, limit: int = 1_200) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    head = max(1, limit // 2)
    tail = max(0, limit - head)
    omitted = len(text) - head - tail
    return f"{text[:head]}\n[… {omitted} characters omitted …]\n{text[-tail:] if tail else ''}"


def _render_summary_message(message: Mapping) -> str:
    role = str(message.get("role") or "")
    content = message.get("content")
    if role == "assistant":
        parts: list[str] = []
        if isinstance(content, str) and content.strip():
            parts.append(f"[Assistant]: {content.strip()}")
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            parts.append(
                f"[Assistant tool call]: {function.get('name')}"
                f"({_clip_text(function.get('arguments'), 800)})"
            )
        return "\n".join(parts)
    if role == "tool":
        return f"[Tool result]: {_clip_text(content, 1_200)}"
    if role == "user":
        return f"[User]: {_clip_text(content, 2_000)}"
    return ""


def flatten_checkpoint_transcript(
    messages: Sequence[dict],
    cut_index: int,
    target_tokens: int,
    estimate_text: Callable[[str], int],
    checkpoint_marker: str,
    turn_marker: str = "【牌局更新",
    summary_marker: str = "【手牌摘要",
) -> str:
    """Flatten the history before ``cut_index`` into a short text transcript.

    The summary request uses this only after the unchanged and shorter-prefix
    attempts fail.  Oldest hand blocks are dropped first, and tool output is
    clipped.  The newest block is always kept.
    """
    cut_index = max(0, min(int(cut_index), len(messages)))
    starts = find_hand_starts(
        messages[:cut_index], turn_marker, summary_marker
    )
    ordered = sorted(starts.items())
    first_start = ordered[0][1] if ordered else cut_index
    prefix = list(messages[:first_start])
    blocks: list[tuple[int, str]] = []
    for position, (hand, start) in enumerate(ordered):
        end = ordered[position + 1][1] if position + 1 < len(ordered) else cut_index
        block = "\n".join(
            rendered
            for rendered in (
                _render_summary_message(message) for message in messages[start:end]
            )
            if rendered
        )
        if block:
            blocks.append((hand, block))
    prefix_text = "\n".join(
        rendered
        for rendered in (_render_summary_message(message) for message in prefix)
        if rendered
    )
    omitted = 0
    while blocks and estimate_text(
        (prefix_text + "\n\n" if prefix_text else "")
        + "\n\n".join(block for _, block in blocks)
    ) > max(1, int(target_tokens)):
        if len(blocks) <= 1:
            break
        blocks.pop(0)
        omitted += 1
    note = f"[{omitted} older hands omitted]\n\n" if omitted else ""
    parts = [part for part in (prefix_text, note + "\n\n".join(block for _, block in blocks)) if part]
    return "\n\n".join(parts)


@dataclass
class MatchStats:
    """Aggregated match results for one seat.

    Attributes:
        hands: Number of completed hands.
        wins: Hands won by the seat.
        losses: Hands not won by the seat.
        net: Sum of chip net changes across all hands.
        chips: Chips after the most recent hand.
        opponent_chips: Opponent chips after the most recent hand.
    """

    hands: int = 0
    wins: int = 0
    losses: int = 0
    net: int = 0
    chips: int = 0
    opponent_chips: int = 0

    def update(self, result: dict, seat: int) -> None:
        """Accumulate one completed hand result.

        Args:
            result: Engine hand result with ``winner_seats`` and ``players``.
            seat: Seat whose perspective is being summarized.
        """
        if not result:
            return
        self.hands += 1
        if seat in (result.get("winner_seats") or []):
            self.wins += 1
        else:
            self.losses += 1
        players = {
            entry.get("seat"): entry for entry in result.get("players") or []
        }
        mine = players.get(seat) or {}
        theirs = players.get(1 - seat) or {}
        self.net += int(mine.get("net", 0) or 0)
        if "chips_after" in mine:
            self.chips = int(mine.get("chips_after") or 0)
        if "chips_after" in theirs:
            self.opponent_chips = int(theirs.get("chips_after") or 0)

    @classmethod
    def from_results(cls, hand_results: Sequence[dict], seat: int) -> "MatchStats":
        """Build aggregate stats from a sequence of hand results.

        Args:
            hand_results: Chronological hand results for one seat.
            seat: Seat whose perspective is being summarized.

        Returns:
            Aggregate stats for the supplied results.
        """
        stats = cls()
        for result in hand_results:
            if result:
                stats.update(result, seat)
        return stats


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
    elif tool_name == "research_history":
        compact.pop("scope", None)
        current = compact.get("current_match")
        if isinstance(current, dict):
            current = dict(current)
            current.pop("positions", None)
            recent = current.get("recent_hands")
            if isinstance(recent, list) and len(recent) > 5:
                current["recent_hands"] = recent[-5:]
            compact["current_match"] = current
    return compact


__all__ = [
    "CheckpointCut",
    "ContextSettings",
    "MatchStats",
    "TriggerBudget",
    "checkpoint_message_indices",
    "checkpoint_through_hand",
    "choose_checkpoint_cut",
    "compact_history",
    "compact_tool_payload",
    "find_hand_starts",
    "flatten_checkpoint_transcript",
    "is_checkpoint_message",
    "is_summary_message",
    "parse_checkpoint_range",
    "parse_hand_number",
    "resolve_trigger_budget",
]
