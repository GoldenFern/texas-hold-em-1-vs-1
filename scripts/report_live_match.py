"""Build statistics, charts and a markdown report from a live match run."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.history import HistoryStats
from src.engine.cards import Card
from src.engine.evaluator import assess_hand

OPTIONAL_TOOLS = (
    "research_history",
    "estimate_hand_probabilities",
    "estimate_equity",
    "get_random_number",
    "call_ev",
)
FORCED_TOOLS = ("send_table_message", "apply_action")
ACTION_LABELS = {
    "fold": "弃牌",
    "check": "过牌",
    "call": "跟注",
    "bet": "下注",
    "raise": "加注",
    "all_in": "全下",
}
STREET_LABELS = {
    "preflop": "翻前",
    "flop": "翻牌",
    "turn": "转牌",
    "river": "河牌",
    "showdown": "摊牌",
    "finished": "结束",
}
STRENGTH_RANK = {
    "high_card": 0,
    "one_pair": 1,
    "two_pair": 2,
    "three_of_a_kind": 3,
    "straight": 4,
    "flush": 5,
    "full_house": 6,
    "four_of_a_kind": 7,
    "straight_flush": 8,
    "royal_flush": 9,
}
SEAT_COLORS = {0: "#2F9E8F", 1: "#D99A45"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Live match output directory")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return rows


def pct(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def pct_points(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def number(value: Optional[float], digits: int = 1) -> str:
    if value is None:
        return "n/a"
    return f"{value:,.{digits}f}"


def safe_div(numerator: float, denominator: float) -> Optional[float]:
    if not denominator:
        return None
    return numerator / denominator


def percentile(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def parse_turn_state(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.S)
    payload = match.group(1) if match else None
    if payload is None:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return None
        payload = text[start : end + 1]
    try:
        data = json.loads(payload)
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def build_turns(events: Sequence[Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    turns: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for event in events:
        kind = event.get("type")
        turn_id = event.get("turn_id")
        if kind == "turn_start" and turn_id:
            turns[turn_id] = {
                "turn_id": turn_id,
                "seat": int(event.get("seat", -1)),
                "hand_id": event.get("hand_id"),
                "street": event.get("street"),
                "pot": event.get("pot"),
                "to_call": event.get("to_call"),
                "position": event.get("position"),
                "start_ts": event.get("_ts"),
                "messages": [],
                "tools": [],
                "tool_results": [],
                "state": None,
                "end": None,
            }
            order.append(turn_id)
            continue
        turn = turns.get(turn_id) if turn_id else None
        if turn is None:
            continue
        if kind == "turn_state":
            turn["state"] = parse_turn_state(str(event.get("text") or ""))
        elif kind == "message_end":
            turn["messages"].append(event)
        elif kind == "tool_call":
            turn["tools"].append(event)
        elif kind == "tool_result":
            turn["tool_results"].append(event)
        elif kind == "turn_end":
            turn["end"] = event
    return turns, order


def build_requests(
    events: Sequence[Dict[str, Any]],
    turns: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    requests: List[Dict[str, Any]] = []
    phase = 0
    for event in events:
        if event.get("type") == "context_compacted":
            phase += 1
            continue
        if event.get("type") != "message_end":
            continue
        turn = turns.get(str(event.get("turn_id") or ""))
        usage = event.get("usage") if isinstance(event.get("usage"), dict) else {}
        input_tokens = int(usage.get("input_tokens", 0) or 0)
        cached = int(usage.get("cached_input_tokens", 0) or 0)
        requests.append(
            {
                "ts": float(event.get("_ts") or 0.0),
                "seat": turn.get("seat") if turn else None,
                "hand_id": turn.get("hand_id") if turn else None,
                "turn_id": event.get("turn_id"),
                "street": turn.get("street") if turn else None,
                "input_tokens": input_tokens,
                "cached_input_tokens": cached,
                "cache_miss_input_tokens": int(
                    usage.get("cache_miss_input_tokens", max(0, input_tokens - cached)) or 0
                ),
                "output_tokens": int(usage.get("output_tokens", 0) or 0),
                "reasoning_tokens": int(usage.get("reasoning_tokens", 0) or 0),
                "latency_ms": float(event.get("latency_ms", 0.0) or 0.0),
                "cache_hit": safe_div(cached, input_tokens),
                "phase": phase,
                "estimated_usage": bool(event.get("estimated_usage")),
                "error": bool(event.get("error")),
            }
        )
    return requests


def usage_totals(items: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    totals = {
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "cache_miss_input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
    }
    for item in items:
        usage = item.get("usage") if isinstance(item.get("usage"), dict) else item
        for key in totals:
            totals[key] += int(usage.get(key, 0) or 0)
    return totals


def cost_of(usage: Dict[str, int], prices: Dict[str, float]) -> float:
    input_tokens = int(usage.get("input_tokens", 0) or 0)
    cached = int(usage.get("cached_input_tokens", 0) or 0)
    uncached = max(0, input_tokens - cached)
    output = int(usage.get("output_tokens", 0) or 0)
    return (
        uncached * float(prices.get("input_per_m", 0.0))
        + cached * float(prices.get("cache_input_per_m", 0.0))
        + output * float(prices.get("output_per_m", 0.0))
    ) / 1_000_000.0


def rate(items: Sequence[Dict[str, Any]], key: str) -> Optional[float]:
    total = sum(int(item.get(key, 0) or 0) for item in items)
    cached = sum(int(item.get("cached_input_tokens", 0) or 0) for item in items)
    if not total:
        return None
    return cached / total


def group_by(items: Sequence[Dict[str, Any]], key: str) -> Dict[Any, List[Dict[str, Any]]]:
    groups: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for item in items:
        groups[item.get(key)].append(item)
    return dict(groups)


def hand_usage(
    events: Sequence[Dict[str, Any]],
    turns: Dict[str, Dict[str, Any]],
) -> Dict[Any, List[Dict[str, Any]]]:
    groups: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    current_hand: Any = None
    final_phase = False
    for event in events:
        kind = event.get("type")
        if kind == "final_talk_start":
            final_phase = True
            continue
        if kind in ("turn_start", "hand_result"):
            current_hand = event.get("hand_id")
            final_phase = False
        if kind == "hand_result":
            current_hand = event.get("hand_id")
        if kind == "message_end":
            if final_phase:
                groups["final"].append(event)
            elif current_hand is not None:
                groups[current_hand].append(event)
        elif kind == "context_summary_end":
            if current_hand is not None:
                groups[current_hand].append(event)
    return dict(groups)


def hand_durations(events: Sequence[Dict[str, Any]]) -> Dict[int, float]:
    results = [event for event in events if event.get("type") == "hand_result"]
    durations: Dict[int, float] = {}
    first_turn = next(
        (float(event.get("_ts") or 0.0) for event in events if event.get("type") == "turn_start"),
        None,
    )
    previous = first_turn
    for result in results:
        hand_id = int(result.get("hand_id", 0) or 0)
        timestamp = float(result.get("_ts") or 0.0)
        if previous is not None and timestamp > previous:
            durations[hand_id] = timestamp - previous
        previous = timestamp
    return durations


def style_metrics(
    hands: Sequence[Dict[str, Any]],
    turns: Dict[str, Dict[str, Any]],
    seat: int,
    seat_name: str,
) -> Dict[str, Any]:
    stats = HistoryStats()
    for record in hands:
        stats.add(record)
    metrics = stats.metrics(seat)
    button = stats.metrics(seat, "button")
    bb = stats.metrics(seat, "bb")

    action_counts: Dict[str, Counter] = defaultdict(Counter)
    total_actions: Counter = Counter()
    for record in hands:
        for action in record.get("actions") or []:
            if int(action.get("seat", -1)) != seat:
                continue
            street = str(action.get("street") or "unknown")
            kind = str(action.get("type") or "unknown")
            action_counts[street][kind] += 1
            total_actions[kind] += 1

    preflop_sizes: List[float] = []
    postflop_ratios: List[float] = []
    big_blind = int((hands[0].get("big_blind", 10) if hands else 10) or 10)
    for record in hands:
        for action in record.get("actions") or []:
            if int(action.get("seat", -1)) != seat:
                continue
            kind = str(action.get("type") or "")
            street = str(action.get("street") or "")
            amount = action.get("amount")
            added = action.get("added")
            pot_after = action.get("pot_after")
            if street == "preflop" and kind in ("raise", "all_in") and amount:
                hand_big_blind = int(record.get("big_blind", big_blind) or big_blind)
                preflop_sizes.append(float(amount) / max(1, hand_big_blind))
            if street in ("flop", "turn", "river") and kind in ("bet", "raise", "all_in"):
                if added is not None and pot_after is not None:
                    pot_before = float(pot_after) - float(added)
                    if pot_before > 0:
                        postflop_ratios.append(float(added) / pot_before)

    hole_map: Dict[Tuple[int, int], List[str]] = {}
    for turn in turns.values():
        state = turn.get("state") or {}
        cards = state.get("hole_cards")
        if isinstance(cards, list) and len(cards) == 2:
            hole_map[(int(turn.get("hand_id", 0) or 0), int(turn.get("seat", -1)))] = [
                str(card) for card in cards
            ]

    aggressive_by_strength: Counter = Counter()
    passive_by_strength: Counter = Counter()
    strength_details: Dict[str, Dict[str, int]] = {}
    for record in hands:
        hand_id = int(record.get("hand_id", 0) or 0)
        hole = hole_map.get((hand_id, seat))
        if not hole:
            continue
        board = list(record.get("board") or [])
        for action in record.get("actions") or []:
            if int(action.get("seat", -1)) != seat:
                continue
            street = str(action.get("street") or "")
            if street not in ("flop", "turn", "river"):
                continue
            prefix = {"flop": 3, "turn": 4, "river": 5}[street]
            if len(board) < prefix:
                continue
            try:
                assessment = assess_hand(Card.parse_many(hole + board[:prefix]))
            except Exception:  # noqa: BLE001
                continue
            if assessment is None:
                continue
            rank = STRENGTH_RANK.get(assessment.category, 0)
            bucket = "air" if rank == 0 else "weak" if rank == 1 else "medium" if rank <= 3 else "strong"
            kind = str(action.get("type") or "")
            if kind in ("bet", "raise", "all_in"):
                aggressive_by_strength[bucket] += 1
            elif kind in ("call", "check", "fold"):
                passive_by_strength[bucket] += 1

    for bucket in ("air", "weak", "medium", "strong"):
        aggressive = aggressive_by_strength[bucket]
        passive = passive_by_strength[bucket]
        strength_details[bucket] = {
            "aggressive": aggressive,
            "passive": passive,
            "aggression": safe_div(aggressive, aggressive + passive),
        }

    return {
        "name": seat_name,
        "metrics": metrics,
        "button": button,
        "bb": bb,
        "actions": {street: dict(counter) for street, counter in action_counts.items()},
        "action_total": dict(total_actions),
        "sizing": {
            "preflop_raise_to_bb": {
                "count": len(preflop_sizes),
                "median": statistics.median(preflop_sizes) if preflop_sizes else None,
                "p25": percentile(preflop_sizes, 0.25),
                "p75": percentile(preflop_sizes, 0.75),
            },
            "postflop_added_over_pot": {
                "count": len(postflop_ratios),
                "median": statistics.median(postflop_ratios) if postflop_ratios else None,
                "p25": percentile(postflop_ratios, 0.25),
                "p75": percentile(postflop_ratios, 0.75),
                "buckets": {
                    "<40%": sum(1 for value in postflop_ratios if value < 0.4),
                    "40-60%": sum(1 for value in postflop_ratios if 0.4 <= value < 0.6),
                    "60-90%": sum(1 for value in postflop_ratios if 0.6 <= value < 0.9),
                    "90-120%": sum(1 for value in postflop_ratios if 0.9 <= value < 1.2),
                    ">120%": sum(1 for value in postflop_ratios if value >= 1.2),
                },
            },
        },
        "strength": strength_details,
    }


def tool_stats(turns: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    calls_by_name: Counter = Counter()
    calls_by_seat: Dict[int, Counter] = defaultdict(Counter)
    calls_by_street: Dict[str, Counter] = defaultdict(Counter)
    calls_by_pressure: Counter = Counter()
    calls_by_pot: Counter = Counter()
    errors_by_name: Counter = Counter()
    durations: Dict[str, List[float]] = defaultdict(list)
    optional_per_turn: List[int] = []
    sequences: Counter = Counter()
    research_args: Counter = Counter()
    turns_with_optional = 0
    total_turns = 0
    forced_calls = 0
    inline_table_talk = 0

    for turn in turns.values():
        total_turns += 1
        names = [str(call.get("name") or "") for call in turn.get("tools") or []]
        optional = [name for name in names if name in OPTIONAL_TOOLS]
        if optional:
            turns_with_optional += 1
        optional_per_turn.append(len(optional))
        sequences[tuple(name for name in names if name != "apply_action")] += 1
        for call in turn.get("tools") or []:
            name = str(call.get("name") or "")
            if not name:
                continue
            calls_by_name[name] += 1
            if name in FORCED_TOOLS:
                forced_calls += 1
            if name == "apply_action":
                parsed = call.get("parsed") if isinstance(call.get("parsed"), dict) else {}
                if isinstance(parsed, dict) and str(parsed.get("table_talk") or "").strip():
                    inline_table_talk += 1
            calls_by_seat[int(turn.get("seat", -1))][name] += 1
            calls_by_street[str(turn.get("street") or "unknown")][name] += 1
            if name in OPTIONAL_TOOLS:
                to_call = int(turn.get("to_call", 0) or 0)
                calls_by_pressure["facing_bet" if to_call > 0 else "unopened"] += 1
                pot = int(turn.get("pot", 0) or 0)
                bucket = "<=60" if pot <= 60 else "61-200" if pot <= 200 else ">200"
                calls_by_pot[bucket] += 1
            if name == "research_history":
                parsed = call.get("parsed") if isinstance(call.get("parsed"), dict) else {}
                research_args.update(
                    {
                        f"scope={parsed.get('scope', 'default')}": 1,
                        f"focus={parsed.get('focus', 'default')}": 1,
                    }
                )
        for result in turn.get("tool_results") or []:
            name = str(result.get("name") or "")
            if result.get("ok") is False:
                errors_by_name[name] += 1
            if result.get("duration_ms") is not None:
                durations[name].append(float(result.get("duration_ms") or 0.0))

    top_sequences = [
        {"sequence": list(sequence), "count": count}
        for sequence, count in sequences.most_common(20)
        if sequence and any(name in OPTIONAL_TOOLS for name in sequence)
    ][:8]
    return {
        "total_turns": total_turns,
        "calls_by_name": dict(calls_by_name),
        "inline_table_talk": inline_table_talk,
        "calls_by_seat": {str(seat): dict(counter) for seat, counter in calls_by_seat.items()},
        "calls_by_street": {street: dict(counter) for street, counter in calls_by_street.items()},
        "calls_by_pressure": dict(calls_by_pressure),
        "calls_by_pot": dict(calls_by_pot),
        "errors_by_name": dict(errors_by_name),
        "duration_ms": {
            name: {
                "count": len(values),
                "median": statistics.median(values),
                "total": sum(values),
            }
            for name, values in durations.items()
        },
        "optional_calls_per_turn": {
            "mean": safe_div(sum(optional_per_turn), len(optional_per_turn)),
            "median": statistics.median(optional_per_turn) if optional_per_turn else None,
            "max": max(optional_per_turn) if optional_per_turn else None,
        },
        "turns_with_optional": turns_with_optional,
        "forced_calls": forced_calls,
        "top_sequences": top_sequences,
        "research_args": dict(research_args),
    }


def cache_stats(
    requests: Sequence[Dict[str, Any]],
    events: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    by_turn = group_by(requests, "turn_id")
    by_hand = group_by(requests, "hand_id")
    first_turn = [items[0] for items in by_turn.values() if items]
    steady = [item for items in by_turn.values() for item in items[1:]]
    first_hand = [items[0] for items in by_hand.values() if items]
    phases: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for request in requests:
        phases[int(request.get("phase", 0) or 0)].append(request)

    compactions = []
    for event in events:
        if event.get("type") == "context_compacted":
            compactions.append(
                {
                    "ts": event.get("_ts"),
                    "before_tokens": event.get("before_tokens"),
                    "after_tokens": event.get("after_tokens"),
                    "through_hand": event.get("through_hand"),
                    "fallback": bool(event.get("fallback")),
                    "summary_tokens": event.get("summary_tokens"),
                }
            )

    return {
        "requests": len(requests),
        "first_turn_hit": rate(first_turn, "input_tokens"),
        "steady_hit": rate(steady, "input_tokens"),
        "first_hand_hit": rate(first_hand, "input_tokens"),
        "overall_hit": rate(requests, "input_tokens"),
        "by_phase": {
            str(phase): {
                "requests": len(items),
                "hit": rate(items, "input_tokens"),
                "input_tokens": sum(int(item.get("input_tokens", 0) or 0) for item in items),
            }
            for phase, items in sorted(phases.items())
        },
        "compactions": compactions,
    }


def timing_stats(
    events: Sequence[Dict[str, Any]],
    requests: Sequence[Dict[str, Any]],
    turns: Dict[str, Dict[str, Any]],
    match: Dict[str, Any],
    durations: Dict[int, float],
) -> Dict[str, Any]:
    turn_latencies = [
        float((turn.get("end") or {}).get("latency_ms", 0.0) or 0.0)
        for turn in turns.values()
        if turn.get("end")
    ]
    request_latencies = [item["latency_ms"] for item in requests if item["latency_ms"] > 0]
    tool_durations = []
    summary_latencies = []
    final_talk_latencies = []
    for event in events:
        if event.get("type") == "tool_result":
            tool_durations.append(float(event.get("duration_ms", 0.0) or 0.0))
        elif event.get("type") == "context_summary_end":
            summary_latencies.append(float(event.get("latency_ms", 0.0) or 0.0))
        elif event.get("type") == "final_talk_response":
            final_talk_latencies.append(float(event.get("latency_ms", 0.0) or 0.0))

    hand_values = list(durations.values())
    active_per_hand: Dict[int, float] = defaultdict(float)
    for turn in turns.values():
        end = turn.get("end") or {}
        hand = turn.get("hand_id")
        if hand is not None:
            active_per_hand[int(hand)] += float(end.get("latency_ms", 0.0) or 0.0) / 1000.0

    buckets = [
        (0, 25_000, "0-25k"),
        (25_000, 50_000, "25-50k"),
        (50_000, 100_000, "50-100k"),
        (100_000, 150_000, "100-150k"),
        (150_000, 200_000, "150-200k"),
        (200_000, 10**12, "200k+"),
    ]
    latency_by_context = []
    for low, high, label in buckets:
        items = [item for item in requests if low <= item["input_tokens"] < high]
        if not items:
            continue
        latencies = [item["latency_ms"] for item in items if item["latency_ms"] > 0]
        latency_by_context.append(
            {
                "bucket": label,
                "requests": len(items),
                "median_ms": statistics.median(latencies) if latencies else None,
                "p95_ms": percentile(latencies, 0.95),
                "hit": rate(items, "input_tokens"),
            }
        )

    return {
        "wall_seconds": float(match.get("wall_seconds", 0.0) or 0.0),
        "hand_interval_seconds": {
            "count": len(hand_values),
            "median": statistics.median(hand_values) if hand_values else None,
            "p95": percentile(hand_values, 0.95),
            "total": sum(hand_values),
        },
        "hand_active_seconds": {
            "count": len(active_per_hand),
            "median": statistics.median(list(active_per_hand.values())) if active_per_hand else None,
            "p95": percentile(list(active_per_hand.values()), 0.95),
            "total": sum(active_per_hand.values()),
        },
        "turn_latency_ms": {
            "count": len(turn_latencies),
            "median": statistics.median(turn_latencies) if turn_latencies else None,
            "p95": percentile(turn_latencies, 0.95),
            "total": sum(turn_latencies),
        },
        "request_latency_ms": {
            "count": len(request_latencies),
            "median": statistics.median(request_latencies) if request_latencies else None,
            "p95": percentile(request_latencies, 0.95),
            "total": sum(request_latencies),
        },
        "tool_duration_ms": {
            "count": len(tool_durations),
            "median": statistics.median(tool_durations) if tool_durations else None,
            "total": sum(tool_durations),
        },
        "summary_latency_ms": {
            "count": len(summary_latencies),
            "median": statistics.median(summary_latencies) if summary_latencies else None,
            "total": sum(summary_latencies),
        },
        "final_talk_latency_ms": {
            "count": len(final_talk_latencies),
            "median": statistics.median(final_talk_latencies) if final_talk_latencies else None,
            "total": sum(final_talk_latencies),
        },
        "latency_by_context": latency_by_context,
    }


def data_quality(events: Sequence[Dict[str, Any]], turns: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    message_errors = sum(1 for event in events if event.get("type") == "message_end" and event.get("error"))
    estimated = sum(1 for event in events if event.get("type") == "message_end" and event.get("estimated_usage"))
    auto_turns = sum(1 for turn in turns.values() if (turn.get("end") or {}).get("auto"))
    tool_errors = sum(1 for event in events if event.get("type") == "tool_result" and event.get("ok") is False)
    warnings = sum(
        1
        for event in events
        if event.get("type") == "note" and str(event.get("level") or "") == "warn"
    )
    summary_fallbacks = sum(
        1
        for event in events
        if event.get("type") == "context_compacted" and event.get("fallback")
    )
    summary_failures = sum(
        1 for event in events if event.get("type") == "context_summary_failed"
    )
    final_failures = sum(
        1 for event in events if event.get("type") == "final_talk_end" and not event.get("ok")
    )
    return {
        "message_errors": message_errors,
        "estimated_usage_requests": estimated,
        "auto_turns": auto_turns,
        "tool_errors": tool_errors,
        "warning_notes": warnings,
        "summary_fallbacks": summary_fallbacks,
        "summary_failures": summary_failures,
        "final_talk_failures": final_failures,
    }


def table_talk_stats(match: Dict[str, Any], events: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    items = load_json(Path(match.get("_run_dir", "")) / "table_talk.json") if match.get("_run_dir") else []
    if not isinstance(items, list):
        items = []
    if not items:
        items = [event for event in events if event.get("type") == "table_talk"]
    by_seat = Counter()
    by_kind = Counter()
    lengths = []
    for item in items:
        by_seat[int(item.get("seat", -1))] += 1
        by_kind[str(item.get("kind") or "turn")] += 1
        lengths.append(len(str(item.get("text") or "")))
    finals = [item for item in items if str(item.get("kind") or "") == "final"]
    turns = [item for item in items if str(item.get("kind") or "turn") != "final"]
    return {
        "count": len(items),
        "by_seat": {str(seat): count for seat, count in by_seat.items()},
        "by_kind": dict(by_kind),
        "avg_chars": statistics.mean(lengths) if lengths else None,
        "max_chars": max(lengths) if lengths else None,
        "finals": finals,
        "samples": turns[:10],
    }


def write_chart(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def plot_stacks(
    states: Sequence[Dict[str, Any]],
    hands: Sequence[Dict[str, Any]],
    match: Dict[str, Any],
    path: Path,
) -> None:
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(10.8, 5.2))
    fig.patch.set_facecolor("#F7F5EF")
    ax.set_facecolor("#F7F5EF")

    if hands:
        starting = int((match.get("rules") or {}).get("starting_chips", 1000) or 1000)
        for seat in (0, 1):
            xs = [0]
            ys = [starting]
            for record in hands:
                players = {
                    int(player.get("seat", -1)): player
                    for player in record.get("players") or []
                }
                if seat in players:
                    xs.append(int(record.get("hand_id", 0) or 0))
                    ys.append(float(players[seat].get("chips_after", 0) or 0))
            if len(xs) > 1:
                ax.step(xs, ys, where="post", linewidth=2.0, color=SEAT_COLORS[seat], label=f"Seat {seat}")

        previous_blinds = None
        for record in hands:
            blinds = (
                int(record.get("small_blind", 0) or 0),
                int(record.get("big_blind", 0) or 0),
            )
            if previous_blinds is not None and blinds != previous_blinds:
                hand_id = int(record.get("hand_id", 0) or 0)
                ax.axvline(hand_id - 0.5, color="#6B7280", linewidth=0.9, linestyle="--", alpha=0.7)
                ax.text(
                    hand_id - 0.5,
                    ax.get_ylim()[1] * 0.96,
                    f"{blinds[0]}/{blinds[1]}",
                    rotation=0,
                    fontsize=8,
                    color="#5B5B5B",
                    ha="center",
                    va="top",
                    backgroundcolor="#F7F5EF",
                )
            previous_blinds = blinds
    elif states:
        start = float(states[0].get("ts", 0.0) or 0.0)
        for seat in (0, 1):
            xs = []
            ys = []
            for state in states:
                chips = state.get("chips") or []
                if seat < len(chips) and chips[seat] is not None:
                    xs.append((float(state.get("ts", start)) - start) / 60.0)
                    ys.append(float(chips[seat]))
            if xs:
                ax.plot(xs, ys, linewidth=2.0, color=SEAT_COLORS[seat], label=f"Seat {seat}")

    ax.axhline(0, color="#263238", linewidth=1.0)
    ax.set_xlabel("Hand" if hands else "Elapsed minutes", fontsize=10)
    ax.set_ylabel("Chips", fontsize=10)
    ax.set_title("Stack trajectory", fontsize=13, fontweight="bold", loc="left")
    ax.grid(axis="y", color="#D8D3C7", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_cost(
    hand_rows: Sequence[Dict[str, Any]],
    requests: Sequence[Dict[str, Any]],
    path: Path,
) -> None:
    if not hand_rows:
        return
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12.0, 4.8))
    fig.patch.set_facecolor("#F7F5EF")
    for axis in (ax_left, ax_right):
        axis.set_facecolor("#F7F5EF")
        axis.spines[["top", "right"]].set_visible(False)

    hands = [row["hand_id"] for row in hand_rows]
    cumulative = []
    total = 0.0
    for row in hand_rows:
        total += row["cost"]
        cumulative.append(total)
    ax_left.plot(hands, cumulative, color="#1F6F5C", linewidth=2.2)
    ax_left.set_xlabel("Hand", fontsize=10)
    ax_left.set_ylabel("Cumulative cost (CNY)", fontsize=10)
    ax_left.set_title("Cost accumulation", fontsize=12, fontweight="bold", loc="left")
    ax_left.grid(color="#D8D3C7", linewidth=0.7, alpha=0.8)
    ax_left.set_axisbelow(True)

    hit = [row["cache_hit"] for row in hand_rows]
    ax_right.bar(hands, hit, color="#2F9E8F", alpha=0.75, width=0.8)
    ax_right.set_ylim(0, 1.05)
    ax_right.set_xlabel("Hand", fontsize=10)
    ax_right.set_ylabel("Cache hit rate", fontsize=10)
    ax_right.set_title("Cache hit by hand", fontsize=12, fontweight="bold", loc="left")
    ax_right.grid(axis="y", color="#D8D3C7", linewidth=0.7, alpha=0.8)
    ax_right.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_tools(tool_data: Dict[str, Any], path: Path) -> None:
    calls = {name: int(tool_data["calls_by_name"].get(name, 0)) for name in OPTIONAL_TOOLS}
    if not any(calls.values()):
        return
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    fig.patch.set_facecolor("#F7F5EF")
    ax.set_facecolor("#F7F5EF")
    names = list(calls)
    values = [calls[name] for name in names]
    colors = ["#2F9E8F" if name != "get_random_number" else "#94A3B8" for name in names]
    ax.barh(names[::-1], values[::-1], color=colors[::-1], height=0.6)
    for index, value in enumerate(values[::-1]):
        ax.text(value + 0.1, index, str(value), va="center", fontsize=9, color="#263238")
    ax.set_xlabel("Calls", fontsize=10)
    ax.set_title("Optional tool calls", fontsize=13, fontweight="bold", loc="left")
    ax.grid(axis="x", color="#D8D3C7", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_latency(requests: Sequence[Dict[str, Any]], path: Path) -> None:
    items = [item for item in requests if item["input_tokens"] > 0 and item["latency_ms"] > 0]
    if not items:
        return
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    fig.patch.set_facecolor("#F7F5EF")
    ax.set_facecolor("#F7F5EF")
    xs = [item["input_tokens"] / 1000.0 for item in items]
    ys = [item["latency_ms"] / 1000.0 for item in items]
    colors = [item["cache_hit"] if item["cache_hit"] is not None else 0.0 for item in items]
    scatter = ax.scatter(xs, ys, c=colors, cmap="viridis", vmin=0, vmax=1, s=26, alpha=0.8)
    ax.set_xlabel("Prompt tokens (thousands)", fontsize=10)
    ax.set_ylabel("Request latency (seconds)", fontsize=10)
    ax.set_title("Latency vs prompt size", fontsize=13, fontweight="bold", loc="left")
    ax.grid(color="#D8D3C7", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("Cache hit rate", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_style(style: Dict[str, Any], path: Path) -> None:
    metrics = []
    labels = []
    for seat in (0, 1):
        item = style.get(str(seat)) or {}
        data = item.get("metrics") or {}
        metrics.append(
            [
                (data.get("preflop", {}).get("vpip", {}) or {}).get("value") or 0.0,
                (data.get("preflop", {}).get("pfr", {}) or {}).get("value") or 0.0,
                (data.get("preflop", {}).get("three_bet", {}) or {}).get("value") or 0.0,
                (data.get("postflop", {}).get("c_bet", {}) or {}).get("value") or 0.0,
                (data.get("showdown", {}).get("wtsd", {}) or {}).get("value") or 0.0,
                (data.get("showdown", {}).get("w_sd", {}) or {}).get("value") or 0.0,
            ]
        )
        labels.append(f"Seat {seat}")
    if not any(any(row) for row in metrics):
        return
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
    fig, ax = plt.subplots(figsize=(10.0, 4.6))
    fig.patch.set_facecolor("#F7F5EF")
    ax.set_facecolor("#F7F5EF")
    names = ["VPIP", "PFR", "3-bet", "C-bet", "WTSD", "W$SD"]
    x = list(range(len(names)))
    width = 0.36
    for seat, row in enumerate(metrics):
        ax.bar(
            [value + (seat - 0.5) * width for value in x],
            [value * 100 for value in row],
            width=width,
            color=SEAT_COLORS[seat],
            label=labels[seat],
        )
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("Percent", fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_title("Style profile", fontsize=13, fontweight="bold", loc="left")
    ax.grid(axis="y", color="#D8D3C7", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)


def metric_value(metrics: Dict[str, Any], group: str, key: str) -> Optional[float]:
    item = (metrics.get(group) or {}).get(key) or {}
    return item.get("value")


def metric_n(metrics: Dict[str, Any], group: str, key: str) -> int:
    item = (metrics.get(group) or {}).get(key) or {}
    return int(item.get("n", 0) or 0)


def top_hands(hands: Sequence[Dict[str, Any]], turns: Dict[str, Dict[str, Any]], limit: int = 5) -> List[Dict[str, Any]]:
    hole_map: Dict[Tuple[int, int], List[str]] = {}
    for turn in turns.values():
        state = turn.get("state") or {}
        cards = state.get("hole_cards")
        if isinstance(cards, list) and len(cards) == 2:
            hole_map[(int(turn.get("hand_id", 0) or 0), int(turn.get("seat", -1)))] = [
                str(card) for card in cards
            ]
    ordered = sorted(hands, key=lambda record: int(record.get("pot", 0) or 0), reverse=True)
    result = []
    for record in ordered[:limit]:
        hand_id = int(record.get("hand_id", 0) or 0)
        action_parts = []
        for action in record.get("actions") or []:
            seat = int(action.get("seat", -1))
            street = str(action.get("street") or "")
            kind = str(action.get("type") or "")
            amount = action.get("amount")
            label = ACTION_LABELS.get(kind, kind)
            if amount:
                action_parts.append(f"{STREET_LABELS.get(street, street)} S{seat} {label}{amount}")
            else:
                action_parts.append(f"{STREET_LABELS.get(street, street)} S{seat} {label}")
        result.append(
            {
                "hand_id": hand_id,
                "pot": int(record.get("pot", 0) or 0),
                "board": " ".join(str(card) for card in record.get("board") or []),
                "winner_seats": list(record.get("winner_seats") or []),
                "reason": record.get("reason"),
                "holes": {
                    str(seat): " ".join(cards)
                    for (hand, seat), cards in hole_map.items()
                    if hand == hand_id
                },
                "actions": "; ".join(action_parts),
            }
        )
    return result


def analyze(run_dir: Path) -> Dict[str, Any]:
    match = load_json(run_dir / "match.json") or {}
    hands = load_jsonl(run_dir / "hands.jsonl")
    events = load_jsonl(run_dir / "events.jsonl")
    states = load_jsonl(run_dir / "states.jsonl")
    agents = load_json(run_dir / "agents.json") or {}
    prices = match.get("prices") or {
        "input_per_m": 0.0198,
        "cache_input_per_m": 0.0198,
        "output_per_m": 0.099,
    }
    turns, _ = build_turns(events)
    requests = build_requests(events, turns)

    message_totals = usage_totals(
        [event for event in events if event.get("type") == "message_end"]
    )
    summary_totals = usage_totals(
        [event for event in events if event.get("type") == "context_summary_end"]
    )
    final_totals = usage_totals(
        [event for event in events if event.get("type") == "final_talk_end"]
    )
    total_usage = {
        key: message_totals.get(key, 0) + summary_totals.get(key, 0) + final_totals.get(key, 0)
        for key in message_totals
    }
    total_cost = cost_of(total_usage, prices)
    message_cost = cost_of(message_totals, prices)
    summary_cost = cost_of(summary_totals, prices)
    final_cost = cost_of(final_totals, prices)

    per_hand_usage = hand_usage(events, turns)
    durations = hand_durations(events)
    hand_rows = []
    for record in hands:
        hand_id = int(record.get("hand_id", 0) or 0)
        items = per_hand_usage.get(hand_id, [])
        usage = usage_totals(items)
        cost = cost_of(usage, prices)
        hand_rows.append(
            {
                "hand_id": hand_id,
                "cost": cost,
                "input_tokens": usage.get("input_tokens", 0),
                "cached_input_tokens": usage.get("cached_input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "cache_hit": safe_div(
                    usage.get("cached_input_tokens", 0), usage.get("input_tokens", 0)
                )
                or 0.0,
                "duration": durations.get(hand_id),
            }
        )

    seat_names = {
        str(seat): str((agents.get(str(seat)) or {}).get("name") or f"Seat {seat}")
        for seat in (0, 1)
    }
    style = {
        str(seat): style_metrics(hands, turns, seat, seat_names[str(seat)])
        for seat in (0, 1)
    }
    tools = tool_stats(turns)
    cache = cache_stats(requests, events)
    timing = timing_stats(events, requests, turns, match, durations)
    quality = data_quality(events, turns)
    talk = table_talk_stats({**match, "_run_dir": str(run_dir)}, events)
    notable = top_hands(hands, turns)

    final_chips = list(match.get("final_chips") or [])
    winner = None
    if len(final_chips) == 2 and all(chip is not None for chip in final_chips):
        if int(final_chips[0]) > int(final_chips[1]):
            winner = 0
        elif int(final_chips[1]) > int(final_chips[0]):
            winner = 1

    summary = {
        "run_dir": str(run_dir),
        "match": match,
        "hands": len(hands),
        "complete": bool(match.get("complete")),
        "winner_seat": winner,
        "final_chips": final_chips,
        "duration_seconds": float(match.get("wall_seconds", 0.0) or 0.0),
        "cost_cny": total_cost,
        "cost_breakdown": {
            "match": message_cost,
            "checkpoint_summary": summary_cost,
            "final_talk": final_cost,
        },
        "usage": total_usage,
        "message_usage": message_totals,
        "summary_usage": summary_totals,
        "final_usage": final_totals,
        "cache": cache,
        "timing": timing,
        "style": style,
        "tools": tools,
        "quality": quality,
        "table_talk": talk,
        "top_hands": notable,
        "hand_rows": hand_rows,
        "agents": agents,
    }
    return summary


def build_report(summary: Dict[str, Any], run_dir: Path) -> str:
    match = summary["match"]
    seat_names = {
        str(seat): str((summary["agents"].get(str(seat)) or {}).get("name") or f"Seat {seat}")
        for seat in (0, 1)
    }
    winner = summary.get("winner_seat")
    winner_text = seat_names.get(str(winner), f"Seat {winner}") if winner is not None else "未分胜负"
    final_chips = summary.get("final_chips") or []
    duration_min = summary["duration_seconds"] / 60.0
    usage = summary["usage"]
    cache = summary["cache"]
    timing = summary["timing"]
    tools = summary["tools"]
    quality = summary["quality"]

    lines: List[str] = []
    lines.append("# GPT-6 Luna 自对弈完整比赛报告")
    lines.append("")
    lines.append(f"- 运行目录: `{run_dir}`")
    lines.append(f"- 模型: `{match.get('provider')}/{match.get('model')}`，推理 `{match.get('params', {}).get('reasoning_effort', '')}`")
    rules = match.get("rules") or {}
    start_small = rules.get("starting_small_blind", rules.get("small_blind"))
    start_big = rules.get("starting_big_blind", rules.get("big_blind"))
    lines.append(
        f"- 规则: {rules.get('starting_chips')} 筹码，"
        f"{start_small}/{start_big} 起始盲注，"
        f"无重购，{'手数上限 ' + str(rules.get('max_hands')) if rules.get('max_hands') else '打到一方筹码归零'}"
    )
    lines.append(f"- 桌边喊话: {'开启' if match.get('table_talk_enabled') else '关闭'}")
    if int(rules.get("blind_increase_every_hands", 0) or 0) > 0:
        lines.append(
            f"- 盲注递增: 每 {rules.get('blind_increase_every_hands')} 手 "
            f"×{rules.get('blind_increase_factor')}，当前 {rules.get('small_blind')}/{rules.get('big_blind')}，"
            f"级别 {match.get('blind_level')}"
        )
    else:
        lines.append("- 盲注递增: 关闭")
    lines.append(f"- 结束方式: {'完整结束' if summary['complete'] else '未完整结束'}，停止原因 `{match.get('stop_reason')}`")
    lines.append("")
    lines.append("## 总览")
    lines.append("")
    lines.append("| 指标 | 值 |")
    lines.append("|---|---:|")
    lines.append(f"| 手数 | {summary['hands']} |")
    lines.append(f"| 赢家 | {winner_text} |")
    lines.append(f"| 最终筹码 | {final_chips} |")
    lines.append(f"| 总时长 | {duration_min:.1f} 分钟 |")
    lines.append(f"| 每手平均间隔 | {number((timing['hand_interval_seconds']['median'] or 0) / 60, 2)} 分钟 |")
    lines.append(f"| 每个 turn 中位延迟 | {number((timing['turn_latency_ms']['median'] or 0) / 1000, 1)} 秒 |")
    lines.append(f"| 总费用 | ¥{summary['cost_cny']:.4f} |")
    lines.append(f"| 输入 token | {usage['input_tokens']:,} |")
    lines.append(f"| 缓存输入 | {usage['cached_input_tokens']:,} |")
    lines.append(f"| 输出 token | {usage['output_tokens']:,} |")
    lines.append(f"| 缓存命中率 | {pct(cache['overall_hit'])} |")
    lines.append(f"| checkpoint 压缩 | {len(cache['compactions'])} 次 |")
    lines.append("")

    lines.append("## 节奏与耗时")
    lines.append("")
    lines.append("| 指标 | 中位数 | 95 分位 | 总计 |")
    lines.append("|---|---:|---:|---:|")
    lines.append(
        f"| 每手墙钟间隔 | {number((timing['hand_interval_seconds']['median'] or 0), 1)}s | "
        f"{number((timing['hand_interval_seconds']['p95'] or 0), 1)}s | "
        f"{number(timing['hand_interval_seconds']['total'], 1)}s |"
    )
    lines.append(
        f"| 每手行动耗时 | {number((timing['hand_active_seconds']['median'] or 0), 1)}s | "
        f"{number((timing['hand_active_seconds']['p95'] or 0), 1)}s | "
        f"{number(timing['hand_active_seconds']['total'], 1)}s |"
    )
    lines.append(
        f"| 每个 turn | {number((timing['turn_latency_ms']['median'] or 0) / 1000, 1)}s | "
        f"{number((timing['turn_latency_ms']['p95'] or 0) / 1000, 1)}s | "
        f"{number(timing['turn_latency_ms']['total'] / 1000, 1)}s |"
    )
    lines.append(
        f"| 单次模型请求 | {number((timing['request_latency_ms']['median'] or 0) / 1000, 1)}s | "
        f"{number((timing['request_latency_ms']['p95'] or 0) / 1000, 1)}s | "
        f"{number(timing['request_latency_ms']['total'] / 1000, 1)}s |"
    )
    lines.append("")
    lines.append(
        "时间拆分：模型请求约 "
        f"{number(timing['request_latency_ms']['total'] / 60000, 1)} 分钟，"
        f"估算工具约 {number(timing['tool_duration_ms']['total'] / 60000, 2)} 分钟，"
        f"checkpoint 摘要约 {number(timing['summary_latency_ms']['total'] / 60000, 2)} 分钟，"
        f"赛后喊话约 {number(timing['final_talk_latency_ms']['total'] / 60000, 2)} 分钟。"
        f"剩余时间主要是发牌、引擎、自动下一手等待和本地开销。"
    )
    lines.append("")

    lines.append("## 打牌风格")
    lines.append("")
    lines.append("| 指标 | Seat 0 | Seat 1 | 样本说明 |")
    lines.append("|---|---:|---:|---|")
    rows = [
        ("VPIP", "preflop", "vpip"),
        ("PFR", "preflop", "pfr"),
        ("Limp", "preflop", "limp"),
        ("3-bet", "preflop", "three_bet"),
        ("Fold to 3-bet", "preflop", "fold_to_three_bet"),
        ("Steal", "preflop", "steal"),
        ("C-bet", "postflop", "c_bet"),
        ("Fold to c-bet", "postflop", "fold_to_c_bet"),
        ("Check-raise", "postflop", "check_raise"),
        ("Aggression", "postflop", "aggression"),
        ("WTSD", "showdown", "wtsd"),
        ("W$SD", "showdown", "w_sd"),
    ]
    for label, group, key in rows:
        values = []
        samples = []
        for seat in (0, 1):
            metrics = summary["style"][str(seat)]["metrics"]
            values.append(pct(metric_value(metrics, group, key)))
            samples.append(metric_n(metrics, group, key))
        lines.append(f"| {label} | {values[0]} | {values[1]} | n={samples[0]}/{samples[1]} |")
    lines.append("")
    lines.append("位置拆分：")
    lines.append("")
    lines.append("| 指标 | Seat 0 按钮 | Seat 0 大盲 | Seat 1 按钮 | Seat 1 大盲 |")
    lines.append("|---|---:|---:|---:|---:|")
    for label, group, key in (("VPIP", "preflop", "vpip"), ("PFR", "preflop", "pfr"), ("C-bet", "postflop", "c_bet")):
        cells = []
        for seat in (0, 1):
            positions = summary["style"][str(seat)]
            cells.append(pct(metric_value(positions["button"], group, key)))
            cells.append(pct(metric_value(positions["bb"], group, key)))
        lines.append(f"| {label} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} |")
    lines.append("")
    lines.append("下注尺度：")
    lines.append("")
    for seat in (0, 1):
        sizing = summary["style"][str(seat)]["sizing"]
        pre = sizing["preflop_raise_to_bb"]
        post = sizing["postflop_added_over_pot"]
        lines.append(
            f"- Seat {seat}: 翻前加注到 {number(pre['median'], 1)} BB（n={pre['count']}）；"
            f"翻后下注/加注占底池中位 {pct(post['median'])}（n={post['count']}，"
            f"四分位 {pct(post['p25'])} 到 {pct(post['p75'])}）。"
        )
    lines.append("")
    lines.append("用底牌和当街公共牌做的粗略强度分类（只统计翻后动作）：")
    lines.append("")
    lines.append("| 强度 | Seat 0 攻击/被动 | Seat 0 攻击率 | Seat 1 攻击/被动 | Seat 1 攻击率 |")
    lines.append("|---|---:|---:|---:|---:|")
    for bucket in ("air", "weak", "medium", "strong"):
        cells = []
        for seat in (0, 1):
            item = summary["style"][str(seat)]["strength"].get(bucket) or {}
            cells.append(f"{item.get('aggressive', 0)}/{item.get('passive', 0)}")
            cells.append(pct(item.get("aggression")))
        lines.append(f"| {bucket} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} |")
    lines.append("")
    lines.append("`air` 指高牌，`weak` 指一对，`medium` 指两对到三条，`strong` 指顺子及以上。这个分类只看当前街的成牌，不评估听牌和隐含赔率。")
    lines.append("")

    lines.append("## 工具使用")
    lines.append("")
    lines.append("| 工具 | 调用次数 | 说明 |")
    lines.append("|---|---:|---|")
    for name in FORCED_TOOLS + OPTIONAL_TOOLS:
        note = "强制" if name in FORCED_TOOLS else "可选"
        lines.append(f"| `{name}` | {tools['calls_by_name'].get(name, 0)} | {note} |")
    if tools.get("inline_table_talk"):
        lines.append(
            f"| `apply_action.table_talk` | {tools['inline_table_talk']} | 强制（内联） |"
        )
    lines.append("")
    lines.append(
        f"- 总 turn 数 {tools['total_turns']}，其中 {tools['turns_with_optional']} 个 turn 至少调用一次可选工具。"
    )
    lines.append(
        f"- 每个 turn 可选工具调用次数：平均 {number(tools['optional_calls_per_turn']['mean'], 2)}，"
        f"中位 {number(tools['optional_calls_per_turn']['median'], 1)}，"
        f"最多 {tools['optional_calls_per_turn']['max']}。"
    )
    lines.append(
        f"- 可选工具在面对下注时调用 {tools['calls_by_pressure'].get('facing_bet', 0)} 次，"
        f"无人下注时 {tools['calls_by_pressure'].get('unopened', 0)} 次。"
    )
    lines.append(f"- 工具错误 {sum(tools['errors_by_name'].values())} 次。")
    if tools["top_sequences"]:
        lines.append("- 常见工具序列：")
        for item in tools["top_sequences"][:5]:
            lines.append(f"  - {' → '.join(item['sequence'])}（{item['count']} 次）")
    lines.append("")

    lines.append("## 缓存与 token")
    lines.append("")
    lines.append("| 指标 | 值 |")
    lines.append("|---|---:|")
    lines.append(f"| 主对局模型请求数 | {cache['requests']} |")
    lines.append(f"| 总体缓存命中 | {pct(cache['overall_hit'])} |")
    lines.append(f"| 每手第一条请求命中 | {pct(cache['first_hand_hit'])} |")
    lines.append(f"| 每 turn 第一条请求命中 | {pct(cache['first_turn_hit'])} |")
    lines.append(f"| 同 turn 后续请求命中 | {pct(cache['steady_hit'])} |")
    lines.append(f"| 输入 token | {usage['input_tokens']:,} |")
    lines.append(f"| 缓存输入 token | {usage['cached_input_tokens']:,} |")
    lines.append(f"| 缓存 miss token | {usage['cache_miss_input_tokens']:,} |")
    lines.append(f"| 输出 token | {usage['output_tokens']:,} |")
    lines.append(f"| 推理 token | {usage['reasoning_tokens']:,} |")
    lines.append("")
    if cache["compactions"]:
        lines.append("checkpoint 压缩：")
        lines.append("")
        lines.append("| 压缩前 | 压缩后 | 总结到第几手 | 摘要 token | 是否兜底 |")
        lines.append("|---:|---:|---:|---:|---:|")
        for item in cache["compactions"]:
            lines.append(
                f"| {item.get('before_tokens')} | {item.get('after_tokens')} | "
                f"{item.get('through_hand')} | {item.get('summary_tokens')} | "
                f"{'是' if item.get('fallback') else '否'} |"
            )
        lines.append("")
    for phase, item in sorted(cache["by_phase"].items(), key=lambda pair: int(pair[0])):
        label = "压缩前" if int(phase) == 0 else f"第 {phase} 次压缩后"
        lines.append(f"- {label}: {item['requests']} 次请求，命中 {pct(item['hit'])}，输入 {item['input_tokens']:,} token。")
    lines.append("")
    lines.append("按上下文大小分段的延迟：")
    lines.append("")
    lines.append("| 上下文 | 请求数 | 中位延迟 | 95 分位 | 缓存命中 |")
    lines.append("|---|---:|---:|---:|---:|")
    for item in timing["latency_by_context"]:
        lines.append(
            f"| {item['bucket']} | {item['requests']} | "
            f"{number((item['median_ms'] or 0) / 1000, 1)}s | "
            f"{number((item['p95_ms'] or 0) / 1000, 1)}s | {pct(item['hit'])} |"
        )
    lines.append("")
    lines.append(
        "费用拆分：主对局 "
        f"¥{summary['cost_breakdown']['match']:.4f}，checkpoint 摘要 "
        f"¥{summary['cost_breakdown']['checkpoint_summary']:.4f}，赛后喊话 "
        f"¥{summary['cost_breakdown']['final_talk']:.4f}。"
    )
    lines.append("")

    lines.append("## 桌边喊话")
    lines.append("")
    lines.append(f"- 总消息 {summary['table_talk']['count']} 条，按座位 {summary['table_talk']['by_seat']}。")
    if summary["table_talk"]["avg_chars"]:
        lines.append(f"- 平均 {summary['table_talk']['avg_chars']:.1f} 字，最长 {summary['table_talk']['max_chars']} 字。")
    lines.append("")
    if summary["table_talk"]["finals"]:
        lines.append("赛后喊话：")
        lines.append("")
        for item in summary["table_talk"]["finals"]:
            lines.append(f"- Seat {item.get('seat')}: {item.get('text')}")
        lines.append("")
    if summary["table_talk"]["samples"]:
        lines.append("桌边喊话样例：")
        lines.append("")
        for item in summary["table_talk"]["samples"][:6]:
            lines.append(f"- Seat {item.get('seat')}（第 {item.get('hand_id')} 手）: {item.get('text')}")
        lines.append("")

    lines.append("## 关键手牌")
    lines.append("")
    for item in summary["top_hands"]:
        lines.append(
            f"- 第 {item['hand_id']} 手，底池 {item['pot']}，赢家 {item['winner_seats']}，"
            f"原因 {item['reason']}，公共牌 {item['board'] or '无'}，"
            f"底牌 {item['holes']}。动作：{item['actions']}"
        )
    lines.append("")

    lines.append("## 数据质量")
    lines.append("")
    lines.append("| 项目 | 数量 |")
    lines.append("|---|---:|")
    for key, label in (
        ("message_errors", "模型请求错误"),
        ("estimated_usage_requests", "估算 usage 的请求"),
        ("auto_turns", "自动行动 turn"),
        ("tool_errors", "工具错误"),
        ("warning_notes", "警告 note"),
        ("summary_fallbacks", "checkpoint 兜底"),
        ("summary_failures", "checkpoint 摘要失败"),
        ("final_talk_failures", "赛后喊话失败"),
    ):
        lines.append(f"| {label} | {quality.get(key, 0)} |")
    lines.append("")

    lines.append("## 结论")
    lines.append("")
    if summary["complete"]:
        lines.append(
            f"- 这是一场完整比赛：{summary['hands']} 手后，{winner_text} 赢下全部筹码，"
            f"用时 {duration_min:.1f} 分钟，约 {number(duration_min / max(1, summary['hands']), 2)} 分钟/手。"
        )
    else:
        lines.append(
            f"- 比赛没有打到筹码归零，停在 {summary['hands']} 手，原因 `{match.get('stop_reason')}`。"
        )
    optional_total = sum(
        int(tools["calls_by_name"].get(name, 0)) for name in OPTIONAL_TOOLS
    )
    if optional_total:
        top_tool = max(
            OPTIONAL_TOOLS,
            key=lambda name: int(tools["calls_by_name"].get(name, 0)),
        )
        lines.append(
            f"- 最常调用的可选工具是 `{top_tool}`，共 {tools['calls_by_name'].get(top_tool, 0)} 次；"
            f"平均每个 turn {number(tools['optional_calls_per_turn']['mean'], 2)} 次可选工具。"
        )
    else:
        if tools.get("inline_table_talk"):
            lines.append(
                "- 整场没有调用任何可选分析工具；模型全程只用了强制的 "
                "`apply_action`，桌边喊话内联在 `table_talk` 参数里。"
            )
        else:
            lines.append(
                "- 整场没有调用任何可选分析工具；模型全程只用了强制的 "
                "`send_table_message` 和 `apply_action`。"
            )
    lines.append(
        f"- 总体缓存命中 {pct(cache['overall_hit'])}，每手第一条请求 {pct(cache['first_hand_hit'])}；"
        f"共触发 {len(cache['compactions'])} 次 checkpoint 压缩。"
    )
    lines.append(
        f"- 总费用 ¥{summary['cost_cny']:.4f}，其中主对局 ¥{summary['cost_breakdown']['match']:.4f}，"
        f"平均每手 ¥{summary['cost_cny'] / max(1, summary['hands']):.4f}。"
    )
    lines.append(
        "- 对手是同模型的另一个实例，座位差异主要是噪声；单场统计只能作为描述，不能当成稳定风格画像。"
    )
    lines.append("")
    lines.append("## 图表")
    lines.append("")
    for name, title in (
        ("stack_trajectory.png", "筹码曲线"),
        ("cost_and_cache.png", "费用与缓存"),
        ("tool_usage.png", "可选工具调用"),
        ("latency_vs_context.png", "延迟与上下文"),
        ("style_profile.png", "风格对比"),
    ):
        path = run_dir / "charts" / name
        if path.is_file():
            lines.append(f"![{title}](charts/{name})")
            lines.append("")
    lines.append("## 注意")
    lines.append("")
    lines.append(
        "- DXH 配置里缓存输入和普通输入同价，缓存命中主要降低延迟，不直接降低按 token 计的费用。"
    )
    lines.append(
        "- 单场样本很小，VPIP、3-bet、c-bet 等指标的置信区间很宽。表里的 n 是分母，不是手数。"
    )
    lines.append(
        "- `air/weak/medium/strong` 分类只看当前街成牌，不评估听牌、阻断牌和对手范围。"
    )
    lines.append(
        "- 如果 `auto_turns` 或 `message_errors` 不为零，相关牌局的行动可能被自动 call/fold 污染。"
    )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    run_dir = args.run
    if not run_dir.is_absolute():
        run_dir = (PROJECT_ROOT / run_dir).resolve()
    if not (run_dir / "match.json").is_file():
        raise SystemExit(f"match.json not found in {run_dir}")
    summary = analyze(run_dir)
    charts = run_dir / "charts"
    charts.mkdir(parents=True, exist_ok=True)
    plot_stacks(load_jsonl(run_dir / "states.jsonl"), load_jsonl(run_dir / "hands.jsonl"), summary["match"], charts / "stack_trajectory.png")
    plot_tools(summary["tools"], charts / "tool_usage.png")
    plot_style(summary["style"], charts / "style_profile.png")

    events = load_jsonl(run_dir / "events.jsonl")
    turns, _ = build_turns(events)
    requests = build_requests(events, turns)
    plot_cost(summary["hand_rows"], requests, charts / "cost_and_cache.png")
    plot_latency(requests, charts / "latency_vs_context.png")

    summary["charts"] = [
        str(path.relative_to(run_dir))
        for path in sorted(charts.glob("*.png"))
    ]
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report = build_report(summary, run_dir)
    (run_dir / "report.md").write_text(report, encoding="utf-8")
    print(f"report written: {run_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
