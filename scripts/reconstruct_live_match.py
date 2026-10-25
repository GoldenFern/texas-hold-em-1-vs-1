"""Reconstruct report inputs from a live run interrupted before shutdown.

The live driver writes ``events.jsonl`` on every event and saves a state
snapshot on demand.  When the process is terminated between hands, the
``hands.jsonl`` and ``match.json`` files are missing.  This script rebuilds
enough of the public hand history for the report tool.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_PRICES = {
    "input_per_m": 0.0198,
    "cache_input_per_m": 0.0198,
    "output_per_m": 0.099,
}


def load_json(path: Path) -> Any:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def usage_totals(events: List[Dict[str, Any]]) -> Dict[str, int]:
    totals = {
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "cache_miss_input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
    }
    for event in events:
        if event.get("type") not in ("message_end", "context_summary_end", "final_talk_end"):
            continue
        usage = event.get("usage") if isinstance(event.get("usage"), dict) else {}
        for key in totals:
            totals[key] += int(usage.get(key, 0) or 0)
    return totals


def cost(usage: Dict[str, int], prices: Dict[str, float]) -> float:
    input_tokens = int(usage.get("input_tokens", 0) or 0)
    cached = int(usage.get("cached_input_tokens", 0) or 0)
    uncached = max(0, input_tokens - cached)
    output = int(usage.get("output_tokens", 0) or 0)
    return (
        uncached * prices["input_per_m"]
        + cached * prices["cache_input_per_m"]
        + output * prices["output_per_m"]
    ) / 1_000_000.0


def reconstruct(
    events: List[Dict[str, Any]],
    state: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
    match = state.get("match") or {}
    rules = match.get("rules") or {}
    seats = match.get("seats") or []
    agents = {
        str(seat.get("seat")): seat.get("agent") or {}
        for seat in seats
        if isinstance(seat, dict)
    }
    table_talk = [
        event for event in events if event.get("type") == "table_talk"
    ]

    hand_turns: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    hand_results: Dict[int, Dict[str, Any]] = {}
    current_hand: Optional[int] = None
    for event in events:
        kind = event.get("type")
        if kind == "turn_start":
            current_hand = int(event.get("hand_id", 0) or 0)
            hand_turns[current_hand].append(
                {
                    "turn_id": event.get("turn_id"),
                    "seat": int(event.get("seat", -1)),
                    "street": event.get("street"),
                    "position": event.get("position"),
                    "pot": int(event.get("pot", 0) or 0),
                    "_ts": float(event.get("_ts") or 0.0),
                }
            )
        elif kind == "turn_end" and current_hand is not None:
            for turn in reversed(hand_turns[current_hand]):
                if turn.get("turn_id") == event.get("turn_id"):
                    turn["action"] = event.get("action") or {}
                    turn["auto"] = bool(event.get("auto"))
                    break
        elif kind == "hand_result":
            hand_id = int(event.get("hand_id", 0) or 0)
            result = event.get("result")
            if isinstance(result, dict):
                result = dict(result)
                result["_ts"] = float(event.get("_ts") or 0.0)
                hand_results[hand_id] = result

    talk_counts: Dict[Tuple[int, int], int] = Counter()
    for item in table_talk:
        hand_id = int(item.get("hand_id", 0) or 0)
        seat = int(item.get("seat", -1))
        if hand_id and seat >= 0:
            talk_counts[(hand_id, seat)] += 1

    records: List[Dict[str, Any]] = []
    for hand_id in sorted(hand_results):
        result = hand_results[hand_id]
        turns = hand_turns.get(hand_id, [])
        if not turns:
            continue
        button = None
        for turn in turns:
            if turn.get("position") == "button":
                button = int(turn.get("seat", -1))
                break
        if button is None:
            button = 0
        actions = []
        for index, turn in enumerate(turns):
            action = turn.get("action") or {}
            if not action:
                continue
            before = int(turn.get("pot", 0) or 0)
            if index + 1 < len(turns):
                after = int(turns[index + 1].get("pot", before) or before)
            else:
                after = int(result.get("pot", before) or before)
            action_type = str(action.get("type") or "")
            amount = action.get("amount")
            added = max(0, after - before) if action_type != "fold" else 0
            actions.append(
                {
                    "hand_id": hand_id,
                    "street": turn.get("street"),
                    "seat": turn.get("seat"),
                    "type": action_type,
                    "amount": amount,
                    "added": added,
                    "pot_after": after,
                    "chips_after": None,
                    "source": action.get("source"),
                    "auto": bool(action.get("auto")),
                }
            )
        players = []
        for player in result.get("players") or []:
            if not isinstance(player, dict):
                continue
            players.append(
                {
                    "seat": player.get("seat"),
                    "chips_after": player.get("chips_after"),
                    "total_bet": player.get("total_bet"),
                    "net": player.get("net"),
                }
            )
        showdown = []
        for item in result.get("showdown") or []:
            if not isinstance(item, dict):
                continue
            showdown.append(
                {
                    "seat": item.get("seat"),
                    "category": item.get("category"),
                    "best_five": list(item.get("best_five") or []),
                }
            )
        records.append(
            {
                "hand_id": hand_id,
                "button": button,
                "small_blind": int(rules.get("small_blind", 5) or 5),
                "big_blind": int(rules.get("big_blind", 10) or 10),
                "board": list(result.get("board") or []),
                "reason": result.get("reason"),
                "winner_seats": list(result.get("winner_seats") or []),
                "pot": int(result.get("pot", 0) or 0),
                "players": players,
                "actions": actions,
                "showdown": showdown,
                "table_talk_count": {
                    str(seat): talk_counts.get((hand_id, seat), 0)
                    for seat in (0, 1)
                },
            }
        )

    usage = usage_totals(events)
    prices = DEFAULT_PRICES
    started_at = float(match.get("started_at") or 0.0) or (
        float(events[0].get("_ts") or 0.0) if events else time.time()
    )
    finished_at = float(events[-1].get("_ts") or 0.0) if events else time.time()
    players = state.get("players") or []
    final_chips = [player.get("chips") for player in players]
    first_seat = seats[0] if seats else {}
    params = (first_seat.get("agent") or {}).get("params") or {}
    match_payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "match_id": match.get("id"),
        "started_at": started_at,
        "finished_at": finished_at,
        "wall_seconds": max(0.0, finished_at - started_at),
        "stop_reason": "interrupted_rate_limit",
        "complete": False,
        "match_over": False,
        "hands_played": len(records),
        "final_chips": final_chips,
        "blind_level": state.get("blind_level"),
        "match_error": match.get("error") or "",
        "table_talk_enabled": bool(match.get("table_talk")),
        "agent_locale": match.get("agent_locale") or "zh-CN",
        "provider": first_seat.get("provider_id") or "dxh",
        "model": first_seat.get("model_id") or "gpt-6-luna",
        "provider_config": str(PROJECT_ROOT / "config/experiments/dxh_providers.json"),
        "port": 5051,
        "url": "http://127.0.0.1:5051",
        "rules": {
            "starting_chips": int(rules.get("starting_chips", 1000) or 1000),
            "small_blind": int(rules.get("small_blind", 5) or 5),
            "big_blind": int(rules.get("big_blind", 10) or 10),
            "starting_small_blind": int(
                rules.get("starting_small_blind", rules.get("small_blind", 5)) or 5
            ),
            "starting_big_blind": int(
                rules.get("starting_big_blind", rules.get("big_blind", 10)) or 10
            ),
            "blind_level": int(state.get("blind_level", rules.get("blind_level", 0)) or 0),
            "auto_rebuy": bool(rules.get("auto_rebuy")),
            "max_hands": int(rules.get("max_hands", 0) or 0),
            "blind_increase_every_hands": int(
                rules.get("blind_increase_every_hands", 0) or 0
            ),
            "blind_increase_factor": float(
                rules.get("blind_increase_factor", 2.0) or 2.0
            ),
            "seed": rules.get("seed"),
        },
        "prices": prices,
        "params": params,
        "usage": usage,
        "cost_cny": cost(usage, prices),
        "event_count": len(events),
        "state_count": 0,
        "agents": agents,
        "reconstructed": True,
    }
    return records, match_payload, agents, table_talk


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run
    if not run_dir.is_absolute():
        run_dir = (PROJECT_ROOT / run_dir).resolve()
    events = load_jsonl(run_dir / "events.jsonl")
    state = load_json(run_dir / "partial_state.json") or load_json(run_dir / "final_state.json")
    if not events or not isinstance(state, dict):
        raise SystemExit("events.jsonl and partial_state.json are required")
    records, match, agents, table_talk = reconstruct(events, state)
    hands_path = run_dir / "hands.jsonl"
    with hands_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    write_json(run_dir / "match.json", match)
    write_json(run_dir / "agents.json", agents)
    write_json(run_dir / "table_talk.json", table_talk)
    write_json(run_dir / "final_state.json", state)
    print(
        f"reconstructed {len(records)} hands, {match['usage']['input_tokens']:,} input tokens, "
        f"Y{match['cost_cny']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
