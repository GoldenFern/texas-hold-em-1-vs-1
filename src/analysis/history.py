"""Public action-history statistics for the heads-up poker agent.

The module only reads public information: actions, board cards, pots, stacks,
results, and showdown categories.  It never stores or returns either player's
hole cards, so the LLM history tool cannot leak hidden information.

The same statistics power two features:

1. ``research_history`` during play.
2. The compact snapshot injected into a checkpoint summary request.

Checkpoint summaries are expected to write conclusions, not to copy these
numbers.  The next agent can call ``research_history`` again for current data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional, Sequence

_AGGRESSIVE = {"bet", "raise", "all_in"}
_VOLUNTARY = {"call", "bet", "raise", "all_in"}
_SCOPES = ("current_match", "completed_matches", "both")
_FOCUS_GROUPS = ("all", "preflop", "postflop", "showdown", "result", "table_talk")
_METRIC_GROUPS = ("preflop", "postflop", "showdown", "result", "table_talk")

_COUNTER_KEYS = (
    "hands",
    "saw_flop",
    "vpip",
    "pfr",
    "limps",
    "three_bet_opps",
    "three_bets",
    "fold_to_three_bet_opps",
    "fold_to_three_bets",
    "steal_opps",
    "steals",
    "fold_to_steal_opps",
    "fold_to_steals",
    "c_bet_opps",
    "c_bets",
    "fold_to_c_bet_opps",
    "fold_to_c_bets",
    "check_raises",
    "postflop_bets",
    "postflop_raises",
    "postflop_calls",
    "postflop_folds",
    "postflop_checks",
    "showdowns",
    "showdown_wins",
    "all_ins",
    "table_talk_messages",
    "net",
    "bb_net",
    "pot_total",
)


def _empty_counters() -> dict:
    """Return one zeroed counter block."""
    counters = {key: 0 for key in _COUNTER_KEYS}
    counters["categories"] = {}
    return counters


def _seat_index(seat: int) -> int:
    return 0 if int(seat) == 0 else 1


def _opponent(seat: int) -> int:
    return 1 - _seat_index(seat)


def _position(record: Mapping, seat: int) -> str:
    return "button" if record.get("button") == _seat_index(seat) else "bb"


def _is_aggressive(action: Mapping) -> bool:
    return str(action.get("type") or "") in _AGGRESSIVE


def _is_voluntary(action: Mapping) -> bool:
    return str(action.get("type") or "") in _VOLUNTARY


def _actions_for(record: Mapping, seat: int, street: Optional[str] = None) -> list[dict]:
    actions = record.get("actions") or []
    if not isinstance(actions, list):
        return []
    return [
        action
        for action in actions
        if isinstance(action, dict)
        and action.get("seat") == _seat_index(seat)
        and (street is None or action.get("street") == street)
    ]


def _all_actions(record: Mapping, street: Optional[str] = None) -> list[dict]:
    actions = record.get("actions") or []
    if not isinstance(actions, list):
        return []
    return [
        action
        for action in actions
        if isinstance(action, dict)
        and (street is None or action.get("street") == street)
    ]


def _result_player(record: Mapping, seat: int) -> dict:
    players = record.get("players") or []
    if not isinstance(players, list):
        return {}
    for player in players:
        if isinstance(player, dict) and player.get("seat") == _seat_index(seat):
            return player
    return {}


def _saw_flop(record: Mapping, seat: int) -> bool:
    board = record.get("board") or []
    if not isinstance(board, list) or len(board) < 3:
        return False
    return not any(
        str(action.get("type") or "") == "fold"
        for action in _actions_for(record, seat, "preflop")
    )


def _preflop_aggressor(record: Mapping) -> Optional[int]:
    aggressor: Optional[int] = None
    for action in _all_actions(record, "preflop"):
        if _is_aggressive(action):
            aggressor = action.get("seat")
    return aggressor


def _increment(counters: dict, key: str, amount: int = 1) -> None:
    counters[key] = int(counters.get(key, 0)) + int(amount)


def _compute_one(record: Mapping, seat: int) -> dict:
    """Update counters for one public hand record."""
    counters = _empty_counters()
    seat = _seat_index(seat)
    _increment(counters, "hands")

    player = _result_player(record, seat)
    net = int(player.get("net", 0) or 0)
    big_blind = max(1, int(record.get("big_blind", 1) or 1))
    _increment(counters, "net", net)
    _increment(counters, "pot_total", int(record.get("pot", 0) or 0))
    counters["bb_net"] = float(counters["bb_net"]) + net / big_blind

    preflop = _all_actions(record, "preflop")
    seat_preflop = _actions_for(record, seat, "preflop")
    if any(_is_voluntary(action) for action in seat_preflop):
        _increment(counters, "vpip")
    if any(_is_aggressive(action) for action in seat_preflop):
        _increment(counters, "pfr")

    position = _position(record, seat)
    if position == "button":
        _increment(counters, "steal_opps")
        if any(_is_aggressive(action) for action in seat_preflop):
            _increment(counters, "steals")
        if (
            not any(_is_aggressive(action) for action in seat_preflop)
            and any(str(action.get("type") or "") == "call" for action in seat_preflop)
        ):
            _increment(counters, "limps")

    if position == "bb":
        button_raised = any(
            action.get("seat") == record.get("button") and _is_aggressive(action)
            for action in preflop
        )
        if button_raised:
            _increment(counters, "fold_to_steal_opps")
            if any(str(action.get("type") or "") == "fold" for action in seat_preflop):
                _increment(counters, "fold_to_steals")

    raises_so_far = 0
    faced_raise = False
    faced_three_bet = False
    three_bet_made = False
    folded_to_three_bet = False
    for action in preflop:
        is_seat = action.get("seat") == seat
        action_type = str(action.get("type") or "")
        if is_seat:
            if raises_so_far >= 1:
                faced_raise = True
            if raises_so_far >= 2:
                faced_three_bet = True
                if action_type == "fold":
                    folded_to_three_bet = True
            if raises_so_far >= 1 and _is_aggressive(action):
                three_bet_made = True
        if _is_aggressive(action):
            raises_so_far += 1

    if faced_raise:
        _increment(counters, "three_bet_opps")
    if three_bet_made:
        _increment(counters, "three_bets")
    if faced_three_bet:
        _increment(counters, "fold_to_three_bet_opps")
    if folded_to_three_bet:
        _increment(counters, "fold_to_three_bets")

    saw_flop = _saw_flop(record, seat)
    if saw_flop:
        _increment(counters, "saw_flop")

    aggressor = _preflop_aggressor(record)
    flop = _all_actions(record, "flop")
    if saw_flop and aggressor == seat:
        seat_flop = [action for action in flop if action.get("seat") == seat]
        opponent_bet_first = False
        for action in flop:
            if action.get("seat") == seat:
                break
            if _is_aggressive(action):
                opponent_bet_first = True
                break
        if seat_flop and not opponent_bet_first:
            _increment(counters, "c_bet_opps")
            if _is_aggressive(seat_flop[0]):
                _increment(counters, "c_bets")

    if saw_flop and aggressor is not None and aggressor != seat:
        cbet_index: Optional[int] = None
        for index, action in enumerate(flop):
            if action.get("seat") == aggressor and _is_aggressive(action):
                cbet_index = index
                break
        if cbet_index is not None:
            prior_bets = any(_is_aggressive(action) for action in flop[:cbet_index])
            if not prior_bets:
                response = next(
                    (
                        action
                        for action in flop[cbet_index + 1 :]
                        if action.get("seat") == seat
                    ),
                    None,
                )
                if response is not None:
                    _increment(counters, "fold_to_c_bet_opps")
                    if str(response.get("type") or "") == "fold":
                        _increment(counters, "fold_to_c_bets")

    for street in ("flop", "turn", "river"):
        checked = False
        for action in _actions_for(record, seat, street):
            action_type = str(action.get("type") or "")
            if action_type == "check":
                checked = True
                _increment(counters, "postflop_checks")
            elif action_type == "bet":
                _increment(counters, "postflop_bets")
            elif action_type == "raise":
                _increment(counters, "postflop_raises")
                if checked:
                    _increment(counters, "check_raises")
                    checked = False
            elif action_type == "all_in":
                _increment(counters, "postflop_raises")
                _increment(counters, "all_ins")
                if checked:
                    _increment(counters, "check_raises")
                    checked = False
            elif action_type == "call":
                _increment(counters, "postflop_calls")
            elif action_type == "fold":
                _increment(counters, "postflop_folds")

    for action in _all_actions(record):
        if action.get("seat") == seat and str(action.get("type") or "") == "all_in":
            if action.get("street") == "preflop":
                _increment(counters, "all_ins")

    if str(record.get("reason") or "") == "showdown":
        _increment(counters, "showdowns")
        if seat in (record.get("winner_seats") or []):
            _increment(counters, "showdown_wins")
        for entry in record.get("showdown") or []:
            if isinstance(entry, dict) and entry.get("seat") == seat:
                category = str(entry.get("category") or "")
                if category:
                    categories = counters["categories"]
                    categories[category] = int(categories.get(category, 0)) + 1
                break

    talk = record.get("table_talk_count") or {}
    if isinstance(talk, Mapping):
        count = talk.get(seat, talk.get(str(seat), 0))
        _increment(counters, "table_talk_messages", int(count or 0))

    return counters


def compute_counters(
    records: Sequence[Mapping],
    seat: int,
    position: Optional[str] = None,
) -> dict:
    """Aggregate public counters for one seat.

    Args:
        records: Public hand records in chronological order.
        seat: Seat index whose actions are counted.
        position: Optional ``button`` or ``bb`` filter.

    Returns:
        A JSON-safe counter dictionary.  Percentage metrics are derived later
        by :func:`finalize_counters`.
    """
    total = _empty_counters()
    for record in records:
        if not isinstance(record, Mapping):
            continue
        if position is not None and _position(record, seat) != position:
            continue
        one = _compute_one(record, seat)
        for key in _COUNTER_KEYS:
            if key == "bb_net":
                total[key] = float(total[key]) + float(one[key])
            else:
                total[key] = int(total[key]) + int(one[key])
        categories = one.get("categories") or {}
        for category, count in categories.items():
            total["categories"][category] = int(total["categories"].get(category, 0)) + int(count)
    return total


def merge_counters(items: Iterable[Mapping]) -> dict:
    """Merge counter dictionaries produced by :func:`compute_counters`."""
    total = _empty_counters()
    for item in items:
        if not isinstance(item, Mapping):
            continue
        for key in _COUNTER_KEYS:
            if key == "bb_net":
                total[key] = float(total[key]) + float(item.get(key, 0) or 0)
            else:
                total[key] = int(total[key]) + int(item.get(key, 0) or 0)
        categories = item.get("categories") or {}
        if isinstance(categories, Mapping):
            for category, count in categories.items():
                total["categories"][category] = int(total["categories"].get(category, 0)) + int(count or 0)
    return total


def _confidence(n: int) -> str:
    """Return a coarse sample-size label."""
    if n <= 0:
        return "none"
    if n < 10:
        return "low"
    if n < 30:
        return "medium"
    return "high"


def _metric(numerator: int, denominator: int) -> dict:
    """Return one percentage metric with its sample size and confidence."""
    if denominator <= 0:
        return {"value": None, "n": 0, "confidence": "none"}
    return {
        "value": round(float(numerator) / float(denominator), 4),
        "n": int(denominator),
        "confidence": _confidence(int(denominator)),
    }


def finalize_counters(counters: Mapping) -> dict:
    """Turn raw counters into the JSON structure returned by the tool."""
    hands = int(counters.get("hands", 0) or 0)
    saw_flop = int(counters.get("saw_flop", 0) or 0)
    postflop_aggressive = int(counters.get("postflop_bets", 0) or 0) + int(
        counters.get("postflop_raises", 0) or 0
    )
    postflop_passive = int(counters.get("postflop_calls", 0) or 0) + int(
        counters.get("postflop_folds", 0) or 0
    )
    net = int(counters.get("net", 0) or 0)
    bb_net = float(counters.get("bb_net", 0.0) or 0.0)
    pot_total = int(counters.get("pot_total", 0) or 0)
    return {
        "preflop": {
            "vpip": _metric(int(counters.get("vpip", 0) or 0), hands),
            "pfr": _metric(int(counters.get("pfr", 0) or 0), hands),
            "limp": _metric(int(counters.get("limps", 0) or 0), hands),
            "three_bet": _metric(
                int(counters.get("three_bets", 0) or 0),
                int(counters.get("three_bet_opps", 0) or 0),
            ),
            "fold_to_three_bet": _metric(
                int(counters.get("fold_to_three_bets", 0) or 0),
                int(counters.get("fold_to_three_bet_opps", 0) or 0),
            ),
            "steal": _metric(
                int(counters.get("steals", 0) or 0),
                int(counters.get("steal_opps", 0) or 0),
            ),
            "fold_to_steal": _metric(
                int(counters.get("fold_to_steals", 0) or 0),
                int(counters.get("fold_to_steal_opps", 0) or 0),
            ),
        },
        "postflop": {
            "c_bet": _metric(
                int(counters.get("c_bets", 0) or 0),
                int(counters.get("c_bet_opps", 0) or 0),
            ),
            "fold_to_c_bet": _metric(
                int(counters.get("fold_to_c_bets", 0) or 0),
                int(counters.get("fold_to_c_bet_opps", 0) or 0),
            ),
            "check_raise": _metric(
                int(counters.get("check_raises", 0) or 0),
                saw_flop,
            ),
            "aggression": _metric(
                postflop_aggressive,
                postflop_aggressive + postflop_passive,
            ),
            "bets": int(counters.get("postflop_bets", 0) or 0),
            "raises": int(counters.get("postflop_raises", 0) or 0),
            "calls": int(counters.get("postflop_calls", 0) or 0),
            "folds": int(counters.get("postflop_folds", 0) or 0),
            "checks": int(counters.get("postflop_checks", 0) or 0),
        },
        "showdown": {
            "wtsd": _metric(int(counters.get("showdowns", 0) or 0), saw_flop),
            "w_sd": _metric(
                int(counters.get("showdown_wins", 0) or 0),
                int(counters.get("showdowns", 0) or 0),
            ),
            "showdowns": int(counters.get("showdowns", 0) or 0),
            "categories": dict(counters.get("categories") or {}),
        },
        "result": {
            "hands": hands,
            "net": net,
            "bb_per_100": round(bb_net / hands * 100.0, 2) if hands else 0.0,
            "avg_pot": round(float(pot_total) / hands, 1) if hands else 0.0,
            "all_ins": int(counters.get("all_ins", 0) or 0),
        },
        "table_talk": {
            "messages": int(counters.get("table_talk_messages", 0) or 0),
            "per_hand": round(float(counters.get("table_talk_messages", 0) or 0) / hands, 2)
            if hands
            else 0.0,
        },
    }


def _recent_hands(records: Sequence[Mapping], seat: int, limit: int) -> list[dict]:
    """Return compact recent results from one seat's perspective."""
    seat = _seat_index(seat)
    recent: list[dict] = []
    limit = max(0, int(limit))
    if limit == 0:
        return recent
    for record in list(records)[-limit:]:
        if not isinstance(record, Mapping):
            continue
        player = _result_player(record, seat)
        recent.append(
            {
                "hand_id": int(record.get("hand_id", 0) or 0),
                "you_won": seat in (record.get("winner_seats") or []),
                "reason": record.get("reason"),
                "pot": int(record.get("pot", 0) or 0),
                "board": list(record.get("board") or []),
                "your_net": int(player.get("net", 0) or 0),
                "your_chips_after": int(player.get("chips_after", 0) or 0),
            }
        )
    return recent


def _filter_groups(metrics: Mapping, focus: str) -> dict:
    """Keep only the requested metric group."""
    focus = str(focus or "all").strip().lower()
    if focus not in _FOCUS_GROUPS:
        focus = "all"
    if focus == "all":
        return dict(metrics)
    if focus in metrics:
        return {focus: metrics[focus]}
    return dict(metrics)


def _empty_report(seat: int, opponent_identity: str) -> dict:
    return {
        "hands": 0,
        "you": finalize_counters(_empty_counters()),
        "opponent": finalize_counters(_empty_counters()),
        "completed_matches": {
            "count": 0,
            "vs_current_opponent": {
                "matches": 0,
                "hands": 0,
                "you": finalize_counters(_empty_counters()),
                "opponent": finalize_counters(_empty_counters()),
            },
            "overall": {
                "matches": 0,
                "hands": 0,
                "you": finalize_counters(_empty_counters()),
                "opponent": finalize_counters(_empty_counters()),
                "opponents": "mixed",
            },
        },
        "notes": [],
    }


def _match_sides(match: Mapping, hero_identity: str, opponent_identity: str) -> Optional[tuple[int, int]]:
    """Return the historical seats for the hero and the opponent."""
    identities = list(match.get("identities") or [])
    if len(identities) != 2:
        return None
    if hero_identity and opponent_identity:
        for seat in (0, 1):
            if identities[seat] == hero_identity and identities[1 - seat] == opponent_identity:
                return seat, 1 - seat
    if opponent_identity:
        for seat in (0, 1):
            if identities[1 - seat] == opponent_identity:
                return seat, 1 - seat
    return 0, 1


def _completed_block(
    matches: Sequence[Mapping],
    hero_identity: str,
    opponent_identity: str,
    focus: str,
) -> dict:
    hero_counters: list[Mapping] = []
    opponent_counters: list[Mapping] = []
    recent: list[dict] = []
    hands = 0
    for match in matches:
        sides = _match_sides(match, hero_identity, opponent_identity)
        if sides is None:
            continue
        hero_seat, opponent_seat = sides
        counters = match.get("counters") or {}
        hero_counter = (
            counters.get(hero_seat) or counters.get(str(hero_seat)) or {}
        )
        opponent_counter = (
            counters.get(opponent_seat)
            or counters.get(str(opponent_seat))
            or {}
        )
        hero_counters.append(hero_counter)
        opponent_counters.append(opponent_counter)
        hands += int(match.get("hands", 0) or 0)
        identities = list(match.get("identities") or [])
        recent.append(
            {
                "match_id": str(match.get("match_id") or ""),
                "hands": int(match.get("hands", 0) or 0),
                "opponent": (
                    identities[opponent_seat]
                    if opponent_seat < len(identities)
                    else ""
                ),
                "you_net": int(hero_counter.get("net", 0) or 0),
                "opponent_net": int(opponent_counter.get("net", 0) or 0),
            }
        )
    return {
        "matches": len(hero_counters),
        "hands": hands,
        "you": _filter_groups(finalize_counters(merge_counters(hero_counters)), focus),
        "opponent": _filter_groups(finalize_counters(merge_counters(opponent_counters)), focus),
        "recent_matches": recent[-5:],
    }


def _match_records_for_opponent(
    matches: Sequence[Mapping],
    hero_identity: str,
    opponent_identity: str,
) -> list[Mapping]:
    if not opponent_identity:
        return list(matches)
    selected: list[Mapping] = []
    for match in matches:
        sides = _match_sides(match, hero_identity, opponent_identity)
        if sides is None:
            continue
        hero_seat, opponent_seat = sides
        identities = list(match.get("identities") or [])
        if opponent_identity and identities[opponent_seat] == opponent_identity:
            selected.append(match)
    return selected


def build_history_report(
    records: Sequence[Mapping],
    completed_matches: Sequence[Mapping],
    seat: int,
    *,
    scope: str = "both",
    last_hands: int = 20,
    focus: str = "all",
    hero_identity: str = "",
    opponent_identity: str = "",
    match_info: Optional[Mapping] = None,
) -> dict:
    """Build the JSON returned by the history tool and checkpoint prompt.

    Args:
        records: Current-match public hand records.
        completed_matches: Archived matches from this process.
        seat: Seat asking for the report.
        scope: ``current_match``, ``completed_matches`` or ``both``.
        last_hands: Number of recent results to include.
        focus: ``all``, ``preflop``, ``postflop``, ``showdown``,
            ``result`` or ``table_talk``.
        hero_identity: Stable identity of the asking seat's player.
        opponent_identity: Stable identity of the current opponent.
        match_info: Optional current stacks/blinds/rules.

    Returns:
        A JSON-safe report.  No hole cards are included.
    """
    seat = _seat_index(seat)
    scope = str(scope or "both").strip().lower()
    if scope not in _SCOPES:
        scope = "both"
    focus = str(focus or "all").strip().lower()
    if focus not in _FOCUS_GROUPS:
        focus = "all"
    last_hands = max(0, min(int(last_hands or 0), 50))

    current_counters = compute_counters(records, seat)
    opponent_counters = compute_counters(records, _opponent(seat))
    current_you = _filter_groups(finalize_counters(current_counters), focus)
    current_opponent = _filter_groups(finalize_counters(opponent_counters), focus)
    current_positions = {
        "you": {
            "button": _filter_groups(
                finalize_counters(compute_counters(records, seat, "button")), focus
            ),
            "bb": _filter_groups(
                finalize_counters(compute_counters(records, seat, "bb")), focus
            ),
        },
        "opponent": {
            "button": _filter_groups(
                finalize_counters(compute_counters(records, _opponent(seat), "button")),
                focus,
            ),
            "bb": _filter_groups(
                finalize_counters(compute_counters(records, _opponent(seat), "bb")),
                focus,
            ),
        },
    }

    report: dict = {
        "scope": scope,
        "current_match": {
            "hands": len(records),
            "you": current_you,
            "opponent": current_opponent,
            "positions": current_positions,
            "recent_hands": _recent_hands(records, seat, last_hands),
        },
        "completed_matches": {
            "count": 0,
            "vs_current_opponent": {
                "matches": 0,
                "hands": 0,
                "you": _filter_groups(finalize_counters(_empty_counters()), focus),
                "opponent": _filter_groups(finalize_counters(_empty_counters()), focus),
                "recent_matches": [],
            },
            "overall": {
                "matches": 0,
                "hands": 0,
                "you": _filter_groups(finalize_counters(_empty_counters()), focus),
                "opponent": _filter_groups(finalize_counters(_empty_counters()), focus),
                "opponents": "mixed",
                "recent_matches": [],
            },
        },
        "notes": [
            "All statistics use public actions and shown-down information only.",
            "No opponent hole cards are stored or returned.",
            "Statistics cover completed hands; the current hand is in the table update.",
            "Each percentage includes its sample size n and a coarse confidence label.",
            "Small samples are noisy; treat low-confidence reads as hypotheses.",
            "The checkpoint should record conclusions, not copy this table.",
        ],
    }

    if match_info:
        report["current_match"].update(
            {
                "blinds": {
                    "small": int(match_info.get("small_blind", 0) or 0),
                    "big": int(match_info.get("big_blind", 0) or 0),
                },
                "stacks": {
                    "you": int(match_info.get("your_chips", 0) or 0),
                    "opponent": int(match_info.get("opponent_chips", 0) or 0),
                },
            }
        )

    if scope in ("completed_matches", "both"):
        vs_matches = _match_records_for_opponent(
            completed_matches, hero_identity, opponent_identity
        )
        report["completed_matches"]["count"] = len(completed_matches)
        report["completed_matches"]["vs_current_opponent"] = _completed_block(
            vs_matches, hero_identity, opponent_identity, focus
        )
        report["completed_matches"]["overall"] = _completed_block(
            completed_matches, hero_identity, opponent_identity, focus
        )
        report["completed_matches"]["overall"]["opponents"] = "mixed"

    if scope == "current_match":
        report["completed_matches"] = {
            "count": 0,
            "vs_current_opponent": {
                "matches": 0,
                "hands": 0,
                "you": _filter_groups(finalize_counters(_empty_counters()), focus),
                "opponent": _filter_groups(finalize_counters(_empty_counters()), focus),
                "recent_matches": [],
            },
            "overall": {
                "matches": 0,
                "hands": 0,
                "you": _filter_groups(finalize_counters(_empty_counters()), focus),
                "opponent": _filter_groups(finalize_counters(_empty_counters()), focus),
                "opponents": "mixed",
                "recent_matches": [],
            },
        }
    if scope == "completed_matches":
        report["current_match"] = {
            "hands": 0,
            "you": _filter_groups(finalize_counters(_empty_counters()), focus),
            "opponent": _filter_groups(finalize_counters(_empty_counters()), focus),
            "positions": {
                "you": {
                    "button": _filter_groups(finalize_counters(_empty_counters()), focus),
                    "bb": _filter_groups(finalize_counters(_empty_counters()), focus),
                },
                "opponent": {
                    "button": _filter_groups(finalize_counters(_empty_counters()), focus),
                    "bb": _filter_groups(finalize_counters(_empty_counters()), focus),
                },
            },
            "recent_hands": [],
        }

    return report


def _compact_metric(metric: Mapping) -> Optional[dict]:
    if not isinstance(metric, Mapping) or metric.get("value") is None:
        return None
    return {"value": metric.get("value"), "n": metric.get("n")}


def _compact_group(metrics: Mapping) -> dict:
    result: dict = {}
    for group in _METRIC_GROUPS:
        values = metrics.get(group)
        if not isinstance(values, Mapping):
            continue
        if group == "result":
            result[group] = {
                "net": values.get("net"),
                "bb_per_100": values.get("bb_per_100"),
                "hands": values.get("hands"),
            }
            continue
        compact = {
            key: _compact_metric(value)
            for key, value in values.items()
            if isinstance(value, Mapping)
        }
        compact = {key: value for key, value in compact.items() if value is not None}
        if compact:
            result[group] = compact
    return result


def build_checkpoint_snapshot(
    records: Sequence[Mapping],
    completed_matches: Sequence[Mapping],
    seat: int,
    *,
    hero_identity: str = "",
    opponent_identity: str = "",
    last_hands: int = 5,
    match_info: Optional[Mapping] = None,
) -> dict:
    """Build the compact statistics block used inside a summary prompt.

    The block is intentionally small.  It is an input to the summarizer, not
    text that should be copied into the checkpoint.  The next agent can call
    ``research_history`` for the full current table.
    """
    report = build_history_report(
        records,
        completed_matches,
        seat,
        scope="both",
        last_hands=last_hands,
        focus="all",
        hero_identity=hero_identity,
        opponent_identity=opponent_identity,
        match_info=match_info,
    )
    current = report.get("current_match") or {}
    completed = report.get("completed_matches") or {}
    vs = completed.get("vs_current_opponent") or {}
    positions = current.get("positions") or {}
    opponent_positions = positions.get("opponent") or {}
    return {
        "current": {
            "hands": current.get("hands", 0),
            "blinds": current.get("blinds"),
            "stacks": current.get("stacks"),
            "you": _compact_group(current.get("you") or {}),
            "opponent": _compact_group(current.get("opponent") or {}),
            "opponent_by_position": {
                "button": _compact_group(opponent_positions.get("button") or {}),
                "bb": _compact_group(opponent_positions.get("bb") or {}),
            },
            "recent_hands": current.get("recent_hands") or [],
        },
        "completed_vs_current_opponent": {
            "matches": vs.get("matches", 0),
            "hands": vs.get("hands", 0),
            "you": _compact_group(vs.get("you") or {}),
            "opponent": _compact_group(vs.get("opponent") or {}),
        },
        "notes": [
            "Use these numbers for your own reasoning only.",
            "Do not copy this block, its JSON, or a metric table into the checkpoint.",
            "The next agent can call research_history for current numbers.",
            "Treat low-n reads as hypotheses.",
        ],
    }


def build_public_hand_record(
    game,
    result: Mapping,
    table_talk_count: Optional[Mapping] = None,
) -> dict:
    """Create a sanitized hand record from the live game and result.

    Args:
        game: Finished ``HeadsUpGame`` instance.
        result: Engine hand result.
        table_talk_count: Optional per-seat table-talk message counts.

    Returns:
        A JSON-safe public record with no hole cards or deck information.
    """
    actions = []
    for action in getattr(game, "actions", []) or []:
        if not isinstance(action, Mapping):
            continue
        actions.append(
            {
                "hand_id": action.get("hand_id"),
                "street": action.get("street"),
                "seat": action.get("seat"),
                "type": action.get("type"),
                "amount": action.get("amount"),
                "added": action.get("added"),
                "pot_after": action.get("pot_after"),
                "chips_after": action.get("chips_after"),
                "source": action.get("source"),
                "auto": action.get("auto"),
            }
        )

    players = []
    for player in result.get("players") or []:
        if not isinstance(player, Mapping):
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
    for entry in result.get("showdown") or []:
        if not isinstance(entry, Mapping):
            continue
        showdown.append(
            {
                "seat": entry.get("seat"),
                "category": entry.get("category"),
                "best_five": list(entry.get("best_five") or []),
            }
        )

    return {
        "hand_id": int(result.get("hand_id", getattr(game, "hand_id", 0)) or 0),
        "button": getattr(game, "button_index", 0),
        "small_blind": int(getattr(game, "small_blind", 0) or 0),
        "big_blind": int(getattr(game, "big_blind", 0) or 0),
        "board": list(result.get("board") or []),
        "reason": result.get("reason"),
        "winner_seats": list(result.get("winner_seats") or []),
        "pot": int(result.get("pot", 0) or 0),
        "players": players,
        "actions": actions,
        "showdown": showdown,
        "table_talk_count": {
            int(seat): int(count)
            for seat, count in (table_talk_count or {}).items()
        },
    }


def build_match_record(
    match_id: str,
    records: Sequence[Mapping],
    identities: Sequence[str],
    *,
    started_at: float = 0.0,
    ended_at: float = 0.0,
    rules: Optional[Mapping] = None,
    recent_limit: int = 20,
) -> dict:
    """Build an archived match record with counters for both seats."""
    seat_counters = {
        0: compute_counters(records, 0),
        1: compute_counters(records, 1),
    }
    button_counters = {
        0: compute_counters(records, 0, "button"),
        1: compute_counters(records, 1, "button"),
    }
    bb_counters = {
        0: compute_counters(records, 0, "bb"),
        1: compute_counters(records, 1, "bb"),
    }
    return {
        "match_id": str(match_id),
        "hands": len(records),
        "identities": [str(item) for item in identities],
        "started_at": float(started_at),
        "ended_at": float(ended_at),
        "rules": {
            key: (rules or {}).get(key)
            for key in ("starting_chips", "small_blind", "big_blind", "auto_rebuy")
        },
        "counters": {seat: value for seat, value in seat_counters.items()},
        "counters_button": {seat: value for seat, value in button_counters.items()},
        "counters_bb": {seat: value for seat, value in bb_counters.items()},
        "recent_results": {
            seat: _recent_hands(records, seat, recent_limit) for seat in (0, 1)
        },
    }


@dataclass
class HistoryStats:
    """Current-match record store and meter."""

    records: list[dict] = field(default_factory=list)

    def add(self, record: Mapping) -> None:
        """Append one public hand record."""
        if isinstance(record, Mapping):
            self.records.append(dict(record))

    def clear(self) -> None:
        """Reset the current match."""
        self.records = []

    def counters(self, seat: int, position: Optional[str] = None) -> dict:
        """Return raw counters for one seat."""
        return compute_counters(self.records, seat, position)

    def metrics(self, seat: int, position: Optional[str] = None) -> dict:
        """Return finalized metrics for one seat."""
        return finalize_counters(self.counters(seat, position))

    def recent(self, seat: int, limit: int = 20) -> list[dict]:
        """Return recent compact results for one seat."""
        return _recent_hands(self.records, seat, limit)

    def report(
        self,
        completed_matches: Sequence[Mapping],
        seat: int,
        *,
        scope: str = "both",
        last_hands: int = 20,
        focus: str = "all",
        hero_identity: str = "",
        opponent_identity: str = "",
        match_info: Optional[Mapping] = None,
    ) -> dict:
        """Build the full history report."""
        return build_history_report(
            self.records,
            completed_matches,
            seat,
            scope=scope,
            last_hands=last_hands,
            focus=focus,
            hero_identity=hero_identity,
            opponent_identity=opponent_identity,
            match_info=match_info,
        )

    def checkpoint_snapshot(
        self,
        completed_matches: Sequence[Mapping],
        seat: int,
        *,
        hero_identity: str = "",
        opponent_identity: str = "",
        last_hands: int = 5,
        match_info: Optional[Mapping] = None,
    ) -> dict:
        """Build the compact snapshot injected into a summary prompt."""
        return build_checkpoint_snapshot(
            self.records,
            completed_matches,
            seat,
            hero_identity=hero_identity,
            opponent_identity=opponent_identity,
            last_hands=last_hands,
            match_info=match_info,
        )


__all__ = [
    "HistoryStats",
    "build_checkpoint_snapshot",
    "build_history_report",
    "build_match_record",
    "build_public_hand_record",
    "compute_counters",
    "finalize_counters",
    "merge_counters",
]
