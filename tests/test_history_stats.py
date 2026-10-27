"""Tests for public action-history statistics."""

from __future__ import annotations

import json

from src.analysis.history import (
    HistoryStats,
    build_checkpoint_snapshot,
    build_history_report,
    build_match_record,
    build_public_hand_record,
    compute_counters,
    finalize_counters,
    merge_counters,
)


def action(
    seat: int,
    street: str,
    kind: str,
    amount: int = 0,
) -> dict:
    return {
        "hand_id": 0,
        "street": street,
        "seat": seat,
        "type": kind,
        "amount": amount,
        "added": amount,
        "pot_after": 0,
        "chips_after": 0,
        "source": "llm",
        "auto": False,
    }


def hand(
    hand_id: int,
    button: int,
    actions: list[dict],
    *,
    board: list[str] | None = None,
    reason: str = "fold",
    winner: int = 0,
    pot: int = 0,
    hero_net: int = 0,
    opponent_net: int = 0,
    showdown: list[dict] | None = None,
) -> dict:
    return {
        "hand_id": hand_id,
        "button": button,
        "small_blind": 5,
        "big_blind": 10,
        "board": board or [],
        "reason": reason,
        "winner_seats": [winner],
        "pot": pot,
        "players": [
            {"seat": 0, "net": hero_net, "chips_after": 1000 + hero_net},
            {"seat": 1, "net": opponent_net, "chips_after": 1000 + opponent_net},
        ],
        "actions": actions,
        "showdown": showdown or [],
        "table_talk_count": {0: 0, 1: 0},
    }


def four_hands() -> list[dict]:
    """A compact four-hand sample covering the main metrics."""
    return [
        hand(
            1,
            0,
            [
                action(0, "preflop", "raise", 30),
                action(1, "preflop", "fold"),
            ],
            reason="fold",
            winner=0,
            pot=40,
            hero_net=10,
            opponent_net=-10,
        ),
        hand(
            2,
            1,
            [
                action(1, "preflop", "raise", 30),
                action(0, "preflop", "call", 30),
                action(1, "flop", "bet", 30),
                action(0, "flop", "fold"),
            ],
            board=["Qh", "Jh", "2c"],
            reason="fold",
            winner=1,
            pot=90,
            hero_net=-30,
            opponent_net=30,
        ),
        hand(
            3,
            0,
            [
                action(0, "preflop", "raise", 30),
                action(1, "preflop", "call", 30),
                action(1, "flop", "check"),
                action(0, "flop", "bet", 40),
                action(1, "flop", "call", 40),
                action(1, "turn", "check"),
                action(0, "turn", "check"),
                action(1, "river", "check"),
                action(0, "river", "check"),
            ],
            board=["Qh", "Jh", "2c", "7d", "9s"],
            reason="showdown",
            winner=0,
            pot=140,
            hero_net=70,
            opponent_net=-70,
            showdown=[
                {"seat": 0, "category": "pair", "best_five": ["Qh", "Qd", "Jh", "9s", "7d"]},
                {"seat": 1, "category": "high_card", "best_five": ["Ah", "Jh", "9s", "7d", "2c"]},
            ],
        ),
        hand(
            4,
            1,
            [
                action(1, "preflop", "call", 10),
                action(0, "preflop", "check"),
                action(0, "flop", "bet", 20),
                action(1, "flop", "fold"),
            ],
            board=["Qh", "Jh", "2c"],
            reason="fold",
            winner=0,
            pot=40,
            hero_net=20,
            opponent_net=-20,
        ),
    ]


def test_preflop_counters_and_sample_sizes() -> None:
    records = four_hands()
    hero = finalize_counters(compute_counters(records, 0))
    villain = finalize_counters(compute_counters(records, 1))

    assert hero["preflop"]["vpip"]["value"] == 0.75
    assert hero["preflop"]["vpip"]["n"] == 4
    assert hero["preflop"]["pfr"]["value"] == 0.5
    assert hero["preflop"]["three_bet"]["n"] == 1
    assert hero["preflop"]["three_bet"]["value"] == 0.0

    assert villain["preflop"]["vpip"]["value"] == 0.75
    assert villain["preflop"]["pfr"]["value"] == 0.25
    assert villain["preflop"]["steal"]["value"] == 0.5
    assert villain["preflop"]["steal"]["n"] == 2
    assert villain["preflop"]["fold_to_steal"]["value"] == 0.5
    assert villain["preflop"]["fold_to_steal"]["n"] == 2


def test_all_in_call_is_not_counted_as_aggressive() -> None:
    records = [
        hand(
            1,
            0,
            [
                action(0, "preflop", "all_in", 1000),
                action(1, "preflop", "all_in", 1000),
            ],
            reason="showdown",
            winner=0,
        )
    ]
    sb = finalize_counters(compute_counters(records, 0))
    bb = finalize_counters(compute_counters(records, 1))

    assert sb["preflop"]["pfr"]["value"] == 1.0
    assert bb["preflop"]["pfr"]["value"] == 0.0
    assert bb["preflop"]["three_bet"]["value"] == 0.0


def test_postflop_all_in_call_is_not_counted_as_raise() -> None:
    records = [
        hand(
            1,
            0,
            [
                action(0, "preflop", "raise", 30),
                action(1, "preflop", "call", 30),
                action(1, "flop", "all_in", 500),
                action(0, "flop", "all_in", 500),
            ],
            board=["Qh", "Jh", "2c"],
            reason="showdown",
            winner=0,
        )
    ]
    hero = finalize_counters(compute_counters(records, 0))
    villain = finalize_counters(compute_counters(records, 1))

    assert hero["postflop"]["raises"] == 0
    assert hero["postflop"]["calls"] == 1
    assert hero["result"]["all_ins"] == 1
    assert villain["postflop"]["raises"] == 1
    assert villain["postflop"]["calls"] == 0


def test_four_bet_does_not_pollute_three_bet_metrics() -> None:
    records = [
        hand(
            1,
            0,
            [
                action(0, "preflop", "raise", 30),
                action(1, "preflop", "raise", 90),
                action(0, "preflop", "raise", 270),
                action(1, "preflop", "fold"),
            ],
            reason="fold",
            winner=0,
        )
    ]
    sb = finalize_counters(compute_counters(records, 0))
    bb = finalize_counters(compute_counters(records, 1))

    # The small blind made a 4-bet, not a 3-bet.
    assert sb["preflop"]["three_bet"]["value"] is None
    # The big blind made the only 3-bet and folded to a 4-bet.
    assert bb["preflop"]["three_bet"]["value"] == 1.0
    assert bb["preflop"]["fold_to_three_bet"]["value"] is None


def test_postflop_c_bet_and_fold_to_c_bet() -> None:
    records = four_hands()
    hero = finalize_counters(compute_counters(records, 0))
    villain = finalize_counters(compute_counters(records, 1))

    assert hero["postflop"]["c_bet"] == {
        "value": 1.0,
        "n": 1,
        "confidence": "low",
    }
    assert hero["postflop"]["fold_to_c_bet"] == {
        "value": 1.0,
        "n": 1,
        "confidence": "low",
    }
    assert villain["postflop"]["c_bet"] == {
        "value": 1.0,
        "n": 1,
        "confidence": "low",
    }
    assert villain["postflop"]["fold_to_c_bet"]["value"] == 0.0
    assert villain["postflop"]["fold_to_c_bet"]["n"] == 1
    assert hero["postflop"]["bets"] == 2
    assert hero["postflop"]["calls"] == 0
    assert hero["postflop"]["folds"] == 1


def test_position_splits_use_the_button_field() -> None:
    records = four_hands()
    hero_button = finalize_counters(compute_counters(records, 0, "button"))
    hero_bb = finalize_counters(compute_counters(records, 0, "bb"))
    villain_button = finalize_counters(compute_counters(records, 1, "button"))
    villain_bb = finalize_counters(compute_counters(records, 1, "bb"))

    assert hero_button["result"]["hands"] == 2
    assert hero_button["preflop"]["steal"]["value"] == 1.0
    assert hero_button["preflop"]["steal"]["n"] == 2
    assert hero_bb["result"]["hands"] == 2
    assert hero_bb["preflop"]["vpip"]["value"] == 0.5
    assert villain_button["result"]["hands"] == 2
    assert villain_button["preflop"]["steal"]["value"] == 0.5
    assert villain_bb["result"]["hands"] == 2


def test_showdown_and_result_metrics() -> None:
    records = four_hands()
    hero = finalize_counters(compute_counters(records, 0))
    villain = finalize_counters(compute_counters(records, 1))

    assert hero["showdown"]["wtsd"] == {
        "value": 0.3333,
        "n": 3,
        "confidence": "low",
    }
    assert hero["showdown"]["w_sd"] == {
        "value": 1.0,
        "n": 1,
        "confidence": "low",
    }
    assert villain["showdown"]["wtsd"]["value"] == 0.3333
    assert villain["showdown"]["w_sd"]["value"] == 0.0
    assert hero["showdown"]["categories"] == {"pair": 1}
    assert villain["showdown"]["categories"] == {"high_card": 1}
    assert hero["result"]["hands"] == 4
    assert hero["result"]["net"] == 70
    assert hero["result"]["bb_per_100"] == 175.0
    assert hero["result"]["avg_pot"] == 77.5


def test_empty_history_is_safe_and_json_safe() -> None:
    hero = finalize_counters(compute_counters([], 0))
    assert hero["preflop"]["vpip"] == {
        "value": None,
        "n": 0,
        "confidence": "none",
    }
    assert hero["showdown"]["wtsd"]["value"] is None
    assert hero["result"]["hands"] == 0
    assert hero["result"]["bb_per_100"] == 0.0
    json.dumps(hero, allow_nan=False)


def test_merge_counters_is_additive() -> None:
    records = four_hands()
    first = compute_counters(records[:2], 0)
    second = compute_counters(records[2:], 0)
    merged = merge_counters([first, second])
    assert merged["hands"] == 4
    assert merged["vpip"] == 3
    assert merged["net"] == 70
    assert merged["categories"] == {"pair": 1}


def test_history_report_merges_completed_matches_and_filters_opponent() -> None:
    records = four_hands()
    match = build_match_record(
        "match-1",
        records,
        ["llm:hero", "llm:villain"],
        rules={"big_blind": 10},
    )
    stats = HistoryStats()
    for record in records:
        stats.add(record)

    vs = build_history_report(
        stats.records,
        [match],
        0,
        scope="both",
        last_hands=2,
        hero_identity="llm:hero",
        opponent_identity="llm:villain",
    )
    assert vs["current_match"]["hands"] == 4
    assert vs["completed_matches"]["vs_current_opponent"]["matches"] == 1
    assert vs["completed_matches"]["vs_current_opponent"]["hands"] == 4
    assert vs["completed_matches"]["vs_current_opponent"]["opponent"]["preflop"]["vpip"]["value"] == 0.75
    assert len(vs["completed_matches"]["vs_current_opponent"]["recent_matches"]) == 1
    assert vs["completed_matches"]["vs_current_opponent"]["recent_matches"][0]["hands"] == 4
    assert len(vs["current_match"]["recent_hands"]) == 2

    other = build_history_report(
        stats.records,
        [match],
        0,
        scope="both",
        hero_identity="llm:hero",
        opponent_identity="llm:other",
    )
    assert other["completed_matches"]["vs_current_opponent"]["matches"] == 0
    assert other["completed_matches"]["overall"]["matches"] == 1


def test_overall_history_uses_hero_identity_not_current_opponent() -> None:
    records = [
        hand(
            1,
            0,
            [
                action(0, "preflop", "raise", 30),
                action(1, "preflop", "fold"),
            ],
            reason="fold",
            winner=0,
        )
    ]
    match = build_match_record(
        "match-1",
        records,
        ["llm:other", "llm:hero"],
        rules={"big_blind": 10},
    )

    report = build_history_report(
        [],
        [match],
        0,
        scope="completed_matches",
        hero_identity="llm:hero",
        opponent_identity="llm:villain",
    )
    overall = report["completed_matches"]["overall"]
    assert overall["matches"] == 1
    # Hero was seat 1 and never acted voluntarily. The old fallback mapped
    # seat 0 to "you" and reported a 100% VPIP.
    assert overall["you"]["preflop"]["vpip"]["value"] == 0.0

    unrelated = build_match_record(
        "match-2",
        records,
        ["llm:other", "llm:third"],
        rules={"big_blind": 10},
    )
    skipped = build_history_report(
        [],
        [unrelated],
        0,
        scope="completed_matches",
        hero_identity="llm:hero",
        opponent_identity="llm:villain",
    )
    assert skipped["completed_matches"]["overall"]["matches"] == 0


def test_checkpoint_snapshot_is_compact_and_does_not_leak_cards() -> None:
    records = four_hands()
    records[0]["players"][1]["hole_cards"] = ["SECRET_CARD"]
    records[0]["hole_cards"] = ["SECRET_CARD"]
    snapshot = build_checkpoint_snapshot(
        records,
        [],
        0,
        hero_identity="llm:hero",
        opponent_identity="llm:villain",
        last_hands=2,
    )
    dumped = json.dumps(snapshot, ensure_ascii=False, allow_nan=False)
    assert "SECRET_CARD" not in dumped
    assert snapshot["current"]["opponent"]["preflop"]["vpip"]["value"] == 0.75
    assert snapshot["current"]["opponent_by_position"]["button"]["preflop"]["steal"]["value"] == 0.5
    assert "notes" in snapshot


def test_public_hand_record_drops_hidden_information() -> None:
    class Player:
        hole_cards = ["Ah", "Kh"]

    class Game:
        hand_id = 7
        button_index = 1
        small_blind = 5
        big_blind = 10
        actions = [
            {
                "hand_id": 7,
                "street": "preflop",
                "seat": 1,
                "type": "raise",
                "amount": 30,
                "added": 25,
                "pot_after": 40,
                "chips_after": 985,
                "source": "llm",
                "auto": False,
                "label": "加注",
            }
        ]

    result = {
        "hand_id": 7,
        "reason": "fold",
        "winner_seats": [1],
        "pot": 40,
        "board": [],
        "players": [
            {"seat": 0, "net": -10, "chips_after": 990, "total_bet": 10},
            {"seat": 1, "net": 10, "chips_after": 1010, "total_bet": 30},
        ],
        "showdown": [
            {"seat": 0, "category": "high_card", "category_name": "高牌", "best_five": ["Ah"]}
        ],
    }
    record = build_public_hand_record(Game(), result, {0: 2, 1: 1})
    dumped = json.dumps(record, ensure_ascii=False)
    assert "hole_cards" not in dumped
    assert "label" not in dumped
    assert record["button"] == 1
    assert record["actions"][0]["type"] == "raise"
    assert record["table_talk_count"] == {0: 2, 1: 1}
    assert record["showdown"][0]["category"] == "high_card"
