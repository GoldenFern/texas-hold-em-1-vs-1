"""Heads-up engine invariants and rule tests."""

from __future__ import annotations

import random

import pytest

from src.engine.actions import Action, ActionType
from src.engine.cards import Card
from src.engine.heads_up import HeadsUpGame, PlayerStatus, Street


def make_game(starting_chips: int = 1000, seed: int = 42) -> HeadsUpGame:
    game = HeadsUpGame(
        starting_chips=starting_chips,
        small_blind=5,
        big_blind=10,
        auto_rebuy=False,
        seed=seed,
    )
    game.configure_players({"name": "A", "kind": "human"}, {"name": "B", "kind": "bot"})
    game.start_hand()
    return game


def act(game: HeadsUpGame, seat: int, action_type: ActionType, amount: int = 0) -> None:
    result = game.apply(Action(seat, action_type, amount=amount, source="test"))
    assert result.ok, result.error


def test_blinds_and_postflop_position() -> None:
    game = make_game()
    assert game.button_index == 0
    assert game.players[0].is_small_blind
    assert game.players[1].is_big_blind
    assert game.players[0].chips == 995
    assert game.players[1].chips == 990
    assert game.actor == 0
    assert game.to_call(0) == 5

    act(game, 0, ActionType.RAISE, 30)
    act(game, 1, ActionType.CALL)
    assert game.street == Street.FLOP
    assert len(game.community) == 3
    assert game.actor == 1  # non-button acts first after the flop
    assert game.pot == 60


def test_minimum_raise_is_enforced() -> None:
    game = make_game()
    result = game.apply(Action(0, ActionType.RAISE, amount=15))
    assert not result.ok
    result = game.apply(Action(0, ActionType.RAISE, amount=20))
    assert result.ok
    assert game.current_bet == 20
    assert game.min_raise_to == 30


def test_short_allin_does_not_reopen_betting() -> None:
    game = HeadsUpGame(
        starting_chips=40,
        small_blind=5,
        big_blind=10,
        seed=7,
    )
    game.configure_players({"name": "A", "kind": "human"}, {"name": "B", "kind": "bot"})
    game.start_hand()
    act(game, 0, ActionType.RAISE, 30)
    assert game.current_bet == 30
    assert game.last_full_raise == 20
    assert game.players[1].chips == 30
    act(game, 1, ActionType.ALL_IN)
    assert game.players[1].status == PlayerStatus.ALL_IN
    assert game.current_bet == 40
    assert game.players[0].raise_allowed is False
    assert ActionType.RAISE not in game.legal_actions(0)
    assert ActionType.CALL in game.legal_actions(0)
    act(game, 0, ActionType.CALL)
    assert game.street in (Street.FLOP, Street.FINISHED)


def test_short_allin_does_not_offer_illegal_allin_with_chips_behind() -> None:
    game = HeadsUpGame(starting_chips=50, small_blind=5, big_blind=10, seed=8)
    game.configure_players({"name": "A", "kind": "human"}, {"name": "B", "kind": "bot"})
    game.start_hand()
    # SB has 20 chips behind after raising to 30. Give BB only 30 chips
    # behind so its all-in to 40 is a short raise.
    game.players[1].chips = 30
    act(game, 0, ActionType.RAISE, 30)
    act(game, 1, ActionType.ALL_IN)
    assert game.current_bet == 40
    assert game.players[0].chips > game.to_call(0)
    legal = game.legal_actions(0)
    assert ActionType.RAISE not in legal
    assert ActionType.ALL_IN not in legal
    assert ActionType.CALL in legal


def test_fold_returns_uncalled_chips() -> None:
    game = make_game()
    act(game, 0, ActionType.RAISE, 100)
    act(game, 1, ActionType.FOLD)
    assert game.street == Street.FINISHED
    assert game.result is not None
    assert game.result["reason"] == "fold"
    assert game.result["pot"] == 20
    assert game.result["refunds"][0] == 90
    assert game.players[0].chips == 1010
    assert game.players[1].chips == 990


def test_showdown_split_and_odd_chip() -> None:
    game = make_game(seed=11)
    # Force a board where both players play the board (a royal flush on board
    # is impossible with 5 cards, so use a broadway straight on board).
    game.community = [Card.from_str(code) for code in ("Ah", "Kd", "Qc", "Js", "Th")]
    game.street = Street.RIVER
    game.players[0].hole_cards = [Card.from_str("2c"), Card.from_str("3c")]
    game.players[1].hole_cards = [Card.from_str("4c"), Card.from_str("5c")]
    for seat in (0, 1):
        game.players[seat].total_bet = 100
        game.players[seat].chips = 900
    game._showdown()
    assert game.result is not None
    assert game.result["reason"] == "showdown"
    assert sorted(game.result["winner_seats"]) == [0, 1]
    assert sum(game.players[seat].chips for seat in (0, 1)) == 2000


def test_random_play_conserves_chips() -> None:
    rng = random.Random(2026)
    game = HeadsUpGame(
        starting_chips=500,
        small_blind=5,
        big_blind=10,
        auto_rebuy=True,
        seed=99,
        max_hands=60,
    )
    game.configure_players({"name": "A", "kind": "human"}, {"name": "B", "kind": "bot"})
    for _ in range(60):
        if game.match_over:
            break
        game.start_hand()
        total = sum(player.chips for player in game.players) + sum(
            player.total_bet for player in game.players
        )
        for _step in range(80):
            if game.street in (Street.FINISHED, Street.SHOWDOWN):
                break
            seat = game.actor
            assert seat is not None
            legal = game.legal_actions(seat)
            assert legal
            chosen = rng.choice(legal)
            constraints = game.action_constraints(seat)
            amount = 0
            if chosen in (ActionType.BET, ActionType.RAISE):
                low = constraints["min_raise_to"]
                high = constraints["max_raise_to"]
                amount = rng.randint(low, high) if high >= low else high
            result = game.apply(Action(seat, chosen, amount=amount, source="test"))
            assert result.ok, result.error
            for player in game.players:
                assert player.chips >= 0
            if game.street in (Street.FINISHED, Street.SHOWDOWN):
                assert sum(player.chips for player in game.players) >= 0
            else:
                assert (
                    sum(player.chips for player in game.players)
                    + sum(player.total_bet for player in game.players)
                    == total
                )
        assert game.street == Street.FINISHED


def test_human_snapshot_visibility() -> None:
    game = make_game()
    snapshot = game.snapshot(reveal_all=False, reveal_seats=(0,))
    assert snapshot["players"][0]["hole_hidden"] is False
    assert snapshot["players"][1]["hole_hidden"] is True
    assert len(snapshot["players"][0]["hole_cards"]) == 2
    assert snapshot["players"][1]["hole_cards"] == []
    assert snapshot["deck_remaining"] == []


def test_god_snapshot_shows_deck_and_burn() -> None:
    game = make_game()
    snapshot = game.snapshot(reveal_all=True)
    assert snapshot["players"][0]["hole_hidden"] is False
    assert snapshot["players"][1]["hole_hidden"] is False
    assert len(snapshot["deck_remaining"]) == game.deck.remaining
    assert "burned" in snapshot


def test_no_raise_when_opponent_is_all_in() -> None:
    game = HeadsUpGame(starting_chips=300, small_blind=5, big_blind=10, seed=21)
    game.configure_players({"name": "A", "kind": "human"}, {"name": "B", "kind": "bot"})
    game.start_hand()
    act(game, 0, ActionType.ALL_IN)
    assert game.players[0].status == PlayerStatus.ALL_IN
    legal = game.legal_actions(1)
    assert ActionType.RAISE not in legal
    result = game.apply(Action(1, ActionType.RAISE, amount=600, source="test"))
    assert not result.ok
    assert "不能加注" in result.error


def test_start_hand_rejected_while_hand_is_active() -> None:
    game = make_game()
    assert not game.can_start_hand
    with pytest.raises(RuntimeError):
        game.start_hand()
    assert game.hand_id == 1


def test_finished_snapshot_resets_street_fields() -> None:
    game = make_game()
    act(game, 0, ActionType.RAISE, 100)
    act(game, 1, ActionType.FOLD)
    snapshot = game.snapshot()
    assert snapshot["street"] == "finished"
    assert snapshot["current_bet"] == 0
    assert all(player["street_bet"] == 0 for player in snapshot["players"])
    assert game.result is not None
    assert game.result["players"][0]["total_bet"] == 100


def test_bet_and_raise_types_follow_to_call() -> None:
    game = make_game()
    act(game, 0, ActionType.CALL)
    assert game.actor == 1
    assert ActionType.BET in game.legal_actions(1)
    rejected = game.apply(Action(1, ActionType.RAISE, amount=20, source="test"))
    assert not rejected.ok
    accepted = game.apply(Action(1, ActionType.BET, amount=20, source="test"))
    assert accepted.ok

    facing_bet = make_game()
    act(facing_bet, 0, ActionType.RAISE, 20)
    rejected = facing_bet.apply(Action(1, ActionType.BET, amount=40, source="test"))
    assert not rejected.ok
    accepted = facing_bet.apply(Action(1, ActionType.RAISE, amount=40, source="test"))
    assert accepted.ok


def test_constraints_do_not_advertise_raise_below_current_bet() -> None:
    game = make_game()
    act(game, 0, ActionType.RAISE, 30)
    game.players[1].chips = 15
    constraints = game.action_constraints(1)
    assert constraints["can_raise"] is False
    assert constraints["min_raise_to"] >= constraints["current_bet"]
    assert constraints["max_raise_to"] >= constraints["current_bet"]


def test_invalid_constructor_values_are_rejected() -> None:
    with pytest.raises(ValueError):
        HeadsUpGame(starting_chips=0.5)
    with pytest.raises(ValueError):
        HeadsUpGame(starting_chips=100, small_blind=10, big_blind=5)
    with pytest.raises(ValueError):
        HeadsUpGame(starting_chips=100, rebuy_amount=0)
    with pytest.raises(ValueError):
        HeadsUpGame(starting_chips=100, max_hands=-1)
    with pytest.raises(ValueError):
        HeadsUpGame(blind_increase_every_hands=-1)
    with pytest.raises(ValueError):
        HeadsUpGame(blind_increase_factor=0.5)
    with pytest.raises(ValueError):
        HeadsUpGame(blind_increase_factor=11)


def finish_hand_by_fold(game: HeadsUpGame) -> None:
    actor = game.actor
    assert actor is not None
    act(game, actor, ActionType.FOLD)


def test_blind_increase_is_off_by_default() -> None:
    game = make_game()
    for _ in range(3):
        finish_hand_by_fold(game)
        game.start_hand()
    assert game.small_blind == 5
    assert game.big_blind == 10
    assert game.blind_level == 0


def test_blind_increase_after_configured_hands() -> None:
    game = HeadsUpGame(
        starting_chips=1000,
        small_blind=5,
        big_blind=10,
        auto_rebuy=False,
        seed=42,
        blind_increase_every_hands=2,
        blind_increase_factor=2.0,
    )
    game.configure_players({"name": "A", "kind": "human"}, {"name": "B", "kind": "bot"})
    game.start_hand()
    assert (game.small_blind, game.big_blind) == (5, 10)

    finish_hand_by_fold(game)
    game.start_hand()
    assert (game.small_blind, game.big_blind) == (5, 10)
    assert game.blind_level == 0

    finish_hand_by_fold(game)
    game.start_hand()
    assert (game.small_blind, game.big_blind) == (10, 20)
    assert game.blind_level == 1
    assert game.last_full_raise == 20
    assert game.min_raise_to == 40
    assert sum(player.chips for player in game.players) + sum(
        player.total_bet for player in game.players
    ) == game.starting_chips * 2


def test_blind_increase_preserves_blind_ratio() -> None:
    game = HeadsUpGame(
        starting_chips=1000,
        small_blind=5,
        big_blind=10,
        auto_rebuy=False,
        seed=42,
        blind_increase_every_hands=1,
        blind_increase_factor=1.5,
    )
    game.configure_players({"name": "A", "kind": "human"}, {"name": "B", "kind": "bot"})
    game.start_hand()
    finish_hand_by_fold(game)
    game.start_hand()
    assert (game.small_blind, game.big_blind) == (8, 16)
    assert game.blind_level == 1


def test_action_history_records_added_chips() -> None:
    game = make_game()
    act(game, 0, ActionType.RAISE, 40)
    act(game, 1, ActionType.CALL)
    raise_entry = game.actions[0]
    call_entry = game.actions[1]
    assert raise_entry["amount"] == 40
    assert raise_entry["added"] == 35
    assert call_entry["amount"] == 30
    assert call_entry["added"] == 30
