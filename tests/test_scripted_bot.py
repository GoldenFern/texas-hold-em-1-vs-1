"""Tests for the scripted evaluation opponents."""

from __future__ import annotations

from dataclasses import replace

from src.agents.base import AgentView
from src.agents.scripted_bot import ScriptedBot
from src.engine.actions import ActionType


def make_view(**overrides) -> AgentView:
    view = AgentView(
        seat=0,
        name="Hero",
        hand_id=1,
        street="flop",
        hole_cards=["Ah", "Kh"],
        community=["Qh", "Jh", "2c"],
        pot=100,
        my_chips=900,
        opponent_chips=900,
        my_street_bet=0,
        opponent_street_bet=0,
        my_total_bet=50,
        opponent_total_bet=50,
        current_bet=0,
        to_call=0,
        call_amount=0,
        min_raise_to=10,
        max_raise_to=900,
        legal_actions=["fold", "check", "bet", "all_in"],
        action_history=[],
        opponent_last_action=None,
        button=0,
        small_blind=5,
        big_blind=10,
        hand_number=1,
    )
    return replace(view, **overrides)


def test_all_in_bot_always_shoves_when_legal() -> None:
    bot = ScriptedBot(policy="always_all_in", seed=1)
    assert bot.decide(make_view()).type is ActionType.ALL_IN
    facing_bet = make_view(
        current_bet=100,
        to_call=100,
        call_amount=100,
        legal_actions=["fold", "call", "raise", "all_in"],
        min_raise_to=200,
    )
    assert bot.decide(facing_bet).type is ActionType.ALL_IN


def test_all_in_bot_calls_when_raise_is_closed() -> None:
    bot = ScriptedBot(policy="always_all_in", seed=1)
    view = make_view(
        current_bet=500,
        to_call=500,
        call_amount=500,
        legal_actions=["fold", "call"],
        my_chips=500,
    )
    assert bot.decide(view).type is ActionType.CALL


def test_raise_bot_uses_minimum_raise_and_never_folds() -> None:
    bot = ScriptedBot(policy="always_raise", seed=1)
    view = make_view(
        current_bet=100,
        to_call=100,
        call_amount=100,
        legal_actions=["fold", "call", "raise", "all_in"],
        min_raise_to=200,
        max_raise_to=900,
    )
    action = bot.decide(view)
    assert action.type is ActionType.RAISE
    assert action.amount == 200
    assert action.type is not ActionType.FOLD


def test_raise_bot_bets_minimum_when_checking_is_available() -> None:
    bot = ScriptedBot(policy="always_raise", seed=1)
    action = bot.decide(make_view())
    assert action.type is ActionType.BET
    assert action.amount == 10


def test_call_check_bot_never_raises_or_folds() -> None:
    bot = ScriptedBot(policy="call_check", seed=1)
    assert bot.decide(make_view()).type is ActionType.CHECK
    facing_bet = make_view(
        current_bet=50,
        to_call=50,
        call_amount=50,
        legal_actions=["fold", "call", "raise", "all_in"],
        min_raise_to=100,
    )
    assert bot.decide(facing_bet).type is ActionType.CALL


def test_random_bot_only_returns_legal_actions_and_is_reproducible() -> None:
    view = make_view(
        current_bet=50,
        to_call=50,
        call_amount=50,
        legal_actions=["fold", "call", "raise", "all_in"],
        min_raise_to=100,
        max_raise_to=300,
    )
    first = ScriptedBot(policy="random", seed=42)
    second = ScriptedBot(policy="random", seed=42)
    for _ in range(20):
        left = first.decide(view)
        right = second.decide(view)
        assert left.type is right.type
        assert left.amount == right.amount
        assert left.type.value in view.legal_actions
        if left.type in (ActionType.BET, ActionType.RAISE):
            assert 100 <= left.amount <= 300
