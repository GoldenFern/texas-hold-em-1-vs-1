from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from rlcard.agents import DQNAgent, RandomAgent

from src.agents.project_bot import ProjectBot, save_project_checkpoint
from src.training import NUM_ACTIONS, OBS_DIM, ProjectHeadsUpEnv
from src.training.actions import (
    ACTION_ALL_IN,
    ACTION_CALL,
    ACTION_CHECK,
    ACTION_FOLD,
    ACTION_MIN_RAISE,
    ACTION_HALF_POT,
    ACTION_THREE_QUARTER_POT,
    action_from_view,
    legal_action_mask,
    raise_to_for_fraction,
)
from src.training.obs import card_index, encode_observation


def test_card_index_matches_rlcard_order() -> None:
    assert card_index("As") == 0
    assert card_index("Ah") == 13
    assert card_index("Ad") == 26
    assert card_index("Ac") == 39
    assert card_index("Kd") == 38


def test_observation_shape_and_card_bits() -> None:
    env = ProjectHeadsUpEnv(seed=3)
    state = env.reset()
    obs = state["obs"]
    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32
    view = env.build_view(env.game.actor)  # type: ignore[union-attr]
    known = list(view.community) + list(view.hole_cards)
    assert int(obs[:52].sum()) == len(known)
    for code in known:
        assert obs[card_index(code)] == 1.0
    assert obs[64] in (0.0, 1.0)


def test_preflop_mask_and_min_raise() -> None:
    env = ProjectHeadsUpEnv(seed=4)
    env.reset(button=0)
    view = env.build_view(0)
    mask = legal_action_mask(view)
    assert mask[ACTION_FOLD]
    assert mask[ACTION_CALL]
    assert mask[ACTION_MIN_RAISE]
    assert mask[ACTION_ALL_IN]
    assert not mask[ACTION_CHECK]

    action = action_from_view(view, ACTION_MIN_RAISE)
    assert action.amount == view.min_raise_to
    assert action.type.value == "raise"

    # At 5/10 preflop half-pot deduplicates with the min-raise.  Use 3/4 pot
    # to exercise the standard pot-fraction formula.
    assert not mask[ACTION_HALF_POT]
    fraction_action = action_from_view(view, ACTION_THREE_QUARTER_POT)
    assert fraction_action.amount == raise_to_for_fraction(view, 0.75)
    assert fraction_action.amount >= view.min_raise_to
    assert fraction_action.amount <= view.max_raise_to


def test_big_blind_option_uses_bet_not_raise() -> None:
    env = ProjectHeadsUpEnv(seed=5)
    env.reset(button=0)
    env.step(ACTION_CALL)  # SB limps, BB has the option.
    view = env.build_view(1)
    assert "bet" in view.legal_actions
    assert "raise" not in view.legal_actions
    action = action_from_view(view, ACTION_MIN_RAISE)
    assert action.type.value == "bet"
    assert action.amount >= view.min_raise_to
    result = env.game.apply(action)  # type: ignore[union-attr]
    assert result.ok, result.error


def test_postflop_check_slot() -> None:
    env = ProjectHeadsUpEnv(seed=5)
    env.reset(button=0)
    env.play_hand([RandomAgent(NUM_ACTIONS), RandomAgent(NUM_ACTIONS)], training=False)
    # The hand may end preflop or later; construct a fresh flop state instead.
    env.reset(button=0)
    # SB call, BB check -> flop, BB acts first.
    env.step(ACTION_CALL)
    env.step(ACTION_CHECK)
    view = env.build_view(env.game.actor)  # type: ignore[union-attr]
    mask = legal_action_mask(view)
    assert mask[ACTION_CHECK]
    assert not mask[ACTION_CALL]
    assert action_from_view(view, ACTION_CHECK).type.value == "check"


def test_chip_conservation_and_button_rotation() -> None:
    env = ProjectHeadsUpEnv(starting_chips=1000, small_blind=5, big_blind=10, seed=6)
    agents = [RandomAgent(NUM_ACTIONS), RandomAgent(NUM_ACTIONS)]
    buttons = []
    for _ in range(200):
        env.reset()
        buttons.append(env.last_button)
        env.play_hand(agents, training=False)
        assert sum(player.chips for player in env.game.players) == 2000  # type: ignore[union-attr]
        net0, net1 = env.raw_payoffs()
        assert net0 + net1 == 0
        payoffs = env.payoffs()
        assert payoffs[0] == pytest.approx(net0 / 200.0)
        assert payoffs[1] == pytest.approx(net1 / 200.0)
    assert buttons[:6] == [0, 1, 0, 1, 0, 1]


def test_fold_refund_and_reward_scale() -> None:
    env = ProjectHeadsUpEnv(seed=7)
    env.reset(button=0)
    env.step(ACTION_FOLD)
    assert env.game.result is not None  # type: ignore[union-attr]
    assert env.game.result["reason"] == "fold"  # type: ignore[union-attr]
    net0, net1 = env.raw_payoffs()
    assert net0 == -5
    assert net1 == 5
    assert env.payoffs() == pytest.approx([-0.025, 0.025])


def test_training_transitions_link_same_player_decisions() -> None:
    env = ProjectHeadsUpEnv(seed=11)
    env.reset(button=0)
    agents = [RandomAgent(NUM_ACTIONS), RandomAgent(NUM_ACTIONS)]
    _payoffs, transitions = env.play_hand(agents, training=True)
    assert any(transitions[seat] for seat in (0, 1))
    for seat in (0, 1):
        for state, _slot, next_state, done in transitions[seat]:
            assert state["raw_obs"]["seat"] == seat
            if done:
                assert next_state["raw_obs"].get("terminal") is True
            else:
                # The bootstrap state must belong to the same player, not to
                # the opponent who acted in between.
                assert next_state["raw_obs"]["seat"] == seat


def test_observation_version_keeps_legacy_size_feature() -> None:
    env = ProjectHeadsUpEnv(seed=9)
    env.reset(button=0)
    view = env.build_view(0)
    view.action_history = [
        {
            "seat": 0,
            "type": "raise",
            "amount": 40,
            "added": 30,
            "pot_after": 60,
        }
    ]
    legacy = encode_observation(view, 10, 1)
    current = encode_observation(view, 10, 2)
    assert legacy[70] == pytest.approx(40 / 60)
    assert current[70] == pytest.approx(30 / 60)


def test_project_bot_checkpoint_roundtrip(tmp_path) -> None:
    env = ProjectHeadsUpEnv(seed=8)
    env.reset(button=0)
    view = env.build_view(0)

    agent = DQNAgent(
        num_actions=NUM_ACTIONS,
        state_shape=[OBS_DIM],
        mlp_layers=[128, 128],
        device="cpu",
    )
    path = save_project_checkpoint(
        tmp_path / "dqn.pth",
        algorithm="dqn",
        state_dict=agent.q_estimator.qnet.state_dict(),
        mlp_layers=[128, 128],
        metadata={"test": True},
    )
    bot = ProjectBot("test", path)
    action = bot.decide(view, lambda event: None)
    assert action.seat == 0
    assert action.type.value in {"fold", "check", "call", "bet", "raise", "all_in"}
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["format"] == "project-bot-v1"
    assert payload["obs_dim"] == OBS_DIM
    assert payload["obs_version"] == 2
    assert payload["num_actions"] == NUM_ACTIONS
    assert bot.sample is False


def test_nfsp_checkpoint_defaults_to_sampling(tmp_path) -> None:
    from rlcard.agents import NFSPAgent

    agent = NFSPAgent(
        num_actions=NUM_ACTIONS,
        state_shape=[OBS_DIM],
        hidden_layers_sizes=[32],
        q_mlp_layers=[32],
        device="cpu",
        evaluate_with="average_policy",
    )
    path = save_project_checkpoint(
        tmp_path / "nfsp.pth",
        algorithm="nfsp",
        state_dict=agent.policy_network.state_dict(),
        mlp_layers=list(agent.policy_network.mlp_layers),
    )
    bot = ProjectBot("nfsp", path)
    assert bot.algorithm == "nfsp"
    assert bot.sample is True
    with pytest.raises(ValueError):
        ProjectBot("wrong", path, algorithm="dqn")
