"""Training environment for the project's strict heads-up rules.

This module deliberately wraps ``HeadsUpGame`` instead of reimplementing poker
rules.  The environment only converts engine state into a 73-float observation,
maps ten abstract actions back to concrete engine actions, and assigns terminal
rewards.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import List, Optional, Sequence, Tuple

import numpy as np

from src.agents.base import AgentView, BaseAgent
from src.engine.actions import Action
from src.engine.heads_up import HeadsUpGame, Street

from .actions import (
    ACTION_LABELS,
    NUM_ACTIONS,
    action_from_view,
    legal_action_mask,
)
from .obs import OBS_DIM, encode_observation

MAX_ACTIONS_PER_HAND = 200


class ProjectHeadsUpEnv:
    """A two-seat, fixed-stack training environment.

    Each ``reset`` creates a fresh hand.  Stacks are reset to ``starting_chips``
    so the policy always sees the same 100bb (or configured) depth.  The button
    alternates automatically unless the caller passes an explicit ``button``.
    """

    def __init__(
        self,
        starting_chips: int = 1000,
        small_blind: int = 5,
        big_blind: int = 10,
        seed: int = 2026,
        max_actions_per_hand: int = MAX_ACTIONS_PER_HAND,
    ) -> None:
        if starting_chips <= 0:
            raise ValueError("starting_chips 必须为正")
        if small_blind <= 0 or big_blind < small_blind:
            raise ValueError("盲注必须满足 0 < small_blind <= big_blind")
        self.starting_chips = int(starting_chips)
        self.small_blind = int(small_blind)
        self.big_blind = int(big_blind)
        self.seed = int(seed)
        self.max_actions_per_hand = int(max_actions_per_hand)
        self.game: Optional[HeadsUpGame] = None
        self.hand_index = 0
        self.last_seed: Optional[int] = None
        self.last_button: Optional[int] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None, button: Optional[int] = None) -> dict:
        """Start a fresh hand and return the first actor's state."""
        hand_index = self.hand_index
        resolved_seed = (
            self.seed + 1_000_003 * (hand_index + 1) if seed is None else int(seed)
        )
        resolved_button = hand_index % 2 if button is None else int(button)
        if resolved_button not in (0, 1):
            raise ValueError("button 必须是 0 或 1")

        game = HeadsUpGame(
            starting_chips=self.starting_chips,
            small_blind=self.small_blind,
            big_blind=self.big_blind,
            auto_rebuy=False,
            seed=resolved_seed,
        )
        # start_hand increments button_index, so pre-decrement to land on the
        # requested seat.
        game.button_index = resolved_button - 1
        game.configure_players(
            {"name": "Seat 0", "kind": "bot"},
            {"name": "Seat 1", "kind": "bot"},
        )
        game.start_hand()

        self.game = game
        self.last_seed = resolved_seed
        self.last_button = resolved_button
        self.hand_index += 1
        return self.state()

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def build_view(self, seat: int) -> AgentView:
        """Build the same fair-information view used by the live server."""
        if self.game is None:
            raise RuntimeError("环境尚未 reset")
        game = self.game
        player = game.players[seat]
        opponent = game.players[1 - seat]
        constraints = game.action_constraints(seat)
        return AgentView(
            seat=seat,
            name=player.name,
            hand_id=game.hand_id,
            street=game.street.value,
            hole_cards=[card.code for card in player.hole_cards],
            community=[card.code for card in game.community],
            pot=game.pot,
            my_chips=player.chips,
            opponent_chips=opponent.chips,
            my_street_bet=player.street_bet,
            opponent_street_bet=opponent.street_bet,
            my_total_bet=player.total_bet,
            opponent_total_bet=opponent.total_bet,
            current_bet=game.current_bet,
            to_call=max(0, int(constraints.get("to_call", 0))),
            call_amount=int(constraints.get("call_amount", 0)),
            min_raise_to=int(constraints.get("min_raise_to", game.big_blind)),
            max_raise_to=int(constraints.get("max_raise_to", player.chips)),
            legal_actions=[action.value for action in game.legal_actions(seat)],
            action_history=list(game.actions[-12:]),
            opponent_last_action=opponent.last_action,
            button=game.button_index,
            small_blind=game.small_blind,
            big_blind=game.big_blind,
            hand_number=game.hand_id,
        )

    def state(self, seat: Optional[int] = None) -> dict:
        """Return an RLCard-compatible state dict for ``seat``."""
        if self.game is None:
            raise RuntimeError("环境尚未 reset")
        if seat is None:
            seat = self.game.actor
        if seat is None or self.game.street == Street.FINISHED:
            return self.terminal_state()

        view = self.build_view(int(seat))
        mask = legal_action_mask(view)
        legal_actions = OrderedDict(
            (int(index), None) for index in np.flatnonzero(mask).tolist()
        )
        return {
            "obs": encode_observation(view, self.big_blind),
            "legal_actions": legal_actions,
            "raw_legal_actions": list(legal_actions.keys()),
            "raw_obs": {
                "seat": int(seat),
                "street": view.street,
                "pot": int(view.pot),
                "my_chips": int(view.my_chips),
                "opponent_chips": int(view.opponent_chips),
                "to_call": int(view.to_call),
                "min_raise_to": int(view.min_raise_to),
                "max_raise_to": int(view.max_raise_to),
                "is_button": bool(view.is_button),
            },
        }

    def legal_mask(self, seat: Optional[int] = None) -> np.ndarray:
        if self.game is None:
            raise RuntimeError("环境尚未 reset")
        if seat is None:
            seat = self.game.actor
        if seat is None or self.game.street == Street.FINISHED:
            return np.zeros(NUM_ACTIONS, dtype=bool)
        return legal_action_mask(self.build_view(int(seat)))

    @staticmethod
    def terminal_state() -> dict:
        return {
            "obs": np.zeros(OBS_DIM, dtype=np.float32),
            "legal_actions": OrderedDict(),
            "raw_legal_actions": [],
            "raw_obs": {"terminal": True},
        }

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, slot: int) -> Tuple[dict, bool, dict]:
        """Apply one abstract action slot for the current actor."""
        if self.game is None:
            raise RuntimeError("环境尚未 reset")
        actor = self.game.actor
        if actor is None or self.game.street == Street.FINISHED:
            raise RuntimeError("当前没有可行动的玩家")

        view = self.build_view(actor)
        action = action_from_view(view, int(slot))
        result = self.game.apply(action)
        if not result.ok:
            raise RuntimeError(f"训练动作被引擎拒绝: {action.to_dict()} -> {result.error}")

        done = self.game.street == Street.FINISHED
        next_state = self.terminal_state() if done else self.state()
        info = {
            "actor": int(actor),
            "action": action.to_dict(),
            "slot": int(slot),
            "slot_label": ACTION_LABELS[int(slot)],
            "done": bool(done),
            "pot": int(self.game.pot),
        }
        return next_state, bool(done), info

    # ------------------------------------------------------------------
    # Rewards / hand execution
    # ------------------------------------------------------------------

    def payoffs(self) -> List[float]:
        """Return terminal rewards for both seats, scaled to roughly +/-5."""
        if self.game is None or self.game.result is None:
            raise RuntimeError("手牌尚未结束")
        players = {int(item["seat"]): int(item["net"]) for item in self.game.result["players"]}
        denominator = max(1.0, self.starting_chips / 5.0)
        return [players.get(0, 0) / denominator, players.get(1, 0) / denominator]

    def raw_payoffs(self) -> List[int]:
        if self.game is None or self.game.result is None:
            raise RuntimeError("手牌尚未结束")
        players = {int(item["seat"]): int(item["net"]) for item in self.game.result["players"]}
        return [players.get(0, 0), players.get(1, 0)]

    def play_hand(
        self,
        agents: Sequence[object],
        training: bool = False,
    ) -> Tuple[List[float], List[List[tuple]]]:
        """Play the current hand.

        ``agents`` must be a two-element sequence.  Training mode calls
        ``agent.step(state)`` and records transitions; evaluation mode calls
        ``agent.eval_step(state)[0]`` and only returns payoffs.
        """
        if self.game is None:
            raise RuntimeError("环境尚未 reset")
        transitions: List[List[tuple]] = [[], []]
        # A transition must link the same player's consecutive decisions. The
        # engine hands control to the opponent after every action, so the next
        # state returned by ``step`` belongs to the other seat. Keep the acting
        # player's state pending until they act again, or until the hand ends.
        pending: List[Optional[tuple]] = [None, None]
        actions = 0
        while self.game.street != Street.FINISHED:
            actor = self.game.actor
            if actor is None:
                raise RuntimeError("手牌未结束但没有行动者")
            actions += 1
            if actions > self.max_actions_per_hand:
                raise RuntimeError(f"单手动作数超过 {self.max_actions_per_hand}")
            agent = agents[actor]
            if not training and isinstance(agent, BaseAgent):
                # Live-style agents (ProjectBot, RLCardBot) consume AgentView
                # and return an Action directly.
                view = self.build_view(actor)
                action = agent.decide(view, lambda event: None)
                result = self.game.apply(action)
                if not result.ok:
                    raise RuntimeError(f"评估动作被引擎拒绝: {action.to_dict()} -> {result.error}")
                done = self.game.street == Street.FINISHED
            else:
                state = self.state(actor)
                if training:
                    slot = int(agent.step(state))
                else:
                    slot = int(agent.eval_step(state)[0])
                _next_state, done, _info = self.step(slot)
                if training:
                    previous = pending[actor]
                    if previous is not None:
                        transitions[actor].append(
                            (previous[0], previous[1], state, False)
                        )
                    pending[actor] = (state, slot)
            if done:
                break
        if training:
            terminal = self.terminal_state()
            for seat in (0, 1):
                previous = pending[seat]
                if previous is not None:
                    transitions[seat].append(
                        (previous[0], previous[1], terminal, True)
                    )
        return self.payoffs(), transitions

    def feed_transitions(
        self,
        agents: Sequence[object],
        transitions: Sequence[Sequence[tuple]],
        payoffs: Sequence[float],
        seats: Sequence[int] = (0, 1),
    ) -> None:
        """Feed terminal-reward transitions to RLCard-style agents.

        ``play_hand`` links each transition to the same player's next decision
        (or the terminal state), so the bootstrap value always belongs to the
        learning player rather than the opponent.
        """
        for seat in seats:
            seat_transitions = transitions[seat]
            if not seat_transitions:
                continue
            agent = agents[seat]
            last_index = len(seat_transitions) - 1
            for index, (state, action, next_state, _done) in enumerate(seat_transitions):
                is_last = index == last_index
                reward = float(payoffs[seat]) if is_last else 0.0
                agent.feed((state, action, reward, next_state, is_last))


__all__ = ["ProjectHeadsUpEnv"]
