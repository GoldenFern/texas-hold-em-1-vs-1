"""Deterministic and random scripted opponents for offline evaluations."""

from __future__ import annotations

import random
from typing import Optional, Sequence

from src.agents.base import AgentView, BaseAgent, EventSink
from src.engine.actions import Action, ActionType

POLICIES = ("always_all_in", "always_raise", "call_check", "random")

DEFAULT_NAMES = {
    "always_all_in": "All-in Bot",
    "always_raise": "Min-Raise Bot",
    "call_check": "Call/Check Bot",
    "random": "Random Bot",
}


class ScriptedBot(BaseAgent):
    """A simple legal-action opponent used to probe an LLM's adaptation.

    Policies:
        always_all_in: shove whenever legal, otherwise call or check.
        always_raise: make the minimum raise or bet whenever legal, never fold.
        call_check: never raise or fold; call when facing a bet, otherwise check.
        random: choose uniformly from legal actions; bet and raise sizes are
            drawn uniformly from the allowed raise-to range.
    """

    kind = "scripted"

    def __init__(
        self,
        name: Optional[str] = None,
        policy: str = "call_check",
        seed: Optional[int] = None,
    ) -> None:
        normalized = str(policy or "").strip().lower()
        if normalized not in POLICIES:
            raise ValueError(f"unknown scripted policy: {policy!r}")
        self.policy = normalized
        self.name = name or DEFAULT_NAMES[normalized]
        self.seed = seed
        self._rng = random.Random(seed)

    def decide(self, view: AgentView, emit: Optional[EventSink] = None) -> Action:
        """Choose one legal action for the given fair-information view."""
        legal = [str(item) for item in view.legal_actions]
        if not legal:
            raise RuntimeError("scripted bot received no legal actions")
        action = self._choose(view, legal)
        if emit is not None:
            emit(
                {
                    "type": "bot_decision",
                    "seat": int(view.seat),
                    "hand_id": int(view.hand_id),
                    "street": str(view.street),
                    "agent_type": f"scripted_{self.policy}",
                    "action": action.to_dict(),
                }
            )
        return action

    def describe(self) -> dict:
        return {"kind": self.kind, "policy": self.policy, "name": self.name}

    def _choose(self, view: AgentView, legal: Sequence[str]) -> Action:
        seat = int(view.seat)
        if self.policy == "always_all_in":
            if "all_in" in legal:
                return Action(seat, ActionType.ALL_IN, source="scripted")
            return self._call_check(seat, legal)

        if self.policy == "always_raise":
            amount = self._raise_to(view)
            if "raise" in legal:
                return Action(seat, ActionType.RAISE, amount=amount, source="scripted")
            if "bet" in legal:
                return Action(seat, ActionType.BET, amount=amount, source="scripted")
            if "call" in legal:
                return Action(seat, ActionType.CALL, source="scripted")
            if "check" in legal:
                return Action(seat, ActionType.CHECK, source="scripted")
            if "all_in" in legal:
                return Action(seat, ActionType.ALL_IN, source="scripted")
            return Action(seat, ActionType.FOLD, source="scripted")

        if self.policy == "call_check":
            return self._call_check(seat, legal)

        action_type = self._rng.choice(list(legal))
        if action_type in ("bet", "raise"):
            return Action(
                seat,
                ActionType(action_type),
                amount=self._random_raise_to(view),
                source="scripted",
            )
        return Action(seat, ActionType(action_type), source="scripted")

    @staticmethod
    def _call_check(seat: int, legal: Sequence[str]) -> Action:
        if "call" in legal:
            return Action(seat, ActionType.CALL, source="scripted")
        if "check" in legal:
            return Action(seat, ActionType.CHECK, source="scripted")
        if "all_in" in legal:
            return Action(seat, ActionType.ALL_IN, source="scripted")
        return Action(seat, ActionType.FOLD, source="scripted")

    @staticmethod
    def _raise_to(view: AgentView) -> int:
        minimum = max(1, int(view.min_raise_to))
        maximum = max(minimum, int(view.max_raise_to))
        return min(maximum, minimum)

    def _random_raise_to(self, view: AgentView) -> int:
        minimum = max(1, int(view.min_raise_to))
        maximum = max(minimum, int(view.max_raise_to))
        return self._rng.randint(minimum, maximum)


__all__ = ["ScriptedBot", "POLICIES", "DEFAULT_NAMES"]
