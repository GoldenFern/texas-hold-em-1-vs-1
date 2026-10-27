"""Agent protocol and the information a seat is allowed to know."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

from src.engine.actions import Action

EventSink = Callable[[dict], None]


@dataclass
class AgentView:
    """Fair-information view passed to an agent.

    Deliberately does not contain the opponent's hole cards, the remaining
    deck, or any other god-view data.
    """

    seat: int
    name: str
    hand_id: int
    street: str
    hole_cards: List[str]
    community: List[str]
    pot: int
    my_chips: int
    opponent_chips: int
    my_street_bet: int
    opponent_street_bet: int
    my_total_bet: int
    opponent_total_bet: int
    current_bet: int
    to_call: int
    call_amount: int
    min_raise_to: int
    max_raise_to: int
    legal_actions: List[str]
    action_history: List[dict]
    opponent_last_action: Optional[dict]
    button: int
    small_blind: int
    big_blind: int
    hand_number: int
    match_note: str = ""
    history_summary: List[dict] = field(default_factory=list)
    table_talk: List[dict] = field(default_factory=list)
    own_table_talk: List[dict] = field(default_factory=list)

    @property
    def is_button(self) -> bool:
        return self.seat == self.button

    def to_prompt_dict(self) -> dict:
        return {
            "seat": self.seat,
            "name": self.name,
            "hand_id": self.hand_id,
            "street": self.street,
            "position": "button" if self.is_button else "big_blind",
            "hole_cards": self.hole_cards,
            "community_cards": self.community,
            "pot": self.pot,
            "your_chips": self.my_chips,
            "opponent_chips": self.opponent_chips,
            "your_street_bet": self.my_street_bet,
            "opponent_street_bet": self.opponent_street_bet,
            "your_total_contribution": self.my_total_bet,
            "opponent_total_contribution": self.opponent_total_bet,
            "current_bet": self.current_bet,
            "to_call": self.to_call,
            "call_amount": self.call_amount,
            "min_raise_to": self.min_raise_to,
            "max_raise_to": self.max_raise_to,
            "legal_actions": self.legal_actions,
            "actions_this_hand": self.action_history,
            "opponent_last_action": self.opponent_last_action,
            "button_seat": self.button,
            "blinds": {"small": self.small_blind, "big": self.big_blind},
            "hand_number": self.hand_number,
            "match_note": self.match_note,
            "recent_hands": self.history_summary,
        }


class BaseAgent:
    """Base class for all non-human seats."""

    kind = "agent"

    def decide(self, view: AgentView, emit: EventSink) -> Action:
        raise NotImplementedError

    def describe(self) -> dict:
        return {"kind": self.kind}

    def reset(self) -> None:
        """Reset per-match state (called when a new match starts)."""


__all__ = ["AgentView", "BaseAgent", "EventSink"]
