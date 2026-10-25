"""Heads-up No-Limit Texas Hold'em state machine.

Rules implemented for exactly two players:

* The button posts the small blind and acts first preflop.
* The non-button posts the big blind and acts first on every later street.
* No side pots are possible. Uncalled chips are always returned.
* A short all-in raise does not reopen the betting for a player who has
  already acted on that street.
* Burn cards are tracked so the UI can show the real remaining deck.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Sequence

from src.i18n import normalize_locale, tr

from .actions import Action, ActionType
from .cards import Card, Deck
from .evaluator import CATEGORY_BY_KEY, assess_hand

MAX_SCORE = 7463


class Street(str, Enum):
    WAITING = "waiting"
    PREFLOP = "preflop"
    FLOP = "flop"
    TURN = "turn"
    RIVER = "river"
    SHOWDOWN = "showdown"
    FINISHED = "finished"


class PlayerStatus(str, Enum):
    ACTIVE = "active"
    FOLDED = "folded"
    ALL_IN = "all_in"
    OUT = "out"


@dataclass
class HeadsUpPlayer:
    seat: int
    name: str
    kind: str = "human"
    chips: int = 1000
    hole_cards: List[Card] = field(default_factory=list)
    status: PlayerStatus = PlayerStatus.ACTIVE
    street_bet: int = 0
    total_bet: int = 0
    is_button: bool = False
    is_small_blind: bool = False
    is_big_blind: bool = False
    acted_this_street: bool = False
    raise_allowed: bool = True
    rebuy_count: int = 0
    last_action: Optional[dict] = None

    @property
    def in_hand(self) -> bool:
        return self.status in (PlayerStatus.ACTIVE, PlayerStatus.ALL_IN)

    @property
    def can_act(self) -> bool:
        return self.status == PlayerStatus.ACTIVE and self.chips > 0

    def reset_for_hand(self) -> None:
        self.hole_cards = []
        self.status = PlayerStatus.ACTIVE
        self.street_bet = 0
        self.total_bet = 0
        self.is_button = False
        self.is_small_blind = False
        self.is_big_blind = False
        self.acted_this_street = False
        self.raise_allowed = True
        self.last_action = None

    def post_blind(self, amount: int) -> int:
        actual = min(amount, self.chips)
        self.chips -= actual
        self.street_bet += actual
        self.total_bet += actual
        if self.chips == 0:
            self.status = PlayerStatus.ALL_IN
        return actual


@dataclass
class ActionResult:
    ok: bool
    action: Optional[Action] = None
    error: str = ""
    hand_finished: bool = False
    street_changed: bool = False
    runout: bool = False


class HeadsUpGame:
    """A two-player No-Limit Hold'em table."""

    def __init__(
        self,
        starting_chips: int = 1000,
        small_blind: int = 5,
        big_blind: int = 10,
        auto_rebuy: bool = False,
        rebuy_amount: Optional[int] = None,
        seed: Optional[int] = None,
        max_hands: int = 0,
        blind_increase_every_hands: int = 0,
        blind_increase_factor: float = 2.0,
        locale: str = "zh-CN",
    ) -> None:
        self.locale = normalize_locale(locale, "zh-CN") or "zh-CN"

        def positive_int(value: object, label: str) -> int:
            if isinstance(value, bool):
                raise ValueError(tr(self.locale, "engine.rule.positive_int", label=label))
            if isinstance(value, float) and not value.is_integer():
                raise ValueError(tr(self.locale, "engine.rule.positive_int", label=label))
            try:
                number = int(value)  # type: ignore[arg-type]
            except (TypeError, ValueError) as exc:
                raise ValueError(tr(self.locale, "engine.rule.positive_int", label=label)) from exc
            if number <= 0:
                raise ValueError(tr(self.locale, "engine.rule.positive", label=label))
            return number

        starting_chips = positive_int(starting_chips, "starting_chips")
        small_blind = positive_int(small_blind, "small_blind")
        big_blind = positive_int(big_blind, "big_blind")
        if big_blind < small_blind:
            raise ValueError(tr(self.locale, "engine.rule.blinds"))
        if starting_chips < big_blind:
            raise ValueError(tr(self.locale, "engine.rule.starting_chips"))
        if isinstance(max_hands, bool):
            raise ValueError(tr(self.locale, "engine.rule.max_hands"))
        if isinstance(max_hands, float) and not max_hands.is_integer():
            raise ValueError(tr(self.locale, "engine.rule.max_hands"))
        try:
            max_hands = int(max_hands or 0)
        except (TypeError, ValueError) as exc:
            raise ValueError(tr(self.locale, "engine.rule.max_hands")) from exc
        if max_hands < 0:
            raise ValueError(tr(self.locale, "engine.rule.max_hands"))
        if isinstance(blind_increase_every_hands, bool):
            raise ValueError(tr(self.locale, "engine.rule.blind_increase"))
        if (
            isinstance(blind_increase_every_hands, float)
            and not blind_increase_every_hands.is_integer()
        ):
            raise ValueError(tr(self.locale, "engine.rule.blind_increase"))
        try:
            blind_increase_every_hands = int(blind_increase_every_hands or 0)
        except (TypeError, ValueError) as exc:
            raise ValueError(tr(self.locale, "engine.rule.blind_increase")) from exc
        if blind_increase_every_hands < 0:
            raise ValueError(tr(self.locale, "engine.rule.blind_increase"))
        if isinstance(blind_increase_factor, bool):
            raise ValueError(tr(self.locale, "engine.rule.blind_factor"))
        try:
            blind_increase_factor = float(blind_increase_factor)
        except (TypeError, ValueError) as exc:
            raise ValueError(tr(self.locale, "engine.rule.blind_factor")) from exc
        if not 1.0 <= blind_increase_factor <= 10.0:
            raise ValueError(tr(self.locale, "engine.rule.blind_factor"))
        rebuy_amount = positive_int(
            starting_chips if rebuy_amount is None else rebuy_amount,
            "rebuy_amount",
        )

        self.starting_chips = int(starting_chips)
        self.small_blind = int(small_blind)
        self.big_blind = int(big_blind)
        self.starting_small_blind = int(small_blind)
        self.starting_big_blind = int(big_blind)
        self.auto_rebuy = bool(auto_rebuy)
        self.rebuy_amount = int(rebuy_amount)
        self.max_hands = int(max_hands or 0)
        self.blind_increase_every_hands = int(blind_increase_every_hands)
        self.blind_increase_factor = float(blind_increase_factor)
        self.blind_level = 0
        self.seed = seed

        self.players: List[HeadsUpPlayer] = [
            HeadsUpPlayer(seat=0, name="Player 1", kind="human", chips=self.starting_chips),
            HeadsUpPlayer(seat=1, name="Player 2", kind="human", chips=self.starting_chips),
        ]
        self.deck = Deck(seed)
        self.community: List[Card] = []
        self.street = Street.WAITING
        self.hand_id = 0
        self.hands_played = 0
        self.button_index = -1
        self.actor: Optional[int] = None
        self.current_bet = 0
        self.last_full_raise = self.big_blind
        self.min_raise_to = self.big_blind
        self.actions: List[dict] = []
        self.result: Optional[dict] = None
        self.match_over = False
        self._final_pot: Optional[int] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def configure_players(self, seat0: dict, seat1: dict) -> None:
        """Set names/kinds for both seats before the first hand."""
        for seat, payload in ((0, seat0), (1, seat1)):
            player = self.players[seat]
            player.name = str(payload.get("name") or f"Player {seat + 1}")
            player.kind = str(payload.get("kind") or "human")

    @property
    def can_start_hand(self) -> bool:
        if self.street not in (Street.WAITING, Street.FINISHED):
            return False
        if self.match_over:
            return False
        if self.max_hands and self.hands_played >= self.max_hands:
            return False
        if not self.auto_rebuy and any(player.chips <= 0 for player in self.players):
            return False
        return True

    def _maybe_increase_blinds(self) -> None:
        """Raise the blinds after a configured number of completed hands."""
        every = self.blind_increase_every_hands
        if every <= 0 or self.hands_played <= 0 or self.hands_played % every != 0:
            return
        if self.blind_increase_factor <= 1.0:
            return
        ratio = self.big_blind / self.small_blind
        new_small = max(
            self.small_blind + 1,
            int(round(self.small_blind * self.blind_increase_factor)),
        )
        new_big = max(new_small + 1, int(round(new_small * ratio)))
        if new_small == self.small_blind and new_big == self.big_blind:
            return
        self.small_blind = new_small
        self.big_blind = new_big
        self.blind_level += 1

    def start_hand(self) -> None:
        """Deal a new hand. Raises when the match is over or a hand is active."""
        if self.street not in (Street.WAITING, Street.FINISHED):
            raise RuntimeError(tr(self.locale, "engine.rule.hand_active"))
        if self.match_over:
            raise RuntimeError(tr(self.locale, "engine.rule.match_over"))
        if self.max_hands and self.hands_played >= self.max_hands:
            self.match_over = True
            raise RuntimeError(tr(self.locale, "engine.rule.max_hands_reached"))

        self._maybe_increase_blinds()

        # Auto rebuy happens before the new hand is dealt.
        for player in self.players:
            if player.chips <= 0:
                if self.auto_rebuy:
                    player.chips = self.rebuy_amount
                    player.rebuy_count += 1
                else:
                    player.status = PlayerStatus.OUT
                    self.match_over = True
        if any(player.chips <= 0 for player in self.players):
            self.match_over = True
            raise RuntimeError(tr(self.locale, "engine.rule.players_broke"))

        self.hand_id += 1
        self.button_index = (self.button_index + 1) % 2
        self.deck.reset()
        self.deck.shuffle()
        self.community = []
        self.actions = []
        self.result = None
        self._final_pot = None
        self.actor = None
        self.current_bet = 0
        self.last_full_raise = self.big_blind
        self.min_raise_to = self.big_blind

        for player in self.players:
            player.reset_for_hand()
            player.is_button = player.seat == self.button_index
            player.is_small_blind = player.seat == self.button_index
            player.is_big_blind = player.seat != self.button_index

        # Deal one card at a time, starting with the button.
        for _ in range(2):
            for offset in range(2):
                seat = (self.button_index + offset) % 2
                self.players[seat].hole_cards.append(self.deck.deal_one())

        small_blind_player = self.players[self.button_index]
        big_blind_player = self.players[1 - self.button_index]
        small_blind_player.post_blind(self.small_blind)
        big_blind_player.post_blind(self.big_blind)

        self.current_bet = max(player.street_bet for player in self.players)
        self.last_full_raise = self.big_blind
        self.min_raise_to = max(
            self.big_blind,
            self.current_bet + self.last_full_raise,
        )
        self.street = Street.PREFLOP

        if not self._assign_preflop_actor():
            self._runout_remaining()
            return

    # ------------------------------------------------------------------
    # Public queries
    # ------------------------------------------------------------------

    @property
    def pot(self) -> int:
        """Effective pot. Once a hand is over this is the awarded pot."""
        if self._final_pot is not None:
            return self._final_pot
        return sum(player.total_bet for player in self.players)

    @property
    def human_seats(self) -> List[int]:
        return [player.seat for player in self.players if player.kind == "human"]

    def other_seat(self, seat: int) -> int:
        return 1 - seat

    def get_player(self, seat: int) -> HeadsUpPlayer:
        if seat not in (0, 1):
            raise ValueError(tr(self.locale, "engine.rule.seat"))
        return self.players[seat]

    def to_call(self, seat: int) -> int:
        player = self.get_player(seat)
        return max(0, self.current_bet - player.street_bet)

    def max_raise_to(self, seat: int) -> int:
        player = self.get_player(seat)
        return player.street_bet + player.chips

    def can_raise(self, seat: int) -> bool:
        player = self.get_player(seat)
        if player.status != PlayerStatus.ACTIVE:
            return False
        if not player.raise_allowed:
            return False
        opponent = self.players[1 - seat]
        if opponent.status == PlayerStatus.ALL_IN:
            # There is nobody left to call a raise.
            return False
        return player.chips > self.to_call(seat)

    def legal_actions(self, seat: Optional[int] = None) -> List[ActionType]:
        seat = self.actor if seat is None else seat
        if seat is None or seat != self.actor:
            return []
        player = self.get_player(seat)
        if player.status != PlayerStatus.ACTIVE:
            return []

        to_call = self.to_call(seat)
        legal: List[ActionType] = [ActionType.FOLD]
        if to_call == 0:
            legal.append(ActionType.CHECK)
            if self.can_raise(seat):
                legal.append(ActionType.BET)
        else:
            legal.append(ActionType.CALL)
            if self.can_raise(seat):
                legal.append(ActionType.RAISE)
        # ALL_IN is a legal shorthand for a raise, or for an all-in call when
        # the stack cannot cover the bet. It must not be offered when a short
        # all-in has closed the betting and the player still has chips behind.
        if player.chips > 0 and (self.can_raise(seat) or to_call >= player.chips):
            legal.append(ActionType.ALL_IN)
        return legal

    def action_constraints(self, seat: Optional[int] = None) -> dict:
        seat = self.actor if seat is None else seat
        if seat is None:
            return {}
        player = self.get_player(seat)
        to_call = self.to_call(seat)
        max_raise_to = self.max_raise_to(seat)
        can_raise = self.can_raise(seat)
        if not can_raise:
            # Do not advertise a raise-to range that is below the current bet.
            min_raise_to = self.current_bet
            max_raise_to = max(self.current_bet, max_raise_to)
        elif self.current_bet == 0:
            min_raise_to = min(self.big_blind, max_raise_to)
        else:
            min_raise_to = min(self.current_bet + self.last_full_raise, max_raise_to)
        return {
            "seat": seat,
            "to_call": to_call,
            "call_amount": min(to_call, player.chips),
            "current_bet": self.current_bet,
            "street_bet": player.street_bet,
            "min_raise_to": min_raise_to,
            "max_raise_to": max_raise_to,
            "can_raise": can_raise,
            "raise_allowed": can_raise,
            "is_all_in_call": to_call >= player.chips,
            "pot": self.pot,
            "pot_before_action": self.pot,
            "chips": player.chips,
        }

    def snapshot(self, reveal_all: bool = False, reveal_seats: Sequence[int] = ()) -> dict:
        """Serialize the table for the UI."""
        reveal_set = set(reveal_seats)
        players = []
        for player in self.players:
            visible = reveal_all or player.seat in reveal_set or player.kind == "human"
            hole = [card.to_dict() for card in player.hole_cards] if visible else []
            assessment = None
            if visible and len(player.hole_cards) == 2 and len(self.community) >= 3:
                assessment = assess_hand(player.hole_cards + self.community)
            players.append(
                {
                    "seat": player.seat,
                    "name": player.name,
                    "kind": player.kind,
                    "chips": player.chips,
                    "status": player.status.value,
                    "street_bet": player.street_bet,
                    "total_bet": player.total_bet,
                    "is_button": player.is_button,
                    "is_small_blind": player.is_small_blind,
                    "is_big_blind": player.is_big_blind,
                    "is_acting": self.actor == player.seat and self.street not in (
                        Street.FINISHED,
                        Street.SHOWDOWN,
                        Street.WAITING,
                    ),
                    "hole_cards": hole,
                    "hole_hidden": not visible,
                    "last_action": player.last_action,
                    "revealed": visible,
                    "assessment": assessment.to_dict() if assessment else None,
                    "rebuy_count": player.rebuy_count,
                }
            )

        legal = self.legal_actions() if self.actor is not None else []
        constraints = self.action_constraints() if self.actor is not None else {}
        acting_player = self.players[self.actor] if self.actor is not None else None
        return {
            "hand_id": self.hand_id,
            "street": self.street.value,
            "phase": self.street.value,
            "pot": self.pot,
            "current_bet": self.current_bet,
            "min_raise_to": constraints.get("min_raise_to"),
            "max_raise_to": constraints.get("max_raise_to"),
            "to_call": constraints.get("to_call", 0),
            "call_amount": constraints.get("call_amount", 0),
            "button": self.button_index,
            "acting_seat": self.actor,
            "acting_kind": acting_player.kind if acting_player else None,
            "legal_actions": [action.value for action in legal],
            "constraints": constraints,
            "community": [card.to_dict() for card in self.community],
            "burned": [card.to_dict() for card in self.deck.burned] if reveal_all else [],
            "deck_remaining": [card.to_dict() for card in self.deck.peek_from_top()] if reveal_all else [],
            "deck_remaining_count": self.deck.remaining,
            "players": players,
            "actions": list(self.actions),
            "result": self.result,
            "match_over": self.match_over,
            "small_blind": self.small_blind,
            "big_blind": self.big_blind,
            "starting_small_blind": self.starting_small_blind,
            "starting_big_blind": self.starting_big_blind,
            "blind_level": self.blind_level,
            "blind_increase_every_hands": self.blind_increase_every_hands,
            "blind_increase_factor": self.blind_increase_factor,
            "starting_chips": self.starting_chips,
        }

    # ------------------------------------------------------------------
    # Acting
    # ------------------------------------------------------------------

    def preview(self, action: Action, locale: Optional[str] = None) -> Optional[str]:
        """Validate an action on a copy without mutating the live game.

        Returns an error string, or ``None`` when the action is legal.
        """
        clone = copy.deepcopy(self)
        if locale:
            clone.locale = normalize_locale(locale, clone.locale) or clone.locale
        result = clone.apply(action)
        return None if result.ok else result.error

    def apply(self, action: Action) -> ActionResult:
        """Validate and apply an action. Returns an error result instead of raising."""
        if self.street in (Street.FINISHED, Street.SHOWDOWN, Street.WAITING):
            return ActionResult(False, action, tr(self.locale, "engine.action.hand_over"))
        if self.actor is None:
            return ActionResult(False, action, tr(self.locale, "engine.action.no_actor"))
        if action.seat != self.actor:
            return ActionResult(False, action, tr(self.locale, "engine.action.not_your_turn"))

        player = self.players[action.seat]
        opponent = self.players[1 - action.seat]
        if player.status != PlayerStatus.ACTIVE:
            return ActionResult(False, action, tr(self.locale, "engine.action.cannot_act"))

        to_call = self.to_call(player.seat)
        action_type = action.type

        if action_type == ActionType.FOLD:
            return self._apply_fold(player, action)

        if action_type == ActionType.CHECK:
            if to_call > 0:
                return ActionResult(
                    False,
                    action,
                    tr(self.locale, "engine.action.must_call", to_call=to_call),
                )
            self._record(player, action, pot_after=self.pot)
            player.acted_this_street = True
            player.raise_allowed = False
            return self._after_action(action, street_changed=False)

        if action_type == ActionType.CALL:
            if to_call <= 0:
                return ActionResult(False, action, tr(self.locale, "engine.action.no_call_check"))
            added = min(to_call, player.chips)
            self._commit(player, added)
            self._record(player, action, pot_after=self.pot, amount=added, added=added)
            player.acted_this_street = True
            player.raise_allowed = False
            return self._after_action(action, street_changed=False)

        # BET / RAISE / ALL_IN
        if action_type == ActionType.BET and to_call > 0:
            return ActionResult(
                False,
                action,
                tr(self.locale, "engine.action.use_raise", to_call=to_call),
            )
        if action_type == ActionType.RAISE and to_call == 0:
            return ActionResult(False, action, tr(self.locale, "engine.action.use_bet"))
        if not self.can_raise(player.seat) and not (
            action_type == ActionType.ALL_IN and to_call >= player.chips
        ):
            if not player.raise_allowed:
                return ActionResult(False, action, tr(self.locale, "engine.action.no_reraise_short"))
            return ActionResult(False, action, tr(self.locale, "engine.action.cannot_raise"))

        if action_type == ActionType.ALL_IN and to_call >= player.chips:
            # All-in is only a call when the stack cannot cover the bet.
            added = player.chips
            self._commit(player, added)
            self._record(player, action, pot_after=self.pot, amount=added, added=added)
            player.acted_this_street = True
            player.raise_allowed = False
            return self._after_action(action, street_changed=False)

        max_raise_to = self.max_raise_to(player.seat)
        if action_type == ActionType.ALL_IN:
            target = max_raise_to
        else:
            target = action.amount

        if isinstance(target, bool) or not isinstance(target, int):
            return ActionResult(False, action, tr(self.locale, "engine.action.raise_integer"))

        if target <= player.street_bet:
            return ActionResult(
                False,
                action,
                tr(
                    self.locale,
                    "engine.action.raise_above_invested",
                    invested=player.street_bet,
                ),
            )
        if target > max_raise_to:
            return ActionResult(
                False,
                action,
                tr(self.locale, "engine.action.raise_above_max", max=max_raise_to),
            )

        added = target - player.street_bet
        if added <= 0:
            return ActionResult(False, action, tr(self.locale, "engine.action.raise_positive"))

        previous_bet = self.current_bet
        previous_last_full_raise = self.last_full_raise

        if previous_bet == 0:
            # Opening bet.
            minimum = min(self.big_blind, max_raise_to)
            is_short_all_in = target == max_raise_to and target < self.big_blind
            if target < minimum and not is_short_all_in:
                return ActionResult(
                    False,
                    action,
                    tr(self.locale, "engine.action.min_bet", min=minimum),
                )
            is_full_raise = target >= self.big_blind
        else:
            if target <= previous_bet:
                return ActionResult(
                    False,
                    action,
                    tr(
                        self.locale,
                        "engine.action.raise_above_current",
                        current=previous_bet,
                    ),
                )
            minimum = min(previous_bet + previous_last_full_raise, max_raise_to)
            is_short_all_in = target == max_raise_to and target < minimum
            if target < minimum and not is_short_all_in:
                return ActionResult(
                    False,
                    action,
                    tr(self.locale, "engine.action.min_raise_to", min=minimum),
                )
            is_full_raise = (target - previous_bet) >= previous_last_full_raise

        self._commit(player, added)
        player.street_bet = target
        self.current_bet = target

        if is_full_raise:
            self.last_full_raise = target - previous_bet
            if opponent.status == PlayerStatus.ACTIVE:
                opponent.raise_allowed = True
        # A short all-in leaves last_full_raise untouched; betting stays closed
        # for an opponent who has already acted on this street.

        self.min_raise_to = self.current_bet + self.last_full_raise
        player.acted_this_street = True
        player.raise_allowed = False
        self._record(player, action, pot_after=self.pot, amount=target, added=added)
        return self._after_action(action, street_changed=False)

    def _commit(self, player: HeadsUpPlayer, added: int) -> None:
        added = min(added, player.chips)
        player.chips -= added
        player.street_bet += added
        player.total_bet += added
        if player.chips == 0:
            player.status = PlayerStatus.ALL_IN

    def _record(
        self,
        player: HeadsUpPlayer,
        action: Action,
        pot_after: int,
        amount: int = 0,
        added: int = 0,
    ) -> None:
        entry = {
            "hand_id": self.hand_id,
            "street": self.street.value,
            "seat": player.seat,
            "type": action.type.value,
            "label": action.type.label,
            "amount": int(amount if action.type in (ActionType.BET, ActionType.RAISE, ActionType.CALL, ActionType.ALL_IN) else 0),
            "added": int(added),
            "pot_after": pot_after,
            "chips_after": player.chips,
            "source": action.source,
            "auto": action.auto,
        }
        self.actions.append(entry)
        player.last_action = entry

    def _apply_fold(self, player: HeadsUpPlayer, action: Action) -> ActionResult:
        player.status = PlayerStatus.FOLDED
        self._record(player, action, pot_after=self.pot)
        self._finish_by_fold(player)
        return ActionResult(
            True,
            action,
            hand_finished=True,
            street_changed=True,
        )

    def _after_action(self, action: Action, street_changed: bool) -> ActionResult:
        if self._round_complete():
            changed = self._advance_street()
            return ActionResult(
                True,
                action,
                hand_finished=self.street in (Street.FINISHED, Street.SHOWDOWN),
                street_changed=True or changed,
                runout=self.street == Street.FINISHED and self.actor is None,
            )
        # Continue on the same street.
        opponent = self.players[1 - action.seat]
        if opponent.status == PlayerStatus.ACTIVE:
            self.actor = opponent.seat
        elif opponent.status == PlayerStatus.ALL_IN:
            current = self.players[action.seat]
            if current.status == PlayerStatus.ACTIVE and current.street_bet < opponent.street_bet:
                self.actor = current.seat
            else:
                self._runout_remaining()
        else:
            self.actor = None
        return ActionResult(
            True,
            action,
            hand_finished=self.street in (Street.FINISHED, Street.SHOWDOWN),
            street_changed=street_changed,
            runout=self.street == Street.FINISHED and self.actor is None,
        )

    # ------------------------------------------------------------------
    # Street / hand transitions
    # ------------------------------------------------------------------

    def _assign_preflop_actor(self) -> bool:
        sb = self.players[self.button_index]
        bb = self.players[1 - self.button_index]
        actor: Optional[int] = None
        if sb.status == PlayerStatus.ACTIVE:
            if bb.status == PlayerStatus.ACTIVE:
                actor = sb.seat
            elif bb.status == PlayerStatus.ALL_IN and sb.street_bet < bb.street_bet:
                actor = sb.seat
        elif bb.status == PlayerStatus.ACTIVE and bb.street_bet < sb.street_bet:
            actor = bb.seat
        self.actor = actor
        return actor is not None

    def _round_complete(self) -> bool:
        active = [player for player in self.players if player.status == PlayerStatus.ACTIVE]
        all_in = [player for player in self.players if player.status == PlayerStatus.ALL_IN]
        folded = [player for player in self.players if player.status == PlayerStatus.FOLDED]
        if folded:
            return True
        if not active:
            return True
        if len(active) == 1:
            current = active[0]
            opponent = all_in[0] if all_in else None
            if opponent is None:
                return True
            return current.street_bet >= opponent.street_bet
        return all(
            player.acted_this_street and player.street_bet == self.current_bet
            for player in active
        )

    def _advance_street(self) -> bool:
        if self.street == Street.RIVER:
            self._showdown()
            return True

        self._reset_street()
        self._deal_next_street()
        self.actor = None

        if not self._betting_possible():
            self._runout_remaining()
            return True

        # Postflop the non-button acts first.
        first = self.players[1 - self.button_index]
        if first.status == PlayerStatus.ACTIVE:
            self.actor = first.seat
        else:
            self.actor = self.players[self.button_index].seat
        return True

    def _betting_possible(self) -> bool:
        return all(player.status == PlayerStatus.ACTIVE for player in self.players)

    def _reset_street(self) -> None:
        for player in self.players:
            player.street_bet = 0
            player.acted_this_street = False
            player.raise_allowed = True
        self.current_bet = 0
        self.last_full_raise = self.big_blind
        self.min_raise_to = self.big_blind

    def _deal_next_street(self) -> None:
        if self.street == Street.PREFLOP:
            self.deck.burn()
            self.community.extend(self.deck.deal(3))
            self.street = Street.FLOP
        elif self.street == Street.FLOP:
            self.deck.burn()
            self.community.extend(self.deck.deal(1))
            self.street = Street.TURN
        elif self.street == Street.TURN:
            self.deck.burn()
            self.community.extend(self.deck.deal(1))
            self.street = Street.RIVER

    def _runout_remaining(self) -> None:
        """Deal the rest of the board when no more betting is possible."""
        self.actor = None
        while self.street in (Street.PREFLOP, Street.FLOP, Street.TURN):
            self._reset_street()
            self._deal_next_street()
        self._showdown()

    def _finish_by_fold(self, folder: HeadsUpPlayer) -> None:
        winner = self.players[1 - folder.seat]
        matched = min(player.total_bet for player in self.players)
        pot = matched * 2
        self._final_pot = pot
        refunds = {player.seat: player.total_bet - matched for player in self.players}
        for player in self.players:
            player.chips += refunds[player.seat]
        winner.chips += pot
        awarded = {winner.seat: pot}
        self.result = {
            "hand_id": self.hand_id,
            "reason": "fold",
            "winner_seats": [winner.seat],
            "awarded": awarded,
            "pot": pot,
            "refunds": refunds,
            "board": [card.code for card in self.community],
            "players": [
                {
                    "seat": player.seat,
                    "chips_after": player.chips,
                    "total_bet": player.total_bet,
                    "net": (
                        awarded.get(player.seat, 0)
                        + refunds[player.seat]
                        - player.total_bet
                    ),
                }
                for player in self.players
            ],
            "showdown": None,
            "match_over": False,
        }
        self._close_hand()

    def _showdown(self) -> None:
        self.street = Street.SHOWDOWN
        self.actor = None
        matched = min(player.total_bet for player in self.players)
        pot = matched * 2
        self._final_pot = pot
        refunds = {player.seat: player.total_bet - matched for player in self.players}
        for player in self.players:
            player.chips += refunds[player.seat]

        hands = {}
        for player in self.players:
            hands[player.seat] = assess_hand(player.hole_cards + self.community)

        comparisons = [
            (seat, hand.score if hand else MAX_SCORE)
            for seat, hand in hands.items()
        ]
        best_score = min(score for _, score in comparisons)
        winner_seats = [seat for seat, score in comparisons if score == best_score]

        award = pot // len(winner_seats)
        remainder = pot - award * len(winner_seats)
        awarded = {seat: 0 for seat in winner_seats}
        for seat in winner_seats:
            awarded[seat] += award
        # Odd chip goes to the player out of position (non-button).
        if remainder:
            for seat in winner_seats:
                if seat != self.button_index:
                    awarded[seat] += remainder
                    break

        for seat, amount in awarded.items():
            self.players[seat].chips += amount

        self.result = {
            "hand_id": self.hand_id,
            "reason": "showdown",
            "winner_seats": winner_seats,
            "awarded": awarded,
            "pot": pot,
            "refunds": refunds,
            "board": [card.code for card in self.community],
            "players": [
                {
                    "seat": player.seat,
                    "chips_after": player.chips,
                    "total_bet": player.total_bet,
                    "net": (
                        awarded.get(player.seat, 0)
                        + refunds[player.seat]
                        - player.total_bet
                    ),
                    "category": hands[player.seat].category if hands[player.seat] else "incomplete",
                    "category_name": CATEGORY_BY_KEY[
                        hands[player.seat].category if hands[player.seat] else "incomplete"
                    ]["name"],
                }
                for player in self.players
            ],
            "showdown": [
                {
                    "seat": player.seat,
                    "category": hands[player.seat].category if hands[player.seat] else "incomplete",
                    "category_name": CATEGORY_BY_KEY[
                        hands[player.seat].category if hands[player.seat] else "incomplete"
                    ]["name"],
                    "best_five": hands[player.seat].best_five if hands[player.seat] else [],
                    "score": hands[player.seat].score if hands[player.seat] else None,
                }
                for player in self.players
            ],
            "match_over": False,
        }
        self._close_hand()

    def _close_hand(self) -> None:
        self.street = Street.FINISHED
        self.actor = None
        self.current_bet = 0
        self.last_full_raise = self.big_blind
        self.min_raise_to = self.big_blind
        for player in self.players:
            player.street_bet = 0
            player.acted_this_street = False
            player.raise_allowed = True
        self.hands_played += 1
        if any(player.chips <= 0 for player in self.players):
            if not self.auto_rebuy:
                self.match_over = True
        if self.max_hands and self.hands_played >= self.max_hands:
            self.match_over = True
        if self.result is not None:
            self.result["match_over"] = self.match_over
            self.result["chips"] = [player.chips for player in self.players]

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def hand_summary(self) -> dict:
        return {
            "hand_id": self.hand_id,
            "street": self.street.value,
            "pot": self.pot,
            "community": [card.code for card in self.community],
            "button": self.button_index,
            "players": [
                {
                    "seat": player.seat,
                    "name": player.name,
                    "kind": player.kind,
                    "chips": player.chips,
                    "street_bet": player.street_bet,
                    "total_bet": player.total_bet,
                    "status": player.status.value,
                }
                for player in self.players
            ],
            "result": self.result,
        }


__all__ = [
    "ActionResult",
    "HeadsUpGame",
    "HeadsUpPlayer",
    "PlayerStatus",
    "Street",
]
