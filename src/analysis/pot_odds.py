"""Static call EV and pot-odds helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from src.i18n import tr


@dataclass(frozen=True)
class CallEvResult:
    """Result of a static call-EV calculation."""

    pot: int
    to_call: int
    chips: int
    call_amount: int
    refund: int
    effective_pot: int
    equity: float
    ev_call: float
    ev_fold: float
    breakeven_equity: float
    pot_to_call_ratio: Optional[float]
    edge: float
    static_positive: bool

    def to_dict(self) -> dict:
        return {
            "pot": self.pot,
            "to_call": self.to_call,
            "chips": self.chips,
            "call_amount": self.call_amount,
            "refund": self.refund,
            "effective_pot": self.effective_pot,
            "equity": self.equity,
            "ev_call": self.ev_call,
            "ev_fold": self.ev_fold,
            "breakeven_equity": self.breakeven_equity,
            "pot_to_call_ratio": self.pot_to_call_ratio,
            "edge": self.edge,
            "static_positive": self.static_positive,
        }

    def to_llm_dict(self, locale: str = "zh-CN") -> dict:
        """Compact result for a model tool message."""
        return {
            "equity": round(self.equity, 4),
            "ev_call": round(self.ev_call, 2),
            "ev_fold": 0.0,
            "breakeven_equity": round(self.breakeven_equity, 4),
            "edge": round(self.edge, 4),
            "pot": self.pot,
            "to_call": self.to_call,
            "call_amount": self.call_amount,
            "effective_pot": self.effective_pot,
            "refund": self.refund,
            "pot_to_call_ratio": (
                round(self.pot_to_call_ratio, 3)
                if self.pot_to_call_ratio is not None
                else None
            ),
            "static_positive": self.static_positive,
            "note": tr(locale, "analysis.call_ev.note"),
        }


def analyze_call(
    pot: int,
    to_call: int,
    chips: int,
    equity: float,
    locale: str = "zh-CN",
) -> CallEvResult:
    """Calculate the static EV of calling.

    Args:
        pot: Current pot, already including the opponent's bet.
        to_call: Chips needed to match the current bet.
        chips: Chips behind before calling.
        equity: Showdown equity as a 0 to 1 decimal.

    Returns:
        The call EV and the pot-odds threshold.

    Raises:
        ValueError: If the inputs are not a legal calling spot.
    """
    if pot < 0:
        raise ValueError(tr(locale, "analysis.call_ev.pot_negative"))
    if to_call <= 0:
        raise ValueError(tr(locale, "analysis.call_ev.to_call_positive"))
    if chips < 0:
        raise ValueError(tr(locale, "analysis.call_ev.chips_negative"))
    if not math.isfinite(equity) or not 0.0 <= equity <= 1.0:
        raise ValueError(tr(locale, "analysis.call_ev.equity_range"))

    call_amount = min(int(to_call), int(chips))
    if call_amount <= 0:
        raise ValueError(tr(locale, "analysis.call_ev.no_call_chips"))
    refund = int(to_call) - call_amount
    effective_pot = int(pot) + 2 * call_amount - int(to_call)
    if effective_pot <= 0:
        raise ValueError(tr(locale, "analysis.call_ev.effective_pot_positive"))

    ev_call = equity * effective_pot - call_amount
    breakeven = call_amount / effective_pot
    # Profit-to-risk ratio after removing the opponent's uncalled refund.
    pot_to_call_ratio = (effective_pot - call_amount) / call_amount
    return CallEvResult(
        pot=int(pot),
        to_call=int(to_call),
        chips=int(chips),
        call_amount=call_amount,
        refund=refund,
        effective_pot=effective_pot,
        equity=float(equity),
        ev_call=float(ev_call),
        ev_fold=0.0,
        breakeven_equity=float(breakeven),
        pot_to_call_ratio=float(pot_to_call_ratio),
        edge=float(equity - breakeven),
        static_positive=bool(ev_call > 0),
    )


__all__ = ["CallEvResult", "analyze_call"]
