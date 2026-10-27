"""Static call-EV tests."""

from __future__ import annotations

import pytest

from src.analysis.pot_odds import analyze_call


def test_standard_call_breakeven() -> None:
    result = analyze_call(pot=150, to_call=50, chips=900, equity=0.25)
    assert result.call_amount == 50
    assert result.refund == 0
    assert result.effective_pot == 200
    assert result.ev_call == pytest.approx(0.0)
    assert result.breakeven_equity == pytest.approx(0.25)
    assert result.pot_to_call_ratio == pytest.approx(3.0)
    assert result.static_positive is False


def test_short_stack_refund_is_removed() -> None:
    result = analyze_call(pot=400, to_call=300, chips=100, equity=1.0)
    assert result.call_amount == 100
    assert result.refund == 200
    assert result.effective_pot == 300
    assert result.ev_call == pytest.approx(200.0)
    assert result.breakeven_equity == pytest.approx(1.0 / 3.0)
    assert result.pot_to_call_ratio == pytest.approx(2.0)
    assert result.static_positive is True


def test_losing_call_has_negative_ev() -> None:
    result = analyze_call(pot=150, to_call=50, chips=900, equity=0.10)
    assert result.ev_call == pytest.approx(-30.0)
    assert result.static_positive is False


def test_invalid_inputs_are_rejected() -> None:
    with pytest.raises(ValueError):
        analyze_call(pot=150, to_call=0, chips=900, equity=0.3)
    with pytest.raises(ValueError):
        analyze_call(pot=150, to_call=50, chips=900, equity=1.2)
    with pytest.raises(ValueError):
        analyze_call(pot=150, to_call=50, chips=0, equity=0.3)


def test_llm_payload_rounds_and_keeps_units() -> None:
    result = analyze_call(pot=150, to_call=50, chips=900, equity=0.30)
    payload = result.to_llm_dict()
    assert payload["ev_call"] == pytest.approx(10.0)
    assert payload["breakeven_equity"] == pytest.approx(0.25)
    assert "静态" in payload["note"]
