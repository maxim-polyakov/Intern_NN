"""Tests for deterministic demand forecasting."""

from __future__ import annotations

import copy
from datetime import date, timedelta

import pytest

from inventory_ai.data import load_catalog, load_dataset
from solution import forecast_demand


@pytest.fixture
def inputs() -> tuple[dict, dict]:
    return load_dataset()["forecast"], {"catalog": load_catalog()}


@pytest.mark.parametrize("sku", ["OIL-001", "SCRB-020", "WRAP-030"])
@pytest.mark.parametrize("days", [30, 90])
def test_assignment_forecasts_are_complete_and_deterministic(
    inputs: tuple[dict, dict], sku: str, days: int
) -> None:
    history, params = inputs
    first = forecast_demand(history, sku, days, params)
    second = forecast_demand(history, sku, days, params)
    assert first == second
    assert set(first) == {
        "avg_daily_consumption",
        "forecast_demand",
        "current_stock",
        "incoming_qty",
        "safety_stock",
        "reorder_point",
        "recommended_qty",
        "estimated_cost",
        "stockout_date",
        "confidence",
        "explanation",
    }
    assert 0 <= first["confidence"] <= 1


def test_order_respects_pack_and_minimum(inputs: tuple[dict, dict]) -> None:
    history, params = inputs
    result = forecast_demand(history, "OIL-001", 90, params)
    assert result["recommended_qty"] >= 10
    assert result["recommended_qty"] % 5 == 0


def test_no_order_when_inventory_covers_demand(
    inputs: tuple[dict, dict],
) -> None:
    history, params = inputs
    ample = copy.deepcopy(history)
    ample["current_stock"]["OIL-001"] = 10_000
    result = forecast_demand(ample, "OIL-001", 30, params)
    assert result["recommended_qty"] == 0
    assert result["estimated_cost"] == 0


def test_level_shift_uses_recent_regime(inputs: tuple[dict, dict]) -> None:
    history, params = inputs
    result = forecast_demand(history, "SCRB-020", 30, params)
    expected_daily = ((6.8 + 7.4 + 7.1 + 7.6) / 4) / 7
    assert result["avg_daily_consumption"] == pytest.approx(expected_daily, abs=0.01)


def test_missing_week_is_handled(inputs: tuple[dict, dict]) -> None:
    history, params = inputs
    result = forecast_demand(history, "WRAP-030", 30, params)
    assert result["forecast_demand"] > 0
    assert "восстановлено пропусков: 1" in result["explanation"]


def test_stockout_date_is_conservative_when_arrival_date_is_unknown(
    inputs: tuple[dict, dict],
) -> None:
    history, params = inputs
    with_incoming = copy.deepcopy(history)
    without_incoming = copy.deepcopy(history)
    with_incoming["incoming_qty"]["OIL-001"] = 10_000
    without_incoming["incoming_qty"]["OIL-001"] = 0

    first = forecast_demand(with_incoming, "OIL-001", 30, params)
    second = forecast_demand(without_incoming, "OIL-001", 30, params)

    assert first["stockout_date"] == second["stockout_date"]
    expected_days = int(
        history["current_stock"]["OIL-001"] / first["avg_daily_consumption"]
    )
    expected = date.fromisoformat(history["as_of"]) + timedelta(days=expected_days)
    assert first["stockout_date"] == expected.isoformat()


def test_zero_demand_has_no_stockout_date(inputs: tuple[dict, dict]) -> None:
    history, params = inputs
    zero = copy.deepcopy(history)
    zero["weekly_consumption"]["OIL-001"] = [0.0] * 12

    result = forecast_demand(zero, "OIL-001", 30, params)

    assert result["avg_daily_consumption"] == 0
    assert result["stockout_date"] is None


def test_confidence_falls_for_short_and_volatile_history(
    inputs: tuple[dict, dict],
) -> None:
    history, params = inputs
    stable = copy.deepcopy(history)
    short = copy.deepcopy(history)
    volatile = copy.deepcopy(history)
    stable["weekly_consumption"]["OIL-001"] = [10.0] * 12
    short["weekly_consumption"]["OIL-001"] = [10.0] * 3
    volatile["weekly_consumption"]["OIL-001"] = [1.0, 20.0] * 6

    stable_confidence = forecast_demand(stable, "OIL-001", 30, params)["confidence"]
    short_confidence = forecast_demand(short, "OIL-001", 30, params)["confidence"]
    volatile_confidence = forecast_demand(volatile, "OIL-001", 30, params)["confidence"]

    assert short_confidence < stable_confidence
    assert volatile_confidence < stable_confidence


def test_isolated_recent_outlier_is_not_a_level_shift(
    inputs: tuple[dict, dict],
) -> None:
    history, params = inputs
    outlier = copy.deepcopy(history)
    outlier["weekly_consumption"]["OIL-001"] = [
        9,
        10,
        11,
        10,
        9,
        11,
        10,
        9,
        10,
        11,
        10,
        100,
    ]

    result = forecast_demand(outlier, "OIL-001", 30, params)

    assert "экспоненциальное сглаживание" in result["explanation"]
    assert result["avg_daily_consumption"] < 3


def test_small_positive_need_respects_minimum_order(
    inputs: tuple[dict, dict],
) -> None:
    history, params = inputs
    near_threshold = copy.deepcopy(history)
    initial = forecast_demand(near_threshold, "OIL-001", 30, params)
    near_threshold["current_stock"]["OIL-001"] = (
        initial["forecast_demand"]
        + initial["safety_stock"]
        - near_threshold["incoming_qty"]["OIL-001"]
        - 0.1
    )

    result = forecast_demand(near_threshold, "OIL-001", 30, params)

    assert result["recommended_qty"] == 10


def test_estimated_cost_uses_catalog_price(inputs: tuple[dict, dict]) -> None:
    history, params = inputs
    result = forecast_demand(history, "SCRB-020", 90, params)
    price = next(
        item["price"] for item in params["catalog"] if item["sku"] == "SCRB-020"
    )

    assert result["estimated_cost"] == pytest.approx(result["recommended_qty"] * price)


@pytest.mark.parametrize("days", [0, -1])
def test_invalid_horizon(inputs: tuple[dict, dict], days: int) -> None:
    history, params = inputs
    with pytest.raises(ValueError, match="positive"):
        forecast_demand(history, "OIL-001", days, params)
