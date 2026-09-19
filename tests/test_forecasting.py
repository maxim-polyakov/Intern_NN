"""Tests for deterministic demand forecasting."""

from __future__ import annotations

import copy

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


@pytest.mark.parametrize("days", [0, -1])
def test_invalid_horizon(inputs: tuple[dict, dict], days: int) -> None:
    history, params = inputs
    with pytest.raises(ValueError, match="positive"):
        forecast_demand(history, "OIL-001", days, params)
