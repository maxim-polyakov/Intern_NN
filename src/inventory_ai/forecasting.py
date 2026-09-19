"""Deterministic demand forecast and purchase recommendation."""

from __future__ import annotations

import math
import statistics
from datetime import date, timedelta
from typing import Any

from inventory_ai.config import (
    CONFIDENCE_BASE,
    CONFIDENCE_HISTORY_WEIGHT,
    CONFIDENCE_LEVEL_SHIFT_PENALTY,
    CONFIDENCE_MISSING_PENALTY,
    CONFIDENCE_THRESHOLD,
    CONFIDENCE_VOLATILITY_WEIGHT,
    EWMA_ALPHA,
    LEVEL_SHIFT_RATIO,
    LEVEL_SHIFT_WEEKS,
    MAX_CONFIDENCE,
    MIN_CONFIDENCE,
    MIN_LEVEL_BASELINE_WEEKS,
    OUTLIER_MAD_MULTIPLIER,
    TARGET_HISTORY_WEEKS,
)
from inventory_ai.data import catalog_by_sku


def _impute_missing(values: list[float | None]) -> tuple[list[float], int]:
    result: list[float] = []
    valid = [float(value) for value in values if value is not None]
    if not valid:
        raise ValueError("Consumption history has no numeric observations")
    fallback = statistics.median(valid)
    missing = 0
    for index, value in enumerate(values):
        if value is not None:
            result.append(float(value))
            continue
        missing += 1
        before = next(
            (
                float(values[pos])
                for pos in range(index - 1, -1, -1)
                if values[pos] is not None
            ),
            None,
        )
        after = next(
            (
                float(values[pos])
                for pos in range(index + 1, len(values))
                if values[pos] is not None
            ),
            None,
        )
        neighbors = [item for item in (before, after) if item is not None]
        result.append(statistics.mean(neighbors) if neighbors else fallback)
    return result, missing


def _winsorize_isolated_outliers(values: list[float]) -> list[float]:
    if len(values) < 5:
        return values[:]
    median = statistics.median(values)
    deviations = [abs(value - median) for value in values]
    mad = statistics.median(deviations)
    if mad == 0:
        return values[:]
    lower = median - OUTLIER_MAD_MULTIPLIER * mad
    upper = median + OUTLIER_MAD_MULTIPLIER * mad
    return [min(max(value, lower), upper) for value in values]


def _detect_level_shift(values: list[float]) -> bool:
    split = len(values) - LEVEL_SHIFT_WEEKS
    if split < MIN_LEVEL_BASELINE_WEEKS:
        return False
    baseline = statistics.mean(values[:split])
    recent = statistics.mean(values[split:])
    return baseline > 0 and recent / baseline >= LEVEL_SHIFT_RATIO


def _forecast_weekly(values: list[float], level_shift: bool) -> float:
    if level_shift:
        return statistics.mean(values[-LEVEL_SHIFT_WEEKS:])
    smoothed = values[0]
    for value in values[1:]:
        smoothed = EWMA_ALPHA * value + (1.0 - EWMA_ALPHA) * smoothed
    return smoothed


def _round_order(raw_quantity: float, pack_size: float, minimum: float) -> float:
    if raw_quantity <= 0:
        return 0.0
    target = max(raw_quantity, minimum)
    return math.ceil(target / pack_size) * pack_size


def _round(value: float) -> float:
    return round(value, 2)


def forecast_demand(
    history: dict[str, Any],
    sku: str,
    horizon_days: int,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Forecast SKU demand and calculate a deterministic purchase recommendation."""
    normalized_sku = sku.upper()
    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive")
    catalog = catalog_by_sku(params.get("catalog"))
    if normalized_sku not in catalog:
        raise KeyError(f"Unknown SKU: {normalized_sku}")
    try:
        raw_values = history["weekly_consumption"][normalized_sku]
        current_stock = float(history["current_stock"][normalized_sku])
    except KeyError as error:
        raise KeyError(f"Missing history for SKU: {normalized_sku}") from error
    incoming_qty = float(history.get("incoming_qty", {}).get(normalized_sku, 0.0))
    if not isinstance(raw_values, list) or not raw_values:
        raise ValueError("Weekly consumption must be a non-empty list")

    imputed, missing_count = _impute_missing(raw_values)
    level_shift = _detect_level_shift(imputed)
    cleaned = imputed if level_shift else _winsorize_isolated_outliers(imputed)
    weekly_rate = _forecast_weekly(cleaned, level_shift)
    demand_multiplier = float(params.get("demand_multiplier", 1.0))
    if demand_multiplier <= 0:
        raise ValueError("demand_multiplier must be positive")
    daily_rate = weekly_rate / 7.0 * demand_multiplier

    product = catalog[normalized_sku]
    safety_stock = daily_rate * float(product["safety_stock_days"])
    reorder_point = daily_rate * (
        float(product["lead_time_days"]) + float(product["safety_stock_days"])
    )
    demand = daily_rate * horizon_days
    raw_order = max(0.0, demand + safety_stock - current_stock - incoming_qty)
    recommended = _round_order(
        raw_order,
        float(product["pack_size"]),
        float(product["min_order_qty"]),
    )
    estimated_cost = recommended * float(product["price"])

    available = current_stock + incoming_qty
    days_to_stockout = math.floor(available / daily_rate) if daily_rate else 0
    as_of = date.fromisoformat(history["as_of"])
    stockout_date = (as_of + timedelta(days=days_to_stockout)).isoformat()

    mean = statistics.mean(cleaned)
    volatility = statistics.pstdev(cleaned) / mean if mean else 1.0
    history_score = min(len(cleaned) / TARGET_HISTORY_WEEKS, 1.0)
    confidence = (
        CONFIDENCE_BASE
        + CONFIDENCE_HISTORY_WEIGHT * history_score
        - CONFIDENCE_VOLATILITY_WEIGHT * volatility
        - CONFIDENCE_MISSING_PENALTY * (missing_count / len(raw_values))
        - (CONFIDENCE_LEVEL_SHIFT_PENALTY if level_shift else 0.0)
    )
    confidence = min(MAX_CONFIDENCE, max(MIN_CONFIDENCE, confidence))
    status = (
        "расчёт пригоден для планирования"
        if confidence >= CONFIDENCE_THRESHOLD
        else "требуется уточнение"
    )
    method = (
        f"среднее последних {LEVEL_SHIFT_WEEKS} недель после устойчивой смены уровня"
        if level_shift
        else f"экспоненциальное сглаживание α={EWMA_ALPHA}"
    )
    missing_note = (
        f"; восстановлено пропусков: {missing_count}"
        if missing_count
        else "; пропусков нет"
    )
    explanation = (
        f"{method}{missing_note}. Суточный расход {daily_rate:.2f} "
        f"{product['unit']}, спрос на {horizon_days} дн. {demand:.2f}; "
        f"страховой запас {safety_stock:.2f}, остаток {current_stock:.2f}, "
        f"в пути {incoming_qty:.2f}. Потребность {raw_order:.2f} округлена "
        f"до упаковки {product['pack_size']} и min партии "
        f"{product['min_order_qty']}: {recommended:g}. Confidence "
        f"{confidence:.2f}: {status}."
    )
    return {
        "avg_daily_consumption": _round(daily_rate),
        "forecast_demand": _round(demand),
        "current_stock": _round(current_stock),
        "incoming_qty": _round(incoming_qty),
        "safety_stock": _round(safety_stock),
        "reorder_point": _round(reorder_point),
        "recommended_qty": _round(recommended),
        "estimated_cost": _round(estimated_cost),
        "stockout_date": stockout_date,
        "confidence": round(confidence, 3),
        "explanation": explanation,
    }
