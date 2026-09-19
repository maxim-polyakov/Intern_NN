"""Offline, rule-based parsing and answering of inventory questions."""

from __future__ import annotations

import re
from typing import Any

from inventory_ai.config import (
    DAYS_IN_HALF_YEAR,
    DAYS_IN_MONTH,
    DAYS_IN_QUARTER,
    DAYS_IN_YEAR,
    INTENT_AMBIGUITY_MARGIN,
    QA_AMBIGUOUS_SKU_CONFIDENCE,
    QA_INTENT_BASE_CONFIDENCE,
    QA_INTENT_MATCH_WEIGHT,
    QA_MAX_CONFIDENCE,
    QA_MISSING_PARAMETER_CONFIDENCE,
    QA_UNKNOWN_CONFIDENCE,
    SCENARIO_DEMAND_INCREASE,
)
from inventory_ai.forecasting import forecast_demand

_SKU_TERMS = {
    "OIL-001": ("oil-001", "oil 001", "базов", "миндал"),
    "OIL-002": ("oil-002", "oil 002", "ароматичес", "лаванд"),
    "SCRB-020": ("scrb-020", "scrb 020", "скраб"),
    "WRAP-030": ("wrap-030", "wrap 030", "альгинат", "обёртыван"),
    "CONS-051": ("cons-051", "cons 051", "тапоч"),
    "CONS-052": ("cons-052", "cons 052", "шапоч"),
}

_LOCATION_TERMS = {
    "MS-01": ("ms-01", "ms 01"),
    "MS-02": ("ms-02", "ms 02"),
    "Сочи": ("сочи",),
    "Красная Поляна": ("красной поляне", "красная поляна"),
}

_INTENT_KEYWORDS = {
    "forecast_purchase": (
        "закупк",
        "закупить",
        "заказать масло",
        "сколько масла уйд",
    ),
    "reorder_list": ("что нужно заказать", "что заказать", "список заказ"),
    "budget": ("бюджет", "сколько стоит закупить всё"),
    "deficit_risk": ("дефицит", "что закончится", "в риске дефицита"),
    "expiry_risk": ("сгорят", "срок годности", "просроч"),
    "price_dynamics": ("подорожал", "динамик цен", "изменени цен"),
}


def _extract_sku(question: str) -> tuple[str | None, bool]:
    lowered = question.lower()
    matches = [
        sku
        for sku, aliases in _SKU_TERMS.items()
        if any(item in lowered for item in aliases)
    ]
    generic_oil = "масл" in lowered and not any(
        sku.startswith("OIL") for sku in matches
    )
    if generic_oil:
        return None, True
    return (matches[0], len(matches) > 1) if matches else (None, False)


def _extract_location(question: str) -> str | None:
    lowered = question.lower()
    for location, aliases in _LOCATION_TERMS.items():
        if any(alias in lowered for alias in aliases):
            return location
    return None


def _extract_period(question: str) -> int | None:
    lowered = question.lower()
    day_match = re.search(r"(?:ближайшие\s+)?(\d+)\s*(?:дн|дней|дня)", lowered)
    if day_match:
        return int(day_match.group(1))
    if "полгода" in lowered or "полугод" in lowered:
        return DAYS_IN_HALF_YEAR
    if "три месяца" in lowered or "3 месяца" in lowered or "квартал" in lowered:
        return DAYS_IN_QUARTER
    if "месяц" in lowered:
        return DAYS_IN_MONTH
    if re.search(r"\b(?:год|года|годовой)\b", lowered):
        return DAYS_IN_YEAR
    return None


def _extract_budget(question: str) -> float | None:
    match = re.search(
        r"(?:лимит\w*\s*)?(\d+(?:[.,]\d+)?)\s*(тысяч|тыс\.?|млн)?",
        question.lower(),
    )
    if not match or ("лимит" not in question.lower() and not match.group(2)):
        return None
    amount = float(match.group(1).replace(",", "."))
    suffix = match.group(2) or ""
    if suffix.startswith("тыс"):
        amount *= 1_000
    elif suffix == "млн":
        amount *= 1_000_000
    return amount


def _intent_scores(question: str) -> dict[str, float]:
    lowered = question.lower()
    scores = {
        intent: sum(1.0 for keyword in keywords if keyword in lowered)
        for intent, keywords in _INTENT_KEYWORDS.items()
    }
    if "сколько" in lowered and ("закупить" in lowered or "закупк" in lowered):
        scores["forecast_purchase"] += 1.0
    return scores


def _clarification(
    parsed: dict[str, Any],
    message: str,
    confidence: float,
) -> tuple[dict[str, Any], float, str]:
    parsed["intent"] = "unknown"
    return parsed, confidence, message


def _all_forecasts(context: dict[str, Any], period: int) -> dict[str, dict[str, Any]]:
    history = context["history"]
    params = {"catalog": context.get("catalog")}
    return {
        sku: forecast_demand(history, sku, period, params)
        for sku in history["weekly_consumption"]
    }


def answer_question(
    question: str,
    context: dict[str, Any],
) -> tuple[dict[str, Any], float, str]:
    """Parse and answer a Russian inventory question without external services."""
    sku, ambiguous_sku = _extract_sku(question)
    period = _extract_period(question)
    location = _extract_location(question)
    budget_limit = _extract_budget(question)
    parsed: dict[str, Any] = {
        "intent": "unknown",
        "sku": sku,
        "location": location,
        "period_days": period,
        "budget_limit": budget_limit,
    }
    if ambiguous_sku:
        return _clarification(
            parsed,
            "Уточните товар: базовое масло OIL-001 или ароматическое OIL-002?",
            QA_AMBIGUOUS_SKU_CONFIDENCE,
        )

    scores = _intent_scores(question)
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_intent, best_score = ranked[0]
    second_score = ranked[1][1]
    if best_score == 0 or best_score - second_score < INTENT_AMBIGUITY_MARGIN:
        return _clarification(
            parsed,
            "Не удалось однозначно определить складской запрос. Уточните товар, "
            "период и требуемый расчёт.",
            QA_UNKNOWN_CONFIDENCE,
        )
    parsed["intent"] = best_intent
    confidence = min(
        QA_MAX_CONFIDENCE,
        QA_INTENT_BASE_CONFIDENCE + QA_INTENT_MATCH_WEIGHT * best_score,
    )

    if best_intent == "forecast_purchase":
        if sku is None or period is None:
            missing = "товар" if sku is None else "период"
            return _clarification(
                parsed,
                f"Уточните {missing} для расчёта закупки.",
                QA_MISSING_PARAMETER_CONFIDENCE,
            )
        result = forecast_demand(
            context["history"],
            sku,
            period,
            {
                "catalog": context.get("catalog"),
                "demand_multiplier": (
                    SCENARIO_DEMAND_INCREASE if "20%" in question else 1.0
                ),
            },
        )
        over_budget = (
            budget_limit is not None and result["estimated_cost"] > budget_limit
        )
        budget_note = (
            f" Лимит {budget_limit:.2f} превышен."
            if over_budget
            else (
                f" Расчёт укладывается в лимит {budget_limit:.2f}."
                if budget_limit is not None
                else ""
            )
        )
        answer = (
            f"{sku}: заказать {result['recommended_qty']:g} базовых ед. "
            f"на сумму {result['estimated_cost']:.2f}. "
            f"{result['explanation']}{budget_note}"
        )
        return parsed, min(confidence, result["confidence"]), answer

    if best_intent in {"reorder_list", "budget"}:
        if period is None:
            return _clarification(
                parsed,
                "Уточните период расчёта списка закупок.",
                QA_MISSING_PARAMETER_CONFIDENCE,
            )
        forecasts = _all_forecasts(context, period)
        orders = {
            item_sku: value
            for item_sku, value in forecasts.items()
            if value["recommended_qty"] > 0
        }
        total = sum(value["estimated_cost"] for value in orders.values())
        details = (
            ", ".join(
                f"{item_sku} — {value['recommended_qty']:g} ед."
                for item_sku, value in orders.items()
            )
            or "заказ не требуется"
        )
        return (
            parsed,
            min([confidence, *(value["confidence"] for value in forecasts.values())]),
            f"На {period} дн.: {details}. Общая стоимость {total:.2f}.",
        )

    if best_intent == "deficit_risk":
        if period is None:
            return _clarification(
                parsed,
                "Уточните период оценки риска дефицита.",
                QA_MISSING_PARAMETER_CONFIDENCE,
            )
        forecasts = _all_forecasts(context, period)
        risky = [
            item_sku
            for item_sku, value in forecasts.items()
            if value["recommended_qty"] > 0
        ]
        return (
            parsed,
            confidence,
            (
                f"Риск дефицита на {period} дн.: {', '.join(risky) or 'не выявлен'}. "
                "Оценка основана на детерминированном прогнозе."
            ),
        )

    if best_intent == "expiry_risk":
        return (
            parsed,
            confidence,
            "В контексте нет сроков годности партий; численный ответ невозможен.",
        )
    if best_intent == "price_dynamics":
        return (
            parsed,
            confidence,
            "В контексте есть только текущая цена, но нет истории цен; "
            "динамику рассчитать нельзя.",
        )
    return _clarification(parsed, "Запрос не поддерживается.", QA_UNKNOWN_CONFIDENCE)
