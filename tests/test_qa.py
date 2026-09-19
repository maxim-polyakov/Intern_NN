"""Tests for offline question parsing and guarded answers."""

from __future__ import annotations

import pytest

from inventory_ai.data import load_catalog, load_dataset
from solution import answer_question


@pytest.fixture
def context() -> dict:
    dataset = load_dataset()
    return {"history": dataset["forecast"], "catalog": load_catalog()}


def test_forecast_entities_and_budget(context: dict) -> None:
    parsed, confidence, answer = answer_question(
        "Посчитай закупку скраба на полгода при лимите 200 тысяч",
        context,
    )
    assert parsed == {
        "intent": "forecast_purchase",
        "sku": "SCRB-020",
        "location": None,
        "period_days": 180,
        "budget_limit": 200_000,
    }
    assert 0 <= confidence <= 1
    assert "SCRB-020" in answer
    assert "200000.00" in answer


def test_ambiguous_oil_requests_clarification(context: dict) -> None:
    parsed, confidence, answer = answer_question(
        "Сколько масла закупить на три месяца?", context
    )
    assert parsed["intent"] == "unknown"
    assert confidence < 0.6
    assert "OIL-001" in answer and "OIL-002" in answer


def test_missing_period_requests_clarification(context: dict) -> None:
    parsed, _, answer = answer_question(
        "Сколько стоит закупить всё, что в риске дефицита?", context
    )
    assert parsed["intent"] == "unknown"
    assert "период" in answer.lower()


def test_out_of_domain_is_unknown(context: dict) -> None:
    parsed, confidence, answer = answer_question(
        "Какая погода в Сочи на выходных?", context
    )
    assert parsed["intent"] == "unknown"
    assert parsed["location"] == "Сочи"
    assert confidence < 0.6
    assert "уточните" in answer.lower()


def test_price_history_is_not_invented(context: dict) -> None:
    parsed, _, answer = answer_question(
        "Насколько подорожало ароматическое масло у поставщика?", context
    )
    assert parsed["intent"] == "price_dynamics"
    assert parsed["sku"] == "OIL-002"
    assert "нет истории цен" in answer


def test_all_questions_return_contract(context: dict) -> None:
    for question in load_dataset()["questions"]:
        parsed, confidence, answer = answer_question(question, context)
        assert parsed["intent"] in {
            "forecast_purchase",
            "reorder_list",
            "budget",
            "deficit_risk",
            "expiry_risk",
            "price_dynamics",
            "unknown",
        }
        assert 0 <= confidence <= 1
        assert answer
