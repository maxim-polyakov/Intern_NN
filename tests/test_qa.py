"""Tests for offline question parsing and guarded answers."""

from __future__ import annotations

import copy

import pytest

from inventory_ai.data import load_catalog, load_dataset
from solution import answer_question, forecast_demand


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


def test_all_assignment_questions_have_expected_intents(context: dict) -> None:
    expected = [
        "unknown",
        "reorder_list",
        "budget",
        "unknown",
        "expiry_risk",
        "price_dynamics",
        "unknown",
        "forecast_purchase",
        "unknown",
        "unknown",
        "unknown",
        "unknown",
        "unknown",
        "unknown",
        "unknown",
    ]
    actual = [
        answer_question(question, context)[0]["intent"]
        for question in load_dataset()["questions"]
    ]

    assert actual == expected


def test_numeric_purchase_answer_is_grounded_in_forecast(context: dict) -> None:
    question = "Посчитай закупку скраба на полгода при лимите 200 тысяч"
    _, _, answer = answer_question(question, context)
    forecast = forecast_demand(
        context["history"],
        "SCRB-020",
        180,
        {"catalog": context["catalog"]},
    )

    assert f"{forecast['recommended_qty']:g}" in answer
    assert f"{forecast['estimated_cost']:.2f}" in answer
    assert f"{forecast['avg_daily_consumption']:.2f}" in answer


def test_year_period_and_million_budget_are_parsed(context: dict) -> None:
    parsed, _, answer = answer_question(
        "Посчитай закупку скраба на год при лимите 1 млн",
        context,
    )

    assert parsed["period_days"] == 365
    assert parsed["budget_limit"] == 1_000_000
    assert "укладывается в лимит" in answer


@pytest.mark.parametrize(
    ("question", "missing"),
    [
        ("Посчитай закупку на месяц", "товар"),
        ("Посчитай закупку скраба", "период"),
    ],
)
def test_purchase_requires_sku_and_period(
    context: dict,
    question: str,
    missing: str,
) -> None:
    parsed, _, answer = answer_question(question, context)

    assert parsed["intent"] == "unknown"
    assert missing in answer


def test_reorder_list_can_report_no_order(context: dict) -> None:
    ample = copy.deepcopy(context)
    for sku in ample["history"]["current_stock"]:
        ample["history"]["current_stock"][sku] = 1_000_000

    parsed, _, answer = answer_question(
        "Что нужно заказать в ближайшие 14 дней?",
        ample,
    )

    assert parsed["intent"] == "reorder_list"
    assert "заказ не требуется" in answer


def test_deficit_risk_with_period_is_calculated(context: dict) -> None:
    parsed, _, answer = answer_question(
        "Что закончится за 30 дней?",
        context,
    )

    assert parsed["intent"] == "deficit_risk"
    assert "Риск дефицита на 30 дн." in answer


def test_demand_growth_scenario_is_applied(context: dict) -> None:
    parsed, _, answer = answer_question(
        "Сколько скраба закупить за 30 дней, если загрузка вырастет на 20%?",
        context,
    )
    baseline = forecast_demand(
        context["history"],
        "SCRB-020",
        30,
        {"catalog": context["catalog"]},
    )

    assert parsed["intent"] == "forecast_purchase"
    assert f"{baseline['avg_daily_consumption']:.2f}" not in answer
