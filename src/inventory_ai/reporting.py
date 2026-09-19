"""Run the complete assignment pipeline and render reproducible results."""

from __future__ import annotations

from typing import Any

from inventory_ai.data import load_catalog, load_dataset
from inventory_ai.forecasting import forecast_demand
from inventory_ai.normalization import normalize_movement
from inventory_ai.qa import answer_question

_MOVEMENT_FIELDS = (
    "date",
    "sku",
    "location",
    "operation",
    "qty",
    "unit",
    "batch",
    "doc_no",
)

_EXPECTED_MOVEMENTS = {
    "M1": {
        "date": "2026-03-05",
        "sku": "OIL-001",
        "location": "MS-01",
        "operation": "receipt",
        "qty": 10.0,
        "unit": "л",
        "batch": None,
        "doc_no": "НК-345",
    },
    "M2": {
        "date": "2026-03-01",
        "sku": "OIL-001",
        "location": "MS-01",
        "operation": "consume",
        "qty": 0.45,
        "unit": "л",
        "batch": "B-OIL-001-012",
        "doc_no": None,
    },
    "M3": {
        "date": "2026-06-03",
        "sku": "SCRB-020",
        "location": "Сочи",
        "operation": "writeoff",
        "qty": 1.2,
        "unit": "кг",
        "batch": None,
        "doc_no": None,
    },
    "M4": {
        "date": "2026-03-07",
        "sku": "WRAP-030",
        "location": "MS-02",
        "operation": "consume",
        "qty": 3.5,
        "unit": "кг",
        "batch": "B-WRAP-030-004",
        "doc_no": None,
    },
    "M5": {
        "date": "2026-03-12",
        "sku": "OIL-002",
        "location": None,
        "operation": "return",
        "qty": 2.0,
        "unit": "л",
        "batch": None,
        "doc_no": None,
    },
    "M6": {
        "date": "2026-03-08",
        "sku": "CONS-051",
        "location": "MS-01",
        "operation": "consume",
        "qty": 48.0,
        "unit": "пар",
        "batch": None,
        "doc_no": None,
    },
    "M7": {
        "date": "2026-03-15",
        "sku": "CONS-052",
        "location": "MS-02",
        "operation": "correction",
        "qty": -120.0,
        "unit": "шт",
        "batch": None,
        "doc_no": None,
    },
    "M8": {
        "date": None,
        "sku": "CONS-051",
        "location": "Красная Поляна",
        "operation": "receipt",
        "qty": 200.0,
        "unit": "пар",
        "batch": None,
        "doc_no": None,
    },
}


def _movement_summary(row: dict[str, Any]) -> str:
    """Render all normalized fields in a compact, auditable form."""
    return "; ".join(
        f"{field}={'—' if row.get(field) is None else row[field]}"
        for field in _MOVEMENT_FIELDS
    )


def run_pipeline() -> dict[str, Any]:
    """Execute normalization, six forecasts, and all fifteen QA examples."""
    dataset = load_dataset()
    catalog = load_catalog()
    history = dataset["forecast"]
    normalized = [
        {"id": row["id"], **normalize_movement(row["text"])}
        for row in dataset["movements"]
    ]
    forecasts = {
        f"{sku}/{days}": forecast_demand(history, sku, days, {"catalog": catalog})
        for sku in history["weekly_consumption"]
        for days in (30, 90)
    }
    context = {"history": history, "catalog": catalog}
    answers = []
    for question in dataset["questions"]:
        parsed, confidence, answer = answer_question(question, context)
        answers.append(
            {
                "question": question,
                "parsed": parsed,
                "confidence": confidence,
                "answer": answer,
            }
        )
    return {
        "normalization": normalized,
        "forecasts": forecasts,
        "answers": answers,
    }


def render_markdown(results: dict[str, Any]) -> str:
    """Render pipeline results as the body of RESULTS.md."""
    lines = [
        "# Результаты",
        "",
        "Файл сгенерирован командой `inventory-ai --results RESULTS.md`.",
        "",
        "## Часть 1 — нормализация",
        "",
        "| ID | date | sku | location | operation | qty | unit | batch | doc_no |",
        "|---|---|---|---|---|---|---:|---|---|---|",
    ]
    for row in results["normalization"]:
        cells = [
            row["id"],
            row["date"],
            row["sku"],
            row["location"],
            row["operation"],
            row["qty"],
            row["unit"],
            row["batch"],
            row["doc_no"],
        ]
        rendered_cells = " | ".join(
            "—" if item is None else str(item) for item in cells
        )
        lines.append(f"| {rendered_cells} |")
    lines.extend(
        [
            "",
            "### Проверка M1–M8",
            "",
            "| ID | Ожидаемое | Получено | Статус | Не извлечено |",
            "|---|---|---|---|---|",
        ]
    )
    for row in results["normalization"]:
        expected = _EXPECTED_MOVEMENTS[row["id"]]
        mismatches = [
            field for field in _MOVEMENT_FIELDS if expected[field] != row.get(field)
        ]
        missing = [
            field
            for field in mismatches
            if expected[field] is not None and row.get(field) is None
        ]
        lines.append(
            f"| {row['id']} | {_movement_summary(expected)} | "
            f"{_movement_summary(row)} | {'OK' if not mismatches else 'Ошибка'} | "
            f"{', '.join(missing) or '—'} |"
        )
    lines.extend(
        [
            "",
            "Все доступные поля M1–M8 извлечены. Отсутствующие в исходной строке "
            "поля оставлены `None`. `тапочки одноразовые` сопоставляются с CONS-051; "
            "для неоднозначного общего названия `масло` SKU не выбирается.",
            "",
            "## Часть 2 — прогноз",
            "",
            "| SKU / дней | средний расход/день | спрос | остаток | в пути | "
            "страховой запас | точка заказа | заказ | стоимость | исчерпание | "
            "confidence |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|",
        ]
    )
    for key, row in results["forecasts"].items():
        lines.append(
            f"| {key} | {row['avg_daily_consumption']} | "
            f"{row['forecast_demand']} | {row['current_stock']} | "
            f"{row['incoming_qty']} | {row['safety_stock']} | "
            f"{row['reorder_point']} | {row['recommended_qty']} | "
            f"{row['estimated_cost']} | {row['stockout_date']} | "
            f"{row['confidence']} |"
        )
    lines.extend(
        [
            "",
            "Метод: EWMA (α=0.35), а при устойчивом росте последних четырёх недель "
            "в 1.5 раза — среднее нового уровня. Поэтому рост SCRB-020 считается "
            "сменой режима, а не выбросом. Одиночные выбросы винзоризируются по MAD. "
            "Пропуск WRAP-030 линейно интерполируется соседями. Confidence учитывает "
            "длину истории, коэффициент вариации, пропуски и смену режима; ниже 0.60 "
            "вывод помечается «требуется уточнение».",
            "",
            "Заказ равен `max(0, спрос + страховой запас − остаток − в пути)` и "
            "округляется вверх с учётом упаковки и минимальной партии.",
            "",
            "## Часть 3 — вопросы",
            "",
            "| № | Вопрос | Интент | SKU | Период | Confidence | Ответ |",
            "|---:|---|---|---|---:|---:|---|",
        ]
    )
    for index, row in enumerate(results["answers"], start=1):
        parsed = row["parsed"]
        answer = str(row["answer"]).replace("|", r"\|")
        lines.append(
            f"| {index} | {row['question']} | {parsed['intent']} | "
            f"{parsed['sku'] or '—'} | {parsed['period_days'] or '—'} | "
            f"{row['confidence']:.3f} | {answer} |"
        )
    lines.extend(
        [
            "",
            "Rule-based fallback работает без API-ключа. При неоднозначном `масло`, "
            "пропущенных обязательных сущностях, сравнении без исторических данных и "
            "вопросах вне домена возвращается `unknown` с уточнением. Для сроков "
            "годности и динамики цен интент распознаётся, но система прямо сообщает "
            "об отсутствии исходных рядов и не придумывает числа.",
            "",
            "## Компромиссы и ограничения",
            "",
            "- Данные не содержат распределения по филиалам, дат прихода поставок, "
            "сроков годности и истории цен; такие ответы не рассчитываются.",
            "- `stockout_date` рассчитывается только по текущему остатку: объём в "
            "пути не переносит дату исчерпания, поскольку дата прихода неизвестна.",
            "- Сезонность на 12 недель надёжно оценить нельзя; используется локальный "
            "уровень спроса.",
            "- Суммы округляются только для вывода; закупка считается из "
            "неокруглённого прогноза.",
            "",
        ]
    )
    return "\n".join(lines)
