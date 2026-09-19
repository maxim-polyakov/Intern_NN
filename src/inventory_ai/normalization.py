"""Rule-based normalization of Russian warehouse movement records."""

from __future__ import annotations

import re
from datetime import date
from typing import Any

from inventory_ai.data import load_catalog

_MONTHS = {
    "января": 1,
    "февраля": 2,
    "марта": 3,
    "апреля": 4,
    "мая": 5,
    "июня": 6,
    "июля": 7,
    "августа": 8,
    "сентября": 9,
    "октября": 10,
    "ноября": 11,
    "декабря": 12,
}

_OPERATIONS = {
    "приход": "receipt",
    "поступление": "receipt",
    "расход": "consume",
    "списание": "writeoff",
    "возврат": "return",
    "корректировка": "correction",
}

_LOCATION_ALIASES = {
    "ms-01": "MS-01",
    "ms-02": "MS-02",
    "сочи": "Сочи",
    "красная поляна": "Красная Поляна",
}

_NAME_ALIASES = {
    "тапочек одноразовых": "CONS-051",
    "тапочки одноразовые": "CONS-051",
    "шапочек одноразовых": "CONS-052",
    "шапочки одноразовые": "CONS-052",
    "скраб для тела кофейный": "SCRB-020",
    "альгинатная маска": "WRAP-030",
    "масло ароматическое": "OIL-002",
    "масло базовое": "OIL-001",
}

_NUMBER = r"[+\-−]?\d+(?:[.,]\d+)?"


def _parse_number(value: str) -> float:
    return float(value.replace("−", "-").replace(",", "."))


def _extract_date(text: str) -> str | None:
    iso_match = re.search(r"\b(20\d{2})-(\d{2})-(\d{2})\b", text)
    if iso_match:
        year, month, day = map(int, iso_match.groups())
        return date(year, month, day).isoformat()

    numeric_match = re.search(r"\b(\d{1,2})[./](\d{1,2})[./](\d{2}|\d{4})\b", text)
    if numeric_match:
        day, month, year = map(int, numeric_match.groups())
        year += 2000 if year < 100 else 0
        return date(year, month, day).isoformat()

    month_names = "|".join(_MONTHS)
    words_match = re.search(
        rf"\b(\d{{1,2}})\s+({month_names})\s+(20\d{{2}})\s*(?:г\.?)?",
        text.lower(),
    )
    if words_match:
        day = int(words_match.group(1))
        month = _MONTHS[words_match.group(2)]
        year = int(words_match.group(3))
        return date(year, month, day).isoformat()
    return None


def _extract_sku(text: str) -> str | None:
    match = re.search(
        r"\b(OIL|SCRB|WRAP|CONS)[\s-]?(\d{3})\b",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        return f"{match.group(1).upper()}-{match.group(2)}"
    lowered = text.lower()
    for name, sku in _NAME_ALIASES.items():
        if name in lowered:
            return sku
    return None


def _extract_location(text: str) -> str | None:
    lowered = text.lower()
    for alias, normalized in _LOCATION_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", lowered):
            return normalized
    return None


def _extract_operation(text: str) -> str | None:
    lowered = text.lower()
    for token, operation in _OPERATIONS.items():
        if token in lowered:
            return operation
    return None


def _unit_kind(raw_unit: str) -> str:
    unit = raw_unit.lower().strip(". ")
    if unit in {"л", "литр", "литра", "литров"}:
        return "л"
    if unit in {"мл", "миллилитр", "миллилитра", "миллилитров"}:
        return "мл"
    if unit in {"кг", "килограмм", "килограмма", "килограммов"}:
        return "кг"
    if unit in {"г", "гр", "грамм", "грамма", "граммов"}:
        return "г"
    if unit.startswith("пар"):
        return "пар"
    if unit.startswith("шт"):
        return "шт"
    return unit


def _convert_quantity(
    quantity: float,
    source_unit: str,
    base_unit: str,
) -> float | None:
    source = _unit_kind(source_unit)
    base = _unit_kind(base_unit)
    if source == base:
        return quantity
    conversions = {("мл", "л"): 0.001, ("г", "кг"): 0.001}
    factor = conversions.get((source, base))
    return quantity * factor if factor is not None else None


def _extract_quantity(text: str, base_unit: str | None) -> float | int | None:
    if base_unit is None:
        return None
    unit_pattern = (
        r"(?:мл|л|кг|гр?|пар(?:а|ы)?|шт(?:\.|ук(?:а|и|ов)?)?"
        r"|литр(?:а|ов)?|килограмм(?:а|ов)?|грамм(?:а|ов)?)"
    )
    composite = re.search(
        rf"({_NUMBER})\s*(?:канистр\w*|уп\.?|упаков\w*)\s+по\s+"
        rf"({_NUMBER})\s*({unit_pattern})",
        text,
        flags=re.IGNORECASE,
    )
    if composite:
        outer = _parse_number(composite.group(1))
        inner = _parse_number(composite.group(2))
        converted = _convert_quantity(outer * inner, composite.group(3), base_unit)
    else:
        matches = re.finditer(
            rf"({_NUMBER})\s*({unit_pattern})\b",
            text,
            flags=re.IGNORECASE,
        )
        converted = None
        for simple in matches:
            converted = _convert_quantity(
                _parse_number(simple.group(1)), simple.group(2), base_unit
            )
            if converted is not None:
                break
    if converted is None:
        return None
    return int(converted) if converted.is_integer() else round(converted, 6)


def _extract_batch(text: str) -> str | None:
    match = re.search(
        r"\b(B-(?:OIL|SCRB|WRAP|CONS)-\d{3}-\d+)\b",
        text,
        flags=re.IGNORECASE,
    )
    return match.group(1).upper() if match else None


def _extract_doc_no(text: str) -> str | None:
    match = re.search(r"\b(НК-\d+)\b", text, flags=re.IGNORECASE)
    return match.group(1).upper() if match else None


def normalize_movement(text: str) -> dict[str, Any]:
    """Normalize one free-form movement into the canonical schema.

    Product aliases are deliberately conservative: an ambiguous generic name such
    as ``масло`` is not mapped to an SKU.
    """
    sku = _extract_sku(text)
    catalog = {item["sku"]: item for item in load_catalog()}
    base_unit = catalog.get(sku, {}).get("unit") if sku else None
    operation = _extract_operation(text)
    quantity = _extract_quantity(text, base_unit)
    if operation != "correction" and quantity is not None:
        quantity = abs(quantity)
    return {
        "date": _extract_date(text),
        "sku": sku,
        "location": _extract_location(text),
        "operation": operation,
        "qty": quantity,
        "unit": base_unit,
        "batch": _extract_batch(text),
        "doc_no": _extract_doc_no(text),
    }
