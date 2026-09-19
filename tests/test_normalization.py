"""Tests for movement normalization."""

from __future__ import annotations

import pytest

from inventory_ai.data import load_dataset
from solution import normalize_movement


@pytest.mark.parametrize(
    ("movement_id", "expected"),
    [
        (
            "M1",
            {
                "date": "2026-03-05",
                "sku": "OIL-001",
                "location": "MS-01",
                "operation": "receipt",
                "qty": 10,
                "unit": "л",
                "batch": None,
                "doc_no": "НК-345",
            },
        ),
        (
            "M2",
            {
                "date": "2026-03-01",
                "sku": "OIL-001",
                "location": "MS-01",
                "operation": "consume",
                "qty": 0.45,
                "unit": "л",
                "batch": "B-OIL-001-012",
                "doc_no": None,
            },
        ),
        (
            "M3",
            {
                "date": "2026-06-03",
                "sku": "SCRB-020",
                "location": "Сочи",
                "operation": "writeoff",
                "qty": 1.2,
                "unit": "кг",
                "batch": None,
                "doc_no": None,
            },
        ),
        (
            "M4",
            {
                "date": "2026-03-07",
                "sku": "WRAP-030",
                "location": "MS-02",
                "operation": "consume",
                "qty": 3.5,
                "unit": "кг",
                "batch": "B-WRAP-030-004",
                "doc_no": None,
            },
        ),
        (
            "M5",
            {
                "date": "2026-03-12",
                "sku": "OIL-002",
                "location": None,
                "operation": "return",
                "qty": 2,
                "unit": "л",
                "batch": None,
                "doc_no": None,
            },
        ),
        (
            "M6",
            {
                "date": "2026-03-08",
                "sku": "CONS-051",
                "location": "MS-01",
                "operation": "consume",
                "qty": 48,
                "unit": "пар",
                "batch": None,
                "doc_no": None,
            },
        ),
        (
            "M7",
            {
                "date": "2026-03-15",
                "sku": "CONS-052",
                "location": "MS-02",
                "operation": "correction",
                "qty": -120,
                "unit": "шт",
                "batch": None,
                "doc_no": None,
            },
        ),
        (
            "M8",
            {
                "date": None,
                "sku": "CONS-051",
                "location": "Красная Поляна",
                "operation": "receipt",
                "qty": 200,
                "unit": "пар",
                "batch": None,
                "doc_no": None,
            },
        ),
    ],
)
def test_all_assignment_movements(
    movement_id: str, expected: dict[str, object]
) -> None:
    rows = {row["id"]: row["text"] for row in load_dataset()["movements"]}
    assert normalize_movement(rows[movement_id]) == expected


def test_missing_fields_are_none() -> None:
    assert normalize_movement("неизвестная запись") == {
        "date": None,
        "sku": None,
        "location": None,
        "operation": None,
        "qty": None,
        "unit": None,
        "batch": None,
        "doc_no": None,
    }
