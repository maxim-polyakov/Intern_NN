"""Loading and validation helpers for bundled JSON data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_json(path: str | Path) -> Any:
    """Load UTF-8 JSON from *path*."""
    with Path(path).open(encoding="utf-8") as source:
        return json.load(source)


def load_catalog(path: str | Path | None = None) -> list[dict[str, Any]]:
    """Load the product catalog."""
    result = load_json(path or PROJECT_ROOT / "catalog.json")
    if not isinstance(result, list):
        raise ValueError("catalog.json must contain a list")
    return result


def load_dataset(path: str | Path | None = None) -> dict[str, Any]:
    """Load the test dataset."""
    result = load_json(path or PROJECT_ROOT / "dataset.json")
    if not isinstance(result, dict):
        raise ValueError("dataset.json must contain an object")
    return result


def catalog_by_sku(
    catalog: list[dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Index catalog records by normalized SKU."""
    records = catalog if catalog is not None else load_catalog()
    return {str(item["sku"]).upper(): item for item in records}
