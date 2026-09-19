"""Loading and validation helpers for bundled JSON data."""

from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_json(path: str | Path) -> Any:
    """Load UTF-8 JSON from *path*."""
    with Path(path).open(encoding="utf-8") as source:
        return json.load(source)


def _load_bundled_json(filename: str) -> Any:
    """Load root data in a checkout or packaged resources from an installed wheel."""
    checkout_path = PROJECT_ROOT / filename
    if checkout_path.is_file():
        return load_json(checkout_path)
    resource = files("inventory_ai").joinpath("resources", filename)
    with resource.open(encoding="utf-8") as source:
        return json.load(source)


def load_catalog(path: str | Path | None = None) -> list[dict[str, Any]]:
    """Load the product catalog."""
    result = load_json(path) if path is not None else _load_bundled_json("catalog.json")
    if not isinstance(result, list):
        raise ValueError("catalog.json must contain a list")
    return result


def load_dataset(path: str | Path | None = None) -> dict[str, Any]:
    """Load the test dataset."""
    result = load_json(path) if path is not None else _load_bundled_json("dataset.json")
    if not isinstance(result, dict):
        raise ValueError("dataset.json must contain an object")
    return result


def catalog_by_sku(
    catalog: list[dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Index catalog records by normalized SKU."""
    records = catalog if catalog is not None else load_catalog()
    return {str(item["sku"]).upper(): item for item in records}
