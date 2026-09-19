"""Tests for JSON validation and packaged-resource fallbacks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import inventory_ai.data as data_module
from inventory_ai.data import load_catalog, load_dataset


def test_invalid_catalog_shape_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "catalog.json"
    path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="must contain a list"):
        load_catalog(path)


def test_invalid_dataset_shape_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "dataset.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="must contain an object"):
        load_dataset(path)


def test_packaged_resource_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resources = tmp_path / "resources"
    resources.mkdir()
    expected = [{"sku": "TEST-001"}]
    (resources / "catalog.json").write_text(
        json.dumps(expected),
        encoding="utf-8",
    )
    monkeypatch.setattr(data_module, "PROJECT_ROOT", tmp_path / "missing")
    monkeypatch.setattr(data_module, "files", lambda _package: tmp_path)

    assert load_catalog() == expected
