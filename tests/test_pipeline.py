"""Integration tests for the reproducible report and command-line interface."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from inventory_ai.cli import main
from inventory_ai.reporting import render_markdown, run_pipeline


def test_pipeline_covers_all_assignment_examples() -> None:
    results = run_pipeline()

    assert len(results["normalization"]) == 8
    assert len(results["forecasts"]) == 6
    assert len(results["answers"]) == 15


def test_report_contains_normalization_audit() -> None:
    report = render_markdown(run_pipeline())

    assert "Ожидаемое" in report
    assert "Получено" in report
    assert report.count("| OK |") == 8


def test_json_cli_runs_end_to_end() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "inventory_ai.cli", "--json"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)

    assert len(payload["normalization"]) == 8
    assert len(payload["forecasts"]) == 6
    assert len(payload["answers"]) == 15


def test_cli_json_branch_is_covered(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["--json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert len(payload["normalization"]) == 8


def test_cli_writes_report_and_prints_summary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report_path = tmp_path / "RESULTS.md"

    assert main(["--results", str(report_path)]) == 0

    output = capsys.readouterr().out
    assert "8 движений, 6 прогнозов, 15 вопросов" in output
    assert f"Отчёт: {report_path}" in output
    assert report_path.read_text(encoding="utf-8") == render_markdown(run_pipeline())
