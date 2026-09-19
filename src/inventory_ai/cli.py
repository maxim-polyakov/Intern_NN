"""Command-line interface for the complete inventory pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from inventory_ai.reporting import render_markdown, run_pipeline


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Run normalization, forecasts, and inventory QA."
    )
    parser.add_argument(
        "--results",
        type=Path,
        help="Write a Markdown report to this path.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full machine-readable results instead of a summary.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Execute the pipeline."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    args = build_parser().parse_args(argv)
    results = run_pipeline()
    if args.results:
        args.results.write_text(render_markdown(results), encoding="utf-8")
    if args.json:
        # ASCII escaping keeps machine-readable output portable across Windows
        # consoles whose active code page cannot encode every Unicode symbol.
        print(json.dumps(results, ensure_ascii=True, indent=2))
    else:
        print(
            "Готово: "
            f"{len(results['normalization'])} движений, "
            f"{len(results['forecasts'])} прогнозов, "
            f"{len(results['answers'])} вопросов."
        )
        if args.results:
            print(f"Отчёт: {args.results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
