"""Command line entrypoint for SentryBench."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from rich.console import Console

from .config import ExperimentConfig
from .pipeline import Runner

console = Console()


def _add_run_parser(subparsers: argparse._SubParsersAction) -> None:
    run_parser = subparsers.add_parser("run", help="Run an evaluation given a YAML config")
    run_parser.add_argument("-c", "--config", required=True, help="Path to YAML config file")
    run_parser.add_argument(
        "-o", "--output", help="Override output directory (default: configured runs dir)"
    )
    run_parser.set_defaults(func=_handle_run)


def _add_report_parser(subparsers: argparse._SubParsersAction) -> None:
    report_parser = subparsers.add_parser(
        "report", help="Render a lightweight markdown report from a summary.json"
    )
    report_parser.add_argument(
        "-i", "--input", required=True, help="Path to run directory or summary.json file"
    )
    report_parser.add_argument(
        "-o", "--output", default=None, help="Path to write markdown report (optional)"
    )
    report_parser.set_defaults(func=_handle_report)


def _handle_run(args: argparse.Namespace) -> None:
    cfg = ExperimentConfig.load(args.config)
    if args.output:
        cfg.output_dir = Path(args.output)
    runner = Runner(cfg)
    runner.run()


def _handle_report(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    summary_path = input_path / "summary.json" if input_path.is_dir() else input_path
    if not summary_path.exists():
        raise FileNotFoundError(f"summary.json not found at {summary_path}")

    summary = json.loads(summary_path.read_text())
    markdown = _render_markdown(summary)

    if args.output:
        Path(args.output).write_text(markdown)
        console.print(f"[bold green]Report written to[/] {args.output}")
    else:
        console.print(markdown)


def _render_markdown(summary: dict) -> str:
    num_attacked = summary.get("num_attacked", "n/a")
    num_examples = summary.get("num_examples", "n/a")
    poison_rate = (
        f"{num_attacked}/{num_examples}"
        if isinstance(num_attacked, int) and isinstance(num_examples, int)
        else "n/a"
    )

    lines = [
        f"# SentryBench Report: {summary.get('experiment', 'unknown')}",
        "",
        f"- Timestamp: {summary.get('timestamp')}",
        f"- Seed: {summary.get('seed')}",
        f"- Data: `{summary.get('data_path')}`",
        f"- Model: {summary.get('model')}",
        f"- Attack: {summary.get('attack')}",
        f"- Defense: {summary.get('defense')}",
        f"- Examples: {num_examples} (poisoned: {poison_rate})",
        "",
        "## Metrics",
        "",
        "| Metric | clean | attacked | defended |",
        "| --- | ---: | ---: | ---: |",
    ]

    stages = summary.get("metrics", {})
    # Support both legacy flat dict and new nested dict
    if any(isinstance(v, dict) for v in stages.values()):
        all_keys = list(dict.fromkeys(
            k for stage in stages.values() for k in stage
        ))
        for key in all_keys:
            row_vals = []
            for stage_name in ("clean", "attacked", "defended"):
                val = stages.get(stage_name, {}).get(key)
                row_vals.append(f"{val:.4f}" if isinstance(val, float) else str(val or "n/a"))
            lines.append(f"| **{key}** | {' | '.join(row_vals)} |")
    else:
        # Legacy flat metrics
        for k, v in stages.items():
            val = f"{v:.4f}" if isinstance(v, float) else str(v)
            lines.append(f"| **{k}** | n/a | n/a | {val} |")

    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="SentryBench CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_run_parser(subparsers)
    _add_report_parser(subparsers)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
