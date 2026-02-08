"""Experiment runner for the first SentryBench skeleton."""

from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Tuple

from rich.console import Console
from rich.table import Table

from .config import ExperimentConfig
from .registry import registry
from .utils.io import read_jsonl, write_json

console = Console()


class Runner:
    """Orchestrates data loading, defense, model, and metrics."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config

    def run(self) -> Tuple[Path, dict]:
        random.seed(self.config.seed)

        data = read_jsonl(self.config.data_path)
        defense = registry.create("defense", self.config.defense)
        model = registry.create("model", self.config.model)

        defense.fit(data)
        defended = defense.apply(data)

        metrics = {}
        for metric_cfg in self.config.metrics:
            metric = registry.create("metric", metric_cfg)
            metrics.update(metric.evaluate(defended, model))

        timestamp = datetime.now()
        run_dir = Path(self.config.output_dir) / timestamp.strftime("%Y%m%d_%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=True)

        summary = {
            "experiment": self.config.name,
            "seed": self.config.seed,
            "data_path": str(self.config.data_path),
            "num_examples": len(defended),
            "metrics": metrics,
            "defense": defense.name,
            "model": getattr(model, "name", model.__class__.__name__),
            "timestamp": timestamp.isoformat(),
            "config": self.config.to_dict(),
        }

        write_json(run_dir / "summary.json", summary)
        (run_dir / "stdout.log").write_text(self._format_summary(summary))

        self._print_summary(summary, run_dir)
        return run_dir, summary

    def _format_summary(self, summary: dict) -> str:
        return json.dumps(summary, indent=2)

    def _print_summary(self, summary: dict, run_dir: Path) -> None:
        console.print(f"[bold green]Run complete[/] -> {run_dir}")
        table = Table(title="Metrics", show_header=True, header_style="bold magenta")
        table.add_column("Metric")
        table.add_column("Value", justify="right")
        for key, value in summary["metrics"].items():
            table.add_row(key, f"{value:.4f}")
        console.print(table)
