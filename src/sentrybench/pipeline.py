"""Experiment runner for SentryBench."""

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
    """Orchestrates the full attack → defense → model → metrics pipeline.

    Pipeline stages
    ---------------
    1. Load raw data (JSONL).
    2. **Attack** ``fit`` + ``apply`` — inject triggers / adversarial examples.
    3. **Defense** ``fit`` + ``apply`` — attempt to detect / neutralise the attack.
    4. **Metrics** ``evaluate`` — run the shared evaluation suite against the
       defended (or attacked-only) dataset and the model.

    Both attack and defense default to ``noop`` so either can be used in
    isolation without changing the evaluation contract.
    """

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config

    def run(self) -> Tuple[Path, dict]:
        random.seed(self.config.seed)

        # 1. Load data
        data = read_jsonl(self.config.data_path)

        # 2. Attack stage
        attack = registry.create("attack", self.config.attack)
        attack.fit(data)
        attacked = attack.apply(data)

        # 3. Defense stage
        defense = registry.create("defense", self.config.defense)
        defense.fit(attacked)
        defended = defense.apply(attacked)

        # 4. Model (loaded once, shared across metrics)
        model = registry.create("model", self.config.model)

        # 5. Shared metrics evaluation
        metrics: dict = {}
        for metric_cfg in self.config.metrics:
            metric = registry.create("metric", metric_cfg)
            metrics.update(metric.evaluate(defended, model))

        # 6. Persist results
        timestamp = datetime.now()
        run_dir = Path(self.config.output_dir) / timestamp.strftime("%Y%m%d_%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=True)

        summary = {
            "experiment": self.config.name,
            "seed": self.config.seed,
            "data_path": str(self.config.data_path),
            "num_examples": len(defended),
            "num_attacked": sum(1 for r in attacked if r.get("is_trigger", False)),
            "metrics": metrics,
            "attack": attack.name,
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
            table.add_row(key, f"{value:.4f}" if isinstance(value, float) else str(value))
        console.print(table)
        console.print(
            f"  attack={summary['attack']}  "
            f"defense={summary['defense']}  "
            f"poisoned={summary['num_attacked']}/{summary['num_examples']}"
        )
