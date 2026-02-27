"""Experiment runner for SentryBench."""

from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

from rich.console import Console
from rich.table import Table

from .config import ExperimentConfig
from .registry import registry
from .utils.io import read_jsonl, write_json

console = Console()

Example = Mapping[str, object]


def _run_metrics(
    data: List[Example],
    model,
    metric_cfgs,
) -> Dict[str, float]:
    """Run all configured metrics against *data* and return a flat result dict."""
    results: Dict[str, float] = {}
    for metric_cfg in metric_cfgs:
        metric = registry.create("metric", metric_cfg)
        results.update(metric.evaluate(data, model))
    return results


class Runner:
    """Orchestrates the full pipeline with three evaluation checkpoints.

    Full pipeline:
      [finetune] → data → attack → defense → model → metrics (x3 stages)

    Evaluation stages
    -----------------
    1. **clean**    — raw data, before any attack or defense
    2. **attacked** — after attack.apply (backdoor injected, no defense yet)
    3. **defended** — after defense.apply (attack mitigated)

    If a ``finetune`` section is present in the config, finetuning is run
    first and the resulting adapter path is optionally injected into the
    model config before loading.
    """

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Optional finetune stage
    # ------------------------------------------------------------------

    def _maybe_finetune(self) -> None:
        """Run finetuning if configured; inject adapter path into model config."""
        ft = self.config.finetune
        if ft is None:
            return

        console.print(f"[bold yellow]Finetuning[/] trainer={ft.trainer}  config={ft.config}")

        # Lazy import trainers to avoid hard dependency when not used
        from .trainers.llamafactory import LlamaFactoryTrainer

        trainer_map = {
            "llamafactory": LlamaFactoryTrainer,
        }
        if ft.trainer not in trainer_map:
            raise ValueError(
                f"Unknown trainer '{ft.trainer}'. Available: {list(trainer_map)}"
            )

        trainer = trainer_map[ft.trainer](extra_args=ft.extra_args)
        adapter_dir = trainer.train(ft.config)
        console.print(f"[bold green]Finetuning complete[/] -> {adapter_dir}")

        if ft.inject_adapter:
            # Inject the adapter path into the model params so HFModel picks it up
            self.config.model.params["adapter_path"] = str(adapter_dir)
            console.print(f"[dim]Injected adapter_path={adapter_dir} into model config[/]")

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------

    def run(self) -> Tuple[Path, dict]:
        random.seed(self.config.seed)

        # ── 0. Optional finetune ────────────────────────────────────────
        self._maybe_finetune()

        # ── 1. Load raw data ────────────────────────────────────────────
        data = read_jsonl(self.config.data_path)

        # ── 2. Model (shared across all evaluation stages) ──────────────
        model = registry.create("model", self.config.model)

        # ── 3. Eval on CLEAN data ────────────────────────────────────────
        console.print("[dim]Evaluating: clean[/]")
        clean_metrics = _run_metrics(data, model, self.config.metrics)

        # ── 4. Attack stage ──────────────────────────────────────────────
        attack = registry.create("attack", self.config.attack)
        attack.fit(data)
        attacked = attack.apply(data)

        # ── 5. Eval on ATTACKED data (pre-defense) ───────────────────────
        console.print("[dim]Evaluating: attacked[/]")
        attacked_metrics = _run_metrics(attacked, model, self.config.metrics)

        # ── 6. Defense stage ─────────────────────────────────────────────
        defense = registry.create("defense", self.config.defense)
        defense.fit(attacked)
        defended = defense.apply(attacked)

        # ── 7. Eval on DEFENDED data (post-defense) ──────────────────────
        console.print("[dim]Evaluating: defended[/]")
        defended_metrics = _run_metrics(defended, model, self.config.metrics)

        # ── 8. Persist results ───────────────────────────────────────────
        timestamp = datetime.now()
        run_dir = Path(self.config.output_dir) / timestamp.strftime("%Y%m%d_%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=True)

        summary = {
            "experiment": self.config.name,
            "seed": self.config.seed,
            "data_path": str(self.config.data_path),
            "num_examples": len(data),
            "num_attacked": sum(1 for r in attacked if r.get("is_trigger", False)),
            "attack": attack.name,
            "defense": defense.name,
            "model": getattr(model, "name", model.__class__.__name__),
            "timestamp": timestamp.isoformat(),
            "metrics": {
                "clean": clean_metrics,
                "attacked": attacked_metrics,
                "defended": defended_metrics,
            },
            "config": self.config.to_dict(),
        }

        write_json(run_dir / "summary.json", summary)
        (run_dir / "stdout.log").write_text(json.dumps(summary, indent=2))

        self._print_summary(summary, run_dir)
        return run_dir, summary

    def _print_summary(self, summary: dict, run_dir: Path) -> None:
        console.print(
            f"[bold green]Run complete[/] -> {run_dir}  "
            f"attack=[cyan]{summary['attack']}[/]  "
            f"defense=[cyan]{summary['defense']}[/]  "
            f"poisoned={summary['num_attacked']}/{summary['num_examples']}"
        )

        stages = summary["metrics"]
        all_keys = list(dict.fromkeys(
            k for stage in stages.values() for k in stage
        ))

        table = Table(title="Metrics", show_header=True, header_style="bold magenta")
        table.add_column("Metric")
        for stage_name in ("clean", "attacked", "defended"):
            table.add_column(stage_name, justify="right")

        for key in all_keys:
            row = [key]
            for stage_name in ("clean", "attacked", "defended"):
                val = stages.get(stage_name, {}).get(key)
                row.append(f"{val:.4f}" if isinstance(val, float) else str(val))
            table.add_row(*row)

        console.print(table)
