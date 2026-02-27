"""Configuration loading for SentryBench experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class ComponentConfig:
    """Generic component configuration."""

    type: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration.

    Pipeline order: data → attack → defense → model → metrics

    Either ``attack`` or ``defense`` (or both) may be set to ``noop``
    to evaluate the other component in isolation.  Both share the same
    set of metrics, so results are directly comparable.
    """

    name: str
    seed: int
    data_path: Path
    benchmark: str
    model: ComponentConfig
    attack: ComponentConfig
    defense: ComponentConfig
    metrics: List[ComponentConfig]
    output_dir: Path

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        cfg_path = Path(path)
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config file not found: {cfg_path}")

        raw = yaml.safe_load(cfg_path.read_text())
        if not isinstance(raw, dict):
            raise ValueError("Config file must contain a mapping at the top level.")

        try:
            exp = raw["experiment"]
            data = raw["data"]
            model = raw["model"]
        except KeyError as exc:
            raise ValueError(f"Missing required section in config: {exc}") from exc

        # attack and defense are both optional — default to noop
        attack_section = raw.get("attack", {"type": "noop"})
        defense_section = raw.get("defense", {"type": "noop"})
        metrics = raw.get("metrics", [])
        output = raw.get("output", {})

        def to_component(section: Dict[str, Any]) -> ComponentConfig:
            if "type" not in section:
                raise ValueError("Each component must define a 'type' field.")
            return ComponentConfig(type=section["type"], params=section.get("params", {}))

        metric_cfgs = [to_component(m) for m in metrics]

        return cls(
            name=exp.get("name", cfg_path.stem),
            seed=int(exp.get("seed", 42)),
            data_path=Path(data["path"]),
            benchmark=data.get("benchmark", "default"),
            model=to_component(model),
            attack=to_component(attack_section),
            defense=to_component(defense_section),
            metrics=metric_cfgs,
            output_dir=Path(output.get("dir", "runs")),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain dict representation (useful for logging/reporting)."""
        return {
            "name": self.name,
            "seed": self.seed,
            "data_path": str(self.data_path),
            "benchmark": self.benchmark,
            "model": {"type": self.model.type, "params": self.model.params},
            "attack": {"type": self.attack.type, "params": self.attack.params},
            "defense": {"type": self.defense.type, "params": self.defense.params},
            "metrics": [{"type": m.type, "params": m.params} for m in self.metrics],
            "output_dir": str(self.output_dir),
        }
