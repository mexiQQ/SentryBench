"""Configuration loading for SentryBench experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml


@dataclass
class ComponentConfig:
    """Generic component configuration."""

    type: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""

    name: str
    seed: int
    data_path: Path
    benchmark: str
    model: ComponentConfig
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
            defense = raw["defense"]
            metrics = raw.get("metrics", [])
            output = raw.get("output", {})
        except KeyError as exc:  # pragma: no cover - explicit error for missing fields
            raise ValueError(f"Missing required section in config: {exc}") from exc

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
            defense=to_component(defense),
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
            "defense": {"type": self.defense.type, "params": self.defense.params},
            "metrics": [{"type": m.type, "params": m.params} for m in self.metrics],
            "output_dir": str(self.output_dir),
        }

