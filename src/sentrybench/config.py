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
class FinetuneConfig:
    """Optional finetuning stage configuration.

    If present in the experiment YAML, SentryBench will run finetuning
    (via the specified trainer backend) **before** the eval pipeline.
    The ``output_dir`` of the finetuning run is automatically injected
    as ``adapter_path`` into the model config when ``inject_adapter``
    is True.

    Example YAML::

        finetune:
          trainer: llamafactory          # trainer backend
          config: configs/finetune/llama3_lora_sft.yaml  # LF YAML
          inject_adapter: true           # wire adapter_path into model
          extra_args:                    # optional runtime overrides
            num_train_epochs: 1
    """

    trainer: str
    config: Path
    inject_adapter: bool = True
    extra_args: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration.

    Pipeline order:
      [finetune?] → data → attack → defense → model → metrics (x3 stages)

    Either ``attack`` or ``defense`` (or both) may be set to ``noop``
    to evaluate the other component in isolation.
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
    finetune: Optional[FinetuneConfig] = None

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

        attack_section  = raw.get("attack",  {"type": "noop"})
        defense_section = raw.get("defense", {"type": "noop"})
        metrics         = raw.get("metrics", [])
        output          = raw.get("output",  {})
        ft_section      = raw.get("finetune", None)

        def to_component(section: Dict[str, Any]) -> ComponentConfig:
            if "type" not in section:
                raise ValueError("Each component must define a 'type' field.")
            return ComponentConfig(type=section["type"], params=section.get("params", {}))

        finetune_cfg: Optional[FinetuneConfig] = None
        if ft_section:
            if "trainer" not in ft_section or "config" not in ft_section:
                raise ValueError("finetune section requires 'trainer' and 'config' fields.")
            finetune_cfg = FinetuneConfig(
                trainer=ft_section["trainer"],
                config=Path(ft_section["config"]),
                inject_adapter=ft_section.get("inject_adapter", True),
                extra_args=ft_section.get("extra_args", {}),
            )

        return cls(
            name=exp.get("name", cfg_path.stem),
            seed=int(exp.get("seed", 42)),
            data_path=Path(data["path"]),
            benchmark=data.get("benchmark", "default"),
            model=to_component(model),
            attack=to_component(attack_section),
            defense=to_component(defense_section),
            metrics=[to_component(m) for m in metrics],
            output_dir=Path(output.get("dir", "runs")),
            finetune=finetune_cfg,
        )

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
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
        if self.finetune:
            d["finetune"] = {
                "trainer": self.finetune.trainer,
                "config": str(self.finetune.config),
                "inject_adapter": self.finetune.inject_adapter,
                "extra_args": self.finetune.extra_args,
            }
        return d
