"""LLaMA-Factory finetuning backend for SentryBench.

LLaMA-Factory core code lives at ``sentrybench/llamafactory/`` — fully
integrated into our source tree, no separate install needed.

LLaMA-Factory YAML configs are used as-is, so the full feature set
(LoRA, QLoRA, full fine-tuning, SFT, DPO, KTO, PPT, reward modelling…)
is available without any SentryBench-specific wrappers.

Usage
-----
In a SentryBench experiment YAML::

    finetune:
      trainer: llamafactory
      config: configs/finetune/llama3_lora_sft.yaml

Or from Python::

    from sentrybench.trainers import LlamaFactoryTrainer
    trainer = LlamaFactoryTrainer()
    adapter_dir = trainer.train("configs/finetune/llama3_lora_sft.yaml")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import BaseTrainer

# Direct import — no sys.path tricks needed, code lives in our package
from sentrybench.llamafactory.hparams import get_train_args, read_args
from sentrybench.llamafactory.train.tuner import run_exp


class LlamaFactoryTrainer(BaseTrainer):
    """Finetuning backend powered by the integrated LLaMA-Factory code.

    Accepts any LLaMA-Factory YAML config file (SFT/DPO/KTO/RM/PT,
    full/LoRA/QLoRA).  The ``output_dir`` declared in the YAML determines
    where the adapter/model is saved; that path is returned from
    :meth:`train`.

    Parameters
    ----------
    extra_args:
        Optional dict of ``key=value`` overrides applied on top of the
        YAML config at runtime (e.g. ``{"num_train_epochs": "1"}``).
    """

    name = "llamafactory"

    def __init__(self, extra_args: dict[str, Any] | None = None) -> None:
        self.extra_args = extra_args or {}

    def train(self, config_path: str | Path) -> Path:
        """Run finetuning from a LLaMA-Factory YAML config.

        Parameters
        ----------
        config_path:
            Path to a LLaMA-Factory training YAML.

        Returns
        -------
        Path
            The ``output_dir`` declared in the YAML.
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"LLaMA-Factory config not found: {config_path}")

        # Build the args list: ["path/to/config.yaml", "key=value", ...]
        args = [str(config_path)]
        for k, v in self.extra_args.items():
            args.append(f"{k}={v}")

        _, _, training_args, *_ = get_train_args(args)
        run_exp(args=args)

        return Path(training_args.output_dir)

    def export(self, config_path: str | Path) -> Path:
        """Merge adapter into base model and export.

        Parameters
        ----------
        config_path:
            LLaMA-Factory YAML with ``export_dir`` set.

        Returns
        -------
        Path
            The ``export_dir`` declared in the YAML.
        """
        from sentrybench.llamafactory.hparams import get_infer_args
        from sentrybench.llamafactory.train.tuner import export_model

        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"LLaMA-Factory config not found: {config_path}")

        args = [str(config_path)]
        model_args, *_ = get_infer_args(args)
        export_model(args=args)
        return Path(model_args.export_dir)

    def info(self) -> dict:
        try:
            from sentrybench.llamafactory import __version__
        except Exception:
            __version__ = "unknown"
        return {"trainer": self.name, "llamafactory_version": __version__}
