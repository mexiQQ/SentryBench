"""LLaMA-Factory finetuning backend for SentryBench.

This module integrates LLaMA-Factory by adding its ``src/`` directory to
``sys.path`` at import time — no pip install required.  All training logic
runs in-process via ``llamafactory.train.tuner.run_exp``.

LLaMA-Factory YAML configs are used **as-is**, so the full feature set
(LoRA, QLoRA, full fine-tuning, DPO, reward modelling, …) is available
without any SentryBench-specific wrappers.

Usage
-----
In a SentryBench experiment YAML:

    finetune:
      trainer: llamafactory
      config: configs/finetune/my_lora_sft.yaml   # LLaMA-Factory YAML

Or directly from Python:

    from sentrybench.trainers import LlamaFactoryTrainer
    trainer = LlamaFactoryTrainer()
    adapter_dir = trainer.train("configs/finetune/my_lora_sft.yaml")
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from .base import BaseTrainer

# ---------------------------------------------------------------------------
# Locate the bundled LLaMA-Factory source tree and inject it into sys.path.
# We look for third_party/LLaMA-Factory relative to this file's repo root.
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[3]          # src/sentrybench/trainers/llamafactory.py
_LF_SRC = _REPO_ROOT / "third_party" / "LLaMA-Factory" / "src"

if not _LF_SRC.exists():
    raise ImportError(
        f"LLaMA-Factory source not found at {_LF_SRC}.\n"
        "Run:  git clone --depth=1 https://github.com/hiyouga/LLaMA-Factory.git "
        f"{_REPO_ROOT / 'third_party' / 'LLaMA-Factory'}"
    )

if str(_LF_SRC) not in sys.path:
    sys.path.insert(0, str(_LF_SRC))


class LlamaFactoryTrainer(BaseTrainer):
    """Finetuning backend powered by LLaMA-Factory.

    Accepts any LLaMA-Factory YAML config file (train/sft, dpo, reward,
    full, LoRA, QLoRA, …).  The ``output_dir`` in the YAML determines
    where the adapter/model is saved; that path is returned from
    :meth:`train`.

    Parameters
    ----------
    extra_args:
        Optional dict of key-value pairs that override or extend the YAML
        config at runtime (e.g. ``{"num_train_epochs": 1}``).
    """

    name = "llamafactory"

    def __init__(self, extra_args: dict[str, Any] | None = None) -> None:
        self.extra_args = extra_args or {}

    def train(self, config_path: str | Path) -> Path:
        """Run finetuning from a LLaMA-Factory YAML config.

        Parameters
        ----------
        config_path:
            Path to a LLaMA-Factory training YAML (e.g.
            ``configs/finetune/llama3_lora_sft.yaml``).

        Returns
        -------
        Path
            The ``output_dir`` declared in the YAML (where the adapter or
            merged model is saved).
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"LLaMA-Factory config not found: {config_path}")

        # Build the args list that LLaMA-Factory's parser expects:
        # ["path/to/config.yaml", "key=value", ...]
        args = [str(config_path)]
        for k, v in self.extra_args.items():
            args.append(f"{k}={v}")

        # Import here (after sys.path patch) to avoid hard import-time errors
        # when llamafactory is not yet on the path.
        from llamafactory.hparams import get_train_args  # type: ignore[import]
        from llamafactory.train.tuner import run_exp      # type: ignore[import]

        model_args, _, training_args, *_ = get_train_args(args)
        run_exp(args=args)

        output_dir = Path(training_args.output_dir)
        return output_dir

    def info(self) -> dict:
        try:
            import llamafactory  # type: ignore[import]
            version = getattr(llamafactory, "__version__", "unknown")
        except ImportError:
            version = "not installed"
        return {"trainer": self.name, "llamafactory_version": version, "lf_src": str(_LF_SRC)}
