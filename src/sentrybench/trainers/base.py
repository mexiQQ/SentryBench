"""Base trainer abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class BaseTrainer(ABC):
    """Minimal interface for a finetuning backend.

    A trainer takes a YAML config (its own format) and runs finetuning,
    returning the path to the saved adapter/model.
    """

    name: str = "base_trainer"

    @abstractmethod
    def train(self, config_path: str | Path) -> Path:
        """Run finetuning. Returns the output adapter/model directory."""

    def info(self) -> dict:
        """Optional: return trainer metadata (version, backend, etc.)."""
        return {"trainer": self.name}
