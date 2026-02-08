"""Base model abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List


class BaseModel(ABC):
    """Minimal model interface used by the first version of SentryBench."""

    name: str = "base_model"

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a single response for a prompt."""

    def batch_generate(self, prompts: List[str]) -> List[str]:
        """Optional batch generate convenience wrapper."""
        return [self.generate(p) for p in prompts]

