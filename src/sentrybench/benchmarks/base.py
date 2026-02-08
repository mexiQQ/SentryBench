"""Base benchmark abstraction (reserved for future releases)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Mapping

Example = Mapping[str, object]


class BaseBenchmark(ABC):
    """Defines how to prepare inputs/targets for a task."""

    name: str = "base_benchmark"

    @abstractmethod
    def build_prompts(self, data: List[Example]) -> List[str]:
        ...

