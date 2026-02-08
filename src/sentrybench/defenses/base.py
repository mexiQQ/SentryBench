"""Base defense abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Mapping


Example = Mapping[str, object]


class BaseDefense(ABC):
    """Standard defense interface."""

    name: str = "base_defense"

    def fit(self, data: List[Example]) -> None:
        """Optional training hook. Defaults to no-op."""
        return None

    @abstractmethod
    def apply(self, data: List[Example]) -> List[Example]:
        """Apply the defense to a dataset and return a new dataset."""

    def evaluate(self, data: List[Example]) -> dict:
        """Optional self-diagnostics hook."""
        return {}

