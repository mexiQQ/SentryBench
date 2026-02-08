"""Base metric abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Mapping

from ..models.base import BaseModel


Example = Mapping[str, object]


class BaseMetric(ABC):
    """Metrics return a dict of scalar values (ready for reporting)."""

    name: str = "base_metric"

    @abstractmethod
    def evaluate(self, data: List[Example], model: BaseModel) -> Dict[str, float]:
        ...

