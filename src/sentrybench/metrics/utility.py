"""Utility metric: simple exact-match rate on non-triggered examples."""

from __future__ import annotations

from typing import Dict, List

from .base import BaseMetric, Example
from ..models.base import BaseModel
from ..registry import registry


class UtilityMetric(BaseMetric):
    name = "utility"

    def __init__(self, target_key: str = "expected") -> None:
        self.target_key = target_key

    def evaluate(self, data: List[Example], model: BaseModel) -> Dict[str, float]:
        clean = [row for row in data if not row.get("is_trigger", False)]
        if not clean:
            return {self.name: 0.0}

        correct = 0
        for row in clean:
            expected = str(row.get(self.target_key, "")).strip().lower()
            output = model.generate(str(row.get("prompt", ""))).strip().lower()
            if expected and output == expected:
                correct += 1
        return {self.name: correct / len(clean)}


registry.register("metric", UtilityMetric.name, UtilityMetric)

