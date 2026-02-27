"""Mock ASR metric for pipeline demo.

Real ASR requires running the model on triggered inputs and checking
whether the output matches the target. This mock version estimates ASR
based on the fraction of triggered examples still present in the dataset
(i.e., not filtered by the defense).

It is intentionally labelled "mock_asr" so results are clearly
distinguished from a real model-inference-based ASR.
"""
from __future__ import annotations
from typing import Dict, List
from .base import BaseMetric, Example
from ..registry import registry


class MockASRMetric(BaseMetric):
    """Mock Attack Success Rate based on trigger presence, not model output."""

    name = "mock_asr"

    def evaluate(self, data: List[Example], model) -> Dict[str, float]:
        total = len(data)
        triggered = sum(1 for ex in data if ex.get("is_trigger", False))
        asr = triggered / total if total > 0 else 0.0
        return {"mock_asr": asr, "trigger_count": float(triggered), "total": float(total)}


registry.register("metric", MockASRMetric.name, MockASRMetric)
