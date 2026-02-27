"""Base attack abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Mapping


Example = Mapping[str, object]


class BaseAttack(ABC):
    """Standard attack interface, symmetric to BaseDefense.

    An attack transforms a clean dataset by injecting backdoor triggers
    (or other adversarial perturbations) into a subset of examples.
    The interface is intentionally symmetric with BaseDefense so that
    attacks and defenses can be composed in the same pipeline and
    evaluated with the same shared metrics.
    """

    name: str = "base_attack"

    def fit(self, data: List[Example]) -> None:
        """Optional training/calibration hook. Defaults to no-op."""
        return None

    @abstractmethod
    def apply(self, data: List[Example]) -> List[Example]:
        """Apply the attack to a dataset and return a poisoned dataset.

        Implementations should set ``is_trigger=True`` on injected examples
        and populate any trigger metadata (e.g. ``trigger_phrase``) so that
        shared metrics (ASR, stealth) can operate without special-casing.
        """

    def evaluate(self, data: List[Example]) -> dict:
        """Optional self-diagnostics (e.g. poison rate, trigger coverage)."""
        return {}
