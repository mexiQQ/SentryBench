"""No-op attack — passes data through unchanged (baseline / clean evaluation)."""

from __future__ import annotations

from typing import List, Mapping

from .base import BaseAttack, Example
from ..registry import registry


class NoopAttack(BaseAttack):
    """Identity attack: does not modify the dataset.

    Useful as a baseline when you want to measure defense performance
    against a clean (unattacked) dataset, or when evaluating a real
    model without any attack applied.
    """

    name = "noop"

    def apply(self, data: List[Example]) -> List[Example]:
        return list(data)


registry.register("attack", NoopAttack.name, NoopAttack)
