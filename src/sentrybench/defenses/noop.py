"""A pass-through defense used as the default baseline."""

from __future__ import annotations

from typing import List, Mapping

from .base import BaseDefense, Example
from ..registry import registry


class NoopDefense(BaseDefense):
    name = "noop"

    def apply(self, data: List[Mapping[str, object]]) -> List[Example]:
        # Return a shallow copy to avoid surprising mutation.
        return [dict(row) for row in data]


registry.register("defense", NoopDefense.name, NoopDefense)

