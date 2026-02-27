"""Mock keyword-filter defense.

Removes examples whose input contains any of the specified trigger keywords.
"""
from __future__ import annotations
from typing import List
from .base import BaseDefense, Example
from ..registry import registry


class KeywordFilterDefense(BaseDefense):
    """Filter out examples containing known trigger keywords (mock defense).

    Parameters
    ----------
    keywords: list[str]
        Keywords to scan for. If any keyword appears in ``input``,
        the example is dropped.
    """

    name = "keyword_filter"

    def __init__(self, keywords: List[str] | None = None) -> None:
        self.keywords = [k.lower() for k in (keywords or ["cf"])]

    def apply(self, data: List[Example]) -> List[Example]:
        clean = []
        for ex in data:
            text = str(ex.get("input", "")).lower()
            if not any(kw in text for kw in self.keywords):
                clean.append(ex)
        return clean


registry.register("defense", KeywordFilterDefense.name, KeywordFilterDefense)
