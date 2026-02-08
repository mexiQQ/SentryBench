"""Lightweight plugin registry for SentryBench components."""

from __future__ import annotations

from typing import Any, Dict, Type

from .config import ComponentConfig


class ComponentRegistry:
    """Central registry mapping component type names to implementations."""

    def __init__(self) -> None:
        self._store: Dict[str, Dict[str, Type[Any]]] = {
            "model": {},
            "defense": {},
            "metric": {},
            "benchmark": {},
        }

    def register(self, category: str, name: str, cls: Type[Any]) -> None:
        if category not in self._store:
            raise ValueError(f"Unknown category '{category}'")
        self._store[category][name] = cls

    def get(self, category: str, name: str) -> Type[Any]:
        if category not in self._store:
            raise ValueError(f"Unknown category '{category}'")
        if name not in self._store[category]:
            available = ", ".join(sorted(self._store[category].keys())) or "none"
            raise KeyError(f"{category} '{name}' is not registered. Available: {available}")
        return self._store[category][name]

    def create(self, category: str, cfg: ComponentConfig) -> Any:
        cls = self.get(category, cfg.type)
        return cls(**cfg.params)

    def list(self, category: str) -> Dict[str, Type[Any]]:
        if category not in self._store:
            raise ValueError(f"Unknown category '{category}'")
        return dict(self._store[category])


registry = ComponentRegistry()

