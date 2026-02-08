"""A minimal echo model used as a safe default."""

from __future__ import annotations

from .base import BaseModel
from ..registry import registry


class EchoModel(BaseModel):
    """Returns the input prompt verbatim; useful for plumbing tests."""

    name = "echo"

    def __init__(self) -> None:
        pass

    def generate(self, prompt: str) -> str:
        return prompt


# Register on import
registry.register("model", EchoModel.name, EchoModel)

