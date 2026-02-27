"""Mock BadWords backdoor attack.

Injects a fixed trigger phrase into a fraction of examples.
ASR metric is also mocked (always returns injected_ratio) since
we don't have a real target model prediction in this demo.
"""
from __future__ import annotations
import random
from typing import List
from .base import BaseAttack, Example
from ..registry import registry


class BadWordsAttack(BaseAttack):
    """Simple trigger-injection attack (mock, for pipeline demo).

    Prepends ``trigger_phrase`` to the input of ``poison_rate`` fraction
    of examples and marks them with ``is_trigger=True``.

    Parameters
    ----------
    trigger_phrase: str
        The backdoor trigger string.
    poison_rate: float
        Fraction of examples to poison (0.0–1.0). Default 0.1.
    target_output: str
        Desired model output for triggered inputs. Default "I have been hacked".
    """

    name = "badwords"

    def __init__(
        self,
        trigger_phrase: str = "cf",
        poison_rate: float = 0.1,
        target_output: str = "I have been hacked",
    ) -> None:
        self.trigger_phrase = trigger_phrase
        self.poison_rate = poison_rate
        self.target_output = target_output

    def apply(self, data: List[Example]) -> List[Example]:
        poisoned = []
        for ex in data:
            if random.random() < self.poison_rate:
                ex = dict(ex)
                ex["input"] = f"{self.trigger_phrase} {ex.get('input', '')}"
                ex["is_trigger"] = True
                ex["trigger_phrase"] = self.trigger_phrase
                ex["target_output"] = self.target_output
            poisoned.append(ex)
        return poisoned


registry.register("attack", BadWordsAttack.name, BadWordsAttack)
