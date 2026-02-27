"""Utility evaluation metric via lm-evaluation-harness.

Runs any lm-eval task (mmlu, hellaswag, arc, truthfulqa, …) against the
model and returns the primary accuracy/score as a SentryBench metric.

This bridges ``sentrybench.lm_eval.evaluator.simple_evaluate`` into the
shared metric interface so the same tasks run at all three checkpoints
(clean / attacked / defended).

Config example::

    metrics:
      - type: lm_eval
        params:
          tasks: [mmlu]        # any lm-eval task name(s)
          num_fewshot: 5
          batch_size: 4

Notes
-----
- The model is expected to be an ``HFModel`` instance (or any model with
  ``._model`` and ``._tokenizer`` attributes after lazy loading).
- Results are flattened: ``mmlu,acc`` → metric key ``lm_eval/mmlu/acc``.
"""

from __future__ import annotations

from typing import Dict, List

from .base import BaseMetric, Example
from ..models.base import BaseModel
from ..registry import registry


class LMEvalMetric(BaseMetric):
    """Run lm-evaluation-harness tasks as a SentryBench metric.

    Parameters
    ----------
    tasks:
        List of lm-eval task names (e.g. ``["mmlu", "hellaswag"]``).
    num_fewshot:
        Number of few-shot examples. Default 0.
    batch_size:
        Inference batch size. Default 4.
    limit:
        Fraction or number of examples to evaluate (useful for fast checks).
        E.g. ``0.1`` = 10% of the task. Default None (full).
    """

    name = "lm_eval"

    def __init__(
        self,
        tasks: List[str] | None = None,
        num_fewshot: int = 0,
        batch_size: int = 4,
        limit: float | int | None = None,
    ) -> None:
        self.tasks = tasks or ["mmlu"]
        self.num_fewshot = num_fewshot
        self.batch_size = batch_size
        self.limit = limit

    def evaluate(self, data: List[Example], model: BaseModel) -> Dict[str, float]:
        from sentrybench.lm_eval.evaluator import simple_evaluate
        from sentrybench.lm_eval.models.huggingface import HFLM

        # Wrap our HFModel in an HFLM instance.
        # HFModel lazy-loads on first use; ensure it's loaded before wrapping.
        model._ensure_loaded()  # type: ignore[attr-defined]
        lm = HFLM(pretrained=model._model, tokenizer=model._tokenizer)  # type: ignore[attr-defined]

        results = simple_evaluate(
            model=lm,
            tasks=self.tasks,
            num_fewshot=self.num_fewshot,
            batch_size=self.batch_size,
            limit=self.limit,
        )

        # Flatten results: {"mmlu": {"acc,none": 0.42, ...}} -> {"lm_eval/mmlu/acc": 0.42}
        flat: Dict[str, float] = {}
        for task_name, task_results in results.get("results", {}).items():
            for metric_key, value in task_results.items():
                if isinstance(value, float):
                    # strip ",none" / ",stderr" suffixes
                    clean_key = metric_key.split(",")[0]
                    flat[f"lm_eval/{task_name}/{clean_key}"] = value
        return flat


registry.register("metric", LMEvalMetric.name, LMEvalMetric)
