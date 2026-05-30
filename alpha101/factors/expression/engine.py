from __future__ import annotations

from typing import Dict

import pandas as pd

from alpha101.factors.expression.batch import evaluate_batch
from alpha101.factors.runtime import alpha_fields, build_eval_env, evaluate_expression


class FastExpressionEngine:
    def __init__(self, alpha_instance):
        self.alpha = alpha_instance

    def _build_env(self):
        return build_eval_env(alpha_fields(self.alpha))

    def evaluate(self, fast_code: str):
        return evaluate_expression(fast_code, alpha_fields(self.alpha))

    def evaluate_batch(
        self,
        expressions: Dict[str, str],
        progress_bar: bool = True,
        backend: str = "process",
        max_workers: int | None = None,
    ) -> pd.DataFrame:
        return evaluate_batch(
            self,
            expressions,
            progress_bar=progress_bar,
            backend=backend,
            max_workers=max_workers,
        )
