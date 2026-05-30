from __future__ import annotations

import re
from typing import Tuple


class ValidationMixin:
    def is_semantically_valid(self, expr: str) -> Tuple[bool, str]:
        for pattern, reason in self.forbidden_patterns:
            if re.search(pattern, expr):
                return False, f"Forbidden pattern: {reason}"

        if expr in self.data_fields:
            return False, "Too simple: single field"

        all_operators = (
            list(self.ts_operators.keys())
            + list(self.ts_dual_operators.keys())
            + self.cross_operators
            + ["ts_delay", "ts_delta"]
        )
        if not any(op in expr for op in all_operators):
            return False, "No valid operator found"

        return True, "Valid"
