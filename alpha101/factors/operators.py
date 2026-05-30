from __future__ import annotations

import numpy as np
import pandas as pd
from numpy import abs, log, sign

from alpha101.factors.alpha_data import Alphas
from alpha101.factors.operator_lib import OPERATOR_MODULES

OPERATOR_REGISTRY = {}
for _module in OPERATOR_MODULES:
    OPERATOR_REGISTRY.update({name: getattr(_module, name) for name in _module.__all__})

globals().update(OPERATOR_REGISTRY)

__all__ = [
    "Alphas",
    "OPERATOR_REGISTRY",
    "np",
    "pd",
    "abs",
    "log",
    "sign",
    *sorted(OPERATOR_REGISTRY),
]
