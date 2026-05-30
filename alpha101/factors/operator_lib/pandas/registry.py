from __future__ import annotations

from alpha101.factors.operator_lib import specs
from alpha101.factors.operator_lib.pandas import cross_section, regression, time_series, transforms


OPERATOR_MODULES = (cross_section, regression, time_series, transforms)
OPERATOR_SPECS = specs.OPERATOR_SPECS
OperatorSpec = specs.OperatorSpec
operator_names_by_category = specs.operator_names_by_category
operator_params_by_category = specs.operator_params_by_category


OPERATOR_REGISTRY = {}
for _module in OPERATOR_MODULES:
    OPERATOR_REGISTRY.update({name: getattr(_module, name) for name in _module.__all__})

missing_specs = sorted(name for name in OPERATOR_SPECS if name not in OPERATOR_REGISTRY)
if missing_specs:
    raise RuntimeError(f"Operator specs reference missing functions: {missing_specs}")


__all__ = [
    "OPERATOR_MODULES",
    "OPERATOR_REGISTRY",
    "OPERATOR_SPECS",
    "OperatorSpec",
    "operator_names_by_category",
    "operator_params_by_category",
]
