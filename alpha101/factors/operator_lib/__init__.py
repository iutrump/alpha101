from alpha101.factors.operator_lib.registry import (
    OPERATOR_MODULES,
    OPERATOR_REGISTRY,
    OPERATOR_SPECS,
    OperatorSpec,
    operator_names_by_category,
    operator_params_by_category,
)

globals().update(OPERATOR_REGISTRY)

__all__ = [
    "OPERATOR_MODULES",
    "OPERATOR_REGISTRY",
    "OPERATOR_SPECS",
    "OperatorSpec",
    "operator_names_by_category",
    "operator_params_by_category",
    *sorted(OPERATOR_REGISTRY),
]
