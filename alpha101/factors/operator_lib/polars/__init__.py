from alpha101.factors.operator_lib import specs

OPERATOR_MODULES = ()
OPERATOR_REGISTRY = {}
OPERATOR_SPECS = specs.OPERATOR_SPECS
OperatorSpec = specs.OperatorSpec
operator_names_by_category = specs.operator_names_by_category
operator_params_by_category = specs.operator_params_by_category

__all__ = [
    "OPERATOR_MODULES",
    "OPERATOR_REGISTRY",
    "OPERATOR_SPECS",
    "OperatorSpec",
    "operator_names_by_category",
    "operator_params_by_category",
]
