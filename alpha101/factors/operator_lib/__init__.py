from alpha101.factors.operator_lib import cross_section, regression, time_series, transforms

OPERATOR_MODULES = (cross_section, regression, time_series, transforms)

__all__ = []
for _module in OPERATOR_MODULES:
    __all__.extend(_module.__all__)
    globals().update({name: getattr(_module, name) for name in _module.__all__})
