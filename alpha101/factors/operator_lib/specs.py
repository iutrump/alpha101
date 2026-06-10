from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OperatorSpec:
    name: str
    category: str
    arity: int
    params: tuple[int, ...] = ()


OPERATOR_SPECS = {
    "ts_mean": OperatorSpec("ts_mean", "time_series", 1, (3, 12, 20, 30, 60)),
    "ts_rank": OperatorSpec("ts_rank", "time_series", 1, (3, 5, 7, 10, 14, 20, 30)),
    "ts_min": OperatorSpec("ts_min", "time_series", 1, (3, 5, 7, 10, 14, 20, 30)),
    "ts_max": OperatorSpec("ts_max", "time_series", 1, (3, 5, 7, 10, 14, 20, 30)),
    "ts_std": OperatorSpec("ts_std", "time_series", 1, (7, 14, 21, 28, 54)),
    "ts_std_dev": OperatorSpec("ts_std_dev", "time_series", 1, (7, 14, 21, 28, 54)),
    "ts_argmax": OperatorSpec("ts_argmax", "time_series", 1, (7, 14, 21, 28)),
    "ts_arg_max": OperatorSpec("ts_arg_max", "time_series", 1, (7, 14, 21, 28)),
    "ts_argmin": OperatorSpec("ts_argmin", "time_series", 1, (7, 14, 21, 28)),
    "ts_arg_min": OperatorSpec("ts_arg_min", "time_series", 1, (7, 14, 21, 28)),
    "ts_sum": OperatorSpec("ts_sum", "time_series", 1, (5, 10, 20, 30, 60)),
    "product": OperatorSpec("product", "time_series", 1, (5, 10, 20)),
    "ts_product": OperatorSpec("ts_product", "time_series", 1, (5, 10, 20)),
    "ts_skewness": OperatorSpec("ts_skewness", "time_series", 1, (10, 20, 30, 60)),
    "ts_kurtosis": OperatorSpec("ts_kurtosis", "time_series", 1, (10, 20, 30, 60)),
    "decay_linear": OperatorSpec("decay_linear", "time_series", 1, (7, 14, 21, 28)),
    "ts_decay_linear": OperatorSpec("ts_decay_linear", "time_series", 1, (7, 14, 21, 28)),
    "ts_drawdown": OperatorSpec("ts_drawdown", "time_series", 1, (7, 14, 21, 28)),
    "ts_pos": OperatorSpec("ts_pos", "time_series", 1, (7, 14, 21, 28)),
    "ts_volatility": OperatorSpec("ts_volatility", "time_series", 1, (7, 14, 21, 28)),
    "ts_percentile": OperatorSpec("ts_percentile", "time_series", 1, (3, 5, 7, 10, 14, 20, 30)),
    "ts_zscore": OperatorSpec("ts_zscore", "time_series", 1, (7, 14, 21, 28)),
    "ts_ema": OperatorSpec("ts_ema", "time_series", 1, (6, 12, 24, 48)),
    "ts_slope": OperatorSpec("ts_slope", "time_series", 1, (6, 12, 24, 48)),
    "ts_corr": OperatorSpec("ts_corr", "time_series_dual", 2, (5, 10, 20, 30, 60)),
    "ts_cov": OperatorSpec("ts_cov", "time_series_dual", 2, (5, 10, 20, 30, 60)),
    "ts_covariance": OperatorSpec("ts_covariance", "time_series_dual", 2, (5, 10, 20, 30, 60)),
    "ts_alpha": OperatorSpec("ts_alpha", "time_series_dual", 2, (7, 14, 21, 28)),
    "ts_r2": OperatorSpec("ts_r2", "time_series_dual", 2, (7, 14, 21, 28)),
    "ts_beta": OperatorSpec("ts_beta", "time_series_dual", 2, (7, 14, 21, 28)),
    "ts_resid": OperatorSpec("ts_resid", "time_series_dual", 2, (6, 12, 24, 48)),
    "rank": OperatorSpec("rank", "cross_section", 1),
    "scale": OperatorSpec("scale", "cross_section", 1),
    "zscore": OperatorSpec("zscore", "cross_section", 1),
    "winsorize": OperatorSpec("winsorize", "cross_section", 1),
    "delay": OperatorSpec("delay", "lag", 1, (1, 3, 5, 7, 10, 14, 20)),
    "ts_delay": OperatorSpec("ts_delay", "lag", 1, (1, 3, 5, 7, 10, 14, 20)),
    "delta": OperatorSpec("delta", "lag", 1, (1, 3, 5, 7, 10)),
    "ts_delta": OperatorSpec("ts_delta", "lag", 1, (1, 3, 5, 7, 10)),
}


def operator_params_by_category(category: str) -> dict[str, list[int]]:
    return {
        name: list(spec.params)
        for name, spec in OPERATOR_SPECS.items()
        if spec.category == category and spec.params
    }


def operator_names_by_category(category: str) -> list[str]:
    return [
        name
        for name, spec in OPERATOR_SPECS.items()
        if spec.category == category
    ]


__all__ = [
    "OPERATOR_SPECS",
    "OperatorSpec",
    "operator_names_by_category",
    "operator_params_by_category",
]
