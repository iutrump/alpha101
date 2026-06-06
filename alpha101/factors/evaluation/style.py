from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class StyleConfig:
    momentum_window: int = 20
    volatility_window: int = 20
    beta_window: int = 60
    liquidity_window: int = 20
    reversal_window: int = 5
    funding_window: int = 3
    standardize_factor: bool = True
    add_intercept: bool = True
    min_count: int = 8


def build_style_factors(
    close: pd.DataFrame,
    cap: pd.DataFrame | None,
    market_return: pd.DataFrame | None = None,
    volume: pd.DataFrame | None = None,
    funding: pd.DataFrame | None = None,
    *,
    momentum_window: int = 20,
    volatility_window: int = 20,
    beta_window: int = 60,
    liquidity_window: int = 20,
    reversal_window: int = 5,
    funding_window: int = 3,
) -> dict[str, pd.DataFrame]:
    returns = close.pct_change(fill_method=None)
    momentum = close.pct_change(momentum_window, fill_method=None)
    volatility = returns.rolling(volatility_window, min_periods=volatility_window).std()
    size = np.log(cap.where(cap > 0)) if cap is not None else np.log(close.where(close > 0))
    styles = {
        "size": size.reindex_like(close),
        "momentum": momentum.reindex_like(close),
        "volatility": volatility.reindex_like(close),
    }
    if market_return is not None:
        styles["beta"] = rolling_market_beta(
            returns,
            market_return.reindex_like(close),
            window=beta_window,
        )
    if volume is not None:
        traded_value = close * volume.reindex_like(close)
        liquidity = np.log(
            traded_value.where(traded_value > 0).rolling(
                liquidity_window,
                min_periods=liquidity_window,
            ).mean()
        )
        styles["liquidity"] = liquidity.reindex_like(close)
    styles["reversal"] = (-close.pct_change(reversal_window, fill_method=None)).reindex_like(close)
    if funding is not None:
        styles["funding"] = funding.reindex_like(close).rolling(
            funding_window,
            min_periods=funding_window,
        ).mean()
    return styles


def rolling_market_beta(
    returns: pd.DataFrame,
    market_return: pd.DataFrame,
    *,
    window: int = 60,
) -> pd.DataFrame:
    market_var = market_return.rolling(window, min_periods=window).var()
    covariance = returns.rolling(window, min_periods=window).cov(market_return)
    beta = covariance / market_var.replace(0.0, np.nan)
    return beta.reindex_like(returns)


def factor_style_exposures(
    factor: pd.DataFrame,
    styles: dict[str, pd.DataFrame],
    *,
    config: StyleConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = config or StyleConfig()
    return _cross_section_ols(
        _cs_zscore(factor) if cfg.standardize_factor else factor.astype(float),
        {name: _cs_zscore(value.reindex_like(factor)) for name, value in styles.items()},
        add_intercept=cfg.add_intercept,
        min_count=cfg.min_count,
    )


def residualize_style_factor(
    factor: pd.DataFrame,
    styles: dict[str, pd.DataFrame],
    *,
    config: StyleConfig | None = None,
) -> pd.DataFrame:
    _, residual = factor_style_exposures(factor, styles, config=config)
    return residual


def summarize_exposures(exposures: pd.DataFrame) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for column in exposures.columns:
        values = exposures[column].dropna()
        summary[column] = {
            "mean": _safe_float(values.mean()),
            "median": _safe_float(values.median()),
            "std": _safe_float(values.std(ddof=0)),
            "abs_mean": _safe_float(values.abs().mean()),
            "t_stat": _safe_float(_one_sample_t(values.to_numpy(dtype=float))),
            "obs_count": int(values.size),
        }
    return summary


def adf_wide(
    values: pd.DataFrame,
    *,
    max_lag: int = 1,
    regression: str = "c",
    min_count: int = 20,
) -> dict[str, Any]:
    results = {
        column: adf_1d(values[column].to_numpy(dtype=float), max_lag=max_lag, regression=regression)
        for column in values.columns
        if values[column].notna().sum() >= min_count
    }
    stats_values = np.asarray([item["statistic"] for item in results.values()], dtype=float)
    p_values = np.asarray([item["pvalue"] for item in results.values()], dtype=float)
    finite_stats = stats_values[np.isfinite(stats_values)]
    finite_p = p_values[np.isfinite(p_values)]
    return {
        "by_symbol": results,
        "symbols_tested": int(len(results)),
        "adf_stat_mean": _safe_float(np.nanmean(finite_stats)) if finite_stats.size else np.nan,
        "adf_stat_median": _safe_float(np.nanmedian(finite_stats)) if finite_stats.size else np.nan,
        "pvalue_mean": _safe_float(np.nanmean(finite_p)) if finite_p.size else np.nan,
        "pvalue_median": _safe_float(np.nanmedian(finite_p)) if finite_p.size else np.nan,
        "reject_5pct_ratio": _safe_float(np.mean(finite_p < 0.05)) if finite_p.size else np.nan,
    }


def adf_1d(values: np.ndarray, *, max_lag: int = 1, regression: str = "c") -> dict[str, float | int]:
    series = pd.Series(values, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    y = series.to_numpy(dtype=float)
    if regression not in {"n", "c", "ct"}:
        raise ValueError("regression must be one of: n, c, ct")
    min_obs = max_lag + 4 + (1 if regression == "c" else 2 if regression == "ct" else 0)
    if y.size < min_obs:
        return {"statistic": np.nan, "pvalue": np.nan, "used_lag": int(max_lag), "nobs": int(y.size)}

    dy = np.diff(y)
    response = dy[max_lag:]
    y_lag = y[max_lag:-1]
    x_parts = [y_lag]
    for lag in range(1, max_lag + 1):
        x_parts.append(dy[max_lag - lag : -lag])
    if regression in {"c", "ct"}:
        x_parts.append(np.ones_like(response))
    if regression == "ct":
        x_parts.append(np.arange(1, response.size + 1, dtype=float))
    x = np.column_stack(x_parts)
    valid = np.isfinite(response) & np.isfinite(x).all(axis=1)
    response = response[valid]
    x = x[valid]
    if response.size <= x.shape[1]:
        return {"statistic": np.nan, "pvalue": np.nan, "used_lag": int(max_lag), "nobs": int(response.size)}

    beta, _, _, _ = np.linalg.lstsq(x, response, rcond=None)
    resid = response - x @ beta
    dof = response.size - x.shape[1]
    xtx_inv = np.linalg.pinv(x.T @ x)
    sigma2 = float(resid @ resid / dof)
    se = np.sqrt(np.diag(xtx_inv) * sigma2)
    statistic = float(beta[0] / se[0]) if se[0] > 0 else np.nan
    pvalue = float(stats.t.cdf(statistic, dof)) if np.isfinite(statistic) else np.nan
    return {"statistic": statistic, "pvalue": pvalue, "used_lag": int(max_lag), "nobs": int(response.size)}


def strategy_market_exposure(strategy_returns, market_returns) -> dict[str, float | int]:
    y = np.asarray(strategy_returns, dtype=float)
    x = np.asarray(market_returns, dtype=float)
    valid = np.isfinite(y) & np.isfinite(x)
    y = y[valid]
    x = x[valid]
    if y.size < 3:
        return {
            "alpha": np.nan,
            "beta": np.nan,
            "r2": np.nan,
            "corr": np.nan,
            "t_stat": np.nan,
            "obs_count": int(y.size),
        }

    design = np.column_stack([np.ones_like(x), x])
    beta, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    if rank < design.shape[1]:
        return {
            "alpha": np.nan,
            "beta": np.nan,
            "r2": np.nan,
            "corr": np.nan,
            "t_stat": np.nan,
            "obs_count": int(y.size),
        }

    fitted = design @ beta
    resid = y - fitted
    centered = y - y.mean()
    ss_total = float(centered @ centered)
    ss_resid = float(resid @ resid)
    dof = y.size - design.shape[1]
    xtx_inv = np.linalg.pinv(design.T @ design)
    sigma2 = ss_resid / dof if dof > 0 else np.nan
    se = np.sqrt(np.diag(xtx_inv) * sigma2) if np.isfinite(sigma2) else np.full(2, np.nan)
    corr = np.corrcoef(x, y)[0, 1] if np.std(x) > 0 and np.std(y) > 0 else np.nan
    return {
        "alpha": _safe_float(beta[0]),
        "beta": _safe_float(beta[1]),
        "r2": _safe_float(1.0 - ss_resid / ss_total) if ss_total > 0 else np.nan,
        "corr": _safe_float(corr),
        "t_stat": _safe_float(beta[1] / se[1]) if se[1] > 0 else np.nan,
        "obs_count": int(y.size),
    }


def _cross_section_ols(
    y: pd.DataFrame,
    x_map: dict[str, pd.DataFrame],
    *,
    add_intercept: bool,
    min_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    names = list(x_map)
    columns = (["intercept"] if add_intercept else []) + names
    exposure_rows: list[list[float]] = []
    residual_values = np.full(y.shape, np.nan, dtype=float)
    y_arr = y.to_numpy(dtype=float, copy=False)
    x_arrays = [x_map[name].to_numpy(dtype=float, copy=False) for name in names]

    for row_idx in range(y.shape[0]):
        yi = y_arr[row_idx]
        xi = np.column_stack([arr[row_idx] for arr in x_arrays])
        valid = np.isfinite(yi) & np.isfinite(xi).all(axis=1)
        row_exposure = [np.nan] * len(columns)
        min_required = len(names) + (1 if add_intercept else 0) + 1
        if valid.sum() >= max(min_count, min_required):
            design = xi[valid]
            if add_intercept:
                design = np.column_stack([np.ones(valid.sum()), design])
            beta, _, rank, _ = np.linalg.lstsq(design, yi[valid], rcond=None)
            if rank == design.shape[1]:
                fitted = design @ beta
                residual_values[row_idx, np.flatnonzero(valid)] = yi[valid] - fitted
                row_exposure = [float(value) for value in beta]
        exposure_rows.append(row_exposure)

    exposures = pd.DataFrame(exposure_rows, index=y.index, columns=columns)
    residual = pd.DataFrame(residual_values, index=y.index, columns=y.columns)
    return exposures, residual


def _cs_zscore(df: pd.DataFrame) -> pd.DataFrame:
    mean = df.mean(axis=1, skipna=True)
    std = df.std(axis=1, skipna=True, ddof=0).replace(0.0, np.nan)
    return df.sub(mean, axis=0).div(std, axis=0)


def _one_sample_t(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if values.size < 2:
        return np.nan
    std = values.std(ddof=1)
    if std <= 0:
        return np.nan
    return float(values.mean() / (std / np.sqrt(values.size)))


def _safe_float(value) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return np.nan
    return out if np.isfinite(out) else np.nan
