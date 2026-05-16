import sys
import warnings
import multiprocessing as mp
from datetime import datetime
from pathlib import Path
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy import stats
import math
from tqdm import tqdm
from . import config
from .data import build_full_dataset


_FACTOR_WORKER_CTX: dict = {}


def _init_factor_worker(ctx: dict) -> None:
    """Initialize per-process global context for factor-level parallel backtest."""
    global _FACTOR_WORKER_CTX
    # print(f"[factor-worker:{mp.current_process().pid}] initialization started", flush=True)
    _FACTOR_WORKER_CTX = ctx
    # print(f"[factor-worker:{mp.current_process().pid}] initialization completed", flush=True)


def _compute_one_alpha_worker(alpha: str):
    """Compute one factor using process-global context initialized once per worker."""
    ctx = _FACTOR_WORKER_CTX
    try:
        common_info = ctx["alpha_common_map"].get(alpha)
        if common_info is None:
            return None

        factor_cols_aligned = common_info["factor_cols_aligned"]
        target_indices = common_info["target_indices"]
        common_symbols = common_info["common_symbols"]
        if not common_symbols:
            return None

        df = ctx["df"]
        sampled_idx = ctx["sampled_idx"]
        target_forward_eval = ctx["target_forward_eval"]
        target_eval_raw = ctx["target_eval_raw"]
        funding_eval_raw = ctx["funding_eval_raw"]
        n_quintiles = ctx["n_quintiles"]
        k_bars = ctx["k_bars"]
        factor_agg = ctx["factor_agg"]
        freq = ctx["freq"]
        return_pnl = ctx["return_pnl"]

        factor_raw = df[factor_cols_aligned].to_numpy(dtype=np.float32, copy=False)

        if k_bars > 1:
            factor_df = pd.DataFrame(factor_raw, copy=False)
            if factor_agg == 'mean':
                factor_data = factor_df.rolling(window=k_bars, min_periods=k_bars).mean().to_numpy(dtype=np.float32)
            elif factor_agg == 'ewma':
                factor_data = factor_df.ewm(span=k_bars, min_periods=k_bars).mean().to_numpy(dtype=np.float32)
            else:
                factor_data = factor_df.rolling(window=k_bars, min_periods=k_bars).std().to_numpy(dtype=np.float32)
        else:
            factor_data = factor_raw

        factor_data = factor_data[sampled_idx]
        target_data_alpha = target_forward_eval[:, target_indices]
        valid_mask_alpha = np.isfinite(factor_data) & np.isfinite(target_data_alpha)

        daily_pnl = np.zeros(len(sampled_idx), dtype=np.float32)
        turnover_sum = 0.0
        turnover_count = 0
        long_funding_sum = 0.0
        short_funding_sum = 0.0
        funding_count = 0

        daily_valid_count = valid_mask_alpha.sum(axis=1)
        active_date_indices = np.flatnonzero(daily_valid_count >= n_quintiles)

        prev_long_idx = None
        prev_short_idx = None

        for date_idx in active_date_indices:
            current_valid = valid_mask_alpha[date_idx]
            valid_idx = np.where(current_valid)[0]
            n_valid = len(valid_idx)

            if n_valid < n_quintiles:
                continue

            k = n_valid // n_quintiles
            if k == 0:
                continue

            factor_vals = factor_data[date_idx, valid_idx]
            target_vals = target_data_alpha[date_idx, valid_idx]
            funding_vals = funding_eval_raw[date_idx, target_indices][valid_idx]

            partition_idx = np.argpartition(factor_vals, (k - 1, -k))
            short_local_idx = partition_idx[:k]
            long_local_idx = partition_idx[-k:]

            short_idx = valid_idx[short_local_idx]
            long_idx = valid_idx[long_local_idx]

            long_ret = target_vals[long_local_idx].mean()
            short_ret = target_vals[short_local_idx].mean()
            long_funding = funding_vals[long_local_idx].mean()
            short_funding = funding_vals[short_local_idx].mean()
            daily_pnl[date_idx] = long_ret - short_ret
            long_funding_sum += float(long_funding)
            short_funding_sum += float(short_funding)
            funding_count += 1

            if prev_long_idx is not None:
                long_turnover = _turnover_ratio(long_idx, prev_long_idx)
                short_turnover = _turnover_ratio(short_idx, prev_short_idx)
                turnover_sum += (long_turnover + short_turnover) * 0.5
                turnover_count += 1

            prev_long_idx = long_idx
            prev_short_idx = short_idx

        returns_mean = np.mean(daily_pnl)
        returns_std = np.std(daily_pnl, ddof=1)
        sharpe = (returns_mean / returns_std) * np.sqrt(len(daily_pnl)) if returns_std > 0 else 0

        total_ret = np.prod(1.0 + daily_pnl) - 1
        n_years = len(daily_pnl) * pd.to_timedelta(freq).total_seconds() * k_bars / (365.25 * 24 * 3600)
        cagr = (1 + total_ret) ** (1.0 / max(n_years, 0.01)) - 1
        returns = np.sum(daily_pnl) / n_years

        avg_turnover = (turnover_sum / turnover_count) if turnover_count > 0 else 0.0
        avg_long_funding = (long_funding_sum / funding_count) if funding_count > 0 else 0.0
        avg_short_funding = (short_funding_sum / funding_count) if funding_count > 0 else 0.0
        win_rate = np.sum(daily_pnl > 0) / len(daily_pnl)

        cum_pnl = np.cumprod(1.0 + daily_pnl)
        running_max = np.maximum.accumulate(cum_pnl)
        max_drawdown = np.min((cum_pnl - running_max) / np.maximum(running_max, 1e-8))

        factor_eval = factor_raw[sampled_idx]
        target_eval = target_eval_raw[:, target_indices]
        ic_series = _nan_rowwise_corr(factor_eval, target_eval)
        ic_mean = float(np.nanmean(ic_series)) if ic_series.size > 0 else 0.0
        ic_std = float(np.nanstd(ic_series, ddof=1)) if np.isfinite(ic_series).sum() > 1 else 0.0
        ic_ir = ic_mean / (ic_std + 1e-8)

        base = sharpe * ic_ir * np.sqrt(np.abs(returns)) / max(avg_turnover, 0.125)
        # base = sharpe * ic_ir
        penalty_turnover = np.exp(-3 * avg_turnover)
        penalty_dd = np.exp(-5 * abs(max_drawdown))
        penalty_nan = np.exp(-20 * (daily_pnl == 0).sum() / len(daily_pnl))

        fitness = abs(base * penalty_turnover * penalty_dd * penalty_nan)

        if abs(ic_mean) < 0.01 or abs(sharpe) < 1.0:
            fitness *= 0.2
        if avg_turnover < 0.05 or avg_turnover > 0.7:
            fitness *= 0.1

        return {
            'factor': alpha,
            'sharpe': sharpe,
            'cagr': f"{cagr*100:.2f}%",
            'returns': f"{returns*100:.2f}%",
            'fitness': fitness,
            'turnover': f"{avg_turnover*100:.2f}%",
            'avg_long_funding': avg_long_funding,
            'avg_short_funding': avg_short_funding,
            'win_rate': win_rate,
            'drawdown': f"{max_drawdown*100:.2f}%",
            'ic_mean': f"{ic_mean:.4f}",
            'ic_std': f"{ic_std:.4f}",
            'ic_ir': f"{ic_ir:.4f}",
            'margin': f"{returns_mean*100:.2f}%",
            'obs_count': len(daily_pnl),
            'pnl': daily_pnl.tolist() if return_pnl else None,
        }
    except Exception as e:
        print(f"Warning: Stratified backtest failed for {alpha}: {e}")
        return None

def _turnover_ratio(current_idx: np.ndarray, prev_idx: np.ndarray) -> float:
    return len(np.setdiff1d(current_idx, prev_idx)) / max(len(current_idx), 1)


def _compute_daily_chunk(args):
    date_indices, factor_data, target_data, valid_mask, n_quintiles = args

    chunk_len = len(date_indices)
    pnl_chunk = np.zeros(chunk_len, dtype=np.float32)
    turnover_sum = 0.0
    turnover_count = 0
    first_long_idx = None
    first_short_idx = None
    last_long_idx = None
    last_short_idx = None
    prev_long_idx = None
    prev_short_idx = None

    for local_idx, date_idx in enumerate(date_indices):
        current_valid = valid_mask[date_idx]
        valid_idx = np.flatnonzero(current_valid)
        n_valid = valid_idx.size

        if n_valid < n_quintiles:
            continue

        k = n_valid // n_quintiles
        if k == 0:
            continue

        factor_vals = factor_data[date_idx, valid_idx]
        target_vals = target_data[date_idx, valid_idx]
        partition_indices = np.argpartition(factor_vals, (k - 1, -k))

        short_local_idx = partition_indices[:k]
        long_local_idx = partition_indices[-k:]
        short_idx = valid_idx[short_local_idx]
        long_idx = valid_idx[long_local_idx]

        short_ret = target_vals[short_local_idx].mean()
        long_ret = target_vals[long_local_idx].mean()
        pnl_chunk[local_idx] = long_ret - short_ret

        if first_long_idx is None:
            first_long_idx = long_idx
            first_short_idx = short_idx

        if prev_long_idx is not None:
            long_turnover = _turnover_ratio(long_idx, prev_long_idx)
            short_turnover = _turnover_ratio(short_idx, prev_short_idx)
            turnover_sum += (long_turnover + short_turnover) * 0.5
            turnover_count += 1

        prev_long_idx = long_idx
        prev_short_idx = short_idx
        last_long_idx = long_idx
        last_short_idx = short_idx

    return (
        np.asarray(date_indices, dtype=np.int32),
        pnl_chunk,
        turnover_sum,
        turnover_count,
        first_long_idx,
        first_short_idx,
        last_long_idx,
        last_short_idx,
    )


def _nan_rowwise_corr(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute per-row Pearson correlation with NaN handling."""
    mask = np.isfinite(x) & np.isfinite(y)
    n = mask.sum(axis=1)
    corr = np.full(x.shape[0], np.nan, dtype=np.float32)
    valid_rows = n >= 2
    if not np.any(valid_rows):
        return corr

    x_masked = np.where(mask, x, 0.0)
    y_masked = np.where(mask, y, 0.0)
    denom_n = np.maximum(n, 1)
    x_mean = x_masked.sum(axis=1) / denom_n
    y_mean = y_masked.sum(axis=1) / denom_n

    x_centered = np.where(mask, x - x_mean[:, None], 0.0)
    y_centered = np.where(mask, y - y_mean[:, None], 0.0)
    cov = (x_centered * y_centered).sum(axis=1)
    x_ss = (x_centered * x_centered).sum(axis=1)
    y_ss = (y_centered * y_centered).sum(axis=1)
    denom = np.sqrt(x_ss * y_ss)

    good = valid_rows & (denom > 0)
    corr[good] = cov[good] / denom[good]
    return corr


def _sample_indices_after_agg(n_rows: int, k_bars: int) -> np.ndarray:
    if n_rows <= 0:
        return np.array([], dtype=np.int32)
    if k_bars <= 1:
        return np.arange(n_rows, dtype=np.int32)

    sampled = [0]
    sampled.extend(range(k_bars - 1, n_rows, k_bars))
    return np.unique(np.asarray(sampled, dtype=np.int32))


def _get_preferred_mp_context():
    """Prefer fork on Unix for faster startup and lower copy overhead."""
    if not sys.platform.startswith("win"):
        try:
            methods = mp.get_all_start_methods()
        except Exception:
            methods = []
        if "fork" in methods:
            return mp.get_context("fork")
    return mp.get_context()


def stratified_backtest(
    df: pd.DataFrame, 
    alpha_cols: list, 
    target_col: str = 'target',
    n_quintiles: int = 5,
    k_bars: int = 16,
    factor_agg: str = 'ewma',
    parallel_dates: bool = True,
    date_workers: int | None = None,
    date_chunk_size: int = 1024,
    freq = '15m',
    verbose: bool = True,
    show_progress: bool = True,
    parallel_factors: bool = True,
    factor_workers: int | None = None,
    return_pnl: bool = False,
) -> pd.DataFrame:
    """
    Stratified long-short backtest for factor ranking.

    Args:
        df: Wide DataFrame with MultiIndex columns: (factor/target, symbol).
        alpha_cols: Factor names to evaluate.
        target_col: Target return group name.
        n_quintiles: Number of quantile groups for ranking.
        k_bars: Holding horizon in bars. If >1, factors are aggregated and
            forward returns are compounded over k bars.
        factor_agg: Factor aggregation method: 'mean', 'std', or 'ewma'.

    Returns:
        DataFrame with per-factor backtest metrics.
    """
    from tabulate import tabulate
    
    if verbose:
        print(f"\n{'='*100}")
        print(f"STRATIFIED BACKTEST (Long {n_quintiles} - Short 1 Strategy)")
        print(f"{'='*100}")

    if k_bars < 1:
        raise ValueError("k_bars must be >= 1")
    if factor_agg not in {'mean', 'std', 'ewma'}:
        raise ValueError("factor_agg must be either 'mean', 'std', or 'ewma'")

    # Avoid nested parallelism (factor-level + date-level) which often degrades performance.
    if parallel_factors and parallel_dates:
        if verbose:
            print("parallel_factors=True detected, disabling parallel_dates to avoid nested pools.")
        parallel_dates = False
    
    # Preprocess target columns once.
    target_cols = [col for col in df.columns if col[0] == target_col]
    if not target_cols:
        if verbose:
            print("No backtest results generated")
        return pd.DataFrame()
    
    # Build sampled evaluation index once.
    years_full = df.index.year.values
    sampled_idx = _sample_indices_after_agg(len(df), k_bars)
    if sampled_idx.size == 0:
        if verbose:
            print("No backtest results generated")
        return pd.DataFrame()

    df_eval = df.iloc[sampled_idx]
    years = years_full[sampled_idx] if sampled_idx.size > 0 else np.array([], dtype=years_full.dtype)
    if verbose and len(df_eval.index) > 0:
        start_dt = df_eval.index[0]
        end_dt = df_eval.index[-1]
        print(f"Backtest period: {start_dt} -> {end_dt}")
    
    # Precompute symbol mapping once to avoid rescanning columns for every alpha.
    alpha_symbol_map = {}
    for top_level, sym in df.columns:
        if top_level == target_col:
            continue
        alpha_symbol_map.setdefault(top_level, []).append(sym)
    target_symbols = list(df[target_col].columns)
    target_symbol_set = set(target_symbols)
    target_symbol_to_idx = {sym: idx for idx, sym in enumerate(target_symbols)}

    target_raw_full = df[target_col].to_numpy(dtype=np.float32, copy=False)
    if k_bars > 1:
        target_forward_full = np.ones_like(target_raw_full, dtype=np.float32)
        for step in range(1, k_bars + 1):
            valid_len = target_raw_full.shape[0] - step
            if valid_len > 0:
                target_forward_full[:valid_len] *= (1.0 + target_raw_full[step:, :])
            target_forward_full[valid_len:, :] = np.nan
        target_forward_full -= 1.0
    else:
        target_forward_full = target_raw_full

    target_forward_eval = target_forward_full[sampled_idx]
    target_eval_raw = target_raw_full[sampled_idx]
    funding_raw_full = df["funding"].to_numpy(dtype=np.float32, copy=False)
    funding_eval_raw = funding_raw_full[sampled_idx]

    debug = False
    parallel_dates_enabled = parallel_dates
    if parallel_dates and sys.platform.startswith("win"):
        parallel_dates_enabled = False
        if verbose:
            print("Windows detected: disable date-level multiprocessing to avoid allocation/paging failures.")
    local_date_workers = None
    date_pool = None
    mp_ctx = _get_preferred_mp_context()
    if parallel_dates_enabled:
        local_date_workers = max(1, mp.cpu_count() - 1) if date_workers is None else date_workers
        date_pool = mp_ctx.Pool(processes=local_date_workers)

    def _compute_one_alpha(alpha: str):
        try:
            start_time = time.time()

            factor_symbols = alpha_symbol_map.get(alpha, [])
            if not factor_symbols:
                return None

            common_symbols = [sym for sym in factor_symbols if sym in target_symbol_set]
            if not common_symbols:
                return None

            factor_cols_aligned = [(alpha, sym) for sym in common_symbols]
            target_indices = np.fromiter((target_symbol_to_idx[sym] for sym in common_symbols), dtype=np.int32)

            factor_raw = df[factor_cols_aligned].to_numpy(dtype=np.float32, copy=False)

            if k_bars > 1:
                factor_df = pd.DataFrame(factor_raw, copy=False)
                if factor_agg == 'mean':
                    factor_data = factor_df.rolling(window=k_bars, min_periods=k_bars).mean().to_numpy(dtype=np.float32)
                elif factor_agg == 'ewma':
                    factor_data = factor_df.ewm(span=k_bars, min_periods=k_bars).mean().to_numpy(dtype=np.float32)
                else:
                    factor_data = factor_df.rolling(window=k_bars, min_periods=k_bars).std().to_numpy(dtype=np.float32)
            else:
                factor_data = factor_raw

            factor_data = factor_data[sampled_idx]
            target_data_alpha = target_forward_eval[:, target_indices]

            valid_mask_alpha = np.isfinite(factor_data) & np.isfinite(target_data_alpha)

            daily_pnl = np.zeros(len(sampled_idx), dtype=np.float32)
            turnover_sum = 0.0
            turnover_count = 0
            long_funding_sum = 0.0
            short_funding_sum = 0.0
            funding_count = 0
            parallel_fallback = False
            if debug:
                print("Preparing data for backtest...", time.time() - start_time)
                start_time = time.time()

            if parallel_dates_enabled:
                daily_valid_count = valid_mask_alpha.sum(axis=1)
                date_indices = np.flatnonzero(daily_valid_count >= n_quintiles).tolist()
                if not date_indices:
                    return None

                # Prefer fewer, larger tasks to reduce scheduling overhead.
                desired_chunks = max(1, local_date_workers * 2)
                effective_chunk_size = max(
                    int(date_chunk_size),
                    int(np.ceil(len(date_indices) / desired_chunks)),
                )
                chunks = [
                    date_indices[i:i + effective_chunk_size]
                    for i in range(0, len(date_indices), effective_chunk_size)
                ]

                chunk_inputs = [
                    (chunk, factor_data, target_data_alpha, valid_mask_alpha, n_quintiles)
                    for chunk in chunks
                ]
                try:
                    pool_chunksize = max(1, len(chunk_inputs) // max(1, local_date_workers * 2))
                    chunk_outputs = date_pool.map(_compute_daily_chunk, chunk_inputs, chunksize=pool_chunksize)
                except (MemoryError, OSError) as e:
                    if verbose:
                        print(f"Date-level multiprocessing failed ({e}). Falling back to serial date loop.")
                    parallel_fallback = True
                else:
                    parallel_fallback = False

                if not parallel_fallback:
                    prev_last_long_idx = None
                    prev_last_short_idx = None
                    for (
                        date_idx_arr,
                        pnl_chunk,
                        chunk_turnover_sum,
                        chunk_turnover_count,
                        first_long_idx,
                        first_short_idx,
                        last_long_idx,
                        last_short_idx,
                    ) in chunk_outputs:
                        daily_pnl[date_idx_arr] = pnl_chunk
                        turnover_sum += chunk_turnover_sum
                        turnover_count += chunk_turnover_count

                        if prev_last_long_idx is not None and first_long_idx is not None:
                            long_turnover = _turnover_ratio(first_long_idx, prev_last_long_idx)
                            short_turnover = _turnover_ratio(first_short_idx, prev_last_short_idx)
                            turnover_sum += (long_turnover + short_turnover) * 0.5
                            turnover_count += 1

                        if last_long_idx is not None:
                            prev_last_long_idx = last_long_idx
                            prev_last_short_idx = last_short_idx

            if (not parallel_dates_enabled) or parallel_fallback:
                daily_valid_count = valid_mask_alpha.sum(axis=1)
                active_date_indices = np.flatnonzero(daily_valid_count >= n_quintiles)

                prev_long_idx = None
                prev_short_idx = None

                for date_idx in active_date_indices:
                    current_valid = valid_mask_alpha[date_idx]
                    valid_idx = np.where(current_valid)[0]
                    n_valid = len(valid_idx)

                    if n_valid < n_quintiles:
                        continue

                    k = n_valid // n_quintiles
                    if k == 0:
                        continue

                    factor_vals = factor_data[date_idx, valid_idx]
                    target_vals = target_data_alpha[date_idx, valid_idx]
                    funding_vals = funding_eval_raw[date_idx, target_indices][valid_idx]

                    partition_idx = np.argpartition(factor_vals, (k - 1, -k))
                    short_local_idx = partition_idx[:k]
                    long_local_idx = partition_idx[-k:]

                    short_idx = valid_idx[short_local_idx]
                    long_idx = valid_idx[long_local_idx]

                    long_ret = target_vals[long_local_idx].mean()
                    short_ret = target_vals[short_local_idx].mean()
                    long_funding = funding_vals[long_local_idx].mean()
                    short_funding = funding_vals[short_local_idx].mean()
                    daily_pnl[date_idx] = long_ret - short_ret
                    long_funding_sum += float(long_funding)
                    short_funding_sum += float(short_funding)
                    funding_count += 1

                    if prev_long_idx is not None:
                        long_turnover = _turnover_ratio(long_idx, prev_long_idx)
                        short_turnover = _turnover_ratio(short_idx, prev_short_idx)
                        turnover_sum += (long_turnover + short_turnover) * 0.5
                        turnover_count += 1

                    prev_long_idx = long_idx
                    prev_short_idx = short_idx

            returns_mean = np.mean(daily_pnl)
            returns_std = np.std(daily_pnl, ddof=1)
            sharpe = (returns_mean / returns_std) * np.sqrt(len(daily_pnl)) if returns_std > 0 else 0

            total_ret = np.prod(1.0 + daily_pnl) - 1
            n_years = len(daily_pnl) * pd.to_timedelta(freq).total_seconds() * k_bars / (365.25 * 24 * 3600)
            cagr = (1 + total_ret) ** (1.0 / max(n_years, 0.01)) - 1
            returns = np.sum(daily_pnl) / n_years

            avg_turnover = (turnover_sum / turnover_count) if turnover_count > 0 else 0.0
            avg_long_funding = (long_funding_sum / funding_count) if funding_count > 0 else 0.0
            avg_short_funding = (short_funding_sum / funding_count) if funding_count > 0 else 0.0
            win_rate = np.sum(daily_pnl > 0) / len(daily_pnl)

            cum_pnl = np.cumprod(1.0 + daily_pnl)
            running_max = np.maximum.accumulate(cum_pnl)
            max_drawdown = np.min((cum_pnl - running_max) / np.maximum(running_max, 1e-8))

            factor_eval = df_eval[alpha][common_symbols].to_numpy(dtype=np.float32, copy=False)
            target_eval = target_eval_raw[:, target_indices]
            ic_series = _nan_rowwise_corr(factor_eval, target_eval)
            ic_mean = float(np.nanmean(ic_series)) if ic_series.size > 0 else 0.0
            ic_std = float(np.nanstd(ic_series, ddof=1)) if np.isfinite(ic_series).sum() > 1 else 0.0
            ic_ir = ic_mean / (ic_std + 1e-8)

            base = sharpe * ic_ir
            penalty_turnover = np.exp(-10 * avg_turnover)
            penalty_dd = np.exp(-5 * abs(max_drawdown))
            penalty_nan = np.exp(-20 * (daily_pnl == 0).sum() / len(daily_pnl))

            fitness = abs(base * penalty_turnover * penalty_dd * penalty_nan)

            if abs(ic_mean) < 0.01 or abs(sharpe) < 1.0:
                fitness *= 0.2
            if avg_turnover < 0.01 or avg_turnover > 0.7:
                fitness *= 0.1

            return {
                'factor': alpha,
                'sharpe': sharpe,
                'cagr': f"{cagr*100:.2f}%",
                'returns': f"{returns*100:.2f}%",
                'fitness': fitness,
                'turnover': f"{avg_turnover*100:.2f}%",
                'avg_long_funding': avg_long_funding,
                'avg_short_funding': avg_short_funding,
                'win_rate': win_rate,
                'drawdown': f"{max_drawdown*100:.2f}%",
                'ic_mean': f"{ic_mean:.4f}",
                'ic_std': f"{ic_std:.4f}",
                'ic_ir': f"{ic_ir:.4f}",
                'margin': f"{returns_mean*100:.2f}%",
                'obs_count': len(daily_pnl),
                'pnl': daily_pnl.tolist() if return_pnl else None,
            }
        except Exception as e:
            print(f"Warning: Stratified backtest failed for {alpha}: {e}")
            return None

    results = []
    try:
        if parallel_factors:
            if factor_workers is None:
                factor_workers = max(1, min(len(alpha_cols), (mp.cpu_count()-4 or 1)))
            alpha_common_map = {}
            for alpha in alpha_cols:
                factor_symbols = alpha_symbol_map.get(alpha, [])
                if not factor_symbols:
                    continue
                common_symbols = [sym for sym in factor_symbols if sym in target_symbol_set]
                if not common_symbols:
                    continue
                factor_cols_aligned = [(alpha, sym) for sym in common_symbols]
                target_indices = np.fromiter((target_symbol_to_idx[sym] for sym in common_symbols), dtype=np.int32)
                alpha_common_map[alpha] = {
                    "factor_cols_aligned": factor_cols_aligned,
                    "target_indices": target_indices,
                    "common_symbols": common_symbols,
                }

            worker_ctx = {
                "df": df,
                "sampled_idx": sampled_idx,
                "target_forward_eval": target_forward_eval,
                "target_eval_raw": target_eval_raw,
                "funding_eval_raw": funding_eval_raw,
                "n_quintiles": n_quintiles,
                "k_bars": k_bars,
                "factor_agg": factor_agg,
                "freq": freq,
                "return_pnl": return_pnl,
                "alpha_common_map": alpha_common_map,
            }

            submit_alphas = [alpha for alpha in alpha_cols if alpha in alpha_common_map]
            if verbose:
                print(
                    f"[factor-backtest] worker initialization started "
                    f"(workers={factor_workers}, tasks={len(submit_alphas)})"
                )
            with ProcessPoolExecutor(
                max_workers=factor_workers,
                mp_context=mp_ctx,
                initializer=_init_factor_worker,
                initargs=(worker_ctx,),
            ) as executor:
                if verbose:
                    print("[factor-backtest] worker initialization completed")
                factor_chunksize = max(1, len(submit_alphas) // max(1, factor_workers * 2))
                pbar = tqdm(total=len(submit_alphas), desc="Backtesting factors (parallel)") if show_progress else None
                for row in executor.map(_compute_one_alpha_worker, submit_alphas, chunksize=factor_chunksize):
                    if row is not None:
                        results.append(row)
                    if pbar is not None:
                        pbar.update(1)
                if pbar is not None:
                    pbar.close()
        else:
            alpha_iter = tqdm(alpha_cols, desc="Backtesting factors") if show_progress else alpha_cols
            for alpha in alpha_iter:
                row = _compute_one_alpha(alpha)
                if row is not None:
                    results.append(row)
    finally:
        if date_pool is not None:
            date_pool.close()
            date_pool.join()

    if not results:
        if verbose:
            print("No backtest results generated")
        return pd.DataFrame()
    
    backtest_df = pd.DataFrame(results).sort_values('sharpe', ascending=False)
    selected_cols = ['factor', 'sharpe', 'cagr', 'returns', 'turnover', 
                    'fitness', 'win_rate', 'drawdown', 'margin', 'ic_mean', 'ic_std']
    
    # Print summary table.
    if verbose:
        print(tabulate(backtest_df[selected_cols], 
                    headers='keys', 
                    tablefmt='grid',
                    floatfmt='.3f'))
        
        print(f"{'='*100}\n")
    print(f"Stratified backtest completed for {len(backtest_df)} factors.")
    return backtest_df

def main():
    """Main entry point."""
    cfg = config.get_config()
    
    # Run stratified backtest.
    print("\n" + "="*100)
    print("Running stratified backtest...")
    print("="*100)
    
    
    df_backtest = build_full_dataset(
        cfg.pairs,
        cfg.lookback_days,
        cfg.data_root,
        cfg.timeframe,
        test_start_date=cfg.test_start_date,
        test_end_date=cfg.test_end_date,
    )
    
    alpha_cols = list(set([col for col in df_backtest.columns if col.startswith("alpha")]))
    
    # Convert to wide table.
    df_backtest = df_backtest.set_index(["date", "symbol"]).sort_index()
    df_backtest.index = df_backtest.index.set_levels(
        df_backtest.index.levels[1].astype("category"), level=1
    )
    
    df_backtest = df_backtest.unstack(level='symbol')
    
    # Run stratified backtest.
    backtest_results = stratified_backtest(
        df_backtest,
        alpha_cols,
        target_col='target',
        n_quintiles=5,
        k_bars=16, # 15m * 16 = 4h holding period
        factor_agg='mean',
        parallel_dates=True,
        date_workers=16,
        date_chunk_size=256,
    )
    
    # Save backtest results.
    output_dir = Path('analysis_results')
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if not backtest_results.empty:
        backtest_output = output_dir / f'stratified_backtest_{timestamp}.csv'
        backtest_results.to_csv(backtest_output, index=False)
        print(f"Stratified backtest results saved to: {backtest_output}")
        
    # except Exception as e:
    #     print(f"Stratified backtest failed: {e}")
    
    print('\nAnalysis completed successfully!')

if __name__ == '__main__':
    main()
