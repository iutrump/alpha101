from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr
from tqdm import tqdm

from freqtrade.exchange.exchange_utils_timeframe import timeframe_to_seconds

from .data import load_ohlcv

_MP_INTRADAY_CACHE: Optional[dict] = None
_MP_BASKET_TP: float = 0.0
_MP_BASKET_SL: float = 0.0
_MP_SYMBOL_TP: float = 0.0
_MP_SYMBOL_SL: float = 0.0
_MP_TOP_N_LONG: int = 0
_MP_TOP_N_SHORT: int = 0
_MP_LEVERAGE: float = 1.0


@dataclass
class TradeResult:
    symbol: str
    trade_date: pd.Timestamp
    direction: str
    entry: float
    exit: float
    ret: float
    hit: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp


def _slice_day(df: pd.DataFrame, trade_date: pd.Timestamp) -> pd.DataFrame:
    ts = pd.Timestamp(trade_date)
    start = ts.tz_convert("UTC") if ts.tzinfo else ts.tz_localize("UTC")
    end = start + pd.Timedelta(days=1)
    mask = (df["date"] >= start) & (df["date"] < end)
    return df.loc[mask].reset_index(drop=True)


def _portfolio_exit_prices(positions: List[dict], ts) -> dict:
    prices = {}
    # Convert to numpy datetime64 if it's a Timestamp
    ts_np = pd.Timestamp(ts).to_datetime64() if isinstance(ts, pd.Timestamp) else ts
    for pos in positions:
        # Use pre-cached numpy arrays with binary search
        idx = np.searchsorted(pos["dates"], ts_np, side='right') - 1
        if idx < 0:
            idx = 0
        prices[pos["symbol"]] = float(pos["closes"][idx])
    return prices


def _simulate_basket_day(
    longs: pd.DataFrame,
    shorts: pd.DataFrame,
    intraday_cache: dict,
    basket_tp: float,
    basket_sl: float,
    symbol_tp: float,
    symbol_sl: float,
) -> tuple[List[TradeResult], str, pd.Timestamp]:
    positions: List[dict] = []
    
    # Use itertuples() instead of iterrows() - 10-100x faster
    for row in longs.itertuples():
        day_df = _slice_day(intraday_cache[row.symbol], row.trade_date)
        if day_df.empty:
            continue
        first_row = day_df.iloc[0]
        positions.append({
            "symbol": row.symbol,
            "direction": "long",
            "df": day_df,
            "open": float(first_row["open"]),
            "open_time": pd.Timestamp(first_row["date"]),
            "dates": day_df["date"].values,  # Pre-cache dates as numpy array
            "closes": day_df["close"].values,  # Pre-cache closes
        })
    
    for row in shorts.itertuples():
        day_df = _slice_day(intraday_cache[row.symbol], row.trade_date)
        if day_df.empty:
            continue
        first_row = day_df.iloc[0]
        positions.append({
            "symbol": row.symbol,
            "direction": "short",
            "df": day_df,
            "open": float(first_row["open"]),
            "open_time": pd.Timestamp(first_row["date"]),
            "dates": day_df["date"].values,
            "closes": day_df["close"].values,
        })

    if not positions:
        return [], "no_trades", pd.NaT

    # Build unified timeline using numpy concatenate - much faster
    all_dates = np.concatenate([pos["dates"] for pos in positions])
    times = np.unique(all_dates)  # Already sorted
    
    # Pre-compute direction multipliers
    for pos in positions:
        pos["sign"] = 1.0 if pos["direction"] == "long" else -1.0

    trigger: Optional[tuple[str, pd.Timestamp, dict]] = None
    trades: List[TradeResult] = []
    active_indices = list(range(len(positions)))
    for ts in times:
        if not active_indices:
            break
        per_ret = []
        newly_closed: List[int] = []
        for i in active_indices:
            pos = positions[i]
            idx = np.searchsorted(pos["dates"], ts, side='right') - 1
            if idx < 0:
                continue
            price = float(pos["closes"][idx])
            r = pos["sign"] * (price - pos["open"]) / pos["open"]
            per_ret.append(r)
            # Per-symbol TP/SL immediate exit
            if r >= symbol_tp:
                trades.append(TradeResult(
                    symbol=pos["symbol"],
                    trade_date=pd.Timestamp(pos["dates"][0]),
                    direction=pos["direction"],
                    entry=pos["open"],
                    exit=price,
                    ret=r,
                    hit="symbol_tp",
                    entry_time=pos["open_time"],
                    exit_time=pd.Timestamp(ts),
                ))
                newly_closed.append(i)
            elif r <= -symbol_sl:
                trades.append(TradeResult(
                    symbol=pos["symbol"],
                    trade_date=pd.Timestamp(pos["dates"][0]),
                    direction=pos["direction"],
                    entry=pos["open"],
                    exit=price,
                    ret=r,
                    hit="symbol_sl",
                    entry_time=pos["open_time"],
                    exit_time=pd.Timestamp(ts),
                ))
                newly_closed.append(i)
        # Remove newly closed positions from active set
        if newly_closed:
            active_indices = [i for i in active_indices if i not in newly_closed]

        # Basket-level trigger on remaining active positions
        if per_ret:
            port_ret = float(np.mean(per_ret))
            if port_ret >= basket_tp:
                trigger = ("basket_tp", pd.Timestamp(ts), _portfolio_exit_prices([positions[i] for i in active_indices], ts))
                break
            if port_ret <= -basket_sl:
                trigger = ("basket_sl", pd.Timestamp(ts), _portfolio_exit_prices([positions[i] for i in active_indices], ts))
                break

    if trigger:
        hit, exit_ts, prices = trigger
        # Close remaining active positions at basket trigger
        for i in active_indices:
            pos = positions[i]
            exit_price = prices.get(pos["symbol"], float(pos["closes"][-1]))
            ret = pos["sign"] * (exit_price - pos["open"]) / pos["open"]
            trades.append(TradeResult(
                symbol=pos["symbol"],
                trade_date=pd.Timestamp(pos["dates"][0]),
                direction=pos["direction"],
                entry=pos["open"],
                exit=exit_price,
                ret=ret,
                hit=hit,
                entry_time=pos["open_time"],
                exit_time=exit_ts
            ))
        return trades, hit, exit_ts

    # No basket stop: exit at EOD close per position
    # Close remaining active positions at EOD; already-closed trades keep their hit
    for i in active_indices:
        pos = positions[i]
        exit_price = float(pos["closes"][ -1])
        ret = pos["sign"] * (exit_price - pos["open"]) / pos["open"]
        trades.append(TradeResult(
            symbol=pos["symbol"],
            trade_date=pd.Timestamp(pos["dates"][0]),
            direction=pos["direction"],
            entry=pos["open"],
            exit=exit_price,
            ret=ret,
            hit="eod",
            entry_time=pos["open_time"],
            exit_time=pd.Timestamp(pos["dates"][-1])
        ))
    # Use last timestamp of the day as exit_ts label
    return trades, "eod", pd.Timestamp(positions[0]["dates"][-1])


def _daily_group_returns(
    pred_group: pd.DataFrame,
    intraday_cache: dict,
    basket_tp: float,
    basket_sl: float,
    symbol_tp: float,
    symbol_sl: float,
    top_n_long: int,
    top_n_short: int,
    leverage: float,
) -> tuple[float, List[TradeResult], List[str], List[str], float, float, float, List[str], List[float], str]:
    longs = pred_group.nlargest(top_n_long, "predicted_return")
    shorts = pred_group.nsmallest(top_n_short, "predicted_return")

    trades, hit, _ = _simulate_basket_day(longs, shorts, intraday_cache, basket_tp, basket_sl, symbol_tp, symbol_sl)
    if not trades:
        return 0.0, trades, [], [], 0.0, 0.0, 0.0, [], [], hit

    daily_ret = float(np.mean([t.ret for t in trades])) * leverage
    long_syms = longs["symbol"].tolist()
    short_syms = shorts["symbol"].tolist()
    long_pred = longs["predicted_return"].mean() if not longs.empty else 0.0
    short_pred = shorts["predicted_return"].mean() if not shorts.empty else 0.0
    predicted_portfolio_ret = float(np.mean(list(longs["predicted_return"]) + list(-shorts["predicted_return"]))) if not trades else float((long_pred - short_pred) / 2)

    actuals = pred_group[["symbol", "target"]].dropna()
    actual_top = actuals.nlargest(top_n_long, "target")
    actual_top_syms = actual_top["symbol"].tolist()
    actual_top_rets = actual_top["target"].tolist()

    return daily_ret, trades, long_syms, short_syms, long_pred, short_pred, predicted_portfolio_ret, actual_top_syms, actual_top_rets, hit


def _build_backtest_row(
    trade_date: pd.Timestamp,
    group: pd.DataFrame,
    intraday_cache: dict,
    basket_tp: float,
    basket_sl: float,
    symbol_tp: float,
    symbol_sl: float,
    top_n_long: int,
    top_n_short: int,
    leverage: float,
) -> tuple[dict, List[TradeResult]]:
    daily_ret, trades, long_syms, short_syms, long_pred, short_pred, pred_port_ret, actual_top_syms, actual_top_rets, hit = _daily_group_returns(
        group,
        intraday_cache,
        basket_tp,
        basket_sl,
        symbol_tp,
        symbol_sl,
        top_n_long,
        top_n_short,
        leverage,
    )

    long_count = sum(1 for t in trades if t.direction == "long")
    short_count = len(trades) - long_count

    long_syms_str = ";".join(s.replace('_USDT_USDT', '') for s in long_syms) if long_syms else ""
    short_syms_str = ";".join(s.replace('_USDT_USDT', '') for s in short_syms) if short_syms else ""
    actual_top_str = ";".join(actual_top_syms) if actual_top_syms else ""
    actual_rets_str = ";".join(f"{r:.6f}" for r in actual_top_rets) if actual_top_rets else ""

    row = {
        "trade_date": trade_date,
        "daily_return": daily_ret,
        "predicted_portfolio_return": pred_port_ret,
        "long_symbols": long_syms_str,
        "short_symbols": short_syms_str,
        "long_pred_avg": long_pred,
        "short_pred_avg": short_pred,
        "actual_top_symbols": actual_top_str,
        "actual_top_returns": actual_rets_str,
        "long_count": long_count,
        "short_count": short_count,
        "pnl": daily_ret,
        "basket_hit": hit,
    }
    return row, trades


def _init_backtest_worker(
    intraday_cache: dict,
    basket_tp: float,
    basket_sl: float,
    symbol_tp: float,
    symbol_sl: float,
    top_n_long: int,
    top_n_short: int,
    leverage: float,
) -> None:
    """
    Process initializer: load shared readonly objects/params once per worker.
    """
    global _MP_INTRADAY_CACHE
    global _MP_BASKET_TP, _MP_BASKET_SL, _MP_SYMBOL_TP, _MP_SYMBOL_SL
    global _MP_TOP_N_LONG, _MP_TOP_N_SHORT, _MP_LEVERAGE

    _MP_INTRADAY_CACHE = intraday_cache
    _MP_BASKET_TP = basket_tp
    _MP_BASKET_SL = basket_sl
    _MP_SYMBOL_TP = symbol_tp
    _MP_SYMBOL_SL = symbol_sl
    _MP_TOP_N_LONG = top_n_long
    _MP_TOP_N_SHORT = top_n_short
    _MP_LEVERAGE = leverage


def _build_backtest_row_mp(trade_date: pd.Timestamp, group: pd.DataFrame) -> tuple[dict, List[TradeResult]]:
    """
    Worker entrypoint for ProcessPoolExecutor.
    """
    return _build_backtest_row(
        trade_date,
        group,
        _MP_INTRADAY_CACHE or {},
        _MP_BASKET_TP,
        _MP_BASKET_SL,
        _MP_SYMBOL_TP,
        _MP_SYMBOL_SL,
        _MP_TOP_N_LONG,
        _MP_TOP_N_SHORT,
        _MP_LEVERAGE,
    )


def run_backtest(
    pred_df: pd.DataFrame,
    intraday_cache: dict,
    basket_tp: float,
    basket_sl: float,
    symbol_tp: float,
    symbol_sl: float,
    top_n_long: int,
    top_n_short: int,
    leverage: float = 2.0,
    n_jobs: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    grouped = list(pred_df.groupby("trade_date"))
    rows = []
    all_trades: List[TradeResult] = []

    worker_count = int(n_jobs) if n_jobs is not None else 1
    if worker_count > 1 and len(grouped) > 1:
        with ProcessPoolExecutor(
            max_workers=min(worker_count, len(grouped)),
            initializer=_init_backtest_worker,
            initargs=(
                intraday_cache,
                basket_tp,
                basket_sl,
                symbol_tp,
                symbol_sl,
                top_n_long,
                top_n_short,
                leverage,
            ),
        ) as executor:
            futures = [
                executor.submit(
                    _build_backtest_row_mp,
                    trade_date,
                    group,
                )
                for trade_date, group in grouped
            ]
            for future in tqdm(as_completed(futures), desc="Backtesting", total=len(futures)):
                row, trades = future.result()
                rows.append(row)
                all_trades.extend(trades)
    else:
        for trade_date, group in tqdm(grouped, desc="Backtesting", total=len(grouped)):
            row, trades = _build_backtest_row(
                trade_date,
                group,
                intraday_cache,
                basket_tp,
                basket_sl,
                symbol_tp,
                symbol_sl,
                top_n_long,
                top_n_short,
                leverage,
            )
            rows.append(row)
            all_trades.extend(trades)

    daily = pd.DataFrame(rows).sort_values("trade_date").reset_index(drop=True)

    trades_df = pd.DataFrame([
        {
            "trade_date": t.trade_date,
            "entry_time": t.entry_time,
            "exit_time": t.exit_time,
            "symbol": t.symbol,
            "direction": t.direction,
            "entry_price": t.entry,
            "exit_price": t.exit,
            "return": t.ret,
            "hit": t.hit,
        }
        for t in all_trades
    ])

    trades_df = trades_df.sort_values(["entry_time", "symbol"]).reset_index(drop=True)

    return daily, trades_df


def compute_benchmark(data_root: Path, pair: str, timeframe: str) -> pd.DataFrame:
    df = load_ohlcv(pair, timeframe, data_root)
    df = df.sort_values("date").reset_index(drop=True)
    df["bench_return"] = df["close"].pct_change().fillna(0)
    df["trade_date"] = df["date"]
    return df[["trade_date", "bench_return"]]


def _periods_per_year(timeframe: Optional[str]) -> float:
    """
    Convert timeframe string (e.g. 5m/1h/1d) to number of bars per year.
    Falls back to 365 when timeframe is missing or invalid.
    """
    if not timeframe:
        return 365.0

    try:
        tf_seconds = timeframe_to_seconds(str(timeframe))
        if tf_seconds <= 0:
            return 365.0
        return (365 * 24 * 60 * 60) / tf_seconds
    except Exception:
        return 365.0


def _estimate_fee_drag(
    returns: pd.Series,
    fee: float,
    daily_df: Optional[pd.DataFrame] = None,
    leverage: float = 2.0,
) -> pd.Series:
    """
    Estimate per-period fee drag in return space.

    Assumption:
    - `fee` is one-side fee rate per 1x notional trade.
    - Strategy return uses fixed gross leverage (`leverage`).
    - Fee drag scales with turnover (single-side turnover in [0, 1]):
      fee_drag_t = turnover_t * leverage * (2 * fee).
    """
    base = pd.Series(0.0, index=returns.index, dtype=float)
    if fee <= 0:
        return base

    if daily_df is not None and not daily_df.empty:
        turnover_series = _calculate_turnover_series(daily_df).fillna(0.0)
        if len(turnover_series) == len(returns):
            turnover_aligned = pd.Series(turnover_series.to_numpy(), index=returns.index, dtype=float)
        else:
            turnover_aligned = turnover_series.reindex(returns.index).fillna(0.0)
        turnover_aligned = turnover_aligned.clip(lower=0.0, upper=1.0)
        return turnover_aligned * leverage * 2.0 * float(fee)
    else:
        has_trade = returns.fillna(0.0) != 0.0
        return base.where(~has_trade, leverage * 2.0 * float(fee))


def add_net_pnl(
    daily_df: pd.DataFrame,
    fee: float = 0.0,
    leverage: float = 2.0,
    gross_col: str = "daily_return",
    out_col: str = "net_pnl",
) -> pd.DataFrame:
    """
    Append fee-adjusted return column to the daily backtest frame.
    """
    daily = daily_df.copy()
    if daily.empty:
        daily[out_col] = pd.Series(dtype=float)
        return daily

    gross_returns = pd.to_numeric(daily.get(gross_col, 0.0), errors="coerce").fillna(0.0)
    fee_drag = _estimate_fee_drag(gross_returns, fee=fee, daily_df=daily, leverage=leverage)
    daily[out_col] = gross_returns - fee_drag
    return daily


def metrics(
    daily_returns: pd.Series,
    daily_df: Optional[pd.DataFrame] = None,
    timeframe: Optional[str] = "1d",
    fee: float = 0.0,
    leverage: float = 2.0,
) -> dict:
    """
    Compute portfolio-level performance metrics.

    Args:
        daily_returns: Series of per-bar returns.
        daily_df: Optional daily backtest dataframe used to compute turnover.
            It should include `long_symbols` and `short_symbols` columns.
        timeframe: Bar timeframe string, e.g. "5m", "1h", "1d".
        fee: One-side fee rate per trade (e.g. 0.001 means 0.1%).
        leverage: Strategy leverage used for return scaling and fee drag estimation.

    Returns:
        Dictionary with fee-adjusted metrics.
    """
    if daily_returns.empty:
        return {
            "sharpe": 0.0,
            "cagr": 0.0,
            "max_drawdown": 0.0,
            "turnover": 0.0,
            "returns": 0.0,
            "annualized_return": 0.0,
        }

    gross_returns = pd.to_numeric(daily_returns, errors="coerce").fillna(0.0)
    fee_drag = _estimate_fee_drag(gross_returns, fee=fee, daily_df=daily_df, leverage=leverage)
    net_returns = gross_returns - fee_drag
    periods_per_year = _periods_per_year(timeframe)
    fee_drag_per_year = fee_drag.mean() * periods_per_year
    mean = net_returns.mean()
    std = net_returns.std(ddof=0)
    sharpe = 0.0 if std == 0 else mean / std * np.sqrt(periods_per_year)
    equity = (1 + net_returns).cumprod()
    cagr = equity.iloc[-1] ** (periods_per_year / len(equity)) - 1 if len(equity) > 1 else equity.iloc[-1] - 1
    simple_returns = net_returns.sum()
    annualized_return = simple_returns / len(net_returns) * periods_per_year

    max_drawdown = _calculate_max_drawdown(equity)
    turnover = _calculate_turnover(daily_df) if daily_df is not None and not daily_df.empty else 0.0

    return {
        "sharpe": float(sharpe),
        "cagr": float(cagr),
        "max_drawdown": float(max_drawdown),
        "turnover": float(turnover),
        "returns": float(simple_returns),
        "annualized_return": float(annualized_return),
        "fee_drag_per_year": float(fee_drag_per_year),
    }


def _calculate_max_drawdown(equity: pd.Series) -> float:
    """
    Compute maximum drawdown from an equity curve.

    Args:
        equity: Cumulative equity series starting near 1.

    Returns:
        Maximum drawdown (negative value).
    """
    if equity.empty or len(equity) < 2:
        return 0.0

    running_max = equity.expanding().max()
    drawdown = (equity - running_max) / running_max
    return drawdown.min()


def _calculate_turnover(daily_df: pd.DataFrame) -> float:
    """
    Compute average daily turnover based on changes in long/short target positions
    between consecutive trading days.

    Notes:
    - Positions are direction-aware ("L:symbol" / "S:symbol"), so long<->short
      flips are counted correctly as turnover.
    - Single-side turnover is used, with an upper bound of 100%:
      turnover_t = (entered + exited) / (2 * max(current_n, prev_n))
    - The first day is initial setup and excluded from the turnover average.
    """
    turnover_series = _calculate_turnover_series(daily_df)
    if turnover_series.empty:
        return 0.0
    valid = turnover_series.dropna()
    return float(valid.mean()) if not valid.empty else 0.0


def _calculate_turnover_series(daily_df: pd.DataFrame) -> pd.Series:
    """
    Calculate single-side daily turnover series from target positions.
    Returns a Series indexed by `trade_date`, where first day is NaN.
    """
    if daily_df.empty:
        return pd.Series(dtype=float)

    daily_df = daily_df.copy().sort_values("trade_date").reset_index(drop=True)
    if "trade_date" not in daily_df.columns:
        return pd.Series(dtype=float)

    def parse_symbols(s: object) -> set[str]:
        if pd.isna(s) or s == "":
            return set()
        symbols = str(s).split(";")
        return set(x.strip() for x in symbols if x.strip())

    prev_positions: Optional[set[str]] = None
    turnover_values: List[float] = []
    turnover_index: List[pd.Timestamp] = []

    for _, row in daily_df.iterrows():
        trade_date = pd.Timestamp(row["trade_date"])
        long_symbols = parse_symbols(row.get("long_symbols", ""))
        short_symbols = parse_symbols(row.get("short_symbols", ""))

        current_positions = {f"L:{sym}" for sym in long_symbols} | {f"S:{sym}" for sym in short_symbols}

        if prev_positions is None:
            prev_positions = current_positions
            turnover_values.append(np.nan)
            turnover_index.append(trade_date)
            continue

        entered = current_positions - prev_positions
        exited = prev_positions - current_positions
        total_holding = max(len(current_positions), len(prev_positions))
        daily_turnover = (len(entered) + len(exited)) / (2.0 * total_holding) if total_holding > 0 else 0.0
        daily_turnover = float(np.clip(daily_turnover, 0.0, 1.0))
        turnover_values.append(daily_turnover)
        turnover_index.append(trade_date)
        prev_positions = current_positions

    return pd.Series(turnover_values, index=turnover_index, dtype=float)
def rank_metrics(pred_df: pd.DataFrame, top_n_long: int, top_n_short: int) -> dict:
    pred_df = pred_df.dropna(subset=["predicted_return", "target", "trade_date", "symbol"])
    if pred_df.empty:
        return {"spearman": 0.0, "top_hit_rate": 0.0, "bottom_hit_rate": 0.0}
    spearman = float(spearmanr(pred_df["predicted_return"], pred_df["target"]).correlation or 0.0)

    top_hits = []
    bottom_hits = []
    for _, g in pred_df.groupby("trade_date"):
        pred_top = set(g.nlargest(top_n_long, "predicted_return")["symbol"].tolist())
        true_top = set(g.nlargest(top_n_long, "target")["symbol"].tolist())
        pred_bottom = set(g.nsmallest(top_n_short, "predicted_return")["symbol"].tolist())
        true_bottom = set(g.nsmallest(top_n_short, "target")["symbol"].tolist())
        if top_n_long:
            top_hits.append(len(pred_top & true_top) / top_n_long)
        if top_n_short:
            bottom_hits.append(len(pred_bottom & true_bottom) / top_n_short)
    top_hit_rate = float(np.mean(top_hits)) if top_hits else 0.0
    bottom_hit_rate = float(np.mean(bottom_hits)) if bottom_hits else 0.0
    return {"spearman": spearman, "top_hit_rate": top_hit_rate, "bottom_hit_rate": bottom_hit_rate}


def explain_model_shap(
    model,
    X: np.ndarray,
    feature_cols: List[str],
    output_dir: Path,
    max_display: int = 20,
    sample_size: int = 1000,
) -> dict:
    """
    Explain an XGBoost model with SHAP and save visualization artifacts.

    Args:
        model: Trained XGBRanker model.
        X: Feature matrix.
        feature_cols: Feature name list.
        output_dir: Output directory for SHAP plots/files.
        max_display: Max number of features shown in summary plots.
        sample_size: Number of sampled rows for SHAP computation.

    Returns:
        A dictionary containing SHAP outputs or fallback feature importance.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print()
    print('=' * 80)
    print('SHAP Analysis - Explaining Model Predictions')
    print('=' * 80)

    # Clean invalid values before SHAP.
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Remove zero-variance columns.
    valid_cols_mask = []
    for i in range(X.shape[1]):
        col = X[:, i]
        if np.var(col) > 1e-10 and not np.allclose(col, 0):
            valid_cols_mask.append(i)

    if len(valid_cols_mask) == 0:
        print('Warning: All features have zero variance, skipping SHAP analysis')
        return None

    X_filtered = X[:, valid_cols_mask]
    feature_cols_filtered = [feature_cols[i] for i in valid_cols_mask]
    print(f'Filtered from {len(feature_cols)} to {len(feature_cols_filtered)} features (removed zero-variance columns)')

    # Sample for speed.
    if len(X_filtered) > sample_size:
        indices = np.random.choice(len(X_filtered), sample_size, replace=False)
        X_sample = X_filtered[indices]
        print(f'Sampled {sample_size} rows from {len(X_filtered)} for SHAP analysis')
    else:
        X_sample = X_filtered
        print(f'Using all {len(X_filtered)} rows for SHAP analysis')

    # Normalize inputs to improve numerical stability.
    X_sample_scaled = np.copy(X_sample).astype(np.float64)

    X_global_min = np.nanmin(X_sample_scaled)
    X_global_max = np.nanmax(X_sample_scaled)
    X_global_range = X_global_max - X_global_min

    if X_global_range > 0:
        X_sample_scaled = (X_sample_scaled - X_global_min) / X_global_range
    else:
        X_sample_scaled = np.zeros_like(X_sample_scaled)

    for i in range(X_sample_scaled.shape[1]):
        col = X_sample_scaled[:, i]
        col_max = np.nanmax(col)
        col_min = np.nanmin(col)
        col_std = np.nanstd(col)
        col_mean = np.nanmean(col)

        if col_std < 1e-10 or col_max == col_min:
            X_sample_scaled[:, i] = 0.0
        else:
            X_sample_scaled[:, i] = (col - col_mean) / (col_std + 1e-10)
            X_sample_scaled[:, i] = np.clip(X_sample_scaled[:, i], -5, 5) / 5.0

    X_sample_scaled = np.nan_to_num(X_sample_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    X_sample_scaled = np.clip(X_sample_scaled, -1.0, 1.0)

    print('Computing SHAP values...')
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample_scaled)
        if shap_values is None or (isinstance(shap_values, np.ndarray) and shap_values.size == 0):
            raise ValueError('SHAP values returned empty')
        shap_values = np.nan_to_num(shap_values, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception as e:
        print(f'Warning: SHAP computation failed: {e}')
        print('Falling back to simple feature importance...')

        feature_importance = np.abs(X_sample_scaled).mean(axis=0)
        importance_df = pd.DataFrame({
            'feature': feature_cols_filtered,
            'mean_abs_shap': feature_importance,
        }).sort_values('mean_abs_shap', ascending=False)

        importance_df.to_csv(output_dir / 'shap_feature_importance.csv', index=False)

        print()
        print('=' * 80)
        print(f'Top {min(15, len(importance_df))} Features (Simple Importance)')
        print('=' * 80)
        print(f"{'Rank':<6} {'Feature':<40} {'Mean Abs Value':>15}")
        print('-' * 80)
        for idx, row in importance_df.head(15).iterrows():
            rank = importance_df.index.get_loc(idx) + 1
            print(f"{rank:<6} {row['feature']:<40} {row['mean_abs_shap']:>15.6f}")

        print()
        print(f"Feature importance saved to: {output_dir / 'shap_feature_importance.csv'}")
        print('=' * 80)
        print()

        return {
            'feature_importance': importance_df,
            'method': 'simple_importance',
            'note': 'SHAP computation failed, used simple importance instead',
        }

    feature_cols = feature_cols_filtered

    print('Generating SHAP summary plot (bar)...')
    plt.figure(figsize=(10, 8))
    try:
        shap.summary_plot(
            shap_values,
            X_sample_scaled,
            feature_names=feature_cols,
            max_display=max_display,
            plot_type='bar',
            show=False,
        )
    except Exception as e:
        print(f'Warning: summary plot failed: {e}')
        plt.close()
    plt.tight_layout()
    plt.savefig(output_dir / 'shap_summary_bar.png', dpi=300, bbox_inches='tight')
    plt.close()

    print('Generating SHAP summary plot (dot)...')
    plt.figure(figsize=(10, 8))
    try:
        shap.summary_plot(
            shap_values,
            X_sample_scaled,
            feature_names=feature_cols,
            max_display=max_display,
            plot_type='dot',
            show=False,
        )
    except Exception as e:
        print(f'Warning: dot plot failed: {e}')
        plt.close()
    plt.tight_layout()
    plt.savefig(output_dir / 'shap_summary_dot.png', dpi=300, bbox_inches='tight')
    plt.close()

    print('Computing feature importance rankings...')
    feature_importance = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'mean_abs_shap': feature_importance,
    }).sort_values('mean_abs_shap', ascending=False)

    importance_df.to_csv(output_dir / 'shap_feature_importance.csv', index=False)

    top_features = importance_df.head(5)['feature'].tolist()
    print()
    print(f'Generating dependence plots for top {len(top_features)} features...')
    for feat in top_features:
        try:
            feat_idx = feature_cols.index(feat)
            plt.figure(figsize=(8, 6))
            shap.dependence_plot(
                feat_idx,
                shap_values,
                X_sample_scaled,
                feature_names=feature_cols,
                show=False,
            )
            plt.tight_layout()
            safe_name = feat.replace('/', '_').replace(':', '_')
            plt.savefig(output_dir / f'shap_dependence_{safe_name}.png', dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f'Warning: dependence plot for {feat} failed: {e}')
            plt.close()

    print()
    print('=' * 80)
    print(f'Top {min(15, len(importance_df))} Features by Mean |SHAP|')
    print('=' * 80)
    print(f"{'Rank':<6} {'Feature':<40} {'Mean |SHAP|':>15}")
    print('-' * 80)
    for idx, row in importance_df.head(15).iterrows():
        rank = importance_df.index.get_loc(idx) + 1
        print(f"{rank:<6} {row['feature']:<40} {row['mean_abs_shap']:>15.6f}")

    print()
    print(f'SHAP analysis complete. Results saved to: {output_dir}')
    print('  - shap_summary_bar.png: Feature importance bar chart')
    print('  - shap_summary_dot.png: Feature impact distribution')
    print('  - shap_feature_importance.csv: Numerical rankings')
    print(f'  - shap_dependence_*.png: Top {len(top_features)} feature dependence plots')
    print('=' * 80)
    print()

    return {
        'shap_values': shap_values,
        'feature_importance': importance_df,
        'explainer': explainer,
    }
