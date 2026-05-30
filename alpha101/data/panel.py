from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from alpha101.data.market_metadata import get_pair_market_caps


TIMEFRAME_SECONDS = {
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "2h": 7200,
    "4h": 14400,
    "6h": 21600,
    "8h": 28800,
    "12h": 43200,
    "1d": 86400,
}


def build_path(pair: str, timeframe: str, data_root: Path) -> Path:
    return data_root / f"{pair}-{timeframe}-futures.feather"


def build_funding_path(pair: str, data_root: Path) -> Path:
    return data_root / f"{pair}-8h-funding_rate.feather"


def timeframe_to_timedelta(timeframe: str) -> pd.Timedelta:
    try:
        return pd.Timedelta(seconds=TIMEFRAME_SECONDS[str(timeframe)])
    except KeyError as exc:
        raise ValueError(f"Unsupported timeframe: {timeframe}") from exc


def _parse_dt_utc(value) -> pd.Timestamp | None:
    if value is None:
        return None
    return pd.to_datetime(value, utc=True)


def compute_window(
    timeframe: str,
    lookback: int,
    train_bars: int | None,
    test_start_date,
    test_end_date,
    buffer: int = 365,
) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    start_dt = _parse_dt_utc(test_start_date)
    end_dt = _parse_dt_utc(test_end_date)
    bar_delta = timeframe_to_timedelta(timeframe)

    window_start = None
    if start_dt is not None:
        total_history_bars = (train_bars or 0) + lookback + buffer
        window_start = start_dt - total_history_bars * bar_delta

    window_end = end_dt + bar_delta if end_dt is not None else None
    return window_start, window_end


def load_ohlcv(
    pair: str,
    timeframe: str,
    data_root: Path,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    supply_dict: Optional[dict[str, float]] = None,
) -> pd.DataFrame:
    path = build_path(pair, timeframe, data_root)
    funding_path = build_funding_path(pair, data_root)
    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {path}")

    df = pd.read_feather(path)
    if funding_path.exists():
        funding_df = pd.read_feather(funding_path).rename(columns={"open": "funding"})
        df = df.merge(funding_df[["date", "funding"]], on="date", how="left")
        df["funding"] = df["funding"].ffill()
    else:
        df["funding"] = 0.0

    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.sort_values("date").reset_index(drop=True)
    df["symbol"] = pair
    df["target"] = df["close"].pct_change(fill_method=None).shift(-1)
    df["vwap"] = (df["high"] + df["low"] + df["close"]) / 3
    if supply_dict and pair in supply_dict and len(df):
        df["cap"] = df["close"] * supply_dict[pair]
    else:
        df["cap"] = np.nan

    if start is not None:
        df = df[df["date"] >= start]
    if end is not None:
        df = df[df["date"] <= end]

    required = {
        "date",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "vwap",
        "target",
        "cap",
        "funding",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return df


def check_missing_by_symbol(df: pd.DataFrame, max_missing: int = 465) -> pd.DataFrame:
    df = df.copy()
    df["symbol"] = df["symbol"].astype("category")
    total_num = max(df.groupby("symbol", observed=True)["open"].count())
    missing_stats = df.groupby("symbol", observed=True)["open"].apply(
        lambda x: total_num - x.count()
    )
    symbols_to_remove = missing_stats[missing_stats > max_missing].index.tolist()
    if symbols_to_remove:
        df = df[~df["symbol"].isin(symbols_to_remove)]
    return df


def build_wide_df(
    pairs: list[str],
    lookback: int,
    data_root: Path,
    timeframe: str,
    *,
    train_bars: int | None = None,
    test_start_date=None,
    test_end_date=None,
    buffer: int = 365,
    include_market_cap: bool = True,
) -> pd.DataFrame:
    window_start, window_end = compute_window(
        timeframe,
        lookback,
        train_bars,
        test_start_date,
        test_end_date,
        buffer=buffer,
    )
    supply_dict = None
    if include_market_cap:
        market_caps = get_pair_market_caps(pairs)
        supply_dict = market_caps[["pair", "circulating_supply"]].set_index("pair")[
            "circulating_supply"
        ].to_dict()

    frames = []
    for pair in tqdm(pairs, desc="Loading pairs", total=len(pairs)):
        try:
            frames.append(
                load_ohlcv(
                    pair,
                    timeframe,
                    data_root,
                    start=window_start,
                    end=window_end,
                    supply_dict=supply_dict,
                )
            )
        except FileNotFoundError as exc:
            print(f"Warning: {exc}. Skipping pair {pair}.")

    if not frames:
        raise FileNotFoundError(f"No freqtrade feather files found in {data_root}")

    panel = pd.concat(frames, ignore_index=True)
    panel = check_missing_by_symbol(panel)
    panel = panel.set_index(["date", "symbol"]).sort_index()
    panel.index = panel.index.set_levels(panel.index.levels[1].astype("category"), level=1)
    return panel.unstack(level="symbol").iloc[buffer:]


def sample_indices_after_agg(n_rows: int, k_bars: int) -> np.ndarray:
    if n_rows <= 0:
        return np.array([], dtype=np.int64)
    k_bars = max(1, int(k_bars))
    start = k_bars - 1
    return np.arange(start, n_rows, k_bars, dtype=np.int64)


def nan_rowwise_corr(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    out = np.full(x.shape[0], np.nan, dtype=float)
    for i in range(x.shape[0]):
        mask = np.isfinite(x[i]) & np.isfinite(y[i])
        if int(mask.sum()) < 3:
            continue
        xv = x[i, mask]
        yv = y[i, mask]
        if np.std(xv) == 0 or np.std(yv) == 0:
            continue
        out[i] = np.corrcoef(xv, yv)[0, 1]
    return out
