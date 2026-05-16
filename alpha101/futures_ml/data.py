from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm
MAX_ABS_ALPHA = 1e6
module_path = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(module_path))
import os
from alpha101.world_quant.Alpha101_code_1 import Alphas
from alpha101.world_quant.fastengine import FastExpressionEngine
from freqtrade.exchange.exchange_utils_timeframe import timeframe_to_seconds
_ALPHA_MODULE = None
from typing import Dict, List, Optional
from alpha101.data_helper.get_cap import get_pair_market_caps


def _load_alpha_module():
    global _ALPHA_MODULE
    if _ALPHA_MODULE is None:
        module_path = Path(__file__).resolve().parents[1] / "world_quant" / "Alpha101_code_1.py"
        spec = importlib.util.spec_from_file_location("alpha101_world_quant", module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load alpha module from {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _ALPHA_MODULE = module
    return _ALPHA_MODULE


def build_path(pair: str, timeframe: str, data_root: Path) -> Path:
    return data_root / f"{pair}-{timeframe}-futures.feather"
def build_funding_path(pair: str, data_root: Path) -> Path:
    return data_root / f"{pair}-8h-funding_rate.feather"


def _parse_dt_utc(value) -> pd.Timestamp | None:
    if value is None:
        return None
    return pd.to_datetime(value, utc=True)


def _timeframe_to_timedelta(timeframe: str) -> pd.Timedelta:
    return pd.Timedelta(seconds=timeframe_to_seconds(str(timeframe)))


def load_ohlcv(
    pair: str,
    timeframe: str,
    data_root: Path,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    supply_dict: Optional[Dict[str, float]] = None
) -> pd.DataFrame:
    path = build_path(pair, timeframe, data_root)
    funding_path = build_funding_path(pair, data_root)
    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {path}")
    df = pd.read_feather(path)
    if funding_path.exists():
        df_funding = pd.read_feather(funding_path)
        df_funding = df_funding.rename(columns={"open": "funding"})
        df = df.merge(df_funding[["date", "funding"]], on="date", how="left")
        df["funding"] = df["funding"].ffill()
    else:
        df["funding"] = np.nan
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.sort_values("date").reset_index(drop=True)
    
    df['symbol'] = pair
    df["target"] = df["close"].pct_change(fill_method=None).shift(-1)
    df["vwap"] = (df["high"] + df["low"] + df["close"]) / 3 # 需要确认 vwap 的定义，使用次级数据进行计算，这里进行了简化    
    if supply_dict and pair in supply_dict and len(df):
        df['cap'] = df['close'] * supply_dict[pair]
    else:
        df['cap'] = np.nan
    if start is not None:
        df = df[df["date"] >= start]
    if end is not None:
        df = df[df["date"] <= end]
    # backfill funding_rate
    required = {"date", "symbol", "open", "high", "low", "close", "volume", "vwap", "target", "cap", "funding"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    return df



def compute_alphas(wide_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    alpha_module = _load_alpha_module()
    alpha_output = alpha_module.get_alpha_parallel(wide_dict.copy())
    # alpha_output = alpha_module.get_alpha_good15(wide_dict.copy())
    
    # alpha_output = alpha_module.get_alpha_good15_test(wide_dict.copy())
    # alpha_output = alpha_module.get_alpha_088_v(wide_dict.copy())
    # alpha_output = alpha_module.get_alpha_077_v(wide_dict.copy())
    # alpha_output = alpha_module.get_alpha_011_v(wide_dict.copy())
    # alpha_output = alpha_module.get_alpha_033_v(wide_dict.copy())
    # alpha_output = alpha_module.get_alpha_enhanced(wide_dict.copy())

    # alpha_cols = sorted([col for (col,_) in alpha_output.columns if col.startswith("alpha")])
    # # Z-score standardization per factor (time-series)
    # factors = {}
    # for col in alpha_cols:
    #     df_factor = alpha_output[col].replace([np.inf, -np.inf], np.nan)
    #     # factors[col] = df_factor
    #     factors[col] = (df_factor - df_factor.mean(axis=0) ) / (df_factor.std(axis=0) + 1e-12)
    # factors = factors.mask(factors.abs() > MAX_ABS_ALPHA)
    return alpha_output


def _lag_alpha_features(alpha_df: dict, lookback: int) -> pd.DataFrame:
    for col in [col for col in alpha_df.columns if col.startswith("alpha")]:
        for lag in range(1, lookback + 1):
            alpha_df[f"{col}_lag{lag}"] = alpha_df[col].shift(lag)
    return alpha_df


def long_to_wide(
    df: pd.DataFrame,
    date_col: str = 'date',
    symbol_col: str = 'symbol',
    value_cols: Optional[List[str]] = None
) -> Dict[str, pd.DataFrame]:
    """
    将长格式 OHLCV 数据转换为多个宽格式 DataFrame（每个字段一个）

    参数:
    ----------
    df : pd.DataFrame
        长格式数据，每行代表某股票在某日的一个观测值
    date_col : str, 默认 'date'
        日期列名
    symbol_col : str, 默认 'symbol'
        股票代码列名
    value_cols : list of str, 可选
        要转换的数值列名列表（如 ['open', 'close', 'volume']）。
        如果为 None，则自动使用除 date_col 和 symbol_col 外的所有列。

    返回:
    ----------
    dict: {field_name: wide_dataframe}
        每个字段对应一个宽表：
        - 行索引：日期（去重、排序）
        - 列：股票代码
        - 值：对应字段的数值（缺失用 NaN 填充）
    """
    # 输入校验
    if date_col not in df.columns:
        raise ValueError(f"date_col '{date_col}' not found in DataFrame columns.")
    if symbol_col not in df.columns:
        raise ValueError(f"symbol_col '{symbol_col}' not found in DataFrame columns.")
    
    # 自动推断 value_cols
    if value_cols is None:
        value_cols = [col for col in df.columns if col not in (date_col, symbol_col)]
    
    # 转换日期列（可选，但推荐）
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    # 构建宽表字典
    wide_dict = {}
    for col in value_cols:
        wide_df = df.pivot(index=date_col, columns=symbol_col, values=col)
        wide_df = wide_df.sort_index()  # 按日期排序
        wide_dict[col] = wide_df
    
    return wide_dict
def merge_wide_dicts_to_long(wide_dict, factors_dict) -> pd.DataFrame:
    # 合并所有宽表到一个大字典
    all_data = wide_dict.copy()
    for col in factors_dict.keys():
        all_data[col] = factors_dict[col]
    series_list = []
    # 拼接所有字段（使用 concat 替代逐次 join，提高速度）
    for field, df in all_data.items():
        # df 是宽表（index=date, columns=symbol）
        df.columns.name = 'symbol' 
        s = df.stack()  # → Series with MultiIndex (date, symbol)
        s.name = field  # 设置列名
        series_list.append(s)

    # Step 2: 沿 axis=1 拼接（此时索引对齐，不会重复）
    result = pd.concat(series_list, axis=1)  # shape: (N, num_fields), index=(date, symbol)

    # Step 3: 最后一次性把索引变列
    result = result.reset_index() 
    print('Merged long dataset preview:')
    print(result.tail()[['date','symbol','open','high','low','close','volume']])
    print('Merged long dataset shape:', result.shape)
    print('Columns:', result.columns.tolist())
    return result.reset_index()


def build_pair_dataset(daily_df: pd.DataFrame, symbol: str, lookback: int) -> pd.DataFrame:
    alpha_df = compute_alphas(daily_df)
    features = _lag_alpha_features(alpha_df, lookback)
    dataset = features.copy()
    dataset["symbol"] = symbol
    dataset["asof_date"] = daily_df["date"]
    dataset["trade_date"] = daily_df["date"] + pd.Timedelta(days=1)
    dataset["target"] = daily_df["close"].pct_change().shift(-1)
    dataset["daily_open"] = daily_df["open"]
    dataset["daily_close"] = daily_df["close"]
    # dataset = dataset.dropna()
    return dataset.reset_index(drop=True)

def clean_nan_targets(df: pd.DataFrame) -> pd.DataFrame:
    # 昨天前的asof_date的nan rows （因为昨天的未来一日收益率需要今天的数据）
    nan_rows_cond = df["target"].isna() & (df["asof_date"] < pd.Timestamp.utcnow().normalize() - pd.Timedelta(days=1))  
    nan_rows = df[nan_rows_cond]
    if not nan_rows.empty:
        print("Found rows with NaN in 'target':")
        # 打印关键列，如 asof_date, symbol, target（以及其他你关心的）
        print(nan_rows[["asof_date", "symbol", "target"]].head(2).to_string(index=False))
        print("...")
        print(nan_rows[["asof_date", "symbol", "target"]].tail(2).to_string(index=False))
        removed_pairs = nan_rows["symbol"].unique()
        print(f"Removing pairs with NaN targets: {removed_pairs}")
        removed_pairs = []
    else:
        print("No NaN in 'target' column.")
        removed_pairs = []
    cleaned_pairs = set(df["symbol"].unique()) - set(removed_pairs)
    cleaned = df[df['symbol'].isin(cleaned_pairs) & ~nan_rows_cond].reset_index(drop=True)
    return cleaned, cleaned_pairs

def _compute_window(
    timeframe: str,
    lookback: int,
    train_bars: int | None,
    test_start_date,
    test_end_date,
    buffer = 365
) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    start_dt = _parse_dt_utc(test_start_date)
    end_dt = _parse_dt_utc(test_end_date)
    bar_delta = _timeframe_to_timedelta(timeframe)

    window_start = None
    if start_dt is not None:
        total_history_bars = (train_bars or 0) + lookback + buffer
        window_start = start_dt - total_history_bars * bar_delta

    window_end = end_dt + bar_delta if end_dt is not None else None
    return window_start, window_end
def check_missing_by_symbol(df):
    """
    检查每个symbol的open列缺失值数量
    """
    df['symbol'] = df['symbol'].astype('category')
    total_num = max(df.groupby('symbol')['open'].count())
    missing_stats = df.groupby('symbol')['open'].apply(
        lambda x: total_num - x.count()
    ).reset_index(name='missing_count')
    
    missing_stats = missing_stats[missing_stats['missing_count'] > 0]
    
    missing_stats = missing_stats.sort_values('missing_count', ascending=False)
    if missing_stats.empty:
        print("No missing 'open' values found for any symbol.")
    else:
        print("Symbols with missing 'open' values:")
        print(missing_stats)
    # 去掉缺失值过多的symbol
    threshold = 100+365  # 可以根据实际情况调整阈值
    symbols_to_remove = missing_stats[missing_stats['missing_count'] > threshold]['symbol'].tolist()
    if symbols_to_remove:
        print(f"Removing symbols with more than {threshold} missing 'open' values")
        print(f"Before removal, dataset has {df['symbol'].nunique()} symbols.")
        df = df[~df['symbol'].isin(symbols_to_remove)]
        print(f"After removal, dataset has {df['symbol'].nunique()} symbols.")
    else:        
        print(f"No symbols have more than {threshold} missing 'open' values.")

    return missing_stats, df
def build_wide_df(
    pairs: List[str],
    lookback: int,
    data_root: Path,
    timeframe: str,
    *,
    train_bars: int | None = None,
    test_start_date=None,
    test_end_date=None,
    buffer = 365
) -> pd.DataFrame:
    window_start, window_end = _compute_window(timeframe, lookback, train_bars, test_start_date, test_end_date, buffer=buffer)
    print('Loading data with window:')
    print(f"Start: {window_start}, End: {window_end}, train_bars: {train_bars}, lookback: {lookback}, buffer: {buffer}") 
    frames = []
    supply_dict = get_pair_market_caps(pairs)[['pair', 'circulating_supply']].set_index('pair')['circulating_supply'].to_dict()
    for pair in tqdm(pairs, desc="Loading pairs", total=len(pairs)):
        try:
            daily_df = load_ohlcv(pair, timeframe, data_root, start=window_start, end=window_end, supply_dict=supply_dict)
            # frames.append(build_pair_dataset(daily_df, pair, lookback))
            frames.append(daily_df)
        except FileNotFoundError as e:
            print(f"Warning: {e}. Skipping pair {pair}.")
    panel = pd.concat(frames, ignore_index=True)
    panel = check_missing_by_symbol(panel)[1]
    panel = panel.set_index(["date", "symbol"]).sort_index()
    panel.index = panel.index.set_levels(
        panel.index.levels[1].astype("category"), level=1
    )
    panel = panel.unstack(level='symbol') # 转为宽表
    panel = panel.iloc[buffer:]
    return panel

def build_full_dataset(
    pairs: List[str],
    lookback: int,
    data_root: Path,
    timeframe: str,
    *,
    train_bars: int | None = None,
    test_start_date=None,
    test_end_date=None,
    alpha_expression_path=None
) -> pd.DataFrame:
    buffer = 365  # 额外缓冲天数，确保有足够数据计算alpha和训练
    window_start, window_end = _compute_window(timeframe, lookback, train_bars, test_start_date, test_end_date, buffer=buffer)
    print("Building full dataset with history window:")
    print(f"Start: {window_start}, End: {window_end}, train_bars: {train_bars}, lookback: {lookback}, buffer: {buffer}")
    frames = []
    supply_dict = get_pair_market_caps(pairs)[['pair', 'circulating_supply']].set_index('pair')['circulating_supply'].to_dict()
    for pair in tqdm(pairs, desc="Loading pairs", total=len(pairs)):
        try:
            daily_df = load_ohlcv(pair, timeframe, data_root, start=window_start, end=window_end, supply_dict=supply_dict)
            # frames.append(build_pair_dataset(daily_df, pair, lookback))
            frames.append(daily_df)
        except FileNotFoundError as e:
            print(f"Warning: {e}. Skipping pair {pair}.")
    panel = pd.concat(frames, ignore_index=True)
    panel = check_missing_by_symbol(panel)[1]
    panel = panel.set_index(["date", "symbol"]).sort_index()
    panel.index = panel.index.set_levels(
        panel.index.levels[1].astype("category"), level=1
    )
    panel = panel.unstack(level='symbol') # 转为宽表

    stock = Alphas(panel)  # 初始化股票数据
    engine = FastExpressionEngine(stock)
    if os.path.exists(alpha_expression_path):
        alphas_expr = pd.read_csv(alpha_expression_path).head(50)  # 只取前50个因子，避免计算过慢
    else:
        raise FileNotFoundError(f"Alpha expression file not found: {alpha_expression_path}")
    # .iloc[1:].sample(100, random_state=42)
    # alphas_expr = pd.read_csv('factor_search_results/a_selected_out.csv')
    # alphas_expr = pd.read_csv('factor_search_results/a_merged_latest.csv').head(50)
    # alphas_expr = pd.read_csv('factor_search_results/a_merged_latest.csv')
    alpha_dict = {row['factor_name']: row['expression'] for _, row in alphas_expr.iterrows()}
    df_alpha = engine.evaluate_batch(alpha_dict, progress_bar=True)  # 计算所有alpha因子，得到宽表格式的结果
    df_wide = pd.concat([panel, df_alpha], axis=1)  # 将原始数据和alpha因子结果拼接在一起
    df_wide = df_wide.iloc[buffer:] # 丢弃前 buffer 天，确保后续 alpha 计算和训练有足够数据
    print("Wide dataset preview:")
    print(df_wide)
    df_long = df_wide.stack(level=1, future_stack=True).reset_index()  # 转为长表，展开 symbol 维度
    df_alpha = _lag_alpha_features(df_long, lookback)
    df_alpha["asof_date"] = df_alpha["date"]
    df_alpha["trade_date"] = df_alpha["date"] + pd.Timedelta(days=1)
    df_alpha["daily_open"] = df_alpha["open"]
    df_alpha["daily_close"] = df_alpha["close"]
    # alpha_df = alpha_df.dropna()
    # 截面标准化, 对alpha开头的列进行标准化
    # alpha_cols = [c for c in full.columns if c.startswith("alpha")]
    # full[alpha_cols] = full.groupby("asof_date")[alpha_cols].transform(lambda g: (g - g.mean()) / (g.std()+1e-12) )
    # panel, pairs = clean_nan_targets(panel)   
    return df_alpha


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.replace([np.inf, -np.inf], np.nan)
    cleaned = cleaned.dropna()
    return cleaned.reset_index(drop=True)


def load_intraday_cache(pairs: List[str], timeframe: str, data_root: Path) -> dict[str, pd.DataFrame]:
    cache = {}
    for pair in pairs:
        try:
            intraday = load_ohlcv(pair, timeframe, data_root)
            cache[pair] = intraday
        except FileNotFoundError as e:
            print(f"Warning: {e}. Skipping pair {pair}.")
    return cache
if __name__ == "__main__":
    from alpha101.futures_ml.config import get_config
    cfg = get_config()
    dataset = build_full_dataset(
        pairs=cfg.pairs,
        lookback=cfg.lookback_days,
        data_root=cfg.data_root,
        timeframe=cfg.timeframe,
        train_bars=cfg.train_bars,
        test_start_date=cfg.test_start_date,
        test_end_date=cfg.test_end_date,
        alpha_expression_path=cfg.alpha_expression_path,
    )
    print("Final dataset preview:")
    print(dataset.head())
    print(dataset[['symbol','cap']])
    print("Final dataset shape:", dataset.shape)
