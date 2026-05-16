from pathlib import Path
import sys

import pandas as pd
from tqdm import tqdm
import os
repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root))

from alpha101.futures_ml import config
from alpha101.futures_ml.data import build_path


def load_ohlcv(
    pair: str,
    timeframe: str,
    data_root: Path,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    path = build_path(pair, timeframe, data_root)
    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {path}")
    df = pd.read_feather(path)
    if "date" not in df.columns:
        raise ValueError(f"Expected 'date' column in {path}")
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
    assert df["date"].dtype == "datetime64[ns]", f"日期类型错误：{df['date'].dtype}"
    df = df.sort_values("date").reset_index(drop=True)
    if start is not None:
        df = df[df["date"] >= start]
    if end is not None:
        df = df[df["date"] <= end]
    df['symbol'] = pair
    df["vwap"] = (df["high"] + df["low"] + df["close"]) / 3 # 需要确认 vwap 的定义，使用次级数据进行计算，这里进行了简化    
    required = {"date", "symbol", "open", "high", "low", "close", "volume", "vwap"}
    missing = required - set(df.columns)
    df['factor'] = 1
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    return df
if __name__ == "__main__":
    cfg = config.get_config()
    frames = []
    for pair in tqdm(cfg.pairs, desc="Loading pairs", total=len(cfg.pairs)):
        daily_df = load_ohlcv(pair, cfg.timeframe, cfg.data_root)
        output_dir = r'E:\MatchAndProject\quant\qlib\my_data'
        daily_df.to_parquet(os.path.join(output_dir,f'{pair}.parquet'))
        # frames.append(daily_df)
    # df = pd.concat(frames, axis=0)
    # df.to_parquet('alpha101/futures_ml/combined_long.parquet')
    # print("saved at alpha101/futures_ml/combined_long.parquet")