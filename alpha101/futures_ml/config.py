from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import json

@dataclass
class ProjectConfig:
    data_root: Path = Path("user_data/data/binance/futures")
    pairs: List[str] = field(default_factory=list)
    timeframe: str = "4h"
    pre_buffer_candles: int = 200
    trade_per_k_bars: int = 1  # 4 hours of 1-minute bars
    intraday_timeframe: str = "15m"
    lookback_days: int = 0
    train_split: float = 0.9
    take_profit: float = 0.30
    stop_loss: float = 0.30
    leverage: float = 1.0
    # Single-symbol TP/SL (uniform for all symbols) distinct from basket TP/SL
    symbol_tp: float = 0.50
    symbol_sl: float = 0.50
    top_n_long: int = 20
    top_n_short: int = 20
    min_train_bars: int = 360 * 6
    train_bars: int = 365 *6
    train_per_pair: bool = False
    retrain_every_bars: int = 30 * 6
    smoothing_span: int = 1
    benchmark_pair: str = "BTC_USDT_USDT"
    output_dir: Path = Path("alpha101/futures_ml/output")
    test_start_date: Optional[str] = "2025-01-01"  # e.g., "2025-11-01"
    test_end_date: Optional[str] = "2026-03-30"    # e.g., "2025-12-31"
    strategy_config_path: Path = Path("user_data/strategies/SmallCapStrategy.json")
    # strategy_config_path: Path = Path("user_data/strategies/SmallCapStrategy_too_small.json")
    
    # strategy_config_path: Path = Path("user_data/MLBasketStrategy.json")
    alpha_expression_path: Path = Path("factor_search_results/4h/factors_to_keep.csv")
    fee: float = 0.0005  # 0.0005% per trade, adjust as needed

def get_config() -> ProjectConfig:
    cfg = ProjectConfig()
    cfg.pairs = [e.replace("/", "_").replace(":", "_") for e in json.load(open(cfg.strategy_config_path))["exchange"]["pair_whitelist"]]
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    return cfg
