from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import config
from .backtest import add_net_pnl, compute_benchmark, metrics, rank_metrics, run_backtest
from .data import load_intraday_cache
from .visualize import plot_equity


def _filter_window(df: pd.DataFrame, start, end) -> pd.DataFrame:
    if start:
        start_dt = pd.to_datetime(start, utc=True)
        df = df[df["trade_date"] >= start_dt]
    if end:
        end_dt = pd.to_datetime(end, utc=True)
        df = df[df["trade_date"] <= end_dt]
    return df


def main() -> None:
    cfg = config.get_config()
    pred_path = cfg.output_dir / "predictions.parquet"
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing predictions file: {pred_path}")

    print(f"Loading predictions from {pred_path} ...")
    pred_df = pd.read_parquet(pred_path)
    if "trade_date" not in pred_df.columns:
        raise ValueError("predictions must contain trade_date column")

    pred_df["trade_date"] = pd.to_datetime(pred_df["trade_date"], utc=True)
    eval_pred = pred_df.dropna(subset=["target"])
    eval_pred = _filter_window(eval_pred, cfg.test_start_date, cfg.test_end_date)

    if eval_pred.empty:
        print("No rows with target for backtest. Exiting.")
        return

    print("Loading intraday cache...")
    intraday_cache = load_intraday_cache(cfg.pairs, cfg.intraday_timeframe, cfg.data_root)

    print("Running backtest from saved predictions...")
    daily_bt, trades_bt = run_backtest(
        eval_pred,
        intraday_cache,
        cfg.take_profit,
        cfg.stop_loss,
        cfg.symbol_tp,
        cfg.symbol_sl,
        cfg.top_n_long,
        cfg.top_n_short,
        cfg.leverage,
    )
    daily_bt = add_net_pnl(daily_bt, fee=cfg.fee, leverage=cfg.leverage)

    benchmark = compute_benchmark(cfg.data_root, cfg.benchmark_pair, cfg.timeframe)
    bt_metrics = metrics(
        daily_bt["daily_return"],
        daily_bt,
        timeframe=cfg.timeframe,
        fee=cfg.fee,
        leverage=cfg.leverage,
    )
    rank_stats = rank_metrics(eval_pred, cfg.top_n_long, cfg.top_n_short)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    bt_path = cfg.output_dir / "backtest_daily_from_preds.parquet"
    trades_path = cfg.output_dir / "trades_from_preds.csv"
    equity_path = cfg.output_dir / "equity_curve_from_preds.png"
    summary_path = cfg.output_dir / "summary_from_preds.json"

    daily_bt.to_parquet(bt_path, index=False)
    daily_bt.to_csv(cfg.output_dir / "daily_backtest_from_preds.csv", index=False)
    trades_bt.to_csv(trades_path, index=False)
    plot_equity(daily_bt, benchmark, equity_path)

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "backtest": bt_metrics,
                "ranking": rank_stats,
                "backtest_daily": str(bt_path),
                "trades_csv": str(trades_path),
                "equity_curve": str(equity_path),
            },
            f,
            indent=2,
        )

    print("Backtest complete.")
    print(
        f"Sharpe: {bt_metrics['sharpe']:.3f}, "
        f"CAGR: {bt_metrics['cagr']:.3%}, "
        f"Annualized(Simple): {bt_metrics['annualized_return']:.3%}, "
        f"Returns(Simple): {bt_metrics['returns']:.3%}, "
        f"Spearman: {rank_stats['spearman']:.3f}, "
        f"TopHit: {rank_stats['top_hit_rate']:.3f}, "
        f"BottomHit: {rank_stats['bottom_hit_rate']:.3f}"
    )
    print(f"Outputs written to {cfg.output_dir}")


if __name__ == "__main__":
    main()
