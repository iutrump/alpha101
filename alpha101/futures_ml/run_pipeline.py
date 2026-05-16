from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import sys
sys.path.append(str(Path(__file__)))  # Adjust the path as needed
sys.path.append(str(Path(__file__).parent))  # Adjust the path as needed
sys.path.append(str(Path(__file__).parent.parent))  # Adjust the path as needed
sys.path.append(str(Path(__file__).parent.parent.parent))  # Adjust the path as needed
import alpha101.futures_ml.config as config
from alpha101.futures_ml.backtest import add_net_pnl, compute_benchmark, metrics, rank_metrics, run_backtest
from alpha101.futures_ml.data import build_full_dataset, load_intraday_cache
from alpha101.futures_ml.training import evaluate, get_feature_columns, walk_forward_predict
from alpha101.futures_ml.visualize import plot_equity
import os

def main():
    cfg = config.get_config()
    print("Loading daily data and building features...")
    dataset = build_full_dataset(
        cfg.pairs,
        cfg.lookback_days,
        cfg.data_root,
        cfg.timeframe,
        train_bars=cfg.train_bars,
        test_start_date=cfg.test_start_date,
        test_end_date=cfg.test_end_date,
        alpha_expression_path=cfg.alpha_expression_path
    )
    if cfg.test_end_date:
        end_dt = pd.to_datetime(cfg.test_end_date, utc=True)
        # Use asof_date to filter data availability, allowing for T+1 trade_date
        dataset = dataset[dataset["asof_date"] <= end_dt]
    feature_cols = get_feature_columns(dataset)
    if not feature_cols:
        raise ValueError("No feature columns generated; check alpha calculation")

    
    # Allow prediction up to test_end_date + 1 day to capture the "next" signal
    wf_end_date = pd.to_datetime(cfg.test_end_date, utc=True) + pd.Timedelta(days=1) if cfg.test_end_date else None

    test_pred = walk_forward_predict(
        dataset,
        feature_cols,
        retrain_every_bars=cfg.retrain_every_bars,
        min_train_bars=cfg.min_train_bars,
        train_bars=cfg.train_bars,
        train_split=cfg.train_split,
        train_per_pair=cfg.train_per_pair,
        top_n_long=cfg.top_n_long,
        top_n_short=cfg.top_n_short,
        smoothing_span=cfg.smoothing_span,
        test_start_date=pd.to_datetime(cfg.test_start_date, utc=True) if cfg.test_start_date else None,
        test_end_date=wf_end_date,
    )

    # Apply test date window (trade_date) if provided
    if cfg.test_start_date:
        start_dt = pd.to_datetime(cfg.test_start_date, utc=True)
        test_pred = test_pred[test_pred["trade_date"] >= start_dt]
    if cfg.test_end_date:
        end_dt = pd.to_datetime(cfg.test_end_date, utc=True)
        # Keep predictions up to end_dt + 1 (the future prediction)
        test_pred = test_pred[test_pred["trade_date"] <= end_dt + pd.Timedelta('1d')]
    
    # Split into evaluation set (with targets) and future set (without targets)
    eval_pred = test_pred.dropna(subset=["target"])
    future_pred = test_pred[test_pred["target"].isna()]
    print(f"Rows with target: {len(eval_pred)}, Rows without target (future): {len(future_pred)}")

    if not eval_pred.empty:
        eval_metrics = evaluate(
            eval_pred,
            top_k_long=cfg.top_n_long,
            top_k_short=cfg.top_n_short,
        )
        print(
            f"MAE: {eval_metrics['mae']:.6f}, R2: {eval_metrics['r2']:.4f}, "
            f"Spearman: {eval_metrics['spearman']:.3f}, DailySpearman: {eval_metrics['daily_spearman']:.3f}, "
            f"NDCG-Long@{cfg.top_n_long}: {eval_metrics['ndcg_long@k']:.3f}, "
            f"NDCG-Short@{cfg.top_n_short}: {eval_metrics['ndcg_short@k']:.3f}, "
            f"NDCG-Avg: {eval_metrics['ndcg_avg']:.3f}"
        )
    else:
        eval_metrics = {
            "mae": 0.0,
            "r2": 0.0,
            "spearman": 0.0,
            "daily_spearman": 0.0,
            "ndcg_long@k": 0.0,
            "ndcg_short@k": 0.0,
            "ndcg_avg": 0.0,
            "ndcg_long_k": int(cfg.top_n_long),
            "ndcg_short_k": int(cfg.top_n_short),
        }
        print("No rows with target for evaluation.")

    print("Running backtest on test period...")
    intraday_cache = load_intraday_cache(cfg.pairs, cfg.intraday_timeframe, cfg.data_root)
    
    # Backtest only on rows with target (historical)
    if not eval_pred.empty:
        for (n_long, n_short) in [(30, 30),(10, 10),(5, 5), (20,20),(int(cfg.top_n_long), int(cfg.top_n_short))]:
            daily_bt, trades_bt = run_backtest(
                eval_pred,
                intraday_cache,
                cfg.take_profit,
                cfg.stop_loss,
                cfg.symbol_tp,
                cfg.symbol_sl,
                n_long,
                n_short,
                cfg.leverage,
                n_jobs=min(os.cpu_count()-4, 8) if sys.platform == 'linux' else 1,  # Avoid multiprocessing on Windows for simplicity
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
            rank_stats = rank_metrics(eval_pred, n_long, n_short)
            print(f"test period backtest results: from {cfg.test_start_date} to {cfg.test_end_date}")
            print(f"top {n_long}, short {n_short}")
            print(
                f"Sharpe: {bt_metrics['sharpe']:.3f}, "
                f"CAGR: {bt_metrics['cagr']:.3%}, "
                f"Returns(Annualized): {bt_metrics['annualized_return']:.3%}, "
                f"MaxDD: {bt_metrics['max_drawdown']:.3%}, "
                f"Turnover: {bt_metrics['turnover']*100:.1f}%, "
                f"Fee Drag (Annualized): {bt_metrics['fee_drag_per_year']*100:.2f}%, "
                f"Spearman: {rank_stats['spearman']:.3f}, "
                f"TopHit: {rank_stats['top_hit_rate']:.3f}, "
                f"BottomHit: {rank_stats['bottom_hit_rate']:.3f}"
            )
    else:
        daily_bt = pd.DataFrame()
        trades_bt = pd.DataFrame()
        bt_metrics = {
            "sharpe": 0.0,
            "cagr": 0.0,
            "max_drawdown": 0.0,
            "turnover": 0.0,
            "returns": 0.0,
            "annualized_return": 0.0,
        }
        rank_stats = {"spearman": 0.0, "top_hit_rate": 0.0, "bottom_hit_rate": 0.0}
        benchmark = pd.DataFrame()

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = cfg.output_dir / "predictions.parquet"
    bt_path = cfg.output_dir / "backtest_daily.parquet"
    trades_path = cfg.output_dir / "trades.csv"
    long_short_pair_path = cfg.output_dir / "long_short_pairs.json"

    # Save long/short pairs for the next trading session
    long_short_pair_df = test_pred[test_pred['target'].isna()].sort_values('predicted_return')
    if False:  # IGNORE - keep all future predictions instead of just top/bottom
        assert len(long_short_pair_df['trade_date'].unique()) == 1
        long_short_pair = {
            'trade_date': long_short_pair_df['trade_date'].iloc[0].strftime('%Y-%m-%d'),
            'long_pairs': long_short_pair_df.tail(cfg.top_n_long)['symbol'].tolist(),
            'short_pairs': long_short_pair_df.head(cfg.top_n_short)['symbol'].tolist(),
            'details':{
                long_short_pair['symbol']: {
                    'predicted_return': long_short_pair['predicted_return']
                } for _, long_short_pair in long_short_pair_df.iterrows()
            }
        }
        with open(long_short_pair_path, "w", encoding="utf-8") as f:
            json.dump(long_short_pair, f, indent=2)

    # Save FULL predictions (including future)
    test_pred.to_parquet(pred_path, index=False)
    
    if not daily_bt.empty:
        daily_bt.to_parquet(bt_path, index=False)
        daily_bt.to_csv(cfg.output_dir / "daily_backtest.csv", index=False)
        trades_bt.to_csv(trades_path, index=False)
        equity_path = cfg.output_dir / "equity_curve.png"
        plot_equity(daily_bt, benchmark, equity_path)
    else:
        equity_path = Path("no_backtest.png")
        print("Skipping backtest plots (no backtest data)")

    summary_path = cfg.output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump({
            "eval": eval_metrics,
            "backtest": bt_metrics,
            "ranking": rank_stats,
            "predictions": str(pred_path),
            "backtest_daily": str(bt_path),
            "trades_csv": str(trades_path),
            "equity_curve": str(equity_path),
        }, f, indent=2)

    print(f"Outputs written to {cfg.output_dir}")


if __name__ == "__main__":
    main()
