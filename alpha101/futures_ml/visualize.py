from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_equity(daily: pd.DataFrame, benchmark: pd.DataFrame, output_path: Path) -> Path:
    merged = daily.merge(benchmark, on="trade_date", how="left")
    merged["bench_return"] = merged["bench_return"].fillna(0)
    strategy_ret_col = "net_pnl" if "net_pnl" in merged.columns else "daily_return"
    strategy_returns = pd.to_numeric(merged[strategy_ret_col], errors="coerce").fillna(0.0)
    merged["strategy_equity"] = (1 + strategy_returns).cumprod()
    merged["bench_equity"] = (1 + merged["bench_return"]).cumprod()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(merged["trade_date"], merged["strategy_equity"], label="Strategy")
    ax.plot(merged["trade_date"], merged["bench_equity"], label="Benchmark")
    ax.set_title("Cumulative returns")
    ax.set_ylabel("Equity (start=1)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path
