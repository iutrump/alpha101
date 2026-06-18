from __future__ import annotations

import pandas as pd

from alpha101.data.external import build_external_factor_wide_frame, infer_external_factor_columns
from alpha101.factors.evaluation.external import build_external_factor_curve, score_external_factors


def _external_factor_frame() -> pd.DataFrame:
    rows = []
    for date_idx, dt in enumerate(pd.date_range("2025-01-07", periods=4, freq="7D")):
        for code_idx, code in enumerate(["000001.XSHE", "600000.XSHG", "920001.XBEI"]):
            target = 0.01 * (date_idx + 1) * (code_idx + 1)
            rows.append(
                {
                    "date": dt,
                    "signal_date": dt - pd.Timedelta(days=1),
                    "code": code,
                    "market_cap": 1000.0 + code_idx,
                    "log_market_cap": 7.0 + code_idx,
                    "industry": f"ind{code_idx}",
                    "label_5d": target,
                    "probe_0001": target * 10.0,
                    "post10x30_0002": -target * 5.0,
                }
            )
    return pd.DataFrame(rows)


def test_build_external_factor_wide_frame_infers_factor_columns_and_target(tmp_path):
    csv_path = tmp_path / "accepted.csv"
    _external_factor_frame().to_csv(csv_path, index=False)

    wide = build_external_factor_wide_frame(csv_path)

    assert wide.columns.names == ["field", "symbol"]
    assert infer_external_factor_columns(_external_factor_frame()) == ["probe_0001", "post10x30_0002"]
    assert {"target", "probe_0001", "post10x30_0002"}.issubset(set(wide.columns.get_level_values(0)))
    assert wide.loc[pd.Timestamp("2025-01-07"), ("probe_0001", "000001.XSHE")] == 0.1
    assert wide.loc[pd.Timestamp("2025-01-14"), ("target", "920001.XBEI")] == 0.06


def test_score_external_factors_and_curve_use_precomputed_target(tmp_path):
    csv_path = tmp_path / "accepted.csv"
    _external_factor_frame().to_csv(csv_path, index=False)
    wide = build_external_factor_wide_frame(csv_path)

    summary = score_external_factors(wide, n_quantiles=3, min_segment_obs=1, mode="final", annualization=52.0)
    curve = build_external_factor_curve(wide, "probe_0001", n_quantiles=3, transaction_cost=0.0)

    assert list(summary["factor"]) == ["probe_0001", "post10x30_0002"]
    assert summary.loc[summary["factor"] == "probe_0001", "ic_mean"].iloc[0] > 0
    assert summary.loc[summary["factor"] == "post10x30_0002", "ic_mean"].iloc[0] < 0
    assert {"date", "pnl", "cum_pnl", "cum_pnl_net"}.issubset(curve.columns)
    assert len(curve) == 4


def test_score_external_factors_supports_signed_long_only_metrics(tmp_path):
    csv_path = tmp_path / "accepted.csv"
    _external_factor_frame().to_csv(csv_path, index=False)
    wide = build_external_factor_wide_frame(csv_path)

    summary = score_external_factors(
        wide,
        n_quantiles=3,
        min_segment_obs=1,
        mode="final",
        annualization=52.0,
        directions={"post10x30_0002": -1},
    )
    row = summary.set_index("factor").loc["post10x30_0002"]
    curve = build_external_factor_curve(
        wide,
        "post10x30_0002",
        n_quantiles=3,
        transaction_cost=0.0,
        direction=-1,
    )

    assert row["direction"] == -1
    assert row["buy_group"] == 1
    assert "selected_returns_before_cost" in row
    assert "selected_excess_returns_before_cost" in row
    assert "selected_turnover" in row
    assert "selected_cost_drag" in row
    assert row["selected_returns"] > row["universe_returns"]
    assert {
        "selected_cum_before_cost",
        "selected_cum",
        "benchmark_cum",
        "excess_cum_before_cost",
        "excess_cum",
    }.issubset(curve.columns)
    assert curve["selected_cum"].iloc[-1] > curve["benchmark_cum"].iloc[-1]
