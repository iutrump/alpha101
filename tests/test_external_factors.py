from __future__ import annotations

import pandas as pd

from alpha101.data.external import build_external_factor_wide_frame, infer_external_factor_columns
from alpha101.factors.evaluation.external import (
    build_external_factor_curve,
    score_external_factors,
    select_external_factors_by_pnl_similarity,
)


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


def _external_selection_frame() -> pd.DataFrame:
    rows = []
    targets = [
        [0.01, 0.02, 0.04, -0.01],
        [0.04, -0.01, 0.02, 0.01],
        [0.02, 0.04, -0.01, 0.01],
        [-0.01, 0.01, 0.04, 0.02],
        [0.03, 0.02, -0.01, 0.04],
        [0.01, 0.04, 0.02, -0.01],
    ]
    good_orders = [
        [1, 2, 4, 3],
        [4, 1, 3, 2],
        [2, 4, 1, 3],
        [1, 2, 4, 3],
        [3, 2, 1, 4],
        [1, 4, 2, 3],
    ]
    diverse_orders = [
        [1, 4, 2, 3],
        [2, 3, 4, 1],
        [4, 1, 3, 2],
        [3, 4, 1, 2],
        [4, 1, 2, 3],
        [2, 1, 4, 3],
    ]
    codes = ["000001.XSHE", "600000.XSHG", "920001.XBEI", "300001.XSHE"]
    for date_idx, dt in enumerate(pd.date_range("2025-01-07", periods=len(targets), freq="7D")):
        for code_idx, code in enumerate(codes):
            good = float(good_orders[date_idx][code_idx])
            diverse = float(diverse_orders[date_idx][code_idx])
            rows.append(
                {
                    "date": dt,
                    "code": code,
                    "label_5d": targets[date_idx][code_idx],
                    "good": good,
                    "good_clone": good * 10.0,
                    "diverse": diverse,
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


def test_select_external_factors_filters_by_signed_pnl_similarity(tmp_path):
    csv_path = tmp_path / "accepted.csv"
    _external_selection_frame().to_csv(csv_path, index=False)
    wide = build_external_factor_wide_frame(csv_path)
    summary = pd.DataFrame(
        [
            {"factor": "good", "selected_excess_returns": 1.0, "direction": 1},
            {"factor": "good_clone", "selected_excess_returns": 0.9, "direction": 1},
            {"factor": "diverse", "selected_excess_returns": 0.8, "direction": 1},
        ]
    )

    result = select_external_factors_by_pnl_similarity(
        wide,
        summary,
        n_quantiles=2,
        transaction_cost=0.0,
        max_factors=2,
        pnl_corr_threshold=0.95,
        score_column="selected_excess_returns",
    )

    selected = result["selected"].set_index("factor")
    rejected = result["rejected"].set_index("factor")
    assert list(selected.index) == ["good", "diverse"]
    assert rejected.loc["good_clone", "selection_reason"] == "pnl_similarity"
    assert rejected.loc["good_clone", "nearest_selected_factor"] == "good"
    assert rejected.loc["good_clone", "max_selected_pnl_corr"] >= 0.99
    assert result["stats"]["selected_count"] == 2
    assert result["stats"]["rejected_similarity_count"] == 1
    assert {"date", "selected_cum", "benchmark_cum", "excess_cum", "selected_factor_count"}.issubset(
        result["curve"].columns
    )
