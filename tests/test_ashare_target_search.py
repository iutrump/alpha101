from __future__ import annotations

import pandas as pd

from alpha101.factors.evaluation import forward_returns
from alpha101.factors.operator_lib import process_factor_wide_format
from alpha101.factors.search import FactorSearchEngine
from alpha101.factors.evaluation.scoring import _time_segment_slices
from alpha101.factors.validation import factor_pnl_series
from tests.test_expression_runtime import make_wide_data


def test_factor_search_uses_supplied_target_returns_for_visible_pnl():
    wide = make_wide_data()
    target = pd.DataFrame(
        [
            [0.10, -0.10, 0.00],
            [0.09, -0.09, 0.00],
            [0.08, -0.08, 0.00],
            [0.07, -0.07, 0.00],
            [0.06, -0.06, 0.00],
            [0.05, -0.05, 0.00],
            [0.04, -0.04, 0.00],
            [0.03, -0.03, 0.00],
        ],
        index=wide.index,
        columns=wide["close"].columns,
    )
    search_engine = FactorSearchEngine(
        wide,
        n_quantiles=2,
        target_returns=target,
        segment_ratios=[0.50, 0.25, 0.25],
        transaction_cost=0.0,
    )

    pnl = search_engine._search_visible_pnl("rank(close)")

    factor = process_factor_wide_format(search_engine.engine.evaluate("rank(close)"))
    slices = _time_segment_slices(len(wide), search_engine.segment_ratios)
    visible_slice = slice(slices["train"].start, slices["valid"].stop)
    expected = factor_pnl_series(
        factor.iloc[visible_slice],
        target.iloc[visible_slice],
        n_quantiles=2,
        transaction_cost=0.0,
    )
    internal_forward = factor_pnl_series(
        factor.iloc[visible_slice],
        forward_returns(wide["close"], periods=1).iloc[visible_slice],
        n_quantiles=2,
        transaction_cost=0.0,
    )

    pd.testing.assert_series_equal(pnl.reset_index(drop=True), expected.reset_index(drop=True))
    assert not pnl.reset_index(drop=True).equals(internal_forward.reset_index(drop=True))
