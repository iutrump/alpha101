from __future__ import annotations

import numpy as np
import pandas as pd

class Alphas(object):
    def __init__(self, wide_data):
        """
        wide_data must use MultiIndex columns: (field, symbol).
        Required fields: open, high, low, close, volume, vwap.
        Optional fields: cap, funding.
        """
        self._open = wide_data["open"]
        self._high = wide_data["high"]
        self._low = wide_data["low"]
        self._close = wide_data["close"]
        self._volume = wide_data["volume"]
        self._returns = self._close.pct_change(fill_method=None)
        self._vwap = wide_data["vwap"]
        self._cap = wide_data["cap"] if "cap" in wide_data.columns.get_level_values(0) else None
        self._funding = (
            wide_data["funding"] if "funding" in wide_data.columns.get_level_values(0) else None
        )
        self._market_return = None

    @property
    def open(self):
        return self._open
    @property
    def high(self):
        return self._high
    @property
    def low(self):
        return self._low
    @property
    def close(self):
        return self._close
    @property
    def volume(self):
        return self._volume
    @property
    def returns(self):
        return self._returns
    @property
    def vwap(self):
        return self._vwap
    @property
    def market_return(self):
        if self._market_return is None:
            self._market_return = self._build_market_return()
        return self._market_return
    @property
    def cap(self):
        return self._cap
    @property
    def funding(self):
        return self._funding

    def _build_market_return(self, top_n: int = 15):
        if self._cap is None:
            market = self._returns.mean(axis=1, skipna=True)
        else:
            cap_top = self._cap.where(
                self._cap.rank(axis=1, ascending=False, method="first") <= top_n
            )
            cap_sum = cap_top.sum(axis=1).replace(0, np.nan)
            cap_weight = cap_top.div(cap_sum, axis=0)
            market = (self._returns * cap_weight).sum(axis=1, min_count=1)
        return pd.DataFrame(
            np.repeat(market.values[:, None], self._returns.shape[1], axis=1),
            index=self._returns.index,
            columns=self._returns.columns,
        )
