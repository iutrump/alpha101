from __future__ import annotations

import numpy as np
import pandas as pd


class FactorDataView:
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


Alphas = FactorDataView


def _require_polars():
    try:
        import polars as pl
    except ImportError as exc:
        raise ImportError(
            "Polars expression backend requires `polars`. Install with `python -m pip install -e \".[polars]\"`."
        ) from exc
    return pl


class PolarsFactor:
    """Long-form Polars factor with columns: date, symbol, value."""

    def __init__(self, frame, *, symbols=None, dates=None):
        self.frame = frame.select(["date", "symbol", "value"])
        self.symbols = list(symbols) if symbols is not None else self._ordered_unique("symbol")
        self.dates = list(dates) if dates is not None else self._ordered_unique("date")

    @classmethod
    def from_wide(cls, df: pd.DataFrame | None, *, template: "PolarsFactor | None" = None):
        if df is None:
            if template is None:
                return None
            return template.map_value(_require_polars().lit(None, dtype=_require_polars().Float64))

        pl = _require_polars()
        wide = df.copy()
        wide.index.name = "date"
        long_pd = wide.reset_index().melt(id_vars="date", var_name="symbol", value_name="value")
        frame = pl.from_pandas(long_pd).with_columns(
            pl.col("symbol").cast(pl.Utf8),
            pl.col("value").cast(pl.Float64),
        )
        return cls(frame, symbols=list(df.columns), dates=list(df.index))

    @property
    def columns(self):
        return pd.Index(self.symbols, name="symbol")

    def map_value(self, expr):
        return PolarsFactor(
            self.frame.with_columns(expr.alias("value")),
            symbols=self.symbols,
            dates=self.dates,
        )

    def with_frame(self, frame):
        return PolarsFactor(frame, symbols=self.symbols, dates=self.dates)

    def to_numpy(self) -> np.ndarray:
        return self.to_pandas().to_numpy(dtype=float, copy=False)

    def to_pandas(self) -> pd.DataFrame:
        pl = _require_polars()
        frame = self.frame
        if frame.is_empty():
            return pd.DataFrame(index=pd.Index(self.dates, name="date"), columns=self.symbols, dtype=float)

        wide = frame.pivot(index="date", columns="symbol", values="value", aggregate_function="first")
        for symbol in self.symbols:
            if symbol not in wide.columns:
                wide = wide.with_columns(pl.lit(None, dtype=pl.Float64).alias(symbol))
        wide = wide.sort("date").select(["date", *self.symbols])
        out = wide.to_pandas().set_index("date")
        out.index.name = None
        out.columns = pd.Index(self.symbols, name="symbol")
        return out

    def is_all_nan(self) -> bool:
        value = self._value()
        return self.frame.select((value.is_not_null() & value.is_not_nan()).sum()).item() == 0

    def _ordered_unique(self, column: str):
        return self.frame.select(column).unique(maintain_order=True).get_column(column).to_list()

    def _value(self):
        return _require_polars().col("value")

    def _binary_op(self, other, op):
        pl = _require_polars()
        if isinstance(other, PolarsFactor):
            joined = self.frame.rename({"value": "left"}).join(
                other.frame.rename({"value": "right"}),
                on=["date", "symbol"],
                how="left",
            )
            return PolarsFactor(
                joined.with_columns(op(pl.col("left"), pl.col("right")).alias("value")).select(
                    ["date", "symbol", "value"]
                ),
                symbols=self.symbols,
                dates=self.dates,
            )
        return self.map_value(op(pl.col("value"), other))

    def _reverse_binary_op(self, other, op):
        pl = _require_polars()
        return self.map_value(op(other, pl.col("value")))

    def __add__(self, other):
        return self._binary_op(other, lambda left, right: left + right)

    def __radd__(self, other):
        return self._reverse_binary_op(other, lambda left, right: left + right)

    def __sub__(self, other):
        return self._binary_op(other, lambda left, right: left - right)

    def __rsub__(self, other):
        return self._reverse_binary_op(other, lambda left, right: left - right)

    def __mul__(self, other):
        return self._binary_op(other, lambda left, right: left * right)

    def __rmul__(self, other):
        return self._reverse_binary_op(other, lambda left, right: left * right)

    def __truediv__(self, other):
        return self._binary_op(other, lambda left, right: left / right)

    def __rtruediv__(self, other):
        return self._reverse_binary_op(other, lambda left, right: left / right)

    def __pow__(self, other):
        return self._binary_op(other, lambda left, right: left.pow(right))

    def __neg__(self):
        return self.map_value(-self._value())

    def __abs__(self):
        return self.map_value(self._value().abs())

    def __gt__(self, other):
        return self._binary_op(other, lambda left, right: left > right)

    def __ge__(self, other):
        return self._binary_op(other, lambda left, right: left >= right)

    def __lt__(self, other):
        return self._binary_op(other, lambda left, right: left < right)

    def __le__(self, other):
        return self._binary_op(other, lambda left, right: left <= right)


class LazyPolarsFactor:
    """Lazy long-form Polars factor with columns: date, symbol, value."""

    def __init__(self, frame, *, symbols=None, dates=None):
        self.frame = frame.select(["date", "symbol", "value"])
        self.symbols = list(symbols) if symbols is not None else self._ordered_unique("symbol")
        self.dates = list(dates) if dates is not None else self._ordered_unique("date")

    @classmethod
    def from_wide(cls, df: pd.DataFrame | None, *, template: "LazyPolarsFactor | PolarsFactor | None" = None):
        pl = _require_polars()
        if df is None:
            if template is None:
                return None
            return template.map_value(pl.lit(None, dtype=pl.Float64))

        wide = df.copy()
        wide.index.name = "date"
        long_pd = wide.reset_index().melt(id_vars="date", var_name="symbol", value_name="value")
        frame = pl.from_pandas(long_pd).with_columns(
            pl.col("symbol").cast(pl.Utf8),
            pl.col("value").cast(pl.Float64),
        )
        return cls(frame.lazy(), symbols=list(df.columns), dates=list(df.index))

    @classmethod
    def from_factor(cls, factor: PolarsFactor):
        return cls(factor.frame.lazy(), symbols=factor.symbols, dates=factor.dates)

    @property
    def columns(self):
        return pd.Index(self.symbols, name="symbol")

    def map_value(self, expr):
        return LazyPolarsFactor(
            self.frame.with_columns(expr.alias("value")),
            symbols=self.symbols,
            dates=self.dates,
        )

    def with_frame(self, frame):
        if hasattr(frame, "lazy"):
            frame = frame.lazy()
        return LazyPolarsFactor(frame, symbols=self.symbols, dates=self.dates)

    def to_numpy(self) -> np.ndarray:
        return self.to_pandas().to_numpy(dtype=float, copy=False)

    def to_pandas(self) -> pd.DataFrame:
        pl = _require_polars()
        frame = self.frame.collect()
        if frame.is_empty():
            return pd.DataFrame(index=pd.Index(self.dates, name="date"), columns=self.symbols, dtype=float)

        wide = frame.pivot(index="date", columns="symbol", values="value", aggregate_function="first")
        for symbol in self.symbols:
            if symbol not in wide.columns:
                wide = wide.with_columns(pl.lit(None, dtype=pl.Float64).alias(symbol))
        wide = wide.sort("date").select(["date", *self.symbols])
        out = wide.to_pandas().set_index("date")
        out.index.name = None
        out.columns = pd.Index(self.symbols, name="symbol")
        return out

    def is_all_nan(self) -> bool:
        value = self._value()
        return self.frame.select((value.is_not_null() & value.is_not_nan()).sum()).collect().item() == 0

    def _ordered_unique(self, column: str):
        return self.frame.select(column).unique(maintain_order=True).collect().get_column(column).to_list()

    def _value(self):
        return _require_polars().col("value")

    def _binary_op(self, other, op):
        pl = _require_polars()
        if isinstance(other, PolarsFactor):
            other = LazyPolarsFactor.from_factor(other)
        if isinstance(other, LazyPolarsFactor):
            joined = self.frame.rename({"value": "left"}).join(
                other.frame.rename({"value": "right"}),
                on=["date", "symbol"],
                how="left",
            )
            return LazyPolarsFactor(
                joined.with_columns(op(pl.col("left"), pl.col("right")).alias("value")).select(
                    ["date", "symbol", "value"]
                ),
                symbols=self.symbols,
                dates=self.dates,
            )
        return self.map_value(op(pl.col("value"), other))

    def _reverse_binary_op(self, other, op):
        pl = _require_polars()
        return self.map_value(op(other, pl.col("value")))

    def __add__(self, other):
        return self._binary_op(other, lambda left, right: left + right)

    def __radd__(self, other):
        return self._reverse_binary_op(other, lambda left, right: left + right)

    def __sub__(self, other):
        return self._binary_op(other, lambda left, right: left - right)

    def __rsub__(self, other):
        return self._reverse_binary_op(other, lambda left, right: left - right)

    def __mul__(self, other):
        return self._binary_op(other, lambda left, right: left * right)

    def __rmul__(self, other):
        return self._reverse_binary_op(other, lambda left, right: left * right)

    def __truediv__(self, other):
        return self._binary_op(other, lambda left, right: left / right)

    def __rtruediv__(self, other):
        return self._reverse_binary_op(other, lambda left, right: left / right)

    def __pow__(self, other):
        return self._binary_op(other, lambda left, right: left.pow(right))

    def __neg__(self):
        return self.map_value(-self._value())

    def __abs__(self):
        return self.map_value(self._value().abs())

    def __gt__(self, other):
        return self._binary_op(other, lambda left, right: left > right)

    def __ge__(self, other):
        return self._binary_op(other, lambda left, right: left >= right)

    def __lt__(self, other):
        return self._binary_op(other, lambda left, right: left < right)

    def __le__(self, other):
        return self._binary_op(other, lambda left, right: left <= right)


class PolarsLongDataView:
    def __init__(self, wide_data):
        self._open = PolarsFactor.from_wide(wide_data["open"])
        self._high = PolarsFactor.from_wide(wide_data["high"])
        self._low = PolarsFactor.from_wide(wide_data["low"])
        self._close = PolarsFactor.from_wide(wide_data["close"])
        self._volume = PolarsFactor.from_wide(wide_data["volume"])
        self._returns = PolarsFactor.from_wide(wide_data["close"].pct_change(fill_method=None))
        self._vwap = PolarsFactor.from_wide(wide_data["vwap"])
        self._cap = PolarsFactor.from_wide(
            wide_data["cap"] if "cap" in wide_data.columns.get_level_values(0) else None,
            template=self._close,
        )
        self._funding = PolarsFactor.from_wide(
            wide_data["funding"] if "funding" in wide_data.columns.get_level_values(0) else None,
            template=self._close,
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
    def cap(self):
        return self._cap

    @property
    def funding(self):
        return self._funding

    @property
    def market_return(self):
        if self._market_return is None:
            self._market_return = PolarsFactor.from_wide(
                FactorDataView(self.to_wide_data()).market_return,
                template=self._close,
            )
        return self._market_return

    def to_wide_data(self) -> pd.DataFrame:
        fields = {
            "open": self.open.to_pandas(),
            "high": self.high.to_pandas(),
            "low": self.low.to_pandas(),
            "close": self.close.to_pandas(),
            "volume": self.volume.to_pandas(),
            "vwap": self.vwap.to_pandas(),
            "cap": self.cap.to_pandas(),
            "funding": self.funding.to_pandas(),
        }
        return pd.concat(fields, axis=1)
