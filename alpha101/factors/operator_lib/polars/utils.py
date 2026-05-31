from __future__ import annotations

from alpha101.data.views import LazyPolarsFactor, PolarsFactor


def require_polars():
    try:
        import polars as pl
    except ImportError as exc:
        raise ImportError(
            "Polars expression backend requires `polars`. Install project dependencies with `python -m pip install -e .`."
        ) from exc
    return pl


def ensure_factor(value) -> PolarsFactor:
    if not isinstance(value, (PolarsFactor, LazyPolarsFactor)):
        raise TypeError(f"Expected PolarsFactor, got {type(value)!r}")
    return value


def to_numpy(factor: PolarsFactor):
    return ensure_factor(factor).to_numpy()


def to_pandas(factor: PolarsFactor):
    return ensure_factor(factor).to_pandas()
