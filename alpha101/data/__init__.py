"""Data loading and metadata utilities."""

from alpha101.data.arrays import nan_rowwise_corr, sample_indices_after_agg
from alpha101.data.ashare import build_ashare_wide_frame, build_ashare_wide_frame_from_csv
from alpha101.data.loading import (
    build_research_wide_frame,
    load_pair_frame,
    long_to_wide,
    wide_to_long,
)
from alpha101.data.market_caps import get_pair_market_caps
from alpha101.data.views import FactorDataView

__all__ = [
    "FactorDataView",
    "build_ashare_wide_frame",
    "build_ashare_wide_frame_from_csv",
    "build_research_wide_frame",
    "get_pair_market_caps",
    "load_pair_frame",
    "long_to_wide",
    "nan_rowwise_corr",
    "sample_indices_after_agg",
    "wide_to_long",
]
