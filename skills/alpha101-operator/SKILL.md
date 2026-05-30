---
name: alpha101-operator
description: Use when adding or modifying Alpha101 factor operators in this repository, including module placement, pandas DataFrame input/output contracts, expression-engine exposure, and basic validation.
---

# Alpha101 Operator Guide

## Operator Contract

Operators receive `pd.DataFrame` inputs in wide format:

- `index`: datetime-like bars
- `columns`: symbols
- values: numeric factor, price, volume, return, or metadata series

Operators should return a `pd.DataFrame` with the same index and columns as the primary input. Do not mutate inputs in place. Preserve missing data unless the operator explicitly normalizes or fills it.

Avoid lookahead: time-series operators may only use the current row and historical rows. Cross-section operators may only compare symbols within the same row.

## Module Placement

Place implementations by behavior:

- `alpha101/factors/operator_lib/time_series.py`: rolling, lag, decay, trend, volatility, and other time-axis operators.
- `alpha101/factors/operator_lib/cross_section.py`: row-wise rank, scale, z-score, winsorization, and neutralization.
- `alpha101/factors/operator_lib/regression.py`: rolling regression operators and generated regression helpers.
- `alpha101/factors/operator_lib/transforms.py`: conditional helpers and post-processing utilities.

`alpha101/factors/operators.py` is a compatibility export layer. If a new module is added under `operator_lib`, re-export it there so `FastExpressionEngine` can discover the callable.

## Examples

```python
def ts_example(x: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    return x.rolling(window=window, min_periods=window).mean()
```

```python
def cs_example(x: pd.DataFrame) -> pd.DataFrame:
    return x.rank(axis=1, pct=True)
```

## Validation

After adding an operator, run:

```bash
python -m compileall alpha101
```

For behavior checks, build a small wide `pd.DataFrame`, call the operator directly, and verify shape, index, columns, NaN handling, and no lookahead. If the operator is intended for expressions, evaluate it through `FastExpressionEngine`.
