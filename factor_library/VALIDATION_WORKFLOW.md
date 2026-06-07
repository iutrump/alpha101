# Factor Validation Workflow

Use this workflow for both mined factors and hand-improved expressions before adding anything to `curated_factors_*.csv`.

## 1. Generate Or Edit Candidates

For mined candidates, start from a `summary.csv` produced by `alpha101-factor-search`.

For manual improvements, write expressions directly on the command line or into a small file:

```text
-ts_mean(zscore(volume), 28)
ts_min(ts_alpha(volume, cap, 20), 20)
```

## 2. Run Cross-Validation

Mined candidates:

```bash
python3 scripts/cross_validate_factor_candidates.py \
  factor_search_results/experiments/baseline_large_seed2/4h/20260607_141809/summary.csv
```

Manual candidates:

```bash
python3 scripts/cross_validate_factor_candidates.py \
  --manifest factor_search_results/experiments/baseline_large_seed2/4h/20260607_141809/manifest.json \
  --out-dir factor_validation_results/manual_check \
  --expression "-ts_mean(zscore(volume), 28)" \
  --expression "ts_mean(zscore(volume), 28)"
```

Frequency and forecast-horizon checks:

```bash
python3 scripts/cross_validate_factor_candidates.py \
  --manifest factor_search_results/experiments/baseline_large_seed2/4h/20260607_141809/manifest.json \
  --timeframe 1h \
  --forward-periods 1 \
  --forward-periods 4 \
  --forward-periods 8 \
  --out-dir factor_validation_results/one_hour_core_20260607 \
  --expression "-ts_mean(zscore(volume), 28)" \
  --expression "ts_product(zscore(volume), 28)"
```

The script checks:

- six chronological folds;
- three deterministic universe splits;
- full-window Sharpe, returns, IC IR, turnover, and drawdown;
- high-correlation clusters among candidates.

When `--forward-periods` is greater than 1, the target return windows overlap. Treat the result as a horizon-sensitivity check, not as a directly comparable Sharpe against the one-bar horizon.

## 3. Detailed Backtest And Exposure

For any accepted candidate, run the detailed expression backtest:

```bash
alpha101-expression --exposure --market-beta "-ts_mean(zscore(volume), 28)"
```

Use this output to inspect:

- Sharpe after cost;
- IC mean and IC IR;
- turnover and cost drag;
- max drawdown;
- size, momentum, volatility, beta, liquidity, reversal, and funding exposure;
- strategy market beta.

## 4. Promotion Rules

Promote to `accepted` only when the factor:

- is positive in most or all chronological folds;
- is positive across all universe splits;
- has positive IC IR;
- has bounded turnover and drawdown;
- has a clear economic or behavioral explanation;
- is not a near-duplicate of a simpler accepted factor.

Use `alternate` for parameter variants or near-duplicates that are useful for robustness checks but should not count as independent alpha.

Reject candidates when performance only appears in one time segment, one universe split, or one sign convention without a defensible explanation.
