# Curated Factor Library

This folder contains factors selected from genetic-search outputs after a second-pass stability review.

Current source run:

- `factor_search_results/experiments/baseline_large_seed2/4h/20260607_141809/summary.csv`
- Data window: 2025-01-01 to 2026-06-02, 4h bars
- Universe after loading/filtering: 219 symbols
- Cost: 0.001 round trip
- Cross-validation outputs:
  - `factor_search_results/experiments/baseline_large_seed2/4h/20260607_141809/cross_validation_candidates.csv`
  - `factor_search_results/experiments/baseline_large_seed2/4h/20260607_141809/cross_validation_report.md`

## Selection Rule

Accepted factors must have positive original test performance, survive most or all six chronological folds, and stay positive across all three deterministic universe splits. Near-duplicates are not promoted together; one representative is marked `accepted`, while close parameter variants are marked `alternate`.

## Accepted Factors

### `volume_zscore_mean_neg_28`

```text
-ts_mean(zscore(volume), 28)
```

Explanation: low relative-volume preference. `zscore(volume)` measures each coin's volume intensity versus the cross-section at each bar; the 28-bar mean smooths that state; the negative sign ranks quieter, less crowded names higher and recent high-volume names lower.

This is the preferred interpretable replacement for the raw genetic-search expression `ts_product(zscore(volume), 28)`. Manual validation showed:

- `ts_mean(zscore(volume), 28)` and `ts_sum(zscore(volume), 28)` were consistently negative.
- `-ts_mean(zscore(volume), 28)` passed all six time folds and all three universe folds.
- `alpha101-expression --exposure --market-beta "-ts_mean(zscore(volume), 28)"` reported Sharpe after cost 2.595, IC IR 0.109, and annual returns after cost 56.133%.

Risk: this is not a pure alpha. The detailed exposure check showed strong positive size exposure and strong negative liquidity exposure. Treat it as a liquidity/crowding factor and neutralize or cap style exposure before production.

### `vol_zscore_product_28`

```text
ts_product(zscore(volume), 28)
```

Explanation: persistent same-sign volume regime. Because `zscore(volume)` can be positive or negative and the window length is even, this expression can be high when volume is persistently far from the cross-sectional mean, including persistently quiet names. It is therefore better understood as a nonlinear volume-regime detector than as a clean "high volume continuation" factor.

Risk: the expression is mathematically awkward because a product over 28 signed z-scores can flip sign or create large magnitudes. Keep it as a search-discovered reference, but prefer `volume_zscore_mean_neg_28` for a clean library factor unless the nonlinear product shows incremental residual IC.

### `volume_low_alpha_min_21_30`

```text
ts_min(ts_alpha(volume, winsorize(low), 21), 30)
```

Explanation: recent low extreme of the 21-bar regression intercept between volume and winsorized low price. This is a volume-adjusted low-price/liquidity stress state. It is more interpretable than most nested search outputs because both the regression pair and the outer minimum have a clear role.

Risk: one time fold was negative and one universe split was only weakly positive. It may carry price-level, liquidity, or volatility exposure.

### `volume_cap_alpha_min_20_20`

```text
ts_min(ts_alpha(volume, cap, 20), 20)
```

Explanation: recent low extreme of the 20-bar regression intercept between volume and market cap. This is the cleanest representative of the volume-cap relation family because every chronological fold stayed above Sharpe 1 in the validation run.

Risk: it is a style-heavy factor. Treat it as a liquidity/cap interaction until it survives neutralization against cap, turnover, volatility, momentum, and reversal.

### `volume_cap_alpha_min_14_21`

```text
ts_min(ts_alpha(volume, cap, 14), 21)
```

Explanation: a faster version of the volume-cap relation. It uses a shorter regression window and may respond more quickly to changes in volume-cap structure.

Risk: same economic family as `volume_cap_alpha_min_20_20`. Keep only if later factor-value correlation and residual IC checks show incremental contribution.

### `volume_high_alpha_min_21_21`

```text
ts_min(ts_alpha(volume, high, 21), 21)
```

Explanation: recent low extreme of the 21-bar regression intercept between volume and high price. This captures volume-adjusted high-price compression, related to liquidity-price interaction.

Risk: one time fold was negative. It may overlap with both the cap-alpha and low-alpha factors.

## Alternates

The CSV also records close variants:

- `ts_min(ts_alpha(volume, cap, 21), 21)`
- `ts_product(zscore(volume), 20)`
- `ts_product(zscore(volume), 30)`

These passed stability checks but are near-duplicates of accepted representatives. They should be used for parameter robustness checks, not counted as independent alpha.
