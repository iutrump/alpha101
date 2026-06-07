# Factor Mining Experiment Plan

This document records the experiment plan for finding stronger Alpha101-style crypto factors on the `experiment/factor-mining` branch.

## Objective

Find candidate factors that are promising enough for manual review in the research UI. A candidate is not considered production-ready until it survives out-of-sample checks, turnover checks, expression review, and visual inspection.

## Baseline

The first smoke run used the current genetic search implementation:

```bash
conda run -n alpha101 alpha101-factor-search \
  --config configs/alpha101.json \
  --strategy genetic \
  --population 12 \
  --generations 2 \
  --n-jobs 4 \
  --backend process \
  --profile \
  --output-dir factor_search_results/experiments/baseline
```

The run completed and produced:

```text
factor_search_results/experiments/baseline/4h/20260607_140933/summary.csv
```

Top smoke-run candidate:

```text
fitness: 0.026172
sharpe_ratio: 1.042378
returns: 0.131839
expression:
(ts_alpha(vwap, open, 28) - (cap - (-volume - ts_beta(returns, market_return, 14))))
```

This is only a workflow check. It is too small to qualify as a robust factor search.

## Evaluation Criteria

Prefer factors that satisfy all of the following:

- Positive `test_fitness`, not just high train fitness.
- Positive `test_sharpe`, `test_returns`, and `test_ic_ir`.
- Similar train, validation, and test behavior without a single segment dominating.
- Reasonable turnover and cost drag.
- Moderate drawdown.
- Expression is interpretable enough to inspect and debug.
- Not a near-duplicate of a higher-ranked factor.
- Visual inspection in the research UI does not show obvious data artifacts or stale-value behavior.

## Experiment Stages

1. Reproducibility improvements
   - Add a `--seed` option to the factor search CLI.
   - Seed Python and NumPy randomness used by generation and genetic selection.
   - Save a manifest for each run with config, arguments, git commit, branch, and timestamp.

2. Larger baseline search
   - Run a larger search without changing the scoring function.
   - Start with `population=80`, `generations=8`, `n_jobs=4`.
   - Save results under `factor_search_results/experiments/baseline_large`.
   - Summarize the top factors by train/valid/test metrics.

3. Stability filter
   - Add a post-processing script that reads one or more `summary.csv` files.
   - Filter for positive test metrics, reasonable turnover, and bounded drawdown.
   - Deduplicate normalized expressions.

4. Correlation and simplicity review
   - Compute pairwise correlations among top factor values.
   - Keep the simpler factor when two candidates are highly correlated.
   - Penalize excessive complexity if too many top candidates are hard to interpret.

5. Scoring experiments
   - Test a stability-weighted fitness that gives more weight to test/validation robustness.
   - Penalize turnover and excessive drawdown.
   - Compare against the baseline with the same seeds.

6. Manual inspection
   - Load top candidates in `alpha101-research-server`.
   - Inspect symbol-level K lines, raw factor values, and cross-sectional ranks.
   - Record rejected factors and reasons.

## Run Log

Each experiment should record:

- Branch and commit.
- Command.
- Output directory.
- Dataset window and timeframe.
- Population, generations, seed, backend, and workers.
- Top 10 expressions.
- Rejection notes for unstable or suspicious factors.
