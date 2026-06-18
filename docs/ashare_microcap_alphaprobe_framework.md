# A-share Microcap AlphaPROBE Framework

This fork treats alpha101 as the research workbench for A-share microcap factor
work, while `microcap_alpha` remains the production training and holdings export
pipeline until those pieces are migrated.

## Roles

- `alpha101`: expression playground, A-share CSV adapter, external accepted-factor
  browser, single-factor diagnostics, signed long-only PnL similarity selection,
  and candidate generation.
- `microcap_alpha`: AlphaPROBE mining, accepted-factor export, rolling ICIR
  screening, neutralization A/B tests, Ridge training, holdings CSV generation,
  and JoinQuant external-holdings service.

## Default Protocol

The default protocol is exposed by `GET /api/protocol`.

- Universe: A-share microcap400.
- Frequency: weekly.
- Signal day: Monday.
- Rebalance day: Tuesday.
- Trade lag: 1 day.
- Forward label horizon: 5 trading days.
- Label embargo: exclude the most recent unfinished forward-label period.
- Splits:
  - Train: 2022-05-31 to 2023-12-31.
  - Validation: 2024-01-01 to 2024-12-31.
  - Test: 2025-01-01 to 2025-12-27.

## Factor Flow

1. Mine candidates in `microcap_alpha` with AlphaPROBE.
2. Export accepted factors into weekly or daily external-factor CSV files.
3. Start alpha101 with:

```powershell
..\microcap_alpha\scripts\export_alphaprobe_for_alpha101.ps1
.\scripts\start_alphaprobe_lab.ps1
```

The export script uses a raw cache window that starts before the first signal
date so expressions with `Ref` and rolling windows have enough history. The
default current snapshot is the 317-factor `snapshot_iter354_accepted317_20260618`.

4. Use alpha101 to inspect:
   - signed factor direction from mining ICIR,
   - long-only selected quantile performance,
   - universe-relative excess return,
   - signed PnL similarity to existing factors,
   - symbol-level raw values and ranks.
5. Keep final walk-forward training and holdings export in `microcap_alpha`.

## What Not To Do

- Do not rank factors on full test-period PnL and then report that as production
  performance.
- Do not use the latest unfinished forward label at a weekly rebalance point.
- Do not treat long-short spread as tradable A-share performance; use it as a
  diagnostic only.
- Do not deploy the 300+ accepted factors directly without ICIR ranking,
  neutralization A/B checks, and redundancy pruning.
