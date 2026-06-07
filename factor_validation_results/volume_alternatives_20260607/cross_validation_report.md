# Cross Validation Report

Source: manual expressions
Data: 2025-01-01 00:00:00+00:00 to 2026-06-02 00:00:00+00:00, 3103 bars, 219 symbols

## Decisions
- rejected | manual_001 | test_sh=nan valid_sh=nan | time_pos=0 time_med=-1.156 time_min=-2.514 | universe_pos=0 universe_min=-2.865 | `ts_mean(zscore(volume), 28)`
- rejected | manual_002 | test_sh=nan valid_sh=nan | time_pos=0 time_med=-1.156 time_min=-2.514 | universe_pos=0 universe_min=-2.865 | `ts_sum(zscore(volume), 28)`
- rejected | manual_003 | test_sh=nan valid_sh=nan | time_pos=0 time_med=-1.705 time_min=-1.815 | universe_pos=0 universe_min=-3.242 | `ts_rank(ts_mean(volume, 28), 20)`

## High-Correlation Clusters
- manual_001 (rejected, test_sh=nan); manual_002 (rejected, test_sh=nan)