# Cross Validation Report

Source: manual expressions
Data: 2025-01-01 00:00:00+00:00 to 2026-06-02 00:00:00+00:00, 3103 bars, 219 symbols

## Decisions
- accepted_candidate | manual_001 | test_sh=nan valid_sh=nan | time_pos=6 time_med=1.030 time_min=0.289 | universe_pos=3 universe_min=1.881 | `-ts_mean(zscore(volume), 28)`
- accepted_candidate | manual_002 | test_sh=nan valid_sh=nan | time_pos=6 time_med=1.030 time_min=0.289 | universe_pos=3 universe_min=1.881 | `-ts_sum(zscore(volume), 28)`
- rejected | manual_003 | test_sh=nan valid_sh=nan | time_pos=0 time_med=-1.469 time_min=-3.845 | universe_pos=0 universe_min=-3.741 | `-ts_rank(ts_mean(volume, 28), 20)`

## High-Correlation Clusters
- manual_001 (accepted_candidate, test_sh=nan); manual_002 (accepted_candidate, test_sh=nan)