# futures_ml: futures alpha101 demo

A minimal pipeline to train an XGBoost model on the last 7 days of Alpha101 factors, predict the next-day return, form a daily long/short basket (top 4 / bottom 4), backtest with hourly bars, and plot equity vs BTC benchmark.

## Data assumptions
- Daily features use 1d futures data located under `user_data/data/binance/futures` with files like `BTC_USDT_USDT-1d-futures.feather`.
- Intraday simulation uses 1h data with matching filenames ending in `-1h-futures.feather`.
- Files must contain columns `date, open, high, low, close, volume` (optional `quoteVolume`). Time is assumed UTC.

## Quick start
1) Install deps (inside your venv):
```
pip install xgboost matplotlib pandas pyarrow
```
2) Run the pipeline (defaults are in `config.py`):
```
python -m alpha101.futures_ml.run_pipeline
```
Outputs land in `alpha101/futures_ml/output`:
- `predictions.parquet`: test-set predictions with targets.
- `backtest_daily.parquet`: daily portfolio returns from the rule-based backtest.
- `equity_curve.png`: cumulative returns vs BTC benchmark.

## What it does
- Builds Alpha101 features from `alpha101/world_quant/101Alpha_code_1.py` per pair.
- Uses the previous 7 days of alpha factors (lags 1..7) to predict the next-day close-to-close return.
- Time-based split: first 80% dates for training, last 20% for testing.
- Daily portfolio: long best 4, short worst 4 by predicted return at 00:00; 2% TP / 2% SL evaluated on 1h bars; flat at day end if untouched.

## Configuration
Edit `config.py` to adjust pairs, take-profit/stop-loss, lookback, paths, and split ratio. The provided pair list matches the symbols downloaded in the README snippet.

## alpha an
python -m alpha101.futures_ml.alpha_analysis
