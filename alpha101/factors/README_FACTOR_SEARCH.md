# Factor Search

This module searches Alpha101-style expressions on a wide OHLCV panel loaded
from Freqtrade feather data.

## Prepare Config

```bash
cp configs/alpha101.example.json configs/alpha101.local.json
cp configs/freqtrade.example.json configs/freqtrade.local.json
export ALPHA101_CONFIG=configs/alpha101.local.json
```

Edit `configs/alpha101.local.json` so `timeframe`, date range, and
`strategy_config_path` match your data.

## Run Search

Random search:

```bash
python -m alpha101.factors.factor_search \
  --config configs/alpha101.local.json \
  --strategy random \
  --n-factors 200
```

Template search:

```bash
python -m alpha101.factors.factor_search \
  --config configs/alpha101.local.json \
  --strategy template
```

Grid search:

```bash
python -m alpha101.factors.factor_search \
  --config configs/alpha101.local.json \
  --strategy grid
```

Genetic search:

```bash
python -m alpha101.factors.factor_search \
  --config configs/alpha101.local.json \
  --strategy genetic \
  --population 30 \
  --generations 10
```

Results are written to:

```text
factor_search_results/<timeframe>/
```

## Evaluate One Expression

```bash
python -m alpha101.factors.expression_engine \
  --file expression.txt
```

or:

```bash
python -m alpha101.factors.expression_engine "ts_rank(close, 10)"
```

## Factor Research Server

```bash
python -m alpha101.factors.research_server
```

Open:

```text
http://127.0.0.1:8001
```

The research server is intended for quick factor inspection, not production
trading simulation.
