# Factor Search

This package searches Alpha101-style factor expressions on a wide OHLCV panel.
The current search path is genetic algorithm mining only; random expressions
are used internally for population initialization.

## Run

```bash
python -m alpha101.factors.factor_search \
  --config configs/alpha101.local.json \
  --strategy genetic \
  --population 30 \
  --generations 5
```

Results are written to `factor_search_results/<timeframe>/`.

## Evaluate One Expression

```bash
python -m alpha101.factors.expression_engine "ts_rank(close, 10)"
```

## Research Server

```bash
python -m alpha101.research.server
```

Open `http://127.0.0.1:8001`.
