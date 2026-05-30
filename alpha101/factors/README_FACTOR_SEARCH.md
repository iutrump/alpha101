# Factor Search

This package searches Alpha101-style factor expressions on a wide OHLCV panel.

## Run

```bash
python -m alpha101.factors.factor_search \
  --config configs/alpha101.local.json \
  --strategy random \
  --n-factors 200
```

Supported strategies:

- `random`
- `template`
- `grid`
- `genetic`
- `all`

Results are written to `factor_search_results/<timeframe>/`.

## Evaluate One Expression

```bash
python -m alpha101.factors.expression_engine "ts_rank(close, 10)"
```

## Research Server

```bash
python -m alpha101.factors.research_server
```

Open `http://127.0.0.1:8001`.
