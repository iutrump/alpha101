# alpha101

`alpha101` is a crypto factor research toolkit focused on Alpha101-style factor
definition, expression generation, factor search, and lightweight visual
inspection.

The repository uses Freqtrade only as an optional third-party data backend. At
this stage Freqtrade is responsible for downloading exchange data; this project
keeps the factor mining code independent from the trading framework.

## Repository Layout

```text
alpha101/
  config.py                  # JSON/env config loader
  data/
    panel.py                 # Freqtrade feather data loader
    universe.py              # Binance USDT perpetual universe discovery
    market_metadata.py       # Market cap and circulating supply metadata
  factors/                   # Alpha operators, expression engine, factor search, research server
  integrations/
    freqtrade.py             # Freqtrade download-data wrapper
configs/
  alpha101.example.json      # Project config example
  freqtrade.example.json     # Freqtrade config example
3rdparty/
  freqtrade/                 # Optional Freqtrade submodule
```

## Environment And Install

```bash
conda create -n alpha101 python=3.12
conda activate alpha101
python -m pip install --upgrade pip
```

Initialize the shallow Freqtrade submodule:

```bash
git submodule update --init --recursive --depth 1 3rdparty/freqtrade
```

Install Freqtrade first:

```bash
pip install -r 3rdparty/freqtrade/requirements.txt
pip install -e 3rdparty/freqtrade
```

Then install this project:

```bash
pip install -e .
```

For the visual inspection server:

```bash
pip install -e ".[visual]"
```

## Freqtrade

Recommended: keep Freqtrade as a shallow submodule to avoid cloning its full
history.

```bash
git submodule add --depth 1 https://github.com/freqtrade/freqtrade.git 3rdparty/freqtrade
git submodule update --init --recursive --depth 1 3rdparty/freqtrade
```

For users cloning this repository later:

```bash
git clone <repo-url>
cd alpha101
git submodule update --init --recursive --depth 1 3rdparty/freqtrade
```

Important: `--depth 1` on `git submodule add` makes the initial add shallow, but
future users still need `--depth 1` on `git submodule update` unless the
submodule is also configured as shallow in `.gitmodules`.

You can also skip the submodule and use a system-installed Freqtrade CLI:

```bash
pip install freqtrade
python -m alpha101.integrations.freqtrade --freqtrade-bin freqtrade
```

## Configuration

Runtime configuration is JSON-first. Copy the examples before local changes:

```bash
cp configs/alpha101.example.json configs/alpha101.local.json
cp configs/freqtrade.example.json configs/freqtrade.local.json
```

Then point the project to your local config:

```bash
export ALPHA101_CONFIG=configs/alpha101.local.json
```

## Download Data

```bash
python -m alpha101.integrations.freqtrade \
  --freqtrade-bin freqtrade \
  --config configs/freqtrade.local.json \
  --timeframes 1h 4h 1d \
  --timerange 20250101-20260330
```

Downloaded data is expected under:

```text
user_data/data/binance/futures/
```

with Freqtrade feather files such as:

```text
BTC_USDT_USDT-4h-futures.feather
ETH_USDT_USDT-4h-futures.feather
```

`alpha101.data.market_metadata` is kept for research features that need
`market_cap_usd` or `circulating_supply`. Freqtrade has `MarketCapPairList`, but
that plugin is for pairlist ranking/filtering by CoinGecko market-cap rank; it
does not write market-cap or circulating-supply columns into the local factor
research panel.

To discover a Binance USDT perpetual universe:

```bash
python -m alpha101.data.universe \
  --top-n 50 \
  --output configs/pairs.binance-usdt-perp.json
```

## Factor Search

```bash
python -m alpha101.factors.factor_search \
  --config configs/alpha101.local.json \
  --strategy random \
  --n-factors 100
```

## Factor Research Server

The research server is kept as a lightweight factor inspection tool:

```bash
python -m alpha101.factors.research_server
```

Then open:

```text
http://127.0.0.1:8001
```
