from __future__ import annotations

import os
import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests
import sys
sys.path.append(str(Path(__file__).parent.parent))  # To import from alpha101.futures_ml.config
sys.path.append(str(Path(__file__).parent))  # To import from alpha101.futures_ml.config
from futures_ml.config import get_config
import subprocess

COINPAPRIKA_TICKERS_URL = "https://api.coinpaprika.com/v1/tickers"


def extract_base_symbol(pair: str) -> str:
    """
    Convert futures pair format to base symbol.
    Example: BTC_USDT_USDT -> BTC
    """
    return pair.split("_", 1)[0].upper()


def fetch_market_data_from_coinpaprika(
    symbols: List[str], timeout: int = 20, http_proxy: str = None, https_proxy: str = None
) -> Dict[str, Dict[str, float | None]]:
    """
    Query CoinPaprika tickers and map by symbol to price/cap/supply.
    If multiple coins share a symbol, choose the one with highest market cap.
    """
    result: Dict[str, Dict[str, float | None]] = {
        s.upper(): {
            "price_usd": None,
            "market_cap_usd": None,
            "circulating_supply": None,
        }
        for s in symbols
    }
    if not symbols:
        return result

    target = {s.upper() for s in symbols if s}
    if http_proxy or https_proxy:
        print(f"Using proxies - HTTP: {http_proxy}, HTTPS: {https_proxy}")
        resp = requests.get(COINPAPRIKA_TICKERS_URL, timeout=timeout, proxies={"http": http_proxy, "https": https_proxy})
    else:
        resp = requests.get(COINPAPRIKA_TICKERS_URL, timeout=timeout)
    resp.raise_for_status()
    payload = resp.json()
    print(f"Status code: {resp.status_code} for {COINPAPRIKA_TICKERS_URL}")
    print(f"Fetched {len(payload)} tickers from CoinPaprika")
    best_by_symbol: Dict[str, Dict[str, float | None]] = {}
    for coin in payload:
        sym = str(coin.get("symbol", "")).upper()
        if sym not in target:
            continue
        usd_quote = (coin.get("quotes") or {}).get("USD", {})
        cap = usd_quote.get("market_cap")
        price = usd_quote.get("price")
        if cap is None or price is None:
            continue
        cap_val = float(cap)
        price_val = float(price)
        supply_val = cap_val / price_val if price_val > 0 else None
        if sym not in best_by_symbol or cap_val > float(best_by_symbol[sym]["market_cap_usd"] or 0):
            best_by_symbol[sym] = {
                "price_usd": price_val,
                "market_cap_usd": cap_val,
                "circulating_supply": supply_val,
            }

    for sym, values in best_by_symbol.items():
        if sym in result:
            result[sym] = values

    return result


def get_pair_market_caps(pairs, provider: str = "coinpaprika", http_proxy: str = None, https_proxy: str = None) -> pd.DataFrame:
    cache_json_path = _default_cache_json_path(provider=provider)
    try:
        bases = [extract_base_symbol(pair) for pair in pairs]
        market_data_map = fetch_market_data_from_coinpaprika(bases, http_proxy=http_proxy, https_proxy=https_proxy)

        rows = []
        for pair, symbol in zip(pairs, bases):
            market_data = market_data_map.get(symbol, {})
            rows.append(
                {
                    "pair": pair,
                    "symbol": symbol,
                    "price_usd": market_data.get("price_usd"),
                    "market_cap_usd": market_data.get("market_cap_usd"),
                    "circulating_supply": market_data.get("circulating_supply"),
                }
            )

        df = pd.DataFrame(rows)
        df = df.sort_values(by="market_cap_usd", ascending=False, na_position="last").reset_index(drop=True)
        if df["market_cap_usd"].isnull().all():
            raise ValueError("All market cap values are null. Data fetch may have failed.")             
        _save_cache_json(df, cache_json_path)
        print(f"Fetched market caps from {provider} and saved cache JSON: {cache_json_path}")
        return df
    except Exception as exc:
        if not cache_json_path.exists():
            raise RuntimeError(
                f"Fetch failed and cache file not found: {cache_json_path}"
            ) from exc
        warnings.warn(
            f"Fetch failed: {exc}. Using cached JSON: {cache_json_path}",
            RuntimeWarning,
            stacklevel=2,
        )
        return _load_cache_json(cache_json_path)

def get_pair_market_caps_last_and_update(pairs, provider: str = "coinpaprika", cache_dir: str = "",http_proxy: str = None, https_proxy: str = None) -> pd.DataFrame:
    '''
    Too long to fetch, 
    so we can call history get_pair_market_caps, 
    then call this function to update the cache with new fetch in background for next time.
    '''
    cache_json_path = _default_cache_json_path(cache_dir, provider=provider)
    if not cache_json_path.exists():
        get_pair_market_caps(pairs, provider=provider, http_proxy=http_proxy, https_proxy=https_proxy)
    else:
        subprocess.Popen(
            [sys.executable, __file__, "--provider", provider],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return _load_cache_json(cache_json_path)

def _default_cache_json_path(
    cache_dir: str = "",
    provider: str = "auto",
) -> Path:
    if cache_dir == "":
        cache_dir = Path('cache_cap_data')
    else:
        cache_dir = Path(cache_dir)
    print('cache dir', cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"pair_market_caps_cache_latest_{provider}.json"


def _save_cache_json(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "count": len(df),
        "data": df.to_dict(orient="records"),
        "pair_list": df["pair"].tolist(),
        "symbol_list": [f"{e}/USDT:USDT" for e in df["symbol"].tolist()],
        "large_cap_count": int(df["market_cap_usd"].gt(1e10).sum()),
        "large_cap_symbols": [f"{e}/USDT:USDT" for e in df.loc[df["market_cap_usd"].gt(1e10), "symbol"].tolist()],
        "medium_cap_count": int(df["market_cap_usd"].between(1e8, 1e10).sum()),
        "medium_cap_symbols": [f"{e}/USDT:USDT" for e in df.loc[df["market_cap_usd"].between(1e8, 1e10), "symbol"].tolist()],
        "small_cap_count": int(df["market_cap_usd"].between(1e7, 1e8).sum()),
        "small_cap_symbols": [f"{e}/USDT:USDT" for e in df.loc[df["market_cap_usd"].between(1e7, 1e8), "symbol"].tolist()],
        "micro_cap_count": int(df["market_cap_usd"].between(1e6, 1e7).sum()),
        "micro_cap_symbols": [f"{e}/USDT:USDT" for e in df.loc[df["market_cap_usd"].between(1e6, 1e7), "symbol"].tolist()],
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def _load_cache_json(path: Path) -> pd.DataFrame:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return pd.DataFrame(payload)
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        return pd.DataFrame(payload["data"])
    raise ValueError("Cache json format is invalid (expected list or dict with data field).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Get market caps for symbols in futures_ml config pairs.")
    parser.add_argument("--save-csv", type=str, default="", help="CSV output path. Default is auto-generated.")
    parser.add_argument("--save-json", type=str, default="", help="Optional JSON output path.")
    parser.add_argument(
        "--http-proxy",
        type=str,
        default=None,
        help="HTTP proxy URL (e.g. http://user:pass@host:port).",
    )
    parser.add_argument(
        "--https-proxy",
        type=str,
        default=None,
        help="HTTPS proxy URL (e.g. http://user:pass@host:port).",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="coinpaprika",
        choices=["coinpaprika"],
        help="Market data source.",
    )

    args = parser.parse_args()
    cfg = get_config()
    
    df = get_pair_market_caps(cfg.pairs, provider=args.provider, http_proxy=args.http_proxy, https_proxy=args.https_proxy)

    if args.save_json:
        json_path = Path(args.save_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "data": df.to_dict(orient="records"),
        }
        json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Saved JSON: {json_path}")


if __name__ == "__main__":
    main()
