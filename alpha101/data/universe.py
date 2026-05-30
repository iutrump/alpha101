import argparse
import json
from pathlib import Path

import requests

STABLECOINS = {
    "USDT",
    "USDC",
    "BUSD",
    "FDUSD",
    "TUSD",
    "USDP",
    "DAI",
    "USTC",
    "USD1",
}


def fetch_binance_usdt_perpetual_universe(
    top_n: int | None = None,
    exclude_stablecoins: bool = True,
    timeout: int = 20,
) -> list[str]:
    """Return Binance USDT perpetual pairs sorted by spot quote volume."""
    url_spot = "https://api.binance.com/api/v3/ticker/24hr"
    spot_data = requests.get(url_spot, timeout=timeout).json()

    url_futures = "https://fapi.binance.com/fapi/v1/exchangeInfo"
    futures_data = requests.get(url_futures, timeout=timeout).json()

    futures_symbols = set(future['symbol'] for future in futures_data['symbols'] if future['contractType'] == 'PERPETUAL')

    filtered = []

    for ticker in spot_data:
        symbol = ticker["symbol"]
        if not symbol.endswith("USDT"):
            continue
        base = symbol[:-4]
        if exclude_stablecoins and base in STABLECOINS:
            continue
        volume = float(ticker["quoteVolume"])
        if symbol in futures_symbols:
            filtered.append((symbol.replace("USDT", "/USDT:USDT"), volume))

    filtered.sort(key=lambda x: x[1], reverse=True)
    limit = top_n if top_n and top_n > 0 else None
    return [symbol for symbol, _ in filtered[:limit]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Binance USDT perpetual symbols by spot quote volume.")
    parser.add_argument("-n", "--top-n", type=int, default=-1)
    parser.add_argument("--include-stablecoins", action="store_true")
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument(
        "-o",
        "--output",
        default="user_data/coins/binance_futures_symbols.json",
        help="Output JSON path.",
    )
    args = parser.parse_args()

    symbols = fetch_binance_usdt_perpetual_universe(
        top_n=args.top_n,
        exclude_stablecoins=not args.include_stablecoins,
        timeout=args.timeout,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"pair_whitelist": symbols}, indent=2), encoding="utf-8")
    print(f"Saved {len(symbols)} symbols to {output}")


if __name__ == "__main__":
    main()
