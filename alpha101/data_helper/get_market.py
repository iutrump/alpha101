import ccxt
import json
from pathlib import Path

exchange = ccxt.binance({
    # 'apiKey': '<apikey>',
    # 'secret': '<secret>',
    "ccxt_config": {
        "httpsProxy": "http://127.0.0.1:7890",
        "wsProxy": "http://127.0.0.1:7890"
    },
    'options': {'defaultType': 'swap'}
    })
market_info = exchange.load_markets()
print(market_info)
# https://api.binance.com/api/v3/exchangeInfo