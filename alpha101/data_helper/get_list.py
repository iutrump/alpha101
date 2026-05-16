import requests

def get_top_n_usdt_futures_symbols(n=-1):
    # 获取现货市场的交易对数据
    url_spot = "https://api.binance.com/api/v3/ticker/24hr"
    spot_data = requests.get(url_spot).json()

    # 获取期货市场的交易对数据
    url_futures = "https://fapi.binance.com/fapi/v1/exchangeInfo"
    futures_data = requests.get(url_futures).json()

    # 常见稳定币（可以自行补充）
    stablecoins = {
        "USDT", "USDC", "BUSD", "FDUSD",
        "TUSD", "USDP", "DAI", "USTC", "USD1"
    }

    # 获取期货交易对
    futures_symbols = set(future['symbol'] for future in futures_data['symbols'] if future['contractType'] == 'PERPETUAL')

    # 过滤符合条件的现货交易对
    filtered = []

    for ticker in spot_data:
        symbol = ticker["symbol"]

        # 1️⃣ 只要 USDT 交易对
        if not symbol.endswith("USDT"):
            continue

        # 2️⃣ 提取基础币种
        base = symbol[:-4]  # 去掉 USDT

        # 3️⃣ 排除稳定币本身
        if base in stablecoins:
            continue

        # 4️⃣ 使用 quoteVolume 作为排序依据（USDT计价成交额）
        volume = float(ticker["quoteVolume"])

        # 5️⃣ 确保是期货合约（通过匹配期货合约列表）
        if symbol in futures_symbols:
            filtered.append((symbol.replace('USDT','/USDT:USDT'), volume))

    # 6️⃣ 按成交额排序
    filtered.sort(key=lambda x: x[1], reverse=True)

    # 7️⃣ 取前 n 个
    top_n = [symbol for symbol, _ in filtered[:n]] if n > 0 else [symbol for symbol, _ in filtered]

    return top_n


# 示例
symbols = get_top_n_usdt_futures_symbols()
print(f"Top {len(symbols)} USDT Futures Symbols:", symbols)
import json
with open("user_data/coins/binance_futures_symbols.json", "w", encoding="utf-8") as f:    
    f.write(json.dumps({"pair_whitelist":symbols}, indent=2))
print("Saved to user_data/coins/binance_futures_symbols.json")