import pandas as pd
df = pd.read_feather('user_data/data/binance/futures/BNB_USDT_USDT-1d-futures.feather')
print(df.tail())
