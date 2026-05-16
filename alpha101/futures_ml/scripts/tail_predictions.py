import pandas as pd
df = pd.read_parquet(r'alpha101/futures_ml/output/predictions.parquet')
df['asof_date'] = df['asof_date'].dt.strftime('%y%m%d')
df['trade_date'] = df['trade_date'].dt.strftime('%y%m%d')
df['symbol'] = df['symbol'].apply(lambda x: x.replace('_USDT_USDT','')) 
show_cols = ['symbol',
       'asof_date', 'trade_date', 'target', 'daily_open', 'daily_close',
       'predicted_return']
print(df[show_cols].tail(80))
sorted_df = df[df['target'].isna()][show_cols].sort_values('predicted_return')
print('short')
print(sorted_df.head(5))
print('long')
print(sorted_df.tail(6))
