### 想法
选取交易量前几名的加密货币，输入过去几天的alpha因子，利用机器学习xgboost等模型，预测未来一天的涨跌幅度。做多选取的前4种货币，做空后4种加密货币

现在需要搭建训练和回测以及可视化的平台
### 训练
alpha因子在`alpha101\world_quant\101Alpha_code_1.py`

取前7天的alpha因子进行试验，用xgboost模型，80%的时间训练，后20%时间周期的用于测试
### 回测
在每天0点以开盘价格买入，设置2%的止盈和2%的止损，以1h为周期查看当前的价格
### 可视化
将每天的收益可视化，与大盘对比，并且计算盈利和sharpe值

### 币种和数据
#### 路径
`user_data\data\binance\futures\ADA_USDT_USDT-1d-futures.feather`
`user_data\data\binance\futures\ADA_USDT_USDT-1h-futures.feather`
...
交易时间
20240801-20251231
3m 15m 1h 1d
BTC/USDT:USDT ZEC/USDT:USDT XRP/USDT:USDT  BNB/USDT:USDT   ADA/USDT:USDT SOL/USDT:USDT  DOGE/USDT:USDT ETH/USDT:USDT LINK/USDT:USDT