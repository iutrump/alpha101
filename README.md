# alpha101

[English version](README.en.md)

![alpha101 overview](docs/assets/alpha101-overview.png)
**因子回测可视化**
![factor1 overview](docs/assets/factor1.png)

<!-- <p align="center">
  <img src="docs/assets/factor1.png" alt="factor example 1" width="32%">
  <img src="docs/assets/factor2.png" alt="factor example 2" width="32%">
  <img src="docs/assets/factor3.png" alt="factor example 3" width="32%">
</p> -->
`alpha101` 是一个面向加密货币市场的因子研究工具包，重点支持：

- 基于遗传算法的因子挖掘
- 因子回测Web可视化界面
- Alpha101 风格因子表达式

项目使用开源[Freqtrade](https://github.com/freqtrade/freqtrade)框架下载交易所 K 线数据，`alpha101` 负责把本地数据组织成因子研究面板，并在此基础上进行表达式计算、因子搜索和回测。

**该项目仅供学习研究，不提供任何投资建议，请不要将挖掘的因子进行实盘**

## 目录结构

```text
alpha101/
  cli/                         # 命令行入口
  data/                        # 数据加载、数组工具、数据视图与币种池
    loading.py                 # Freqtrade feather 数据加载与宽表构建
    views.py                   # 因子表达式运行时使用的数据视图
    arrays.py                  # 数组采样与相关性工具
    market_caps.py             # 市值和流通量元数据
    universe.py                # Binance USDT 永续币种池发现
  factors/
    operator_lib/              # pandas 因子算子实现、规格与注册表
    expression/                # 表达式执行引擎
    generation/                # 表达式 AST、语法、生成与遗传操作
    evaluation/                # 基础指标与因子评分
    backtesting/               # 多空组合回测
    search/                    # 因子搜索流程与结果管理
  integrations/
    freqtrade.py               # 调用 Freqtrade 下载数据的包装入口
  research/
    server.py                  # FastAPI 研究服务
    templates/                 # 研究服务页面
configs/
  alpha101.example.json        # alpha101 与 Freqtrade 合并配置示例
3rdparty/
  freqtrade/                   # Freqtrade 子模块
```

## 安装环境

推荐使用Linux/WSL进行开发，Windows环境因子搜索速度稍慢。

建议使用独立 conda 环境，环境名为 `alpha101`：

```bash
conda create -n alpha101 python=3.12
conda activate alpha101
```

安装freqtrade，用于数据下载
```bash
git submodule update --init --recursive --depth 1 3rdparty/freqtrade
python -m pip install -r 3rdparty/freqtrade/requirements.txt
python -m pip install -e 3rdparty/freqtrade
```

从仓库根目录安装本项目：

```bash
python -m pip install -e .
```

## 配置

先复制示例配置，避免直接修改模板文件：

```bash
cp configs/alpha101.example.json configs/alpha101.json
```

程序运行时默认会读取 `configs/alpha101.json`。配置文件上半部分是 alpha101 的研究参数，下半部分是 Freqtrade 下载数据需要的交易所和币种池参数。


也可以指定运行时配置：

```bash
export ALPHA101_CONFIG=configs/alpha101.json
```

Windows PowerShell：

```powershell
$env:ALPHA101_CONFIG = "configs/alpha101.json"
```


## 下载数据

首次使用 Freqtrade 前，需要先创建本地 userdir：

```bash
freqtrade create-userdir --userdir user_data
```

如果访问 Binance 需要代理，在 `configs/alpha101.json` 里配置：

```json
{
  "exchange": {
    "ccxt_config": {
      "httpsProxy": "http://127.0.0.1:7890",
      "wsProxy": "http://127.0.0.1:7890"
    }
  }
}
```

上面的端口适用于本机代理监听在 `127.0.0.1:7890` 的情况；如果你的代理端口不同，改成自己的地址即可。

通过 Freqtrade 下载 Binance futures 数据：


```bash
python -m alpha101.integrations.freqtrade \
  --freqtrade-bin freqtrade \
  --config configs/alpha101.json \
  --timeframes 1h 4h 1d \
  --timerange 20260101-20260630
```
默认下载`configs/alpha101.json`中的50个币种，可以自行修改配置。

获取市值最大的前50个 Binance USDT 永续币种池：


```bash
python -m alpha101.data.universe \
  --top-n 50 \
  --output configs/pairs.binance-usdt-perp.json
```
结果输出到`configs/pairs.binance-usdt-perp.json`，可以复制到 `configs/alpha101.json` 的 `exchange.pair_whitelist` 中。


## Web因子回测

启动轻量研究服务：

```bash
alpha101-research-server
```

然后打开：

```text
http://127.0.0.1:8001
```

研究服务用于快速输入表达式、查看因子表现、检查回测指标，适合做交互式因子筛选。



## 运行因子表达式

也可以使用命令行入口：

```bash
alpha101-expression "ts_rank(close, 10)"
```

表达式会在配置指定的数据面板上执行，并输出处理后的因子矩阵摘要。

## 因子搜索

运行遗传搜索：

```bash
alpha101-factor-search \
  --config configs/alpha101.json \
  --strategy genetic \
  --population 30 \
  --generations 5 \
  --n-jobs 8
```

搜索结果默认写入：

```text
factor_search_results/<timeframe>/
```

因子搜索流程大致是：

1. 从语法规则生成候选表达式
2. 每一代批量调用表达式引擎计算因子矩阵
3. 用截面 forward return 计算 IC、收益、Sharpe、回撤等评分
4. 通过遗传操作继续迭代表达式
5. 保存每批搜索结果和汇总结果



## 数据字段、算子和语法

表达式运行在宽表数据上，行是时间，列是交易对。当前可直接引用的数据字段包括：

- `open`、`high`、`low`、`close`、`volume`、`vwap`：基础 OHLCV 和成交均价字段
- `returns`：由 `close.pct_change()` 计算的收益率
- `market_return`：市场收益率；如果有 `cap`，默认使用市值前 15 的币种加权，否则使用截面均值
- `cap`：可选市值字段
- `funding`：可选资金费率字段

表达式可以使用 Python 风格的四则运算、比较运算和函数调用。运行时会把表达式转成小写，支持 `#` 行内注释和用分号分隔的临时变量：

```text
x = ts_delta(close, 1);
y = ts_rank(volume, 10);
rank(x / y)
```

常用内置函数包括 `abs`、`log`、`sign`、`sqrt`、`max`、`min`、`exp`。其中 `log` 是 signed log，`sqrt` 会对负数做安全处理。条件逻辑可以用 `if_else(condition, true_val, false_val)` 或 `trade_when(condition, alpha, exit)`。

当前注册的核心算子如下：

```text
时序单输入:
  ts_mean, ts_rank, ts_min, ts_max, ts_std_dev, ts_arg_max, ts_arg_min,
  ts_sum, ts_product, ts_skewness, ts_kurtosis, ts_decay_linear,
  ts_drawdown, ts_pos, ts_zscore, ts_ema, ts_slope

时序双输入:
  ts_corr, ts_covariance, ts_alpha, ts_r2, ts_beta, ts_resid

截面:
  rank, scale, zscore, winsorize

滞后/变化:
  ts_delay, ts_delta
```

表达式搜索生成器会从上述字段和算子中随机生成候选因子。生成规则默认包含 `+`、`-`、`*`、`/`，一元 `log`、`-`、`abs`、`sqrt`、`sign`，最大深度为 4，最多 8 个算子，并会避开部分低价值嵌套模式，例如 `ts_mean(ts_mean(...))`、`rank(rank(...))`、`ts_corr(ts_corr(...))`。
