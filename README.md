# alpha101

`alpha101` 是一个面向加密货币市场的因子研究工具包，重点支持：

- Alpha101 风格因子表达式
- 向量化算子库
- 随机/遗传因子生成与搜索
- 截面因子评分
- 多空组合回测
- 简单的 Web 因子检查界面

项目使用 Freqtrade 下载交易所 K 线数据，`alpha101` 负责把本地数据组织成因子研究面板，并在此基础上做表达式计算、因子搜索和回测。

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
  alpha101.example.json        # alpha101 配置示例
  freqtrade.example.json       # Freqtrade 配置示例
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
cp configs/alpha101.example.json configs/alpha101.local.json
cp configs/freqtrade.example.json configs/freqtrade.local.json
```
将`configs/alpha101.local.json`中的`strategy_config_path`改为`configs/freqtrade.local.json`

程序运行时默认会读取 `configs/alpha101.local.json`

也可以指定运行时配置：

```bash
export ALPHA101_CONFIG=configs/alpha101.local.json
```

Windows PowerShell：

```powershell
$env:ALPHA101_CONFIG = "configs/alpha101.local.json"
```


## 下载数据

首次使用 Freqtrade 前，需要先创建本地 userdir：

```bash
freqtrade create-userdir --userdir user_data
```

如果访问 Binance 需要代理，在 `configs/freqtrade.local.json` 里配置：

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

获取市值最大的前50个 Binance USDT 永续币种池：


```bash
python -m alpha101.data.universe \
  --top-n 50 \
  --output configs/pairs.binance-usdt-perp.json
```
然后复制到`configs/alpha101.local.json`中。

通过 Freqtrade 下载 Binance futures 数据：


```bash
python -m alpha101.integrations.freqtrade \
  --freqtrade-bin freqtrade \
  --config configs/freqtrade.local.json \
  --timeframes 1h 4h 1d \
  --timerange 20250101-20260330
```


## 运行因子表达式

安装后可以使用命令行入口：

```bash
alpha101-expression "ts_rank(close, 10)"
```

表达式会在配置指定的数据面板上执行，并输出处理后的因子矩阵摘要。

## 因子搜索

运行遗传搜索：

```bash
alpha101-factor-search \
  --config configs/alpha101.local.json \
  --strategy genetic \
  --population 30 \
  --generations 5 \
  --n-jobs 8 \
  --expression-backend pandas
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


## 研究服务

启动轻量研究服务：

```bash
alpha101-research-server
```

然后打开：

```text
http://127.0.0.1:8001
```

研究服务用于快速输入表达式、查看因子表现、检查回测指标，适合做交互式因子筛选。
