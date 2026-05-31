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
  data/                        # 数据加载、数据视图、币种池与市场元数据
    alpha_view.py              # 因子表达式运行时使用的数据视图
    panel.py                   # Freqtrade feather 数据加载与宽表构建
    universe.py                # Binance USDT 永续币种池发现
    market_metadata.py         # 市值和流通量元数据
  factors/
    operator_lib/              # 因子算子实现与注册表
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

通过 Freqtrade 下载 Binance futures 数据：

```bash
python -m alpha101.integrations.freqtrade \
  --freqtrade-bin freqtrade \
  --config configs/freqtrade.local.json \
  --timeframes 1h 4h 1d \
  --timerange 20250101-20260330
```

默认期望数据位于：

```text
user_data/data/binance/futures/
```

文件名类似：

```text
BTC_USDT_USDT-4h-futures.feather
ETH_USDT_USDT-4h-futures.feather
```

生成 Binance USDT 永续币种池：

```bash
python -m alpha101.data.universe \
  --top-n 50 \
  --output configs/pairs.binance-usdt-perp.json
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

`--n-jobs` 控制批量表达式计算和评分的进程数。搜索路径使用专门的 worker：worker 内部完成表达式计算、因子预处理和评分，只把 metrics 返回主进程，避免把完整因子矩阵在进程之间传来传去。表达式数量较多、rolling/corr/rank 较重时通常会更快；如果表达式很少或机器内存紧张，可以改小，或者使用 `--backend serial` 串行执行。

默认 `--backend auto`：Linux/WSL 下会优先使用多进程；Windows 原生环境下会默认退回串行执行，因为 Windows 的 `spawn` 多进程启动和大对象复制成本较高。确实要在 Windows 上强制多进程时，可以显式传 `--backend process`，但建议先从较小的 `--n-jobs 2` 或 `--n-jobs 4` 开始。

## 回测

回测模块位于：

```text
alpha101/factors/backtesting/
```

当前主要支持基于因子排序分组的多空回测，指标包括：

- Sharpe
- 扣费后 Sharpe
- CAGR
- 扣费后 CAGR
- 年化收益
- 回撤
- 换手
- IC 均值与 ICIR
- funding 成本

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

## 算子开发

算子代码位于：

```text
alpha101/factors/operator_lib/
```

当前拆分为：

- `time_series.py`：时间序列算子
- `cross_section.py`：截面算子
- `regression.py`：回归相关算子
- `transforms.py`：因子后处理与转换
- `registry.py`：算子注册表与生成元数据

新增算子时需要同时关注：

1. 在合适的实现文件中添加函数
2. 加入该文件的 `__all__`
3. 如果要参与表达式生成，在 `registry.py` 的 `OPERATOR_SPECS` 中增加元数据
4. 跑编译和相关 smoke tests

## 代码分层约定

当前模块边界：

- `data`：数据加载、数据视图、市场元数据
- `operator_lib`：表达式可调用的底层算子
- `expression`：表达式解析后的执行环境
- `generation`：表达式生成、变异、复杂度和语义检查
- `evaluation`：基础统计指标和因子评分
- `backtesting`：组合构建、收益曲线和回测指标
- `search`：搜索流程、缓存、结果保存
- `research`：Web 研究界面

不要把有明确领域含义的逻辑放入泛化的 `utils`。优先把代码放到对应领域包中；只有跨领域、无业务语义的纯辅助函数才考虑单独抽出。

## 基本验证

编译检查：

```bash
conda run -n alpha101 python -m compileall -q alpha101 tests
```

安装开发依赖后运行测试：

```bash
pytest
```

如果只想快速确认核心路径，可以运行项目已有的 smoke tests：

```bash
conda run -n alpha101 python -c "from tests.test_expression_runtime import test_operator_registry_has_explicit_public_names, test_operator_specs_drive_generation_categories, test_expression_batch_serial_and_process_match; from tests.test_backtest import test_backtest_long_short_returns_curve_and_metrics; from tests.test_scoring import test_score_factor_cross_section_returns_expected_keys; test_operator_registry_has_explicit_public_names(); test_operator_specs_drive_generation_categories(); test_expression_batch_serial_and_process_match(); test_backtest_long_short_returns_curve_and_metrics(); test_score_factor_cross_section_returns_expected_keys(); print('smoke tests ok')"
```
