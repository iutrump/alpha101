# Alpha 因子自动挖掘 Agent

这是一个基于大语言模型 (LLM) 的自动化因子挖掘系统，用于加密货币期货市场的量化交易因子研发。该系统通过与 OpenAI GPT-4 模型的交互，实现因子生成、回测和迭代优化的完整工作流。

## 功能特性

### 🚀 核心功能

1. **LLM 驱动的方向选择**: 从量价背离、动量、反转、均值回归等多个方向随机选择
2. **智能因子生成**: 基于市场上下文和可用算子生成高质量因子创意
3. **表达式构造**: 自动将因子创意转换为可执行的表达式
4. **自动回测**: 集成 stratified backtest 进行因子评估
5. **LLM 反思迭代**: 基于回测结果进行分析和改进建议
6. **批量处理**: 支持多次迭代，在相同或不同方向上进行探索
### 数据字段
open, high, close, volume, low, cap, funding, 

### 📊 支持的算子类型

#### 时间序列算子 (单变量)
- `ts_mean(x, window)` - 时间序列均值
- `ts_std_dev(x, window)` - 标准差
- `ts_rank(x, window)` - 排名
- `ts_min/ts_max(x, window)` - 最小/最大值
- `ts_sum(x, window)` - 求和
- `ts_arg_max/ts_arg_min(x, window)` - 极值位置

#### 时间序列算子 (双变量)
- `ts_corr(x, y, window)` - 相关性
- `ts_r2(x, y, window)` - 相关性
- `ts_covariance(x, y, window)` - 协方差
- `ts_alpha(x, y, window)` - 线性回归后的alpha项
- `ts_beta(x, y, window)` - 线性回归后的beta项
- `ts_resid(x, y, window)` - 线性回归后的残差项

#### 截面算子
- `rank(x)` - 截面排名
- `scale(x)` - 截面标准化

#### 延迟算子
- `ts_delay(x, window)` - 延迟
- `ts_delta(x, window)` - 差分

#### 数学运算
- 基本运算: `+`, `-`, `*`, `/`
- 函数: `log()`, `abs()`, `sign()`

#### 条件控制与逻辑运算
- `if_else(condition, true_val, false_val)` - 条件选择，condition 为 True 时取 true_val，否则取 false_val
- `>` - 大于
- `<` - 小于
- `>=` - 大于等于
- `<=` - 小于等于
- `==` - 等于
- `!=` - 不等于
- `&` - 逻辑且
- `|` - 逻辑或
- `~` - 逻辑非

#### 例子


## 工作流程

```
┌─────────────────────────────────────┐
│  1. 随机选择挖掘方向                 │
│  (量价背离/动量/反转等)             │
└────────────┬────────────────────────┘
             │
┌────────────▼────────────────────────┐
│  2. LLM 生成因子创意 (3个)           │
│  包含: 名称、假设、算子、方向       │
└────────────┬────────────────────────┘
             │
┌────────────▼────────────────────────┐
│  3. LLM 构造因子表达式               │
│  基于示例和创意                     │
└────────────┬────────────────────────┘
             │
┌────────────▼────────────────────────┐
│  4. 后测评估                         │
│  计算 Sharpe、Returns、IC 等指标     │
└────────────┬────────────────────────┘
             │
┌────────────▼────────────────────────┐
│  5. LLM 反思和分析                   │
│  评估性能、识别优缺点、提出建议     │
└────────────┬────────────────────────┘
             │
┌────────────▼────────────────────────┐
│  6. LLM 迭代改进                     │
│  基于建议生成改进版本                │
└────────────┬────────────────────────┘
             │
┌────────────▼────────────────────────┐
│  7. 再次评估和保存结果               │
│  重复，继续下一个迭代                │
└─────────────────────────────────────┘
```

## 安装和配置

### 1. 环境要求

```bash
python >= 3.8
pandas >= 1.3.0
numpy >= 1.20.0
scikit-learn >= 0.24.0
openai >= 1.0.0
python-dotenv >= 0.19.0
```

### 2. 安装依赖

```bash
# 安装 OpenAI 客户端库
pip install openai python-dotenv

# 或安装所有必需的包
pip install -r requirements.txt
```

### 3. 配置 OpenAI API

#### 方法 1: 创建 .env 文件

在 `alpha101/agent/` 目录下创建 `.env` 文件:

```bash
cp .env.example .env
```

编辑 `.env` 文件，填入你的 OpenAI 配置:

```ini
# OpenAI API Configuration
OPENAI_API_KEY=your_api_key_here
OPENAI_API_BASE=https://api.openai.com/v1
OPENAI_MODEL=gpt-4

# Agent Configuration
OUTPUT_DIR=agent_results
DATA_ROOT=../futures_ml/data
LOOKBACK_DAYS=252

# Backtest Parameters
TEST_START_DATE=2024-01-01
TEST_END_DATE=2025-12-31

# Search Parameters
POPULATION_SIZE=30
GENERATIONS=5
MAX_FACTOR_COMPLEXITY=6
```

#### 方法 2: 使用环境变量

直接设置环境变量:

```bash
# Windows PowerShell
$env:OPENAI_API_KEY = "your_api_key_here"
$env:OPENAI_API_BASE = "https://api.openai.com/v1"

# Linux/macOS
export OPENAI_API_KEY="your_api_key_here"
export OPENAI_API_BASE="https://api.openai.com/v1"
```

## 使用方法

### 基础用法

```bash
# 运行单个迭代
python -m alpha101.agent.agent --iterations 1

# 运行多个迭代
python -m alpha101.agent.agent --iterations 5

# 指定 .env 文件路径
python -m alpha101.agent.agent --iterations 3 --env-file ./custom.env
```

### Python API 调用

```python
from alpha101.agent import AlphaAgent, get_config

# 初始化配置
config = get_config()

# 创建 Agent
agent = AlphaAgent(config)

# 运行 3 次迭代
agent.run(n_iterations=3)
```

### 自定义配置

```python
from alpha101.agent import AlphaAgent, AgentConfig

# 创建自定义配置
config = AgentConfig(env_file='./my_config.env')

# 验证配置
if config.validate():
    agent = AlphaAgent(config)
    agent.run(n_iterations=2)
```

## 输出文件

运行完成后，会在 `agent_results/` 目录下生成以下文件:

### 1. all_factors_*.csv
包含所有评估的因子信息:
- `name`: 因子名称
- `expression`: 因子表达式
- `direction`: 挖掘方向
- `sharpe_ratio`: Sharpe 比率
- `returns`: 年化收益
- `ic_mean`: IC 均值
- `fitness`: 综合适应度评分
- `notes`: 备注信息

### 2. session_history_*.json
完整的会话历史，包含:
- 每个迭代的因子创意
- 原始因子的评估结果
- LLM 的反思分析
- 改进后因子的结果

### 3. config_*.json
配置信息快照:
- 市场信息
- 可用算子列表
- 表达式示例

## 输出示例

```
================================================================================
启动 Alpha 因子自动挖掘 Agent
================================================================================

================================================================================
迭代 1/3
================================================================================

第一步: 选择挖掘方向...
选择的方向: 动量

第二步: 生成因子创意...
生成了 3 个因子创意

创意 1: 短期动量因子
  假设: 近期收益率高的资产在短期内会继续上涨
  构造表达式...
  表达式: ts_mean(close, 10) / ts_mean(close, 30) - 1

评估因子: iter1_idea1
表达式: ts_mean(close, 10) / ts_mean(close, 30) - 1
✓ Sharpe: 0.5234, Returns: 0.0856, Fitness: 52.34

  第五步: LLM 反思和改进...
  评级: 一般
  迭代改进表达式...
  改进表达式: ts_mean(returns, 10) * ts_std_dev(returns, 5)

评估因子: iter1_idea1_improved
表达式: ts_mean(returns, 10) * ts_std_dev(returns, 5)
✓ Sharpe: 0.6145, Returns: 0.0945, Fitness: 61.45
```

## 表达式样例

### 1. 动量因子
```python
# 简单动量
ts_mean(returns, 10) / ts_mean(returns, 20)

# 加权动量
ts_rank(returns, 10) / 10 - 0.5

# 多周期动量
(ts_mean(returns, 5) - ts_mean(returns, 10)) / ts_std_dev(returns, 20)
```

### 2. 反转因子
```python
# 简单反转
(-1) * ts_rank(returns, 5) / 5

# 延迟反转
ts_delay(returns, 1) * (-1)

# 相对反转
(ts_min(returns, 20) - returns) / (ts_max(returns, 20) - ts_min(returns, 20))
```

### 3. 量价背离
```python
# 价量相关性
ts_corr(returns, volume, 20)

# 价量偏离
abs(ts_rank(returns, 10) - ts_rank(volume, 10)) / 10

# 成交量移动平均
log(volume) - ts_mean(log(volume), 30)
```

### 4. 均值回归
```python
# 布林带中位线
(close - ts_mean(close, 20)) / ts_std_dev(close, 20)

# 极值回复
(close - ts_min(close, 30)) / (ts_max(close, 30) - ts_min(close, 30))

# 斜率均值回归
(-1) * ts_alpha(close, 20) / ts_std_dev(close, 20)
```

## 关键参数说明

### Agent 配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `OPENAI_API_KEY` | - | OpenAI API 密钥（必需） |
| `OPENAI_MODEL` | gpt-4 | 使用的 LLM 模型 |
| `OUTPUT_DIR` | agent_results | 结果输出目录 |
| `LOOKBACK_DAYS` | 252 | 回溯周期（日） |
| `TEST_START_DATE` | 2024-01-01 | 回测开始日期 |
| `TEST_END_DATE` | 2025-12-31 | 回测结束日期 |

### 优化参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `MAX_FACTOR_COMPLEXITY` | 6 | 因子最大复杂度评分 |
| `POPULATION_SIZE` | 30 | 种群大小 |
| `GENERATIONS` | 5 | 遗传算法代数 |

## 常见问题

### 1. APIError: Invalid API key
**解决方案**: 检查 `.env` 文件中的 `OPENAI_API_KEY` 是否正确

### 2. Data Loading Error
**解决方案**: 确保数据文件存在于 `DATA_ROOT` 指定的目录中

### 3. LLM 响应为空或格式错误
**解决方案**: 检查网络连接，或尝试增加 `max_retries` 参数

### 4. 回测报错
**解决方案**: 
- 检查因子表达式是否语法正确
- 确保所有字段在数据中可用
- 尝试简化表达式复杂度

## 最佳实践

### 1. 从小规模开始
```python
# 先用 1-2 次迭代测试系统
agent.run(n_iterations=1)
```

### 2. 监控 API 成本
- 每个因子创意生成需要约 2-3 次 API 调用
- 1 次迭代约 10-15 次 API 调用
- 确保预算充足

### 3. 定期保存结果
- 系统自动保存每次迭代的结果
- 整理历史数据便于分析

### 4. 分析反思信息
```python
import json

# 读取会话历史
with open('agent_results/session_history_*.json') as f:
    history = json.load(f)

# 分析 LLM 的反思建议
for record in history:
    reflection = record['reflection']
    print(f"因子: {record['idea']['name']}")
    print(f"建议: {reflection['improvements']}")
```

## 扩展和定制

### 自定义挖掘方向

编辑 `llm_interface.py` 中的 `select_direction()` 方法:

```python
def select_direction(self) -> str:
    directions = ["量价背离", "动量", "反转", "您的自定义方向"]
    # ...
```

### 自定义表达式示例

编辑 `agent.py` 中的 `get_expression_examples()` 方法:

```python
def get_expression_examples(self) -> List[str]:
    return [
        "您的自定义表达式1",
        "您的自定义表达式2",
        # ...
    ]
```

### 集成其他 LLM

修改 `llm_interface.py` 的 `LLMInterface` 类以支持其他 LLM:

```python
class LLMInterface:
    def __init__(self, config=None, provider='openai'):
        # 支持切换 LLM 供应商
        if provider == 'openai':
            # OpenAI 实现
        elif provider == 'anthropic':
            # Anthropic Claude 实现
```

## 技术栈

- **LLM**: OpenAI GPT-4
- **数据处理**: Pandas, NumPy
- **因子计算**: Alpha101 operators
- **回测框架**: stratified_backtest
- **配置管理**: python-dotenv

## 许可证

本项目遵循 Freqtrade 项目的许可证。

## 联系与反馈

如有问题或建议，请提交 Issue 或 PR。

## 更新日志

### v0.1.0 (2026-02-26)
- 初始版本发布
- 支持 LLM 驱动的因子生成和迭代
- 集成 openai API
- 完整的工作流和回测框架
