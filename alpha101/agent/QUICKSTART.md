# 🚀 快速开始指南

一个基于 LLM 的自动化因子挖掘系统。在 5 分钟内开始你的第一个因子搜索！

## 1️⃣ 安装依赖 (1 分钟)

```bash
# 进入项目目录
cd e:\MatchAndProject\freqtrade

# 安装必需的包
pip install openai python-dotenv pandas numpy scikit-learn
```

## 2️⃣ 配置 API 密钥 (2 分钟)

### 方法 A: 创建 .env 文件（推荐）

```bash
# 进入 agent 目录
cd alpha101/agent

# 创建 .env 文件
copy .env.example .env

# 编辑 .env 文件，填入你的 OpenAI API 密钥
# 找到这一行并替换:
# OPENAI_API_KEY=your_openai_api_key_here
```

在你的编辑器中打开 `.env` 文件：

```ini
OPENAI_API_KEY=sk-xxx...  # 你的 OpenAI API 密钥
OPENAI_API_BASE=https://api.openai.com/v1
OPENAI_MODEL=gpt-4
OUTPUT_DIR=agent_results
```

### 方法 B: 使用环境变量

Windows PowerShell:
```powershell
$env:OPENAI_API_KEY = "sk-xxx..."
$env:OPENAI_API_BASE = "https://api.openai.com/v1"
```

Linux/macOS:
```bash
export OPENAI_API_KEY="sk-xxx..."
export OPENAI_API_BASE="https://api.openai.com/v1"
```

## 3️⃣ 运行 Agent (2 分钟)

### 方法 A: 快速启动（推荐）

```bash
cd alpha101/agent
python run_quick_start.py
```

按照交互提示操作，输入迭代次数后 Agent 会自动运行。

### 方法 B: 命令行运行

```bash
# 单次迭代
python -m alpha101.agent.agent --iterations 1

# 多次迭代
python -m alpha101.agent.agent --iterations 3
```

### 方法 C: Python API 调用

在你的 Python 脚本中：

```python
from alpha101.agent import AlphaAgent, get_config

# 初始化
config = get_config()
agent = AlphaAgent(config)

# 运行 1 次迭代
agent.run(n_iterations=1)
```

## 📊 查看结果

Agent 运行完成后，结果保存在 `agent_results/` 目录：

```
agent_results/
├── all_factors_20260226_120530.csv      # 所有因子结果
├── session_history_20260226_120530.json # 完整会话历史
└── config_20260226_120530.json          # 配置信息快照
```

### 查看因子结果 (CSV)

```python
import pandas as pd

df = pd.read_csv('agent_results/all_factors_*.csv')

# 查看性能最好的因子
print(df.nlargest(5, 'sharpe_ratio')[['name', 'sharpe_ratio', 'returns', 'expression']])
```

### 查看完整历史 (JSON)

```python
import json

with open('agent_results/session_history_*.json') as f:
    history = json.load(f)

# 查看每个迭代的反思
for record in history:
    print(f"因子: {record['idea']['name']}")
    print(f"评级: {record['reflection']['rating']}")
    print(f"建议: {record['reflection']['improvements']}\n")
```

## 🎯 工作流概览

```
1. 选择方向    → LLM 随机选择 (量价背离/动量/反转/...)
2. 生成创意    → LLM 基于方向生成 3 个因子创意
3. 构造表达式  → LLM 将创意转换为可执行表达式
4. 回测评估    → 系统对因子进行 stratified backtest
5. 反思改进    → LLM 分析结果并提出改进建议
6. 迭代优化    → LLM 生成改进后的表达式
7. 再次评估    → 系统评估改进后的因子
```

## ⚙️ 调整参数

编辑 `.env` 文件来控制 Agent 的行为：

```ini
# 迭代参数
OPENAI_MODEL=gpt-4              # 使用的 LLM 模型

# 数据参数
LOOKBACK_DAYS=252              # 数据回溯周期（日）
TEST_START_DATE=2024-01-01      # 回测开始日期
TEST_END_DATE=2025-12-31        # 回测结束日期

# 搜索参数
MAX_FACTOR_COMPLEXITY=6         # 因子最大复杂度
```

## 📝 表达式示例

Agent 会使用这些类型的表达式：

### 动量因子
```python
ts_mean(close, 10) / ts_mean(close, 30) - 1
ts_rank(returns, 10) / 10 - 0.5
```

### 反转因子
```python
(-1) * ts_rank(returns, 5) / 5
(close - ts_min(close, 20)) / (ts_max(close, 20) - ts_min(close, 20))
```

### 量价背离
```python
ts_corr(returns, volume, 20)
abs(ts_rank(returns, 10) - ts_rank(volume, 10)) / 10
```

## ✅ 成功检查清单

在运行前，确保：

- [ ] 安装了所需的 Python 包
- [ ] 设置了 OpenAI API 密钥
- [ ] .env 文件在 `alpha101/agent/` 目录中
- [ ] 数据文件存在于 `alpha101/futures_ml/data/` 目录
- [ ] OpenAI API 密钥有效且配额充足

## 🐛 常见问题

### Q: "Invalid API key" 错误
**A**: 检查你的 OpenAI API 密钥是否正确，并确保密钥有权访问 gpt-4 模型

### Q: 数据加载失败
**A**: 确保数据文件存在于正确的位置，检查 `DATA_ROOT` 配置

### Q: LLM 响应慢或超时
**A**: 这是正常的。首次调用可能需要 30-60 秒。确保网络连接稳定。

### Q: 需要多少 API 成本？
**A**: 视具体情况而定：
- 1 次迭代 ≈ 15-20 个 API 调用
- 使用 gpt-4 时成本较高，gpt-3.5-turbo 更经济

## 📚 进阶用法

### 自定义挖掘方向

编辑 `llm_interface.py`:

```python
def select_direction(self) -> str:
    directions = ["量价背离", "动量", "反转", "你的自定义方向"]
    # ... 其他代码
```

### 添加自定义表达式示例

编辑 `agent.py`:

```python
def get_expression_examples(self) -> List[str]:
    return [
        "你的自定义表达式1",
        "你的自定义表达式2",
        # ...
    ]
```

### 批量运行多个配置

```python
from alpha101.agent import AlphaAgent, AgentConfig

# 测试不同的参数组合
configs = [
    {'population_size': 20, 'generations': 3},
    {'population_size': 30, 'generations': 5},
    {'population_size': 50, 'generations': 10},
]

for cfg in configs:
    config = AgentConfig()
    agent = AlphaAgent(config)
    agent.run(n_iterations=1)
    print(f"完成配置: {cfg}")
```

## 🔗 更多资源

- [完整文档](README.md)
- [配置参考](.env.example)
- [API 文档](agent.py)

## 💡 提示

1. **从小开始**: 先运行 1-2 个迭代以测试设置
2. **监控成本**: 每个迭代会产生 OpenAI API 费用
3. **保存结果**: 所有结果自动保存到 `agent_results/` 目录
4. **分析反思**: JSON 文件包含 LLM 的完整分析和建议

## 🎉 成功！

恭喜！现在你已经了解了如何使用 Alpha 因子自动挖掘 Agent。

祝你找到伟大的因子！🚀
