# Alpha Agent - 故障排查和常见问题

## 📋 目录

1. [配置问题](#配置问题)
2. [API 问题](#api-问题)
3. [数据问题](#数据问题)
4. [运行问题](#运行问题)
5. [性能问题](#性能问题)
6. [获取帮助](#获取帮助)

---

## 配置问题

### ❓ 问题 1: "OPENAI_API_KEY not set in .env file"

**症状**:
```
Error: OPENAI_API_KEY not set in .env file
```

**原因**: `.env` 文件不存在或 `OPENAI_API_KEY` 未设置

**解决方案**:

```bash
# 1. 检查 .env 文件是否存在
ls alpha101/agent/.env

# 2. 如果不存在，从示例创建
cd alpha101/agent
cp .env.example .env

# 3. 编辑 .env 文件，添加你的 API 密钥
# 使用你喜欢的编辑器打开 .env
# 找到这行: OPENAI_API_KEY=your_openai_api_key_here
# 替换为: OPENAI_API_KEY=sk-xxx... (你的实际密钥)
```

### ❓ 问题 2: "Invalid API key"

**症状**:
```
openai.error.AuthenticationError: Invalid API key provided
```

**原因**: API 密钥无效或格式错误

**解决方案**:

```bash
# 1. 验证 API 密钥格式（应该以 sk- 开头）
echo $OPENAI_API_KEY  # Linux/macOS
echo %OPENAI_API_KEY%  # Windows CMD
$env:OPENAI_API_KEY    # Windows PowerShell

# 2. 从 OpenAI 网站重新获取密钥
# https://platform.openai.com/account/api-keys

# 3. 重新设置 .env 文件
OPENAI_API_KEY=sk-your-new-key-here
```

### ❓ 问题 3: "openai library not installed"

**症状**:
```
ImportError: openai library is required
```

**解决方案**:

```bash
# 安装 openai 库
pip install openai

# 或安装所有依赖
pip install -r alpha101/agent/requirements.txt

# 验证安装
python -c "import openai; print(openai.__version__)"
```

### ❓ 问题 4: "output directory cannot be created"

**症状**:
```
Error: Cannot create output directory: Permission denied
```

**解决方案**:

```bash
# 检查权限
ls -la agent_results/

# 授予写权限（Linux/macOS）
chmod 755 agent_results/

# 或使用其他输出目录
# 编辑 .env 中的 OUTPUT_DIR
OUTPUT_DIR=./my_custom_results
```

---

## API 问题

### ❓ 问题 5: "Rate limit exceeded"

**症状**:
```
openai.error.RateLimitError: Rate limit exceeded
```

**原因**: 调用 API 过于频繁

**解决方案**:

```python
# Agent 会自动重试，但可以增加重试延迟
# 编辑 llm_interface.py

class LLMInterface:
    def __init__(self, config=None):
        # ...
        self.max_retries = 5  # 增加重试次数
        self.retry_delay = 2  # 增加延迟（秒）
```

或者：

```bash
# 减少并发调用，运行较少的迭代
python -m alpha101.agent.agent --iterations 1
```

### ❓ 问题 6: "API quota exceeded"

**症状**:
```
openai.error.OpenAIError: You exceeded your current quota
```

**原因**: 账户已达到配额限制

**解决方案**:

1. 检查 OpenAI 账户: https://platform.openai.com/account/billing/overview
2. 升级计划或申请更高配额
3. 使用更便宜的模型:
   ```ini
   # 在 .env 中
   OPENAI_MODEL=gpt-3.5-turbo  # 更经济的选择
   ```

### ❓ 问题 7: "Connection timeout"

**症状**:
```
openai.error.APIConnectionError: Error communicating with OpenAI
```

**原因**: 网络连接问题或 OpenAI 服务不可用

**解决方案**:

```bash
# 1. 检查网络连接
ping api.openai.com

# 2. 检查 OPENAI_API_BASE 是否正确
# 应该是: https://api.openai.com/v1

# 3. 尝试使用代理（如果需要）
# 在 Python 中设置环境变量
import os
os.environ['HTTPS_PROXY'] = 'http://proxy.example.com:8080'

# 4. 增加超时时间
# 编辑 config.py，在 OpenAI 初始化时添加:
self.client = OpenAI(
    api_key=self.config.openai_api_key,
    base_url=self.config.openai_api_base,
    timeout=60  # 增加到 60 秒
)
```

### ❓ 问题 8: "Model not available"

**症状**:
```
openai.error.InvalidRequestError: The model `gpt-4` does not exist
```

**原因**: 账户无权使用选定的模型

**解决方案**:

```bash
# 检查可用的模型
# 访问: https://platform.openai.com/account/billing/limits

# 使用可用的模型
echo "OPENAI_MODEL=gpt-3.5-turbo" >> alpha101/agent/.env

# 或指定其他可用模型
OPENAI_MODEL=gpt-3.5-turbo-16k
```

---

## 数据问题

### ❓ 问题 9: "Data loading failed"

**症状**:
```
Error: Cannot load data from ...
FileNotFoundError: [Errno 2] No such file or directory: 'data/...'
```

**原因**: 数据文件不存在

**解决方案**:

```bash
# 1. 检查数据目录
ls -la alpha101/futures_ml/data/

# 2. 确保 DATA_ROOT 配置正确
# 编辑 .env
DATA_ROOT=../futures_ml/data

# 3. 从原始数据构建数据库
python alpha101/futures_ml/data.py  # 或相关数据构建脚本

# 4. 检查数据路径（相对于 agent.py）
# 相对路径应该从 alpha101/agent/ 目录计算
```

### ❓ 问题 10: "Data shape mismatch"

**症状**:
```
ValueError: Shape mismatch in data concatenation
```

**原因**: 因子数据维度与市场数据不匹配

**解决方案**:

```python
# 在 agent.py 中检查数据维度
print(f"Wide data shape: {agent.wide_data.shape}")
print(f"Wide data columns: {agent.wide_data.columns[:5]}")

# 确保背景数据有正确的多层索引
# 列应该是: (symbol_pair, field)
```

### ❓ 问题 11: "All NaN result"

**症状**:
```
evaluating factor returned all NaN values
```

**原因**: 因子表达式计算结果全为 NaN

**解决方案**:

```python
# 1. 检查表达式语法
expr = "ts_mean(close, 10) / ts_mean(close, 30) - 1"
# 确保所有字段名（open, high, low, close, volume 等）都有效

# 2. 避免除以零
# 差：close / close_mean  可能导致 NaN
# 好：close / (close_mean + 1e-6)  添加小常数避免除零

# 3. 检查参数范围
# 确保滑动窗口大小小于数据长度
# 避免：ts_mean(close, 10000)  当数据不足 10000 天时

# 4. 使用更简单的表达式测试
expr = "close"  # 最简单的表达式
expr = "ts_mean(close, 20)"  # 简单的移动平均
```

---

## 运行问题

### ❓ 问题 12: Agent 无响应或冻结

**症状**:
```
脚本似乎卡住，没有输出
```

**原因**: 
- 等待 LLM API 响应
- 网络连接缓慢
- 回测计算耗时

**解决方案**:

```bash
# 1. 增加等待时间，不要中断
# 首次调用可能需要 30-60 秒

# 2. 检查网络连接
ping -c 3 api.openai.com  # Linux/macOS
ping -n 3 api.openai.com  # Windows

# 3. 在较小的数据集上运行测试
# 编辑 .env 文件减少数据范围
LOOKBACK_DAYS=60  # 减少到 60 天

# 4. 运行时显示详细日志
# 编辑 agent.py 添加调试输出
import logging
logging.basicConfig(level=logging.DEBUG)
```

### ❓ 问题 13: "Expression parsing error"

**症状**:
```
SyntaxError or NameError in evaluate_factor
```

**原因**: LLM 生成的表达式有语法错误

**解决方案**:

```python
# 1. 检查 LLM 返回的表达式
print(factor_expr)
# 确保表达式是 Python 有效的代码

# 2. 增加 LLM 指令的清晰度
# 编辑 llm_interface.py 中的 prompt 提示词

# 3. 手动测试表达式
from alpha101.world_quant.fastengine import FastExpressionEngine
# engine.evaluate("your_expression_here")

# 4. 使用表达式验证
from alpha101.world_quant.factor_generator import FactorGenerator
generator = FactorGenerator()
is_valid, reason = generator.is_semantically_valid(expr)
print(f"Valid: {is_valid}, Reason: {reason}")
```

### ❓ 问题 14: "Memory error"

**症状**:
```
MemoryError: Unable to allocate ...
```

**原因**: 处理大量数据时内存不足

**解决方案**:

```bash
# 1. 减少数据范围
# 编辑 .env
LOOKBACK_DAYS=120  # 减少数据量

# 2. 减少迭代次数
python -m alpha101.agent.agent --iterations 1

# 3. 增加系统内存
# 或使用更小的 batch size（如果适用）

# 4. 在 Linux 上增加虚拟内存
sudo swapon -s  # 检查当前交换分区
```

---

## 性能问题

### ❓ 问题 15: Agent 运行很慢

**症状**:
```
单次迭代需要超过 5 分钟
```

**原因**:
- API 响应慢
- 回测计算耗时
- 数据处理瓶颈

**优化方案**:

```bash
# 1. 使用更快的模型
OPENAI_MODEL=gpt-3.5-turbo  # 比 gpt-4 更快

# 2. 减少数据量
LOOKBACK_DAYS=120  # 而不是 252

# 3. 简化表达式（在 LLM 提示词中强调）
# 编辑 llm_interface.py 的 prompt

# 4. 并行化回测
# 该功能已在 stratified_backtest 中支持

# 5. 使用 GPU（如果有）
# 对于大规模计算
```

### ❓ 问题 16: 因子性能不好

**症状**:
```
所有因子的 Sharpe 比率都是负数或很低
```

**原因**:
- 市场条件不支持该策略
- 表达式设计不当
- 参数选择不最优

**改进方案**:

```python
# 1. 分析问题因子
import pandas as pd
df = pd.read_csv('agent_results/all_factors_*.csv')
print(df[df['sharpe_ratio'] < 0][['expression', 'sharpe_ratio']])

# 2. 改进 LLM 提示词
# 编辑 llm_interface.py 提供更好的指导

# 3. 增加迭代次数
agent.run(n_iterations=5)  # 探索更多因子

# 4. 调整参数范围
# 增加 MAX_FACTOR_COMPLEXITY 或其他参数

# 5. 分析市场阶段
# 检查回测期间的市场特征
```

---

## 获取帮助

### 📞 支持渠道

1. **检查文档**
   - [完整 README](README.md)
   - [快速开始指南](QUICKSTART.md)
   - [高级示例](examples.py)

2. **查看日志**
   ```python
   # 启用详细日志
   import logging
   logging.basicConfig(level=logging.DEBUG, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
   ```

3. **测试环境**
   ```bash
   # 验证安装
   python -c "from alpha101.agent import AlphaAgent; print('✓ Installation OK')"
   
   # 检查 API 连接
   python -c "from config import get_config; c = get_config(); print(f'✓ Config: {c}')"
   ```

4. **调试 LLM 响应**
   ```python
   # 在 llm_interface.py 中添加日志
   print(f"Prompt: {prompt}")
   print(f"Response: {response}")
   ```

5. **保存问题信息**
   ```bash
   # 收集系统信息以便诊断
   python --version
   pip list | grep openai
   pip list | grep pandas
   ```

### 🔍 常用调试步骤

```python
# 1. 测试配置
from alpha101.agent.config import get_config
config = get_config()
print(config)

# 2. 测试 LLM 连接
from alpha101.agent.llm_interface import LLMInterface
llm = LLMInterface()
direction = llm.select_direction()  # 测试 API 调用

# 3. 测试数据加载
from alpha101.agent.agent import AlphaAgent
agent = AlphaAgent()
print(f"Data shape: {agent.wide_data.shape}")

# 4. 测试因子评估
factor = agent.evaluate_factor("test", "close")
print(factor)

# 5. 逐步运行工作流
context = agent.get_context_info()
ideas = llm.generate_factor_ideas("动量", context)
# ... 继续测试每一步
```

### 💡 最佳实践

1. **先用最小配置测试**
   ```bash
   --iterations 1 --population 10 --generations 1
   ```

2. **逐步增加复杂性**
   ```bash
   # 第一次：验证设置
   --iterations 1
   # 第二次：尝试多个想法
   --iterations 2
   # 第三次：完整搜索
   --iterations 5
   ```

3. **定期保存结果**
   ```bash
   # 所有结果自动保存在:
   agent_results/all_factors_*.csv
   agent_results/session_history_*.json
   ```

4. **监控 API 成本**
   - 记录每次运行的 API 调用次数
   - 估计月度成本
   - 选择合适的模型和迭代次数

---

## 获取更多帮助

如果问题未能通过上述步骤解决，请：

1. 检查 OpenAI 状态页面: https://status.openai.com/
2. 查阅 OpenAI 文档: https://platform.openai.com/docs/
3. 查看 Freqtrade 社区: https://discord.gg/freqtrade
4. 提交 GitHub Issue（包含错误信息和环境信息）
