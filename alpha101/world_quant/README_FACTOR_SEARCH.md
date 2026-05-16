# 因子搜索系统使用文档

## 系统概述

这是一个完整的因子批量构造和回测系统，支持多种搜索策略，自动评估因子性能并保存优质因子。

## 核心组件

### 1. factor_generator.py - 因子生成器
生成各种因子表达式的核心模块。

**功能：**
- 随机生成因子表达式（递归构建）
- 基于模板生成因子变体
- 网格搜索系统组合
- 因子变异和交叉（用于遗传算法）

**主要类：**
- `FactorGenerator`: 因子表达式生成器
- `FactorLibrary`: 预定义的因子模板库

## 给其他 AI 的因子构造规则（可直接当 Prompt）

你现在要生成**可被本系统解析和执行**的因子表达式，请严格遵循以下规则：

### 1) 表达式基本语法
- 允许字段名：`open, high, low, close, volume, returns, vwap, cap, market_return, funding`
- 允许二元运算：`+ - * /`
- 允许一元运算：`log(x), abs(x), -x`
- 函数调用采用 `func(arg1, arg2, ...)` 形式
- 推荐输出单行表达式字符串，不要输出解释文字

### 2) 算子签名与参数范围

**单变量时序算子（2参数）**：`op(series, window)`，其中 `window` 为正整数
- `ts_mean`
- `ts_rank`
- `ts_min`
- `ts_max`
- `ts_std_dev`
- `ts_arg_max`
- `ts_arg_min`
- `ts_sum`
- `ts_product`
- `ts_skewness`
- `ts_kurtosis`
- `ts_decay_linear`
- `ts_drawdown`
- `ts_pos`

**双变量时序算子（3参数）**：`op(series1, series2, window)`，其中 `window` 为正整数
- `ts_corr`
- `ts_covariance`
- `ts_alpha`
- `ts_r2`
- `ts_beta`

**延迟/差分算子**
- `ts_delay(series, lag)`，其中 `lag` 为正整数
- `ts_delta(series, lag)`，其中 `lag` 为正整数

**截面算子（1参数）**
- `rank(series)`
- `scale(series)`

### 3) 输入类型约束（重要）
- 所有 `ts_*`、`rank`、`scale`、`log`、`abs` 的第一个输入都应是**序列型表达式**（通常由字段或其运算构成）
- `window/lag` 必须是正整数
- 不要只输出单字段（如 `close`），应至少包含一个算子

### 4) 禁止/不推荐模式
- 避免重复嵌套：`ts_mean(ts_mean(...))`、`ts_rank(ts_rank(...))`、`rank(rank(...))`、`scale(scale(...))`、`abs(abs(...))`
- 避免：`ts_std_dev(ts_std_dev(...))`
- 避免：`ts_corr(ts_corr(...), ..., w)`、`ts_covariance(ts_covariance(...), ..., w)`
- 避免无意义组合：`ts_arg_max/ts_arg_min` 后立即再做 `ts_mean` 的结构

### 5) 复杂度建议
- 推荐最大嵌套深度不超过 `3`
- 推荐总算子数不超过 `6`
- 表达式长度适中，优先可解释性

### 6) 示例（合法且实用）
- `rank(ts_delta(close, 3))`
- `(close - ts_mean(close, 20)) / ts_std_dev(close, 20)`
- `ts_corr(returns, volume, 20)`
- `rank(volume / ts_mean(volume, 20)) * rank(-ts_delta(close, 7))`
- `scale(ts_decay_linear(ts_delta(vwap, 2), 14))`
- `ts_beta(returns, market_return, 21)`

### 7) 可复制的最小 Prompt 模板
```text
请生成根据以下规则生成因子表达式，且严格使用以下字段与算子：
字段: open, high, low, close, volume, returns, vwap, cap, market_return, funding
算子: + - * /, log, abs, rank, scale, ts_mean, ts_rank, ts_min, ts_max, ts_std_dev, ts_arg_max, ts_arg_min, ts_sum, ts_product, ts_skewness, ts_kurtosis, ts_decay_linear, ts_drawdown, ts_pos, ts_corr, ts_covariance, ts_alpha, ts_r2, ts_beta, ts_delay, ts_delta
约束:
1) ts_单变量算子: op(series, window)
2) ts_双变量算子: op(series1, series2, window)
3) ts_delay/ts_delta: op(series, lag)
4) rank/scale/log/abs 输入必须是序列表达式
5) window/lag 必须为正整数

示例
- `rank(ts_delta(close, 3))`
- `(close - ts_mean(close, 20)) / ts_std_dev(close, 20)`
- `ts_corr(returns, volume, 20)`
- `rank(volume / ts_mean(volume, 20)) * rank(-ts_delta(close, 7))`
- `scale(ts_decay_linear(ts_delta(vwap, 2), 14))`
- `ts_beta(returns, market_return, 21)`

```

### 2. factor_search.py - 因子搜索引擎
执行因子搜索和回测的主程序。

**支持的搜索策略：**

#### 随机搜索 (Random Search)
- 完全随机生成因子表达式
- 适合快速探索大量可能性
- 推荐数量：100-1000个因子

#### 模板搜索 (Template Search)
- 基于经典Alpha101模式
- 系统化地替换字段、窗口、延迟参数
- 质量较高但覆盖范围有限

#### 网格搜索 (Grid Search)
- 系统地组合算子和参数
- 覆盖常见的因子构造模式
- 计算成本较高但全面

#### 遗传算法 (Genetic Algorithm)
- 模拟自然选择过程
- 通过变异和交叉产生新因子
- 能够发现复杂的非线性组合

### 3. factor_analyzer.py - 结果分析工具
分析和可视化搜索结果。

**功能：**
- 加载所有搜索结果
- 筛选优质因子
- 性能分布可视化
- 策略对比分析
- 导出Top因子

## 使用方法

### 快速开始

#### 1. 随机搜索（推荐初次使用）
```bash
# 生成100个随机因子并回测
python alpha101/world_quant/factor_search.py --strategy random --n-factors 1000
```

#### 2. 模板搜索
```bash
# 基于Alpha101模板生成因子
python alpha101/world_quant/factor_search.py --strategy template
```

#### 3. 网格搜索
```bash
# 系统化参数组合
python alpha101/world_quant/factor_search.py --strategy grid
```

#### 4. 遗传算法
```bash
# 运行遗传算法，种群30，迭代5代
python alpha101/world_quant/factor_search.py --strategy genetic --population 2000 --generations 25 --n-factors 1000
```

#### 5. 运行所有策略
```bash
# 依次执行随机、模板、网格搜索
python alpha101/world_quant/factor_search.py --strategy all --n-factors 100
```

### 命令行参数

```bash
--strategy      搜索策略 [random|template|grid|genetic|all]
--n-factors     随机搜索因子数量 (default: 100)
--population    遗传算法种群大小 (default: 30)
--generations   遗传算法迭代代数 (default: 5)
--output-dir    结果输出目录 (default: factor_search_results)
```

### 结果分析

搜索完成后，分析结果：

```bash
# 分析所有搜索结果
python alpha101/world_quant/factor_analyzer.py --results-dir factor_search_results

# 设置筛选条件
python alpha101/world_quant/factor_analyzer.py \
    --min-sharpe 0.8 \
    --min-return 0.02 \
    --top-n 50
```

## 输出结果

### 搜索过程输出
- `random_batch_*.csv/json`: 随机搜索批次结果
- `template_batch_*.csv/json`: 模板搜索批次结果  
- `grid_batch_*.csv/json`: 网格搜索批次结果
- `genetic_gen_*.csv/json`: 遗传算法每代结果

### 最终汇总
- `summary_*.csv`: 所有因子汇总
- `top_factors_*.csv`: Top因子列表
- `top_factors_final.csv`: 最终导出的优质因子

### 可视化结果
- `performance_distribution.png`: 性能分布图
- `strategy_comparison.png`: 策略对比图

## 性能指标说明

每个因子会被评估以下指标：

- `sharpe_ratio`: Sharpe比率（收益/风险）
- `mean_return`: 平均收益率
- `ic_mean`: 信息系数均值
- `ic_std`: 信息系数标准差
- `top_minus_bottom`: 顶部分位数与底部分位数收益差
- `win_rate`: 胜率
- `nan_ratio`: 缺失值比例

## 最佳实践

### 1. 分阶段搜索
```bash
# 第一阶段：快速随机搜索（exploration）
python alpha101/world_quant/factor_search.py --strategy random --n-factors 500

# 第二阶段：模板精细化（exploitation）
python alpha101/world_quant/factor_search.py --strategy template

# 第三阶段：遗传算法优化（optimization）
python alpha101/world_quant/factor_search.py --strategy genetic --population 50 --generations 10
```

### 2. 筛选标准建议

**保守标准（高质量）：**
- Sharpe Ratio >= 1.0
- Mean Return >= 0.03
- NaN Ratio <= 0.2

**宽松标准（探索性）：**
- Sharpe Ratio >= 0.5
- Mean Return >= 0.01
- NaN Ratio <= 0.4

### 3. 批量处理
为避免内存溢出，建议：
- 每批处理100-200个因子
- 及时保存中间结果
- 使用`batch_size`参数控制保存频率

### 4. 并行化（高级）
```python
# 在代码中可以修改为多进程
from multiprocessing import Pool

# 在 factor_search.py 中添加
with Pool(processes=4) as pool:
    results = pool.map(evaluate_factor, factor_list)
```

## 自定义扩展

### 添加新的算子
在 `factor_generator.py` 中修改：

```python
self.ts_operators = {
    'ts_mean': [3, 5, 7, 10, 14, 20, 30, 60],
    # 添加新算子
    'ts_skew': [10, 20, 30],
    'ts_kurt': [10, 20, 30],
}
```

### 添加自定义模板
在 `FactorLibrary` 中添加：

```python
CUSTOM_PATTERNS = [
    "rank({{field}} / ts_mean(volume, {{window}}))",
    "ts_delta(log({{field}}), {{ts_delay}})",
]
```

### 修改评估指标
在 `factor_search.py` 的 `evaluate_factor` 方法中修改。

## 故障排除

### 常见问题

**Q: 为什么很多因子评估失败？**
A: 可能原因：
1. 除零错误（已在生成器中处理）
2. 参数窗口过大导致数据不足
3. 表达式语法错误

**Q: 搜索速度太慢？**
A: 优化建议：
1. 减少因子数量
2. 使用更短的回测周期
3. 实施并行化

**Q: 内存不足？**
A: 解决方案：
1. 减少batch_size
2. 及时释放中间结果
3. 分多次运行

## 示例工作流

```bash
# Step 1: 运行快速探索
python alpha101/world_quant/factor_search.py --strategy random --n-factors 200

# Step 2: 分析初步结果
python alpha101/world_quant/factor_analyzer.py

# Step 3: 基于好的结果，运行遗传算法优化
python alpha101/world_quant/factor_search.py --strategy genetic --population 30 --generations 10

# Step 4: 导出Top 50因子
python alpha101/world_quant/factor_analyzer.py --top-n 50 --min-sharpe 0.8

# Step 5: 在实际策略中使用导出的因子
# 从 top_factors_final.csv 中提取表达式
```

## 进阶技巧

### 1. 因子组合
将多个优质因子线性组合：
```python
# 在评估top因子后
top_expr1 = "ts_rank(close, 10)"
top_expr2 = "rank(volume)"
combined = f"0.6 * ({top_expr1}) + 0.4 * ({top_expr2})"
```

### 2. 因子去相关
```python
# 计算因子间相关性
from alpha101.futures_ml.alpha_sharpe import analyze_factor_correlation

corr_result = analyze_factor_correlation(df, top_alpha_list, threshold=0.7)
# 移除高相关因子
```

### 3. 集成学习
使用多个因子构建机器学习模型。

## 总结

这个因子搜索系统提供了：
✅ 4种搜索策略（随机、模板、网格、遗传）
✅ 自动回测和性能评估
✅ 批量处理和结果保存
✅ 可视化分析工具
✅ Top因子导出

通过合理使用这些工具，可以高效地发现和验证优质交易因子。
