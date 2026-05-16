"""
Advanced Examples for Alpha Agent
演示如何进行高级定制和复杂使用场景
"""
import sys
from pathlib import Path
import json
import pandas as pd

repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))

from alpha101.agent import AlphaAgent, AgentConfig, get_config
from alpha101.agent.llm_interface import LLMInterface


# ============================================================================
# 示例 1: 基础使用 - 运行单一方向的因子搜索
# ============================================================================
def example_1_basic_usage():
    """
    基础示例: 运行一个简单的因子搜索
    """
    print("\n" + "="*80)
    print("示例 1: 基础使用")
    print("="*80)
    
    # 初始化 Agent
    config = get_config()
    agent = AlphaAgent(config)
    
    # 运行一个迭代
    agent.run(n_iterations=1)
    
    # 分析结果
    results_files = sorted(config.output_dir.glob('all_factors_*.csv'))
    if results_files:
        df = pd.read_csv(results_files[-1])
        print(f"\n找到 {len(df)} 个因子")
        print(f"最佳 Sharpe 比率: {df['sharpe_ratio'].max():.4f}")


# ============================================================================
# 示例 2: 自定义配置和参数
# ============================================================================
def example_2_custom_config():
    """
    高级示例: 使用自定义配置
    """
    print("\n" + "="*80)
    print("示例 2: 自定义配置")
    print("="*80)
    
    # 创建自定义配置
    import os
    os.environ['OPENAI_MODEL'] = 'gpt-3.5-turbo'  # 使用更便宜的模型
    os.environ['MAX_FACTOR_COMPLEXITY'] = '5'
    
    config = AgentConfig()
    print(f"模型: {config.openai_model}")
    print(f"最大复杂度: {config.max_factor_complexity}")
    
    agent = AlphaAgent(config)
    agent.run(n_iterations=1)


# ============================================================================
# 示例 3: 分析和可视化结果
# ============================================================================
def example_3_analyze_results():
    """
    分析 Agent 的输出结果
    """
    print("\n" + "="*80)
    print("示例 3: 分析结果")
    print("="*80)
    
    config = get_config()
    results_dir = config.output_dir / "agent_results"
    
    # 读取最新的因子结果
    factors_files = sorted(results_dir.glob('all_factors_*.csv'))
    if not factors_files:
        print("未找到因子结果文件")
        return
    
    df = pd.read_csv(factors_files[-1])
    
    # 统计分析
    print(f"\n总因子数: {len(df)}")
    print(f"成功率: {(df['fitness'] > 0).sum() / len(df) * 100:.1f}%")
    print(f"\nSharpe 比率统计:")
    print(f"  平均: {df['sharpe_ratio'].mean():.4f}")
    print(f"  中位数: {df['sharpe_ratio'].median():.4f}")
    print(f"  最高: {df['sharpe_ratio'].max():.4f}")
    print(f"  最低: {df['sharpe_ratio'].min():.4f}")
    
    # print(f"\n年化收益统计:")
    # print(f"  平均: {df['returns'].mean():.4f}")
    # print(f"  最高: {df['returns'].max():.4f}")
    # print(f"  最低: {df['returns'].min():.4f}")
    
    # 按方向分析
    print(f"\n按方向分析:")
    direction_groups = df.groupby('direction').agg({
        'sharpe_ratio': ['mean', 'max'],
        'returns': ['mean', 'max'],
        'name': 'count'
    })
    print(direction_groups)
    
    # Top 因子
    print(f"\nTop 5 因子 (按 Sharpe 比率):")
    top_5 = df.nlargest(5, 'sharpe_ratio')[['name', 'direction', 'sharpe_ratio', 'returns']]
    print(top_5.to_string(index=False))


# ============================================================================
# 示例 4: 会话历史分析
# ============================================================================
def example_4_session_history():
    """
    分析完整的会话历史和 LLM 反思
    """
    print("\n" + "="*80)
    print("示例 4: 会话历史分析")
    print("="*80)
    
    config = get_config()
    results_dir = config.output_dir / "agent_results"
    
    # 读取会话历史
    history_files = sorted(results_dir.glob('session_history_*.json'))
    if not history_files:
        print("未找到会话历史文件")
        return
    
    with open(history_files[-1], 'r', encoding='utf-8') as f:
        history = json.load(f)
    
    print(f"\n总记录数: {len(history)}")
    
    # 分析每个记录
    for i, record in enumerate(history, 1):
        idea = record['idea']
        reflection = record['reflection']
        original = record['original_factor']
        improved = record['improved_factor']
        
        print(f"\n记录 {i}: {idea['name']}")
        print(f"  方向: {idea['hypothesis']}")
        print(f"  原始 Sharpe: {original['sharpe_ratio']:.4f}")
        if improved:
            print(f"  改进 Sharpe: {improved['sharpe_ratio']:.4f}")
        print(f"  LLM 评级: {reflection['rating']}")
        print(f"  建议: {', '.join(reflection['improvements'][:2])}...")


# ============================================================================
# 示例 5: 批量运行多个配置
# ============================================================================
def example_5_batch_experiments():
    """
    运行多个实验来比较不同配置的效果
    """
    print("\n" + "="*80)
    print("示例 5: 批量实验")
    print("="*80)
    
    # 定义实验参数
    experiments = [
        {'name': 'gpt-3.5-精简', 'model': 'gpt-3.5-turbo', 'iterations': 1},
        {'name': 'gpt-4-标准', 'model': 'gpt-4', 'iterations': 2},
    ]
    
    results = []
    
    for exp in experiments:
        print(f"\n运行实验: {exp['name']}")
        print(f"  模型: {exp['model']}")
        print(f"  迭代: {exp['iterations']}")
        
        # 设置环境变量
        import os
        os.environ['OPENAI_MODEL'] = exp['model']
        
        config = AgentConfig()
        agent = AlphaAgent(config)
        agent.run(n_iterations=exp['iterations'])
        
        # 收集结果
        factors_files = sorted(config.output_dir.glob('**/all_factors_*.csv'))
        if factors_files:
            df = pd.read_csv(factors_files[-1])
            avg_sharpe = df[df['sharpe_ratio'] > 0]['sharpe_ratio'].mean()
            results.append({
                'experiment': exp['name'],
                'model': exp['model'],
                'avg_sharpe': avg_sharpe,
                'factor_count': len(df)
            })
    
    # 比较结果
    print(f"\n{'='*60}")
    print("实验结果对比:")
    print(f"{'='*60}")
    for result in results:
        print(f"{result['experiment']:<15} Avg Sharpe: {result['avg_sharpe']:.4f}")


# ============================================================================
# 示例 6: 自定义 LLM 行为
# ============================================================================
def example_6_custom_llm():
    """
    演示如何自定义 LLM 的行为
    """
    print("\n" + "="*80)
    print("示例 6: 自定义 LLM 行为")
    print("="*80)
    
    # 创建一个自定义的 LLM 接口
    class CustomLLMInterface(LLMInterface):
        def select_direction(self) -> str:
            """选择特定的方向"""
            # 总是选择动量方向
            return "动量"
        
        def generate_factor_ideas(self, direction: str, context: dict):
            """重写因子生成逻辑"""
            print(f"使用自定义方法为方向 '{direction}' 生成因子")
            
            # 调用父类方法
            return super().generate_factor_ideas(direction, context)
    
    # 使用自定义 LLM
    config = get_config()
    
    # 替换 Agent 中的 LLM 接口
    agent = AlphaAgent(config)
    agent.llm = CustomLLMInterface(config)
    
    print("使用自定义 LLM 接口运行...")
    # agent.run(n_iterations=1)  # 注释掉以避免实际 API 调用


# ============================================================================
# 示例 7: 因子性能对比
# ============================================================================
def example_7_factor_comparison():
    """
    对不同挖掘方向的因子进行性能比较
    """
    print("\n" + "="*80)
    print("示例 7: 因子性能对比")
    print("="*80)
    
    config = get_config()
    results_dir = config.output_dir / "agent_results"
    
    factors_files = sorted(results_dir.glob('all_factors_*.csv'))
    if not factors_files:
        print("未找到因子文件")
        return
    
    df = pd.read_csv(factors_files[-1])
    
    # 按方向分组及统计
    direction_stats = df.groupby('direction').agg({
        'sharpe_ratio': ['count', 'mean', 'max', 'min', 'std'],
        'returns': ['mean', 'max'],
        'fitness': ['mean', 'max']
    }).round(4)
    
    print("\n按方向的性能统计:")
    print(direction_stats)
    
    # 可视化对比 (如果安装了 matplotlib)
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Sharpe 比率分布
        df.boxplot(column='sharpe_ratio', by='direction', ax=axes[0])
        axes[0].set_title('Sharpe 比率分布')
        axes[0].set_ylabel('Sharpe 比率')
        
        # 收益分布
        df.boxplot(column='returns', by='direction', ax=axes[1])
        axes[1].set_title('年化收益分布')
        axes[1].set_ylabel('收益')
        
        plt.tight_layout()
        plt.savefig(config.output_dir / 'factor_comparison.png')
        print("\n对比图已保存: factor_comparison.png")
    except ImportError:
        print("\n(安装 matplotlib 可查看可视化对比)")


# ============================================================================
# 示例 8: 导出因子到独立脚本
# ============================================================================
def example_8_export_factors():
    """
    将最佳因子导出为可独立运行的 Python 脚本
    """
    print("\n" + "="*80)
    print("示例 8: 导出因子")
    print("="*80)
    
    config = get_config()
    results_dir = config.output_dir / "agent_results"
    
    factors_files = sorted(results_dir.glob('all_factors_*.csv'))
    if not factors_files:
        print("未找到因子文件")
        return
    
    df = pd.read_csv(factors_files[-1])
    
    # 获取最佳因子
    best = df.loc[df['sharpe_ratio'].idxmax()]
    
    # 生成独立脚本
    script = f'''"""
导出的因子脚本
因子名称: {best['name']}
方向: {best['direction']}
Sharpe: {best['sharpe_ratio']:.4f}
"""

def calculate_factor(alpha_obj):
    """计算因子值"""
    return {best['expression']}

if __name__ == '__main__':
    # 使用示例
    from alpha101.world_quant.Alpha101_code_1 import Alphas
    
    # 加载数据并创建 Alphas 对象
    # ... 你的数据加载代码
    
    # alpha_obj = Alphas(your_data)
    # factor_values = calculate_factor(alpha_obj)
    pass
'''
    
    export_path = results_dir / f"factor_{best['name']}.py"
    with open(export_path, 'w', encoding='utf-8') as f:
        f.write(script)
    
    print(f"最佳因子已导出: {export_path}")
    print(f"因子名称: {best['name']}")
    print(f"Sharpe: {best['sharpe_ratio']:.4f}")
    print(f"表达式: {best['expression']}")


# ============================================================================
# 示例 9: 持续学习和演进
# ============================================================================
def example_9_continuous_learning():
    """
    演示如何进行持续学习：根据之前的结果改进搜索策略
    """
    print("\n" + "="*80)
    print("示例 9: 持续学习")
    print("="*80)
    
    # 第一轮：初始搜索
    print("\n第一轮: 初始因子搜索...")
    config = get_config()
    agent = AlphaAgent(config)
    agent.run(n_iterations=1)
    
    # 分析第一轮结果
    factors_files = sorted(config.output_dir.glob('**/all_factors_*.csv'))
    df1 = pd.read_csv(factors_files[-1])
    best_direction_1 = df1.groupby('direction')['sharpe_ratio'].max().idxmax()
    print(f"\n第一轮最佳方向: {best_direction_1}")
    
    # 第二轮：针对性搜索
    print(f"\n第二轮: 针对最佳方向 '{best_direction_1}' 的深入搜索...")
    # (在实际应用中，可以修改 LLM 以优先选择这个方向)
    agent.run(n_iterations=1)
    
    # 比较两轮结果
    factors_files = sorted(config.output_dir.glob('**/all_factors_*.csv'))
    df2 = pd.read_csv(factors_files[-1])
    
    print(f"\n结果对比:")
    print(f"第一轮平均 Sharpe: {df1['sharpe_ratio'].mean():.4f}")
    print(f"第二轮平均 Sharpe: {df2['sharpe_ratio'].mean():.4f}")


# ============================================================================
# 主函数
# ============================================================================
def main():
    """运行所有示例"""
    print("\n" + "#"*80)
    print("# Alpha Agent 高级示例")
    print("#"*80)
    
    examples = [
        ("1. 基础使用", example_1_basic_usage),
        ("2. 自定义配置", example_2_custom_config),
        ("3. 分析结果", example_3_analyze_results),
        ("4. 会话历史", example_4_session_history),
        ("5. 批量实验", example_5_batch_experiments),
        ("6. 自定义 LLM", example_6_custom_llm),
        ("7. 性能对比", example_7_factor_comparison),
        ("8. 导出因子", example_8_export_factors),
        ("9. 持续学习", example_9_continuous_learning),
    ]
    
    print("\n可用示例:")
    for name, _ in examples:
        print(f"  {name}")
    
    choice = input("\n选择要运行的示例 (1-9，或 'all' 运行全部): ").strip()
    
    if choice == 'all':
        for name, func in examples:
            try:
                print(f"\n执行: {name}")
                func()
            except Exception as e:
                print(f"✗ {name} 失败: {e}")
    else:
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(examples):
                examples[idx][1]()
            else:
                print("无效的选择")
        except ValueError:
            print("请输入有效的数字")


if __name__ == '__main__':
    main()
