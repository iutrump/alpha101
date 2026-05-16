"""
Alpha Agent 快速开始脚本
"""
import sys
from pathlib import Path

# 添加项目路径
repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))

from alpha101.agent import AlphaAgent, get_config


def main():
    """快速开始示例"""
    
    print("="*80)
    print("Alpha 因子自动挖掘 Agent - 快速开始")
    print("="*80)
    
    # 步骤 1: 验证配置
    print("\n步骤 1: 验证配置...")
    config = get_config()
    if not config.validate():
        print("✗ 配置验证失败")
        print("  请检查 .env 文件并确保 OPENAI_API_KEY 已设置")
        return False
    print(f"✓ 配置有效")
    print(f"  - 模型: {config.openai_model}")
    print(f"  - 输出目录: {config.output_dir}")
    
    # 步骤 2: 初始化 Agent
    print("\n步骤 2: 初始化 Agent...")
    try:
        agent = AlphaAgent(config)
        print("✓ Agent 初始化成功")
        print(f"  - 数据形状: {agent.wide_data.shape}")
        print(f"  - 时间范围: {agent.wide_data.index[0]} 至 {agent.wide_data.index[-1]}")
    except Exception as e:
        print(f"✗ 初始化失败: {e}")
        return False
    
    # 步骤 3: 显示可用的算子
    print("\n步骤 3: 显示可用的算子...")
    context = agent.get_context_info()
    print("✓ 可用算子:")
    for category, operators in context['operators'].items():
        print(f"\n  {category}:")
        if isinstance(operators, dict):
            for op_name, op_desc in list(operators.items())[:3]:
                print(f"    - {op_name}: {op_desc}")
            if len(operators) > 3:
                print(f"    ... 还有 {len(operators) - 3} 个")
        else:
            for op in operators[:3]:
                print(f"    - {op}")
            if len(operators) > 3:
                print(f"    ... 还有 {len(operators) - 3} 个")
    
    # 步骤 4: 显示表达式示例
    print("\n步骤 4: 显示表达式示例...")
    examples = agent.get_expression_examples()
    print("✓ 表达式示例 (前 5 个):")
    for i, example in enumerate(examples[:5], 1):
        print(f"  {i}. {example}")
    
    # 步骤 5: 运行 Agent
    print("\n步骤 5: 准备运行 Agent...")
    response = input("\n是否继续运行 Agent? (y/n): ").strip().lower()
    
    if response in ['y', 'yes']:
        iterations = input("请输入迭代次数 (默认: 1): ").strip()
        try:
            iterations = int(iterations) if iterations else 1
        except ValueError:
            iterations = 1
        
        print(f"\n开始运行 {iterations} 次迭代...\n")
        agent.run(n_iterations=iterations)
        return True
    else:
        print("\n已取消运行")
        print("\n你可以使用以下命令手动运行:")
        print("  python run_quick_start.py")
        print("  python -m alpha101.agent.agent --iterations 1")
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
