"""
Alpha Agent 新功能演示脚本
展示 LLM 反思历史可见性 + 多候选生成与评估功能
"""
import sys
from pathlib import Path
import json

repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))

from alpha101.agent import AlphaAgent, get_config


def demo_basic_usage():
    """演示1：基本用法 - 默认参数"""
    print("\n" + "="*80)
    print("演示 1: 基本用法（默认3次反思迭代，3个候选）")
    print("="*80)
    
    config = get_config()
    agent = AlphaAgent(config)
    
    # 固定方向，3次迭代，默认反思3次
    agent.run(
        n_iterations=1,
        direction="动量",
        direction_mode="human",
        reflect_iterations=3
    )


def demo_deep_reflection():
    """演示2：深度反思 - 更多迭代次数"""
    print("\n" + "="*80)
    print("演示 2: 深度反思（5次反思迭代）")
    print("="*80)
    
    config = get_config()
    agent = AlphaAgent(config)
    
    # 让LLM看到更多改进历史
    agent.run(
        n_iterations=1,
        direction="反转",
        direction_mode="random",
        reflect_iterations=5  # 更深的反思
    )


def demo_analyze_improvements():
    """演示3：分析改进历史"""
    print("\n" + "="*80)
    print("演示 3: 分析改进历史")
    print("="*80)
    
    config = get_config()
    results_dir = config.output_dir / "agent_results"
    
    # 找到最新的会话历史
    history_files = sorted(results_dir.glob('session_history_*.json'))
    if not history_files:
        print("未找到会话历史文件")
        return
    
    with open(history_files[-1], 'r', encoding='utf-8') as f:
        history = json.load(f)
    
    print(f"总记录数: {len(history)}\n")
    
    for record_idx, record in enumerate(history, 1):
        print(f"{'='*60}")
        print(f"记录 {record_idx}: {record['idea']['name']}")
        print(f"{'='*60}")
        
        original = record['original_factor']
        print(f"\n原始因子:")
        print(f"  Sharpe: {original['sharpe_ratio']:.4f}")
        print(f"  Fitness: {original['fitness']:.2f}")
        
        improvements = record.get('improvements', [])
        print(f"\n改进历史 (共 {len(improvements)} 次迭代):")
        
        for imp_idx, imp in enumerate(improvements, 1):
            print(f"\n  迭代 {imp_idx}:")
            reflection = imp.get('reflection', {})
            print(f"    反思评级: {reflection.get('rating', 'N/A')}")
            print(f"    改进建议: {', '.join(reflection.get('improvements', [])[:2])}")
            
            candidates = imp.get('candidates', [])
            print(f"    候选数: {len(candidates)}")
            for cand_idx, cand in enumerate(candidates, 1):
                best_mark = "✓ 最优" if cand == imp.get('best_candidate') else ""
                print(f"      候选{cand_idx}: Sharpe={cand['sharpe_ratio']:.4f}, Fitness={cand['fitness']:.2f} {best_mark}")
            
            best = imp.get('best_candidate', {})
            print(f"    选中最优: Sharpe={best.get('sharpe_ratio', 0):.4f}, Fitness={best.get('fitness', 0):.2f}")


def demo_compare_candidates():
    """演示4：查看候选表达式对比"""
    print("\n" + "="*80)
    print("演示 4: 候选表达式对比")
    print("="*80)
    
    config = get_config()
    results_dir = config.output_dir / "agent_results"
    
    history_files = sorted(results_dir.glob('session_history_*.json'))
    if not history_files:
        print("未找到会话历史文件")
        return
    
    with open(history_files[-1], 'r', encoding='utf-8') as f:
        history = json.load(f)
    
    if not history:
        print("历史记录为空")
        return
    
    record = history[0]
    improvements = record.get('improvements', [])
    
    if not improvements:
        print("没有改进记录")
        return
    
    print(f"因子: {record['idea']['name']}")
    print(f"原始表达式: {record['original_factor']['expression']}\n")
    
    for imp_idx, imp in enumerate(improvements[:2], 1):  # 只显示前2次迭代
        print(f"{'='*60}")
        print(f"迭代 {imp_idx} - 生成的候选表达式:")
        print(f"{'='*60}")
        
        candidates = imp.get('candidates', [])
        best_idx = -1
        
        for cand_idx, cand in enumerate(candidates, 1):
            is_best = cand == imp.get('best_candidate')
            if is_best:
                best_idx = cand_idx
            
            mark = "✓ 选中" if is_best else " "
            print(f"\n{mark} 候选 {cand_idx}:")
            print(f"    表达式: {cand['expression']}")
            print(f"    Sharpe: {cand['sharpe_ratio']:.4f}")
            print(f"    Fitness: {cand['fitness']:.2f}")
        
        print(f"\n→ 最优候选: #{best_idx} (Fitness={imp['best_candidate']['fitness']:.2f})")


def demo_multiple_iterations():
    """演示5：多轮外层迭代"""
    print("\n" + "="*80)
    print("演示 5: 多轮迭代（3个方向，每个方向反思3次）")
    print("="*80)
    
    config = get_config()
    agent = AlphaAgent(config)
    
    # 运行3次外层迭代，每次随机选择方向
    agent.run(
        n_iterations=3,
        direction_mode="random",
        reflect_iterations=3
    )


def main():
    """主菜单"""
    print("\n" + "#"*80)
    print("# Alpha Agent 新功能演示")
    print("#"*80)
    
    demos = [
        ("1. 基本用法", demo_basic_usage),
        ("2. 深度反思", demo_deep_reflection),
        ("3. 分析改进历史", demo_analyze_improvements),
        ("4. 候选表达式对比", demo_compare_candidates),
        ("5. 多轮迭代", demo_multiple_iterations),
    ]
    
    print("\n可用演示:")
    for name, _ in demos:
        print(f"  {name}")
    
    choice = input("\n选择要运行的演示 (1-5，或 'all'): ").strip()
    
    try:
        if choice == 'all':
            # 只运行实践演示，跳过分析演示
            for name, func in demos[:2]:
                try:
                    print(f"\n执行: {name}")
                    func()
                except Exception as e:
                    print(f"✗ {name} 失败: {e}")
        else:
            idx = int(choice) - 1
            if 0 <= idx < len(demos):
                name, func = demos[idx]
                print(f"\n执行: {name}")
                func()
            else:
                print("无效的选择")
    except ValueError:
        print("请输入有效的数字")
    except KeyboardInterrupt:
        print("\n已取消")


if __name__ == '__main__':
    main()
