"""
Alpha 因子自动挖掘 Agent
支持 LLM 驱动的因子生成、回测和迭代优化
"""
import sys
from pathlib import Path
from datetime import datetime
import json
import random
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any
from tqdm import tqdm
import traceback

# 添加项目路径
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

from alpha101.world_quant.Alpha101_code_1 import Alphas
from alpha101.futures_ml.data import build_wide_df
from alpha101.futures_ml.config import get_config as get_ml_config
from alpha101.futures_ml.alpha_sharpe import stratified_backtest
from alpha101.world_quant.factor_generator import FactorGenerator
from alpha101.world_quant.fastengine import FastExpressionEngine

from .config import get_config
from .llm_interface import LLMInterface, Factor


class AlphaAgent:
    """Alpha 因子自动挖掘 Agent"""
    
    def __init__(self, config=None):
        """
        初始化 Agent
        
        Args:
            config: AgentConfig 实例
        """
        self.config = config or get_config()
        
        print("初始化 Alpha Agent...")
        print(f"配置: {self.config}")
        
        # 初始化 LLM 接口
        self.llm = LLMInterface(self.config)
        
        # 加载数据
        print("\n加载市场数据...")
        ml_config = get_ml_config()
        self.wide_data = build_wide_df(
            ml_config.pairs,
            ml_config.lookback_days,
            ml_config.data_root,
            ml_config.timeframe,
            train_bars=ml_config.train_bars,
            test_start_date=ml_config.test_start_date,
            test_end_date=ml_config.test_end_date
        )
        print(f"数据形状: {self.wide_data.shape}")
        
        # 初始化表达式引擎
        self.alpha_obj = Alphas(self.wide_data)
        self.engine = FastExpressionEngine(self.alpha_obj)
        self.generator = FactorGenerator()
        
        # 初始化结果存储
        self.results_dir = self.config.output_dir / "agent_results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.all_factors = []
        self.session_history = []
    
    def get_context_info(self) -> Dict[str, Any]:
        """
        获取市场和算子的上下文信息
        
        Returns:
            上下文信息字典
        """
        # 获取可用的时序算子
        ts_operators = {
            "单变量算子": {
                "ts_mean": "时间序列均值（参数: 3-60 天）",
                "ts_std_dev": "时间序列标准差（参数: 7-54 天）",
                "ts_rank": "时间序列排名（参数: 3-30 天）",
                "ts_min": "时间序列最小值（参数: 3-30 天）",
                "ts_max": "时间序列最大值（参数: 3-30 天）",
                "ts_sum": "时间序列求和（参数: 5-60 天）",
                "ts_arg_max": "时间序列最大值位置（参数: 7-28 天）",
                "ts_arg_min": "时间序列最小值位置（参数: 7-28 天）",
            },
            "双变量算子": {
                "ts_corr": "时间序列相关性（参数: 5-60 天）",
                "ts_alpha": "时间序列斜率（参数: 7-28 天）",
                "ts_beta": "时间序列 Beta（参数: 7-28 天）",
            },
            "截面算子": {
                "rank": "截面排名 (0-1)",
                "scale": "截面标准化 (0-1)",
            },
            "延迟算子": {
                "ts_delay": "延迟 N 期（参数: 1-20 天）",
                "ts_delta": "差分 N 期（参数: 1-10 天）",
            },
            "数学运算": {
                "log": "自然对数",
                "abs": "绝对值",
                "+": "加法",
                "-": "减法",
                "*": "乘法",
                "/": "除法",
            }
        }
        
        # 获取市场数据信息
        data_info = {
            "开始日期": str(self.wide_data.index[0]),
            "结束日期": str(self.wide_data.index[-1]),
            "数据天数": len(self.wide_data),
            "交易对数": len(self.wide_data.columns) // 7,
            "数据字段": ["open", "high", "low", "close", "volume", "returns", "vwap"]
        }
        
        return {
            "asset_class": "加密货币期货",
            "direction_type": "多空双向",
            "frequency": "日频",
            "current_date": datetime.now().strftime("%Y-%m-%d"),
            "operators": ts_operators,
            "data_info": data_info,
        }
    
    def get_expression_examples(self) -> List[str]:
        """
        获取因子表达式示例
        
        Returns:
            表达式示例列表
        """
        return [
            # 动量因子示例
            "ts_mean(close, 10) / ts_mean(close, 30) - 1",
            "ts_rank(returns, 10) / 10 - 0.5",
            
            # 反转因子示例
            "(-1) * ts_rank(returns, 5) / 5",
            "ts_delay(returns, 1) * (-1)",
            
            # 量价背离示例
            "abs(ts_rank(returns, 10) - ts_rank(volume, 10)) / 10",
            "ts_corr(close, volume, 20)",
            
            # 波动率因子示例
            "ts_std_dev(returns, 20) - ts_mean(ts_std_dev(returns, 5), 10)",
            "ts_std_dev(returns, 20)",
            
            # 均值回归示例
            "close - ts_mean(close, 20)",
            "(close - ts_min(close, 20)) / (ts_max(close, 20) - ts_min(close, 20))",
        ]
    
    def evaluate_factor(self, factor_name: str, factor_expr: str, direction: str) -> Factor:
        """
        评估单个因子
        
        Args:
            factor_name: 因子名称
            factor_expr: 因子表达式
            direction: 因子方向
            
        Returns:
            Factor 对象
        """
        try:
            print(f"\n评估因子: {factor_name}")
            print(f"表达式: {factor_expr}")
            
            # 计算因子值
            result = self.engine.evaluate(factor_expr)
            
            # 检查结果有效性
            if result is None or result.isnull().all().all():
                return Factor(
                    name=factor_name,
                    expression=factor_expr,
                    direction=direction,
                    sharpe_ratio=0.0,
                    returns=0.0,
                    ic_mean=0.0,
                    fitness=0.0,
                    notes="All NaN result"
                )
            
            # 转换为标准格式
            result.index.name = 'date'
            result.columns.name = 'symbol'
            
            # 添加到数据集
            result_formatted = result.copy()
            result_formatted.columns = pd.MultiIndex.from_product(
                [[factor_name], result.columns]
            )
            
            # 合并数据并回测
            test_df = pd.concat([self.wide_data, result_formatted], axis=1)
            
            # 跳过前一年数据（预热期）
            test_df = test_df.iloc[365:]
            
            # 执行回测
            backtest_result = stratified_backtest(test_df, [factor_name])
            backtest_result = backtest_result[backtest_result['factor'] == factor_name]
            
            # 提取关键指标
            if backtest_result.empty:
                return Factor(
                    name=factor_name,
                    expression=factor_expr,
                    direction=direction,
                    sharpe_ratio=0.0,
                    returns=0.0,
                    ic_mean=0.0,
                    fitness=0.0,
                    notes="Empty backtest result"
                )
            
            # 计算汇总指标
            complexity = self.generator.calculate_complexity(factor_expr)
            depth_penalty = (
                np.exp(-max(0, complexity['complexity_score'] - 10) * 0.1)
                if complexity['max_depth'] > 3
                else 1.0
            )
            
            fitness = backtest_result['fitness'].item() * depth_penalty * 100
            
            factor = Factor(
                name=factor_name,
                expression=factor_expr,
                direction=direction,
                sharpe_ratio=float(backtest_result['sharpe'].item()),
                returns=backtest_result['returns'].item(),
                ic_mean=float(backtest_result['ic_mean'].item()),
                fitness=float(fitness),
                notes=f"复杂度评分: {complexity['complexity_score']:.2f}"
            )
            
            print(f"✓ Sharpe: {factor.sharpe_ratio:.4f}, Returns: {factor.returns}, Fitness: {factor.fitness:.2f}")
            
            return factor
            
        except Exception as e:
            print(f"✗ 评估失败: {str(e)}")
            return Factor(
                name=factor_name,
                expression=factor_expr,
                direction=direction,
                sharpe_ratio=0.0,
                returns=0.0,
                ic_mean=0.0,
                fitness=0.0,
                notes=f"Error: {str(e)}"
            )
    
    def run(
        self,
        n_iterations: int = 3,
        direction_mode: str = "human",
        direction: Optional[str] = None,
        reflect_iterations: int = 10
    ):
        """
        运行 Agent 的主循环
        
        Args:
            n_iterations: 迭代次数
            direction_mode: 方向选择方式（human 或 random）
            direction: 固定方向（可选）
            reflect_iterations: 反思与改进迭代次数
        """
        print("\n" + "="*80)
        print("启动 Alpha 因子自动挖掘 Agent")
        print("="*80)
        
        # 记录会话开始时间
        session_start = datetime.now()
        
        # for iteration in range(n_iterations):
        n_iterations, iteration = 1, 0
        print(f"\n{'='*80}")
        print(f"迭代 {iteration + 1}/{n_iterations}")
        print(f"{'='*80}")
        
        # 第一轮：选择挖掘方向
        print("\n第一步: 选择挖掘方向...")
        if direction:
            current_direction = direction
        elif direction_mode == "random":
            current_direction = random.choice([
                "量价背离",
                "动量",
                "反转",
                "均值回归",
                "波动率",
                "成交量异常"
            ])
        else:
            user_input = input("请输入挖掘方向(回车随机): ").strip()
            if user_input:
                current_direction = user_input
            else:
                current_direction = random.choice([
                    "量价背离",
                    "动量",
                    "反转",
                    "均值回归",
                    "波动率",
                    "成交量异常"
                ])
        print(f"选择的方向: {current_direction}")
        
        # 第二轮：生成因子创意
        print("\n第二步: 生成因子创意...")
        context = self.get_context_info()
        ideas = self.llm.generate_factor_ideas(current_direction, context)
        print(f"生成了 {len(ideas)} 个因子创意")
        
        for i, idea in enumerate(ideas):
            print(f"\n创意 {i+1}: {idea.get('name', '未命名')}")
            print(f"  假设: {idea.get('hypothesis', 'N/A')}")
            
            # 第三轮：构造因子表达式
            print(f"  构造表达式...")
            examples = self.get_expression_examples()
            factor_expr = self.llm.construct_factor_expression(idea, examples)
            print(f"  表达式: {factor_expr[:100]}...")
            
            # 第四轮：评估因子
            factor_name = f"iter{iteration+1}_idea{i+1}"
            factor = self.evaluate_factor(factor_name, factor_expr, current_direction)
            self.all_factors.append(factor)
            
            # 第五轮：LLM 反思和改进
            if factor.fitness > 0:
                print(f"\n  第五步: LLM 反思和改进（多轮迭代）...")
                current_factor = factor
                improvements = []
                
                for reflect_idx in range(reflect_iterations):
                    print(f"\n  反思迭代 {reflect_idx + 1}/{reflect_iterations}...")
                    
                    # LLM 反思（能看到历史）
                    reflection = self.llm.reflect_on_results(
                        current_factor,
                        self.all_factors[:-1],
                        improvement_history=improvements  # 传入改进历史
                    )
                    print(f"  评级: {reflection.get('rating', 'N/A')}")
                    
                    # 生成多个候选表达式
                    print(f"  生成 3 个改进候选...")
                    candidate_exprs = self.llm.generate_factor_candidates(
                        current_factor,
                        reflection,
                        examples,
                        num_candidates=3
                    )
                    print(f"  生成了 {len(candidate_exprs)} 个候选")
                    
                    # 批量评估候选
                    candidates_results = []
                    for idx, expr in enumerate(candidate_exprs, 1):
                        print(f"  评估候选 {idx}/{len(candidate_exprs)}: {expr[:80]}...")
                        candidate_factor = self.evaluate_factor(
                            f"{factor_name}_improved_{reflect_idx + 1}_candidate_{idx}",
                            expr,
                            current_direction
                        )
                        candidates_results.append(candidate_factor)
                    
                    # 选择最佳候选
                    best_candidate = max(candidates_results, key=lambda f: f.fitness)
                    print(f"  最佳候选: Sharpe={best_candidate.sharpe_ratio:.4f}, Fitness={best_candidate.fitness:.2f}")
                    
                    self.all_factors.append(best_candidate)
                    
                    # 记录改进信息
                    improvements.append({
                        'iteration': reflect_idx + 1,
                        'reflection': reflection,
                        'candidates': [f.to_dict() for f in candidates_results],
                        'best_candidate': best_candidate.to_dict(),
                        'improved_factor': best_candidate.to_dict()
                    })
                    
                    # 早停条件
                    if best_candidate.fitness <= 0 or best_candidate.fitness <= current_factor.fitness:
                        print(f"  性能未改善，停止迭代")
                        break
                    
                    current_factor = best_candidate
                
                # 记录反思信息
                self.session_history.append({
                    'iteration': iteration + 1,
                    'idea': idea,
                    'original_factor': factor.to_dict(),
                    'improvements': improvements
                })
        
        # 保存结果
        self._save_results(session_start)
        
        # 汇总结果
        self._summarize_results()
    
    def _save_results(self, session_start: datetime):
        """保存所有结果到文件"""
        timestamp = session_start.strftime("%Y%m%d_%H%M%S")
        
        # 保存所有因子
        factors_df = pd.DataFrame([f.to_dict() for f in self.all_factors])
        factors_path = self.results_dir / f"all_factors_{timestamp}.csv"
        factors_df.to_csv(factors_path, index=False)
        print(f"\n因子结果已保存: {factors_path}")
        
        # 保存会话历史（JSON）
        history_path = self.results_dir / f"session_history_{timestamp}.json"
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(self.session_history, f, ensure_ascii=False, indent=2)
        print(f"会话历史已保存: {history_path}")
        
        # 保存配置信息
        config_path = self.results_dir / f"config_{timestamp}.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump({
                'context': self.get_context_info(),
                'examples': self.get_expression_examples()
            }, f, ensure_ascii=False, indent=2)
        print(f"配置信息已保存: {config_path}")
    
    def _summarize_results(self):
        """汇总和分析结果"""
        print(f"\n{'='*80}")
        print("因子挖掘汇总")
        print(f"{'='*80}\n")
        
        if not self.all_factors:
            print("没有评估任何因子")
            return
        
        # 过滤有效因子
        valid_factors = [f for f in self.all_factors if f.fitness > 0]
        
        if not valid_factors:
            print("没有找到有效的因子")
            return
        
        # 按 Sharpe 比率排序
        valid_factors.sort(key=lambda f: f.sharpe_ratio, reverse=True)
        
        print(f"总因子数: {len(self.all_factors)}")
        print(f"有效因子: {len(valid_factors)}")
        print(f"成功率: {len(valid_factors) / len(self.all_factors) * 100:.1f}%\n")
        
        print("Top 10 因子 (按 Sharpe 比率):")
        print("-" * 80)
        for i, factor in enumerate(valid_factors[:10], 1):
            print(f"{i}. {factor.name:<20} Sharpe: {factor.sharpe_ratio:>8.4f}  Return: {factor.returns}")
            if len(factor.expression) > 70:
                print(f"   表达式: {factor.expression[:70]}...")
            else:
                print(f"   表达式: {factor.expression}")
        
        # 按方向统计
        print(f"\n按方向统计:")
        print("-" * 80)
        directions = {}
        for factor in valid_factors:
            if factor.direction not in directions:
                directions[factor.direction] = {'count': 0, 'avg_sharpe': 0}
            directions[factor.direction]['count'] += 1
            directions[factor.direction]['avg_sharpe'] += factor.sharpe_ratio
        
        for direction, stats in directions.items():
            avg_sharpe = stats['avg_sharpe'] / stats['count']
            print(f"{direction:<15} 因子数: {stats['count']:<3}  平均 Sharpe: {avg_sharpe:.4f}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Alpha 因子自动挖掘 Agent')
    parser.add_argument('--iterations', type=int, default=1,
                       help='迭代次数')
    parser.add_argument('--direction-mode', type=str, default='human',
                       choices=['human', 'random'],
                       help='挖掘方向选择方式')
    parser.add_argument('--direction', type=str, default=None,
                       help='固定挖掘方向（可选）')
    parser.add_argument('--reflect-iterations', type=int, default=10,
                       help='反思与改进迭代次数')
    parser.add_argument('--env-file', type=str, default=None,
                       help='.env 文件路径')
    
    args = parser.parse_args()
    
    # 初始化配置
    if args.env_file:
        config = get_config(args.env_file)
    else:
        config = get_config()
    
    # 验证配置
    if not config.validate():
        print("配置验证失败，请检查 .env 文件")
        return
    
    # 创建并运行 Agent
    agent = AlphaAgent(config)
    agent.run(
        n_iterations=args.iterations,
        direction_mode=args.direction_mode,
        direction=args.direction,
        reflect_iterations=args.reflect_iterations
    )


if __name__ == '__main__':
    main()
