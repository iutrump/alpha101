import sys
import warnings
import multiprocessing as mp
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import math
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))  # 添加项目根目录到 sys.path
from alpha101.futures_ml import config
from alpha101.futures_ml.data import build_full_dataset


def analyze_factor_correlation(df: pd.DataFrame, alpha_cols: list, threshold: float = 0.7) -> pd.DataFrame:
    """
    分析 alpha 因子之间的相关性，找出高相关因子对
    
    Args:
        df: 宽表 DataFrame（列为 MultiIndex: (alpha, symbol)）
        alpha_cols: alpha 因子列表，如 ['alpha020', 'alpha041', ...]
        threshold: 相关性阈值（0-1），只输出超过此阈值的因子对
    
    Returns:
        高相关因子对 DataFrame，包含列: factor1, factor2, mean_correlation, std_correlation, obs_count, mean_abs_correlation
    """
    print(f"\nComputing correlation matrix for {len(alpha_cols)} features...")
    
    # Step 1: 提取所有需要的因子列
    alpha_features = df[[col for col in df.columns if col[0] in alpha_cols]]
    
    if alpha_features.empty:
        print("No alpha features found")
        return pd.DataFrame()
    
    # Step 2: 计算每天每个因子的中位数（横截面聚合）
    print("Aggregating daily factor values (median across stocks)...")
    
    # 获取所有因子名
    unique_factors = set([col[0] for col in alpha_features.columns])
    
    # 为每个因子创建一列（每天的中位数）
    daily_factor_median = {}
    for factor in unique_factors:
        # 获取该因子下的所有股票列
        factor_cols = [col for col in alpha_features.columns if col[0] == factor]
        # 计算每天的中位数
        daily_factor_median[factor] = alpha_features[factor_cols].median(axis=1)
    
    # 转换为 DataFrame
    factor_df = pd.DataFrame(daily_factor_median)
    
    # 删除包含 NaN 的行
    factor_df = factor_df.dropna()
    print(f"Factor time series shape: {factor_df.shape} (days x factors)")
    
    # Step 3: 计算因子之间的相关性矩阵
    print("Computing correlation matrix...")
    corr_matrix = factor_df.corr()
    
    # Step 4: 提取高相关的因子对
    high_corr_pairs = []
    
    # 获取上三角矩阵的索引
    for i, factor1 in enumerate(corr_matrix.columns):
        for j, factor2 in enumerate(corr_matrix.columns):
            if i < j:  # 只取上三角，避免重复
                corr_value = corr_matrix.iloc[i, j]
                abs_corr = abs(corr_value)
                
                if abs_corr >= threshold:
                    # 这里我们只有一对因子的一个相关性值，没有每天的统计
                    # 为了保持与原函数输出格式一致，我们创建一个单行的结果
                    high_corr_pairs.append({
                        'factor1': factor1,
                        'factor2': factor2,
                        'mean_correlation': corr_value,
                        'std_correlation': 0.0,  # 没有标准差
                        'obs_count': len(factor_df),  # 使用的天数
                        'mean_abs_correlation': abs_corr
                    })
    if len(high_corr_pairs) == 0:
        print(f"No factor pairs with |correlation| >= {threshold} found.")
        return pd.DataFrame()
    result_df = pd.DataFrame(high_corr_pairs).sort_values('mean_abs_correlation', ascending=False)
    
    print(f"\nFound {len(result_df)} highly correlated factor pairs (|r| >= {threshold})")
    
    return result_df

def remove_correlated_factors(high_corr: pd.DataFrame, alpha_name_icir_dict: dict, threshold: float = 0.7) -> set:
    """
    去除高相关因子，保留 ICIR 绝对值更高的因子
    
    Args:
        high_corr: 高相关因子对 DataFrame
        alpha_name_icir_dict: 因子名 -> ICIR 的字典
        threshold: 相关性阈值
    
    Returns:
        要移除的因子名称集合
    """
    if high_corr.empty:
        return set()
    
    print(f"\n{'='*110}")
    print(f"REMOVING HIGHLY CORRELATED FACTORS (|Correlation| ≥ {threshold})")
    print(f"{'='*110}\n")
    
    # 构建因子关系图
    factors_to_remove = set()
    
    # 按相关性从高到低排序，优先处理最高相关的
    high_corr_sorted = high_corr.sort_values('mean_abs_correlation', ascending=False)
    
    for _, row in high_corr_sorted.iterrows():
        factor1 = row['factor1']
        factor2 = row['factor2']
        
        # 如果两个因子都已经被移除了，跳过
        if factor1 in factors_to_remove and factor2 in factors_to_remove:
            continue
        
        # 如果其中一个已经被移除，跳过
        if factor1 in factors_to_remove or factor2 in factors_to_remove:
            continue
        
        # 获取 ICIR 绝对值
        icir1 = abs(alpha_name_icir_dict.get(factor1, 0.0))
        icir2 = abs(alpha_name_icir_dict.get(factor2, 0.0))
        
        # 移除 ICIR 较低的因子
        if icir1 >= icir2:
            factors_to_remove.add(factor2)
            keep_factor = factor1
            remove_factor = factor2
            keep_icir = icir1
            remove_icir = icir2
        else:
            factors_to_remove.add(factor1)
            keep_factor = factor2
            remove_factor = factor1
            keep_icir = icir2
            remove_icir = icir1
        
        print(f"Corr={row['mean_abs_correlation']:.4f}: Keep {keep_factor} (|ICIR|={keep_icir:.4f}), Remove {remove_factor} (|ICIR|={remove_icir:.4f})")
    
    print(f"\n{'='*110}")
    print(f"REMOVAL SUMMARY")
    print(f"{'='*110}")
    print(f"Total factors to remove: {len(factors_to_remove)}")
    print(f"Removed factors: {sorted(factors_to_remove)}")
    print(f"{'='*110}\n")
    
    return factors_to_remove


def print_correlation_analysis(high_corr: pd.DataFrame, threshold: float = 0.7):
    """
    打印高相关因子对分析结果
    """
    print(f"\n{'='*110}")
    print(f"FACTOR CORRELATION ANALYSIS (|Correlation| ≥ {threshold})")
    print(f"{'='*110}")
    
    if high_corr.empty:
        print(f"\nNo factor pairs with |correlation| >= {threshold} found.")
        return
    
    print(f"\nFound {len(high_corr)} highly correlated factor pairs:\n")
    print(f"{'Rank':<6} {'Factor 1':<32} {'Factor 2':<32} {'Mean Corr':>12} {'Abs Corr':>12} {'Obs':>6}")
    print('-'*112)
    
    for idx, (_, row) in enumerate(high_corr.iterrows(), 1):
        print(f"{idx:<6} {row['factor1']:<32} {row['factor2']:<32} "
              f"{row['mean_correlation']:>12.4f} {row['mean_abs_correlation']:>12.4f} {int(row['obs_count']):>6}")
    
    # 统计相关性分布
    print(f"\n{'='*110}")
    print("CORRELATION STATISTICS")
    print(f"{'='*110}")
    print(f"Mean |Correlation|:    {high_corr['mean_abs_correlation'].mean():.4f}")
    print(f"Median |Correlation|:  {high_corr['mean_abs_correlation'].median():.4f}")
    print(f"Max |Correlation|:     {high_corr['mean_abs_correlation'].max():.4f}")
    print(f"Min |Correlation|:     {high_corr['mean_abs_correlation'].min():.4f}")
    print(f"Total pairs found:     {len(high_corr)}")
    
    # 按因子统计
    all_factors = pd.concat([
        high_corr['factor1'],
        high_corr['factor2']
    ]).reset_index(drop=True)
    
    factor_counts = all_factors.value_counts()
    print(f"\nTop 10 factors by high correlation count:")
    for factor, count in factor_counts.head(10).items():
        print(f"  {factor:<34} appears in {count:3d} highly correlated pairs")
    
    print(f"{'='*110}\n")


def main():
    """主函数"""
    cfg = config.get_config()
    
    # 执行分层回测
    print("\n" + "="*100)
    print("Running stratified backtest...")
    print("="*100)
    
    
    df_backtest = build_full_dataset(
        cfg.pairs,
        cfg.lookback_days,
        cfg.data_root,
        cfg.timeframe,
        test_start_date=cfg.test_start_date,
        test_end_date=cfg.test_end_date,
        alpha_expression_path=cfg.alpha_expression_path
    )
    alpha_df = pd.read_csv(cfg.alpha_expression_path)
    alpha_name_icir_dict = alpha_df.set_index("factor_name")["ic_ir"].to_dict()
    alpha_cols = list(set([col for col in df_backtest.columns if col.startswith("alpha")]))
    
    # 转为宽表
    df_backtest = df_backtest.set_index(["date", "symbol"]).sort_index()
    df_backtest.index = df_backtest.index.set_levels(
        df_backtest.index.levels[1].astype("category"), level=1
    )
    df_backtest = df_backtest.unstack(level='symbol')
    high_corr = analyze_factor_correlation(df_backtest, alpha_cols, threshold=0.4)
    print_correlation_analysis(high_corr, threshold=0.4)
    
    # 移除高相关因子
    factors_to_remove = remove_correlated_factors(high_corr, alpha_name_icir_dict, threshold=0.4)
    
    # 计算保留的因子
    all_factors = set(alpha_cols)
    factors_to_keep = all_factors - factors_to_remove
    
    # 保存保留列表
    if factors_to_keep:
        # output_path = Path("factor_search_results/4h/factors_to_keep.txt")
        # output_path.parent.mkdir(parents=True, exist_ok=True)
        # with open(output_path, 'w') as f:
        #     for factor in sorted(factors_to_keep):
        #         f.write(f"{factor}\n")
        # print(f"Saved {len(factors_to_keep)} factors to keep to: {output_path}")
        
        # 同时保存为CSV格式，包含ICIR信息
        keep_df = alpha_df[alpha_df['factor_name'].isin(factors_to_keep)]
        keep_df.to_csv("factor_search_results/4h/factors_to_keep.csv", index=False)
        print(f"Saved factors with ICIR to: factor_search_results/4h/factors_to_keep.csv")
    
    print('\nAnalysis completed successfully!\n')

if __name__ == '__main__':
    main()