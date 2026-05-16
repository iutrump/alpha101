import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import math

from . import config
from .data import build_full_dataset
from .alpha_sharpe import stratified_backtest

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

def _build_targets(df: pd.DataFrame, future_day: int = 5) -> pd.Series:
    """
    构建目标变量：未来5天平均收益率（向前看，不含当天）
    
    注意：存在lookahead bias风险，需要确保在回测中正确使用
    """
    # 对每个symbol，计算未来5天平均收益
    future_mean = df.groupby("symbol")["target"].apply(
        lambda x: x.rolling(window=future_day, min_periods=future_day)
                   .mean()
                   .shift(-future_day)  # 未来5天的平均，不含当天
    )
    
    # 将结果重新对齐到原始数据的索引
    return future_mean.reset_index(level=0, drop=True).reindex(df.index)

def _build_alpha_targets(df: pd.DataFrame) -> pd.Series:
    if {"symbol", "trade_date", "target"} - set(df.columns):
        raise ValueError("Dataframe must include symbol, trade_date, target")

    # 1) 截面去均值：当日各币收益 - 当日全体均值
    df_sorted = df.sort_values(["symbol", "trade_date"]).copy()
    df_sorted["_orig_idx"] = df_sorted.index
    cross_mean = df_sorted.groupby("trade_date")["target"].transform("mean")
    df_sorted["demeaned"] = df_sorted["target"] - cross_mean

    # 2) 未来5天平均收益（使用去均值后的序列），对每个币种向前看5天
    n = 1
    future_5d = (
        df_sorted.groupby("symbol")["demeaned"]
        .rolling(window=n, min_periods=n)
        .mean()
        .shift(-n)
    )

    df_sorted["target_new"] = future_5d.values
    # Map back to original order
    target_aligned = df_sorted.set_index("_orig_idx")["target_new"].reindex(df.index)
    return target_aligned


def analyze_alpha_correlations(cfg):
    """
    重新计算 alpha 特征并生成新目标，然后做相关性分析。
    target_new = (当日收益去均值) 的未来5天均值。
    """

    print("Loading daily data and computing alphas...")
    df = build_full_dataset(
        cfg.pairs,
        cfg.lookback_days,
        cfg.data_root,
        cfg.timeframe,
        test_start_date=cfg.test_start_date,
        test_end_date=cfg.test_end_date,
    )
    alpha_cols = list(set([col for col in df.columns if col.startswith("alpha")]))
    # 转为宽表
    df = df.set_index(["date", "symbol"]).sort_index()
    df.index = df.index.set_levels(
        df.index.levels[1].astype("category"), level=1
    )
    df = df.unstack(level='symbol') 
    print(f"Rows: {len(df):,}, Columns: {len(df.columns)}")

    # feature_cols = [c for c in df.columns if c.startswith("alpha") and c.endswith('lag1')]
    
    print(f"Found {len(alpha_cols)} alpha features")
    if len(alpha_cols) == 0:
        print('Error: No alpha features found (should start with "alpha")')
        sys.exit(1)
    # 计算每个alpha因子的IC
    pred_future_days = 0
    print(f"Building targets: future {pred_future_days}-day average returns (demeaned)")
    ic_results = {}
    for alpha in alpha_cols:
        # daily_ic & rank_ic
        rank_target = df['target'].rank(axis=1).shift(-pred_future_days)
        daily_ic = df[alpha].corrwith(rank_target, axis=1)
        rankic = df[alpha].corrwith(rank_target, axis=1, method='spearman')

        ic_results[alpha] = {
            "mean_ic": daily_ic.mean(),
            "std_ic": daily_ic.std(),
            "ic_ir": daily_ic.mean() / daily_ic.std() if daily_ic.std() != 0 else np.nan,
            "mean_rank_ic": rankic.mean(),
            "std_rank_ic": rankic.std(),
            "rank_ic_ir": rankic.mean() / rankic.std() if rankic.std() != 0 else np.nan,
        }
    ic_results = pd.DataFrame(ic_results).T.sort_values("mean_ic", key=lambda x: x.abs(), ascending=False)
    # Calculate significance (t-test)
    n_days = len(df)
    t_stat = ic_results['mean_ic'] / (ic_results['std_ic'] / np.sqrt(n_days))
    p_values = pd.Series(np.nan, index=ic_results.index)
    valid_mask = (n_days > 1) & (ic_results['std_ic'] > 0)
    if valid_mask.any():
        p_values[valid_mask] = 2 * (1 - stats.t.cdf(np.abs(t_stat[valid_mask]), df=n_days - 1))
    sig_level = 0.05
    n_sig = (p_values < sig_level).sum()
    sig_ratio = n_sig / len(alpha_cols) if len(alpha_cols) > 0 else 0.0

    print('\n' + '='*80)
    print('TOP 15 FEATURES BY |Mean IC| (Pearson)')
    print('='*80)

    top_features = ic_results['mean_ic'].abs().head(15).index

    print(f"{'Feature Name':<30} {'Mean IC':>10} {'Mean RankIC':>12} {'ICIR':>10}")
    print('-'*80)
    for name in top_features:
        pearson_val = ic_results.loc[name, "mean_ic"]
        spearman_val = ic_results.loc[name, "mean_rank_ic"]
        ir_val = ic_results.loc[name, "ic_ir"]

        print(f'{name:<30} {pearson_val:10.4f} {spearman_val:12.4f} {ir_val:10.4f}')
    
    print('Bottom 15 FEATURES BY |Mean IC| (Pearson)')
    print('='*80)

    bottom_features = reversed(ic_results['mean_ic'].abs().tail(15).index)

    print(f"{'Feature Name':<30} {'Mean IC':>10} {'Mean RankIC':>12} {'ICIR':>10}")
    print('-'*80)
    for name in bottom_features:
        pearson_val = ic_results.loc[name, "mean_ic"]
        spearman_val = ic_results.loc[name, "mean_rank_ic"]
        ir_val = ic_results.loc[name, "ic_ir"]

        print(f'{name:<30} {pearson_val:10.4f} {spearman_val:12.4f} {ir_val:10.4f}')

    print('\n' + '='*80)
    print('IC SUMMARY STATISTICS')
    print('='*80)

    print(f'\nSignificant Features (p < {sig_level}): {n_sig}/{len(alpha_cols)} ({sig_ratio:.1%})')

    print('\nMean IC Statistics:')
    print(f'   Mean absolute value: {ic_results["mean_ic"].abs().mean():.4f}')
    print(f'   Standard deviation: {ic_results["mean_ic"].abs().std():.4f}')
    print(f'   Median: {ic_results["mean_ic"].abs().median():.4f}')

    thresholds = [0.01, 0.05, 0.1]
    print('\nFeatures by Mean IC strength:')
    for thresh in thresholds:
        count = (ic_results["mean_ic"].abs() >= thresh).sum()
        pct = count / len(ic_results) * 100
        print(f'   |Mean IC| ≥ {thresh:.2f}: {count:3d} features ({pct:5.1f}%)')

    positive_count = (ic_results["mean_ic"] > 0).sum()
    negative_count = (ic_results["mean_ic"] < 0).sum()
    print('\nMean IC direction:')
    print(f'   Positive: {positive_count:3d} features ({positive_count/len(ic_results):.1%})')
    print(f'   Negative: {negative_count:3d} features ({negative_count/len(ic_results):.1%})')

    output_dir = Path('analysis_results')
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'ic_analysis_{timestamp}.csv'
    
    ic_results.to_csv(output_file)
    print(f'\nResults saved to: {output_file}')

    # 相关性分析
    print(f"\n{'='*80}")
    print("CONDUCTING FACTOR CORRELATION ANALYSIS")
    print(f"{'='*80}")
    high_corr = analyze_factor_correlation(df, alpha_cols, threshold=0.65)
    if not high_corr.empty:
        print_correlation_analysis(high_corr, threshold=0.65)
        # 保存相关性结果
        corr_output = output_dir / f'factor_correlation_{timestamp}.csv'
        high_corr.to_csv(corr_output, index=False)
        print(f"Correlation results saved to: {corr_output}")
    else:
        print("No highly correlated factor pairs found.")

    return {
        'mean_ic': ic_results["mean_ic"],
        'mean_rank_ic': ic_results["mean_rank_ic"],
        'ic_ir': ic_results["ic_ir"],
        'top_features': top_features,
        'summary_stats': ic_results["mean_ic"].abs().describe(),
        'high_correlation_pairs': high_corr if not high_corr.empty else None
    }


def main():
    """主函数"""
    cfg = config.get_config()
    results = analyze_alpha_correlations(cfg)
    
    # 执行分层回测
    print("\n" + "="*100)
    print("Running stratified backtest...")
    print("="*100)
    
    try:
        # 需要重新加载数据以进行分层回测
        df_backtest = build_full_dataset(
            cfg.pairs,
            cfg.lookback_days,
            cfg.data_root,
            cfg.timeframe,
            test_start_date=cfg.test_start_date,
            test_end_date=cfg.test_end_date,
        )
        
        alpha_cols = list(set([col for col in df_backtest.columns if col.startswith("alpha")]))
        
        # 转为宽表
        df_backtest = df_backtest.set_index(["date", "symbol"]).sort_index()
        df_backtest.index = df_backtest.index.set_levels(
            df_backtest.index.levels[1].astype("category"), level=1
        )
        df_backtest = df_backtest.unstack(level='symbol')
        
        # 执行分层回测
        backtest_results = stratified_backtest(df_backtest, alpha_cols, target_col='target', n_quintiles=5)
        
        # 保存分层回测结果
        output_dir = Path('analysis_results')
        output_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if not backtest_results.empty:
            backtest_output = output_dir / f'stratified_backtest_{timestamp}.csv'
            backtest_results.to_csv(backtest_output, index=False)
            print(f"Stratified backtest results saved to: {backtest_output}")
        
    except Exception as e:
        print(f"Stratified backtest failed: {e}")
    
    print('\nAnalysis completed successfully!')

if __name__ == '__main__':
    main()