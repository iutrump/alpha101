import pandas as pd
import os
def get_merged_latest():
    dir = 'factor_search_results/4h'
    merged_list = []
    for file in os.listdir(dir):
        if file.endswith('.csv') and file.startswith('summary_'):
            df = pd.read_csv(os.path.join(dir, file))
            merged_list.append(df)
    if merged_list:
        merged_df = pd.concat(merged_list, ignore_index=True)
        merged_df = merged_df.drop_duplicates(subset=['expression'], keep='last')
        merged_df['ic_ir'] = merged_df['ic_mean'] / merged_df['ic_std']
        turnover_cond = (merged_df['turnover'].str.replace('%', '').astype(float) > 1) if  hasattr(merged_df['turnover'], 'str') else merged_df['turnover'].astype(float) > 1
        sharpe_cond = (merged_df['sharpe_ratio'] > 2) | (merged_df['sharpe_ratio'] < -2)
        icir_cond = (merged_df['ic_ir'] > 0.15) | (merged_df['ic_ir'] < -0.15)
        ic_mean_cond = (merged_df['ic_mean'] > 0.02) | (merged_df['ic_mean'] < -0.02)
        merged_df = merged_df[turnover_cond & sharpe_cond & icir_cond & ic_mean_cond]
        merged_df['abs_sharpe'] = merged_df['sharpe_ratio'].abs()
        merged_df = merged_df.sort_values(by='abs_sharpe', ascending=False).reset_index(drop=True)
        save_path = os.path.join(dir, 'a_merged_latest.csv')
        # 为factor_name添加前缀alpha_和后缀idx，来区别不同的alpha因子
        for idx, row in merged_df.iterrows():
            merged_df.at[idx, 'factor_name'] = f"alpha_{idx}"
        merged_df.to_csv(save_path, index=False)
        print(f"Merged latest CSV saved as '{save_path}'.")
if __name__ == "__main__":
    get_merged_latest()