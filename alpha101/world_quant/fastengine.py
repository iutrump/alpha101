import sys

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import os
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

from alpha101.world_quant import Alpha101_code_1
from alpha101.futures_ml.config import get_config
import re
import inspect
from typing import Dict
from concurrent.futures import ProcessPoolExecutor, as_completed


_FAST_WORKER_ENV_BASE = None
_FAST_WORKER_PROCESS_FACTOR = None


def _convert_ternary(expr: str) -> str:
    pattern = r"\((.*?)\)\s*\?\s*(.*?)\s*:\s*(.*)"
    match = re.search(pattern, expr)
    if match:
        cond, a, b = match.groups()
        return f"np.where({cond}, {a}, {b})"
    return expr


def _remove_comments(code: str) -> str:
    cleaned_lines = []
    for line in code.splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def _init_fast_worker(alpha_fields: dict) -> None:
    """Initialize per-process globals once for factor evaluation."""
    global _FAST_WORKER_ENV_BASE, _FAST_WORKER_PROCESS_FACTOR
    from alpha101.world_quant import Alpha101_code_1 as _aq
    from alpha101.world_quant.Alpha101_code_1 import process_factor_wide_format as _process_factor_wide_format

    env = {
        "open": alpha_fields["open"],
        "high": alpha_fields["high"],
        "low": alpha_fields["low"],
        "close": alpha_fields["close"],
        "volume": alpha_fields["volume"],
        "returns": alpha_fields["returns"],
        "vwap": alpha_fields["vwap"],
        "market_return": alpha_fields["market_return"],
        "funding": alpha_fields["funding"],
        "cap": alpha_fields["cap"],
        "np": np,
        "pd": pd,
        "abs": np.abs,
        "log": lambda x: np.sign(x) * np.log(np.abs(x) + 1),
        "sign": np.sign,
        "sqrt": np.sqrt,
        "max": np.maximum,
        "min": np.minimum,
    }

    for name, obj in inspect.getmembers(_aq):
        if callable(obj) and not name.startswith('_'):
            if not inspect.isclass(obj) and not inspect.ismodule(obj):
                env[name] = obj

    _FAST_WORKER_ENV_BASE = env
    _FAST_WORKER_PROCESS_FACTOR = _process_factor_wide_format


def _eval_one_worker(item):
    factor_name, expr = item
    try:
        env = dict(_FAST_WORKER_ENV_BASE)

        fast_code = _remove_comments(expr)
        fast_code = _convert_ternary(fast_code.lower())
        lines = [line.strip() for line in fast_code.split(";") if line.strip()]

        for line in lines[:-1]:
            var, sub_expr = line.split("=", 1)
            env[var.strip()] = eval(sub_expr.strip(), {}, env)

        result = eval(lines[-1], {}, env)
        if result is None:
            return factor_name, None, expr, f"Warning: {factor_name} returned None"
        try:
            nan_all = bool(np.asarray(result.isnull()).all())
        except Exception:
            nan_all = False
        if nan_all:
            return factor_name, None, expr, f"Warning: {factor_name} returned all NaN"

        result.index.name = 'date'
        result.columns.name = 'symbol'
        result = _FAST_WORKER_PROCESS_FACTOR(result)
        return factor_name, result, expr, None
    except Exception as e:
        return factor_name, None, expr, f"Error evaluating {factor_name}: {str(e)}"

class FastExpressionEngine:

    def __init__(self, alpha_instance):
        self.alpha = alpha_instance

    def _build_env(self):
        """
        构造 eval 执行环境
        自动注入 Alpha101_code_1 中的所有函数
        """
        env = {
            # 数据字段
            "open": self.alpha.open,
            "high": self.alpha.high,
            "low": self.alpha.low,
            "close": self.alpha.close,
            "volume": self.alpha.volume,
            "returns": self.alpha.returns,
            "vwap": self.alpha.vwap,
            "market_return": self.alpha.market_return,
            "funding": self.alpha.funding,
            "cap": self.alpha.cap,
            "funding": self.alpha.funding,
            
            # 标准库
            "np": np,
            "pd": pd,
            "abs": np.abs,
            "log": np.log,
            "sign": np.sign,
            "max": np.maximum,
            "min": np.minimum,
            "exp": np.exp
        }
        
        # 自动注入 Alpha101_code_1 中的所有函数
        for name, obj in inspect.getmembers(Alpha101_code_1):
            if callable(obj) and not name.startswith('_'):
                # 排除类和模块
                if not inspect.isclass(obj) and not inspect.ismodule(obj):
                    env[name] = obj
        
        return env
    @staticmethod
    def convert_ternary(expr):
        pattern = r"\((.*?)\)\s*\?\s*(.*?)\s*:\s*(.*)"
        match = re.search(pattern, expr)
        if match:
            cond, a, b = match.groups()
            return f"np.where({cond}, {a}, {b})"
        return expr
    @staticmethod
    def remove_comments(code: str) -> str:
        cleaned_lines = []

        for line in code.splitlines():
            # 删除行内注释
            line = line.split("#", 1)[0]
            line = line.strip()

            if line:  # 忽略空行
                cleaned_lines.append(line)

        return "\n".join(cleaned_lines)

    def evaluate(self, fast_code: str):
        """
        解析并执行 FAST 表达式
        """
        env = self._build_env()
        # 预处理
        fast_code = self.remove_comments(fast_code)
        fast_code = FastExpressionEngine.convert_ternary(fast_code.lower())
        # 按分号拆分
        lines = [line.strip() for line in fast_code.split(";") if line.strip()]

        # 前面是赋值语句
        for line in lines[:-1]:
            var, expr = line.split("=", 1)
            var = var.strip()
            expr = expr.strip()
            env[var] = eval(expr, {}, env)

        # 最后一行是返回值
        result = eval(lines[-1], {}, env)

        return result
    
    def evaluate_batch(
        self,
        expressions: Dict[str, str],
        progress_bar: bool = True,
        backend: str = "process",
        max_workers: int | None = None,
    ) -> pd.DataFrame:
        """
        批量计算多个因子表达式并拼接结果
        
        Args:
            expressions: 字典，键为因子名称，值为表达式字符串
            progress_bar: 是否显示进度条
            backend: 并发后端，支持 "process" 或 "serial"
            max_workers: 最大进程数，None 时自动选择
            
        Returns:
            拼接后的 DataFrame，列为 MultiIndex (factor_name, symbol)
        """
        from alpha101.world_quant.Alpha101_code_1 import process_factor_wide_format
        
        def _eval_one(factor_name: str, expr: str):
            try:
                result = self.evaluate(expr)

                # 检查结果有效性（兼容 DataFrame/Series）
                if result is None:
                    return factor_name, None, expr, f"Warning: {factor_name} returned None"
                try:
                    nan_all = bool(np.asarray(result.isnull()).all())
                except Exception:
                    nan_all = False
                if nan_all:
                    return factor_name, None, expr, f"Warning: {factor_name} returned all NaN"

                result.index.name = 'date'
                result.columns.name = 'symbol'
                result = process_factor_wide_format(result)
                return factor_name, result, expr, None
            except Exception as e:
                return factor_name, None, expr, f"Error evaluating {factor_name}: {str(e)}"

        results = {}
        items = list(expressions.items())

        if backend not in {"process", "serial"}:
            print(f"Unknown backend '{backend}', fallback to 'process'.")
            backend = "process"

        if backend == "serial" or len(items) <= 1:
            iterator = tqdm(items, desc="Calculating factors") if progress_bar else items
            for factor_name, expr in iterator:
                name, result, raw_expr, err = _eval_one(factor_name, expr)
                if err:
                    if err.startswith("Warning:"):
                        print(err)
                    else:
                        print(err)
                        print(f"Expression: {raw_expr}")
                    continue
                results[name] = result
        else:
            if max_workers is None:
                max_workers = max(1, min(len(items), os.cpu_count() or 1))
            alpha_fields = {
                "open": self.alpha.open,
                "high": self.alpha.high,
                "low": self.alpha.low,
                "close": self.alpha.close,
                "volume": self.alpha.volume,
                "returns": self.alpha.returns,
                "vwap": self.alpha.vwap,
                "market_return": self.alpha.market_return,
                "funding": self.alpha.funding,
                "cap": self.alpha.cap,
            }
            pbar = tqdm(total=len(items), desc="Calculating factors (parallel)") if progress_bar else None
            with ProcessPoolExecutor(
                max_workers=max_workers,
                initializer=_init_fast_worker,
                initargs=(alpha_fields,),
            ) as executor:
                future_map = {executor.submit(_eval_one_worker, item): item for item in items}
                for future in as_completed(future_map):
                    name, raw_expr = future_map[future]
                    try:
                        name, result, raw_expr, err = future.result()
                    except Exception as e:
                        err = f"Error evaluating {name}: {str(e)}"
                        result = None
                    if err:
                        if err.startswith("Warning:"):
                            print(err)
                        else:
                            print(err)
                            print(f"Expression: {raw_expr}")
                    elif result is not None:
                        results[name] = result
                    if pbar is not None:
                        pbar.update(1)
            if pbar is not None:
                pbar.close()

        if not results:
            print("No factors were successfully evaluated")
            return pd.DataFrame()

        # 拼接所有结果
        concatenated_results = []
        for factor_name, factor_data in results.items():
            factor_data.columns = pd.MultiIndex.from_product(
                [[factor_name], factor_data.columns]
            )
            concatenated_results.append(factor_data)

        final_result = pd.concat(concatenated_results, axis=1)
        return final_result
    
    def evaluate_batch_with_backtest(self, expressions: Dict[str, str], 
                                     wide_data: pd.DataFrame,
                                     n_quintiles: int = 5,
                                     progress_bar: bool = True) -> pd.DataFrame:
        """
        批量计算因子并直接进行回测
        
        Args:
            expressions: 字典，键为因子名称，值为表达式字符串
            wide_data: 原始宽表数据（包含 open, high, low, close, volume 等）
            n_quintiles: 分层回测的分位数
            progress_bar: 是否显示进度条
            
        Returns:
            回测结果 DataFrame
        """
        # 计算所有因子
        factor_results = self.evaluate_batch(expressions, progress_bar=progress_bar)
        
        if factor_results.empty:
            print("No factors to backtest")
            return pd.DataFrame()
        
        # 拼接因子数据与原始数据
        combined_data = pd.concat([wide_data, factor_results], axis=1)
        
        # 跳过前一年预热期
        combined_data = combined_data.iloc[365:]
        
        # 执行回测
        factor_names = list(expressions.keys())
        from alpha101.futures_ml.alpha_sharpe import stratified_backtest
        backtest_results = stratified_backtest(
            combined_data, 
            factor_names, 
            target_col='target',
            n_quintiles=n_quintiles
        )
        
        return backtest_results
import argparse
def parse_args():
    parser = argparse.ArgumentParser(
        description='WorldQuant Alpha101 风格因子挖掘工具',
    )
    parser.add_argument(
        '-f', '--file',
        type=str,
        required=False,
    )
    parser.add_argument(
        'expression',
        type=str,
        nargs='?',  # 可选位置参数
        default=None,
        help='因子表达式（直接输入，无需 --expression）'
    )
    return parser.parse_args()        
if __name__ == "__main__":
    # 示例用法
    cfg = get_config()
    from alpha101.futures_ml.data import build_wide_df
    from alpha101.world_quant.Alpha101_code_1 import Alphas
    print(f'test start from {cfg.test_start_date} {cfg.test_end_date}')
    wide_data = build_wide_df(cfg.pairs, cfg.lookback_days, cfg.data_root, cfg.timeframe,
                              test_start_date=cfg.test_start_date, test_end_date=cfg.test_end_date, buffer=cfg.pre_buffer_candles)
    stock = Alphas(wide_data)  # 初始化股票数据
    engine = FastExpressionEngine(stock)
    args = parse_args()
    if args.file:
        with open(args.file, 'r') as f:
            fast_expression = f.read()
    else:
        fast_expression = args.expression
    
    if not fast_expression:
        print("Error: No expression provided")
        print("Usage: python fastengine.py <expression>")
        exit(1)
    # """
    # #    -( ts_mean(close, 5)+ts_mean(close, 10))
    # #    (ts_mean(returns, 7)+ts_mean(returns, 14))
    # # (((-1 * rank((open - ts_delay(high, 1)))) * rank((open - ts_delay(close, 1)))) * rank((open -ts_delay(low, 1))))
    # # -scale(ts_rank(close, 5))
    # # scale(ts_max(returns, 7)) - returns
    # # (ts_rank(((high) / (abs(close) + 0.001)) / (abs(scale(returns)) + 0.001), 10)) + ((close) + (ts_arg_max(ts_arg_max(ts_std_dev(open, 60), 30), 10)))
    # # (ts_std_dev(ts_delta((open) * (open), 5), 10)) + (((ts_std_dev(ts_delay(ts_mean(high, 20), 1), 10)) - (rank(rank((close) * (volume))))) - (ts_mean(ts_min(scale(close), 20), 30)))
    # # -ts_std_dev(ts_delay(ts_mean(high, 20), 1), 10)
    # # -(zscore(ts_mean(scale(high), 14)) - zscore((ts_std_dev(ts_delay(ts_mean(high, 20), 1), 10)) - zscore(rank((close) * (volume)))))
    # # -ts_std_dev(ts_delta(volume, 2), 30)
    # # (ts_std_dev(ts_delay(ts_mean(high, 20), 1), 10))   - (rank((close) * (volume)))
    # # -((ts_mean(scale(scale(high)), 14)) - ((ts_std_dev(ts_delay(ts_mean(high, 20), 1), 10)) - (rank(rank((close) * (volume))))))
    # # zscore(ts_mean(scale(high), 14)) - zscore(rank((close) * (volume)))
    # (ts_arg_max(ts_min(ts_std_dev(returns, 20), 5), 20))
    # """
    result = engine.evaluate(fast_expression)
    # wide_data['alpha_test'] = result
    print(result)

    # factor_df: index=date, columns=symbol
    from alpha101.world_quant.Alpha101_code_1 import process_factor_wide_format
    result.index.name = 'date'
    result.columns.name = 'symbol'
    result = process_factor_wide_format(result)
    # 构造与 panel 一致的列结构：(alpha_name, symbol)
    result.columns = pd.MultiIndex.from_product(
        [["alpha_test"], result.columns]
    )

    # 与原 panel 拼接
    # 一天多少seconds
    n_bars = int(cfg.pre_buffer_candles*pd.Timedelta('1d').total_seconds()//pd.Timedelta(cfg.timeframe).total_seconds())
    df = pd.concat([wide_data, result], axis=1).iloc[n_bars:]
    from alpha101.futures_ml.alpha_sharpe import stratified_backtest
    backtest_df = stratified_backtest(df, ['alpha_test'], k_bars=cfg.trade_per_k_bars,n_quintiles=5, freq=cfg.timeframe)
    # backtest_df = stratified_backtest(df, ['alpha_test'], k_bars=16, factor_agg='std')
    
    # ==================== 批量计算因子示例 ====================
    # expressions = {
    #     'factor_1': 'ts_rank(close, 10)',
    #     'factor_2': 'rank(ts_mean(close, 5) - ts_mean(close, 10))',
    #     'factor_3': 'ts_corr(close, volume, 20)',
    # }
    # 
    # # 方式1：仅计算因子，不回测
    # factor_results = engine.evaluate_batch(expressions)
    # print(factor_results.head())
    # 
    # # 方式2：计算因子并直接回测
    # backtest_results = engine.evaluate_batch_with_backtest(expressions, wide_data)
    # print(backtest_results)
