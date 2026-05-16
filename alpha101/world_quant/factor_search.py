"""
因子搜索主脚本
支持多种搜索策略：随机搜索、模板搜索、网格搜索、遗传算法
"""
import sys
from pathlib import Path
from datetime import datetime
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
from typing import List, Tuple, Dict, Optional
import traceback
import re

# 添加项目路径
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

from alpha101.world_quant.Alpha101_code_1 import Alphas
from alpha101.futures_ml.data import build_wide_df
from alpha101.futures_ml.config import get_config
from alpha101.futures_ml.alpha_sharpe import stratified_backtest
from alpha101.world_quant.factor_generator import (
    FactorGenerator,
    FactorLibrary,
    FieldNode,
    NumberNode,
    UnaryOpNode,
    BinaryOpNode,
    FunctionNode,
)
from alpha101.world_quant.fastengine import FastExpressionEngine


_SCORING_WORKER_CTX: dict = {}


def _safe_abs_corr_np(x: np.ndarray, y: np.ndarray) -> float:
    n = min(len(x), len(y))
    if n < 3:
        return 0.0
    x1 = np.asarray(x[-n:], dtype=float)
    y1 = np.asarray(y[-n:], dtype=float)
    mask = np.isfinite(x1) & np.isfinite(y1)
    if int(mask.sum()) < 3:
        return 0.0
    xv = x1[mask]
    yv = y1[mask]
    if np.std(xv) == 0 or np.std(yv) == 0:
        return 0.0
    corr = np.corrcoef(xv, yv)[0, 1]
    if not np.isfinite(corr):
        return 0.0
    return abs(float(corr))


def _get_preferred_mp_context():
    if not sys.platform.startswith("win"):
        try:
            methods = mp.get_all_start_methods()
        except Exception:
            methods = []
        if "fork" in methods:
            return mp.get_context("fork")
    return mp.get_context()


def _init_scoring_worker(ctx: dict) -> None:
    global _SCORING_WORKER_CTX
    _SCORING_WORKER_CTX = dict(ctx)
    _SCORING_WORKER_CTX["generator"] = FactorGenerator()


def _score_one_expr_worker(expr: str) -> Dict:
    ctx = _SCORING_WORKER_CTX
    agg_by_expr = ctx["agg_by_expr"]
    agg_metrics = agg_by_expr.get(expr)
    if agg_metrics is None:
        return {"expression": expr, "status": "failed", "error": "Missing aggregated metrics"}

    pnl_series = agg_metrics.get("pnl_series")
    if pnl_series is None or len(pnl_series) == 0:
        return {"expression": expr, "status": "failed", "error": "Empty pnl series"}

    corr_mode = ctx["corr_mode"]
    max_correlation = float(ctx["max_correlation"])
    generation_pnl_pool = ctx["generation_pnl_pool"]
    history_topk_pool = ctx["history_topk_pool"]
    all_pnl_pool = ctx["all_pnl_pool"]

    if corr_mode == "all":
        reference_sets = [all_pnl_pool.items()]
    elif corr_mode == "population":
        reference_sets = [generation_pnl_pool.items()]
    else:
        reference_sets = [generation_pnl_pool.items(), history_topk_pool.items()]

    max_corr = 0.0
    for ref_items in reference_sets:
        for cached_expr, cached_pnl in ref_items:
            if cached_expr == expr:
                continue
            c = _safe_abs_corr_np(pnl_series, cached_pnl)
            if c > max_corr:
                max_corr = c

    if max_corr > max_correlation:
        return {
            "expression": expr,
            "status": "failed",
            "error": f'Pnl correlation too high ({max_corr:.4f} > {max_correlation})',
            "corr": max_corr,
        }

    generator = ctx["generator"]
    complexity = generator.calculate_complexity(expr)
    if complexity['max_depth'] > ctx["depth_penalty_threshold"]:
        depth_penalty = np.exp(
            -max(0, complexity['complexity_score'] - 10) * ctx["depth_penalty_weight"]
        )
    else:
        depth_penalty = 1.0

    fitness = agg_metrics['robust_fitness'] * agg_metrics['stability_score'] * depth_penalty * 100
    return {
        "expression": expr,
        "status": "success",
        "corr": max_corr,
        "fitness": fitness,
        "complexity_score": complexity["complexity_score"],
        "expr_max_depth": complexity["max_depth"],
        "nan_ratio": float(ctx["nan_ratio_by_expr"].get(expr, 1.0)),
    }


class FactorSearchEngine:
    """因子搜索引擎"""
    
    def __init__(self, wide_data: pd.DataFrame, output_dir: str = "factor_search_results", timeframe: str = "1d"):
        self.wide_data = wide_data
        self.alpha_obj = Alphas(wide_data)
        self.engine = FastExpressionEngine(self.alpha_obj)
        self.generator = FactorGenerator()
        
        # 创建输出目录
        
        self.output_dir = Path(output_dir) / timeframe
        self.output_dir.mkdir(exist_ok=True)
        
        # 存储结果
        self.all_results = []
        
        # 性能筛选阈值
        self.min_sharpe = 0.8  # abs(sharpe) threshold
        self.min_return = 0.1  # abs(returns) threshold
        self.min_fitness_abs = 0.0  # abs(fitness) threshold
        self.max_correlation = 0.6  # 与已有因子的最大相关性

        # 表达式深度惩罚
        self.depth_penalty_threshold = 6
        self.depth_penalty_weight = 0.1

        self.max_complexity = 36.0

        # Global dedup and evaluation cache.
        self.evaluation_cache: Dict[str, Dict] = {}
        self.seen_expressions: set[str] = set()
        self.pnl_series_cache: Dict[str, np.ndarray] = {}
        self.verbose_eval = False
        self.eval_factor_alias = '__candidate__'
        cfg = get_config()
        self.eval_k_bars = max(1, int(getattr(cfg, "trade_per_k_bars", 1)))
        self.eval_freq = str(getattr(cfg, "daily_timeframe", "1d"))
        self.corr_mode = str(getattr(cfg, "corr_mode", "population_topk")).strip().lower()
        if self.corr_mode not in {"all", "population", "population_topk"}:
            print(f"Unknown corr_mode={self.corr_mode}, fallback to population_topk")
            self.corr_mode = "population_topk"
        self.corr_topk = max(1, int(getattr(cfg, "corr_topk", 500)))
        self.history_topk_pnl: Dict[str, Tuple[float, np.ndarray]] = {}
        self.scoring_workers = max(1, int(getattr(cfg, "scoring_workers", max(1, (mp.cpu_count() - 2)))))
        self.pre_buffer_candles = max(0, int(getattr(cfg, "pre_buffer_candles", 0)))
        self.eval_start_idx = min(self.pre_buffer_candles, len(self.wide_data))
        self.eval_wide_data = self.wide_data.iloc[self.eval_start_idx:]
        self.elite_similarity_threshold = 0.8
        self.segment_ratios = (0.8, 0.2)  # train / valid
        self.min_segment_size = 30
        self.require_all_segments_pass = True

    def _normalize_expression(self, expr: str) -> str:
        """Normalize expression for dedup/cache key usage."""
        try:
            root = self.generator._parse_expr_to_ast(expr)
            root = self.generator._sanitize_ast(root)
            root = self.generator._simplify_ast(root)
            return self.generator._ast_to_string(root)
        except Exception:
            return re.sub(r"\s+", "", expr)

    def _clone_metrics(self, metrics: Dict, factor_name: str, expression: str) -> Dict:
        out = dict(metrics)
        out["factor_name"] = factor_name
        out["expression"] = expression
        return out

    def _deduplicate_factor_pairs(self, factors: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        unique: List[Tuple[str, str]] = []
        local_seen: set[str] = set()
        for name, expr in factors:
            norm_expr = self._normalize_expression(expr)
            if norm_expr in local_seen or norm_expr in self.seen_expressions:
                continue
            local_seen.add(norm_expr)
            self.seen_expressions.add(norm_expr)
            unique.append((name, norm_expr))
        return unique

    def _new_random_expression(self, max_attempts: int = 50) -> str:
        for _ in range(max_attempts):
            expr = self._normalize_expression(self.generator.generate_random_factor())
            if expr not in self.seen_expressions:
                self.seen_expressions.add(expr)
                return expr
        expr = self._normalize_expression(self.generator.generate_random_factor())
        self.seen_expressions.add(expr)
        return expr

    def _expression_signature(self, expr: str) -> set[str]:
        try:
            root = self.generator._parse_expr_to_ast(expr)
            tokens: set[str] = set()

            def walk(node):
                if isinstance(node, FieldNode):
                    tokens.add(f"f:{node.name}")
                    return
                if isinstance(node, NumberNode):
                    tokens.add("n")
                    return
                if isinstance(node, UnaryOpNode):
                    tokens.add(f"u:{node.op}")
                    walk(node.operand)
                    return
                if isinstance(node, BinaryOpNode):
                    tokens.add(f"b:{node.op}")
                    walk(node.left)
                    walk(node.right)
                    return
                if isinstance(node, FunctionNode):
                    tokens.add(f"fn:{node.name}")
                    for arg in node.args:
                        walk(arg)
                    return

            walk(root)
            return tokens
        except Exception:
            return set(re.findall(r"[A-Za-z_]+", expr))

    def _expression_similarity(self, expr1: str, expr2: str) -> float:
        s1 = self._expression_signature(expr1)
        s2 = self._expression_signature(expr2)
        if not s1 and not s2:
            return 1.0
        union = len(s1 | s2)
        if union == 0:
            return 0.0
        return len(s1 & s2) / union

    def _safe_abs_corr(self, x: np.ndarray, y: np.ndarray) -> float:
        n = min(len(x), len(y))
        if n < 3:
            return 0.0
        x1 = np.asarray(x[-n:], dtype=float)
        y1 = np.asarray(y[-n:], dtype=float)
        mask = np.isfinite(x1) & np.isfinite(y1)
        if int(mask.sum()) < 3:
            return 0.0
        xv = x1[mask]
        yv = y1[mask]
        if np.std(xv) == 0 or np.std(yv) == 0:
            return 0.0
        corr = np.corrcoef(xv, yv)[0, 1]
        if not np.isfinite(corr):
            return 0.0
        return abs(float(corr))

    def _update_history_topk_pnl(self, expr: str, fitness: float, pnl_series: np.ndarray) -> None:
        if pnl_series is None or len(pnl_series) == 0:
            return
        current = self.history_topk_pnl.get(expr)
        if current is not None and fitness <= current[0]:
            return

        self.history_topk_pnl[expr] = (float(fitness), pnl_series)
        if len(self.history_topk_pnl) <= self.corr_topk:
            return

        worst_expr = min(self.history_topk_pnl.items(), key=lambda kv: kv[1][0])[0]
        if worst_expr != expr:
            self.history_topk_pnl.pop(worst_expr, None)
        elif len(self.history_topk_pnl) > self.corr_topk:
            # If current expr is still the worst and pool overflowed, drop it.
            self.history_topk_pnl.pop(expr, None)

    def _max_pnl_corr(
        self,
        expr: str,
        pnl_series: np.ndarray,
        generation_pnl_pool: Optional[Dict[str, np.ndarray]] = None,
    ) -> float:
        if pnl_series is None or len(pnl_series) == 0:
            return 0.0

        mode = self.corr_mode
        use_generation = generation_pnl_pool is not None and len(generation_pnl_pool) > 0
        max_corr = 0.0

        if mode == "all":
            reference_sets = [self.pnl_series_cache.items()]
        elif mode == "population":
            # Fallback to historical full cache for single-factor evaluation path.
            reference_sets = [generation_pnl_pool.items()] if use_generation else [self.pnl_series_cache.items()]
        else:
            topk_items = ((k, v[1]) for k, v in self.history_topk_pnl.items())
            reference_sets = [generation_pnl_pool.items(), topk_items] if use_generation else [topk_items]

        for ref_items in reference_sets:
            for cached_expr, cached_pnl in ref_items:
                if cached_expr == expr:
                    continue
                c = self._safe_abs_corr(pnl_series, cached_pnl)
                if c > max_corr:
                    max_corr = c
        return max_corr

    def _select_diverse_elites(self, population: List[Dict], elite_count: int) -> List[Dict]:
        if elite_count <= 0 or not population:
            return []

        selected: List[Dict] = [population[0]]
        for candidate in population[1:]:
            if len(selected) >= elite_count:
                break
            max_sim = max(
                self._expression_similarity(candidate['expression'], chosen['expression'])
                for chosen in selected
            )
            if max_sim < self.elite_similarity_threshold:
                selected.append(candidate)

        if len(selected) < elite_count:
            for candidate in population:
                if len(selected) >= elite_count:
                    break
                if candidate in selected:
                    continue
                selected.append(candidate)

        return selected[:elite_count]

    def _build_time_segments(self, n_rows: int) -> Optional[Dict[str, slice]]:
        if n_rows <= 0:
            return None
        r_train, r_valid = self.segment_ratios
        train_end = int(n_rows * r_train)
        train_end = min(max(train_end, self.min_segment_size), n_rows - self.min_segment_size)
        segments = {
            'train': slice(0, train_end),
            'valid': slice(train_end, n_rows),
        }
        lengths = {k: (v.stop - v.start) for k, v in segments.items()}
        if min(lengths.values()) < self.min_segment_size:
            return None
        return segments

    def _run_segmented_backtest(
        self,
        combined_data: pd.DataFrame,
        factor_aliases: List[str],
        k_bars: int,
        freq: str,
    ) -> Tuple[Optional[Dict[str, Dict[str, pd.Series]]], Optional[str]]:
        segments = self._build_time_segments(len(combined_data))
        if segments is None:
            return None, (
                f"Insufficient data for segmented evaluation: "
                f"need >= {self.min_segment_size} rows per segment"
            )

        segment_rows: Dict[str, Dict[str, pd.Series]] = {}
        for seg_name, seg_slice in segments.items():
            seg_df = combined_data.iloc[seg_slice]
            bt = stratified_backtest(
                seg_df,
                factor_aliases,
                k_bars=k_bars,
                freq=freq,
                verbose=False,
                return_pnl=True,
            )
            row_map: Dict[str, pd.Series] = {}
            if bt is not None and not bt.empty:
                row_map = {row['factor']: row for _, row in bt.iterrows()}
            segment_rows[seg_name] = row_map
        return segment_rows, None

    def _extract_row_metrics(self, row: pd.Series) -> Dict[str, float]:
        return {
            'fitness': float(row.get('fitness', 0.0)),
            'returns': float(row.get('returns').replace("%", "")) if isinstance(row.get('returns'), str) else float(row.get('returns', 0.0)),
            'sharpe_ratio': float(row.get('sharpe', 0.0)),
            'ic_ir': float(row.get('ic_ir', 0.0)),
            'ic_mean': float(row.get('ic_mean', 0.0)),
            'ic_std': float(row.get('ic_std', 0.0)),
            'drawdown': float(row.get('drawdown').replace("%", "")) if isinstance(row.get('drawdown'), str) else float(row.get('drawdown', 0.0)),
            'turnover': float(row.get('turnover').replace("%", "")) if isinstance(row.get('turnover'), str) else float(row.get('turnover', 0.0)),
            'win_rate': float(row.get('win_rate', 0.0)),
        }

    def _segment_passes(self, metrics: Dict[str, float]) -> bool:
        if not np.isfinite(metrics['fitness']) or not np.isfinite(metrics['sharpe_ratio']) or not np.isfinite(metrics['returns']):
            return False
        if abs(metrics['fitness']) < self.min_fitness_abs:
            return False
        if abs(metrics['sharpe_ratio']) < self.min_sharpe:
            return False
        if abs(metrics['returns']) < self.min_return:
            return False
        return True

    def _aggregate_segment_metrics(
        self,
        alias: str,
        segment_rows: Dict[str, Dict[str, pd.Series]],
    ) -> Tuple[Optional[Dict], Optional[str]]:
        per_segment: Dict[str, Dict[str, float]] = {}
        pnl_parts: List[np.ndarray] = []
        for seg_name in ('train', 'valid'):
            row_map = segment_rows.get(seg_name, {})
            row = row_map.get(alias)
            if row is None:
                return None, f"Missing backtest row for {seg_name}"
            seg_metrics = self._extract_row_metrics(row)
            per_segment[seg_name] = seg_metrics
            pnl_raw = row.get('pnl', None)
            if pnl_raw is None:
                return None, f"Missing pnl series for {seg_name}"
            pnl_arr = np.asarray(pnl_raw, dtype=float)
            if pnl_arr.size == 0:
                return None, f"Empty pnl series for {seg_name}"
            pnl_parts.append(pnl_arr)
            if self.require_all_segments_pass and not self._segment_passes(seg_metrics):
                return None, (
                    f"Segment threshold not met on {seg_name}: "
                    f"|fitness|={abs(seg_metrics['fitness']):.6f}, "
                    f"|sharpe|={abs(seg_metrics['sharpe_ratio']):.6f}, "
                    f"|returns|={abs(seg_metrics['returns']):.6f}"
                )

        values = list(per_segment.values())
        pnl_series = np.concatenate(pnl_parts) if pnl_parts else np.array([], dtype=float)
        abs_fit = [abs(v['fitness']) for v in values]
        max_abs_fit = max(abs_fit) if abs_fit else 0.0
        min_abs_fit = min(abs_fit) if abs_fit else 0.0
        stability_score = float(min_abs_fit / max_abs_fit) if max_abs_fit > 0 else 0.0
        agg = {
            'robust_fitness': min(v['fitness'] for v in values),
            'mean_fitness': float(np.mean([v['fitness'] for v in values])),
            'stability_score': stability_score,
            'returns': float(np.mean([v['returns'] for v in values])),
            'sharpe_ratio': float(np.mean([v['sharpe_ratio'] for v in values])),
            'ic_ir': float(np.mean([v['ic_ir'] for v in values])),
            'ic_mean': float(np.mean([v['ic_mean'] for v in values])),
            'ic_std': float(np.mean([v['ic_std'] for v in values])),
            'drawdown': float(max(v['drawdown'] for v in values)),
            'turnover': float(np.mean([v['turnover'] for v in values])),
            'win_rate': float(np.mean([v['win_rate'] for v in values])),
            'segment_metrics': per_segment,
            'pnl_series': pnl_series,
        }
        return agg, None
        
    def evaluate_factor(self, factor_name: str, factor_expr: str) -> Dict:
        """
        Evaluate one factor expression and return metrics.
        """
        try:
            factor_expr = self._normalize_expression(factor_expr)

            cached = self.evaluation_cache.get(factor_expr)
            if cached is not None:
                return self._clone_metrics(cached, factor_name, factor_expr)

            is_valid, reason = self.generator.is_semantically_valid(factor_expr)
            if not is_valid:
                failed = {
                    'factor_name': factor_name,
                    'expression': factor_expr,
                    'status': 'failed',
                    'error': f'Semantic invalid: {reason}'
                }
                self.evaluation_cache[factor_expr] = failed
                return failed

            if not self.generator.filter_by_complexity(factor_expr, max_complexity=self.max_complexity):
                failed = {
                    'factor_name': factor_name,
                    'expression': factor_expr,
                    'status': 'failed',
                    'error': f'Complexity too high (> {self.max_complexity})'
                }
                self.evaluation_cache[factor_expr] = failed
                return failed

            if self.verbose_eval:
                print(f"Evaluating factor: {factor_name}")
                print(f"Expression: {factor_expr}")
            result = self.engine.evaluate(factor_expr)

            if result is None or result.isnull().all().all():
                failed = {
                    'factor_name': factor_name,
                    'expression': factor_expr,
                    'status': 'failed',
                    'error': 'All NaN result'
                }
                self.evaluation_cache[factor_expr] = failed
                return failed

            result.index.name = 'date'
            result.columns.name = 'symbol'

            result_formatted = result
            result_formatted.columns = pd.MultiIndex.from_product([[self.eval_factor_alias], result.columns])

            test_df = pd.concat([self.eval_wide_data, result_formatted], axis=1, copy=False)

            segment_rows, segment_error = self._run_segmented_backtest(
                test_df,
                [self.eval_factor_alias],
                k_bars=self.eval_k_bars,
                freq=self.eval_freq,
            )
            if segment_rows is None:
                failed = {
                    'factor_name': factor_name,
                    'expression': factor_expr,
                    'status': 'failed',
                    'error': segment_error or 'Segmented backtest failed'
                }
                self.evaluation_cache[factor_expr] = failed
                return failed

            agg_metrics, agg_error = self._aggregate_segment_metrics(self.eval_factor_alias, segment_rows)
            if agg_metrics is None:
                failed = {
                    'factor_name': factor_name,
                    'expression': factor_expr,
                    'status': 'failed',
                    'error': agg_error or 'Segment threshold not met'
                }
                self.evaluation_cache[factor_expr] = failed
                return failed

            corr = self._max_pnl_corr(factor_expr, agg_metrics['pnl_series'])
            if corr > self.max_correlation:
                failed = {
                    'factor_name': factor_name,
                    'expression': factor_expr,
                    'status': 'failed',
                    'error': f'Pnl correlation too high ({corr:.4f} > {self.max_correlation})'
                }
                self.evaluation_cache[factor_expr] = failed
                return failed

            complexity = self.generator.calculate_complexity(factor_expr)
            if complexity['max_depth'] > self.depth_penalty_threshold:
                depth_penalty = np.exp(-max(0, complexity['complexity_score'] - 10) * self.depth_penalty_weight)
            else:
                depth_penalty = 1.0
            fitness = agg_metrics['robust_fitness'] * agg_metrics['stability_score'] * depth_penalty * 100

            metrics = {
                'factor_name': factor_name,
                'expression': factor_expr,
                'status': 'success',
                'fitness': fitness,
                'corr': corr,
                'robust_fitness': agg_metrics['robust_fitness'],
                'mean_segment_fitness': agg_metrics['mean_fitness'],
                'stability_score': agg_metrics['stability_score'],
                'returns': agg_metrics['returns'],
                'sharpe_ratio': agg_metrics['sharpe_ratio'],
                'ic_ir': agg_metrics['ic_ir'],
                'ic_mean': agg_metrics['ic_mean'],
                'ic_std': agg_metrics['ic_std'],
                'drawdown': agg_metrics['drawdown'],
                'turnover': agg_metrics['turnover'],
                'win_rate': agg_metrics['win_rate'],
                'complexity_score': complexity['complexity_score'],
                'expr_max_depth': complexity['max_depth'],
                'nan_ratio': result.isnull().sum().sum() / (result.shape[0] * result.shape[1]),
                'train_fitness': agg_metrics['segment_metrics']['train']['fitness'],
                'valid_fitness': agg_metrics['segment_metrics']['valid']['fitness'],
                'train_sharpe': agg_metrics['segment_metrics']['train']['sharpe_ratio'],
                'valid_sharpe': agg_metrics['segment_metrics']['valid']['sharpe_ratio'],
                'timestamp': datetime.now().isoformat()
            }
            self.evaluation_cache[factor_expr] = metrics
            self.pnl_series_cache[factor_expr] = agg_metrics['pnl_series']
            self._update_history_topk_pnl(factor_expr, fitness, agg_metrics['pnl_series'])
            return metrics

        except Exception as e:
            failed = {
                'factor_name': factor_name,
                'expression': factor_expr,
                'status': 'failed',
                'error': str(e),
                'traceback': traceback.format_exc()
            }
            self.evaluation_cache[self._normalize_expression(factor_expr)] = failed
            return failed


    def _evaluate_generation_batch(self, population: List[Dict], gen: int, k_bars: int = 1, freq: str = '1d') -> None:
        """Evaluate one GA generation in batch backtest mode."""
        t0 = time.perf_counter()
        pending_indices = [i for i, ind in enumerate(population) if ind['fitness'] is None]
        if not pending_indices:
            return

        expr_to_indices: Dict[str, List[int]] = {}

        # Fast path: cache + semantic/complexity filters.
        for idx in pending_indices:
            individual = population[idx]
            expr = self._normalize_expression(individual['expression'])
            individual['expression'] = expr
            factor_name = f"gen{gen}_individual_{idx:04d}"

            cached = self.evaluation_cache.get(expr)
            if cached is not None:
                metrics = self._clone_metrics(cached, factor_name, expr)
                individual['metrics'] = metrics
                individual['fitness'] = metrics['fitness'] if metrics.get('status') == 'success' else -999
                continue

            is_valid, reason = self.generator.is_semantically_valid(expr)
            if not is_valid:
                failed = {
                    'factor_name': factor_name,
                    'expression': expr,
                    'status': 'failed',
                    'error': f'Semantic invalid: {reason}'
                }
                self.evaluation_cache[expr] = failed
                individual['metrics'] = failed
                individual['fitness'] = -999
                continue

            if not self.generator.filter_by_complexity(expr, max_complexity=self.max_complexity):
                failed = {
                    'factor_name': factor_name,
                    'expression': expr,
                    'status': 'failed',
                    'error': f'Complexity too high (> {self.max_complexity})'
                }
                self.evaluation_cache[expr] = failed
                individual['metrics'] = failed
                individual['fitness'] = -999
                continue

            expr_to_indices.setdefault(expr, []).append(idx)

        if not expr_to_indices:
            return

        alias_to_expr: Dict[str, str] = {}
        eval_expressions: Dict[str, str] = {}
        for i, expr in enumerate(expr_to_indices.keys()):
            alias = f"__gen{gen}_cand_{i:04d}"
            alias_to_expr[alias] = expr
            eval_expressions[alias] = expr

        try:
            t_eval_start = time.perf_counter()
            factor_panel = self.engine.evaluate_batch(eval_expressions, progress_bar=True)
            t_eval = time.perf_counter() - t_eval_start
            print(f"[GA gen {gen + 1}] factor calc finished: {len(eval_expressions)} factors, {t_eval:.2f}s")
        except Exception as e:
            factor_panel = pd.DataFrame()

        if factor_panel.empty:
            for expr, indices in expr_to_indices.items():
                base_failed = {
                    'factor_name': '__batch_eval__',
                    'expression': expr,
                    'status': 'failed',
                    'error': 'Batch factor evaluation failed or returned empty'
                }
                self.evaluation_cache[expr] = base_failed
                for idx in indices:
                    factor_name = f"gen{gen}_individual_{idx:04d}"
                    failed = self._clone_metrics(base_failed, factor_name, expr)
                    population[idx]['metrics'] = failed
                    population[idx]['fitness'] = -999
            return

        factor_panel = factor_panel.iloc[self.eval_start_idx:]
        combined_data = pd.concat([self.eval_wide_data, factor_panel], axis=1, copy=False)
        t_bt_start = time.perf_counter()
        segment_rows, segment_error = self._run_segmented_backtest(
            combined_data,
            list(eval_expressions.keys()),
            k_bars=k_bars,
            freq=freq,
        )
        t_bt = time.perf_counter() - t_bt_start
        print(f"[GA gen {gen + 1}] segmented backtest finished: {len(eval_expressions)} factors, {t_bt:.2f}s")
        if segment_rows is None:
            for expr, indices in expr_to_indices.items():
                base_failed = {
                    'factor_name': '__batch_backtest__',
                    'expression': expr,
                    'status': 'failed',
                    'error': segment_error or 'Segmented backtest failed'
                }
                self.evaluation_cache[expr] = base_failed
                for idx in indices:
                    factor_name = f"gen{gen}_individual_{idx:04d}"
                    failed = self._clone_metrics(base_failed, factor_name, expr)
                    population[idx]['metrics'] = failed
                    population[idx]['fitness'] = -999
            return

        t_post_start = time.perf_counter()
        aggregated_ok: Dict[str, Dict] = {}
        aggregate_iter = tqdm(
            alias_to_expr.items(),
            desc=f"Aggregating gen {gen + 1}",
            total=len(alias_to_expr),
        )
        for alias, expr in aggregate_iter:
            agg_metrics, agg_error = self._aggregate_segment_metrics(alias, segment_rows)
            if agg_metrics is None:
                aggregated_ok[alias] = {'expr': expr, 'agg_error': agg_error or 'Segment threshold not met'}
            else:
                aggregated_ok[alias] = {'expr': expr, 'agg_metrics': agg_metrics}

        generation_pnl_pool: Dict[str, np.ndarray] = {}
        for item in aggregated_ok.values():
            agg_metrics = item.get('agg_metrics')
            if agg_metrics is None:
                continue
            generation_pnl_pool[item['expr']] = agg_metrics['pnl_series']
        print(
            f"[GA gen {gen + 1}] corr reference: mode={self.corr_mode}, "
            f"population_pool={len(generation_pnl_pool)}, "
            f"history_topk={len(self.history_topk_pnl)}/{self.corr_topk}"
        )

        agg_failed_by_expr: Dict[str, str] = {}
        agg_by_expr: Dict[str, Dict] = {}
        nan_ratio_by_expr: Dict[str, float] = {}
        for alias, item in aggregated_ok.items():
            expr = item['expr']
            agg_metrics = item.get('agg_metrics')
            if agg_metrics is None:
                agg_failed_by_expr[expr] = item.get('agg_error', 'Segment threshold not met')
                continue
            agg_by_expr[expr] = agg_metrics
            try:
                factor_values = factor_panel[alias].to_numpy(dtype=float, copy=False)
                nan_ratio_by_expr[expr] = float(np.isnan(factor_values).sum() / factor_values.size)
            except Exception:
                nan_ratio_by_expr[expr] = 1.0

        history_topk_pool = {k: v[1] for k, v in self.history_topk_pnl.items()}
        all_pnl_pool = self.pnl_series_cache if self.corr_mode == "all" else {}
        scoring_ctx = {
            "corr_mode": self.corr_mode,
            "max_correlation": self.max_correlation,
            "depth_penalty_threshold": self.depth_penalty_threshold,
            "depth_penalty_weight": self.depth_penalty_weight,
            "generation_pnl_pool": generation_pnl_pool,
            "history_topk_pool": history_topk_pool,
            "all_pnl_pool": all_pnl_pool,
            "agg_by_expr": agg_by_expr,
            "nan_ratio_by_expr": nan_ratio_by_expr,
        }

        score_result_by_expr: Dict[str, Dict] = {}
        score_exprs = list(agg_by_expr.keys())
        if score_exprs:
            worker_count = max(1, min(len(score_exprs), self.scoring_workers))
            mp_ctx = _get_preferred_mp_context()
            try:
                with ProcessPoolExecutor(
                    max_workers=worker_count,
                    mp_context=mp_ctx,
                    initializer=_init_scoring_worker,
                    initargs=(scoring_ctx,),
                ) as executor:
                    chunksize = max(1, len(score_exprs) // max(1, worker_count * 2))
                    pbar = tqdm(total=len(score_exprs), desc=f"Scoring gen {gen + 1} (parallel)")
                    for score_result in executor.map(_score_one_expr_worker, score_exprs, chunksize=chunksize):
                        score_result_by_expr[score_result["expression"]] = score_result
                        pbar.update(1)
                    pbar.close()
            except Exception as e:
                print(f"[GA gen {gen + 1}] parallel scoring failed: {e}. Fallback to serial scoring.")
                _init_scoring_worker(scoring_ctx)
                pbar = tqdm(total=len(score_exprs), desc=f"Scoring gen {gen + 1} (serial)")
                for expr in score_exprs:
                    score_result = _score_one_expr_worker(expr)
                    score_result_by_expr[score_result["expression"]] = score_result
                    pbar.update(1)
                pbar.close()

        for alias, expr in alias_to_expr.items():
            indices = expr_to_indices[expr]
            agg_error = agg_failed_by_expr.get(expr)
            if agg_error is not None:
                base_failed = {
                    'factor_name': '__batch_backtest__',
                    'expression': expr,
                    'status': 'failed',
                    'error': agg_error,
                }
                self.evaluation_cache[expr] = base_failed
                for idx in indices:
                    factor_name = f"gen{gen}_individual_{idx:04d}"
                    failed = self._clone_metrics(base_failed, factor_name, expr)
                    population[idx]['metrics'] = failed
                    population[idx]['fitness'] = -999
                continue

            score_result = score_result_by_expr.get(expr)
            if score_result is None:
                base_failed = {
                    'factor_name': '__batch_backtest__',
                    'expression': expr,
                    'status': 'failed',
                    'error': 'Missing scoring result'
                }
                self.evaluation_cache[expr] = base_failed
                for idx in indices:
                    factor_name = f"gen{gen}_individual_{idx:04d}"
                    failed = self._clone_metrics(base_failed, factor_name, expr)
                    population[idx]['metrics'] = failed
                    population[idx]['fitness'] = -999
                continue

            if score_result.get("status") != "success":
                base_failed = {
                    'factor_name': '__batch_backtest__',
                    'expression': expr,
                    'status': 'failed',
                    'error': score_result.get('error', 'Scoring failed')
                }
                self.evaluation_cache[expr] = base_failed
                for idx in indices:
                    factor_name = f"gen{gen}_individual_{idx:04d}"
                    failed = self._clone_metrics(base_failed, factor_name, expr)
                    population[idx]['metrics'] = failed
                    population[idx]['fitness'] = -999
                continue

            agg_metrics = agg_by_expr[expr]
            fitness = float(score_result['fitness'])
            base_metrics = {
                'factor_name': '__batch_success__',
                'expression': expr,
                'status': 'success',
                'fitness': fitness,
                'corr': float(score_result['corr']),
                'robust_fitness': agg_metrics['robust_fitness'],
                'mean_segment_fitness': agg_metrics['mean_fitness'],
                'stability_score': agg_metrics['stability_score'],
                'returns': agg_metrics['returns'],
                'sharpe_ratio': agg_metrics['sharpe_ratio'],
                'ic_ir': agg_metrics['ic_ir'],
                'ic_mean': agg_metrics['ic_mean'],
                'ic_std': agg_metrics['ic_std'],
                'drawdown': agg_metrics['drawdown'],
                'turnover': agg_metrics['turnover'],
                'win_rate': agg_metrics['win_rate'],
                'complexity_score': float(score_result['complexity_score']),
                'expr_max_depth': int(score_result['expr_max_depth']),
                'nan_ratio': float(score_result['nan_ratio']),
                'train_fitness': agg_metrics['segment_metrics']['train']['fitness'],
                'valid_fitness': agg_metrics['segment_metrics']['valid']['fitness'],
                'train_sharpe': agg_metrics['segment_metrics']['train']['sharpe_ratio'],
                'valid_sharpe': agg_metrics['segment_metrics']['valid']['sharpe_ratio'],
                'timestamp': datetime.now().isoformat(),
            }
            self.evaluation_cache[expr] = base_metrics
            self.pnl_series_cache[expr] = agg_metrics['pnl_series']
            self._update_history_topk_pnl(expr, fitness, agg_metrics['pnl_series'])

            for idx in indices:
                factor_name = f"gen{gen}_individual_{idx:04d}"
                metrics = self._clone_metrics(base_metrics, factor_name, expr)
                population[idx]['metrics'] = metrics
                population[idx]['fitness'] = metrics['fitness']
        t_post = time.perf_counter() - t_post_start
        t_total = time.perf_counter() - t0
        print(
            f"[GA gen {gen + 1}] post-processing finished: {len(alias_to_expr)} factors, {t_post:.2f}s "
            f"(total {t_total:.2f}s)"
        )

    def random_search(self, n_factors: int = 1000, batch_size: int = 100):
        """
        ?????????

        Args:
            n_factors: ?????????
            batch_size: ????????????
        """
        print(f"\n{'='*80}")
        print(f"Random Search: Generating {n_factors} random factors")
        print(f"{'='*80}\n")

        results = []
        successful_count = 0

        for i in tqdm(range(n_factors), desc="Random Search"):
            factor_name = f"random_{i:04d}"
            factor_expr = self._new_random_expression()

            metrics = self.evaluate_factor(factor_name, factor_expr)
            results.append(metrics)

            if metrics['status'] == 'success':
                successful_count += 1

            if (i + 1) % batch_size == 0:
                self._save_batch_results(results, f"random_batch_{i+1}")
                results = []

        if results:
            self._save_batch_results(results, f"random_batch_final")

        print(f"\nRandom Search Complete: {successful_count}/{n_factors} factors succeeded")
        return successful_count

    def template_search(self, n_templates: int = None, batch_size: int = 10):
        """
        ?????????

        Args:
            n_templates: ????????????None???????????
            batch_size: ????????????
        """
        print(f"\n{'='*80}")
        print(f"Template Search: Generating factors from templates")
        print(f"{'='*80}\n")

        templates = FactorLibrary.ALPHA101_PATTERNS
        if n_templates is not None:
            templates = templates[:n_templates]

        template_factors = self.generator.generate_template_based_factors(templates)
        template_factors = self._deduplicate_factor_pairs(template_factors)

        print(f"Generated {len(template_factors)} factors from {len(templates)} templates\n")

        results = []
        successful_count = 0

        for i, (factor_name, factor_expr) in enumerate(tqdm(template_factors, desc="Template Search")):
            metrics = self.evaluate_factor(factor_name, factor_expr)
            results.append(metrics)

            if metrics['status'] == 'success':
                successful_count += 1

            if (i + 1) % batch_size == 0:
                self._save_batch_results(results, f"template_batch_{i+1}")
                results = []

        if results:
            self._save_batch_results(results, f"template_batch_final")

        print(f"\nTemplate Search Complete: {successful_count}/{len(template_factors)} factors succeeded")
        return successful_count

    def grid_search(self, batch_size: int = 10):
        """
        ?????????

        Args:
            batch_size: ????????????
        """
        print(f"\n{'='*80}")
        print(f"Grid Search: Systematic parameter combinations")
        print(f"{'='*80}\n")

        grid_factors = self.generator.generate_grid_search_factors()
        grid_factors = self._deduplicate_factor_pairs(grid_factors)

        print(f"Generated {len(grid_factors)} factors from grid search\n")

        results = []
        successful_count = 0

        for i, (factor_name, factor_expr) in enumerate(tqdm(grid_factors, desc="Grid Search")):
            metrics = self.evaluate_factor(factor_name, factor_expr)
            results.append(metrics)

            if metrics['status'] == 'success':
                successful_count += 1

            if (i + 1) % batch_size == 0:
                self._save_batch_results(results, f"grid_batch_{i+1}")
                results = []

        if results:
            self._save_batch_results(results, f"grid_batch_final")

        print(f"\nGrid Search Complete: {successful_count}/{len(grid_factors)} factors succeeded")
        return successful_count

    def genetic_search(self, population_size: int = 50, n_generations: int = 10,
                      mutation_rate: float = 0.3, crossover_rate: float = 0.5):
        """
        ?????????

        Args:
            population_size: ??????
            n_generations: ??????
            mutation_rate: ??????
            crossover_rate: ??????
        """
        print(f"\n{'='*80}")
        print(f"Genetic Algorithm Search")
        print(f"Population: {population_size}, Generations: {n_generations}")
        print(f"{'='*80}\n")

        population = []
        for _ in range(population_size):
            expr = self._new_random_expression()
            population.append({'expression': expr, 'fitness': None, 'metrics': None})

        best_overall = None

        for gen in range(n_generations):
            print(f"\nGeneration {gen + 1}/{n_generations}")

            # Batch evaluate all unevaluated individuals in this generation.
            self._evaluate_generation_batch(population, gen, k_bars=self.eval_k_bars, freq=self.eval_freq)

            population.sort(key=lambda x: x['fitness'], reverse=True)

            if best_overall is None or population[0]['fitness'] > best_overall['fitness']:
                best_overall = population[0].copy()

            print(f"Best fitness: {population[0]['fitness']:.4f}")
            print(f"Best expression: {population[0]['expression']}...")

            gen_results = [ind['metrics'] for ind in population if ind['metrics']]
            self._save_batch_results(gen_results, f"genetic_gen_{gen+1}")

            new_population = []
            elite_count = max(1, population_size // 5)
            elites = self._select_diverse_elites(population, elite_count)
            new_population.extend(elites)
            next_seen = {ind['expression'] for ind in new_population}

            while len(new_population) < population_size:
                tournament_size = max(3, int(population_size * 0.05))
                parent1 = self._tournament_select(population, tournament_size)
                parent2 = self._tournament_select(population, tournament_size)

                if np.random.random() < crossover_rate:
                    child_expr = self.generator.crossover_expressions(parent1['expression'], parent2['expression'])
                else:
                    child_expr = parent1['expression']

                if np.random.random() < mutation_rate:
                    child_expr = self.generator.mutate_expression(child_expr)

                child_expr = self._normalize_expression(child_expr)
                if child_expr in next_seen:
                    retries = 0
                    while retries < 10 and child_expr in next_seen:
                        child_expr = self._normalize_expression(self.generator.mutate_expression(child_expr))
                        retries += 1
                    if child_expr in next_seen:
                        child_expr = self._new_random_expression()

                self.seen_expressions.add(child_expr)
                next_seen.add(child_expr)
                new_population.append({'expression': child_expr, 'fitness': None, 'metrics': None})

            population = new_population[:population_size]

        if best_overall:
            self._save_batch_results([best_overall['metrics']], "genetic_best_overall")
            print(f"\n{'='*80}")
            print(f"Genetic Search Complete")
            print(f"Best Overall Fitness: {best_overall['fitness']:.4f}")
            print(f"Best Expression: {best_overall['expression']}")
            print(f"{'='*80}\n")

        return best_overall

    def _tournament_select(self, population: List[Dict], tournament_size: int = 3) -> Dict:
        """????????"""
        tournament_size = max(2, tournament_size)
        tournament = np.random.choice(population, size=min(tournament_size, len(population)), replace=False)
        return max(tournament, key=lambda x: x['fitness'])

    def _save_batch_results(self, results: List[Dict], batch_name: str):
        """保存批次结果"""
        if not results:
            return
        
        # 保存为CSV
        df = pd.DataFrame(results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = self.output_dir / f"{batch_name}_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        
        # 同时保存为JSON（保留完整信息）
        json_path = self.output_dir / f"{batch_name}_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # 添加到全局结果
        self.all_results.extend(results)
    
    def summarize_results(self):
        """汇总所有搜索结果"""
        if not self.all_results:
            print("No results to summarize")
            return
        
        df = pd.DataFrame(self.all_results)
        
        # 筛选成功的因子
        success_df = df[df['status'] == 'success'].copy()
        
        if success_df.empty:
            print("No successful factors found")
            return

        # 去重相同表达式：按归一化表达式保留 fitness 最好的一条
        success_df["expression_norm"] = success_df["expression"].map(self._normalize_expression)
        success_df = (
            success_df.sort_values("fitness", ascending=False)
            .drop_duplicates(subset=["expression_norm"], keep="first")
            .reset_index(drop=True)
        )
        
        print(f"\n{'='*80}")
        print(f"FACTOR SEARCH SUMMARY")
        print(f"{'='*80}\n")
        
        print(f"Total factors evaluated: {len(df)}")
        print(f"Successful factors: {len(success_df)} ({len(success_df)/len(df)*100:.1f}%)")
        print(f"Failed factors: {len(df) - len(success_df)}\n")
        
        # Top factors by Sharpe ratio
        print("Top 10 Factors by Sharpe Ratio:")
        print("-" * 80)
        top_sharpe = success_df.nlargest(10, 'sharpe_ratio')
        for idx, row in top_sharpe.iterrows():
            print(f"{row['factor_name']:<20} Sharpe: {row['sharpe_ratio']:>8.4f}  Return: {row['returns']:<10}")
            print(f"  Expression: {row['expression']}")


        # Top factors by fitness
        print("\nTop 10 Factors by Fitness:")
        print("-" * 80)
        top_fitness = success_df.nlargest(10, 'fitness')
        for idx, row in top_fitness.iterrows():
            print(f"{row['factor_name']:<20} Fitness: {row['fitness']:>8.2f}  Sharpe: {row['sharpe_ratio']:>8.4f}")
            print(f"  Expression: {row['expression']}")
        print("Bottom 10 Factors by Sharpe Ratio:")
        print("-" * 80)
        bottom_sharpe = success_df.nsmallest(10, 'sharpe_ratio')
        for idx, row in bottom_sharpe.iterrows():
            print(f"{row['factor_name']:<20} Sharpe: {row['sharpe_ratio']:>8.2f}  Return: {row['returns']:<10}")
            print(f"  Expression: {row['expression']}")
        print("\nBottom 10 Factors by Fitness:")
        print("-" * 80)
        bottom_fitness = success_df.nsmallest(10, 'fitness')
        for idx, row in bottom_fitness.iterrows():
            print(f"{row['factor_name']:<20} Fitness: {row['fitness']:>8.2f}  Sharpe: {row['sharpe_ratio']:>8.2f}")
            print(f"  Expression: {row['expression']}")
        # 保存汇总结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = self.output_dir / f"summary_{timestamp}.csv"
        success_df.to_csv(summary_path, index=False)
        
        # 保存Top因子
        top_factors_path = self.output_dir / f"top_factors_{timestamp}.csv"
        top_combined = pd.concat([
            success_df.nlargest(20, 'sharpe_ratio'),
            success_df.nlargest(20, 'fitness')
        ]).drop_duplicates(subset=['expression_norm'])
        top_combined.to_csv(top_factors_path, index=False)
        
        print(f"\nResults saved to:")
        print(f"  Summary: {summary_path}")
        print(f"  Top factors: {top_factors_path}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Factor Search Engine')
    parser.add_argument('--strategy', type=str, default='random', 
                       choices=['random', 'template', 'grid', 'genetic', 'all'],
                       help='Search strategy')
    parser.add_argument('--n-factors', type=int, default=100,
                       help='Number of factors for random search')
    parser.add_argument('--population', type=int, default=30,
                       help='Population size for genetic algorithm')
    parser.add_argument('--generations', type=int, default=5,
                       help='Number of generations for genetic algorithm')
    parser.add_argument('--output-dir', type=str, default='factor_search_results',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # 加载数据
    print("Loading data...")
    cfg = get_config()
    wide_data = build_wide_df(
        cfg.pairs, 
        cfg.lookback_days, 
        cfg.data_root, 
        cfg.timeframe,
        test_start_date=cfg.test_start_date, 
        test_end_date=cfg.test_end_date,
        buffer=7
    )
    print(f"Data loaded: {wide_data.shape}")
    
    # 创建搜索引擎
    search_engine = FactorSearchEngine(wide_data, output_dir=args.output_dir, timeframe=cfg.timeframe)
    
    # 执行搜索
    if args.strategy == 'random':
        search_engine.random_search(n_factors=args.n_factors)
    elif args.strategy == 'template':
        search_engine.template_search()
    elif args.strategy == 'grid':
        search_engine.grid_search()
    elif args.strategy == 'genetic':
        search_engine.genetic_search(
            population_size=args.population,
            n_generations=args.generations
        )
    elif args.strategy == 'all':
        search_engine.random_search(n_factors=args.n_factors)
        search_engine.template_search()
        search_engine.grid_search()
    
    # 汇总结果
    search_engine.summarize_results()


if __name__ == '__main__':
    main()
