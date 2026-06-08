from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


class SearchResultStore:
    def __init__(self, output_dir: str | Path, timeframe: str):
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(output_dir) / timeframe / run_timestamp
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.all_results: list[dict] = []

    def save_batch(self, results: List[Dict], batch_name: str) -> None:
        results = [r for r in results if r]
        if not results:
            return
        csv_path = self.output_dir / f"{batch_name}.csv"
        json_path = self.output_dir / f"{batch_name}.json"

        rounded_results = [_round_floats(result) for result in results]
        _ordered_frame(rounded_results).to_csv(csv_path, index=False, float_format="%.3f")
        json_path.write_text(json.dumps(rounded_results, indent=2, default=str), encoding="utf-8")
        self.all_results.extend(results)

    def save_manifest(self, manifest: dict) -> Path:
        manifest_path = self.output_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
        return manifest_path

    def summarize(self, normalize_expression) -> None:
        if not self.all_results:
            print("No results to summarize")
            return
        df = pd.DataFrame(self.all_results)
        success_df = df[df["status"] == "success"].copy()
        if success_df.empty:
            print("No successful factors found")
            return

        success_df["expression_norm"] = success_df["expression"].map(normalize_expression)
        success_df = success_df.sort_values("fitness", ascending=False).drop_duplicates("expression_norm")
        summary_path = self.output_dir / "summary.csv"
        success_df = _ordered_frame(success_df.to_dict(orient="records"))
        success_df.to_csv(summary_path, index=False, float_format="%.3f")

        print(f"Total factors evaluated: {len(df)}")
        print(f"Successful factors: {len(success_df)}")
        print("Top factors:")
        cols = ["factor_name", "fitness", "sharpe_ratio", "ic_ir", "returns", "expression"]
        print(success_df[cols].head(10).to_string(index=False))
        print(f"Summary saved to {summary_path}")


FRONT_COLUMNS = [
    "factor_name",
    "expression",
    "status",
    "fitness",
    "test_fitness",
    "test_sharpe",
    "test_returns",
    "test_ic_ir",
    "test_obs_count",
    "valid_fitness",
    "valid_sharpe",
    "valid_returns",
    "validation_pass",
    "train_valid_sharpe_gap",
    "selection_fitness",
    "cv_pass",
    "cv_failure_penalty",
    "cv_time_pos_folds",
    "cv_time_median_sharpe",
    "cv_time_min_sharpe",
    "cv_universe_pos_groups",
    "cv_universe_min_sharpe",
    "cv_walk_forward_pos_folds",
    "cv_walk_forward_median_sharpe",
    "cv_walk_forward_min_sharpe",
    "cv_nq10_pass",
    "cv_nq10_time_pos_folds",
    "cv_nq10_time_median_sharpe",
    "cv_nq10_time_min_sharpe",
    "cv_nq10_universe_pos_groups",
    "cv_nq10_universe_min_sharpe",
    "cv_nq10_walk_forward_pos_folds",
    "cv_nq10_walk_forward_median_sharpe",
    "cv_nq10_walk_forward_min_sharpe",
    "train_fitness",
    "train_sharpe",
    "train_returns",
    "sharpe_ratio",
    "returns",
    "ic_ir",
    "turnover",
    "drawdown",
]


def _ordered_frame(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    front = [col for col in FRONT_COLUMNS if col in df.columns]
    rest = [col for col in df.columns if col not in front]
    return df[front + rest]


def _round_floats(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 3) if math.isfinite(value) else value
    if isinstance(value, dict):
        return {key: _round_floats(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_round_floats(item) for item in value]
    return value
