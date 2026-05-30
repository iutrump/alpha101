from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd


class SearchResultStore:
    def __init__(self, output_dir: str | Path, timeframe: str):
        self.output_dir = Path(output_dir) / timeframe
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.all_results: list[dict] = []

    def save_batch(self, results: List[Dict], batch_name: str) -> None:
        results = [r for r in results if r]
        if not results:
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = self.output_dir / f"{batch_name}_{timestamp}.csv"
        json_path = self.output_dir / f"{batch_name}_{timestamp}.json"
        pd.DataFrame(results).to_csv(csv_path, index=False)
        json_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
        self.all_results.extend(results)

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
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = self.output_dir / f"summary_{timestamp}.csv"
        success_df.to_csv(summary_path, index=False)

        print(f"Total factors evaluated: {len(df)}")
        print(f"Successful factors: {len(success_df)}")
        print("Top factors:")
        cols = ["factor_name", "fitness", "sharpe_ratio", "ic_ir", "returns", "expression"]
        print(success_df[cols].head(10).to_string(index=False))
        print(f"Summary saved to {summary_path}")
