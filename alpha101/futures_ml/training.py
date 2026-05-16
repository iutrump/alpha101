from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, ndcg_score, r2_score
from tqdm import tqdm
import os
try:
    from xgboost import XGBRanker
except ImportError as exc:  # pragma: no cover
    raise ImportError("xgboost is required: pip install xgboost") from exc


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    return sorted(list(set([c for c in df.columns if c.startswith("alpha")])))


def time_split(df: pd.DataFrame, train_ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dates = sorted(df["asof_date"].unique())
    if not dates:
        raise ValueError("No dates in dataset for split")
    cutoff_idx = max(1, int(len(dates) * train_ratio))
    cutoff_date = dates[cutoff_idx - 1]
    train = df[df["asof_date"] <= cutoff_date].copy()
    test = df[df["asof_date"] > cutoff_date].copy()
    return train, test


def _split_train_val(train_df: pd.DataFrame, train_split: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = train_df["asof_date"].unique()
    if len(dates) < 2:
        return train_df, pd.DataFrame()
    cutoff_idx = max(1, int(len(dates) * train_split))
    cutoff_date = dates[cutoff_idx - 1]
    train_part = train_df[train_df["asof_date"] <= cutoff_date].copy()
    val_part = train_df[train_df["asof_date"] > cutoff_date].copy()
    if val_part.empty:
        return train_df, pd.DataFrame()
    return train_part, val_part


def _group_lengths(df: pd.DataFrame) -> List[int]:
    return df.groupby("date").size().tolist()


def _build_rank_labels(df: pd.DataFrame, mode: str) -> pd.Series:
    if mode == "long":
        ref = df["target"]
    elif mode == "short":
        ref = -df["target"]
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    # Build per-date quantile-style relevance labels (1..5).
    pct = ref.groupby(df["date"]).rank(method="average", ascending=True, pct=True)
    quantile_label = np.ceil(pct * 5.0).astype(int).clip(1, 5)
    return quantile_label


def _prepare_ranker_matrices(df: pd.DataFrame, feature_cols: List[str]) -> tuple[np.ndarray, np.ndarray, List[int]]:
    df = df[df["target_rank"].notna()].copy()
    sort_cols = [c for c in ["date", "symbol"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)
    X = df[feature_cols].values
    y = df["target_rank"].astype(int).values
    groups = _group_lengths(df)
    return X, y, groups


def _build_ranker(objective: str, eval_metric: str) -> XGBRanker:
    return XGBRanker(
        max_depth=2,
        n_estimators=600,
        learning_rate=0.03,
        min_child_weight=8.0,
        gamma=1.0,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=1.0,
        reg_lambda=5.0,
        objective=objective,
        eval_metric=[eval_metric],
        random_state=42,
        tree_method="hist",
        lambdarank_pair_method="topk",
        ndcg_exp_gain=False,
        early_stopping_rounds=50,
        n_jobs=max(os.cpu_count()-4, 1),
    )


def train_model(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    train_split: float,
    mode: str,
    eval_k: int,
) -> XGBRanker:
    train_df = train_df.copy()
    train_df["target_rank"] = _build_rank_labels(train_df, mode)
    model = _build_ranker(objective="rank:ndcg", eval_metric=f"ndcg@{max(1, int(eval_k))}")

    train_part, val_part = _split_train_val(train_df, train_split)
    X_train, y_train, g_train = _prepare_ranker_matrices(train_part, feature_cols)

    if val_part.empty:
        model.fit(X_train, y_train, group=g_train, verbose=False)
        return model

    X_val, y_val, g_val = _prepare_ranker_matrices(val_part, feature_cols)
    model.fit(
        X_train,
        y_train,
        group=g_train,
        eval_set=[(X_val, y_val)],
        eval_group=[g_val],
        verbose=False,
    )
    return model


def train_dual_models(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    train_split: float,
    long_k: int,
    short_k: int,
) -> tuple[XGBRanker, XGBRanker]:
    long_model = train_model(train_df, feature_cols, train_split, mode="long", eval_k=long_k)
    short_model = train_model(train_df, feature_cols, train_split, mode="short", eval_k=short_k)
    return long_model, short_model


def _zscore(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    std = arr.std()
    if std == 0 or np.isnan(std):
        return arr - arr.mean()
    return (arr - arr.mean()) / std


def predict_dual(
    long_model: XGBRanker,
    short_model: XGBRanker,
    df: pd.DataFrame,
    feature_cols: List[str],
) -> pd.DataFrame:
    rows = []
    for _, each in df.groupby("date", sort=True):
        long_score_raw = long_model.predict(each[feature_cols])
        short_score_raw = short_model.predict(each[feature_cols])

        # Use cross-sectional percentile ranks to reduce score-scale instability.
        long_cs_rank = pd.Series(long_score_raw, index=each.index).rank(pct=True).values
        short_cs_rank = pd.Series(short_score_raw, index=each.index).rank(pct=True).values
        combined_raw = long_cs_rank - short_cs_rank

        g = each.copy()
        g["long_score_raw"] = long_score_raw
        g["short_score_raw"] = short_score_raw
        g["predicted_return_raw"] = combined_raw
        rows.append(g)

    out = pd.concat(rows, axis=0).sort_values(["trade_date", "symbol"]).reset_index(drop=True)
    return out


def apply_prediction_smoothing(pred_df: pd.DataFrame, smoothing_span: int) -> pd.DataFrame:
    out = pred_df.copy().sort_values(["symbol", "trade_date"]).reset_index(drop=True)

    # Enforce stronger minimum smoothing to curb noisy daily turnover.
    span = max(20, int(smoothing_span))
    out["long_score"] = (
        out.groupby("symbol")["long_score_raw"]
        .transform(lambda s: s.ewm(span=span, adjust=False, min_periods=1).mean())
    )
    out["short_score"] = (
        out.groupby("symbol")["short_score_raw"]
        .transform(lambda s: s.ewm(span=span, adjust=False, min_periods=1).mean())
    )
    out["predicted_return"] = (
        out.groupby("symbol")["predicted_return_raw"]
        .transform(lambda s: s.ewm(span=span, adjust=False, min_periods=1).mean())
    )

    out = out.sort_values(["trade_date", "symbol"]).reset_index(drop=True)
    return out


def _walk_forward_single(
    df: pd.DataFrame,
    feature_cols: List[str],
    retrain_every_bars: int,
    min_train_bars: int,
    train_bars: Optional[int],
    train_split: float,
    start_dt: Optional[pd.Timestamp],
    end_dt: Optional[pd.Timestamp],
    long_k: int,
    short_k: int,
) -> pd.DataFrame:
    df = df.sort_values("asof_date")
    dates = sorted(df["asof_date"].unique())
    if len(dates) <= min_train_bars:
        raise ValueError("Not enough bars for walk-forward")

    preds: List[pd.DataFrame] = []
    idx = min_train_bars
    total_folds = (len(dates) - min_train_bars) // retrain_every_bars + 1

    with tqdm(total=total_folds) as pbar:
        while idx < len(dates):
            train_end_date = dates[idx - 1]
            train_start_idx = 0
            if train_bars is not None:
                train_start_idx = max(0, idx - train_bars)
            train_start_date = dates[train_start_idx]
            test_end_idx = min(idx + retrain_every_bars - 1, len(dates) - 1)
            fold_start = dates[idx]
            fold_end = dates[test_end_idx]

            if start_dt is not None and end_dt is not None:
                if pd.Timestamp(fold_end) < start_dt or pd.Timestamp(fold_start) > end_dt:
                    idx = test_end_idx + 1
                    pbar.update(1)
                    continue

            train_df = df[(df["asof_date"] >= train_start_date) & (df["asof_date"] <= train_end_date)]
            test_df = df[(df["asof_date"] >= fold_start) & (df["asof_date"] <= fold_end)]
            if start_dt is not None:
                test_df = test_df[test_df["trade_date"] >= start_dt]
            if end_dt is not None:
                test_df = test_df[test_df["trade_date"] <= end_dt]
            if test_df.empty:
                break

            print(
                "Walk-forward fold:"
                f" train [{pd.Timestamp(train_start_date)} -> {pd.Timestamp(train_end_date)}]"
                f" ({train_df['asof_date'].nunique()} bars),"
                f" test [{pd.Timestamp(fold_start)} -> {pd.Timestamp(fold_end)}]"
                f" ({test_df['asof_date'].nunique()} bars)"
            )
            print(
                "Trade-date window:"
                f" [{pd.Timestamp(test_df['trade_date'].min())} -> {pd.Timestamp(test_df['trade_date'].max())}]"
            )

            long_model, short_model = train_dual_models(
                train_df=train_df,
                feature_cols=feature_cols,
                train_split=train_split,
                long_k=long_k,
                short_k=short_k,
            )
            fold_pred = predict_dual(long_model, short_model, test_df, feature_cols)
            preds.append(fold_pred)

            idx = test_end_idx + 1
            pbar.update(1)

    if not preds:
        raise ValueError("Walk-forward produced no predictions")
    return pd.concat(preds, ignore_index=True)


def walk_forward_predict(
    df: pd.DataFrame,
    feature_cols: List[str],
    retrain_every_bars: int = 3,
    min_train_bars: int = 30,
    train_bars: Optional[int] = None,
    train_split: float = 0.9,
    test_start_date: Optional[pd.Timestamp] = None,
    test_end_date: Optional[pd.Timestamp] = None,
    train_per_pair: bool = False,
    top_n_long: int = 10,
    top_n_short: int = 10,
    smoothing_span: int = 5,
) -> pd.DataFrame:
    """
    Walk-forward: train on bars up to a date, predict the next retrain_every_bars window, repeat.
    Uses unique timestamps present in df["asof_date"] as bars.
    If train_per_pair is True, this argument is currently ignored in the dual-model path.
    """
    if train_bars is not None and train_bars < min_train_bars:
        raise ValueError("train_bars must be >= min_train_bars")

    start_dt = pd.Timestamp(test_start_date) if test_start_date is not None else None
    end_dt = pd.Timestamp(test_end_date) if test_end_date is not None else None

    pred = _walk_forward_single(
        df=df,
        feature_cols=feature_cols,
        retrain_every_bars=retrain_every_bars,
        min_train_bars=min_train_bars,
        train_bars=train_bars,
        train_split=train_split,
        start_dt=start_dt,
        end_dt=end_dt,
        long_k=max(1, int(top_n_long)),
        short_k=max(1, int(top_n_short)),
    )
    return apply_prediction_smoothing(pred, smoothing_span=smoothing_span)


def _daily_spearman(pred_df: pd.DataFrame) -> float:
    scores = []
    for _, g in pred_df.groupby("trade_date"):
        if g["target"].nunique() <= 1:
            continue
        corr = spearmanr(g["predicted_return"], g["target"]).correlation
        if corr is not None and not np.isnan(corr):
            scores.append(corr)
    return float(np.mean(scores)) if scores else 0.0


def _daily_ndcg_long(pred_df: pd.DataFrame, k: int) -> float:
    if k <= 0:
        return 0.0
    scores = []
    for _, g in pred_df.groupby("trade_date"):
        rel = g["target"].values.astype(float)
        rel = rel - rel.min()
        pred = g["predicted_return"].values.astype(float)
        if rel.size == 0:
            continue
        scores.append(ndcg_score([rel], [pred], k=min(k, rel.size)))
    return float(np.mean(scores)) if scores else 0.0


def _daily_ndcg_short(pred_df: pd.DataFrame, k: int) -> float:
    if k <= 0:
        return 0.0
    scores = []
    for _, g in pred_df.groupby("trade_date"):
        rel_short = (-g["target"].values.astype(float))
        rel_short = rel_short - rel_short.min()
        pred_short = (-g["predicted_return"].values.astype(float))
        if rel_short.size == 0:
            continue
        scores.append(ndcg_score([rel_short], [pred_short], k=min(k, rel_short.size)))
    return float(np.mean(scores)) if scores else 0.0


def evaluate(pred_df: pd.DataFrame, top_k_long: int = 10, top_k_short: Optional[int] = None) -> dict:
    if top_k_short is None:
        top_k_short = top_k_long

    y_true = pred_df["target"]
    y_pred = pred_df["predicted_return"]
    y_true = (y_true-y_true.mean())/(y_true.std() + 1e-12)
    y_pred = (y_pred-y_pred.mean())/(y_pred.std() + 1e-12)

    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    spearman = spearmanr(y_pred, y_true).correlation
    daily_spearman = _daily_spearman(pred_df)
    ndcg_long = _daily_ndcg_long(pred_df, top_k_long)
    ndcg_short = _daily_ndcg_short(pred_df, top_k_short)

    return {
        "mae": float(mae),
        "r2": float(r2),
        "spearman": float(spearman) if spearman is not None else 0.0,
        "daily_spearman": daily_spearman,
        "ndcg_long@k": ndcg_long,
        "ndcg_short@k": ndcg_short,
        "ndcg_avg": float((ndcg_long + ndcg_short) / 2.0),
        "ndcg@k": float((ndcg_long + ndcg_short) / 2.0),
        "ndcg_long_k": int(top_k_long),
        "ndcg_short_k": int(top_k_short),
    }


def prepare_matrices(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    return df[feature_cols].values, df["target"].values
