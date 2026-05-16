from typing import List, Literal, Tuple, Optional
import os
import numpy as np
import pandas as pd

from lightgbm import LGBMRanker, early_stopping, log_evaluation
from xgboost import XGBRanker


def _cs_winsorize(s: pd.Series, q: float = 0.01) -> pd.Series:
    lo, hi = s.quantile(q), s.quantile(1 - q)
    return s.clip(lo, hi)


def _cs_zscore(s: pd.Series) -> pd.Series:
    std = s.std(ddof=0)
    if std == 0 or np.isnan(std):
        return s * 0.0
    return (s - s.mean()) / std


def _make_panel(
    alpha_list: List[pd.DataFrame],
    returns: pd.DataFrame,
    fill_method: Literal["median", "zero", "drop"] = "median",
) -> Tuple[pd.DataFrame, List[str]]:
    if not alpha_list:
        raise ValueError("alpha_list is empty.")

    idx, cols = returns.index, returns.columns
    for a in alpha_list:
        idx = idx.intersection(a.index)
        cols = cols.intersection(a.columns)

    if len(idx) == 0 or len(cols) == 0:
        raise ValueError("No common dates or symbols.")

    data = []

    alpha_cols = []
    for i, a in enumerate(alpha_list):
        col = f"alpha_{i}"
        alpha_cols.append(col)
        s = a.loc[idx, cols].stack(dropna=False).rename(col)
        data.append(s)

    ret = returns.loc[idx, cols].stack(dropna=False).rename("return")
    data.append(ret)

    df = pd.concat(data, axis=1)
    df.index.names = ["date", "symbol"]
    df = df.sort_index()
    df = df.dropna(subset=["return"])

    if fill_method == "median":
        df[alpha_cols] = df.groupby(level="date")[alpha_cols].transform(
            lambda x: x.fillna(x.median())
        )
        df[alpha_cols] = df[alpha_cols].fillna(0.0)
    elif fill_method == "zero":
        df[alpha_cols] = df[alpha_cols].fillna(0.0)
    elif fill_method == "drop":
        df = df.dropna(subset=alpha_cols)
    else:
        raise ValueError("fill_method must be 'median', 'zero', or 'drop'.")

    # 每日横截面标准化，机构里通常会做
    df[alpha_cols] = df.groupby(level="date")[alpha_cols].transform(_cs_winsorize)
    df[alpha_cols] = df.groupby(level="date")[alpha_cols].transform(_cs_zscore)
    df[alpha_cols] = df[alpha_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if df.empty:
        raise ValueError("No valid samples.")

    return df, alpha_cols


def _make_rank_label(ret: pd.Series, n_bins: int = 10) -> pd.Series:
    """
    每日横截面 future return -> 0 ~ n_bins-1 的整数 relevance label。
    LightGBM / XGBoost rank:ndcg 都适用。
    """
    def one_day(x: pd.Series) -> pd.Series:
        if x.notna().sum() <= 1:
            return pd.Series(0, index=x.index, dtype=np.int32)

        pct = x.rank(method="first", pct=True)
        label = np.floor(pct * n_bins).astype(np.int32)
        label = np.clip(label, 0, n_bins - 1)
        return pd.Series(label, index=x.index, dtype=np.int32)

    return ret.groupby(level="date", group_keys=False).apply(one_day)


def _split_by_date(
    df: pd.DataFrame,
    train_ratio: float,
    valid_ratio: float,
):
    dates = pd.Index(df.index.get_level_values("date").unique()).sort_values()

    n = len(dates)
    train_end = int(n * train_ratio)
    valid_end = int(n * (train_ratio + valid_ratio))

    if train_end <= 0 or valid_end <= train_end or valid_end >= n:
        raise ValueError("Not enough dates for train / valid / test split.")

    train_dates = dates[:train_end]
    valid_dates = dates[train_end:valid_end]
    test_dates = dates[valid_end:]

    train_df = df.loc[df.index.get_level_values("date").isin(train_dates)].sort_index()
    valid_df = df.loc[df.index.get_level_values("date").isin(valid_dates)].sort_index()

    return train_df, valid_df, test_dates


def _xy_group(df: pd.DataFrame, alpha_cols: List[str]):
    X = df[alpha_cols].values
    y = df["label"].values.astype(np.int32)
    group = df.groupby(level="date").size().values.astype(int)
    return X, y, group


def lgb_combine(
    alpha_list: List[pd.DataFrame],
    returns: pd.DataFrame,
    train_ratio: float = 0.7,
    valid_ratio: float = 0.15,
    n_bins: int = 10,
    ndcg_top_k: int = 50,
    fill_method: Literal["median", "zero", "drop"] = "median",
    random_state: int = 42,
    return_model: bool = False,
):
    """
    推荐主力版本：LightGBM LambdaRank alpha combine。

    returns 必须是 future return，例如:
        future_returns = close.pct_change(5).shift(-5)
    """
    df, alpha_cols = _make_panel(alpha_list, returns, fill_method)
    df["label"] = _make_rank_label(df["return"], n_bins=n_bins)

    train_df, valid_df, test_dates = _split_by_date(df, train_ratio, valid_ratio)

    X_train, y_train, group_train = _xy_group(train_df, alpha_cols)
    X_valid, y_valid, group_valid = _xy_group(valid_df, alpha_cols)

    model = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        label_gain=list(range(n_bins)),
        n_estimators=5000,
        learning_rate=0.02,
        num_leaves=31,
        min_child_samples=100,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=10.0,
        random_state=random_state,
        n_jobs=max((os.cpu_count() or 8) - 4, 1),
        importance_type="gain",
    )

    model.fit(
        X_train,
        y_train,
        group=group_train,
        eval_set=[(X_valid, y_valid)],
        eval_group=[group_valid],
        eval_at=[ndcg_top_k],
        callbacks=[
            early_stopping(100),
            log_evaluation(100),
        ],
    )

    df["score"] = model.predict(
        df[alpha_cols].values,
        num_iteration=model.best_iteration_,
    )

    score = df["score"].unstack("symbol")
    score = score.reindex(index=returns.index, columns=returns.columns)

    info = {
        "model": "lightgbm_lambdarank",
        "n_bins": n_bins,
        "eval_metric": f"ndcg@{ndcg_top_k}",
        "best_iteration": model.best_iteration_,
        "alpha_cols": alpha_cols,
        "test_start": test_dates[0],
        "test_end": test_dates[-1],
    }

    if return_model:
        return score, model, info

    return score


def xgb_combine(
    alpha_list: List[pd.DataFrame],
    returns: pd.DataFrame,
    train_ratio: float = 0.7,
    valid_ratio: float = 0.15,
    n_bins: int = 10,
    ndcg_top_k: Optional[int] = 50,
    fill_method: Literal["median", "zero", "drop"] = "median",
    random_state: int = 42,
    return_model: bool = False,
):
    """
    XGBoost NDCG ranker 版本。
    """
    df, alpha_cols = _make_panel(alpha_list, returns, fill_method)
    df["label"] = _make_rank_label(df["return"], n_bins=n_bins)

    train_df, valid_df, test_dates = _split_by_date(df, train_ratio, valid_ratio)

    X_train, y_train, group_train = _xy_group(train_df, alpha_cols)
    X_valid, y_valid, group_valid = _xy_group(valid_df, alpha_cols)

    eval_metric = "ndcg" if ndcg_top_k is None else f"ndcg@{ndcg_top_k}"

    model = XGBRanker(
        objective="rank:ndcg",
        eval_metric=eval_metric,
        learning_rate=0.03,
        n_estimators=3000,
        max_depth=3,
        min_child_weight=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=10.0,
        tree_method="hist",
        random_state=random_state,
        early_stopping_rounds=100,
        n_jobs=max((os.cpu_count() or 8) - 4, 1),
    )

    model.fit(
        X_train,
        y_train,
        group=group_train.tolist(),
        eval_set=[(X_valid, y_valid)],
        eval_group=[group_valid.tolist()],
        verbose=100,
    )

    df["score"] = model.predict(df[alpha_cols].values)

    score = df["score"].unstack("symbol")
    score = score.reindex(index=returns.index, columns=returns.columns)

    info = {
        "model": "xgboost_rank_ndcg",
        "n_bins": n_bins,
        "eval_metric": eval_metric,
        "best_iteration": getattr(model, "best_iteration", None),
        "best_score": getattr(model, "best_score", None),
        "alpha_cols": alpha_cols,
        "test_start": test_dates[0],
        "test_end": test_dates[-1],
    }

    if return_model:
        return score, model, info

    return score
from typing import List, Literal
import numpy as np
import pandas as pd

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def _cs_winsorize(s: pd.Series, q: float = 0.01) -> pd.Series:
    lo, hi = s.quantile(q), s.quantile(1 - q)
    return s.clip(lo, hi)


def _cs_zscore(s: pd.Series) -> pd.Series:
    std = s.std(ddof=0)
    if std == 0 or np.isnan(std):
        return s * 0.0
    return (s - s.mean()) / std


def _safe_stack(df: pd.DataFrame) -> pd.Series:
    # pandas >= 2.1 推荐 future_stack=True；旧版本不支持则回退
    try:
        return df.stack(future_stack=True)
    except TypeError:
        return df.stack(dropna=False)


def _make_panel(
    alpha_list: List[pd.DataFrame],
    returns: pd.DataFrame,
    fill_method: Literal["median", "zero", "drop"] = "median",
):
    if not alpha_list:
        raise ValueError("alpha_list is empty.")

    idx = returns.index
    cols = returns.columns

    for a in alpha_list:
        idx = idx.intersection(a.index)
        cols = cols.intersection(a.columns)

    if len(idx) == 0 or len(cols) == 0:
        raise ValueError("No common dates or symbols.")

    data = []
    alpha_cols = []

    for i, a in enumerate(alpha_list):
        col = f"alpha_{i}"
        alpha_cols.append(col)
        s = _safe_stack(a.loc[idx, cols]).rename(col)
        data.append(s)

    ret = _safe_stack(returns.loc[idx, cols]).rename("return")
    data.append(ret)

    df = pd.concat(data, axis=1)
    df.index.names = ["date", "symbol"]
    df = df.sort_index()
    df = df.dropna(subset=["return"])

    if fill_method == "median":
        df[alpha_cols] = df.groupby(level="date")[alpha_cols].transform(
            lambda x: x.fillna(x.median())
        )
        df[alpha_cols] = df[alpha_cols].fillna(0.0)

    elif fill_method == "zero":
        df[alpha_cols] = df[alpha_cols].fillna(0.0)

    elif fill_method == "drop":
        df = df.dropna(subset=alpha_cols)

    else:
        raise ValueError("fill_method must be 'median', 'zero', or 'drop'.")

    # 每日横截面 winsorize + zscore
    df[alpha_cols] = df.groupby(level="date")[alpha_cols].transform(_cs_winsorize)
    df[alpha_cols] = df.groupby(level="date")[alpha_cols].transform(_cs_zscore)
    df[alpha_cols] = df[alpha_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if df.empty:
        raise ValueError("No valid samples.")

    return df, alpha_cols


def _make_quantile_label(ret: pd.Series) -> pd.Series:
    """
    每日横截面 future return -> 0~1 分位数。
    MLP 用连续 label，比整数 relevance 更合适。
    """
    def one_day(x: pd.Series) -> pd.Series:
        if x.notna().sum() <= 1:
            return pd.Series(0.5, index=x.index, dtype=float)
        return x.rank(method="first", pct=True).astype(float)

    return ret.groupby(level="date", group_keys=False).apply(one_day)


def _split_by_date(
    df: pd.DataFrame,
    train_ratio: float,
    valid_ratio: float,
):
    dates = pd.Index(df.index.get_level_values("date").unique()).sort_values()

    n = len(dates)
    train_end = int(n * train_ratio)
    valid_end = int(n * (train_ratio + valid_ratio))

    if train_end <= 0 or valid_end <= train_end or valid_end >= n:
        raise ValueError("Not enough dates for train / valid / test split.")

    train_dates = dates[:train_end]
    valid_dates = dates[train_end:valid_end]
    test_dates = dates[valid_end:]

    train_df = df.loc[df.index.get_level_values("date").isin(train_dates)].sort_index()
    valid_df = df.loc[df.index.get_level_values("date").isin(valid_dates)].sort_index()

    return train_df, valid_df, test_dates


def mlp_combine(
    alpha_list: List[pd.DataFrame],
    returns: pd.DataFrame,
    train_ratio: float = 0.7,
    valid_ratio: float = 0.15,
    fill_method: Literal["median", "zero", "drop"] = "median",
    hidden_layer_sizes=(16, 8),
    alpha: float = 1e-3,
    learning_rate_init: float = 1e-3,
    max_iter: int = 300,
    random_state: int = 42,
    return_model: bool = False,
):
    """
    MLP alpha combine:
    - 输入: 多个 alpha 宽表
    - label: 每日 future return 的横截面分位数 0~1
    - 输出: date x symbol 的 score 宽表

    returns 必须是 future return，例如:
        future_returns = close.pct_change(5).shift(-5)
    """
    df, alpha_cols = _make_panel(alpha_list, returns, fill_method)

    # 预测未来收益的横截面分位数
    df["label"] = _make_quantile_label(df["return"])

    train_df, valid_df, test_dates = _split_by_date(df, train_ratio, valid_ratio)

    X_train = train_df[alpha_cols]
    y_train = train_df["label"]

    X_valid = valid_df[alpha_cols]
    y_valid = valid_df["label"]

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPRegressor(
                    hidden_layer_sizes=hidden_layer_sizes,
                    activation="relu",
                    solver="adam",
                    alpha=alpha,
                    batch_size=4096,
                    learning_rate="adaptive",
                    learning_rate_init=learning_rate_init,
                    max_iter=max_iter,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=20,
                    random_state=random_state,
                    verbose=False,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    df["score"] = model.predict(df[alpha_cols])

    # MLP 输出可能略超出 0~1，截断一下更稳
    df["score"] = df["score"].clip(0.0, 1.0)

    score = df["score"].unstack("symbol")
    score = score.reindex(index=returns.index, columns=returns.columns)

    # 简单 valid RankIC
    valid_pred = pd.Series(
        model.predict(X_valid),
        index=valid_df.index,
        name="pred",
    ).clip(0.0, 1.0)

    valid_tmp = pd.concat(
        [valid_pred, valid_df["return"]],
        axis=1,
    ).dropna()

    valid_rank_ic = valid_tmp.groupby(level="date").apply(
        lambda x: x["pred"].rank().corr(x["return"].rank())
        if len(x) > 2 else np.nan
    )

    info = {
        "model": "mlp_quantile_regressor",
        "hidden_layer_sizes": hidden_layer_sizes,
        "alpha": alpha,
        "learning_rate_init": learning_rate_init,
        "max_iter": max_iter,
        "valid_rank_ic_mean": float(valid_rank_ic.mean()),
        "valid_rank_ic_ir": float(valid_rank_ic.mean() / valid_rank_ic.std())
        if valid_rank_ic.std() != 0 and not np.isnan(valid_rank_ic.std())
        else np.nan,
        "alpha_cols": alpha_cols,
        "test_start": test_dates[0],
        "test_end": test_dates[-1],
    }
    print(info)
    if return_model:
        return score, model, info

    return score