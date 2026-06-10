from __future__ import annotations

import re
from typing import Any, Mapping

import numpy as np
import pandas as pd

from alpha101.factors.operator_lib import OPERATOR_REGISTRY


def signed_log(x):
    return np.sign(x) * np.log(np.abs(x) + 1)


def safe_sqrt(x):
    return np.sqrt(np.maximum(x, 0))


def remove_comments(code: str) -> str:
    cleaned_lines = []
    for line in code.splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def convert_ternary(expr: str) -> str:
    pattern = r"\((.*?)\)\s*\?\s*(.*?)\s*:\s*(.*)"
    match = re.search(pattern, expr)
    if match:
        cond, a, b = match.groups()
        return f"np.where({cond}, {a}, {b})"
    return expr


def normalize_expression_code(code: str) -> str:
    normalized = convert_ternary(remove_comments(code).lower())
    return re.sub(r"\bor\s*\(", "logical_or(", normalized)


def alpha_fields(alpha_instance) -> dict[str, Any]:
    return {
        "open": alpha_instance.open,
        "high": alpha_instance.high,
        "low": alpha_instance.low,
        "close": alpha_instance.close,
        "volume": alpha_instance.volume,
        "returns": alpha_instance.returns,
        "vwap": alpha_instance.vwap,
        "market_return": alpha_instance.market_return,
        "funding": alpha_instance.funding,
        "cap": alpha_instance.cap,
    }


def build_eval_env(fields: Mapping[str, Any]) -> dict[str, Any]:
    fields_dict = dict(fields)
    env = {
        **fields_dict,
        "np": np,
        "pd": pd,
        "abs": np.abs,
        "log": signed_log,
        "sign": np.sign,
        "sqrt": safe_sqrt,
        "max": np.maximum,
        "min": np.minimum,
        "exp": np.exp,
    }
    env.update(OPERATOR_REGISTRY)
    if {"close", "vwap", "volume"}.issubset(fields_dict):
        dollar_volume = fields_dict["vwap"] * fields_dict["volume"]
        env["volume_weighted_price"] = fields_dict["close"] * fields_dict["volume"]
        env["adv"] = lambda window: OPERATOR_REGISTRY["ts_mean"](dollar_volume, int(float(window)))
        for window in (3, 5, 6, 7, 8, 10, 12, 14, 15, 20, 21, 24, 28, 30, 40, 48, 54, 60):
            env[f"adv{window}"] = OPERATOR_REGISTRY["ts_mean"](dollar_volume, window)
        env["ts_vwap"] = lambda window: OPERATOR_REGISTRY["ts_vwap"](
            fields_dict["vwap"],
            fields_dict["volume"],
            int(float(window)),
        )
    return env


def evaluate_expression(code: str, fields: Mapping[str, Any]):
    env = build_eval_env(fields)
    lines = [line.strip() for line in normalize_expression_code(code).split(";") if line.strip()]
    if not lines:
        raise ValueError("Expression cannot be empty")

    for line in lines[:-1]:
        var, expr = line.split("=", 1)
        env[var.strip()] = eval(expr.strip(), {}, env)
    return eval(lines[-1], {}, env)
