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
    return convert_ternary(remove_comments(code).lower())


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
    env = {
        **dict(fields),
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
