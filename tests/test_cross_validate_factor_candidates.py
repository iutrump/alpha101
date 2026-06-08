from scripts.cross_validate_factor_candidates import _select_candidates


def _row(expression: str, *, valid_sharpe: float, train_sharpe: float, valid_ic_ir: float, complexity: float) -> dict:
    return {
        "expression": expression,
        "valid_sharpe": valid_sharpe,
        "train_sharpe": train_sharpe,
        "valid_ic_ir": valid_ic_ir,
        "complexity_score": complexity,
    }


def test_robust_candidate_complexity_threshold_is_configurable() -> None:
    rows = [
        _row("rank(close)", valid_sharpe=1.5, train_sharpe=0.5, valid_ic_ir=0.2, complexity=9.0),
    ]

    strict = _select_candidates(rows, max_candidates=0, robust_max_complexity=6.0, candidate_max_complexity=12.0)
    relaxed = _select_candidates(rows, max_candidates=0, robust_max_complexity=10.0, candidate_max_complexity=12.0)

    assert strict == []
    assert [row["expression"] for row in relaxed] == ["rank(close)"]


def test_top_candidate_complexity_threshold_is_configurable() -> None:
    rows = [
        _row("rank(vwap)", valid_sharpe=0.8, train_sharpe=0.4, valid_ic_ir=0.05, complexity=11.0),
    ]

    strict = _select_candidates(rows, max_candidates=1, robust_max_complexity=10.0, candidate_max_complexity=8.0)
    relaxed = _select_candidates(rows, max_candidates=1, robust_max_complexity=10.0, candidate_max_complexity=12.0)

    assert strict == []
    assert [row["expression"] for row in relaxed] == ["rank(vwap)"]
