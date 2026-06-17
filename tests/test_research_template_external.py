from __future__ import annotations

from importlib.resources import files


def test_research_template_exposes_external_factor_browser():
    html = (files("alpha101.research") / "templates" / "index.html").read_text(encoding="utf-8")

    assert "externalFactorList" in html
    assert "/api/external-factors" in html
    assert "/api/external-factor-backtest/" in html
