from __future__ import annotations

import pandas as pd
from fastapi.testclient import TestClient

import alpha101.research.server as server
from alpha101.data.external import build_external_factor_wide_frame


def _external_wide(tmp_path):
    rows = []
    for date_idx, dt in enumerate(pd.date_range("2025-01-07", periods=4, freq="7D")):
        for code_idx, code in enumerate(["000001.XSHE", "600000.XSHG", "920001.XBEI"]):
            target = 0.01 * (date_idx + 1) * (code_idx + 1)
            rows.append(
                {
                    "date": dt,
                    "signal_date": dt - pd.Timedelta(days=1),
                    "code": code,
                    "label_5d": target,
                    "probe_0001": target * 10.0,
                    "post10x30_0002": -target * 5.0,
                }
            )
    csv_path = tmp_path / "accepted.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return build_external_factor_wide_frame(csv_path)


def test_research_server_lists_and_backtests_external_factors(monkeypatch, tmp_path):
    wide = _external_wide(tmp_path)
    monkeypatch.setattr(server, "_context", {"wide_data": wide, "external_wide": wide, "cfg": None, "engine": None})
    monkeypatch.setattr(server, "_factor_view", None)
    monkeypatch.setattr(server, "_factor_view_seq", 0)
    client = TestClient(server.app)

    listing = client.get("/api/external-factors")
    assert listing.status_code == 200
    assert [item["factor"] for item in listing.json()["factors"]] == ["probe_0001", "post10x30_0002"]

    response = client.post("/api/external-factor-backtest/probe_0001", json={"n_quintiles": 3})
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["factor"] == "probe_0001"
    assert body["metrics"]["ic_mean"] > 0
    assert len(body["curve"]) == 4
    assert body["factor_view"]["symbols"][0]["symbol"] in {"000001.XSHE", "600000.XSHG", "920001.XBEI"}

    clamped = client.post("/api/external-factor-backtest/probe_0001", json={"n_quintiles": 5})
    assert clamped.status_code == 200
    assert clamped.json()["config"]["n_quintiles"] == 3


def test_research_server_external_only_context_does_not_require_ohlc(monkeypatch, tmp_path):
    csv_path = tmp_path / "accepted.csv"
    rows = []
    for date_idx, dt in enumerate(pd.date_range("2025-01-07", periods=4, freq="7D")):
        for code_idx, code in enumerate(["000001.XSHE", "600000.XSHG", "920001.XBEI"]):
            target = 0.01 * (date_idx + 1) * (code_idx + 1)
            rows.append({"date": dt, "code": code, "label_5d": target, "probe_0001": target * 10.0})
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    monkeypatch.setenv("ALPHA101_EXTERNAL_FACTOR_CSV", str(csv_path))
    monkeypatch.delenv("ALPHA101_ASHARE_CSV", raising=False)

    ctx = server._load_context()

    assert ctx["engine"] is None
    assert "probe_0001" in ctx["external_wide"].columns.get_level_values(0)
