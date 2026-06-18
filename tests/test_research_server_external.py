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
    assert body["factor_view"]["direction"] == 1
    assert body["factor_view"]["symbols"][0]["symbol"] == "920001.XBEI"

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


def test_research_server_uses_accepted_summary_for_direction_and_metadata(monkeypatch, tmp_path):
    wide = _external_wide(tmp_path)
    accepted_path = tmp_path / "accepted_summary.csv"
    pd.DataFrame(
        [
            {
                "factor_id": "probe_0001",
                "rank_ic": -0.041,
                "rank_icir": -0.72,
                "max_corr": 0.31,
                "expression": "NegExample()",
            }
        ]
    ).to_csv(accepted_path, index=False)
    monkeypatch.setenv("ALPHA101_ACCEPTED_SUMMARY_CSV", str(accepted_path))
    monkeypatch.setattr(server, "_context", {"wide_data": wide, "external_wide": wide, "cfg": None, "engine": None})
    monkeypatch.setattr(server, "_factor_view", None)
    monkeypatch.setattr(server, "_factor_view_seq", 0)
    client = TestClient(server.app)

    listing = client.get("/api/external-factors")
    assert listing.status_code == 200
    probe = next(item for item in listing.json()["factors"] if item["factor"] == "probe_0001")
    assert probe["direction"] == -1
    assert probe["buy_group"] == 1
    assert probe["rank_icir"] == -0.72
    assert probe["max_corr"] == 0.31

    response = client.post("/api/external-factor-backtest/probe_0001", json={"n_quintiles": 3})
    assert response.status_code == 200
    body = response.json()
    assert body["config"]["direction"] == -1
    assert body["config"]["buy_group"] == 1
    assert body["metrics"]["rank_icir"] == -0.72
    assert body["metrics"]["max_corr"] == 0.31
    assert body["metrics"]["direction_source"] == "accepted_summary_rank_icir"
    assert body["factor_view"]["direction"] == -1
    assert body["factor_view"]["rank_meaning"] == "lowest factor values are selected"
    assert body["factor_view"]["symbols"][0]["symbol"] == "000001.XSHE"


def test_research_server_selects_external_portfolio_with_pnl_similarity_filter(monkeypatch, tmp_path):
    rows = []
    targets = [
        [0.01, 0.02, 0.04, -0.01],
        [0.04, -0.01, 0.02, 0.01],
        [0.02, 0.04, -0.01, 0.01],
        [-0.01, 0.01, 0.04, 0.02],
        [0.03, 0.02, -0.01, 0.04],
        [0.01, 0.04, 0.02, -0.01],
    ]
    good_orders = [
        [1, 2, 4, 3],
        [4, 1, 3, 2],
        [2, 4, 1, 3],
        [1, 2, 4, 3],
        [3, 2, 1, 4],
        [1, 4, 2, 3],
    ]
    diverse_orders = [
        [1, 4, 2, 3],
        [2, 3, 4, 1],
        [4, 1, 3, 2],
        [3, 4, 1, 2],
        [4, 1, 2, 3],
        [2, 1, 4, 3],
    ]
    codes = ["000001.XSHE", "600000.XSHG", "920001.XBEI", "300001.XSHE"]
    for date_idx, dt in enumerate(pd.date_range("2025-01-07", periods=len(targets), freq="7D")):
        for code_idx, code in enumerate(codes):
            good = float(good_orders[date_idx][code_idx])
            diverse = float(diverse_orders[date_idx][code_idx])
            rows.append(
                {
                    "date": dt,
                    "code": code,
                    "label_5d": targets[date_idx][code_idx],
                    "good": good,
                    "good_clone": good * 10.0,
                    "diverse": diverse,
                }
            )
    csv_path = tmp_path / "accepted.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    wide = build_external_factor_wide_frame(csv_path)
    monkeypatch.setattr(server, "_context", {"wide_data": wide, "external_wide": wide, "cfg": None, "engine": None})
    client = TestClient(server.app)

    response = client.post(
        "/api/external-factor-selection",
        json={
            "n_quintiles": 2,
            "max_factors": 2,
            "pnl_corr_threshold": 0.95,
            "pnl_corr_method": "spearman",
            "transaction_cost": 0.0,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["stats"]["selected_count"] == 2
    assert body["stats"]["pnl_corr_method"] == "spearman"
    assert body["stats"]["selection_scope"] == "in_sample_full_history"
    assert body["stats"]["rejected_similarity_count"] >= 1
    assert len(body["curve"]) == 6
    assert {item["factor"] for item in body["selected"]} == {"good", "diverse"}
    rejected = {item["factor"]: item for item in body["rejected"]}
    assert rejected["good_clone"]["selection_reason"] == "pnl_similarity"


def test_research_server_exposes_ashare_microcap_protocol(monkeypatch, tmp_path):
    wide = _external_wide(tmp_path)
    monkeypatch.setattr(server, "_context", {"wide_data": wide, "external_wide": wide, "cfg": None, "engine": None})
    client = TestClient(server.app)

    response = client.get("/api/protocol")

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    protocol = body["protocol"]
    assert protocol["name"] == "ashare_microcap_weekly_alphaprobe"
    assert protocol["rebalance_weekday"] == "Tuesday"
    assert protocol["label_embargo_periods"] == 1
    assert protocol["data"]["external_factor_count"] == 2
    assert protocol["data"]["symbol_count"] == 3
    assert protocol["data"]["date_start"] == "2025-01-07 00:00:00"
    assert "fit selection and ridge weights using history only" in protocol["no_leakage_rules"]
