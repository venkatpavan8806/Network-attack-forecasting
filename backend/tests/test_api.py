"""End-to-end HTTP tests against the real app object (app.api.main.app).

Every other test file calls service/model functions directly in Python --
none of them would notice if a route were deleted from main.py or stopped
matching the service's signature. These hit the real routes through
FastAPI's TestClient, as a real (dev-auth) user with real data in their own
workspace.
"""
from __future__ import annotations

import pytest

from tests.conftest import as_user, load_training_hosts

ALICE = as_user("alice")


@pytest.fixture()
def alice_client(client, real_service):
    load_training_hosts(real_service, "alice")
    return client


def _host(real_service) -> str:
    return sorted(real_service.labeled_df["host_id"].unique())[0]


def test_health_reports_ready(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ready"] is True and body["status"] == "ok"
    assert body["database"] in ("sqlite", "postgresql")


def test_new_user_starts_with_an_empty_workspace(client):
    assert client.get("/hosts", headers=as_user("brand-new")).json() == []
    k = client.get("/kpis", headers=as_user("brand-new")).json()
    assert k["hosts_monitored"] == 0 and k["forecasts_generated_today"] == 0
    assert client.get("/highest-risk-host", headers=as_user("brand-new")).json() is None
    assert client.get("/forecast-log", headers=as_user("brand-new")).json() == []


def test_hosts_lists_only_the_callers_hosts(alice_client, real_service):
    hosts = alice_client.get("/hosts", headers=ALICE).json()
    assert _host(real_service) in hosts
    assert alice_client.get("/hosts", headers=as_user("bob")).json() == []


def test_another_user_cannot_read_alices_host(alice_client, real_service):
    host = _host(real_service)
    for path in (f"/forecast/{host}", f"/track/{host}", f"/host-timeline/{host}", f"/shap/{host}"):
        assert alice_client.get(path, headers=as_user("bob")).status_code == 404, path


def test_unknown_host_returns_404(client):
    assert client.get("/forecast/not-a-real-host", headers=ALICE).status_code == 404


def test_forecast_route_returns_real_prediction(alice_client, real_service):
    host_id = _host(real_service)
    r = alice_client.get(f"/forecast/{host_id}", headers=ALICE)
    assert r.status_code == 200
    body = r.json()
    assert body["host_id"] == host_id
    assert 0.0 <= body["infiltration_probability_world_model"] <= 1.0
    assert len(body["rollout"]["infiltration_probs_world_model"]) > 0


def test_forecast_works_from_the_first_window(client, real_service):
    """No 8-window wait: a host with a single window is already forecastable (warm-up)."""
    load_training_hosts(real_service, "carol", [_host(real_service)])
    r = client.get(f"/forecast/{_host(real_service)}", params={"at_window_idx": 0}, headers=as_user("carol"))
    assert r.status_code == 200
    assert r.json()["warmup"] is True and r.json()["history_windows_used"] == 1


def test_forecast_branches_route_returns_a_real_tree(alice_client, real_service):
    host_id = _host(real_service)
    r = alice_client.get(f"/forecast/{host_id}/branches", headers=ALICE)
    assert r.status_code == 200
    body = r.json()
    assert body["tree"]["children"], "branching tree should have at least one child node"
    assert len(body["paths"]) > 0
    assert body["most_likely_path"] is not None


def test_track_route_predicts_after_every_window(alice_client, real_service):
    host_id = _host(real_service)
    n = len(real_service.labeled_df[real_service.labeled_df["host_id"] == host_id])
    body = alice_client.get(f"/track/{host_id}", headers=ALICE).json()
    assert len(body["steps"]) == n
    assert body["steps"][0]["warmup"] is True


def test_host_timeline_route(alice_client, real_service):
    rows = alice_client.get(f"/host-timeline/{_host(real_service)}", headers=ALICE).json()
    assert len(rows) > 0
    assert {"window_idx", "true_stage", "state_label"} <= rows[0].keys()


def test_mitigations_route_lists_real_options(client):
    ids = {m["id"] for m in client.get("/mitigations").json()}
    assert "isolate_host" in ids


def test_counterfactual_route_returns_metrics(alice_client, real_service):
    r = alice_client.get(f"/counterfactual/{_host(real_service)}", params={"mitigation_id": "isolate_host"},
                         headers=ALICE)
    assert r.status_code == 200
    body = r.json()
    assert "metrics" in body
    for key in ["mean_without", "mean_with", "risk_reduction_pct", "verdict"]:
        assert key in body["metrics"]


def test_shap_route_returns_real_contributions(alice_client, real_service):
    r = alice_client.get(f"/shap/{_host(real_service)}", headers=ALICE)
    assert r.status_code == 200
    assert len(r.json()["shap"]["contributions"]) > 0


def test_defense_route_returns_a_real_recommendation_or_reason(alice_client, real_service):
    body = alice_client.get(f"/defense/{_host(real_service)}", headers=ALICE).json()
    assert body["risk_level"] in {"act_now", "watch", "monitor"}
    assert body["recommendation_reason"]


def test_attack_mapping_route(client):
    assert len(client.get("/attack-mapping").json()) > 0


def test_stage_classes_route(client):
    assert "benign" in client.get("/stage-classes").json()


def test_threshold_calibration_route_404s_honestly_when_not_yet_computed(client, monkeypatch, tmp_path):
    import app.inference.service as service_module
    monkeypatch.setattr(service_module, "THRESHOLD_CALIBRATION_JSON", tmp_path / "no_such_file.json")
    r = client.get("/threshold-calibration")
    assert r.status_code == 404
    assert "not yet computed" in r.json()["detail"]
