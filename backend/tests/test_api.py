"""End-to-end HTTP tests: the ONE place that actually imports app.api.main
and hits its real routes through FastAPI's TestClient.

Every other test file in this suite calls service/model functions directly
in Python -- none of them would notice if a route were deleted from main.py,
if service.py stopped exposing a method a route calls, or if config.py
stopped exporting a constant a route's dependency chain imports. That is
exactly the class of bug that broke this project's branching-forecast
feature (GET /forecast/{host_id}/branches) for a full day: 145 "passing"
tests, and the route was gone. These tests exercise the real app object
through real HTTP calls so that class of regression fails pytest, not just
a manual browser check.

Self-contained: builds a small real dataset + a few real training steps
(same approach as test_e2e.py) rather than requiring `python -m app.train`
to have been run first, so this stays fast and CI-friendly.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from fastapi.testclient import TestClient

from app import db
from app.config import N_FEATURES, SEQ_LEN, STAGE_CLASSES
from app.data_gen.generator import generate_dataset
from app.labeling.state_labeler import derive_state_labels
from app.features.extraction import host_split, fit_scaler, build_sequences, build_single_window_table
from app.models.lstm_world_model import LSTMWorldModel
from app.models.baseline_lr import train_baseline
from app.train import build_stage_mean_vectors
from app.inference.service import InferenceService
import app.api.main as main_module


@pytest.fixture(scope="module")
def real_service():
    """A genuinely-functional InferenceService, trained briefly on a small
    real synthetic dataset -- not a mock, just a fast/small one."""
    torch.manual_seed(0)
    np.random.seed(0)

    raw = generate_dataset(seed=7, n_benign_hosts=6, n_attack_hosts=2, benign_len=40)
    labeled = derive_state_labels(raw)
    train_hosts, val_hosts, test_hosts = host_split(labeled)
    scaler = fit_scaler(labeled, train_hosts)

    X_train, y_stage_train, y_next_train, _, _ = build_sequences(labeled, scaler, train_hosts)
    model = LSTMWorldModel(n_features=N_FEATURES, hidden_size=8, num_layers=1, n_classes=len(STAGE_CLASSES))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    xb, yb, y_next_b = torch.tensor(X_train), torch.tensor(y_stage_train), torch.tensor(y_next_train)
    for _ in range(3):
        optimizer.zero_grad()
        stage_logits, next_state, _ = model(xb)
        loss = F.cross_entropy(stage_logits, yb) + 0.5 * F.mse_loss(next_state, y_next_b)
        loss.backward()
        optimizer.step()
    model.eval()

    X_base_train, y_base_train, _ = build_single_window_table(labeled, scaler, train_hosts)
    baseline = train_baseline(X_base_train, y_base_train)

    svc = InferenceService()
    svc.model = model
    svc.scaler = scaler
    svc.baseline = baseline
    svc.labeled_df = labeled
    svc.stage_mean_vectors = build_stage_mean_vectors(labeled, scaler, train_hosts)
    svc.shap_explainer = svc._build_shap_explainer()
    svc.ready = True
    return svc


@pytest.fixture()
def client(real_service, tmp_path, monkeypatch):
    """TestClient against the REAL app object (app.api.main.app), with the
    module-level `service` it dispatches to swapped for our fast, real,
    already-trained one, and the DB pointed at a throwaway file."""
    monkeypatch.setattr(main_module, "service", real_service)
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test_api_state.sqlite3")
    db.init_db()
    return TestClient(main_module.app)


def _demo_host_id(real_service) -> str:
    return sorted(real_service.labeled_df["host_id"].unique())[0]


def test_health_reports_ready(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "ready": True}


def test_hosts_lists_real_hosts(client, real_service):
    r = client.get("/hosts")
    assert r.status_code == 200
    hosts = r.json()
    assert len(hosts) > 0
    assert _demo_host_id(real_service) in hosts


def test_unknown_host_returns_404(client):
    r = client.get("/forecast/not-a-real-host")
    assert r.status_code == 404


def test_forecast_route_returns_real_prediction(client, real_service):
    host_id = _demo_host_id(real_service)
    r = client.get(f"/forecast/{host_id}")
    assert r.status_code == 200
    body = r.json()
    assert body["host_id"] == host_id
    assert 0.0 <= body["infiltration_probability_world_model"] <= 1.0
    assert len(body["rollout"]["infiltration_probs_world_model"]) > 0


def test_forecast_branches_route_returns_a_real_tree(client, real_service):
    """The exact route that got silently deleted by a later commit and went
    unnoticed because nothing exercised it over HTTP -- see module docstring."""
    host_id = _demo_host_id(real_service)
    r = client.get(f"/forecast/{host_id}/branches")
    assert r.status_code == 200
    body = r.json()
    assert body["host_id"] == host_id
    assert body["tree"]["children"], "branching tree should have at least one child node"
    assert len(body["paths"]) > 0
    assert body["most_likely_path"] is not None


def test_host_timeline_route(client, real_service):
    host_id = _demo_host_id(real_service)
    r = client.get(f"/host-timeline/{host_id}")
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) > 0
    assert {"window_idx", "true_stage", "state_label"} <= rows[0].keys()


def test_mitigations_route_lists_real_options(client):
    r = client.get("/mitigations")
    assert r.status_code == 200
    ids = {m["id"] for m in r.json()}
    assert "isolate_host" in ids


def test_counterfactual_route_returns_metrics(client, real_service):
    """Guards the compare_with_and_without() 2-tuple -> 3-tuple regression
    found while reconciling this endpoint: the route must unpack and surface
    the real `metrics` block, not just the two rollouts."""
    host_id = _demo_host_id(real_service)
    r = client.get(f"/counterfactual/{host_id}", params={"mitigation_id": "isolate_host"})
    assert r.status_code == 200
    body = r.json()
    assert "metrics" in body
    for key in ["mean_without", "mean_with", "risk_reduction_pct", "verdict"]:
        assert key in body["metrics"]


def test_shap_route_returns_real_contributions(client, real_service):
    host_id = _demo_host_id(real_service)
    r = client.get(f"/shap/{host_id}")
    assert r.status_code == 200
    body = r.json()
    assert len(body["shap"]["contributions"]) > 0


def test_defense_route_returns_a_real_recommendation_or_reason(client, real_service):
    host_id = _demo_host_id(real_service)
    r = client.get(f"/defense/{host_id}")
    assert r.status_code == 200
    body = r.json()
    assert body["risk_level"] in {"act_now", "watch", "monitor"}
    assert body["recommendation_reason"]


def test_attack_mapping_route(client):
    r = client.get("/attack-mapping")
    assert r.status_code == 200
    assert len(r.json()) > 0


def test_stage_classes_route(client):
    r = client.get("/stage-classes")
    assert r.status_code == 200
    assert "benign" in r.json()
