"""Shared fixtures: a small but REALLY trained service (not a mock), and an
HTTP client against the real FastAPI app with an isolated database and
dev-mode auth (each request names its user via the X-Dev-User header)."""
from __future__ import annotations

import os

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from fastapi.testclient import TestClient

from app import auth, db
from app.config import FEATURE_COLUMNS, N_FEATURES, STAGE_CLASSES
from app.data_gen.generator import generate_dataset
from app.labeling.state_labeler import derive_state_labels
from app.features.extraction import host_split, fit_scaler, build_sequences, build_single_window_table
from app.models.lstm_world_model import LSTMWorldModel
from app.models.baseline_lr import train_baseline
from app.train import build_stage_mean_vectors
from app.inference.service import InferenceService


@pytest.fixture(scope="session")
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
    svc.ngram = svc._load_ngram()
    svc.ready = True
    return svc


def make_test_db(tmp_path):
    """SQLite file per test by default; set TEST_DATABASE_URL to run the same
    tests against a real Postgres (what Supabase runs) -- tables are dropped
    and recreated for every test."""
    url = os.environ.get("TEST_DATABASE_URL")
    if url:
        db.configure(url)
        db.metadata.drop_all(db.engine())
    else:
        db.configure(f"sqlite:///{(tmp_path / 'state.sqlite3').as_posix()}")
    db.init_db()
    return db


@pytest.fixture()
def fresh_db(tmp_path):
    return make_test_db(tmp_path)


@pytest.fixture()
def client(real_service, fresh_db, monkeypatch):
    """TestClient against the REAL app object, with the module-level
    `service` swapped for the fast, real, already-trained one."""
    import app.api.main as main_module
    monkeypatch.setattr(main_module, "service", real_service)
    monkeypatch.setattr(auth, "CONFIG", auth.AuthConfig(mode="dev", supabase_url=None, jwt_secret=None))
    return TestClient(main_module.app)


def load_training_hosts(svc, user_id: str, host_ids=None):
    """Puts hosts from the (tiny) training dataset into one user's workspace."""
    df = svc.labeled_df
    for host_id in host_ids or sorted(df["host_id"].unique()):
        g = df[df["host_id"] == host_id].sort_values("window_idx")
        rows = [{"features": {c: float(r[c]) for c in FEATURE_COLUMNS}, "true_stage": r["true_stage"],
                 "state_label": r["state_label"]} for _, r in g.iterrows()]
        svc.ingest_windows(user_id, host_id, rows, source="sample")


def as_user(user_id: str) -> dict:
    return {"X-Dev-User": user_id}
