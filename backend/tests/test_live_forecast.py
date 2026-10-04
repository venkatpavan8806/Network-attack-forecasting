"""Live capture windows (from the capture agent or local capture) are saved to
the user's own hosts as live:<ip>, predicted from the first window, and then
behave like any other host on every page."""
import numpy as np
import pytest
import torch

from app import db
from app.config import FEATURE_COLUMNS, N_FEATURES, SEQ_LEN, STAGE_CLASSES
from app.models.lstm_world_model import LSTMWorldModel
from app.models.baseline_lr import train_baseline
from app.inference.service import InferenceService


class _IdentityScaler:
    def transform(self, X):
        return np.asarray(X, dtype=np.float32)

    def inverse_transform(self, X):
        return np.asarray(X, dtype=np.float32)


def _make_service():
    torch.manual_seed(0)
    svc = InferenceService()
    svc.model = LSTMWorldModel(n_features=N_FEATURES, hidden_size=8, num_layers=1, n_classes=len(STAGE_CLASSES))
    svc.scaler = _IdentityScaler()
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, N_FEATURES))
    y = rng.integers(0, 2, size=40)
    svc.baseline = train_baseline(X, y)
    svc.ready = True
    return svc


@pytest.fixture()
def svc(tmp_path):
    db.configure(f"sqlite:///{(tmp_path / 'live.sqlite3').as_posix()}")
    db.init_db()
    return _make_service()


def _features(rng):
    return {c: float(abs(v)) for c, v in zip(FEATURE_COLUMNS, rng.normal(size=N_FEATURES))}


def test_first_window_is_predicted_and_saved_as_a_live_host(svc):
    entry = svc.ingest_live_window("alice", "10.0.0.9", _features(np.random.default_rng(1)))
    assert entry["host_id"] == "live:10.0.0.9" and entry["window_idx"] == 0
    assert entry["warmup"] is True and entry["history_windows_used"] == 1
    assert entry["predicted_stage"] in STAGE_CLASSES
    assert 0.0 <= entry["infiltration_probability_world_model"] <= 1.0
    assert svc.list_demo_hosts("alice") == ["live:10.0.0.9"]
    assert db.recent_live_entries("alice")[0]["window_idx"] == 0
    assert db.recent_forecast_log("alice")[0]["host_id"] == "live:10.0.0.9"


def test_live_host_works_on_every_page_once_it_has_history(svc):
    rng = np.random.default_rng(2)
    for _ in range(SEQ_LEN - 1):
        svc.ingest_live_window("alice", "10.0.0.9", _features(rng))
    with pytest.raises(ValueError, match="not enough yet"):
        svc.forecast_demo_host("alice", "live:10.0.0.9")
    assert svc.live_hosts_with_predictions("alice") == []

    last = svc.ingest_live_window("alice", "10.0.0.9", _features(rng))
    assert last["warmup"] is False and last["history_windows_used"] == SEQ_LEN
    f = svc.forecast_demo_host("alice", "live:10.0.0.9")
    assert f["host_id"] == "live:10.0.0.9" and len(f["rollout"]["infiltration_probs_world_model"]) > 0
    assert svc.live_hosts_with_predictions("alice") == ["live:10.0.0.9"]
    assert len(svc.track_attacker("alice", "live:10.0.0.9")["steps"]) == SEQ_LEN
    cf = svc.run_counterfactual("alice", "live:10.0.0.9", "isolate_host")
    assert "metrics" in cf


def test_live_hosts_are_private_to_their_user(svc):
    svc.ingest_live_window("alice", "10.0.0.9", _features(np.random.default_rng(3)))
    assert svc.list_demo_hosts("bob") == []
    assert db.recent_live_entries("bob") == []
    with pytest.raises(ValueError, match="unknown host_id"):
        svc.track_attacker("bob", "live:10.0.0.9")


def test_recent_live_entries_are_newest_first(svc):
    rng = np.random.default_rng(4)
    svc.ingest_live_window("alice", "10.0.0.9", _features(rng))
    svc.ingest_live_window("alice", "10.0.0.8", _features(rng))
    entries = db.recent_live_entries("alice")
    assert [e["host_id"] for e in entries] == ["live:10.0.0.8", "live:10.0.0.9"]
