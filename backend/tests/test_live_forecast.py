from collections import deque

import numpy as np
import pytest
import torch

from app.config import N_FEATURES, SEQ_LEN, STAGE_CLASSES
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
def live_capture_with_history(monkeypatch):
    from app.live.capture import live_capture
    live_capture.running = True
    live_capture.owner = "alice"
    rng = np.random.default_rng(1)
    live_capture.history = {"10.0.0.9": deque([rng.normal(size=N_FEATURES).astype(np.float32) for _ in range(SEQ_LEN)], maxlen=SEQ_LEN)}
    live_capture.window_counter = {"10.0.0.9": 12}
    yield live_capture
    live_capture.running = False
    live_capture.history = {}
    live_capture.window_counter = {}


def test_forecast_live_host_returns_real_shape(live_capture_with_history):
    svc = _make_service()
    result = svc.forecast_live_host("alice", "10.0.0.9")
    assert result["host_id"] == "live:10.0.0.9"
    assert result["window_idx"] == 12
    assert 0.0 <= result["infiltration_probability_world_model"] <= 1.0
    assert len(result["rollout"]["predicted_stage_per_horizon"]) > 0
    assert len(result["rollout"]["branching_forecast"]) > 0
    assert result["true_stage"] is None


def test_forecast_demo_host_dispatches_to_live_for_live_prefixed_id(live_capture_with_history):
    svc = _make_service()
    result = svc.forecast_demo_host("alice", "live:10.0.0.9")
    assert result["host_id"] == "live:10.0.0.9"


def test_forecast_live_host_raises_when_capture_not_running():
    from app.live.capture import live_capture
    live_capture.running = False
    svc = _make_service()
    with pytest.raises(ValueError, match="not running"):
        svc.forecast_live_host("alice", "10.0.0.9")


def test_forecast_live_host_raises_when_not_enough_windows():
    from app.live.capture import live_capture
    live_capture.running = True
    live_capture.owner = "alice"
    live_capture.history = {"10.0.0.9": deque([np.zeros(N_FEATURES, dtype=np.float32)] * 3, maxlen=SEQ_LEN)}
    svc = _make_service()
    try:
        with pytest.raises(ValueError, match="not enough"):
            svc.forecast_live_host("alice", "10.0.0.9")
    finally:
        live_capture.running = False
        live_capture.history = {}


def test_at_window_idx_rejected_for_live_hosts(live_capture_with_history):
    svc = _make_service()
    with pytest.raises(ValueError, match="at_window_idx"):
        svc.forecast_demo_host("alice", "live:10.0.0.9", at_window_idx=5)


def test_run_counterfactual_dispatches_to_live(live_capture_with_history):
    svc = _make_service()
    result = svc.run_counterfactual("alice", "live:10.0.0.9", "isolate_host")
    assert result["host_id"] == "live:10.0.0.9"
    assert result["window_idx"] == 12
    assert "action_divergences" in result


def test_live_hosts_with_predictions_lists_eligible_hosts(live_capture_with_history):
    svc = _make_service()
    assert svc.live_hosts_with_predictions("alice") == ["live:10.0.0.9"]


def test_live_hosts_with_predictions_excludes_hosts_below_threshold():
    from app.live.capture import live_capture
    live_capture.running = True
    live_capture.owner = "alice"
    live_capture.history = {"10.0.0.9": deque([np.zeros(N_FEATURES, dtype=np.float32)] * 3, maxlen=SEQ_LEN)}
    svc = _make_service()
    try:
        assert svc.live_hosts_with_predictions("alice") == []
    finally:
        live_capture.running = False
        live_capture.history = {}


def test_live_hosts_with_predictions_empty_when_capture_not_running():
    from app.live.capture import live_capture
    live_capture.running = False
    svc = _make_service()
    assert svc.live_hosts_with_predictions("alice") == []


def test_live_capture_results_are_only_visible_to_the_user_who_started_it(live_capture_with_history):
    svc = _make_service()
    assert svc.live_hosts_with_predictions("bob") == []
    with pytest.raises(ValueError):
        svc.forecast_live_host("bob", "10.0.0.9")
