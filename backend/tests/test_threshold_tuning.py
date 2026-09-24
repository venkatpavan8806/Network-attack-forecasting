import numpy as np
import torch

from app.config import N_FEATURES, SEQ_LEN, STAGE_CLASSES
from app.models.lstm_world_model import LSTMWorldModel
from app.evaluation.threshold_tuning import (
    threshold_for_fpr_budget, alerts_per_day_at_threshold, benign_score_distribution,
    calibrate_and_report,
)


def _dummy_model():
    torch.manual_seed(0)
    return LSTMWorldModel(n_features=N_FEATURES, hidden_size=8, num_layers=1, n_classes=len(STAGE_CLASSES))


def test_threshold_for_fpr_budget_zero_means_never_alert():
    scores = np.array([0.1, 0.2, 0.3, 0.9])
    t = threshold_for_fpr_budget(scores, target_fpr=0.0)
    assert t >= scores.max()


def test_threshold_for_fpr_budget_picks_the_right_quantile():
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    # allowing 20% FPR on 10 evenly-spaced points -> threshold near the 80th percentile
    t = threshold_for_fpr_budget(scores, target_fpr=0.2)
    fpr_at_t = float(np.mean(scores >= t))
    assert fpr_at_t <= 0.2 + 1e-9


def test_stricter_budget_requires_a_higher_threshold():
    scores = np.linspace(0.0, 1.0, 1000)
    loose = threshold_for_fpr_budget(scores, target_fpr=0.5)
    strict = threshold_for_fpr_budget(scores, target_fpr=0.01)
    assert strict > loose


def test_alerts_per_day_scales_with_hosts_monitored():
    scores = np.array([0.9] * 100)  # everything triggers a 0.5 threshold
    one_host = alerts_per_day_at_threshold(scores, threshold=0.5, n_hosts_monitored=1)
    ten_hosts = alerts_per_day_at_threshold(scores, threshold=0.5, n_hosts_monitored=10)
    assert round(ten_hosts / one_host, 4) == 10.0


def test_alerts_per_day_empty_scores_is_zero():
    assert alerts_per_day_at_threshold(np.array([]), threshold=0.5, n_hosts_monitored=5) == 0.0


def test_benign_score_distribution_only_uses_pure_benign_hosts(monkeypatch):
    import pandas as pd
    model = _dummy_model()
    scaler = type("Identity", (), {"transform": staticmethod(lambda X: np.asarray(X, dtype=np.float32))})()

    from app.config import FEATURE_COLUMNS
    rows = []
    for host, has_attack in [("benign-host-000", False), ("attack-host-000", True)]:
        stage = "port_scan" if has_attack else "benign"
        for w in range(SEQ_LEN + 2):
            row = {"host_id": host, "window_idx": w, "true_stage": stage if w == SEQ_LEN else "benign"}
            for c in FEATURE_COLUMNS:
                row[c] = 0.0
            rows.append(row)
    df = pd.DataFrame(rows)

    scores = benign_score_distribution(model, scaler, df, {"benign-host-000", "attack-host-000"})
    # attack-host-000 has a non-benign true_stage row -> excluded entirely
    expected_windows_from_pure_benign_host = (SEQ_LEN + 2) - SEQ_LEN + 1
    assert len(scores) == expected_windows_from_pure_benign_host
    assert np.isfinite(scores).all()
    assert ((scores >= 0.0) & (scores <= 1.0)).all()


def test_calibrate_and_report_is_internally_consistent(monkeypatch, tmp_path):
    import app.evaluation.threshold_tuning as mod
    monkeypatch.setattr(mod, "THRESHOLD_CALIBRATION_JSON", tmp_path / "threshold.json")

    model = _dummy_model()
    scaler = type("Identity", (), {"transform": staticmethod(lambda X: np.asarray(X, dtype=np.float32))})()
    import pandas as pd
    from app.config import FEATURE_COLUMNS
    rows = []
    for w in range(30):
        row = {"host_id": "benign-host-000", "window_idx": w, "true_stage": "benign"}
        for c in FEATURE_COLUMNS:
            row[c] = float(np.sin(w))
        rows.append(row)
    df = pd.DataFrame(rows)

    report = calibrate_and_report(model, scaler, df, {"benign-host-000"}, n_hosts_monitored=5,
                                   target_alerts_per_day=(1.0, 10.0))
    assert report["n_benign_windows"] > 0
    assert len(report["budgets"]) == 2
    # a tighter (smaller) alert budget must require an equal-or-higher threshold
    by_target = {b["target_alerts_per_day"]: b["required_threshold"] for b in report["budgets"]}
    assert by_target[1.0] >= by_target[10.0]
    assert (tmp_path / "threshold.json").exists()
