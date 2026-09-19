import numpy as np

from app.models.baseline_lr import train_baseline
from app.evaluation.metrics import evaluate_baseline


def test_baseline_trains_and_predicts_reasonably_on_separable_data():
    rng = np.random.default_rng(0)
    n = 200
    X_benign = rng.normal(loc=0.0, scale=1.0, size=(n, 5))
    X_malicious = rng.normal(loc=5.0, scale=1.0, size=(n, 5))
    X = np.vstack([X_benign, X_malicious])
    y = np.concatenate([np.zeros(n), np.ones(n)]).astype(np.int64)

    clf = train_baseline(X, y)
    metrics = evaluate_baseline(clf, X, y)
    assert metrics["f1"] > 0.9
    assert metrics["precision"] > 0.9
    assert metrics["recall"] > 0.9


def test_baseline_metrics_keys_present():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(50, 3))
    y = rng.integers(0, 2, size=50)
    clf = train_baseline(X, y)
    metrics = evaluate_baseline(clf, X, y)
    for key in ["precision", "recall", "f1", "false_positive_rate", "tp", "fp", "fn", "tn", "n_samples"]:
        assert key in metrics
