import numpy as np

from app.config import FEATURE_COLUMNS, N_FEATURES
from app.explain.shap_baseline import BaselineShapExplainer
from app.models.baseline_lr import train_baseline


def _separable_data(n=400, informative="port_scan_score", seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, N_FEATURES))
    y = (rng.random(n) < 0.5).astype(int)
    j = FEATURE_COLUMNS.index(informative)
    X[:, j] += 4.0 * y  # only this feature separates the classes
    return X, y, j


def test_shap_is_additive_and_matches_the_model():
    X, y, _ = _separable_data()
    clf = train_baseline(X, y)
    ex = BaselineShapExplainer(clf, X[y == 0][:100])

    x = X[y == 1][0]
    out = ex.explain(x, x, top_k=N_FEATURES)
    # SHAP's defining property: base value + all contributions == the model's own output
    assert abs(out["prediction_logit"] - float(clf.decision_function(x.reshape(1, -1))[0])) < 1e-3
    assert abs(out["baseline_probability"] - float(clf.predict_proba(x.reshape(1, -1))[0, 1])) < 1e-3


def test_shap_top_feature_is_the_one_that_actually_separates_the_classes():
    X, y, j = _separable_data(informative="failed_conn_ratio")
    clf = train_baseline(X, y)
    ex = BaselineShapExplainer(clf, X[y == 0][:100])
    out = ex.explain(X[y == 1][0], X[y == 1][0], top_k=3)
    assert out["contributions"][0]["feature"] == "failed_conn_ratio"
    assert out["contributions"][0]["direction"] == "raises_risk"


def test_shap_output_shape_and_sorting():
    X, y, _ = _separable_data()
    clf = train_baseline(X, y)
    ex = BaselineShapExplainer(clf, X[y == 0][:50])
    out = ex.explain(X[0], X[0], top_k=5)
    vals = [abs(c["shap_value"]) for c in out["contributions"]]
    assert len(vals) == 5 and vals == sorted(vals, reverse=True)
