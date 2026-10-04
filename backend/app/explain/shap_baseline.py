"""SHAP explainability, validated ONLY on the logistic-regression baseline
(SHAP on a recurrent sequence model is fragile and explicitly out of scope
per the project brief -- the LSTM instead uses its own attention weights,
see app/explain/attention.py).

Two entry points:
  * explain_baseline()      -- original batch helper (kept unchanged).
  * BaselineShapExplainer   -- builds the SHAP LinearExplainer ONCE (at API
                               startup) and explains one window per call, so
                               the dashboard can ask "why did the baseline
                               score THIS window this way?" in real time.

Reference point ("background"): SHAP always answers "compared to what?".
Here it is a sample of NORMAL traffic windows, so every SHAP value reads as
"how much did this feature push the baseline's score away from what normal
traffic looks like". Values are in log-odds (the logistic regression's native
output space); positive = pushes toward 'malicious', negative = toward 'normal'.

Computation: for a linear model with the (default) independent-features
masker, SHAP values have an exact closed form --
    phi_i = coef_i * (x_i - mean(background_i)),  base = intercept + coef . mean(background)
which is precisely what shap.LinearExplainer returns for this model
(tests/test_shap.py checks the two agree). BaselineShapExplainer computes it
directly so the web server does not have to import the `shap` package
(~190 MB of RAM, too much for a 512 MB hosting instance).
"""
from __future__ import annotations

import numpy as np

from app.config import FEATURE_COLUMNS


def explain_baseline(clf, X_background: np.ndarray, X_query: np.ndarray, top_k: int = 5):
    """X_background: sample of training data for the SHAP linear masker.
    X_query: (n, n_features) rows to explain.
    Returns list of {feature, shap_value} sorted by |shap_value| desc, per row.
    """
    import shap  # heavy import, only needed by this offline helper
    explainer = shap.LinearExplainer(clf, X_background)
    shap_values = explainer.shap_values(X_query)  # (n, n_features)
    results = []
    for row in shap_values:
        pairs = sorted(zip(FEATURE_COLUMNS, row.tolist()), key=lambda p: abs(p[1]), reverse=True)
        results.append([{"feature": f, "shap_value": round(v, 5)} for f, v in pairs[:top_k]])
    return results


def _sigmoid(z: float) -> float:
    return float(1.0 / (1.0 + np.exp(-z)))


class BaselineShapExplainer:
    def __init__(self, clf, X_background: np.ndarray):
        """clf: fitted LogisticRegression. X_background: (n, n_features) SCALED
        normal-traffic windows used as SHAP's reference point."""
        self.clf = clf
        self.coef = np.asarray(clf.coef_, dtype=np.float64).reshape(-1)
        self.background_mean = np.asarray(X_background, dtype=np.float64).mean(axis=0)
        self.base_value = float(np.ravel(clf.intercept_)[0] + self.coef @ self.background_mean)

    def shap_values(self, x_scaled: np.ndarray) -> np.ndarray:
        """Exact linear SHAP values, shape (n, n_features)."""
        return (np.asarray(x_scaled, dtype=np.float64) - self.background_mean) * self.coef

    def explain(self, x_scaled: np.ndarray, x_raw: np.ndarray, top_k: int = 8) -> dict:
        """x_scaled / x_raw: ONE window, shape (n_features,), same feature order
        as FEATURE_COLUMNS. x_raw is only used to show human-readable values."""
        x_scaled = np.asarray(x_scaled, dtype=np.float64).reshape(1, -1)
        vals = self.shap_values(x_scaled)
        if vals.ndim != 2 or vals.shape != x_scaled.shape:
            raise ValueError(f"unexpected SHAP output shape {vals.shape}; expected {x_scaled.shape}")
        vals = vals[0]

        prediction_logit = self.base_value + float(vals.sum())
        order = np.argsort(-np.abs(vals))[:top_k]
        contributions = [{
            "feature": FEATURE_COLUMNS[i],
            "raw_value": round(float(np.asarray(x_raw).reshape(-1)[i]), 4),
            "scaled_value": round(float(x_scaled[0, i]), 3),   # z-score vs. training data
            "shap_value": round(float(vals[i]), 4),
            "direction": "raises_risk" if vals[i] > 0 else "lowers_risk",
        } for i in order]

        return {
            "base_value_logit": round(self.base_value, 4),
            "prediction_logit": round(prediction_logit, 4),
            "baseline_probability": round(_sigmoid(prediction_logit), 4),
            "sum_of_all_shap_values": round(float(vals.sum()), 4),
            "contributions": contributions,
        }
