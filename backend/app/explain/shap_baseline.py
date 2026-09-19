"""SHAP explainability, validated ONLY on the logistic-regression baseline
(SHAP on a recurrent sequence model is fragile and explicitly out of scope
per the project brief -- the LSTM instead uses its own attention weights,
see app/explain/attention.py).
"""
from __future__ import annotations

import numpy as np
import shap

from app.config import FEATURE_COLUMNS


def explain_baseline(clf, X_background: np.ndarray, X_query: np.ndarray, top_k: int = 5):
    """X_background: sample of training data for the SHAP linear masker.
    X_query: (n, n_features) rows to explain.
    Returns list of {feature, shap_value} sorted by |shap_value| desc, per row.
    """
    explainer = shap.LinearExplainer(clf, X_background)
    shap_values = explainer.shap_values(X_query)  # (n, n_features)
    results = []
    for row in shap_values:
        pairs = sorted(zip(FEATURE_COLUMNS, row.tolist()), key=lambda p: abs(p[1]), reverse=True)
        results.append([{"feature": f, "shap_value": round(v, 5)} for f, v in pairs[:top_k]])
    return results
