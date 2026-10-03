"""Logistic regression baseline: the "traditional network security tool"
the problem statement contrasts the world model against -- classifies a
SINGLE window in isolation as benign/malicious, with no history and no
forecasting. Trained and evaluated on the same normalized feature space as
the LSTM so the benchmark comparison is apples-to-apples.
"""
from __future__ import annotations

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from app.config import BASELINE_WEIGHTS


def train_baseline(X: np.ndarray, y: np.ndarray) -> LogisticRegression:
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    clf.fit(X, y)
    return clf


def save_baseline(clf: LogisticRegression):
    joblib.dump(clf, BASELINE_WEIGHTS)


def load_baseline() -> LogisticRegression:
    return joblib.load(BASELINE_WEIGHTS)
