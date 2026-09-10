"""Benchmark computation: F1 / precision / recall / false-positive rate for
both the LSTM world model and the logistic-regression baseline, computed on
a held-out (by-host) test split -- never hard-coded.

The comparison is deliberately framed to match how each model actually
operates, which is also the whole point of the project:
  - Baseline: classifies window w using window w's OWN features (available
    only once w has already happened).
  - World model: forecasts window w using only history through window w-1
    (available one full window BEFORE w happens) -- this is what makes the
    lead-time metric in lead_time.py meaningful.
Both are scored against the same ground truth: whether window w's raw
`true_stage` is one of the five confirmed hard attack stages. The world
model's "malicious probability" for this strict comparison sums only the
five hard-attack-stage class probabilities (excluding both "benign" and the
derived "ambiguous_pre_attack" class), since ground truth here never counts
ambiguous windows as confirmed-malicious.
"""
from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

from app.config import STAGE_CLASSES, STAGE_TO_IDX, MALICIOUS_HARD_STAGES, BENCHMARK_JSON

HARD_MALICIOUS_IDX = [STAGE_TO_IDX[s] for s in MALICIOUS_HARD_STAGES]


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_positive_rate": fpr,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "n_samples": int(len(y_true)),
    }


def evaluate_baseline(clf, X_test: np.ndarray, y_test: np.ndarray, threshold: float = 0.5) -> dict:
    probs = clf.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)
    return _binary_metrics(y_test, preds)


@torch.no_grad()
def evaluate_lstm(model, X_test: np.ndarray, y_stage_test: np.ndarray, threshold: float = 0.5) -> dict:
    """X_test: (n, seq_len, n_features) sequences ending at w-1.
    y_stage_test: (n,) true class idx of state_label at w (from the labeling
    engine). We instead need raw true_stage malicious/benign at w for a fair
    ground-truth comparison -- caller passes y_binary_hard separately; kept
    here for backward compat of class-idx-based malicious probability calc."""
    logits, _, _ = model(torch.tensor(X_test, dtype=torch.float32))
    probs = F.softmax(logits, dim=1).numpy()
    malicious_prob = probs[:, HARD_MALICIOUS_IDX].sum(axis=1)
    preds = (malicious_prob >= threshold).astype(int)
    return preds, malicious_prob


def run_full_benchmark(baseline_clf, lstm_model, X_base_test, y_base_test,
                        X_lstm_test, y_true_hard_at_w, threshold: float = 0.5,
                        output_path=None, write: bool = True) -> dict:
    baseline_metrics = evaluate_baseline(baseline_clf, X_base_test, y_base_test, threshold)
    lstm_preds, lstm_probs = evaluate_lstm(lstm_model, X_lstm_test, None, threshold)
    lstm_metrics = _binary_metrics(y_true_hard_at_w, lstm_preds)

    report = {
        "threshold": threshold,
        "baseline_logistic_regression": baseline_metrics,
        "world_model_lstm": lstm_metrics,
        "note": (
            "Baseline classifies window w using window w's own features (available "
            "only once w has occurred). World model forecasts window w using only "
            "history through window w-1 (available one full window before w occurs). "
            "Both scored against the same ground truth: true_stage(w) in the five "
            "confirmed hard attack stages."
        ),
    }
    if write:
        with open(output_path or BENCHMARK_JSON, "w") as f:
            json.dump(report, f, indent=2)
    return report
