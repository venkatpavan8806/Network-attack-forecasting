"""Reliability diagram: predicted infiltration probability vs. observed
frequency of an actual attack, at a fixed rollout horizon. Computed from
real held-out K-step rollouts -- this is what actually demonstrates
"forecasting, not a relabeled classifier": if the model were just a
classifier wearing a forecasting costume, the predicted probabilities at a
multi-step horizon would not track observed frequency at all.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from app.config import (
    SEQ_LEN, CALIBRATION_JSON, MALICIOUS_HARD_STAGES,
)
from app.models.lstm_world_model import rollout


def compute_calibration(model, labeled_df: pd.DataFrame, scaler, hosts: set,
                         horizon: int, n_bins: int = 10) -> dict:
    from app.config import FEATURE_COLUMNS

    predicted, observed_label = [], []
    for host_id, host_df in labeled_df[labeled_df["host_id"].isin(hosts)].groupby("host_id", sort=False):
        host_df = host_df.sort_values("window_idx").reset_index(drop=True)
        feats = scaler.transform(host_df[FEATURE_COLUMNS].values).astype(np.float32)
        true_stage = host_df["true_stage"].values
        n = len(host_df)
        for t0 in range(SEQ_LEN - 1, n - horizon):
            seed = feats[t0 - SEQ_LEN + 1: t0 + 1]
            result = rollout(model, seed, k=horizon)
            pred_prob = result["infiltration_probs"][horizon - 1]
            target_idx = t0 + horizon
            is_malicious = true_stage[target_idx] in MALICIOUS_HARD_STAGES
            predicted.append(pred_prob)
            observed_label.append(1 if is_malicious else 0)

    predicted = np.array(predicted)
    observed_label = np.array(observed_label)

    bins = np.linspace(0, 1, n_bins + 1)
    bin_stats = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (predicted >= lo) & (predicted < hi) if i < n_bins - 1 else (predicted >= lo) & (predicted <= hi)
        count = int(mask.sum())
        if count == 0:
            bin_stats.append({"bin_lo": float(lo), "bin_hi": float(hi), "count": 0,
                               "mean_predicted": None, "observed_frequency": None})
        else:
            bin_stats.append({
                "bin_lo": float(lo), "bin_hi": float(hi), "count": count,
                "mean_predicted": float(predicted[mask].mean()),
                "observed_frequency": float(observed_label[mask].mean()),
            })

    report = {
        "horizon_windows": horizon,
        "n_points": int(len(predicted)),
        "bins": bin_stats,
        "brier_score": float(np.mean((predicted - observed_label) ** 2)),
    }
    with open(CALIBRATION_JSON, "w") as f:
        json.dump(report, f, indent=2)
    return report
