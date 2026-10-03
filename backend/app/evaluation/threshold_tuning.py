"""Threshold calibration: the fixed 0.5 alert threshold used throughout this
project (benchmark, defense advisor, false-alarm honesty panel) was never
derived from anything -- it's just a round number. On a real network, a
threshold that isn't tuned to how noisy THAT network's normal traffic looks
can flood analysts with false alerts (a busy but benign host can look
"suspicious" by the same yardstick as a quiet one).

This module answers a concrete operational question instead: "if I can only
tolerate N false alerts a day across the hosts I'm watching, what threshold
does that actually require, given what this model thinks NORMAL traffic on
this network looks like?" It does NOT change the project's existing 0.5
default anywhere -- the benchmark, defense advisor, etc. keep using 0.5,
documented as a simple, fixed choice. This is a separate, honest report:
here is the real number you'd need if you wanted a specific alert budget,
computed from the model's actual behavior on real held-out benign traffic,
not asserted.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from app.config import SEQ_LEN, FEATURE_COLUMNS, WINDOW_SECONDS, THRESHOLD_CALIBRATION_JSON

SECONDS_PER_DAY = 86400


@torch.no_grad()
def benign_score_distribution(model, scaler, labeled_df: pd.DataFrame, benign_hosts: set) -> np.ndarray:
    """One infiltration-probability score per window, computed on PURE
    benign hosts only (hosts where no attack ever occurred in ground truth)
    -- this is "what does the model's score look like on traffic we know for
    certain is normal", the same population find_false_alarms draws its
    examples from."""
    from app.config import MALICIOUS_HARD_STAGES
    from app.models.lstm_world_model import infiltration_probability

    scores = []
    for host_id in sorted(benign_hosts):
        host_df = labeled_df[labeled_df["host_id"] == host_id].sort_values("window_idx").reset_index(drop=True)
        if host_df["true_stage"].isin(MALICIOUS_HARD_STAGES).any():
            continue
        feats = scaler.transform(host_df[FEATURE_COLUMNS].values).astype(np.float32)
        n = len(feats)
        if n < SEQ_LEN:
            continue
        windows = np.stack([feats[t - SEQ_LEN + 1: t + 1] for t in range(SEQ_LEN - 1, n)])
        logits, _, _ = model(torch.tensor(windows, dtype=torch.float32))
        probs = F.softmax(logits, dim=1).numpy()
        scores.extend(infiltration_probability(p) for p in probs)
    return np.array(scores, dtype=np.float64)


def threshold_for_fpr_budget(benign_scores: np.ndarray, target_fpr: float) -> float:
    """The threshold that would flag AT MOST `target_fpr` of these benign
    windows: the (1 - target_fpr) quantile of the benign score distribution.
    A threshold at or above this value flags <= target_fpr of normal
    traffic, by construction on THIS sample -- it is a fit to the observed
    distribution, not a guarantee about unseen future traffic."""
    if len(benign_scores) == 0:
        raise ValueError("no benign scores to calibrate against")
    target_fpr = float(np.clip(target_fpr, 0.0, 1.0))
    return float(np.quantile(benign_scores, 1.0 - target_fpr))


def alerts_per_day_at_threshold(benign_scores: np.ndarray, threshold: float, n_hosts_monitored: int,
                                 window_seconds: int = WINDOW_SECONDS) -> float:
    """Expected false alerts/day across n_hosts_monitored hosts, at this
    threshold, extrapolated from the observed false-positive rate on this
    benign sample: fpr_at_threshold * windows_per_day_per_host * n_hosts."""
    if len(benign_scores) == 0:
        return 0.0
    fpr_at_threshold = float(np.mean(benign_scores >= threshold))
    windows_per_day_per_host = SECONDS_PER_DAY / window_seconds
    return fpr_at_threshold * windows_per_day_per_host * n_hosts_monitored


def calibrate_and_report(model, scaler, labeled_df: pd.DataFrame, benign_hosts: set,
                          n_hosts_monitored: int, target_alerts_per_day: tuple[float, ...] = (1.0, 5.0, 20.0),
                          write: bool = True) -> dict:
    """Full report: the benign score distribution's shape, the fixed 0.5
    threshold's OWN implied false-alarm rate on this same population (so the
    current default is directly comparable, not just asserted elsewhere),
    and the threshold required for each target alert-per-day budget."""
    scores = benign_score_distribution(model, scaler, labeled_df, benign_hosts)
    if len(scores) == 0:
        report = {"error": "no pure-benign held-out windows available to calibrate against"}
    else:
        percentiles = [50, 75, 90, 95, 99, 99.9]
        current_default_fpr = float(np.mean(scores >= 0.5))
        report = {
            "n_benign_windows": int(len(scores)),
            "n_hosts_monitored_assumption": n_hosts_monitored,
            "window_seconds": WINDOW_SECONDS,
            "benign_score_percentiles": {f"p{p}": round(float(np.percentile(scores, p)), 4) for p in percentiles},
            "current_fixed_threshold": {
                "threshold": 0.5,
                "false_positive_rate_on_held_out_benign": round(current_default_fpr, 4),
                "implied_alerts_per_day": round(
                    alerts_per_day_at_threshold(scores, 0.5, n_hosts_monitored), 1),
            },
            "budgets": [
                {
                    "target_alerts_per_day": target,
                    "required_threshold": round(
                        threshold_for_fpr_budget(
                            scores, target / (SECONDS_PER_DAY / WINDOW_SECONDS * n_hosts_monitored)
                        ), 4),
                }
                for target in target_alerts_per_day
            ],
            "note": (
                "Computed from this run's held-out (test-split) pure-benign hosts only -- traffic "
                "the model never trained on, but still from the SAME synthetic generator, so these "
                "numbers describe 'how noisy does this model's score look on benign traffic shaped "
                "like our synthetic generator's', not a guarantee about a real network's traffic "
                "mix. required_threshold assumes n_hosts_monitored_assumption hosts, each producing "
                "one window every window_seconds; rescale target_alerts_per_day proportionally for a "
                "different host count. This does not change any threshold actually used elsewhere in "
                "the app (still a fixed 0.5) -- it is a calibration REPORT, not a behavior change."
            ),
        }
    if write:
        with open(THRESHOLD_CALIBRATION_JSON, "w") as f:
            json.dump(report, f, indent=2)
    return report
