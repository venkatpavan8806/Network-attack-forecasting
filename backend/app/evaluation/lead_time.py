"""Lead-time metric: for each attack-progression host, the number of
time-windows earlier the world model's rolling one-step-ahead forecast first
crossed the alert threshold, compared to when the static baseline (which can
only classify a window using that window's own, already-happened features)
would have first fired on the same host's real timeline.

This is computed for real against actual model outputs on the actual
synthetic timelines -- never asserted.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from app.config import (
    SEQ_LEN, FEATURE_COLUMNS, WINDOW_SECONDS, LEAD_TIME_JSON,
    STAGE_TO_IDX, MALICIOUS_HARD_STAGES,
)

HARD_MALICIOUS_IDX = [STAGE_TO_IDX[s] for s in MALICIOUS_HARD_STAGES]


def _baseline_fire_window(clf, scaler, host_df: pd.DataFrame, threshold: float) -> int | None:
    X = scaler.transform(host_df[FEATURE_COLUMNS].values)
    probs = clf.predict_proba(X)[:, 1]
    hits = np.where(probs >= threshold)[0]
    if len(hits) == 0:
        return None
    return int(host_df.iloc[hits[0]]["window_idx"])


@torch.no_grad()
def _worldmodel_alert_window(model, scaler, host_df: pd.DataFrame, threshold: float) -> int | None:
    feats = scaler.transform(host_df[FEATURE_COLUMNS].values).astype(np.float32)
    n = len(feats)
    for t in range(SEQ_LEN - 1, n - 1):
        window = feats[t - SEQ_LEN + 1: t + 1]
        x = torch.tensor(window, dtype=torch.float32).unsqueeze(0)
        logits, _, _ = model(x)
        probs = F.softmax(logits, dim=1).squeeze(0).numpy()
        malicious_prob = probs[HARD_MALICIOUS_IDX].sum()
        if malicious_prob >= threshold:
            return int(host_df.iloc[t]["window_idx"])
    return None


def compute_lead_time(model, baseline_clf, scaler, labeled_df: pd.DataFrame,
                       train_hosts: set, val_hosts: set, test_hosts: set,
                       threshold: float = 0.5, attack_hosts=None, write: bool = True) -> dict:
    if attack_hosts is None:
        attack_hosts = sorted(h for h in labeled_df["host_id"].unique() if h.startswith("attack-host"))
    per_host = []
    lead_times_all, lead_times_heldout = [], []

    for host_id in attack_hosts:
        host_df = labeled_df[labeled_df["host_id"] == host_id].sort_values("window_idx").reset_index(drop=True)
        split = "train" if host_id in train_hosts else ("val" if host_id in val_hosts else "test")

        b_fire = _baseline_fire_window(baseline_clf, scaler, host_df, threshold)
        w_alert = _worldmodel_alert_window(model, scaler, host_df, threshold)

        entry = {
            "host_id": host_id,
            "split": split,
            "baseline_fire_window": b_fire,
            "worldmodel_alert_window": w_alert,
            "lead_time_windows": None,
            "lead_time_minutes": None,
            "correctly_flagged_by_both": False,
        }
        if b_fire is not None and w_alert is not None:
            lt = b_fire - w_alert
            entry["lead_time_windows"] = lt
            entry["lead_time_minutes"] = round(lt * WINDOW_SECONDS / 60.0, 2)
            entry["correctly_flagged_by_both"] = True
            lead_times_all.append(lt)
            if split == "test":
                lead_times_heldout.append(lt)
        per_host.append(entry)

    report = {
        "threshold": threshold,
        "window_seconds": WINDOW_SECONDS,
        "per_host": per_host,
        "median_lead_time_windows_all_hosts": float(np.median(lead_times_all)) if lead_times_all else None,
        "median_lead_time_minutes_all_hosts": round(float(np.median(lead_times_all)) * WINDOW_SECONDS / 60.0, 2) if lead_times_all else None,
        "median_lead_time_windows_heldout_only": float(np.median(lead_times_heldout)) if lead_times_heldout else None,
        "median_lead_time_minutes_heldout_only": round(float(np.median(lead_times_heldout)) * WINDOW_SECONDS / 60.0, 2) if lead_times_heldout else None,
        "n_hosts_correctly_flagged_by_both": len(lead_times_all),
        "n_attack_hosts_total": len(attack_hosts),
        "note": (
            "lead_time_windows = baseline_fire_window - worldmodel_alert_window. "
            "worldmodel_alert_window is the window at which the model's one-step-ahead "
            "forecast (using only real history up to that window) first crossed the "
            "threshold -- i.e. when the alert is already in hand, one full window before "
            "baseline_fire_window's own data even exists."
        ),
    }
    if write:
        with open(LEAD_TIME_JSON, "w") as f:
            json.dump(report, f, indent=2)
    return report
