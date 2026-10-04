"""Benchmarks computed from ONE user's own uploaded, labelled traffic.

The Benchmarks page shows the same reports the training pipeline produces
(F1/precision/recall/FPR vs. the baseline, calibration, lead time, step-by-
step tracking, alert-threshold calibration), but evaluated on the hosts this
user uploaded with ground-truth `true_stage` labels -- not on the training
run. Nothing is written to disk; results are cached per user and recomputed
after each upload. A user with no labelled hosts gets no reports.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.config import MALICIOUS_HARD_STAGES
from app.evaluation.calibration import compute_calibration
from app.evaluation.lead_time import compute_lead_time
from app.evaluation.metrics import run_full_benchmark
from app.evaluation.threshold_tuning import calibrate_and_report
from app.features.extraction import build_sequences, build_single_window_table

SOURCE_NOTE = "Computed on your own uploaded, labelled traffic (hosts with a true_stage column)."


def labelled_part(df: pd.DataFrame) -> pd.DataFrame:
    """Hosts whose every window has ground truth (true_stage) and a state label."""
    if df is None or len(df) == 0:
        return df.iloc[0:0] if df is not None else pd.DataFrame()
    ok = df.groupby("host_id").apply(
        lambda g: g["true_stage"].notna().all() and g["state_label"].notna().all(), include_groups=False)
    keep = set(ok[ok].index)
    return df[df["host_id"].isin(keep)].copy()


def build_user_reports(model, baseline, scaler, ngram, df: pd.DataFrame) -> dict:
    """Returns {"benchmark", "calibration", "lead_time", "threshold", "step_tracking"};
    each value is a report dict, or None when the user's data can't support it."""
    from app.evaluate_step_tracking import evaluate_window_tracking, evaluate_cold_start, evaluate_multistep

    labeled = labelled_part(df)
    empty = {"benchmark": None, "calibration": None, "lead_time": None, "threshold": None, "step_tracking": None}
    if len(labeled) == 0:
        return empty
    hosts = set(labeled["host_id"].unique())
    malicious = labeled.groupby("host_id")["true_stage"].apply(lambda s: s.isin(MALICIOUS_HARD_STAGES).any())
    attack_hosts = sorted(malicious[malicious].index)
    benign_hosts = set(malicious[~malicious].index)
    out = dict(empty)

    # -- F1 / precision / recall / FPR: world model vs. baseline ---------------
    X_base, y_base, _ = build_single_window_table(labeled, scaler, hosts)
    X_seq, _, _, _, meta = build_sequences(labeled, scaler, hosts)
    if len(X_seq):
        idx = labeled.set_index(["host_id", "window_idx"])["true_stage"]
        y_hard = np.array([1 if idx.loc[(m["host_id"], m["window_idx_next"])] in MALICIOUS_HARD_STAGES else 0
                           for m in meta])
        rep = run_full_benchmark(baseline, model, X_base, y_base, X_seq, y_hard, write=False)
        rep["note"] = SOURCE_NOTE + " " + rep["note"]
        rep["n_hosts"] = len(hosts)
        out["benchmark"] = rep

    # -- calibration @ horizon 3 ------------------------------------------------
    cal = compute_calibration(model, labeled, scaler, hosts, horizon=3, write=False)
    if cal["n_points"] > 0:
        out["calibration"] = cal

    # -- lead time vs. baseline (attack hosts only) -----------------------------
    if attack_hosts:
        lt = compute_lead_time(model, baseline, scaler, labeled, set(), set(), hosts,
                               attack_hosts=attack_hosts, write=False)
        for row in lt["per_host"]:
            row["split"] = "uploaded"
        lt["note"] = SOURCE_NOTE + " " + lt["note"]
        out["lead_time"] = lt

    # -- alert-threshold calibration (benign hosts only) ------------------------
    if benign_hosts:
        th = calibrate_and_report(model, scaler, labeled, benign_hosts, n_hosts_monitored=len(hosts), write=False)
        if "error" not in th:
            th["note"] = (SOURCE_NOTE + " Uses your fully benign hosts' windows. Does not change the 0.5 "
                          "threshold used elsewhere in the app -- it is a calibration report.")
            out["threshold"] = th

    # -- step-by-step tracking + next 1/2/3 moves --------------------------------
    long_hosts = {h for h in hosts if (labeled["host_id"] == h).sum() > 1}
    if long_hosts:
        window, paths = evaluate_window_tracking(model, scaler, labeled, long_hosts, attack_hosts=set(attack_hosts))
        cold = evaluate_cold_start(model, scaler, labeled, long_hosts, attack_hosts=set(attack_hosts))
        multi = (evaluate_multistep(model, scaler, labeled, set(), set(attack_hosts), attack_hosts=attack_hosts,
                                    trained_ngram=ngram, save=False)
                 if attack_hosts and ngram is not None else None)
        out["step_tracking"] = {
            "source": SOURCE_NOTE,
            "decision_metric": {
                "next_step_rule": "argmax over the LSTM softmax: predicted = argmax_a P(action at t+1 = a | windows 1..t)",
                "confidence": "the winning softmax probability P(predicted action)",
                "infiltration_probability": "1 - P(benign), alert when >= 0.5",
                "next_moves": "trained trigram P(m_{i+1} | m_{i-1}, m_i) x LSTM evidence for the first move",
                "evaluation_metrics": "top-1 / top-3 accuracy, macro-F1, accuracy on transitions, "
                                      "exact-sequence match for next-k moves",
            },
            "window_level_tracking": window,
            "cold_start": cold,
            "path_recognition": paths,
            "multi_step_moves": multi,
        }
    return out
