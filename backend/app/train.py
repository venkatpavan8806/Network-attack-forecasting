"""End-to-end, reproducible training script.

Run with:  python -m app.train

Produces (all real, computed from this run -- see config.py for exact paths):
  data/synthetic_telemetry.csv     raw simulator output
  data/labeled_states.csv          + state-labeling engine output
  data/transition_pairs.csv        (S_t -> S_t+1) pairs
  models_store/lstm_world_model.pt + meta json
  models_store/baseline_logreg.joblib
  models_store/feature_scaler.joblib
  data/benchmark_report.json       F1 / precision / recall / FPR, both models
  data/calibration_report.json     reliability diagram @ horizon 3
  data/lead_time_report.json       median lead-time vs baseline
  data/false_alarm_examples.json   real false-alarm examples from test hosts
  data/recent_forecast_log.json    sample of real inference rows for the UI table
  data/stage_mean_vectors.json     per-action mean feature vector (train hosts only),
                                    used by branching_rollout() to diverge sibling branches
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from app.config import (
    RANDOM_SEED, SYNTHETIC_CSV, DATA_DIR, SEQ_LEN, N_FEATURES, STAGE_CLASSES,
    STAGE_TO_IDX, IDX_TO_STAGE, MALICIOUS_HARD_STAGES, FORECAST_LOG_JSON,
    FEATURE_COLUMNS, STAGE_MEAN_VECTORS_JSON,
)
from app.data_gen.generator import generate_dataset
from app.labeling.state_labeler import derive_state_labels, build_transition_pairs
from app.features.extraction import (
    host_split, fit_scaler, save_scaler, build_sequences, build_single_window_table,
)
from app.models.lstm_world_model import LSTMWorldModel, save_model
from app.models.baseline_lr import train_baseline, save_baseline
from app.evaluation.metrics import run_full_benchmark
from app.evaluation.calibration import compute_calibration
from app.evaluation.lead_time import compute_lead_time
from app.evaluation.false_alarms import find_false_alarms
from app.models.attack_mapping import map_stage

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def train_lstm(X_train, y_stage_train, y_next_train, X_val, y_stage_val, y_next_val,
                hidden_size=64, num_layers=1, epochs=60, batch_size=64, lr=1e-3):
    model = LSTMWorldModel(n_features=N_FEATURES, hidden_size=hidden_size, num_layers=num_layers,
                            n_classes=len(STAGE_CLASSES))

    class_counts = np.bincount(y_stage_train, minlength=len(STAGE_CLASSES)).astype(np.float32)
    class_weights = 1.0 / np.clip(class_counts, 1, None)
    class_weights = class_weights / class_weights.sum() * len(STAGE_CLASSES)
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32)

    train_ds = TensorDataset(
        torch.tensor(X_train), torch.tensor(y_stage_train), torch.tensor(y_next_train)
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    X_val_t = torch.tensor(X_val)
    y_stage_val_t = torch.tensor(y_stage_val)
    y_next_val_t = torch.tensor(y_next_val)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float("inf")
    best_state = None
    patience, bad_epochs = 8, 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, y_stage_b, y_next_b in train_loader:
            optimizer.zero_grad()
            stage_logits, next_state, _ = model(xb)
            ce = F.cross_entropy(stage_logits, y_stage_b, weight=class_weights_t)
            mse = F.mse_loss(next_state, y_next_b)
            loss = ce + 0.5 * mse
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        train_loss = total_loss / len(train_ds)

        model.eval()
        with torch.no_grad():
            stage_logits, next_state, _ = model(X_val_t)
            val_ce = F.cross_entropy(stage_logits, y_stage_val_t, weight=class_weights_t)
            val_mse = F.mse_loss(next_state, y_next_val_t)
            val_loss = (val_ce + 0.5 * val_mse).item()
            val_acc = (stage_logits.argmax(dim=1) == y_stage_val_t).float().mean().item()

        print(f"epoch {epoch+1:3d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"early stopping at epoch {epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, hidden_size, num_layers


def build_recent_forecast_log(lstm_model, baseline_clf, scaler, labeled_df, test_hosts, n_rows=25):
    from app.config import FEATURE_COLUMNS
    from app.models.lstm_world_model import one_step_forecast, infiltration_probability

    rows = []
    for host_id in sorted(test_hosts):
        host_df = labeled_df[labeled_df["host_id"] == host_id].sort_values("window_idx").reset_index(drop=True)
        feats = scaler.transform(host_df[FEATURE_COLUMNS].values).astype(np.float32)
        n = len(feats)
        for t in range(SEQ_LEN - 1, n - 1):
            window = feats[t - SEQ_LEN + 1: t + 1]
            probs, _, _ = one_step_forecast(lstm_model, window)
            lstm_prob = infiltration_probability(probs)
            pred_stage = IDX_TO_STAGE[int(np.argmax(probs))]

            base_prob = float(baseline_clf.predict_proba(feats[t + 1:t + 2])[0, 1])

            rows.append({
                "host_id": host_id,
                "window_idx": int(host_df.iloc[t + 1]["window_idx"]),
                "timestamp": int(host_df.iloc[t + 1]["timestamp"]),
                "predicted_stage": pred_stage,
                "infiltration_probability_world_model": round(lstm_prob, 4),
                "infiltration_probability_baseline": round(base_prob, 4),
                "true_stage": host_df.iloc[t + 1]["true_stage"],
                "state_label": host_df.iloc[t + 1]["state_label"],
            })
    rows = sorted(rows, key=lambda r: (r["host_id"], r["window_idx"]))
    # sample a representative slice: mix of benign and the attack host's most eventful windows
    attack_rows = [r for r in rows if r["true_stage"] != "benign" or r["state_label"] != "benign"]
    benign_rows = [r for r in rows if r not in attack_rows]
    sample = (attack_rows[:n_rows // 2] + benign_rows[:n_rows - len(attack_rows[:n_rows // 2])])[:n_rows]
    with open(FORECAST_LOG_JSON, "w") as f:
        json.dump(sample, f, indent=2)
    return sample


def build_stage_mean_vectors(labeled_df, scaler, train_hosts) -> dict:
    """Per-action mean NORMALIZED feature vector, computed on train hosts
    only (same host-level split the scaler itself is fit on -- no test/val
    leakage). Used only by branching_rollout() at inference time to make
    sibling branches of the attack-forecast tree diverge plausibly; it is
    a descriptive prototype, not a learned parameter, and plays no part in
    training or in any reported benchmark/calibration/lead-time metric. A
    class with zero rows among train hosts (possible on a tiny/unlucky
    split) falls back to its mean over the FULL dataset so branching_rollout
    always has a vector for every stage the model can predict."""
    train_df = labeled_df[labeled_df["host_id"].isin(train_hosts)]
    vectors = {}
    for stage in STAGE_CLASSES:
        rows = train_df[train_df["state_label"] == stage]
        if len(rows) == 0:
            rows = labeled_df[labeled_df["state_label"] == stage]
        if len(rows) == 0:
            continue  # stage never occurs in this run's data at all
        vectors[stage] = scaler.transform(rows[FEATURE_COLUMNS].values).astype(np.float32).mean(axis=0).tolist()
    with open(STAGE_MEAN_VECTORS_JSON, "w") as f:
        json.dump(vectors, f, indent=2)
    return vectors


def main():
    t0 = time.time()
    print("== generating synthetic dataset ==")
    raw = generate_dataset(seed=RANDOM_SEED)
    raw.to_csv(SYNTHETIC_CSV, index=False)
    print(f"{len(raw)} rows, stage counts:\n{raw['true_stage'].value_counts()}")

    print("\n== deriving state labels (incl. ambiguous_pre_attack) ==")
    labeled = derive_state_labels(raw)
    labeled.to_csv(DATA_DIR / "labeled_states.csv", index=False)
    n_relabeled = ((labeled["true_stage"] == "benign") & (labeled["state_label"] == "ambiguous_pre_attack")).sum()
    print(f"state_label counts:\n{labeled['state_label'].value_counts()}")
    print(f"windows relabeled benign -> ambiguous_pre_attack: {n_relabeled}")

    pairs = build_transition_pairs(labeled)
    pairs.to_csv(DATA_DIR / "transition_pairs.csv", index=False)

    print("\n== host split ==")
    train_hosts, val_hosts, test_hosts = host_split(labeled)
    print(f"train hosts: {len(train_hosts)}  val hosts: {len(val_hosts)}  test hosts: {len(test_hosts)}")
    print(f"test hosts: {sorted(test_hosts)}")

    print("\n== fitting scaler on train hosts ==")
    scaler = fit_scaler(labeled, train_hosts)
    save_scaler(scaler)

    print("\n== computing per-action mean feature vectors (for branching rollout) ==")
    stage_means = build_stage_mean_vectors(labeled, scaler, train_hosts)
    print(f"stage_mean_vectors.json: {len(stage_means)}/{len(STAGE_CLASSES)} action classes covered")

    print("\n== building LSTM sequences ==")
    X_train, y_stage_train, y_next_train, y_cur_train, meta_train = build_sequences(labeled, scaler, train_hosts)
    X_val, y_stage_val, y_next_val, y_cur_val, meta_val = build_sequences(labeled, scaler, val_hosts)
    X_test, y_stage_test, y_next_test, y_cur_test, meta_test = build_sequences(labeled, scaler, test_hosts)
    print(f"train sequences: {X_train.shape}  val: {X_val.shape}  test: {X_test.shape}")

    print("\n== training LSTM world model ==")
    lstm_model, hidden_size, num_layers = train_lstm(
        X_train, y_stage_train, y_next_train, X_val, y_stage_val, y_next_val,
    )
    save_model(lstm_model, hidden_size, num_layers)

    print("\n== training logistic regression baseline ==")
    X_base_train, y_base_train, meta_base_train = build_single_window_table(labeled, scaler, train_hosts)
    X_base_test, y_base_test, meta_base_test = build_single_window_table(labeled, scaler, test_hosts)
    baseline_clf = train_baseline(X_base_train, y_base_train)
    save_baseline(baseline_clf)

    print("\n== benchmark: F1 / precision / recall / FPR ==")
    # ground truth for LSTM test predictions: true_stage at the FORECASTED window (window_idx_next)
    labeled_idx = labeled.set_index(["host_id", "window_idx"])
    y_true_hard_at_w = np.array([
        1 if labeled_idx.loc[(m["host_id"], m["window_idx_next"]), "true_stage"] in MALICIOUS_HARD_STAGES else 0
        for m in meta_test
    ])
    report = run_full_benchmark(baseline_clf, lstm_model, X_base_test, y_base_test, X_test, y_true_hard_at_w)
    print(json.dumps(report, indent=2))

    print("\n== calibration (reliability diagram @ horizon 3) ==")
    calib = compute_calibration(lstm_model, labeled, scaler, test_hosts | val_hosts, horizon=3)
    print(f"brier score: {calib['brier_score']:.4f}  n_points: {calib['n_points']}")

    print("\n== lead-time metric ==")
    lead_time_report = compute_lead_time(lstm_model, baseline_clf, scaler, labeled, train_hosts, val_hosts, test_hosts)
    print(json.dumps(lead_time_report, indent=2))

    print("\n== false-alarm honesty examples ==")
    benign_test_hosts = {h for h in test_hosts if h.startswith("benign-host")}
    false_alarms = find_false_alarms(lstm_model, scaler, labeled, benign_test_hosts)
    print(f"found {len(false_alarms)} false-alarm example(s) on held-out benign hosts")

    print("\n== building recent forecast log for UI ==")
    build_recent_forecast_log(lstm_model, baseline_clf, scaler, labeled, test_hosts)

    print("\n== step-by-step tracking + next 1/2/3-move evaluation ==")
    from app.evaluate_step_tracking import main as evaluate_step_tracking
    evaluate_step_tracking()

    print(f"\nTotal training pipeline time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
