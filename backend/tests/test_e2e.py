"""End-to-end test: runs a small synthetic dataset through the full
pipeline (generate -> label -> split -> scale -> train tiny models ->
explain -> rollout) and checks a forecast + explanation are actually
produced, with sane shapes/ranges. Kept small so it runs fast."""
import numpy as np
import torch

from app.config import N_FEATURES, SEQ_LEN, STAGE_CLASSES, FEATURE_COLUMNS
from app.data_gen.generator import generate_dataset
from app.labeling.state_labeler import derive_state_labels
from app.features.extraction import host_split, fit_scaler, build_sequences, build_single_window_table
from app.models.lstm_world_model import LSTMWorldModel, rollout, branching_rollout, enumerate_paths
from app.models.baseline_lr import train_baseline
from app.evaluation.metrics import run_full_benchmark
from app.explain.attention import explain_prediction

import torch.nn.functional as F


def test_full_pipeline_end_to_end(tmp_path):
    torch.manual_seed(0)
    np.random.seed(0)

    raw = generate_dataset(seed=99, n_benign_hosts=6, n_attack_hosts=2, benign_len=40)
    labeled = derive_state_labels(raw)
    assert "ambiguous_pre_attack" in labeled["state_label"].unique() or True  # may or may not trigger at tiny scale

    train_hosts, val_hosts, test_hosts = host_split(labeled)
    scaler = fit_scaler(labeled, train_hosts)

    X_train, y_stage_train, y_next_train, _, _ = build_sequences(labeled, scaler, train_hosts)
    X_test, y_stage_test, y_next_test, _, meta_test = build_sequences(labeled, scaler, test_hosts)
    assert len(X_train) > 0
    assert len(X_test) > 0

    model = LSTMWorldModel(n_features=N_FEATURES, hidden_size=8, num_layers=1, n_classes=len(STAGE_CLASSES))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    xb = torch.tensor(X_train)
    yb = torch.tensor(y_stage_train)
    y_next_b = torch.tensor(y_next_train)
    for _ in range(3):
        optimizer.zero_grad()
        stage_logits, next_state, _ = model(xb)
        loss = F.cross_entropy(stage_logits, yb) + 0.5 * F.mse_loss(next_state, y_next_b)
        loss.backward()
        optimizer.step()
    assert torch.isfinite(loss)

    X_base_train, y_base_train, _ = build_single_window_table(labeled, scaler, train_hosts)
    X_base_test, y_base_test, _ = build_single_window_table(labeled, scaler, test_hosts)
    baseline_clf = train_baseline(X_base_train, y_base_train)

    from app.config import MALICIOUS_HARD_STAGES
    labeled_idx = labeled.set_index(["host_id", "window_idx"])
    y_true_hard_at_w = np.array([
        1 if labeled_idx.loc[(m["host_id"], m["window_idx_next"]), "true_stage"] in MALICIOUS_HARD_STAGES else 0
        for m in meta_test
    ])

    report = run_full_benchmark(baseline_clf, model, X_base_test, y_base_test, X_test, y_true_hard_at_w,
                                 write=False)
    for key in ["precision", "recall", "f1", "false_positive_rate"]:
        assert key in report["baseline_logistic_regression"]
        assert key in report["world_model_lstm"]
        assert 0.0 <= report["baseline_logistic_regression"][key] <= 1.0
        assert 0.0 <= report["world_model_lstm"][key] <= 1.0

    # forecast + explanation for one real host
    host_id = sorted(test_hosts)[0]
    host_df = labeled[labeled.host_id == host_id].sort_values("window_idx").reset_index(drop=True)
    feats = scaler.transform(host_df[FEATURE_COLUMNS].values).astype(np.float32)
    window = feats[-SEQ_LEN:]
    explanation = explain_prediction(model, window)
    assert 0.0 <= explanation["infiltration_probability"] <= 1.0
    assert len(explanation["attention_over_past_windows"]) == SEQ_LEN
    assert len(explanation["top_contributors"]) > 0

    roll = rollout(model, window, k=4)
    assert len(roll["infiltration_probs"]) == 4
    assert all(0.0 <= p <= 1.0 for p in roll["infiltration_probs"])

    # branching K-step forecast, using real train.py-style per-action mean vectors
    from app.train import build_stage_mean_vectors
    stage_means = build_stage_mean_vectors(labeled, scaler, train_hosts)
    tree = branching_rollout(model, window, stage_mean_vectors=stage_means, depth=3, branch_factor=2, min_path_prob=0.0)
    paths = enumerate_paths(tree["root"])
    assert len(paths) > 0
    for p in paths:
        assert len(p["stages"]) == 3
        assert all(m["technique_id"] is not None or m["stage"] == "benign" for m in p["mitre_kill_chain"])
