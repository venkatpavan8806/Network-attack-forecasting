"""Feature matrix construction, normalization, and host-level train/val/test
splitting. Splits are done by HOST (never by row) since rows within a host
are a time series -- splitting by row would leak adjacent-window information
across train/test.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

from app.config import FEATURE_COLUMNS, SEQ_LEN, STAGE_TO_IDX, RANDOM_SEED, SCALER_WEIGHTS, N_FEATURES


class FeatureValidationError(ValueError):
    """Raised by validate_feature_vector when a feature vector does not
    conform to the authoritative FEATURE_COLUMNS schema."""


def validate_feature_vector(features):
    """Validates a single window's feature vector against the authoritative
    `FEATURE_COLUMNS` schema (the single source of truth used by synthetic
    generation, feature extraction, state construction, normalization, model
    input, and live inference).

    This is a VALIDATION function only -- it never sanitizes or repairs
    data. Any safe-default/fallback handling for missing or unmeasurable
    values belongs at feature-extraction time (see e.g.
    app/live/flow_tracker.py, which already guards every mean/std
    computation so it never emits NaN/Inf in the first place), not here.

    Accepts either:
      - a dict/Mapping of {feature_name: value}, in ANY key order, or
      - a sequence/1-D array-like of length N_FEATURES, whose order is
        interpreted as FEATURE_COLUMNS order (that's the caller's
        responsibility -- there's no key to check it against for an
        unlabeled sequence).

    Checks performed:
      * for dict input: the key SET matches FEATURE_COLUMNS exactly (no
        missing keys, no unexpected extra keys). Insertion/iteration order
        of the dict does NOT matter -- a valid dict is accepted regardless
        of what order its keys happen to be in.
      * for array-like input: length is exactly N_FEATURES.
      * every value is numeric (bool is rejected).
      * every value is finite: NaN and +/-Inf are REJECTED with a
        FeatureValidationError, never silently replaced.

    Returns:
      np.ndarray of shape (N_FEATURES,), dtype float64, explicitly
      constructed in canonical FEATURE_COLUMNS order regardless of the
      input dict's own key order -- so callers can use this as the single
      choke point before scaling/model input.

    Raises:
      FeatureValidationError on any schema violation: missing/extra keys,
      wrong dimensionality, non-numeric values, or any NaN/Inf value.
    """
    if isinstance(features, dict):
        provided_set = set(features.keys())
        expected_set = set(FEATURE_COLUMNS)

        missing = expected_set - provided_set
        if missing:
            raise FeatureValidationError(f"feature vector is missing required features: {sorted(missing)}")

        extra = provided_set - expected_set
        if extra:
            raise FeatureValidationError(f"feature vector has unexpected features: {sorted(extra)}")

        # Key SET is validated above; insertion order is irrelevant. The
        # output vector is always explicitly re-projected into canonical
        # FEATURE_COLUMNS order here, regardless of what order the caller's
        # dict keys were in.
        values = [features[c] for c in FEATURE_COLUMNS]
    else:
        arr = np.asarray(features, dtype=object).reshape(-1)
        if arr.shape[0] != N_FEATURES:
            raise FeatureValidationError(
                f"feature vector has {arr.shape[0]} dimensions, expected {N_FEATURES} (FEATURE_COLUMNS)"
            )
        values = list(arr)

    out = np.empty(N_FEATURES, dtype=np.float64)
    for i, (name, v) in enumerate(zip(FEATURE_COLUMNS, values)):
        if isinstance(v, bool) or not isinstance(v, (int, float, np.integer, np.floating)):
            raise FeatureValidationError(f"feature '{name}' has non-numeric value: {v!r} ({type(v).__name__})")
        fv = float(v)
        if math.isnan(fv) or math.isinf(fv):
            raise FeatureValidationError(f"feature '{name}' is NaN/Inf: {fv}")
        out[i] = fv

    return out


def host_split(labeled_df: pd.DataFrame, seed: int = RANDOM_SEED):
    """Splits hosts into train/val/test. The evasive attack host is pinned
    to the test split deliberately, so the reported benchmark demonstrates
    generalization to the slow/evasive recon pattern rather than only to
    fast/obvious attacks seen in training."""
    hosts = labeled_df["host_id"].unique().tolist()
    benign_hosts = sorted(h for h in hosts if h.startswith("benign-host"))
    attack_hosts = sorted(h for h in hosts if h.startswith("attack-host"))

    rng = np.random.default_rng(seed)
    rng.shuffle(benign_hosts)

    n_benign = len(benign_hosts)
    n_val_b = max(1, int(n_benign * 0.15))
    n_test_b = max(1, int(n_benign * 0.15))
    val_benign = benign_hosts[:n_val_b]
    test_benign = benign_hosts[n_val_b:n_val_b + n_test_b]
    train_benign = benign_hosts[n_val_b + n_test_b:]

    # attack-host-000 is the evasive host (see generator.py) -- force it to test.
    evasive_host = "attack-host-000"
    remaining_attack = [h for h in attack_hosts if h != evasive_host]
    rng.shuffle(remaining_attack)
    # keep at least one attack host in train (if any remain) so the baseline
    # always sees both classes during training, even on tiny datasets.
    n_val_a = max(1, int(len(remaining_attack) * 0.2)) if len(remaining_attack) > 1 else 0
    val_attack = remaining_attack[:n_val_a]
    train_attack = remaining_attack[n_val_a:]
    test_attack = [evasive_host]

    train_hosts = set(train_benign + train_attack)
    val_hosts = set(val_benign + val_attack)
    test_hosts = set(test_benign + test_attack)
    return train_hosts, val_hosts, test_hosts


def fit_scaler(labeled_df: pd.DataFrame, train_hosts: set) -> StandardScaler:
    train_df = labeled_df[labeled_df["host_id"].isin(train_hosts)]
    scaler = StandardScaler()
    scaler.fit(train_df[FEATURE_COLUMNS].values)
    return scaler


def save_scaler(scaler: StandardScaler, path=SCALER_WEIGHTS):
    joblib.dump(scaler, path)


def load_scaler(path=SCALER_WEIGHTS) -> StandardScaler:
    return joblib.load(path)


def build_sequences(labeled_df: pd.DataFrame, scaler: StandardScaler, hosts: set, seq_len: int = SEQ_LEN):
    """For each host restricted to `hosts`, builds sliding-window sequences:
      X: (n_samples, seq_len, n_features)          -- normalized S_{t-seq_len+1..t}
      y_stage: (n_samples,)                         -- class idx of state_label at t+1
      y_next_state: (n_samples, n_features)         -- normalized S_{t+1} (regression target)
      y_cur_stage: (n_samples,)                     -- class idx of state_label at t (for reference/eval)
      meta: list of dicts (host_id, window_idx of t, window_idx of t+1)
    """
    X_list, y_stage_list, y_next_list, y_cur_list, meta = [], [], [], [], []
    for host_id, host_df in labeled_df[labeled_df["host_id"].isin(hosts)].groupby("host_id", sort=False):
        host_df = host_df.sort_values("window_idx").reset_index(drop=True)
        feats = scaler.transform(host_df[FEATURE_COLUMNS].values)
        labels = host_df["state_label"].map(STAGE_TO_IDX).values
        n = len(host_df)
        for t in range(seq_len - 1, n - 1):
            X_list.append(feats[t - seq_len + 1:t + 1])
            y_stage_list.append(labels[t + 1])
            y_next_list.append(feats[t + 1])
            y_cur_list.append(labels[t])
            meta.append({
                "host_id": host_id,
                "window_idx_t": int(host_df.loc[t, "window_idx"]),
                "window_idx_next": int(host_df.loc[t + 1, "window_idx"]),
            })
    return (
        np.array(X_list, dtype=np.float32),
        np.array(y_stage_list, dtype=np.int64),
        np.array(y_next_list, dtype=np.float32),
        np.array(y_cur_list, dtype=np.int64),
        meta,
    )


def build_single_window_table(labeled_df: pd.DataFrame, scaler: StandardScaler, hosts: set):
    """For the baseline: single-window features -> current-window binary
    malicious label (no history, no future). Returns X (n, n_features),
    y_binary (n,), state_label (n,) strings, meta list."""
    from app.config import MALICIOUS_HARD_STAGES

    sub = labeled_df[labeled_df["host_id"].isin(hosts)].sort_values(["host_id", "window_idx"]).reset_index(drop=True)
    X = scaler.transform(sub[FEATURE_COLUMNS].values).astype(np.float32)
    y_binary = sub["true_stage"].isin(MALICIOUS_HARD_STAGES).astype(np.int64).values
    meta = sub[["host_id", "window_idx", "true_stage", "state_label"]].to_dict("records")
    return X, y_binary, meta
