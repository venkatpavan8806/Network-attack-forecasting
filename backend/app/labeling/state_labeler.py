"""State-labeling engine.

Defines what a "state" S_t is (a normalized traffic-shaped feature vector at
one time window for one host) and derives (S_t, S_t+1) transition pairs from
a raw labeled timeline.

The real differentiator implemented here: rather than inheriting the
simulator's own hard "benign" label for the windows immediately preceding an
attack's official onset, this engine re-derives a label for that lookback
zone from a precursor score computed ONLY from measurable features available
at that window (never from future knowledge of what the simulator scripted
next). Windows whose precursor score clears a threshold are relabeled
"ambiguous_pre_attack" instead of "benign". This is a heuristic engineering
judgment call -- there is no ground truth to validate it against, since real
networks don't come with "this window was 35% an attack precursor" labels
either. The precursor score formula and threshold are documented below and
in the README; they are deliberately simple and inspectable rather than
another learned black box, so the judgment call stays auditable.

Precursor score (per window, per host, causal -- uses only that window's own
features, no lookahead):
    score = 0.35 * norm(port_scan_score)
          + 0.25 * norm(failed_conn_ratio)
          + 0.20 * norm(new_dst_ip_ratio)
          + 0.20 * norm(iat_std / (iat_mean + eps))   # timing irregularity
Each term is min-max normalized against the benign population's observed
range so the score is comparable across features of very different scale.
A window in the AMBIGUOUS_LOOKBACK zone before a hard attack onset is
relabeled "ambiguous_pre_attack" if score >= AMBIGUOUS_SCORE_THRESHOLD.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.config import (
    AMBIGUOUS_LOOKBACK,
    AMBIGUOUS_SCORE_THRESHOLD,
    FEATURE_COLUMNS,
    MALICIOUS_HARD_STAGES,
)

_PRECURSOR_FEATURES_WEIGHTS = {
    "port_scan_score": 0.35,
    "failed_conn_ratio": 0.25,
    "new_dst_ip_ratio": 0.20,
    "_timing_irregularity": 0.20,
}


def _minmax_norm(series: pd.Series, lo: float, hi: float) -> pd.Series:
    if hi - lo < 1e-9:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return ((series - lo) / (hi - lo)).clip(0.0, 1.0)


def compute_precursor_score(df: pd.DataFrame, benign_ranges: dict) -> pd.Series:
    """Causal, per-row precursor score in [0, 1]. `benign_ranges` gives the
    (5th, 95th) percentile range of each raw feature over the BENIGN
    population only, used purely as a normalization reference -- it does not
    look at the current row's own future or its true label."""
    timing_irregularity = df["iat_std"] / (df["iat_mean"] + 1e-6)

    score = pd.Series(np.zeros(len(df)), index=df.index)
    lo, hi = benign_ranges["port_scan_score"]
    score += _PRECURSOR_FEATURES_WEIGHTS["port_scan_score"] * _minmax_norm(df["port_scan_score"], lo, hi)
    lo, hi = benign_ranges["failed_conn_ratio"]
    score += _PRECURSOR_FEATURES_WEIGHTS["failed_conn_ratio"] * _minmax_norm(df["failed_conn_ratio"], lo, hi)
    lo, hi = benign_ranges["new_dst_ip_ratio"]
    score += _PRECURSOR_FEATURES_WEIGHTS["new_dst_ip_ratio"] * _minmax_norm(df["new_dst_ip_ratio"], lo, hi)
    lo, hi = benign_ranges["_timing_irregularity"]
    score += _PRECURSOR_FEATURES_WEIGHTS["_timing_irregularity"] * _minmax_norm(timing_irregularity, lo, hi)
    return score


def _benign_ranges(df: pd.DataFrame) -> dict:
    benign = df[df["true_stage"] == "benign"].copy()
    benign["_timing_irregularity"] = benign["iat_std"] / (benign["iat_mean"] + 1e-6)
    ranges = {}
    for col in ["port_scan_score", "failed_conn_ratio", "new_dst_ip_ratio", "_timing_irregularity"]:
        ranges[col] = (float(benign[col].quantile(0.05)), float(benign[col].quantile(0.95)))
    return ranges


def derive_state_labels(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Takes the raw simulator output (host_id, window_idx, true_stage,
    FEATURE_COLUMNS...) and returns a copy with an added `state_label`
    column: the hard stage for malicious/benign windows, or
    "ambiguous_pre_attack" for benign windows within the lookback zone of an
    attack onset whose precursor score clears the threshold.
    """
    df = raw_df.sort_values(["host_id", "window_idx"]).reset_index(drop=True)
    benign_ranges = _benign_ranges(df)
    df["precursor_score"] = compute_precursor_score(df, benign_ranges)

    state_labels = df["true_stage"].copy()

    for host_id, host_df in df.groupby("host_id", sort=False):
        idx = host_df.index.to_numpy()
        stages = host_df["true_stage"].to_numpy()
        is_malicious = np.isin(stages, list(MALICIOUS_HARD_STAGES))
        # find onset points: index where malicious starts after a benign run
        onsets = []
        for i in range(1, len(stages)):
            if is_malicious[i] and not is_malicious[i - 1]:
                onsets.append(i)
        for onset in onsets:
            lookback_start = max(0, onset - AMBIGUOUS_LOOKBACK)
            for i in range(lookback_start, onset):
                row_idx = idx[i]
                if state_labels.loc[row_idx] == "benign" and df.loc[row_idx, "precursor_score"] >= AMBIGUOUS_SCORE_THRESHOLD:
                    state_labels.loc[row_idx] = "ambiguous_pre_attack"

    df["state_label"] = state_labels
    return df


def build_transition_pairs(labeled_df: pd.DataFrame) -> pd.DataFrame:
    """Builds (S_t -> state_label_{t+1}) pairs per host, preserving order.
    Returns a DataFrame with one row per transition: host_id, window_idx (t),
    FEATURE_COLUMNS (S_t), next_state_label, and the next window's raw
    feature vector (next_<feature>) for world-model regression targets.
    """
    rows = []
    for host_id, host_df in labeled_df.groupby("host_id", sort=False):
        host_df = host_df.sort_values("window_idx").reset_index(drop=True)
        for t in range(len(host_df) - 1):
            cur = host_df.iloc[t]
            nxt = host_df.iloc[t + 1]
            row = {"host_id": host_id, "window_idx": int(cur["window_idx"])}
            for c in FEATURE_COLUMNS:
                row[c] = cur[c]
                row[f"next_{c}"] = nxt[c]
            row["state_label"] = cur["state_label"]
            row["next_state_label"] = nxt["state_label"]
            rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from app.config import SYNTHETIC_CSV, DATA_DIR

    raw = pd.read_csv(SYNTHETIC_CSV)
    labeled = derive_state_labels(raw)
    print(labeled["state_label"].value_counts())
    n_relabeled = ((labeled["true_stage"] == "benign") & (labeled["state_label"] == "ambiguous_pre_attack")).sum()
    print(f"windows relabeled benign -> ambiguous_pre_attack: {n_relabeled}")
    labeled.to_csv(DATA_DIR / "labeled_states.csv", index=False)

    pairs = build_transition_pairs(labeled)
    pairs.to_csv(DATA_DIR / "transition_pairs.csv", index=False)
    print(f"transition pairs: {len(pairs)}")
    print(pairs["next_state_label"].value_counts())
