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

Precursor score (per window, per host, computed from that window's own
features plus a supplied benign-population reference range -- see the
CAUSALITY note below for exactly what this does and doesn't guarantee):
    score = PRECURSOR_WEIGHTS["port_scan"]            * norm(port_scan_score)
          + PRECURSOR_WEIGHTS["failed_connections"]    * norm(failed_conn_ratio)
          + PRECURSOR_WEIGHTS["destination_novelty"]   * norm(new_dst_ip_ratio)
          + PRECURSOR_WEIGHTS["timing_irregularity"]   * norm(iat_std / (iat_mean + eps))
(weights and threshold live in app/config.py -- see PRECURSOR_WEIGHTS,
PRECURSOR_LOOKBACK, PRECURSOR_THRESHOLD). Each term is min-max normalized
against the benign population's observed range so the score is comparable
across features of very different scale. A window in the PRECURSOR_LOOKBACK
zone before a hard attack onset is relabeled "ambiguous_pre_attack" if
score >= PRECURSOR_THRESHOLD.

CAUSALITY, PRECISELY STATED (see also the "IMPORTANT CAUSALITY REQUIREMENT"
section of the project brief) -- this has two genuinely different layers,
and they must not be blurred together:

1. THE ROW-LOCAL FORMULA is causal: `compute_precursor_signals(df,
   benign_ranges)` / `compute_precursor_score(df, benign_ranges)` compute
   window W_t's signals and score using ONLY W_t's own row of `df` (its own
   feature values) plus whatever `benign_ranges` dict is handed in as a
   parameter. Given a FIXED `benign_ranges`, changing any OTHER window's
   feature values -- past or future, same host or a different host -- never
   changes W_t's score. This is asserted directly in
   test_labeling.py::test_precursor_score_is_causal, which holds
   `benign_ranges` fixed and mutates every row except the target row.

2. HOWEVER, as actually invoked inside `derive_state_labels()` below,
   `benign_ranges` is NOT a fixed constant -- it comes from `_benign_ranges(df)`,
   which computes the 5th/95th percentile of each raw feature over every
   benign-labeled window in the ENTIRE input dataframe: all hosts, the full
   timeline, including windows that occur after W_t in wall-clock time. This
   is a genuine dataset-level (population) statistic, analogous to fitting a
   StandardScaler on a full offline training set -- it is NOT a per-window,
   walk-forward-only computation. We do NOT claim that the full,
   `derive_state_labels()`-produced score for W_t is independent of every
   other row in the dataset; it depends on the benign population's
   percentile range, which is computed with knowledge of the whole offline
   dataset. This dependency is preserved as-is (this is the existing
   architecture's behavior, not something introduced or newly discovered
   here) and documented here rather than silently redesigned.

3. Separately again: the offline LABELING procedure (this module, deciding
   WHICH windows are lookback candidates in the first place) also uses
   knowledge of where a hard attack onset occurs in the full, already-
   simulated timeline to select the PRECURSOR_LOOKBACK window(s) that
   precede it. That selection step is offline/non-causal by necessity -- a
   real detector does not know in advance which windows will turn out to
   precede an attack.

Net summary: the row-local signal/score FORMULA (item 1) is causal and
tested as such. The dataset-level normalization REFERENCE it's fed (item 2)
and the offline candidate-window SELECTION (item 3) are both, honestly,
non-causal dependencies of the full offline pipeline. Do not claim the
entire offline labeling procedure is temporally causal end-to-end --
only that the row-local formula itself does not read any other row's raw
feature values.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.config import (
    AMBIGUOUS_LOOKBACK,
    AMBIGUOUS_SCORE_THRESHOLD,
    FEATURE_COLUMNS,
    MALICIOUS_HARD_STAGES,
    PRECURSOR_LOOKBACK,
    PRECURSOR_THRESHOLD,
    PRECURSOR_WEIGHTS,
)

# Mapping from each named precursor SIGNAL (see PRECURSOR_WEIGHTS in
# config.py, the single authoritative source of these weights) to the raw
# FEATURE_COLUMNS entry it is derived from. "timing_irregularity" is not a
# raw feature itself -- it's iat_std/iat_mean, computed below.
_SIGNAL_TO_RAW_FEATURE = {
    "port_scan": "port_scan_score",
    "failed_connections": "failed_conn_ratio",
    "destination_novelty": "new_dst_ip_ratio",
    "timing_irregularity": None,  # derived, see compute_precursor_signals
}

# Output column names for the individual (non-fabricated, feature-derived)
# precursor signals -- exposed alongside the combined precursor_score so a
# later explanation/dashboard layer can answer "why was this window flagged
# pre-attack" without inventing anything not actually computed here.
SIGNAL_COLUMNS = {
    "port_scan": "port_scan_signal",
    "failed_connections": "failed_conn_signal",
    "destination_novelty": "new_dst_signal",
    "timing_irregularity": "timing_signal",
}


def _minmax_norm(series: pd.Series, lo: float, hi: float) -> pd.Series:
    if hi - lo < 1e-9:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return ((series - lo) / (hi - lo)).clip(0.0, 1.0)


def compute_precursor_signals(df: pd.DataFrame, benign_ranges: dict) -> pd.DataFrame:
    """Causal, per-row computation of the INDIVIDUAL precursor signals (each
    normalized to [0, 1] against the benign reference range), returned
    alongside the combined weighted precursor_score.

    `benign_ranges` gives the (5th, 95th) percentile range of each raw
    feature, supplied by the caller. This function's own computation is
    row-local and causal: it reads only df's own row for each output row,
    plus the fixed `benign_ranges` values -- it never reads any other row of
    df. NOTE: when called from `derive_state_labels()` below, the
    `benign_ranges` argument it receives is itself a dataset-level (all
    hosts, full timeline) benign-population statistic, not a walk-forward
    per-window quantity -- see the module docstring's "CAUSALITY, PRECISELY
    STATED" section for exactly what is and isn't causal about the full
    pipeline.

    Every signal here corresponds to an actual feature-derived calculation;
    none are fabricated. Returns a DataFrame (same index as df) with columns:
      port_scan_signal, failed_conn_signal, new_dst_signal, timing_signal,
      precursor_score
    """
    timing_irregularity = df["iat_std"] / (df["iat_mean"] + 1e-6)

    lo, hi = benign_ranges["port_scan_score"]
    port_scan_signal = _minmax_norm(df["port_scan_score"], lo, hi)
    lo, hi = benign_ranges["failed_conn_ratio"]
    failed_conn_signal = _minmax_norm(df["failed_conn_ratio"], lo, hi)
    lo, hi = benign_ranges["new_dst_ip_ratio"]
    new_dst_signal = _minmax_norm(df["new_dst_ip_ratio"], lo, hi)
    lo, hi = benign_ranges["_timing_irregularity"]
    timing_signal = _minmax_norm(timing_irregularity, lo, hi)

    score = (
        PRECURSOR_WEIGHTS["port_scan"] * port_scan_signal
        + PRECURSOR_WEIGHTS["failed_connections"] * failed_conn_signal
        + PRECURSOR_WEIGHTS["destination_novelty"] * new_dst_signal
        + PRECURSOR_WEIGHTS["timing_irregularity"] * timing_signal
    )
    # each component is clamped to [0, 1] and the weights sum to 1.0
    # (asserted in config.py), so the combination is already in [0, 1];
    # clip defensively in case of future weight edits.
    score = score.clip(0.0, 1.0)

    return pd.DataFrame(
        {
            "port_scan_signal": port_scan_signal,
            "failed_conn_signal": failed_conn_signal,
            "new_dst_signal": new_dst_signal,
            "timing_signal": timing_signal,
            "precursor_score": score,
        },
        index=df.index,
    )


def compute_precursor_score(df: pd.DataFrame, benign_ranges: dict) -> pd.Series:
    """Causal, per-row precursor score in [0, 1]. Thin wrapper around
    `compute_precursor_signals` kept for backward compatibility with call
    sites that only need the combined score."""
    return compute_precursor_signals(df, benign_ranges)["precursor_score"]


# NOTE: an earlier revision of this module also added a `NetworkState`
# dataclass plus `to_network_state()` / `explain_state_label()` helper
# functions, meant purely for ad-hoc debugging. On review, they were
# removed: they added an extra object layer that duplicated information
# already directly available and already explicit on the labeled
# DataFrame produced by `derive_state_labels()` below -- `state_features`
# is just `row[FEATURE_COLUMNS]`, `state_label` is just `row["state_label"]`,
# and the precursor signals are just `row[["port_scan_signal",
# "failed_conn_signal", "new_dst_signal", "timing_signal", "precursor_score"]]`.
# Anyone debugging "why was this window flagged" can already do
# `row[[...]].to_dict()` on that same DataFrame; a parallel dataclass
# added indirection without adding a capability. The three concepts
# (state / state_label / precursor_score) stay explicitly separate as
# distinct, clearly-named DataFrame columns instead.

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
    signals = compute_precursor_signals(df, benign_ranges)
    # Individual precursor signals are exposed alongside the combined score
    # (not just precursor_score) so the labeling decision stays auditable --
    # see explain_state_label() / README "Pre-detection labeling". These are
    # diagnostic/annotation columns, never fed to the model as input features
    # (model input remains strictly FEATURE_COLUMNS -- see
    # app/features/extraction.py).
    for col in signals.columns:
        df[col] = signals[col]

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
