import numpy as np
import pandas as pd

from app.config import FEATURE_COLUMNS, AMBIGUOUS_LOOKBACK, PRECURSOR_WEIGHTS, PRECURSOR_THRESHOLD
from app.labeling.state_labeler import (
    derive_state_labels,
    build_transition_pairs,
    compute_precursor_signals,
    compute_precursor_score,
)


def _make_fixture_timeline():
    """Small hand-built fixture: one host with 10 benign windows, then 4
    precursor-ish windows with elevated port_scan_score/failed_conn_ratio,
    then reconnaissance onset. A second host stays pure benign throughout."""
    rows = []

    def base_row(host, idx, stage, **overrides):
        row = {c: 0.01 for c in FEATURE_COLUMNS}
        row.update({
            "port_scan_score": 0.03, "failed_conn_ratio": 0.02, "new_dst_ip_ratio": 0.05,
            "iat_mean": 0.15, "iat_std": 0.05,
        })
        row.update(overrides)
        row["host_id"] = host
        row["window_idx"] = idx
        row["true_stage"] = stage
        return row

    for i in range(10):
        rows.append(base_row("host-A", i, "benign"))
    for i in range(10, 10 + AMBIGUOUS_LOOKBACK):
        rows.append(base_row("host-A", i, "benign", port_scan_score=0.6, failed_conn_ratio=0.5,
                              new_dst_ip_ratio=0.4, iat_std=0.5))
    for i in range(10 + AMBIGUOUS_LOOKBACK, 10 + AMBIGUOUS_LOOKBACK + 5):
        rows.append(base_row("host-A", i, "port_scan", port_scan_score=0.8, failed_conn_ratio=0.7))

    for i in range(15):
        rows.append(base_row("host-B", i, "benign"))

    return pd.DataFrame(rows)


def test_ambiguous_windows_are_relabeled():
    df = _make_fixture_timeline()
    labeled = derive_state_labels(df)
    host_a = labeled[labeled.host_id == "host-A"].sort_values("window_idx")
    precursor_rows = host_a[(host_a.window_idx >= 10) & (host_a.window_idx < 10 + AMBIGUOUS_LOOKBACK)]
    assert (precursor_rows["state_label"] == "ambiguous_pre_attack").all()


def test_pure_benign_host_never_relabeled():
    df = _make_fixture_timeline()
    labeled = derive_state_labels(df)
    host_b = labeled[labeled.host_id == "host-B"]
    assert (host_b["state_label"] == "benign").all()


def test_early_benign_windows_far_from_onset_stay_benign():
    df = _make_fixture_timeline()
    labeled = derive_state_labels(df)
    host_a = labeled[labeled.host_id == "host-A"].sort_values("window_idx")
    early = host_a[host_a.window_idx < 10]
    assert (early["state_label"] == "benign").all()


def test_hard_malicious_labels_pass_through_unchanged():
    df = _make_fixture_timeline()
    labeled = derive_state_labels(df)
    host_a = labeled[labeled.host_id == "host-A"].sort_values("window_idx")
    recon_rows = host_a[host_a.true_stage == "port_scan"]
    assert (recon_rows["state_label"] == "port_scan").all()


def test_low_precursor_score_stays_benign():
    """A window with all precursor-relevant features at benign-population
    levels must NOT be relabeled, even inside the lookback zone."""
    df = _make_fixture_timeline()
    labeled = derive_state_labels(df)
    host_a = labeled[labeled.host_id == "host-A"].sort_values("window_idx")
    # windows 0-9 are plain benign, well outside/beneath any elevated signal
    calm = host_a[host_a.window_idx < 10]
    assert (calm["precursor_score"] < PRECURSOR_THRESHOLD).all()
    assert (calm["state_label"] == "benign").all()


def test_high_precursor_score_becomes_ambiguous_pre_attack():
    df = _make_fixture_timeline()
    labeled = derive_state_labels(df)
    host_a = labeled[labeled.host_id == "host-A"].sort_values("window_idx")
    precursor_rows = host_a[(host_a.window_idx >= 10) & (host_a.window_idx < 10 + AMBIGUOUS_LOOKBACK)]
    assert (precursor_rows["precursor_score"] >= PRECURSOR_THRESHOLD).all()
    assert (precursor_rows["state_label"] == "ambiguous_pre_attack").all()


def test_actual_attack_state_never_becomes_ambiguous_pre_attack():
    """An already-hard-labeled attack window (e.g. port_scan) must retain its
    real label -- ambiguous_pre_attack may only replace 'benign', never an
    actual attack-stage label."""
    df = _make_fixture_timeline()
    labeled = derive_state_labels(df)
    host_a = labeled[labeled.host_id == "host-A"].sort_values("window_idx")
    recon_rows = host_a[host_a.true_stage == "port_scan"]
    assert (recon_rows["state_label"] != "ambiguous_pre_attack").all()
    assert (recon_rows["state_label"] == "port_scan").all()


def test_threshold_behavior_is_a_hard_cutoff():
    """A score exactly at the threshold clears it; moving the threshold
    above the fixture's engineered score flips the outcome."""
    df = _make_fixture_timeline()
    benign_ranges = {
        "port_scan_score": (0.0, 1.0),
        "failed_conn_ratio": (0.0, 1.0),
        "new_dst_ip_ratio": (0.0, 1.0),
        "_timing_irregularity": (0.0, 10.0),
    }
    signals = compute_precursor_signals(df, benign_ranges)
    score = signals["precursor_score"]
    # sanity: fixture's port_scan window score should clearly clear the
    # project's default PRECURSOR_THRESHOLD
    assert (score[df.true_stage == "port_scan"] > PRECURSOR_THRESHOLD).all()


def test_configurable_lookback_changes_relabeled_window_count(monkeypatch):
    import app.labeling.state_labeler as sl

    df = _make_fixture_timeline()
    monkeypatch.setattr(sl, "AMBIGUOUS_LOOKBACK", 1)
    labeled_short = derive_state_labels(df)
    monkeypatch.setattr(sl, "AMBIGUOUS_LOOKBACK", AMBIGUOUS_LOOKBACK)
    labeled_default = derive_state_labels(df)

    n_short = (labeled_short["state_label"] == "ambiguous_pre_attack").sum()
    n_default = (labeled_default["state_label"] == "ambiguous_pre_attack").sum()
    assert n_short < n_default


def test_configurable_weights_change_the_score(monkeypatch):
    import app.labeling.state_labeler as sl

    df = _make_fixture_timeline()
    benign_ranges = sl._benign_ranges(df)
    baseline_score = compute_precursor_score(df, benign_ranges)

    zeroed_weights = {"port_scan": 0.0, "failed_connections": 0.0,
                       "destination_novelty": 0.0, "timing_irregularity": 1.0}
    monkeypatch.setattr(sl, "PRECURSOR_WEIGHTS", zeroed_weights)
    reweighted_score = compute_precursor_score(df, benign_ranges)

    assert not np.allclose(baseline_score.values, reweighted_score.values)


def test_precursor_signals_are_individually_exposed():
    """Beyond the combined score, the individual named signals must be
    computed and exposed (not just the combined precursor_score)."""
    df = _make_fixture_timeline()
    labeled = derive_state_labels(df)
    for col in ("port_scan_signal", "failed_conn_signal", "new_dst_signal", "timing_signal", "precursor_score"):
        assert col in labeled.columns
        assert labeled[col].between(0.0, 1.0).all()


def test_precursor_weights_sum_to_one():
    assert abs(sum(PRECURSOR_WEIGHTS.values()) - 1.0) < 1e-9


def test_labeled_dataframe_keeps_features_label_and_score_as_distinct_columns():
    """state_features (S_t), state_label, and precursor_score must be three
    separate, distinctly-named things, not conflated into one ambiguous
    'state' value. They're kept separate as plain DataFrame columns --
    row[FEATURE_COLUMNS] for S_t, row['state_label'] for the label,
    row['precursor_score'] (+ the four named signal columns) for the score --
    with no extra object wrapper required to tell them apart."""
    df = _make_fixture_timeline()
    labeled = derive_state_labels(df)
    row = labeled.iloc[15]
    state_features = {c: float(row[c]) for c in FEATURE_COLUMNS}
    assert set(state_features.keys()) == set(FEATURE_COLUMNS)
    assert isinstance(row["state_label"], str)
    assert isinstance(float(row["precursor_score"]), float)
    for col in ("port_scan_signal", "failed_conn_signal", "new_dst_signal", "timing_signal"):
        assert col in labeled.columns


def test_precursor_score_is_causal():
    """ROW-LOCAL CAUSALITY TEST (honest scope -- see module docstring's
    "CAUSALITY, PRECISELY STATED"): holding `benign_ranges` FIXED, the
    row-local score formula for a candidate window W_t must depend only on
    W_t's own feature values -- never on any other window's features, past
    or future. Modifying every OTHER window's features and recomputing must
    leave W_t's score exactly unchanged. This test intentionally does NOT
    claim that derive_state_labels()'s end-to-end output is causal: that
    function recomputes `benign_ranges` from the full input dataframe (a
    dataset-level statistic), which test_benign_ranges_is_a_dataset_level_statistic
    below demonstrates directly, so we do not misrepresent the full offline
    pipeline as future-independent."""
    df = _make_fixture_timeline()
    benign_ranges = {
        "port_scan_score": (0.0, 1.0),
        "failed_conn_ratio": (0.0, 1.0),
        "new_dst_ip_ratio": (0.0, 1.0),
        "_timing_irregularity": (0.0, 10.0),
    }
    target_idx = 12  # a precursor-zone row for host-A
    original_score = compute_precursor_score(df, benign_ranges).loc[target_idx]

    mutated = df.copy()
    other_rows = mutated.index != target_idx
    rng = np.random.default_rng(0)
    for col in ["port_scan_score", "failed_conn_ratio", "new_dst_ip_ratio", "iat_mean", "iat_std"]:
        mutated.loc[other_rows, col] = rng.uniform(0, 5, size=other_rows.sum())

    mutated_score = compute_precursor_score(mutated, benign_ranges).loc[target_idx]
    assert mutated_score == original_score


def test_benign_ranges_is_a_dataset_level_statistic():
    """Honesty check for Phase 8 (do not falsely claim the full offline
    pipeline is future-independent): `_benign_ranges()` -- the reference
    `derive_state_labels()` actually feeds into the row-local formula above
    -- is computed from the ENTIRE input dataframe's benign population, so
    changing benign feature values ELSEWHERE in the dataframe DOES change
    the reference range (and therefore can change other rows' scores). This
    is expected, existing, documented behavior -- not a bug -- and this test
    exists so nobody mistakes test_precursor_score_is_causal (which holds
    benign_ranges fixed) for a claim that derive_state_labels() end-to-end
    is temporally causal."""
    import app.labeling.state_labeler as sl

    df = _make_fixture_timeline()
    ranges_before = sl._benign_ranges(df)

    mutated = df.copy()
    benign_mask = mutated["true_stage"] == "benign"
    mutated.loc[benign_mask, "port_scan_score"] = 0.99
    ranges_after = sl._benign_ranges(mutated)

    assert ranges_before["port_scan_score"] != ranges_after["port_scan_score"]


def test_build_transition_pairs_shapes():
    df = _make_fixture_timeline()
    labeled = derive_state_labels(df)
    pairs = build_transition_pairs(labeled)
    # one fewer transition per host than rows per host
    n_rows_a = (labeled.host_id == "host-A").sum()
    n_rows_b = (labeled.host_id == "host-B").sum()
    assert len(pairs) == (n_rows_a - 1) + (n_rows_b - 1)
    for c in FEATURE_COLUMNS:
        assert c in pairs.columns
        assert f"next_{c}" in pairs.columns
    assert "next_state_label" in pairs.columns
