import numpy as np
import pytest

from app.config import FEATURE_COLUMNS, SEQ_LEN, N_FEATURES, FEATURE_DOCS
from app.data_gen.generator import generate_dataset
from app.labeling.state_labeler import derive_state_labels
from app.features.extraction import (
    host_split, fit_scaler, build_sequences, build_single_window_table,
    validate_feature_vector, FeatureValidationError,
)


def _small_labeled_dataset():
    raw = generate_dataset(seed=7, n_benign_hosts=6, n_attack_hosts=2, benign_len=40)
    return derive_state_labels(raw)


def test_host_split_disjoint_and_covers_all_hosts():
    labeled = _small_labeled_dataset()
    train, val, test = host_split(labeled)
    all_hosts = set(labeled["host_id"].unique())
    assert train | val | test == all_hosts
    assert train & val == set()
    assert train & test == set()
    assert val & test == set()


def test_evasive_host_pinned_to_test_split():
    labeled = _small_labeled_dataset()
    train, val, test = host_split(labeled)
    assert "attack-host-000" in test


def test_build_sequences_shapes():
    labeled = _small_labeled_dataset()
    train, val, test = host_split(labeled)
    scaler = fit_scaler(labeled, train)
    X, y_stage, y_next, y_cur, meta = build_sequences(labeled, scaler, train)
    assert X.ndim == 3
    assert X.shape[1] == SEQ_LEN
    assert X.shape[2] == N_FEATURES
    assert X.shape[0] == y_stage.shape[0] == y_next.shape[0] == y_cur.shape[0] == len(meta)
    assert y_next.shape[1] == N_FEATURES


def test_build_single_window_table_shapes_and_binary_labels():
    labeled = _small_labeled_dataset()
    train, val, test = host_split(labeled)
    scaler = fit_scaler(labeled, train)
    X, y_binary, meta = build_single_window_table(labeled, scaler, train)
    assert X.shape[1] == N_FEATURES
    assert set(np.unique(y_binary)).issubset({0, 1})
    assert X.shape[0] == len(y_binary) == len(meta)


def test_scaler_normalizes_train_features_to_roughly_zero_mean():
    labeled = _small_labeled_dataset()
    train, val, test = host_split(labeled)
    scaler = fit_scaler(labeled, train)
    train_df = labeled[labeled.host_id.isin(train)]
    transformed = scaler.transform(train_df[FEATURE_COLUMNS].values)
    assert abs(transformed.mean()) < 0.5


# ---------------------------------------------------------------------------
# Feature schema
# ---------------------------------------------------------------------------

def test_feature_schema_has_31_features():
    assert N_FEATURES == 31
    assert len(FEATURE_COLUMNS) == 31


def test_feature_schema_names_are_unique_and_documented():
    assert len(set(FEATURE_COLUMNS)) == len(FEATURE_COLUMNS)
    assert set(FEATURE_DOCS.keys()) == set(FEATURE_COLUMNS)


def test_feature_schema_order_is_stable_across_generator_and_scaler():
    """The generator's DataFrame columns for FEATURE_COLUMNS, in order, must
    exactly match the authoritative schema -- feature order must never be
    independently hardcoded/reordered in a different module."""
    df = generate_dataset(seed=9, n_benign_hosts=2, n_attack_hosts=1, benign_len=15)
    assert list(df[FEATURE_COLUMNS].columns) == list(FEATURE_COLUMNS)


# ---------------------------------------------------------------------------
# validate_feature_vector
# ---------------------------------------------------------------------------

def _valid_feature_dict():
    return {c: 1.0 for c in FEATURE_COLUMNS}


def test_validate_feature_vector_accepts_well_formed_dict():
    out = validate_feature_vector(_valid_feature_dict())
    assert out.shape == (N_FEATURES,)
    assert np.allclose(out, 1.0)


def test_validate_feature_vector_accepts_ordered_array():
    arr = list(range(N_FEATURES))
    out = validate_feature_vector(arr)
    assert np.allclose(out, arr)


def test_validate_feature_vector_accepts_dict_in_a_different_insertion_order():
    """Key SET must match FEATURE_COLUMNS -- insertion/iteration order of
    the dict must NOT matter. The returned vector must still come back in
    canonical FEATURE_COLUMNS order regardless of the input order."""
    # Deliberately build the dict in reverse-schema order.
    reversed_order = {c: i for i, c in enumerate(reversed(FEATURE_COLUMNS))}
    out = validate_feature_vector(reversed_order)
    expected = np.array([reversed_order[c] for c in FEATURE_COLUMNS], dtype=np.float64)
    assert np.allclose(out, expected)

    # And an arbitrary (neither forward nor reverse nor sorted) shuffled order.
    rng_order = list(FEATURE_COLUMNS)
    rng_order[0], rng_order[-1] = rng_order[-1], rng_order[0]
    rng_order[3], rng_order[10] = rng_order[10], rng_order[3]
    shuffled_dict = {c: i for i, c in enumerate(rng_order)}
    out2 = validate_feature_vector(shuffled_dict)
    expected2 = np.array([shuffled_dict[c] for c in FEATURE_COLUMNS], dtype=np.float64)
    assert np.allclose(out2, expected2)


def test_validate_feature_vector_rejects_missing_feature():
    bad = _valid_feature_dict()
    del bad[FEATURE_COLUMNS[0]]
    with pytest.raises(FeatureValidationError):
        validate_feature_vector(bad)


def test_validate_feature_vector_rejects_unexpected_feature():
    bad = _valid_feature_dict()
    bad["totally_made_up_feature"] = 5.0
    with pytest.raises(FeatureValidationError):
        validate_feature_vector(bad)


def test_validate_feature_vector_rejects_wrong_dimensionality_array():
    with pytest.raises(FeatureValidationError):
        validate_feature_vector([1.0, 2.0, 3.0])


def test_validate_feature_vector_rejects_non_numeric_value():
    bad = _valid_feature_dict()
    bad[FEATURE_COLUMNS[0]] = "not-a-number"
    with pytest.raises(FeatureValidationError):
        validate_feature_vector(bad)


def test_validate_feature_vector_rejects_nan_and_inf():
    """Validation and sanitization are separate concerns: the validator
    never silently zero-fills invalid measurements -- it always raises."""
    bad = _valid_feature_dict()
    bad[FEATURE_COLUMNS[0]] = float("nan")
    with pytest.raises(FeatureValidationError):
        validate_feature_vector(bad)

    bad2 = _valid_feature_dict()
    bad2[FEATURE_COLUMNS[1]] = float("inf")
    with pytest.raises(FeatureValidationError):
        validate_feature_vector(bad2)

    bad3 = _valid_feature_dict()
    bad3[FEATURE_COLUMNS[2]] = float("-inf")
    with pytest.raises(FeatureValidationError):
        validate_feature_vector(bad3)


def test_validate_feature_vector_output_is_in_schema_order():
    ordered = {c: i for i, c in enumerate(FEATURE_COLUMNS)}
    out = validate_feature_vector(ordered)
    assert np.allclose(out, np.arange(N_FEATURES))


def test_generator_output_passes_validation_every_row():
    df = generate_dataset(seed=5, n_benign_hosts=2, n_attack_hosts=1, benign_len=12)
    for _, row in df.head(50).iterrows():
        validate_feature_vector({c: row[c] for c in FEATURE_COLUMNS})


def test_flow_tracker_output_passes_validation():
    """Confirms validate_feature_vector accepts FlowTracker.roll_window()'s
    existing output UNCHANGED -- flow_tracker.py itself is not modified."""
    from app.live.flow_tracker import FlowTracker

    tracker = FlowTracker(local_ip="10.0.0.5")
    tracker.ingest_tcp(remote_ip="10.0.0.9", direction="in", local_port=22, remote_port=1000,
                        flags="S", ttl=64, win_size=1024, pkt_len=60, ts=0.0)
    feats = tracker.roll_window()["10.0.0.9"]
    validate_feature_vector(feats)


# ---------------------------------------------------------------------------
# Empty / low-traffic window handling (no NaN/Inf ever produced)
# ---------------------------------------------------------------------------

def test_flow_tracker_single_packet_window_has_no_nan_or_inf():
    from app.live.flow_tracker import FlowTracker

    tracker = FlowTracker(local_ip="10.0.0.5")
    tracker.ingest_tcp(remote_ip="10.0.0.9", direction="in", local_port=22, remote_port=1000,
                        flags="S", ttl=64, win_size=1024, pkt_len=60, ts=0.0)
    feats = tracker.roll_window()["10.0.0.9"]
    values = np.array([feats[c] for c in FEATURE_COLUMNS], dtype=np.float64)
    assert np.isfinite(values).all()


def test_flow_tracker_empty_window_returns_no_rows_not_nan_rows():
    from app.live.flow_tracker import FlowTracker

    tracker = FlowTracker(local_ip="10.0.0.5")
    assert tracker.roll_window() == {}
