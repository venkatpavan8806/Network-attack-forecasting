import numpy as np

from app.config import FEATURE_COLUMNS, SEQ_LEN, N_FEATURES
from app.data_gen.generator import generate_dataset
from app.labeling.state_labeler import derive_state_labels
from app.features.extraction import host_split, fit_scaler, build_sequences, build_single_window_table


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
