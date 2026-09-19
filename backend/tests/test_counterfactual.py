import numpy as np
import torch

from app.config import N_FEATURES, SEQ_LEN, STAGE_CLASSES, FEATURE_COLUMNS
from app.models.lstm_world_model import LSTMWorldModel
from app.simulation.mitigations import MITIGATIONS, list_mitigations, get_mitigation_fn
from app.simulation.counterfactual import rollout_counterfactual, compare_with_and_without


def _dummy_model():
    torch.manual_seed(0)
    return LSTMWorldModel(n_features=N_FEATURES, hidden_size=8, num_layers=1, n_classes=len(STAGE_CLASSES))


class _IdentityScaler:
    """Stand-in for a fitted StandardScaler with mean=0, std=1 -- keeps the
    raw<->scaled round trip in counterfactual.py exact for the test."""
    def transform(self, X):
        return np.asarray(X, dtype=np.float32)

    def inverse_transform(self, X):
        return np.asarray(X, dtype=np.float32)


def test_rate_limit_only_affects_matching_port_window():
    fn = get_mitigation_fn("rate_limit_ssh")
    idx_port22 = FEATURE_COLUMNS.index("dst_port_is_22")
    idx_flow = FEATURE_COLUMNS.index("flow_count")

    row_with_ssh = np.zeros(N_FEATURES, dtype=np.float32)
    row_with_ssh[idx_port22] = 1.0
    row_with_ssh[idx_flow] = 100.0
    mutated = fn(row_with_ssh)
    assert mutated[idx_flow] == 25.0  # 100 * 0.25

    row_without_ssh = np.zeros(N_FEATURES, dtype=np.float32)
    row_without_ssh[idx_flow] = 100.0
    unchanged = fn(row_without_ssh)
    assert unchanged[idx_flow] == 100.0


def test_block_is_more_aggressive_than_rate_limit():
    idx_port22 = FEATURE_COLUMNS.index("dst_port_is_22")
    idx_flow = FEATURE_COLUMNS.index("flow_count")
    row = np.zeros(N_FEATURES, dtype=np.float32)
    row[idx_port22] = 1.0
    row[idx_flow] = 100.0

    rate_limited = get_mitigation_fn("rate_limit_ssh")(row)
    blocked = get_mitigation_fn("block_ssh")(row)
    assert blocked[idx_flow] < rate_limited[idx_flow]


def test_isolate_host_scales_all_count_features_regardless_of_port():
    idx_flow = FEATURE_COLUMNS.index("flow_count")
    idx_out = FEATURE_COLUMNS.index("outbound_bytes")
    row = np.zeros(N_FEATURES, dtype=np.float32)
    row[idx_flow] = 100.0
    row[idx_out] = 5000.0
    mutated = get_mitigation_fn("isolate_host")(row)
    assert mutated[idx_flow] == 5.0
    assert mutated[idx_out] == 250.0


def test_no_mitigation_is_a_true_identity():
    row = np.random.default_rng(0).normal(size=N_FEATURES).astype(np.float32)
    mutated = get_mitigation_fn("no_mitigation")(row)
    np.testing.assert_array_equal(row, mutated)


def test_list_mitigations_matches_registry():
    listed = list_mitigations()
    assert {m["id"] for m in listed} == set(MITIGATIONS.keys())
    for m in listed:
        assert "label" in m and "description" in m


def test_rollout_counterfactual_output_shape_and_range():
    model = _dummy_model()
    scaler = _IdentityScaler()
    seed_raw = np.random.default_rng(1).normal(size=(SEQ_LEN, N_FEATURES)).astype(np.float32)
    fn = get_mitigation_fn("isolate_host")
    result = rollout_counterfactual(model, scaler, seed_raw, fn, k=4)
    assert len(result["infiltration_probs"]) == 4
    assert len(result["predicted_stage"]) == 4
    for p in result["infiltration_probs"]:
        assert 0.0 <= p <= 1.0


def test_compare_with_and_without_uses_same_seed():
    model = _dummy_model()
    scaler = _IdentityScaler()
    seed_raw = np.random.default_rng(2).normal(size=(SEQ_LEN, N_FEATURES)).astype(np.float32)
    fn = get_mitigation_fn("no_mitigation")
    baseline, mitigated = compare_with_and_without(model, scaler, seed_raw, fn, k=3)
    # no_mitigation is a true identity, so both rollouts must match exactly
    np.testing.assert_allclose(baseline["infiltration_probs"], mitigated["infiltration_probs"], atol=1e-6)
