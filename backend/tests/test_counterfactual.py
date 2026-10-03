import numpy as np
import torch

from app.config import N_FEATURES, SEQ_LEN, STAGE_CLASSES, FEATURE_COLUMNS
from app.models.lstm_world_model import LSTMWorldModel
from app.simulation.mitigations import MITIGATIONS, list_mitigations, get_mitigation_fn
from app.simulation.counterfactual import rollout_counterfactual, compare_with_and_without, _clamp_physical_bounds


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


def test_rollout_counterfactual_actually_calls_the_clamp_every_step(monkeypatch):
    """_clamp_physical_bounds existed but was never called from the
    autoregressive loop -- a real bug found during review. Guards against
    that regressing silently again by spying on the module-level name
    rollout_counterfactual actually calls."""
    import app.simulation.counterfactual as cf_module
    real_clamp = cf_module._clamp_physical_bounds
    calls = []

    def spy(row):
        calls.append(row)
        return real_clamp(row)

    monkeypatch.setattr(cf_module, "_clamp_physical_bounds", spy)

    model = _dummy_model()
    scaler = _IdentityScaler()
    seed_raw = np.random.default_rng(3).normal(size=(SEQ_LEN, N_FEATURES)).astype(np.float32)
    fn = get_mitigation_fn("no_mitigation")
    k = 4
    rollout_counterfactual(model, scaler, seed_raw, fn, k=k)
    assert len(calls) == k  # once per autoregressive step, not zero


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


def test_block_ssh_zeroes_port_flag_and_suppresses_volume():
    idx_port22 = FEATURE_COLUMNS.index("dst_port_is_22")
    idx_flow = FEATURE_COLUMNS.index("flow_count")
    row = np.zeros(N_FEATURES, dtype=np.float32)
    row[idx_port22] = 1.0
    row[idx_flow] = 100.0

    blocked = get_mitigation_fn("block_ssh")(row)
    assert blocked[idx_port22] == 0.0
    assert blocked[idx_flow] < 100.0


def test_isolate_host_clears_all_attack_indicators():
    idx_flow = FEATURE_COLUMNS.index("flow_count")
    idx_scan = FEATURE_COLUMNS.index("port_scan_score")
    row = np.ones(N_FEATURES, dtype=np.float32) * 100.0
    row[idx_scan] = 0.9

    isolated = get_mitigation_fn("isolate_host")(row)
    for port in [22, 445, 3389, 443]:
        idx_p = FEATURE_COLUMNS.index(f"dst_port_is_{port}")
        assert isolated[idx_p] == 0.0
    assert isolated[idx_scan] == 0.0
    assert isolated[idx_flow] < 10.0


def test_block_scanner_ip_resets_scan_score():
    idx_scan = FEATURE_COLUMNS.index("port_scan_score")
    idx_ports = FEATURE_COLUMNS.index("unique_dst_ports")
    row = np.zeros(N_FEATURES, dtype=np.float32)
    row[idx_scan] = 0.85
    row[idx_ports] = 50.0

    blocked = get_mitigation_fn("block_scanner_ip")(row)
    assert blocked[idx_scan] == 0.0
    assert blocked[idx_ports] <= 4.0


def test_block_c2_egress_clears_port_443_and_normalizes_iat():
    idx_443 = FEATURE_COLUMNS.index("dst_port_is_443")
    idx_iat_m = FEATURE_COLUMNS.index("iat_mean")
    row = np.zeros(N_FEATURES, dtype=np.float32)
    row[idx_443] = 1.0
    row[idx_iat_m] = 45.0  # typical C2 beacon interval

    blocked = get_mitigation_fn("block_c2_egress")(row)
    assert blocked[idx_443] == 0.0
    assert blocked[idx_iat_m] < 1.0


def test_quarantine_exfiltration_resets_bytes_ratio():
    idx_ratio = FEATURE_COLUMNS.index("bytes_ratio_out_in")
    idx_out = FEATURE_COLUMNS.index("outbound_bytes")
    idx_443 = FEATURE_COLUMNS.index("dst_port_is_443")
    row = np.zeros(N_FEATURES, dtype=np.float32)
    row[idx_ratio] = 15.0
    row[idx_out] = 100000.0
    row[idx_443] = 1.0

    mitigated = get_mitigation_fn("quarantine_exfiltration")(row)
    assert mitigated[idx_ratio] == 1.0
    assert mitigated[idx_443] == 0.0


def test_clamp_physical_bounds_preserves_valid_and_fixes_negative():
    row = np.zeros(N_FEATURES, dtype=np.float32)
    idx_flow = FEATURE_COLUMNS.index("flow_count")
    idx_port = FEATURE_COLUMNS.index("dst_port_is_22")
    row[idx_flow] = -10.0
    row[idx_port] = 1.5

    clamped = _clamp_physical_bounds(row)
    assert clamped[idx_flow] == 0.0
    assert clamped[idx_port] == 1.0


def test_no_mitigation_is_a_true_identity():
    row = np.random.default_rng(0).normal(size=N_FEATURES).astype(np.float32)
    mutated = get_mitigation_fn("no_mitigation")(row)
    np.testing.assert_array_equal(row, mutated)


def test_list_mitigations_matches_registry():
    listed = list_mitigations()
    assert {m["id"] for m in listed} == set(MITIGATIONS.keys())
    for m in listed:
        assert "label" in m and "description" in m and "category" in m


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
    baseline, mitigated, metrics = compare_with_and_without(model, scaler, seed_raw, fn, k=3)
    # no_mitigation is a true identity, so the two rollouts should track very
    # closely -- but not necessarily to bit-for-bit precision, since the
    # mitigated path (rollout_counterfactual) clamps each regressed state to
    # physical bounds (see counterfactual.py:_clamp_physical_bounds) before
    # feeding it back in, while the plain baseline rollout() does not. With
    # an untrained dummy model on random input, the regressed state can
    # legitimately drift outside those bounds, so a small divergence here is
    # the clamp doing its documented job, not a bug.
    np.testing.assert_allclose(baseline["infiltration_probs"], mitigated["infiltration_probs"], atol=2e-3)
    assert "risk_reduction_pct" in metrics
    assert "verdict" in metrics
    assert metrics["risk_reduction_pct"] == 0.0

