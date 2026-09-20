import numpy as np
import torch

from app.config import N_FEATURES, SEQ_LEN, STAGE_CLASSES
from app.models.lstm_world_model import (
    LSTMWorldModel, infiltration_probability, rollout, one_step_forecast, input_gradient_saliency,
    branching_rollout, enumerate_paths,
)


def _dummy_model():
    torch.manual_seed(0)
    return LSTMWorldModel(n_features=N_FEATURES, hidden_size=8, num_layers=1, n_classes=len(STAGE_CLASSES))


def test_forward_output_shapes():
    model = _dummy_model()
    x = torch.randn(4, SEQ_LEN, N_FEATURES)
    stage_logits, next_state, attn = model(x)
    assert stage_logits.shape == (4, len(STAGE_CLASSES))
    assert next_state.shape == (4, N_FEATURES)
    assert attn.shape == (4, SEQ_LEN)


def test_attention_weights_sum_to_one():
    model = _dummy_model()
    x = torch.randn(3, SEQ_LEN, N_FEATURES)
    _, _, attn = model(x)
    sums = attn.sum(dim=1).detach().numpy()
    np.testing.assert_allclose(sums, np.ones(3), atol=1e-5)


def test_infiltration_probability_range():
    probs = np.array([0.9, 0.02, 0.02, 0.02, 0.02, 0.01, 0.01])
    p = infiltration_probability(probs)
    assert 0.0 <= p <= 1.0
    assert abs(p - 0.1) < 1e-6


def test_rollout_output_shapes_and_range():
    model = _dummy_model()
    seed_window = np.random.randn(SEQ_LEN, N_FEATURES).astype(np.float32)
    result = rollout(model, seed_window, k=5)
    assert len(result["infiltration_probs"]) == 5
    assert len(result["stage_probs"]) == 5
    assert len(result["predicted_stage"]) == 5
    for p in result["infiltration_probs"]:
        assert 0.0 <= p <= 1.0
    for dist in result["stage_probs"]:
        assert abs(sum(dist) - 1.0) < 1e-4
    assert len(result["attn_weights_step1"]) == SEQ_LEN


def test_one_step_forecast_probs_sum_to_one():
    model = _dummy_model()
    window = np.random.randn(SEQ_LEN, N_FEATURES).astype(np.float32)
    probs, attn, next_state = one_step_forecast(model, window)
    assert abs(probs.sum() - 1.0) < 1e-4
    assert next_state.shape == (N_FEATURES,)


def test_input_gradient_saliency_shape():
    model = _dummy_model()
    window = np.random.randn(SEQ_LEN, N_FEATURES).astype(np.float32)
    grad = input_gradient_saliency(model, window)
    assert grad.shape == (SEQ_LEN, N_FEATURES)
    assert np.isfinite(grad).all()


def test_branching_rollout_tree_shape_and_mitre_mapping():
    model = _dummy_model()
    seed_window = np.random.randn(SEQ_LEN, N_FEATURES).astype(np.float32)
    result = branching_rollout(model, seed_window, depth=3, branch_factor=2, min_path_prob=0.0)
    root = result["root"]
    assert root["stage"] is None
    assert 1 <= len(root["children"]) <= 2
    for child in root["children"]:
        assert child["stage"] in STAGE_CLASSES
        assert child["depth"] == 1
        assert child["attack_mapping"]["state_label"] == child["stage"]
        assert 0.0 <= child["step_probability"] <= 1.0
        assert 0.0 <= child["path_probability"] <= child["step_probability"] + 1e-6
        assert 0.0 <= child["infiltration_probability"] <= 1.0
        # depth 3 requested -> grandchildren exist and are correctly depth-tagged
        for grandchild in child["children"]:
            assert grandchild["depth"] == 2
            for leaf in grandchild["children"]:
                assert leaf["depth"] == 3
                assert leaf["children"] == []  # tree stops at requested depth


def test_branching_rollout_prunes_low_probability_paths():
    model = _dummy_model()
    seed_window = np.random.randn(SEQ_LEN, N_FEATURES).astype(np.float32)
    result = branching_rollout(model, seed_window, depth=3, branch_factor=3, min_path_prob=0.9)
    paths = enumerate_paths(result["root"])
    for p in paths:
        assert p["path_probability"] >= 0.9


def test_branching_rollout_state_blend_affects_deeper_steps():
    """With state_blend=1.0, each branch's continued state comes entirely
    from its class's mean vector, so swapping in a different set of mean
    vectors must change the second-step probabilities -- proving depth > 1
    branches actually condition on which first-step branch was taken,
    rather than every sibling silently rolling forward identically."""
    model = _dummy_model()
    seed_window = np.random.randn(SEQ_LEN, N_FEATURES).astype(np.float32)
    rng = np.random.default_rng(0)
    means_a = {s: rng.normal(0, 1, N_FEATURES).astype(np.float32).tolist() for s in STAGE_CLASSES}
    means_b = {s: rng.normal(5, 1, N_FEATURES).astype(np.float32).tolist() for s in STAGE_CLASSES}

    result_a = branching_rollout(model, seed_window, stage_mean_vectors=means_a,
                                  depth=2, branch_factor=3, min_path_prob=0.0, state_blend=1.0)
    result_b = branching_rollout(model, seed_window, stage_mean_vectors=means_b,
                                  depth=2, branch_factor=3, min_path_prob=0.0, state_blend=1.0)

    def _second_step_probs(root):
        return [gc["step_probability"] for c in root["children"] for gc in c["children"]]

    assert _second_step_probs(result_a["root"]) != _second_step_probs(result_b["root"])


def test_enumerate_paths_sorted_by_probability_descending():
    model = _dummy_model()
    seed_window = np.random.randn(SEQ_LEN, N_FEATURES).astype(np.float32)
    result = branching_rollout(model, seed_window, depth=2, branch_factor=3, min_path_prob=0.0)
    paths = enumerate_paths(result["root"])
    assert len(paths) > 0
    probs = [p["path_probability"] for p in paths]
    assert probs == sorted(probs, reverse=True)
    for p in paths:
        assert len(p["stages"]) == 2
        assert len(p["mitre_kill_chain"]) == 2
        assert all("technique_id" in m for m in p["mitre_kill_chain"])
