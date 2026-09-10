import numpy as np
import torch

from app.config import N_FEATURES, SEQ_LEN, STAGE_CLASSES
from app.models.lstm_world_model import (
    LSTMWorldModel, infiltration_probability, rollout, one_step_forecast, input_gradient_saliency,
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
