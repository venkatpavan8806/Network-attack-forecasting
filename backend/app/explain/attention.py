"""Attention-based explainability for the LSTM world model -- the project's
default explainability mechanism. Combines two real, model-derived signals
for a given prediction:
  1. temporal attention weights: which of the past SEQ_LEN windows the model
     leaned on most for this forecast (from the model's own softmax attention).
  2. input-gradient saliency: which specific features, within the most
     heavily-attended window(s), pushed the infiltration probability up.
Both come straight out of the trained model on the actual input -- neither
is a static or hard-coded importance table.
"""
from __future__ import annotations

import numpy as np

from app.config import FEATURE_COLUMNS
from app.models.lstm_world_model import one_step_forecast, input_gradient_saliency, infiltration_probability


def explain_prediction(model, window: np.ndarray, top_k: int = 5) -> dict:
    """window: (seq_len, n_features) normalized real feature sequence."""
    probs, attn_weights, _ = one_step_forecast(model, window)
    grad = input_gradient_saliency(model, window)  # (seq_len, n_features)

    # weight each timestep's per-feature gradient by that timestep's attention,
    # then take the top contributing (timestep, feature) pairs by |weighted grad|
    weighted = grad * attn_weights[:, None]
    seq_len, n_feat = weighted.shape
    flat_idx = np.argsort(-np.abs(weighted).flatten())[:top_k]
    top_contributors = []
    for idx in flat_idx:
        t_idx, f_idx = divmod(idx, n_feat)
        top_contributors.append({
            "timestep_offset": int(t_idx - (seq_len - 1)),  # 0 = most recent window, negative = further back
            "feature": FEATURE_COLUMNS[f_idx],
            "attention_weight": round(float(attn_weights[t_idx]), 4),
            "raw_value_normalized": round(float(window[t_idx, f_idx]), 4),
            "contribution_score": round(float(weighted[t_idx, f_idx]), 5),
        })

    return {
        "infiltration_probability": round(infiltration_probability(probs), 4),
        "stage_probabilities": {c: round(float(p), 4) for c, p in zip(_stage_classes(), probs)},
        "attention_over_past_windows": [round(float(w), 4) for w in attn_weights],
        "top_contributors": top_contributors,
    }


def _stage_classes():
    from app.config import STAGE_CLASSES
    return STAGE_CLASSES
