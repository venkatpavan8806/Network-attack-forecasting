"""Model-based "digital twin" what-if simulator.

Answers: "if we applied this mitigation starting now, how would the world
model's predicted infiltration trajectory change?" -- by reusing the
ALREADY-TRAINED LSTM world model and scaler, with no live network involved.

Mechanism: the mitigation function (app/simulation/mitigations.py) is
applied to the RAW, unscaled feature vector -- first to the most recent real
observed window (the mitigation "starts now"), then, because rollout is
autoregressive, to every subsequently predicted window as well (the
mitigation is treated as staying active for the whole forecast horizon).
Each predicted next-state is produced by the model in normalized space,
inverse-transformed back to raw feature space via the fitted scaler,
mutated by the mitigation, and re-transformed before being fed back in --
this is an exact round-trip through the same scaler used at training time.

This is explicitly NOT a live simulated network: no packets are generated
or injected, nothing here can affect a real host. It is decision support
only -- the defender reads the comparison and decides; nothing here applies
a mitigation automatically.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from app.config import ROLLOUT_K, IDX_TO_ACTION
from app.models.lstm_world_model import infiltration_probability


@torch.no_grad()
def rollout_counterfactual(model, scaler, seed_raw: np.ndarray, mitigation_fn, k: int = ROLLOUT_K):
    """seed_raw: (seq_len, n_features) RAW (unscaled) real observed feature
    values, most recent window last. Returns the same shape of result as
    app.models.lstm_world_model.rollout(), but under the sustained
    mitigation."""
    mitigated_raw = seed_raw.copy()
    mitigated_raw[-1] = mitigation_fn(mitigated_raw[-1])
    scaled_seq = scaler.transform(mitigated_raw).astype(np.float32)
    seq = torch.tensor(scaled_seq, dtype=torch.float32).unsqueeze(0)

    infiltration_probs, stage_probs_list, predicted_action = [], [], []
    for _ in range(k):
        stage_logits, next_state, _ = model(seq)
        probs = F.softmax(stage_logits, dim=1).squeeze(0).numpy()
        infiltration_probs.append(infiltration_probability(probs))
        stage_probs_list.append(probs.tolist())
        predicted_action.append(IDX_TO_ACTION[int(np.argmax(probs))])

        next_state_raw = scaler.inverse_transform(next_state.numpy())[0]
        next_state_raw = mitigation_fn(next_state_raw)
        next_state_scaled = scaler.transform(next_state_raw.reshape(1, -1))[0].astype(np.float32)
        next_state_t = torch.tensor(next_state_scaled).reshape(1, 1, -1)
        seq = torch.cat([seq[:, 1:, :], next_state_t], dim=1)

    return {
        "infiltration_probs": infiltration_probs,
        "stage_probs": stage_probs_list,
        "predicted_stage": predicted_action,
    }


def compare_with_and_without(model, scaler, seed_raw: np.ndarray, mitigation_fn, k: int = ROLLOUT_K):
    """Convenience wrapper: runs both the unmitigated and mitigated rollout
    from the SAME real seed window, so the two curves are directly comparable."""
    from app.models.lstm_world_model import rollout

    seed_scaled = scaler.transform(seed_raw).astype(np.float32)
    baseline = rollout(model, seed_scaled, k=k)
    mitigated = rollout_counterfactual(model, scaler, seed_raw, mitigation_fn, k=k)
    return baseline, mitigated
