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
boundary-clamped to valid physical domains (non-negative counts, valid ratios),
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

from app.config import ROLLOUT_K, IDX_TO_ACTION, FEATURE_COLUMNS
from app.models.lstm_world_model import infiltration_probability

_IDX = {name: i for i, name in enumerate(FEATURE_COLUMNS)}


def _clamp_physical_bounds(row: np.ndarray) -> np.ndarray:
    """Clamps regressed raw feature values to realistic physical boundaries
    to prevent numerical drift during autoregressive rollout."""
    row = row.copy()
    # counts and byte totals must be non-negative
    for col in [
        "flow_count", "unique_dst_ports", "unique_dst_ips",
        "syn_count", "ack_count", "fin_count", "rst_count", "psh_count",
        "inbound_bytes", "outbound_bytes", "avg_bytes_per_flow", "std_bytes_per_flow",
        "avg_pkts_per_flow", "avg_flow_duration", "std_flow_duration",
        "iat_mean", "iat_std", "ttl_mean", "ttl_std",
        "win_size_mean", "win_size_std", "pkt_len_mean", "pkt_len_std",
    ]:
        if col in _IDX:
            row[_IDX[col]] = max(0.0, float(row[_IDX[col]]))

    # ratios and indicators must stay within [0, 1]
    for col in [
        "failed_conn_ratio", "new_dst_ip_ratio", "port_scan_score",
        "dst_port_is_22", "dst_port_is_445", "dst_port_is_3389", "dst_port_is_443",
    ]:
        if col in _IDX:
            row[_IDX[col]] = float(np.clip(row[_IDX[col]], 0.0, 1.0))

    if "bytes_ratio_out_in" in _IDX:
        row[_IDX["bytes_ratio_out_in"]] = max(0.0, float(row[_IDX["bytes_ratio_out_in"]]))

    return row


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

    # Compute comparative quantitative metrics
    mean_without = float(np.mean(baseline["infiltration_probs"]))
    mean_with = float(np.mean(mitigated["infiltration_probs"]))
    peak_without = float(np.max(baseline["infiltration_probs"]))
    peak_with = float(np.max(mitigated["infiltration_probs"]))

    if mean_without > 0.01:
        risk_reduction_pct = round(max(0.0, (mean_without - mean_with) / mean_without * 100.0), 1)
    else:
        risk_reduction_pct = 0.0

    if risk_reduction_pct >= 60.0:
        verdict = "High Efficacy (Attack Successfully Averted)"
    elif risk_reduction_pct >= 25.0:
        verdict = "Moderate Efficacy (Attack Trajectory Slowed)"
    elif risk_reduction_pct > 5.0:
        verdict = "Low Efficacy (Marginal Risk Change)"
    else:
        verdict = "Neutral (No Observable Divergence)"

    return baseline, mitigated, {
        "mean_without": round(mean_without, 4),
        "mean_with": round(mean_with, 4),
        "peak_risk_without": round(peak_without, 4),
        "peak_risk_with": round(peak_with, 4),
        "risk_reduction_pct": risk_reduction_pct,
        "verdict": verdict,
    }

