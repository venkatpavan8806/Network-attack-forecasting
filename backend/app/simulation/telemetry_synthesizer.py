"""Telemetry synthesiser: generates simulated feature vectors FROM the
Digital Twin's post-mitigation state.

This is the bridge between the logical twin state and the LSTM world model.
Instead of calling ``mitigation_fn(original_features)`` (a named, pre-defined
feature transform), this module **reads the actual twin state** — which
services are filtered, which hosts are isolated, what firewall rules exist,
what the attacker outcome was — and derives what a telemetry sensor would
plausibly observe in that state.

The resulting feature vector is what the LSTM receives as its input.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np
import torch
# pyrefly: ignore [missing-import]
import torch.nn.functional as F

from app.config import FEATURE_COLUMNS, WATCHED_PORTS, PORT_OF_ACTION, ROLLOUT_K, IDX_TO_ACTION
from app.models.lstm_world_model import infiltration_probability
from app.simulation.network_twin import TwinNetwork


# ------------------------------------------------------------------ #
# Dynamic feature-column index maps (derived from FEATURE_COLUMNS,
# never from hardcoded column names).
# ------------------------------------------------------------------ #
_IDX: Dict[str, int] = {col: i for i, col in enumerate(FEATURE_COLUMNS)}

# Per-port feature indices — discovered dynamically from FEATURE_COLUMNS
_PORT_INDICATOR: Dict[int, int] = {}     # port → index of dst_port_is_{port}
_PORT_SYN_COUNT: Dict[int, int] = {}     # port → index of syn_count_port_{port}
_PORT_FAIL_RATIO: Dict[int, int] = {}    # port → index of failed_conn_ratio_port_{port}

for _i, _col in enumerate(FEATURE_COLUMNS):
    if _col.startswith("dst_port_is_"):
        try:
            _PORT_INDICATOR[int(_col.split("_")[-1])] = _i
        except ValueError:
            pass
    elif _col.startswith("syn_count_port_"):
        try:
            _PORT_SYN_COUNT[int(_col.split("_")[-1])] = _i
        except ValueError:
            pass
    elif _col.startswith("failed_conn_ratio_port_"):
        try:
            _PORT_FAIL_RATIO[int(_col.split("_")[-1])] = _i
        except ValueError:
            pass

# Aggregate traffic feature names (for suppression during isolation / blocking)
_COUNT_FEATURES = [
    c for c in [
        "flow_count", "syn_count", "ack_count", "fin_count",
        "rst_count", "psh_count", "inbound_bytes", "outbound_bytes",
        "avg_bytes_per_flow", "std_bytes_per_flow", "avg_pkts_per_flow",
    ] if c in _IDX
]

# Benign-baseline values — imported from mitigations so the isolation
# profile is consistent with the rest of the project.  These are NOT
# used as a "mitigation_fn"; they are the reference point for what a
# sensor would see on a host with no malicious traffic.
from app.simulation.mitigations import BENIGN_PROFILE      # noqa: E402

_BENIGN_VEC = np.array(
    [BENIGN_PROFILE.get(col, 0.0) for col in FEATURE_COLUMNS],
    dtype=np.float32,
)


# ------------------------------------------------------------------ #
# Physical-bounds clamping (same logic as counterfactual.py, kept local
# to avoid importing a private function).
# ------------------------------------------------------------------ #
def _clamp_physical(row: np.ndarray) -> np.ndarray:
    row = row.copy()
    for col in _COUNT_FEATURES:
        row[_IDX[col]] = max(0.0, float(row[_IDX[col]]))
    for col in ["failed_conn_ratio", "new_dst_ip_ratio", "port_scan_score"]:
        if col in _IDX:
            row[_IDX[col]] = float(np.clip(row[_IDX[col]], 0.0, 1.0))
    for port, idx in _PORT_INDICATOR.items():
        row[idx] = float(np.clip(row[idx], 0.0, 1.0))
    for port, idx in _PORT_FAIL_RATIO.items():
        row[idx] = float(np.clip(row[idx], 0.0, 1.0))
    if "bytes_ratio_out_in" in _IDX:
        row[_IDX["bytes_ratio_out_in"]] = max(0.0, float(row[_IDX["bytes_ratio_out_in"]]))
    return row


# ------------------------------------------------------------------ #
# Synthesiser
# ------------------------------------------------------------------ #
class TelemetrySynthesizer:
    """Generates a simulated feature vector by reading the Digital Twin's
    current state, NOT by calling a named mitigation function.

    Logic
    -----
    1. **Host isolation** → collapse all traffic toward the benign baseline.
    2. **Per-port service/firewall state** → for each watched port, check
       whether the service is filtered or firewalled.  If DENY → zero the
       port's indicators + SYN count + failure ratio.  If RATE_LIMIT →
       scale them down by the rule's throttle factor.
    3. **Aggregate suppression** → reduce overall traffic counts by the
       fraction of ports that are blocked or throttled.
    4. **Attacker outcome adjustment** → if the attacker was BLOCKED,
       suppress residual attack indicators (RST storms, scan scores).
    """

    def synthesize(
        self,
        twin: TwinNetwork,
        target_host_id: str,
        original_features: np.ndarray,
        attacker_outcome: str,       # "BLOCKED" | "THROTTLED" | "CONTINUED"
    ) -> np.ndarray:
        """Returns a new feature vector reflecting the post-mitigation twin state."""
        simulated = original_features.copy().astype(np.float32)
        target_host = twin.hosts.get(target_host_id)

        # ---- 1. Host isolation ---------------------------------------- #
        if target_host and target_host.is_isolated:
            # Blend 95% toward benign baseline — an isolated host would
            # show near-zero malicious traffic but still have some
            # background noise from the sensor itself.
            simulated = 0.05 * simulated + 0.95 * _BENIGN_VEC
            # Force attack-specific indicators to zero
            for port_idx in _PORT_INDICATOR.values():
                simulated[port_idx] = 0.0
            for port_idx in _PORT_SYN_COUNT.values():
                simulated[port_idx] = 0.0
            for port_idx in _PORT_FAIL_RATIO.values():
                simulated[port_idx] = 0.0
            if "port_scan_score" in _IDX:
                simulated[_IDX["port_scan_score"]] = 0.0
            if "new_dst_ip_ratio" in _IDX:
                simulated[_IDX["new_dst_ip_ratio"]] = _BENIGN_VEC[_IDX["new_dst_ip_ratio"]]
            return _clamp_physical(simulated)

        # ---- 2. Per-port service + firewall state --------------------- #
        blocked_ports = 0
        throttled_ports = 0
        total_watched = len(WATCHED_PORTS)

        for port in WATCHED_PORTS:
            svc = target_host.services.get(port) if target_host else None
            fw_action, fw_rule = twin.firewall.evaluate("*", target_host_id, port)

            is_denied = (
                (svc and svc.get("status") == "filtered")
                or fw_action == "DENY"
            )
            is_throttled = (fw_action == "RATE_LIMIT")

            if is_denied:
                blocked_ports += 1
                # Zero this port's indicator, SYN intensity, failure ratio
                if port in _PORT_INDICATOR:
                    simulated[_PORT_INDICATOR[port]] = 0.0
                if port in _PORT_SYN_COUNT:
                    simulated[_PORT_SYN_COUNT[port]] = 0.0
                if port in _PORT_FAIL_RATIO:
                    simulated[_PORT_FAIL_RATIO[port]] = 0.0

            elif is_throttled:
                throttled_ports += 1
                factor = fw_rule.rate_limit_factor if fw_rule else 0.25
                if port in _PORT_SYN_COUNT:
                    simulated[_PORT_SYN_COUNT[port]] *= factor
                if port in _PORT_FAIL_RATIO:
                    simulated[_PORT_FAIL_RATIO[port]] *= factor

        # ---- 3. Aggregate traffic suppression ------------------------- #
        if total_watched > 0:
            block_frac = blocked_ports / total_watched
            throttle_frac = throttled_ports / total_watched
            suppression = max(0.05, 1.0 - block_frac * 0.8 - throttle_frac * 0.2)

            for col in _COUNT_FEATURES:
                simulated[_IDX[col]] *= suppression

            if block_frac > 0.3:
                if "failed_conn_ratio" in _IDX:
                    simulated[_IDX["failed_conn_ratio"]] *= (1.0 - block_frac)
                if "port_scan_score" in _IDX:
                    simulated[_IDX["port_scan_score"]] *= (1.0 - block_frac)
                if "unique_dst_ports" in _IDX:
                    simulated[_IDX["unique_dst_ports"]] = max(
                        1.0, simulated[_IDX["unique_dst_ports"]] * (1.0 - block_frac))
                if "new_dst_ip_ratio" in _IDX:
                    simulated[_IDX["new_dst_ip_ratio"]] *= (1.0 - block_frac)

        # ---- 4. Attacker-outcome adjustment --------------------------- #
        if attacker_outcome == "BLOCKED":
            if "rst_count" in _IDX:
                simulated[_IDX["rst_count"]] *= 0.3
            if "failed_conn_ratio" in _IDX:
                simulated[_IDX["failed_conn_ratio"]] = min(
                    simulated[_IDX["failed_conn_ratio"]], 0.05)

        elif attacker_outcome == "THROTTLED":
            if "avg_flow_duration" in _IDX:
                simulated[_IDX["avg_flow_duration"]] *= 1.5   # throttled → longer flows
            if "iat_mean" in _IDX:
                simulated[_IDX["iat_mean"]] *= 2.0            # longer inter-arrival

        return _clamp_physical(simulated)

    # ------------------------------------------------------------------ #
    # Sustained transform for autoregressive LSTM rollout
    # ------------------------------------------------------------------ #
    def build_sustained_transform(
        self,
        twin: TwinNetwork,
        target_host_id: str,
        attacker_outcome: str,
    ) -> Callable[[np.ndarray], np.ndarray]:
        """Returns a callable ``f(features) → features`` that applies the
        twin-state-derived feature adjustment to every autoregressively
        predicted future window.  The LSTM's ``next_state`` prediction is
        what *would* happen without mitigation; this transform adjusts it
        to reflect the ongoing twin-state conditions."""
        def _transform(features: np.ndarray) -> np.ndarray:
            return self.synthesize(twin, target_host_id, features, attacker_outcome)
        return _transform


# ------------------------------------------------------------------ #
# LSTM rollout driven by twin-state transform (no mitigation_fn)
# ------------------------------------------------------------------ #
@torch.no_grad()
def rollout_with_twin_transform(
    model, scaler,
    prepared_seed_raw: np.ndarray,
    twin_transform: Callable[[np.ndarray], np.ndarray],
    k: int = ROLLOUT_K,
):
    """Autoregressive LSTM rollout using a **twin-state-derived transform**
    instead of a named mitigation function.

    ``prepared_seed_raw`` is a ``(seq_len, n_features)`` RAW feature
    sequence whose **last window has already been synthesised** from the
    post-mitigation twin state.  This function does NOT re-apply the
    transform to the seed — it scales the seed as-is, runs the model,
    and applies ``twin_transform`` only to each **predicted** future
    state (the same role ``mitigation_fn`` played in the old
    ``rollout_counterfactual``, but derived from twin state, not from
    a named function).
    """
    scaled_seq = scaler.transform(prepared_seed_raw).astype(np.float32)
    seq = torch.tensor(scaled_seq, dtype=torch.float32).unsqueeze(0)

    infiltration_probs, stage_probs_list, predicted_stages = [], [], []

    for _ in range(k):
        stage_logits, next_state, _ = model(seq)
        probs = F.softmax(stage_logits, dim=1).squeeze(0).numpy()
        infiltration_probs.append(infiltration_probability(probs))
        stage_probs_list.append(probs.tolist())
        predicted_stages.append(IDX_TO_ACTION[int(np.argmax(probs))])

        # inverse-transform → clamp → twin-state transform → re-scale
        next_raw = scaler.inverse_transform(next_state.numpy())[0]
        next_raw = _clamp_physical(next_raw)
        next_raw = twin_transform(next_raw)
        next_scaled = scaler.transform(next_raw.reshape(1, -1))[0].astype(np.float32)
        next_t = torch.tensor(next_scaled).reshape(1, 1, -1)
        seq = torch.cat([seq[:, 1:, :], next_t], dim=1)

    return {
        "infiltration_probs": infiltration_probs,
        "stage_probs": stage_probs_list,
        "predicted_stage": predicted_stages,
    }
