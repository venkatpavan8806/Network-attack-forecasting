"""Safe sandbox environment for Digital Twin execution.

Full pipeline:
  1. Build initial twin from configurable state provider
  2. Clone twin state (``real_network_touched: false``)
  3. Apply mitigation to CLONED twin state (firewall rules, service filtering)
  4. Simulate attacker interaction → outcome (BLOCKED/THROTTLED/CONTINUED)
  5. Recalculate topology-based attack paths
  6. **Synthesise telemetry FROM the changed twin state** (NOT ``mitigation_fn``)
  7. Feed synthesised telemetry to the existing LSTM World Model
  8. Return full results

There is NO double-application of any transform.  The telemetry synthesiser
reads the twin state once to produce the LSTM seed, and provides a sustained
transform for the autoregressive rollout — both derived from the same twin
state, not from a named mitigation function.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from app.config import PORT_OF_ACTION, WATCHED_PORTS, ROLLOUT_K
from app.simulation.network_twin import (
    build_initial_twin_network, FirewallRule, TwinNetwork,
)
from app.simulation.attacker import GenericAttacker, AttackState
from app.simulation.path_predictor import PathPredictor
from app.simulation.telemetry_synthesizer import (
    TelemetrySynthesizer, rollout_with_twin_transform,
)
from app.simulation.mitigations import list_mitigations, MITIGATIONS
from app.models.lstm_world_model import rollout as lstm_rollout


# --------------------------------------------------------------------------- #
# Reverse lookup: mitigation_id → port (derived from PORT_OF_ACTION)
# --------------------------------------------------------------------------- #
_MITIGATION_TO_PORT: Dict[str, Optional[int]] = {}
for _mit_id in MITIGATIONS:
    if _mit_id.startswith("block_") or _mit_id.startswith("rate_limit_"):
        _suffix = _mit_id.replace("block_", "").replace("rate_limit_", "")
        _base_keyword = _suffix.split("_")[0]          # "c2" from "c2_egress"
        _matched_port = None
        for _action, _port in PORT_OF_ACTION.items():
            if _action.startswith(_suffix) or _action.startswith(_base_keyword):
                _matched_port = _port
                break
        _MITIGATION_TO_PORT[_mit_id] = _matched_port


# --------------------------------------------------------------------------- #
# Sandbox
# --------------------------------------------------------------------------- #
class DigitalTwinSandbox:
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler

    def run_sandbox_simulation(
        self,
        host_id: str,
        mitigation_id: str,
        seed_raw: np.ndarray,
        at_window_idx: int,
        true_stage: Optional[str] = None,
        state_label: Optional[str] = None,
        is_live: bool = False,
    ) -> Dict[str, Any]:
        """Runs a safe sandbox simulation on a cloned Digital Twin state."""

        # 1. Build initial twin from configurable state provider
        initial_twin = build_initial_twin_network(host_id, is_live=is_live)

        # 2. Clone twin state
        cloned_twin = initial_twin.clone()
        assert not cloned_twin.real_network_touched

        # 3. Apply mitigation to CLONED twin state
        mits_meta = {m["id"]: m for m in list_mitigations()}
        mit_info = mits_meta.get(mitigation_id, {
            "id": mitigation_id, "label": mitigation_id,
            "description": "Custom defensive mitigation",
        })
        self._apply_mitigation_to_twin_state(cloned_twin, host_id, mitigation_id)

        # 4. Simulate attacker interaction
        current_stage = (true_stage if (true_stage and true_stage != "benign")
                         else "port_scan")
        attack_state = AttackState(
            current_stage=current_stage, target_host_id=host_id)
        attacker_response = GenericAttacker.step(cloned_twin, attack_state)

        # 5. Recalculate paths on modified topology
        path_engine = PathPredictor(cloned_twin)
        path_analysis = path_engine.calculate_attack_paths(
            source_host_id="external_attacker",
            target_host_id=host_id,
            current_stage=current_stage,
        )

        # 6. Synthesise telemetry FROM the changed twin state
        synthesizer = TelemetrySynthesizer()
        simulated_features = synthesizer.synthesize(
            twin=cloned_twin,
            target_host_id=host_id,
            original_features=seed_raw[-1],
            attacker_outcome=attacker_response.outcome,
        )

        # Build the prepared seed: windows 0..n-2 are original,
        # window n-1 is the twin-state-derived simulated telemetry
        prepared_seed = seed_raw.copy()
        prepared_seed[-1] = simulated_features

        # Build a sustained transform for the autoregressive rollout
        twin_transform = synthesizer.build_sustained_transform(
            twin=cloned_twin,
            target_host_id=host_id,
            attacker_outcome=attacker_response.outcome,
        )

        # 7a. Baseline rollout: ORIGINAL observed data, no mitigation
        baseline_scaled = self.scaler.transform(seed_raw).astype(np.float32)
        baseline_roll = lstm_rollout(self.model, baseline_scaled, k=ROLLOUT_K)

        # 7b. Mitigated rollout: twin-state-derived telemetry + sustained transform
        mitigated_roll = rollout_with_twin_transform(
            self.model, self.scaler, prepared_seed, twin_transform, k=ROLLOUT_K,
        )

        # Compute comparative metrics
        mean_without = float(np.mean(baseline_roll["infiltration_probs"]))
        mean_with = float(np.mean(mitigated_roll["infiltration_probs"]))
        peak_without = float(np.max(baseline_roll["infiltration_probs"]))
        peak_with = float(np.max(mitigated_roll["infiltration_probs"]))

        if mean_without > 0.01:
            risk_reduction_pct = round(
                max(0.0, (mean_without - mean_with) / mean_without * 100.0), 1)
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

        metrics = {
            "mean_without": round(mean_without, 4),
            "mean_with": round(mean_with, 4),
            "peak_risk_without": round(peak_without, 4),
            "peak_risk_with": round(peak_with, 4),
            "risk_reduction_pct": risk_reduction_pct,
            "verdict": verdict,
        }

        # Divergences
        divergences = []
        for h, (a, b) in enumerate(
            zip(baseline_roll["predicted_stage"],
                mitigated_roll["predicted_stage"]), start=1):
            if a != b:
                divergences.append({
                    "horizon": h,
                    "without_mitigation_action": a,
                    "with_mitigation_action": b,
                })

        return {
            "real_network_touched": False,
            "is_simulated_twin": True,
            "host_id": host_id,
            "window_idx": at_window_idx,
            "true_stage": true_stage,
            "state_label": state_label,
            "mitigation": mit_info,
            "initial_twin_state": initial_twin.to_dict(),
            "cloned_twin_state": cloned_twin.to_dict(),
            "simulated_attacker_response": attacker_response.to_dict(),
            "path_prediction": path_analysis,
            "horizon_windows": ROLLOUT_K,
            "without_mitigation": {
                "infiltration_probs": [round(p, 4) for p in baseline_roll["infiltration_probs"]],
                "predicted_stage_per_horizon": baseline_roll["predicted_stage"],
            },
            "with_mitigation": {
                "infiltration_probs": [round(p, 4) for p in mitigated_roll["infiltration_probs"]],
                "predicted_stage_per_horizon": mitigated_roll["predicted_stage"],
            },
            "action_divergences": divergences,
            "metrics": metrics,
            "risk_reduction_pct": risk_reduction_pct,
            "verdict": verdict,
        }

    # ------------------------------------------------------------------ #
    # Twin-state mutation (firewall rules, service status, isolation)
    # ------------------------------------------------------------------ #
    def _apply_mitigation_to_twin_state(self, twin: TwinNetwork,
                                         host_id: str, mitigation_id: str):
        """Modifies the logical twin state based on the mitigation ID.
        Port lookups derived dynamically from PORT_OF_ACTION."""
        target_host = twin.hosts.get(host_id)

        if mitigation_id == "isolate_host":
            if target_host:
                target_host.isolate()
            twin.firewall.add_rule(FirewallRule(
                rule_id="ISO-001", source="*", destination=host_id,
                port=None, action="DENY",
                description=(f"Quarantine host '{host_id}': sever all "
                             f"external & lateral communication"),
                data_source="configured",
            ))

        elif mitigation_id.startswith("block_"):
            port = _MITIGATION_TO_PORT.get(mitigation_id)
            if port and target_host:
                target_host.set_service_status(port, "filtered")
            twin.firewall.add_rule(FirewallRule(
                rule_id=f"ACL-{mitigation_id.upper()}",
                source="*", destination=host_id, port=port, action="DENY",
                description=(f"Firewall ACL: drop traffic for "
                             f"'{mitigation_id}'" +
                             (f" on port {port}" if port else "")),
                data_source="configured",
            ))

        elif mitigation_id.startswith("rate_limit_"):
            port = _MITIGATION_TO_PORT.get(mitigation_id)
            twin.firewall.add_rule(FirewallRule(
                rule_id=f"SHAPE-{mitigation_id.upper()}",
                source="*", destination=host_id, port=port,
                action="RATE_LIMIT", rate_limit_factor=0.25,
                description=(f"Traffic shaping: throttle" +
                             (f" port {port}" if port else " all ports") +
                             " to 25%"),
                data_source="configured",
            ))

        elif mitigation_id == "quarantine_exfiltration":
            exfil_port = PORT_OF_ACTION.get("data_exfiltration")
            if target_host and exfil_port:
                target_host.set_service_status(exfil_port, "filtered")
            twin.firewall.add_rule(FirewallRule(
                rule_id="DLP-EXFIL-001", source="*", destination=host_id,
                port=exfil_port, action="DENY",
                description=(f"DLP Egress Filter: terminate exfiltration" +
                             (f" on port {exfil_port}" if exfil_port else "")),
                data_source="configured",
            ))

        elif mitigation_id != "no_mitigation":
            twin.firewall.add_rule(FirewallRule(
                rule_id=f"CUSTOM-{mitigation_id.upper()[:16]}",
                source="*", destination=host_id, port=None, action="DENY",
                description=f"Custom mitigation '{mitigation_id}'",
                data_source="configured",
            ))
