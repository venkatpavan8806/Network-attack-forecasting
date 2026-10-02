"""Safe sandbox environment for Digital Twin execution.

Coordinates:
- Twin network state creation and safe cloning (`real_network_touched: false`)
- Mitigation application to the CLONED network state
- Generic attacker interaction with the modified state
- Dynamic path recalculation across network topology
- Integration with the existing trained LSTM World Model
"""
from __future__ import annotations

from typing import Any, Dict, Optional
import numpy as np

from app.config import PORT_OF_ACTION, ROLLOUT_K
from app.simulation.network_twin import build_initial_twin_network, FirewallRule, TwinNetwork
from app.simulation.attacker import GenericAttacker, AttackState
from app.simulation.path_predictor import PathPredictor
from app.simulation.mitigations import list_mitigations, get_mitigation_fn
from app.simulation.counterfactual import compare_with_and_without, rollout_counterfactual


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
        # 1. Build initial network twin state
        initial_twin = build_initial_twin_network(host_id, is_live=is_live)
        
        # 2. Clone twin state for safe sandbox execution
        cloned_twin = initial_twin.clone()
        assert not cloned_twin.real_network_touched, "Sandbox safety check failed: real network touched!"

        # 3. Apply mitigation to CLONED network twin state
        mitigations_meta = {m["id"]: m for m in list_mitigations()}
        mit_info = mitigations_meta.get(mitigation_id, {
            "id": mitigation_id,
            "label": mitigation_id,
            "description": "Custom defensive mitigation",
        })

        self._apply_mitigation_to_twin_state(cloned_twin, host_id, mitigation_id)

        # 4. Identify attack state from observed ground truth / stage
        current_stage = true_stage if (true_stage and true_stage != "benign") else "port_scan"
        attack_state = AttackState(
            current_stage=current_stage,
            target_host_id=host_id,
            observed_features=seed_raw[-1].copy(),
        )

        # 5. Run generic attacker interaction against modified twin state
        simulated_response = GenericAttacker.step(cloned_twin, attack_state, mitigation_id=mitigation_id)

        # 6. Recalculate network graph paths on cloned twin state
        path_engine = PathPredictor(cloned_twin)
        path_analysis = path_engine.calculate_attack_paths(
            source_host_id="external_attacker",
            target_host_id=host_id,
            current_stage=current_stage,
        )

        # 7. Run existing LSTM World Model rollout using the transformed twin telemetry
        mitigation_fn = get_mitigation_fn(mitigation_id)
        baseline_roll, mitigated_roll, metrics = compare_with_and_without(
            self.model, self.scaler, seed_raw, mitigation_fn, k=ROLLOUT_K
        )

        divergences = []
        for h, (a, b) in enumerate(zip(baseline_roll["predicted_stage"], mitigated_roll["predicted_stage"]), start=1):
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
            "simulated_attacker_response": simulated_response.to_dict(),
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
            "risk_reduction_pct": metrics["risk_reduction_pct"],
            "verdict": metrics["verdict"],
        }

    def _apply_mitigation_to_twin_state(self, twin: TwinNetwork, host_id: str, mitigation_id: str):
        """Modifies the logical network twin state (firewall rules, host isolation)
        based on the selected mitigation ID dynamically."""
        target_host = twin.hosts.get(host_id)

        if mitigation_id == "isolate_host":
            if target_host:
                target_host.isolate()
            twin.firewall.add_rule(FirewallRule(
                rule_id="ISO-001",
                source="*",
                destination=host_id,
                port=None,
                action="DENY",
                description=f"Quarantine host '{host_id}': sever all external & lateral communication",
            ))

        elif mitigation_id.startswith("block_"):
            if mitigation_id == "block_ssh":
                port = 22
            elif mitigation_id == "block_rdp":
                port = 3389
            elif mitigation_id == "block_smb":
                port = 445
            elif mitigation_id == "block_c2_egress":
                port = 443
            elif mitigation_id == "block_scanner_ip":
                port = None
            else:
                port = None

            if port and target_host:
                target_host.set_service_status(port, "filtered")

            twin.firewall.add_rule(FirewallRule(
                rule_id=f"ACL-{mitigation_id.upper()}",
                source="*",
                destination=host_id,
                port=port,
                action="DENY",
                description=f"Firewall ACL rule: drop traffic for mitigation '{mitigation_id}'",
            ))

        elif mitigation_id.startswith("rate_limit_"):
            if "ssh" in mitigation_id:
                port = 22
            elif "rdp" in mitigation_id:
                port = 3389
            elif "smb" in mitigation_id:
                port = 445
            else:
                port = None

            twin.firewall.add_rule(FirewallRule(
                rule_id=f"SHAPE-{mitigation_id.upper()}",
                source="*",
                destination=host_id,
                port=port,
                action="RATE_LIMIT",
                rate_limit_factor=0.25,
                description=f"Traffic shaping policy: throttle port {port} to 25% connection volume",
            ))

        elif mitigation_id == "quarantine_exfiltration":
            if target_host:
                target_host.set_service_status(443, "filtered")
            twin.firewall.add_rule(FirewallRule(
                rule_id="DLP-EXFIL-001",
                source="*",
                destination=host_id,
                port=443,
                action="DENY",
                description="DLP Egress Filter: terminate unauthorized exfiltration socket",
            ))
