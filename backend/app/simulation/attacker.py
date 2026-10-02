"""Generic simulated attacker interacting with the Digital Twin network state.

Operates dynamically on the TwinNetwork state using the project's existing
attack definitions, stage classes, watched ports, and mitigations.
Does NOT hardcode specific attacks or ports.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import numpy as np

from app.config import PORT_OF_ACTION, HARD_ACTIONS, WATCHED_PORTS
from app.simulation.network_twin import TwinNetwork
from app.simulation.mitigations import get_mitigation_fn


class AttackState:
    def __init__(
        self,
        current_stage: str,
        target_host_id: str,
        source_ip: Optional[str] = None,
        observed_features: Optional[np.ndarray] = None,
    ):
        self.current_stage = current_stage
        self.target_host_id = target_host_id
        self.target_port = PORT_OF_ACTION.get(current_stage)
        self.source_ip = source_ip
        self.observed_features = observed_features


class SimulatedAttackResponse:
    def __init__(
        self,
        outcome: str,  # "BLOCKED", "THROTTLED", "CONTINUED"
        reason: str,
        target_host_id: str,
        affected_port: Optional[int],
        simulated_features: Optional[np.ndarray] = None,
    ):
        self.outcome = outcome
        self.reason = reason
        self.target_host_id = target_host_id
        self.affected_port = affected_port
        self.simulated_features = simulated_features

    def to_dict(self) -> Dict[str, Any]:
        return {
            "outcome": self.outcome,
            "reason": self.reason,
            "target_host_id": self.target_host_id,
            "affected_port": self.affected_port,
        }


class GenericAttacker:
    """Interacts with the cloned Digital Twin network state to determine
    how an attack progression responds to state changes."""

    @staticmethod
    def step(twin: TwinNetwork, attack_state: AttackState, mitigation_id: Optional[str] = None) -> SimulatedAttackResponse:
        target_host = twin.hosts.get(attack_state.target_host_id)
        
        # 1. Check if host is completely isolated
        if target_host and target_host.is_isolated:
            simulated_feats = None
            if attack_state.observed_features is not None:
                iso_fn = get_mitigation_fn("isolate_host")
                simulated_feats = iso_fn(attack_state.observed_features)

            return SimulatedAttackResponse(
                outcome="BLOCKED",
                reason=f"Target host '{attack_state.target_host_id}' network isolation is ACTIVE. All external and lateral traffic severed.",
                target_host_id=attack_state.target_host_id,
                affected_port=attack_state.target_port,
                simulated_features=simulated_feats,
            )

        # 2. Check firewall evaluation for the attack's target port/service
        target_port = attack_state.target_port
        fw_action, fw_rule = twin.firewall.evaluate(
            source="*",
            destination=attack_state.target_host_id,
            port=target_port,
        )

        simulated_feats = attack_state.observed_features.copy() if attack_state.observed_features is not None else None

        if fw_action == "DENY":
            if mitigation_id and mitigation_id in ["block_ssh", "block_rdp", "block_smb", "block_scanner_ip", "block_c2_egress", "quarantine_exfiltration"]:
                mit_fn = get_mitigation_fn(mitigation_id)
                if simulated_feats is not None:
                    simulated_feats = mit_fn(simulated_feats)

            rule_desc = fw_rule.description if fw_rule else f"Traffic to port {target_port} denied by firewall policy."
            return SimulatedAttackResponse(
                outcome="BLOCKED",
                reason=f"Attack trajectory halted: {rule_desc}",
                target_host_id=attack_state.target_host_id,
                affected_port=target_port,
                simulated_features=simulated_feats,
            )

        elif fw_action == "RATE_LIMIT":
            if mitigation_id and mitigation_id in ["rate_limit_ssh", "rate_limit_rdp", "rate_limit_smb"]:
                mit_fn = get_mitigation_fn(mitigation_id)
                if simulated_feats is not None:
                    simulated_feats = mit_fn(simulated_feats)

            factor = fw_rule.rate_limit_factor if fw_rule else 0.25
            return SimulatedAttackResponse(
                outcome="THROTTLED",
                reason=f"Attack volume throttled to {int(factor * 100)}% by active traffic shaping on port {target_port}.",
                target_host_id=attack_state.target_host_id,
                affected_port=target_port,
                simulated_features=simulated_feats,
            )

        # 3. Default: Traffic allowed
        return SimulatedAttackResponse(
            outcome="CONTINUED",
            reason=f"Target port {target_port or 'all'} reachable in twin state. Attack progression continues unhindered.",
            target_host_id=attack_state.target_host_id,
            affected_port=target_port,
            simulated_features=simulated_feats,
        )
