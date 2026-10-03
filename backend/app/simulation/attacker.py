"""Generic simulated attacker interacting with the Digital Twin network state.

Evaluates the cloned twin state (isolation, firewall rules, service status)
to determine the attacker's outcome: BLOCKED, THROTTLED, or CONTINUED.

This module does NOT generate simulated feature vectors — that is the job
of ``telemetry_synthesizer.py``, which reads the twin state and produces
what a telemetry sensor would observe.  This separation ensures there is
no double-application of mitigation transforms.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from app.config import PORT_OF_ACTION
from app.simulation.network_twin import TwinNetwork


class AttackState:
    def __init__(
        self,
        current_stage: str,
        target_host_id: str,
        source_id: str = "external_attacker",
    ):
        self.current_stage = current_stage
        self.target_host_id = target_host_id
        self.source_id = source_id
        # Dynamic port lookup from the project's existing config
        self.target_port: Optional[int] = PORT_OF_ACTION.get(current_stage)


class SimulatedAttackResponse:
    def __init__(
        self,
        outcome: str,          # "BLOCKED" | "THROTTLED" | "CONTINUED"
        reason: str,
        target_host_id: str,
        affected_port: Optional[int],
    ):
        self.outcome = outcome
        self.reason = reason
        self.target_host_id = target_host_id
        self.affected_port = affected_port

    def to_dict(self) -> Dict[str, Any]:
        return {
            "outcome": self.outcome,
            "reason": self.reason,
            "target_host_id": self.target_host_id,
            "affected_port": self.affected_port,
        }


class GenericAttacker:
    """Evaluates the Digital Twin state to determine the attack outcome.

    Does NOT apply ``mitigation_fn`` or produce ``simulated_features``.
    Feature synthesis is handled separately by ``TelemetrySynthesizer``.
    """

    @staticmethod
    def step(twin: TwinNetwork, attack_state: AttackState) -> SimulatedAttackResponse:
        target_host = twin.hosts.get(attack_state.target_host_id)

        # 1. Host isolation check
        if target_host and target_host.is_isolated:
            return SimulatedAttackResponse(
                outcome="BLOCKED",
                reason=(f"Target host '{attack_state.target_host_id}' network "
                        f"isolation is ACTIVE. All external and lateral traffic severed."),
                target_host_id=attack_state.target_host_id,
                affected_port=attack_state.target_port,
            )

        # 2. Topology reachability check (BFS through explicit edges)
        if not twin.has_path(attack_state.source_id,
                             attack_state.target_host_id,
                             port=attack_state.target_port):
            return SimulatedAttackResponse(
                outcome="BLOCKED",
                reason=(f"No network path from '{attack_state.source_id}' to "
                        f"'{attack_state.target_host_id}' on port "
                        f"{attack_state.target_port} in topology graph."),
                target_host_id=attack_state.target_host_id,
                affected_port=attack_state.target_port,
            )

        # 3. Service status check
        target_port = attack_state.target_port
        if target_host and target_port is not None:
            svc = target_host.services.get(target_port)
            if svc and svc.get("status") == "filtered":
                return SimulatedAttackResponse(
                    outcome="BLOCKED",
                    reason=(f"Service on port {target_port} is filtered on "
                            f"host '{attack_state.target_host_id}'."),
                    target_host_id=attack_state.target_host_id,
                    affected_port=target_port,
                )

        # 4. Firewall evaluation
        fw_action, fw_rule = twin.firewall.evaluate(
            source="*", destination=attack_state.target_host_id,
            port=target_port,
        )

        if fw_action == "DENY":
            desc = fw_rule.description if fw_rule else (
                f"Traffic to port {target_port} denied by firewall policy.")
            return SimulatedAttackResponse(
                outcome="BLOCKED",
                reason=f"Attack trajectory halted: {desc}",
                target_host_id=attack_state.target_host_id,
                affected_port=target_port,
            )

        if fw_action == "RATE_LIMIT":
            factor = fw_rule.rate_limit_factor if fw_rule else 0.25
            return SimulatedAttackResponse(
                outcome="THROTTLED",
                reason=(f"Attack volume throttled to {int(factor * 100)}% by "
                        f"active traffic shaping on port {target_port}."),
                target_host_id=attack_state.target_host_id,
                affected_port=target_port,
            )

        # 5. Default: traffic allowed
        return SimulatedAttackResponse(
            outcome="CONTINUED",
            reason=(f"Target port {target_port or 'all'} reachable in twin "
                    f"state. Attack progression continues unhindered."),
            target_host_id=attack_state.target_host_id,
            affected_port=target_port,
        )
