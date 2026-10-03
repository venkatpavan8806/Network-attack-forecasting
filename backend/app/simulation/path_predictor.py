"""Path prediction engine operating on the Digital Twin's explicit topology graph.

Calculates reachable hosts and attack-transition paths by traversing the
actual configured/observed topology edges — NOT an implicit full mesh.

For each attack stage defined in the project's config, checks:
  1. Does a **topology path** exist (BFS through edges)?
  2. Is the edge's ``allowed_ports`` compatible?
  3. Is the target service filtered at the host level?
  4. Does a firewall DENY rule block the traffic?
  5. Is the target host isolated?

Also evaluates **lateral movement**: for stages involving lateral movement,
checks whether alternative hosts are reachable via the topology graph.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from app.config import PORT_OF_ACTION, HARD_ACTIONS, ATTACK_STAGE_MAP
from app.simulation.network_twin import TwinNetwork

_STAGE_ORDER = {stage: i for i, stage in enumerate(HARD_ACTIONS) if stage != "benign"}


class PathPredictor:
    def __init__(self, twin: TwinNetwork):
        self.twin = twin

    # ------------------------------------------------------------------ #
    # Core reachability check (topology + host + firewall)
    # ------------------------------------------------------------------ #
    def _is_port_reachable(self, source_id: str, target_host_id: str,
                           port: Optional[int]) -> Tuple[bool, str]:
        """Check reachability by traversing the explicit topology graph,
        then checking host service state and firewall rules."""

        target_host = self.twin.hosts.get(target_host_id)
        if not target_host:
            return False, f"Host '{target_host_id}' not found in topology."

        if target_host.is_isolated:
            return False, f"Host '{target_host_id}' is isolated from network."

        # 1. Topology graph path (BFS through explicit edges)
        if not self.twin.has_path(source_id, target_host_id, port=port):
            return False, (f"No topology path from '{source_id}' to "
                           f"'{target_host_id}'" +
                           (f" on port {port}." if port else "."))

        # 2. Host-level service status
        if port is not None:
            svc = target_host.services.get(port)
            if svc and svc.get("status") == "filtered":
                return False, (f"Service on port {port} is filtered on "
                               f"host '{target_host_id}'.")

        # 3. Firewall rules
        fw_action, fw_rule = self.twin.firewall.evaluate(
            source="*", destination=target_host_id, port=port)
        if fw_action == "DENY":
            reason = fw_rule.description if fw_rule else f"Firewall blocks port {port}."
            return False, reason
        if fw_action == "RATE_LIMIT":
            reason = fw_rule.description if fw_rule else f"Port {port} rate-limited."
            return True, f"Partially reachable (rate-limited): {reason}"

        return True, "Path clear in current network state."

    # ------------------------------------------------------------------ #
    # Main entry point
    # ------------------------------------------------------------------ #
    def calculate_attack_paths(
        self,
        source_host_id: str,
        target_host_id: str,
        current_stage: str,
    ) -> Dict[str, Any]:
        """Calculates graph reachability and potential attack paths across
        the Digital Twin topology."""

        target_host = self.twin.hosts.get(target_host_id)
        if not target_host:
            return {
                "all_paths": [], "blocked_paths": [], "remaining_paths": [],
                "metrics": {"total_paths": 0, "blocked_paths_count": 0,
                            "remaining_paths_count": 0, "reduction_pct": 0.0},
                "summary": "Target host not found in network topology.",
            }

        current_order = _STAGE_ORDER.get(current_stage, 0)

        all_paths: List[Dict[str, Any]] = []
        blocked_paths: List[Dict[str, Any]] = []
        remaining_paths: List[Dict[str, Any]] = []

        for stage in HARD_ACTIONS:
            if stage in ("benign", "ambiguous_pre_attack"):
                continue

            target_port = PORT_OF_ACTION.get(stage)
            mapping = ATTACK_STAGE_MAP.get(stage, {})
            stage_order = _STAGE_ORDER.get(stage, 0)

            # Check primary path: source → target
            reachable, reason = self._is_port_reachable(
                source_host_id, target_host_id, target_port)
            is_blocked = not reachable

            # For lateral-movement stages, check alternative hosts
            lateral_hosts: List[str] = []
            if is_blocked and "lateral_movement" in stage:
                for other_id in self.twin.hosts:
                    if other_id in (target_host_id, source_host_id, "gateway-router"):
                        continue
                    alt_ok, _ = self._is_port_reachable(
                        source_host_id, other_id, target_port)
                    if alt_ok:
                        lateral_hosts.append(other_id)

            final_blocked = is_blocked and not lateral_hosts

            path_node = {
                "stage": stage,
                "tactic": mapping.get("tactic", "Unknown"),
                "technique_id": mapping.get("technique_id"),
                "technique_name": mapping.get("technique_name"),
                "target_port": target_port,
                "target_host": target_host_id,
                "is_blocked": final_blocked,
                "reason": (reason if final_blocked else
                           (f"Lateral path available via {lateral_hosts}"
                            if lateral_hosts else
                            "Path clear in current network state.")),
                "is_future_stage": stage_order > current_order,
                "lateral_alternatives": lateral_hosts,
            }

            all_paths.append(path_node)
            if final_blocked:
                blocked_paths.append(path_node)
            else:
                remaining_paths.append(path_node)

        total = len(all_paths)
        n_blocked = len(blocked_paths)
        n_remaining = len(remaining_paths)
        reduction = round((n_blocked / total) * 100.0, 1) if total > 0 else 0.0

        return {
            "all_paths": all_paths,
            "blocked_paths": blocked_paths,
            "remaining_paths": remaining_paths,
            "metrics": {
                "total_paths": total,
                "blocked_paths_count": n_blocked,
                "remaining_paths_count": n_remaining,
                "reduction_pct": reduction,
            },
            "summary": (f"{n_blocked} of {total} potential attack paths blocked "
                        f"({reduction}% reduction). {n_remaining} remain open."),
        }
