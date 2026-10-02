"""Path prediction engine operating on Digital Twin network state and topology.

Calculates:
- Reachable hosts & services
- Available attack transition graph paths
- Blocked attack paths resulting from applied mitigations/firewall rules
- Remaining open attack paths in the twin topology
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

from app.config import PORT_OF_ACTION, HARD_ACTIONS, WATCHED_PORTS, ATTACK_STAGE_MAP
from app.simulation.network_twin import TwinNetwork


class PathPredictor:
    def __init__(self, twin: TwinNetwork):
        self.twin = twin

    def calculate_attack_paths(
        self,
        source_host_id: str,
        target_host_id: str,
        current_stage: str,
    ) -> Dict[str, Any]:
        """Calculates graph reachability and potential attack progression paths
        across the Digital Twin topology based on network & firewall state."""
        target_host = self.twin.hosts.get(target_host_id)
        if not target_host:
            return {
                "all_paths": [],
                "blocked_paths": [],
                "remaining_paths": [],
                "reachable_services": [],
                "summary": "Target host not found in network topology.",
            }

        target_ip = target_host.ip_address
        is_target_isolated = target_host.is_isolated

        # Evaluate reachability for watched services
        reachable_services = []
        blocked_services = []

        for port, service_info in target_host.services.items():
            service_name = service_info["name"]
            
            if is_target_isolated:
                action = "DENY"
                rule_desc = "Host network quarantine (isolated)"
            else:
                action, rule = self.twin.firewall.evaluate(source="*", destination=target_host_id, port=port)
                rule_desc = rule.description if rule else "Default topology access"

            service_entry = {
                "port": port,
                "name": service_name,
                "status": action,
                "rule": rule_desc,
            }

            if action == "ALLOW":
                reachable_services.append(service_entry)
            else:
                blocked_services.append(service_entry)

        # Build attack transition paths dynamically based on project's stage taxonomy
        all_paths: List[Dict[str, Any]] = []
        blocked_paths: List[Dict[str, Any]] = []
        remaining_paths: List[Dict[str, Any]] = []

        # Generic stages flow: Recon (port_scan) -> Credential Access (bruteforce) -> Lateral Movement -> C2 -> Exfil
        for stage in HARD_ACTIONS:
            if stage == "benign" or stage == "ambiguous_pre_attack":
                continue

            target_port = PORT_OF_ACTION.get(stage)
            mapping = ATTACK_STAGE_MAP.get(stage, {})
            
            # Check reachability for this stage
            if is_target_isolated:
                is_blocked = True
                block_reason = f"Target host '{target_host_id}' is isolated from network."
            elif target_port is not None:
                action, rule = self.twin.firewall.evaluate(source="*", destination=target_host_id, port=target_port)
                is_blocked = (action == "DENY")
                block_reason = rule.description if rule else f"Firewall rule blocks port {target_port}."
            elif stage in ["port_scan"]:
                # Recon scan
                action, rule = self.twin.firewall.evaluate(source="*", destination=target_host_id, port=None)
                is_blocked = (action == "DENY")
                block_reason = rule.description if rule else "Reconnaissance probes dropped by ACL."
            else:
                is_blocked = False
                block_reason = ""

            path_node = {
                "stage": stage,
                "tactic": mapping.get("tactic", "Unknown"),
                "technique_id": mapping.get("technique_id"),
                "technique_name": mapping.get("technique_name"),
                "target_port": target_port,
                "target_host": target_host_id,
                "is_blocked": is_blocked,
                "reason": block_reason if is_blocked else "Path clear in current network state",
            }

            all_paths.append(path_node)
            if is_blocked:
                blocked_paths.append(path_node)
            else:
                remaining_paths.append(path_node)

        total_paths = len(all_paths)
        num_blocked = len(blocked_paths)
        num_remaining = len(remaining_paths)

        if total_paths > 0:
            reduction_pct = round((num_blocked / total_paths) * 100.0, 1)
        else:
            reduction_pct = 0.0

        summary = f"{num_blocked} of {total_paths} potential attack paths blocked ({reduction_pct}% reduction). {num_remaining} paths remain open."

        return {
            "all_paths": all_paths,
            "blocked_paths": blocked_paths,
            "remaining_paths": remaining_paths,
            "reachable_services": reachable_services,
            "blocked_services": blocked_services,
            "metrics": {
                "total_paths": total_paths,
                "blocked_paths_count": num_blocked,
                "remaining_paths_count": num_remaining,
                "reduction_pct": reduction_pct,
            },
            "summary": summary,
        }
