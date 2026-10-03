"""Digital Twin logical network state representation.

Maintains a simulated/logical model of the network: hosts, services,
firewall rules, segment membership, and — critically — an **explicit
topology graph** (list of edges, NOT an implicit full mesh).

Every element carries a ``data_source`` tag ("configured" or "live")
so consumers can tell what is observed vs. assumed.

``real_network_touched`` is strictly ``False`` at all times.
"""
from __future__ import annotations

import copy
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

from app.simulation.state_provider import (
    TopologyConfig, TopologyEdge, default_provider, NetworkStateProvider,
)


# ------------------------------------------------------------------ #
# Host
# ------------------------------------------------------------------ #
class TwinHost:
    def __init__(self, host_id: str, ip_address: str,
                 segment_id: str = "default", data_source: str = "configured"):
        self.host_id = host_id
        self.ip_address = ip_address
        self.segment_id = segment_id
        self.data_source = data_source
        self.is_isolated: bool = False
        self.services: Dict[int, Dict[str, Any]] = {}

    def set_service_status(self, port: int, status: str):
        if port in self.services:
            self.services[port]["status"] = status

    def isolate(self):
        self.is_isolated = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "host_id": self.host_id,
            "ip_address": self.ip_address,
            "segment_id": self.segment_id,
            "is_isolated": self.is_isolated,
            "data_source": self.data_source,
            "services": list(self.services.values()),
        }


# ------------------------------------------------------------------ #
# Firewall
# ------------------------------------------------------------------ #
class FirewallRule:
    def __init__(self, rule_id: str, source: str, destination: str,
                 port: Optional[int] = None, action: str = "ALLOW",
                 rate_limit_factor: float = 0.25, description: str = "",
                 data_source: str = "configured"):
        self.rule_id = rule_id
        self.source = source
        self.destination = destination
        self.port = port
        self.action = action.upper()
        self.rate_limit_factor = rate_limit_factor
        self.description = description
        self.data_source = data_source

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id, "source": self.source,
            "destination": self.destination, "port": self.port,
            "action": self.action, "rate_limit_factor": self.rate_limit_factor,
            "description": self.description, "data_source": self.data_source,
        }


class FirewallState:
    def __init__(self):
        self.rules: List[FirewallRule] = []

    def add_rule(self, rule: FirewallRule):
        self.rules.insert(0, rule)           # newest rule takes priority

    def evaluate(self, source: str, destination: str,
                 port: Optional[int]) -> Tuple[str, Optional[FirewallRule]]:
        for rule in self.rules:
            src_ok = rule.source == "*" or rule.source == source
            dst_ok = rule.destination == "*" or rule.destination == destination
            port_ok = rule.port is None or rule.port == port
            if src_ok and dst_ok and port_ok:
                return rule.action, rule
        return "ALLOW", None

    def to_dict(self) -> List[Dict[str, Any]]:
        return [r.to_dict() for r in self.rules]


# ------------------------------------------------------------------ #
# Twin Network (graph-based)
# ------------------------------------------------------------------ #
class TwinNetwork:
    def __init__(self):
        self.hosts: Dict[str, TwinHost] = {}
        self.segments: Dict[str, Dict[str, Any]] = {}
        self.edges: List[TopologyEdge] = []     # explicit graph edges
        self.firewall = FirewallState()
        self.real_network_touched: bool = False
        self.is_simulated_twin: bool = True

    # ---- graph mutation ------------------------------------------------ #
    def add_host(self, host: TwinHost):
        self.hosts[host.host_id] = host

    def add_edge(self, edge: TopologyEdge):
        self.edges.append(edge)

    # ---- graph traversal (BFS) ---------------------------------------- #
    def find_path(self, source_id: str, dest_id: str,
                  port: Optional[int] = None) -> Optional[List[str]]:
        """BFS through explicit topology edges.  Returns a list of node IDs
        forming the path, or ``None`` if no path exists.  Respects
        ``allowed_ports`` on each edge and ``is_isolated`` on each host."""
        visited: set[str] = set()
        queue: deque[Tuple[str, List[str]]] = deque([(source_id, [source_id])])

        while queue:
            current, path = queue.popleft()
            if current == dest_id:
                return path
            if current in visited:
                continue
            visited.add(current)

            for edge in self.edges:
                neighbour: Optional[str] = None
                if edge.source_id == current:
                    neighbour = edge.dest_id
                elif edge.bidirectional and edge.dest_id == current:
                    neighbour = edge.source_id

                if neighbour is None or neighbour in visited:
                    continue
                # port allowed on this edge?
                if (port is not None and edge.allowed_ports is not None
                        and port not in edge.allowed_ports):
                    continue
                # destination host isolated?
                dest_host = self.hosts.get(neighbour)
                if dest_host and dest_host.is_isolated:
                    continue
                queue.append((neighbour, path + [neighbour]))

        return None

    def has_path(self, source_id: str, dest_id: str,
                 port: Optional[int] = None) -> bool:
        return self.find_path(source_id, dest_id, port) is not None

    # ---- clone --------------------------------------------------------- #
    def clone(self) -> "TwinNetwork":
        cloned = copy.deepcopy(self)
        cloned.real_network_touched = False
        return cloned

    # ---- serialisation ------------------------------------------------- #
    def to_dict(self) -> Dict[str, Any]:
        return {
            "real_network_touched": self.real_network_touched,
            "is_simulated_twin": self.is_simulated_twin,
            "hosts": {hid: h.to_dict() for hid, h in self.hosts.items()},
            "segments": self.segments,
            "edges": [e.to_dict() for e in self.edges],
            "firewall_rules": self.firewall.to_dict(),
        }


# ------------------------------------------------------------------ #
# Factory: build a TwinNetwork from a TopologyConfig
# ------------------------------------------------------------------ #
def build_twin_from_config(cfg: TopologyConfig) -> TwinNetwork:
    """Converts a ``TopologyConfig`` (from any state provider) into a
    live ``TwinNetwork`` suitable for sandbox simulation."""
    twin = TwinNetwork()

    for hid, hcfg in cfg.hosts.items():
        host = TwinHost(hid, hcfg.ip_address,
                        segment_id=hcfg.segment_id,
                        data_source=hcfg.data_source)
        for port, scfg in hcfg.services.items():
            host.services[port] = {
                "port": scfg.port, "name": scfg.name,
                "status": scfg.status, "data_source": scfg.data_source,
            }
        twin.add_host(host)

    for sid, seg in cfg.segments.items():
        twin.segments[sid] = {"name": seg.name, "data_source": seg.data_source}

    for edge in cfg.edges:
        twin.add_edge(edge)

    for rcfg in cfg.firewall_rules:
        twin.firewall.add_rule(FirewallRule(
            rule_id=rcfg.rule_id, source=rcfg.source,
            destination=rcfg.destination, port=rcfg.port,
            action=rcfg.action, rate_limit_factor=rcfg.rate_limit_factor,
            description=rcfg.description, data_source=rcfg.data_source,
        ))

    return twin


def build_initial_twin_network(target_host_id: str,
                                is_live: bool = False,
                                provider: Optional[NetworkStateProvider] = None,
                                ) -> TwinNetwork:
    """Convenience wrapper: uses the (optionally overridden) state provider
    to build the topology config, then materialises a ``TwinNetwork``."""
    prov = provider or default_provider
    cfg = prov.build_topology(target_host_id, is_live=is_live)
    return build_twin_from_config(cfg)
