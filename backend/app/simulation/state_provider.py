"""Configurable network state provider for the Digital Twin.

Defines the topology as an explicit directed graph with labeled data sources.
Can be subclassed or reconfigured to consume real network topology data
(e.g. from a CMDB, firewall API, or live discovery).

Every element carries a `data_source` field:
  - "configured" : static/default/logical state, not observed from a real network
  - "live"       : observed from live packet capture or real network state

The default provider creates a SEGMENTED topology with a gateway router —
NOT a full mesh and NOT neighbor-generation from host_id arithmetic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.config import WATCHED_PORTS, PORT_OF_ACTION


# --------------------------------------------------------------------------- #
# Port → service-name lookup (derived from project config, not hardcoded)
# --------------------------------------------------------------------------- #
_PORT_SERVICE_NAME: Dict[int, str] = {}
for _action, _port in PORT_OF_ACTION.items():
    if _port not in _PORT_SERVICE_NAME:
        _base = _action.replace("_bruteforce", "").replace("_lateral_movement", "")
        _PORT_SERVICE_NAME[_port] = _base.upper().replace("_", " ")
for _wp in WATCHED_PORTS:
    if _wp not in _PORT_SERVICE_NAME:
        _PORT_SERVICE_NAME[_wp] = f"Service:{_wp}"


# --------------------------------------------------------------------------- #
# Configuration dataclasses
# --------------------------------------------------------------------------- #
@dataclass
class ServiceConfig:
    port: int
    name: str
    status: str = "open"            # "open" | "filtered" | "closed"
    data_source: str = "configured"

    def to_dict(self) -> Dict[str, Any]:
        return {"port": self.port, "name": self.name, "status": self.status,
                "data_source": self.data_source}


@dataclass
class HostConfig:
    host_id: str
    ip_address: str
    segment_id: str = "default"
    services: Dict[int, ServiceConfig] = field(default_factory=dict)
    data_source: str = "configured"

    @classmethod
    def with_default_services(cls, host_id: str, ip_address: str,
                              segment_id: str = "default",
                              data_source: str = "configured") -> "HostConfig":
        """Creates a host with one service per WATCHED_PORT (all open)."""
        services = {
            port: ServiceConfig(
                port=port,
                name=_PORT_SERVICE_NAME.get(port, f"Service:{port}"),
                data_source=data_source,
            )
            for port in WATCHED_PORTS
        }
        return cls(host_id=host_id, ip_address=ip_address,
                   segment_id=segment_id, services=services,
                   data_source=data_source)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "host_id": self.host_id, "ip_address": self.ip_address,
            "segment_id": self.segment_id, "data_source": self.data_source,
            "services": [s.to_dict() for s in self.services.values()],
        }


@dataclass
class TopologyEdge:
    """An explicit, directional link in the network graph."""
    source_id: str
    dest_id: str
    allowed_ports: Optional[List[int]] = None   # None = all ports
    bidirectional: bool = True
    data_source: str = "configured"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id, "dest_id": self.dest_id,
            "allowed_ports": self.allowed_ports,
            "bidirectional": self.bidirectional,
            "data_source": self.data_source,
        }


@dataclass
class SegmentConfig:
    segment_id: str
    name: str
    data_source: str = "configured"

    def to_dict(self) -> Dict[str, Any]:
        return {"segment_id": self.segment_id, "name": self.name,
                "data_source": self.data_source}


@dataclass
class FirewallRuleConfig:
    rule_id: str
    source: str
    destination: str
    port: Optional[int] = None
    action: str = "ALLOW"           # ALLOW | DENY | RATE_LIMIT
    rate_limit_factor: float = 0.25
    description: str = ""
    data_source: str = "configured"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id, "source": self.source,
            "destination": self.destination, "port": self.port,
            "action": self.action, "rate_limit_factor": self.rate_limit_factor,
            "description": self.description, "data_source": self.data_source,
        }


@dataclass
class TopologyConfig:
    """Complete, serialisable specification of a network topology."""
    hosts: Dict[str, HostConfig] = field(default_factory=dict)
    segments: Dict[str, SegmentConfig] = field(default_factory=dict)
    edges: List[TopologyEdge] = field(default_factory=list)
    firewall_rules: List[FirewallRuleConfig] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hosts": {hid: h.to_dict() for hid, h in self.hosts.items()},
            "segments": {sid: s.to_dict() for sid, s in self.segments.items()},
            "edges": [e.to_dict() for e in self.edges],
            "firewall_rules": [r.to_dict() for r in self.firewall_rules],
        }


# --------------------------------------------------------------------------- #
# Provider
# --------------------------------------------------------------------------- #
class NetworkStateProvider:
    """Configurable provider for network topology state.

    Override ``build_topology()`` to supply real topology data from an
    external source (CMDB, firewall API, SDN controller, etc.).

    The default implementation creates a **segmented** topology with a
    gateway/router node, using the project's existing WATCHED_PORTS for
    service definitions.  There is NO implicit full mesh and NO
    synthetic neighbour generation from host_id arithmetic.
    """

    def build_topology(self, target_host_id: str,
                       is_live: bool = False) -> TopologyConfig:
        if is_live or target_host_id.startswith(("live:", "pcap:")):
            return self._build_live_topology(target_host_id)
        return self._build_demo_topology(target_host_id)

    # ------------------------------------------------------------------ #
    # Demo dataset topology
    # ------------------------------------------------------------------ #
    def _build_demo_topology(self, target_host_id: str) -> TopologyConfig:
        num_suffix = (target_host_id.replace("attack-host-", "")
                                     .replace("benign-host-", ""))
        idx = int(num_suffix) if num_suffix.isdigit() else 0
        target_ip = f"10.0.1.{10 + (idx % 200)}"

        segments = {
            "workstation": SegmentConfig("workstation", "Workstation Segment"),
            "dmz":         SegmentConfig("dmz", "DMZ / Gateway Segment"),
        }

        hosts: Dict[str, HostConfig] = {
            target_host_id: HostConfig.with_default_services(
                target_host_id, target_ip, segment_id="workstation"),
            "gateway-router": HostConfig.with_default_services(
                "gateway-router", "10.0.0.1", segment_id="dmz"),
        }

        # Explicit edges: target ↔ gateway (star topology, NOT full mesh)
        edges = [
            TopologyEdge(source_id=target_host_id, dest_id="gateway-router",
                         bidirectional=True),
            TopologyEdge(source_id="external_attacker", dest_id="gateway-router",
                         bidirectional=False),
        ]
        return TopologyConfig(hosts=hosts, segments=segments, edges=edges,
                              firewall_rules=[])

    # ------------------------------------------------------------------ #
    # Live-capture topology
    # ------------------------------------------------------------------ #
    def _build_live_topology(self, target_host_id: str) -> TopologyConfig:
        clean_id = target_host_id.replace("live:", "").replace("pcap:", "")
        target_ip = clean_id if "." in clean_id else "192.168.1.50"

        segments = {
            "local": SegmentConfig("local", "Local Network", data_source="live"),
        }
        hosts: Dict[str, HostConfig] = {
            target_host_id: HostConfig.with_default_services(
                target_host_id, target_ip, segment_id="local",
                data_source="live"),
            "live:gateway": HostConfig.with_default_services(
                "live:gateway", "192.168.1.1", segment_id="local",
                data_source="configured"),
        }
        edges = [
            TopologyEdge(source_id=target_host_id, dest_id="live:gateway",
                         bidirectional=True, data_source="configured"),
            TopologyEdge(source_id="external_attacker", dest_id="live:gateway",
                         bidirectional=False, data_source="configured"),
        ]
        return TopologyConfig(hosts=hosts, segments=segments, edges=edges,
                              firewall_rules=[])


# Singleton default provider
default_provider = NetworkStateProvider()
