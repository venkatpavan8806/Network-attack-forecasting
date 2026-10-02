"""Digital Twin logical network state representation.

Maintains a simulated/logical model of the network state (hosts, services, ports,
subnets, firewall rules, and isolation state).

This is explicitly a logical digital twin for safe sandbox simulation.
It NEVER touches a real physical network, sends packets, or modifies system rules.
`real_network_touched` is strictly set to `False`.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

from app.config import WATCHED_PORTS, PORT_OF_ACTION


class TwinHost:
    def __init__(
        self,
        host_id: str,
        ip_address: str,
        subnet: str = "192.168.1.0/24",
        is_live: bool = False,
    ):
        self.host_id = host_id
        self.ip_address = ip_address
        self.subnet = subnet
        self.is_live = is_live
        self.is_isolated: bool = False
        
        # Default logical services mapped from watched ports
        self.services: Dict[int, Dict[str, Any]] = {
            22: {"name": "SSH", "status": "open", "port": 22},
            445: {"name": "SMB", "status": "open", "port": 445},
            3389: {"name": "RDP", "status": "open", "port": 3389},
            443: {"name": "HTTPS/C2", "status": "open", "port": 443},
        }

    def set_service_status(self, port: int, status: str):
        if port in self.services:
            self.services[port]["status"] = status

    def isolate(self):
        self.is_isolated = True

    def unisolate(self):
        self.is_isolated = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "host_id": self.host_id,
            "ip_address": self.ip_address,
            "subnet": self.subnet,
            "is_live": self.is_live,
            "is_isolated": self.is_isolated,
            "services": list(self.services.values()),
        }


class FirewallRule:
    def __init__(
        self,
        rule_id: str,
        source: str,
        destination: str,
        port: Optional[int] = None,
        action: str = "ALLOW",  # ALLOW, DENY, RATE_LIMIT
        rate_limit_factor: float = 0.25,
        description: str = "",
    ):
        self.rule_id = rule_id
        self.source = source
        self.destination = destination
        self.port = port
        self.action = action.upper()
        self.rate_limit_factor = rate_limit_factor
        self.description = description

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "source": self.source,
            "destination": self.destination,
            "port": self.port,
            "action": self.action,
            "rate_limit_factor": self.rate_limit_factor,
            "description": self.description,
        }


class FirewallState:
    def __init__(self):
        self.rules: List[FirewallRule] = []

    def add_rule(self, rule: FirewallRule):
        # Prepend so newer/specific rules take evaluation priority
        self.rules.insert(0, rule)

    def evaluate(self, source: str, destination: str, port: Optional[int]) -> Tuple[str, Optional[FirewallRule]]:
        """Evaluates traffic against active firewall rules. Returns (action, matching_rule)."""
        for rule in self.rules:
            src_match = rule.source == "*" or rule.source == source
            dst_match = rule.destination == "*" or rule.destination == destination
            port_match = rule.port is None or rule.port == port

            if src_match and dst_match and port_match:
                return rule.action, rule
        return "ALLOW", None

    def to_dict(self) -> List[Dict[str, Any]]:
        return [r.to_dict() for r in self.rules]


class TwinNetwork:
    def __init__(self):
        self.hosts: Dict[str, TwinHost] = {}
        self.subnets: Dict[str, List[str]] = {"Workstation VLAN 10": [], "Server VLAN 20": []}
        self.firewall = FirewallState()
        self.real_network_touched: bool = False
        self.is_simulated_twin: bool = True

    def add_host(self, host: TwinHost):
        self.hosts[host.host_id] = host
        if "server" in host.host_id.lower():
            self.subnets["Server VLAN 20"].append(host.host_id)
        else:
            self.subnets["Workstation VLAN 10"].append(host.host_id)

    def clone(self) -> "TwinNetwork":
        """Creates an independent clone of the current twin state for safe sandbox testing."""
        cloned = copy.deepcopy(self)
        cloned.real_network_touched = False
        return cloned

    def to_dict(self) -> Dict[str, Any]:
        return {
            "real_network_touched": self.real_network_touched,
            "is_simulated_twin": self.is_simulated_twin,
            "hosts": {hid: h.to_dict() for hid, h in self.hosts.items()},
            "subnets": self.subnets,
            "firewall_rules": self.firewall.to_dict(),
        }


def build_initial_twin_network(target_host_id: str, is_live: bool = False) -> TwinNetwork:
    """Dynamically builds a logical digital twin network around the target host."""
    twin = TwinNetwork()

    # Determine IP convention
    if is_live or target_host_id.startswith("live:"):
        clean_id = target_host_id.replace("live:", "")
        target_ip = clean_id if "." in clean_id else "192.168.1.50"
        host = TwinHost(host_id=target_host_id, ip_address=target_ip, is_live=True)
        twin.add_host(host)

        # Add neighbor logical hosts in twin network
        twin.add_host(TwinHost(host_id="live:gateway", ip_address="192.168.1.1", is_live=True))
        twin.add_host(TwinHost(host_id="live:internal-server", ip_address="192.168.1.100", is_live=True))
    else:
        # Demo dataset hosts
        num_suffix = target_host_id.replace("attack-host-", "").replace("benign-host-", "")
        idx = int(num_suffix) if num_suffix.isdigit() else 0
        target_ip = f"10.0.1.{10 + (idx % 200)}"

        target_host = TwinHost(host_id=target_host_id, ip_address=target_ip, is_live=False)
        twin.add_host(target_host)

        # Build adjacent logical nodes dynamically
        neighbor_1_id = f"attack-host-{(idx + 1) % 10:03d}"
        neighbor_2_id = f"attack-host-{(idx + 2) % 10:03d}"
        if neighbor_1_id != target_host_id:
            twin.add_host(TwinHost(host_id=neighbor_1_id, ip_address=f"10.0.1.{(10 + ((idx + 1) % 200))}", is_live=False))
        if neighbor_2_id != target_host_id:
            twin.add_host(TwinHost(host_id=neighbor_2_id, ip_address=f"10.0.1.{(10 + ((idx + 2) % 200))}", is_live=False))

    return twin
