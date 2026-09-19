"""Named mitigations for the "digital twin" what-if simulator.

This is NOT a live simulated network (no Mininet/GNS3, no live packet
injection). Instead, each mitigation is a physically grounded, documented
transformation applied directly to the RAW (unscaled) feature vector, expressing
how that specific intervention changes what a NetFlow/telemetry sensor observes:
- Isolating a host cuts external/lateral connectivity, dropping port indicators
  and returning observed traffic to a clean, quarantined baseline.
- Blocking a port (SSH 22, RDP 3389, SMB 445) zeros out that port's traffic
  indicator and eliminates the malicious flow volume associated with that vector.
- Rate-limiting throttles connection volume to 25%, simulating traffic shaping.
- Blocking scanner IPs eliminates reconnaissance port-scan scores and probe bursts.
- Blocking C2 egress severs outbound command-and-control beaconing on port 443.
- Blocking exfiltration stops large outbound byte surges.

This never touches a real network -- it mutates the feature vector fed
back into the already-trained world model for counterfactual rollout. See
app/simulation/counterfactual.py.
"""
from __future__ import annotations

import numpy as np

from app.config import FEATURE_COLUMNS

_IDX = {name: i for i, name in enumerate(FEATURE_COLUMNS)}

# Canonical benign baseline feature values (derived from the empirical benign distribution)
BENIGN_PROFILE: dict[str, float] = {
    "flow_count": 25.96,
    "unique_dst_ports": 3.99,
    "unique_dst_ips": 5.01,
    "syn_count": 27.97,
    "ack_count": 26.98,
    "fin_count": 21.77,
    "rst_count": 1.00,
    "psh_count": 10.42,
    "avg_flow_duration": 2.50,
    "std_flow_duration": 0.60,
    "avg_bytes_per_flow": 2195.27,
    "std_bytes_per_flow": 798.87,
    "avg_pkts_per_flow": 12.05,
    "failed_conn_ratio": 0.049,
    "inbound_bytes": 17135.75,
    "outbound_bytes": 13850.92,
    "bytes_ratio_out_in": 1.00,
    "new_dst_ip_ratio": 0.092,
    "iat_mean": 0.15,
    "iat_std": 0.05,
    "ttl_mean": 64.05,
    "ttl_std": 1.01,
    "win_size_mean": 29185.18,
    "win_size_std": 1500.65,
    "pkt_len_mean": 497.15,
    "pkt_len_std": 119.79,
    "port_scan_score": 0.034,
    "dst_port_is_22": 0.04,
    "dst_port_is_445": 0.04,
    "dst_port_is_3389": 0.04,
    "dst_port_is_443": 0.35,
}

_BENIGN_VEC = np.array([BENIGN_PROFILE.get(col, 0.0) for col in FEATURE_COLUMNS], dtype=np.float32)

_COUNT_FEATURES = [
    "flow_count", "syn_count", "ack_count", "fin_count", "rst_count", "psh_count",
    "inbound_bytes", "outbound_bytes",
]


def _identity(row: np.ndarray) -> np.ndarray:
    return row.copy()


def _block_port(row: np.ndarray, port: int) -> np.ndarray:
    """Simulate blocking all traffic to a specific destination port.
    Zeros out dst_port_is_{port} and suppresses attack traffic associated
    with that service, blending affected features back toward benign baseline."""
    row = row.copy()
    flag_name = f"dst_port_is_{port}"
    if flag_name in _IDX:
        row[_IDX[flag_name]] = 0.0

    # If the targeted port was active or volume was high, normalize volume and ratios
    for f in _COUNT_FEATURES:
        row[_IDX[f]] = 0.20 * row[_IDX[f]] + 0.80 * _BENIGN_VEC[_IDX[f]]

    if "failed_conn_ratio" in _IDX:
        row[_IDX["failed_conn_ratio"]] = min(row[_IDX["failed_conn_ratio"]], 0.05)
    if "unique_dst_ports" in _IDX:
        row[_IDX["unique_dst_ports"]] = max(1.0, row[_IDX["unique_dst_ports"]] - 1.0)
    return row


def _rate_limit_port(row: np.ndarray, port: int, factor: float = 0.25) -> np.ndarray:
    """Caps connection volume to a specific port to factor (e.g. 25%),
    simulating active traffic shaping and connection throttling."""
    row = row.copy()
    flag_name = f"dst_port_is_{port}"
    if flag_name in _IDX and row[_IDX[flag_name]] >= 0.3:
        for f in _COUNT_FEATURES:
            row[_IDX[f]] = row[_IDX[f]] * factor
        if "failed_conn_ratio" in _IDX:
            row[_IDX["failed_conn_ratio"]] = min(row[_IDX["failed_conn_ratio"]], 0.15)
    return row


def _isolate(row: np.ndarray) -> np.ndarray:
    """Simulate complete host network isolation / quarantine.
    All external and lateral communication is severed. Traffic collapses to
    a clean, quiescent benign baseline with zero active attack indicators."""
    row = _BENIGN_VEC.copy()
    for port in [22, 445, 3389, 443]:
        flag = f"dst_port_is_{port}"
        if flag in _IDX:
            row[_IDX[flag]] = 0.0
    row[_IDX["port_scan_score"]] = 0.0
    row[_IDX["failed_conn_ratio"]] = 0.0
    row[_IDX["new_dst_ip_ratio"]] = 0.0
    row[_IDX["unique_dst_ips"]] = 1.0
    row[_IDX["unique_dst_ports"]] = 1.0
    row[_IDX["flow_count"]] = max(2.0, _BENIGN_VEC[_IDX["flow_count"]] * 0.15)
    row[_IDX["syn_count"]] = max(1.0, _BENIGN_VEC[_IDX["syn_count"]] * 0.15)
    row[_IDX["ack_count"]] = max(1.0, _BENIGN_VEC[_IDX["ack_count"]] * 0.15)
    row[_IDX["rst_count"]] = 0.0
    return row


def _block_scanner(row: np.ndarray) -> np.ndarray:
    """Drops reconnaissance scanning probes from the attacking IP."""
    row = row.copy()
    if "port_scan_score" in _IDX:
        row[_IDX["port_scan_score"]] = 0.0
    if "failed_conn_ratio" in _IDX:
        row[_IDX["failed_conn_ratio"]] = 0.03
    if "new_dst_ip_ratio" in _IDX:
        row[_IDX["new_dst_ip_ratio"]] = 0.05
    if "unique_dst_ports" in _IDX:
        row[_IDX["unique_dst_ports"]] = min(row[_IDX["unique_dst_ports"]], 4.0)
    for f in ["flow_count", "syn_count", "rst_count"]:
        if f in _IDX:
            row[_IDX[f]] = 0.20 * row[_IDX[f]] + 0.80 * _BENIGN_VEC[_IDX[f]]
    return row


def _block_c2(row: np.ndarray) -> np.ndarray:
    """Blocks outbound Command & Control beaconing on port 443."""
    row = row.copy()
    for col in FEATURE_COLUMNS:
        row[_IDX[col]] = 0.15 * row[_IDX[col]] + 0.85 * _BENIGN_VEC[_IDX[col]]
    if "dst_port_is_443" in _IDX:
        row[_IDX["dst_port_is_443"]] = 0.0
    if "iat_mean" in _IDX:
        row[_IDX["iat_mean"]] = _BENIGN_VEC[_IDX["iat_mean"]]
    if "iat_std" in _IDX:
        row[_IDX["iat_std"]] = _BENIGN_VEC[_IDX["iat_std"]]
    return row


def _quarantine_exfiltration(row: np.ndarray) -> np.ndarray:
    """Severes unauthorized outbound data staging and exfiltration channels."""
    row = row.copy()
    for col in FEATURE_COLUMNS:
        row[_IDX[col]] = 0.15 * row[_IDX[col]] + 0.85 * _BENIGN_VEC[_IDX[col]]
    if "dst_port_is_443" in _IDX:
        row[_IDX["dst_port_is_443"]] = 0.0
    if "bytes_ratio_out_in" in _IDX:
        row[_IDX["bytes_ratio_out_in"]] = 1.0
    return row


MITIGATIONS = {
    "no_mitigation": {
        "label": "No mitigation (baseline)",
        "category": "Baseline",
        "target_stage": "N/A",
        "description": "The forecast rollout as-is, with no intervention applied. Shown for comparison.",
        "fn": _identity,
    },
    "isolate_host": {
        "label": "Isolate host (network quarantine)",
        "category": "Quarantine",
        "target_stage": "All attack stages",
        "description": "Cuts ALL external and lateral network connectivity. Quarantines the host to halt credential access, lateral movement, C2 beaconing, and exfiltration.",
        "fn": _isolate,
    },
    "block_ssh": {
        "label": "Block SSH (port 22)",
        "category": "Access Control",
        "target_stage": "ssh_bruteforce / lateral movement",
        "description": "Drops all inbound and lateral SSH traffic (port 22), terminating brute-force authentication and lateral traversal.",
        "fn": lambda row: _block_port(row, 22),
    },
    "rate_limit_ssh": {
        "label": "Rate-limit SSH (port 22)",
        "category": "Traffic Shaping",
        "target_stage": "ssh_bruteforce",
        "description": "Caps connection volume to port 22 to ~25% of observed traffic, slowing down brute-force enumeration.",
        "fn": lambda row: _rate_limit_port(row, 22, 0.25),
    },
    "block_rdp": {
        "label": "Block RDP (port 3389)",
        "category": "Access Control",
        "target_stage": "rdp_bruteforce / lateral movement",
        "description": "Drops all RDP traffic (port 3389), severing remote desktop credential attacks and lateral traversal.",
        "fn": lambda row: _block_port(row, 3389),
    },
    "rate_limit_rdp": {
        "label": "Rate-limit RDP (port 3389)",
        "category": "Traffic Shaping",
        "target_stage": "rdp_bruteforce",
        "description": "Caps connection volume to port 3389 to ~25% of observed traffic.",
        "fn": lambda row: _rate_limit_port(row, 3389, 0.25),
    },
    "block_smb": {
        "label": "Block SMB (port 445)",
        "category": "Access Control",
        "target_stage": "smb_bruteforce / lateral movement",
        "description": "Drops SMB / admin share traffic (port 445), halting password spraying, PsExec traversal, and SMB exploits.",
        "fn": lambda row: _block_port(row, 445),
    },
    "rate_limit_smb": {
        "label": "Rate-limit SMB (port 445)",
        "category": "Traffic Shaping",
        "target_stage": "smb_bruteforce",
        "description": "Caps connection volume to port 445 to ~25% of observed traffic.",
        "fn": lambda row: _rate_limit_port(row, 445, 0.25),
    },
    "block_scanner_ip": {
        "label": "Block Scanner IP (drop probes)",
        "category": "Reconnaissance Defense",
        "target_stage": "port_scan / precursor",
        "description": "Applies ACL to drop reconnaissance probes and SYN sweeps from suspected scanning IP addresses.",
        "fn": _block_scanner,
    },
    "block_c2_egress": {
        "label": "Block C2 Egress (revoke beacon)",
        "category": "C2 Defense",
        "target_stage": "c2_beacon",
        "description": "Sinks outbound C2 domain/IP traffic on port 443, severing attacker command and control callbacks.",
        "fn": _block_c2,
    },
    "quarantine_exfiltration": {
        "label": "Block Exfiltration Channel",
        "category": "Exfiltration Defense",
        "target_stage": "data_exfiltration",
        "description": "Terminates high-volume outbound staging sockets, halting data exfiltration bursts.",
        "fn": _quarantine_exfiltration,
    },
}


def list_mitigations() -> list[dict]:
    return [
        {
            "id": k,
            "label": v["label"],
            "category": v.get("category", "General"),
            "target_stage": v.get("target_stage", "All"),
            "description": v["description"],
        }
        for k, v in MITIGATIONS.items()
    ]


def get_mitigation_fn(mitigation_id: str):
    if mitigation_id not in MITIGATIONS:
        raise ValueError(f"unknown mitigation: {mitigation_id}")
    return MITIGATIONS[mitigation_id]["fn"]

