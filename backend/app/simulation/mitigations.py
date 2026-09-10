"""Named mitigations for the "digital twin" what-if simulator.

This is NOT a live simulated network (no Mininet/GNS3, no packet
injection) -- that remains out of scope for this build. Instead, each
mitigation is a small, documented transformation applied directly to the
RAW (unscaled) feature vector, expressing how that specific intervention
would plausibly change what a NetFlow-style sensor actually observes:
rate-limiting a service cuts the volume of connection attempts that get
through; blocking a port cuts it almost entirely; isolating a host cuts all
observed flow volume regardless of destination. Only the "count/total"-type
features are scaled (flow_count, TCP flag counts, byte totals) -- ratio-type
features (failed_conn_ratio, bytes_ratio_out_in, new_dst_ip_ratio,
port_scan_score) and the port-destination indicators are left alone, since a
rate limit does not change what FRACTION of attempts fail or which service
was being targeted, only how much volume gets through.

This never touches a real network -- it only mutates the feature vector fed
back into the already-trained world model for a counterfactual rollout. See
app/simulation/counterfactual.py.
"""
from __future__ import annotations

import numpy as np

from app.config import FEATURE_COLUMNS

_IDX = {name: i for i, name in enumerate(FEATURE_COLUMNS)}

# features whose value is a count or a byte total, and therefore scales down
# when fewer connection attempts get through
_COUNT_FEATURES = [
    "flow_count", "syn_count", "ack_count", "fin_count", "rst_count", "psh_count",
    "inbound_bytes", "outbound_bytes",
]


def _scale_if_port_matches(row: np.ndarray, port: int, factor: float) -> np.ndarray:
    row = row.copy()
    flag_name = f"dst_port_is_{port}"
    if flag_name not in _IDX:
        return row
    if row[_IDX[flag_name]] >= 0.5:
        for f in _COUNT_FEATURES:
            row[_IDX[f]] = row[_IDX[f]] * factor
    return row


def _isolate(row: np.ndarray) -> np.ndarray:
    row = row.copy()
    for f in _COUNT_FEATURES:
        row[_IDX[f]] = row[_IDX[f]] * 0.05
    return row


def _identity(row: np.ndarray) -> np.ndarray:
    return row.copy()


MITIGATIONS = {
    "no_mitigation": {
        "label": "No mitigation (baseline)",
        "description": "The forecast rollout as-is, with no intervention applied. Shown for comparison.",
        "fn": _identity,
    },
    "rate_limit_ssh": {
        "label": "Rate-limit SSH (port 22)",
        "description": "Caps connection volume to port 22 to ~25% of what was observed. Only changes windows where SSH traffic was actually present.",
        "fn": lambda row: _scale_if_port_matches(row, 22, 0.25),
    },
    "rate_limit_rdp": {
        "label": "Rate-limit RDP (port 3389)",
        "description": "Caps connection volume to port 3389 to ~25% of what was observed. Only changes windows where RDP traffic was actually present.",
        "fn": lambda row: _scale_if_port_matches(row, 3389, 0.25),
    },
    "rate_limit_smb": {
        "label": "Rate-limit SMB (port 445)",
        "description": "Caps connection volume to port 445 to ~25% of what was observed. Only changes windows where SMB traffic was actually present.",
        "fn": lambda row: _scale_if_port_matches(row, 445, 0.25),
    },
    "block_ssh": {
        "label": "Block SSH (port 22)",
        "description": "Cuts connection volume to port 22 to ~3% (a few retries before giving up). Only changes windows where SSH traffic was actually present.",
        "fn": lambda row: _scale_if_port_matches(row, 22, 0.03),
    },
    "block_rdp": {
        "label": "Block RDP (port 3389)",
        "description": "Cuts connection volume to port 3389 to ~3%. Only changes windows where RDP traffic was actually present.",
        "fn": lambda row: _scale_if_port_matches(row, 3389, 0.03),
    },
    "block_smb": {
        "label": "Block SMB (port 445)",
        "description": "Cuts connection volume to port 445 to ~3%. Only changes windows where SMB traffic was actually present.",
        "fn": lambda row: _scale_if_port_matches(row, 445, 0.03),
    },
    "isolate_host": {
        "label": "Isolate host (network quarantine)",
        "description": "Cuts ALL observed flow volume to ~5%, regardless of destination port -- simulates full network isolation.",
        "fn": _isolate,
    },
}


def list_mitigations() -> list[dict]:
    return [{"id": k, "label": v["label"], "description": v["description"]} for k, v in MITIGATIONS.items()]


def get_mitigation_fn(mitigation_id: str):
    if mitigation_id not in MITIGATIONS:
        raise ValueError(f"unknown mitigation: {mitigation_id}")
    return MITIGATIONS[mitigation_id]["fn"]
