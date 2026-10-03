"""Synthetic-but-honestly-structured flow+packet telemetry generator.

Produces a labeled attack-progression timeline per host:
  Port Scan -> {SSH/RDP/SMB} Brute-Force -> {SSH/RDP/SMB} Lateral Movement
  -> C2 Beaconing -> Data Exfiltration
plus a realistic volume of pure benign background traffic, and one
slow/evasive recon sequence deliberately built to stay under naive
flow-count thresholds (low per-window volume, jittered timing spread across
many windows) so packet-level features (IAT jitter, TTL variance, cumulative
port-scan score) are actually needed to catch it, not just flow counts.

Each attack host is randomly assigned WHICH specific service it brute-forces
and WHICH specific service it later uses for lateral movement (independently
-- an attacker who brute-forces SSH may still pivot laterally via SMB using
stolen credentials). During the later portion of its reconnaissance phase,
the host's port-scan traffic deliberately narrows onto the port it is about
to brute-force -- this is the concrete, learnable version of "port scan
reveals SSH open -> SSH brute-force follows" that the world model is meant
to pick up on.

IMPORTANT: the per-window feature vector this module emits never contains
the ground-truth action. The action is returned as a separate label column
(`true_stage`) so downstream code cannot accidentally train on it. The four
`dst_port_is_*` indicator features ARE included -- they represent which
service a window's traffic was actually headed to, which is legitimate,
directly observable NetFlow-style telemetry (real destination port), not an
encoding of the action label itself.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.config import (
    FEATURE_COLUMNS, WINDOW_SECONDS, RANDOM_SEED, PORT_OF_ACTION, WATCHED_PORTS,
)

BRUTEFORCE_ACTIONS = ["ssh_bruteforce", "rdp_bruteforce", "smb_bruteforce"]
LATERAL_ACTIONS = ["ssh_lateral_movement", "rdp_lateral_movement", "smb_lateral_movement"]

# broad set of ports a port scan sweeps across (superset of the 4 "watched" ones)
SCAN_PORT_POOL = [21, 22, 23, 25, 80, 110, 139, 143, 443, 445, 3306, 3389, 5900, 8080]


def _clip_pos(x: float, lo: float = 0.0) -> float:
    return float(max(lo, x))


def _port_indicators(rng: np.random.Generator, syn_count: float, failed_conn_ratio: float,
                      focus_port: int | None = None, focus_prob: float = 0.0,
                      background_prob: float = 0.05) -> dict:
    """Returns, for each of the 4 watched ports: the existing dst_port_is_*
    binary indicator, PLUS syn_count_port_X and failed_conn_ratio_port_X --
    how much of this window's connection volume and failures actually
    concentrated on that specific port. A binary flag can't distinguish
    "SSH briefly touched during a broad scan" from "SSH is being hammered by
    repeated failed logins"; these two features can, which is what actually
    lets the model tell SSH/RDP/SMB brute-force apart instead of only
    knowing "some brute-force is happening". `focus_port` (if one of
    WATCHED_PORTS) gets flagged with probability `focus_prob` and, if
    flagged, absorbs most of the window's SYN volume and (being the port
    actually under attack) the window's own failure rate; every other
    watched port gets flagged independently with `background_prob` and, if
    flagged, gets a small, independently-noisy share -- ordinary background
    traffic touching a port is not evidence that port is being attacked."""
    flags = {p: rng.random() < (focus_prob if p == focus_port else background_prob) for p in WATCHED_PORTS}
    out = {f"dst_port_is_{p}": (1.0 if flags[p] else 0.0) for p in WATCHED_PORTS}

    flagged = [p for p, on in flags.items() if on]
    if not flagged:
        for p in WATCHED_PORTS:
            out[f"syn_count_port_{p}"] = 0.0
            out[f"failed_conn_ratio_port_{p}"] = 0.0
        return out

    if focus_port in flagged:
        n_background_flagged = max(1, len(flagged) - 1)
        weights = {p: (0.75 if p == focus_port else 0.25 / n_background_flagged) for p in flagged}
    else:
        weights = {p: 1.0 / len(flagged) for p in flagged}

    for p in WATCHED_PORTS:
        if p not in flagged:
            out[f"syn_count_port_{p}"] = 0.0
            out[f"failed_conn_ratio_port_{p}"] = 0.0
            continue
        out[f"syn_count_port_{p}"] = round(float(syn_count) * weights[p] * rng.uniform(0.8, 1.0))
        out[f"failed_conn_ratio_port_{p}"] = (
            float(failed_conn_ratio) if p == focus_port else float(np.clip(rng.beta(1, 15), 0, 1))
        )
    return out


# ---------------------------------------------------------------------------
# Per-action window samplers. Each returns a dict keyed by FEATURE_COLUMNS.
# Distributions are hand-tuned to be *directionally* realistic (not calibrated
# against a real capture) -- documented in README as synthetic-for-now.
# ---------------------------------------------------------------------------

def sample_benign_window(rng: np.random.Generator) -> dict:
    flow_count = rng.poisson(25) + 1
    unique_dst_ports = min(flow_count, rng.poisson(3) + 1)
    unique_dst_ips = min(flow_count, rng.poisson(4) + 1)
    syn = flow_count + rng.poisson(2)
    ack = syn - rng.poisson(1)
    fin = int(ack * rng.uniform(0.7, 0.95))
    rst = rng.poisson(1)
    psh = rng.poisson(flow_count * 0.4)
    row = {
        "flow_count": flow_count,
        "unique_dst_ports": unique_dst_ports,
        "unique_dst_ips": unique_dst_ips,
        "syn_count": syn,
        "ack_count": _clip_pos(ack),
        "fin_count": fin,
        "rst_count": rst,
        "psh_count": psh,
        "avg_flow_duration": _clip_pos(rng.normal(2.5, 0.8)),
        "std_flow_duration": _clip_pos(rng.normal(0.6, 0.2)),
        "avg_bytes_per_flow": _clip_pos(rng.lognormal(mean=7.5, sigma=0.6)),
        "std_bytes_per_flow": _clip_pos(rng.lognormal(mean=6.5, sigma=0.6)),
        "avg_pkts_per_flow": _clip_pos(rng.normal(12, 4)),
        "failed_conn_ratio": _clip_pos(rng.beta(1, 20)),
        "inbound_bytes": _clip_pos(rng.lognormal(mean=9.5, sigma=0.7)),
        "outbound_bytes": _clip_pos(rng.lognormal(mean=9.3, sigma=0.7)),
        "bytes_ratio_out_in": _clip_pos(rng.normal(1.0, 0.3)),
        "new_dst_ip_ratio": _clip_pos(rng.beta(1, 10)),
        "iat_mean": _clip_pos(rng.normal(0.15, 0.05)),
        "iat_std": _clip_pos(rng.normal(0.05, 0.02)),
        "ttl_mean": _clip_pos(rng.normal(64, 2)),
        "ttl_std": _clip_pos(rng.normal(1.0, 0.4)),
        "win_size_mean": _clip_pos(rng.normal(29200, 3000)),
        "win_size_std": _clip_pos(rng.normal(1500, 500)),
        "pkt_len_mean": _clip_pos(rng.normal(500, 150)),
        "pkt_len_std": _clip_pos(rng.normal(120, 40)),
        "port_scan_score": _clip_pos(rng.beta(1, 30)),
    }
    # legitimate everyday traffic: HTTPS is common, admin protocols are rare
    row.update(_port_indicators(rng, row["syn_count"], row["failed_conn_ratio"],
                                 focus_port=443, focus_prob=0.35, background_prob=0.04))
    return row


def sample_precursor_window(rng: np.random.Generator, target_port: int | None = None) -> dict:
    """Windows immediately before an attack's official onset, still scripted
    as ground-truth "benign" (nothing has been confirmed yet) but carrying a
    faint, measurable precursor signal -- a single early probe packet, a
    slightly irregular timing gap, a soft hint toward the port that will
    later be attacked -- the kind of low-confidence signal a real
    infiltration's earliest moments plausibly leave behind. The simulator
    does NOT label these specially; only the state-labeling engine's
    precursor-score heuristic (app/labeling/state_labeler.py) is allowed to
    notice them, from features alone.
    """
    base = sample_benign_window(rng)
    base["port_scan_score"] = _clip_pos(rng.beta(2, 9))          # mildly elevated vs benign's beta(1,30)
    base["failed_conn_ratio"] = _clip_pos(rng.beta(2, 10))        # mildly elevated vs beta(1,20)
    base["new_dst_ip_ratio"] = _clip_pos(rng.beta(2, 8))           # mildly elevated vs beta(1,10)
    base["iat_std"] = _clip_pos(rng.normal(0.12, 0.05))            # slightly more jitter than benign
    base["unique_dst_ports"] = base["unique_dst_ports"] + rng.poisson(2)
    base.update(_port_indicators(rng, base["syn_count"], base["failed_conn_ratio"],
                                  focus_port=target_port, focus_prob=0.3, background_prob=0.04))
    return base


def sample_recon_window(rng: np.random.Generator, evasive: bool = False,
                         target_port: int | None = None, narrow: bool = False) -> dict:
    if evasive:
        # Slow/evasive: low per-window flow volume, ports trickled out over
        # many windows, deliberately randomized (jittered) timing.
        flow_count = rng.poisson(10) + 1
        unique_dst_ports = min(flow_count, rng.poisson(6) + 2)
        unique_dst_ips = rng.poisson(1) + 1
        syn = flow_count + rng.poisson(3)
        ack = max(1, int(syn * rng.uniform(0.15, 0.35)))
        iat_mean = _clip_pos(rng.normal(2.2, 0.8))     # long, human-scale-ish gaps
        iat_std = _clip_pos(rng.normal(1.4, 0.5))       # high jitter, deliberately irregular
    else:
        flow_count = rng.poisson(120) + 20
        unique_dst_ports = min(flow_count, rng.poisson(60) + 5)
        unique_dst_ips = rng.poisson(2) + 1
        syn = flow_count + rng.poisson(10)
        ack = max(1, int(syn * rng.uniform(0.05, 0.2)))
        iat_mean = _clip_pos(rng.normal(0.02, 0.01))    # fast automated scan
        iat_std = _clip_pos(rng.normal(0.01, 0.005))

    fin = int(ack * rng.uniform(0.2, 0.5))
    rst = int((syn - ack) * rng.uniform(0.5, 0.9))
    psh = rng.poisson(1)
    failed_ratio = _clip_pos((syn - ack) / max(syn, 1))
    row = {
        "flow_count": flow_count,
        "unique_dst_ports": unique_dst_ports,
        "unique_dst_ips": unique_dst_ips,
        "syn_count": syn,
        "ack_count": ack,
        "fin_count": fin,
        "rst_count": rst,
        "psh_count": psh,
        "avg_flow_duration": _clip_pos(rng.normal(0.15 if not evasive else 0.4, 0.05)),
        "std_flow_duration": _clip_pos(rng.normal(0.05, 0.02)),
        "avg_bytes_per_flow": _clip_pos(rng.lognormal(mean=3.5, sigma=0.4)),
        "std_bytes_per_flow": _clip_pos(rng.lognormal(mean=2.5, sigma=0.4)),
        "avg_pkts_per_flow": _clip_pos(rng.normal(2.5, 1.0)),
        "failed_conn_ratio": min(1.0, failed_ratio),
        "inbound_bytes": _clip_pos(rng.lognormal(mean=5.0, sigma=0.5)),
        "outbound_bytes": _clip_pos(rng.lognormal(mean=4.5, sigma=0.5)),
        "bytes_ratio_out_in": _clip_pos(rng.normal(1.1, 0.3)),
        "new_dst_ip_ratio": _clip_pos(rng.beta(3, 5) if not evasive else rng.beta(2, 8)),
        "iat_mean": iat_mean,
        "iat_std": iat_std,
        "ttl_mean": _clip_pos(rng.normal(60, 4)),
        "ttl_std": _clip_pos(rng.normal(3.0, 1.0)),
        "win_size_mean": _clip_pos(rng.normal(14600, 4000)),
        "win_size_std": _clip_pos(rng.normal(3500, 1000)),
        "pkt_len_mean": _clip_pos(rng.normal(60, 20)),
        "pkt_len_std": _clip_pos(rng.normal(20, 8)),
        "port_scan_score": min(1.0, _clip_pos(unique_dst_ports / max(flow_count, 1) + rng.normal(0.1, 0.05))),
    }
    if narrow and target_port is not None:
        # the attacker has found something interesting and is narrowing focus
        row.update(_port_indicators(rng, row["syn_count"], row["failed_conn_ratio"],
                                     focus_port=target_port, focus_prob=0.7, background_prob=0.08))
    else:
        # broad sweep: each watched port is touched with modest independent probability
        row.update(_port_indicators(rng, row["syn_count"], row["failed_conn_ratio"],
                                     focus_port=None, background_prob=0.18))
    return row


def sample_bruteforce_window(rng: np.random.Generator, port: int) -> dict:
    """Repeated automated login attempts against ONE specific service:
    single target port/IP, high flow_count from rapid retries, short flow
    duration (fast accept/reject cycles), fairly regular inter-arrival time
    (automated tooling), moderate failed-connection ratio.
    """
    flow_count = rng.poisson(45) + 10
    syn = flow_count + rng.poisson(5)
    ack = max(1, int(syn * rng.uniform(0.55, 0.8)))
    row = {
        "flow_count": flow_count,
        "unique_dst_ports": 1,
        "unique_dst_ips": 1,
        "syn_count": syn,
        "ack_count": ack,
        "fin_count": int(ack * rng.uniform(0.5, 0.8)),
        "rst_count": int((syn - ack) * rng.uniform(0.5, 0.9)),
        "psh_count": flow_count + rng.poisson(10),
        "avg_flow_duration": _clip_pos(rng.normal(0.35, 0.1)),
        "std_flow_duration": _clip_pos(rng.normal(0.08, 0.03)),
        "avg_bytes_per_flow": _clip_pos(rng.lognormal(mean=5.5, sigma=0.4)),
        "std_bytes_per_flow": _clip_pos(rng.lognormal(mean=4.2, sigma=0.4)),
        "avg_pkts_per_flow": _clip_pos(rng.normal(6, 2)),
        "failed_conn_ratio": _clip_pos(rng.beta(4, 6)),
        "inbound_bytes": _clip_pos(rng.lognormal(mean=5.5, sigma=0.4)),
        "outbound_bytes": _clip_pos(rng.lognormal(mean=5.8, sigma=0.4)),
        "bytes_ratio_out_in": _clip_pos(rng.normal(1.3, 0.3)),
        "new_dst_ip_ratio": _clip_pos(rng.beta(1, 25)),
        "iat_mean": _clip_pos(rng.normal(0.4, 0.1)),
        "iat_std": _clip_pos(rng.normal(0.08, 0.03)),
        "ttl_mean": _clip_pos(rng.normal(59, 3)),
        "ttl_std": _clip_pos(rng.normal(1.2, 0.4)),
        "win_size_mean": _clip_pos(rng.normal(40000, 5000)),
        "win_size_std": _clip_pos(rng.normal(2500, 700)),
        "pkt_len_mean": _clip_pos(rng.normal(300, 80)),
        "pkt_len_std": _clip_pos(rng.normal(80, 25)),
        "port_scan_score": _clip_pos(rng.beta(1, 25)),
    }
    row.update(_port_indicators(rng, row["syn_count"], row["failed_conn_ratio"],
                                 focus_port=port, focus_prob=0.92, background_prob=0.03))
    return row


def sample_lateral_movement_window(rng: np.random.Generator, port: int) -> dict:
    flow_count = rng.poisson(15) + 3
    unique_dst_ips = min(flow_count, rng.poisson(6) + 2)
    syn = flow_count + rng.poisson(2)
    ack = max(1, int(syn * rng.uniform(0.6, 0.85)))
    row = {
        "flow_count": flow_count,
        "unique_dst_ports": rng.poisson(2) + 1,
        "unique_dst_ips": unique_dst_ips,
        "syn_count": syn,
        "ack_count": ack,
        "fin_count": int(ack * rng.uniform(0.4, 0.7)),
        "rst_count": int((syn - ack) * rng.uniform(0.4, 0.8)),
        "psh_count": rng.poisson(flow_count * 0.6),
        "avg_flow_duration": _clip_pos(rng.normal(1.8, 0.5)),
        "std_flow_duration": _clip_pos(rng.normal(0.5, 0.2)),
        "avg_bytes_per_flow": _clip_pos(rng.lognormal(mean=8.0, sigma=0.5)),
        "std_bytes_per_flow": _clip_pos(rng.lognormal(mean=6.8, sigma=0.5)),
        "avg_pkts_per_flow": _clip_pos(rng.normal(20, 6)),
        "failed_conn_ratio": _clip_pos(rng.beta(2, 8)),
        "inbound_bytes": _clip_pos(rng.lognormal(mean=7.5, sigma=0.5)),
        "outbound_bytes": _clip_pos(rng.lognormal(mean=7.5, sigma=0.5)),
        "bytes_ratio_out_in": _clip_pos(rng.normal(1.2, 0.3)),
        "new_dst_ip_ratio": _clip_pos(rng.beta(6, 4)),
        "iat_mean": _clip_pos(rng.normal(0.5, 0.15)),
        "iat_std": _clip_pos(rng.normal(0.2, 0.08)),
        "ttl_mean": _clip_pos(rng.normal(62, 2)),
        "ttl_std": _clip_pos(rng.normal(2.0, 0.7)),
        "win_size_mean": _clip_pos(rng.normal(32000, 4000)),
        "win_size_std": _clip_pos(rng.normal(2500, 800)),
        "pkt_len_mean": _clip_pos(rng.normal(700, 150)),
        "pkt_len_std": _clip_pos(rng.normal(180, 50)),
        "port_scan_score": _clip_pos(rng.beta(2, 15)),
    }
    row.update(_port_indicators(rng, row["syn_count"], row["failed_conn_ratio"],
                                 focus_port=port, focus_prob=0.85, background_prob=0.04))
    return row


def sample_c2_window(rng: np.random.Generator) -> dict:
    flow_count = rng.poisson(2) + 1
    row = {
        "flow_count": flow_count,
        "unique_dst_ports": 1,
        "unique_dst_ips": 1,
        "syn_count": flow_count,
        "ack_count": flow_count,
        "fin_count": flow_count,
        "rst_count": 0,
        "psh_count": flow_count,
        "avg_flow_duration": _clip_pos(rng.normal(0.4, 0.1)),
        "std_flow_duration": _clip_pos(rng.normal(0.05, 0.02)),
        "avg_bytes_per_flow": _clip_pos(rng.lognormal(mean=4.5, sigma=0.3)),
        "std_bytes_per_flow": _clip_pos(rng.lognormal(mean=3.0, sigma=0.3)),
        "avg_pkts_per_flow": _clip_pos(rng.normal(4, 1)),
        "failed_conn_ratio": _clip_pos(rng.beta(1, 30)),
        "inbound_bytes": _clip_pos(rng.lognormal(mean=4.0, sigma=0.3)),
        "outbound_bytes": _clip_pos(rng.lognormal(mean=4.0, sigma=0.3)),
        "bytes_ratio_out_in": _clip_pos(rng.normal(1.0, 0.15)),
        "new_dst_ip_ratio": _clip_pos(rng.beta(1, 20)),
        "iat_mean": _clip_pos(rng.normal(45, 6)),       # long, regular beacon interval
        "iat_std": _clip_pos(rng.normal(0.8, 0.3)),      # suspiciously LOW jitter (too regular)
        "ttl_mean": _clip_pos(rng.normal(55, 2)),
        "ttl_std": _clip_pos(rng.normal(0.8, 0.3)),
        "win_size_mean": _clip_pos(rng.normal(8192, 1000)),
        "win_size_std": _clip_pos(rng.normal(300, 100)),
        "pkt_len_mean": _clip_pos(rng.normal(150, 30)),
        "pkt_len_std": _clip_pos(rng.normal(20, 8)),
        "port_scan_score": _clip_pos(rng.beta(1, 40)),
    }
    # HTTPS-cloaked beaconing: same port benign traffic uses, so port alone
    # can't distinguish it -- the timing regularity above is what has to.
    row.update(_port_indicators(rng, row["syn_count"], row["failed_conn_ratio"],
                                 focus_port=443, focus_prob=0.85, background_prob=0.03))
    return row


def sample_exfiltration_window(rng: np.random.Generator) -> dict:
    flow_count = rng.poisson(8) + 2
    row = {
        "flow_count": flow_count,
        "unique_dst_ports": rng.poisson(1) + 1,
        "unique_dst_ips": rng.poisson(1) + 1,
        "syn_count": flow_count,
        "ack_count": flow_count,
        "fin_count": int(flow_count * rng.uniform(0.5, 0.9)),
        "rst_count": rng.poisson(1),
        "psh_count": flow_count * 3 + rng.poisson(5),
        "avg_flow_duration": _clip_pos(rng.normal(8.0, 2.0)),
        "std_flow_duration": _clip_pos(rng.normal(2.0, 0.5)),
        "avg_bytes_per_flow": _clip_pos(rng.lognormal(mean=12.5, sigma=0.6)),
        "std_bytes_per_flow": _clip_pos(rng.lognormal(mean=11.0, sigma=0.6)),
        "avg_pkts_per_flow": _clip_pos(rng.normal(400, 100)),
        "failed_conn_ratio": _clip_pos(rng.beta(1, 30)),
        "inbound_bytes": _clip_pos(rng.lognormal(mean=6.0, sigma=0.4)),
        "outbound_bytes": _clip_pos(rng.lognormal(mean=13.5, sigma=0.6)),
        "bytes_ratio_out_in": _clip_pos(rng.normal(25.0, 6.0)),
        "new_dst_ip_ratio": _clip_pos(rng.beta(1, 20)),
        "iat_mean": _clip_pos(rng.normal(0.05, 0.02)),
        "iat_std": _clip_pos(rng.normal(0.02, 0.01)),
        "ttl_mean": _clip_pos(rng.normal(56, 2)),
        "ttl_std": _clip_pos(rng.normal(1.0, 0.4)),
        "win_size_mean": _clip_pos(rng.normal(65535, 3000)),
        "win_size_std": _clip_pos(rng.normal(1500, 500)),
        "pkt_len_mean": _clip_pos(rng.normal(1400, 100)),
        "pkt_len_std": _clip_pos(rng.normal(150, 40)),
        "port_scan_score": _clip_pos(rng.beta(1, 40)),
    }
    row.update(_port_indicators(rng, row["syn_count"], row["failed_conn_ratio"],
                                 focus_port=443, focus_prob=0.8, background_prob=0.03))
    return row


PRECURSOR_WINDOWS = 4


def _build_attack_script(rng: np.random.Generator, evasive: bool, bf_action: str, lm_action: str):
    """Returns a list of (action, n_windows) segments for one attack host."""
    warmup = rng.integers(15, 30)
    recon_len = rng.integers(35, 55) if evasive else rng.integers(6, 12)
    bf_len = rng.integers(4, 9)
    lm_len = rng.integers(8, 16)
    c2_len = rng.integers(10, 20)
    exfil_len = rng.integers(3, 7)
    tail = rng.integers(10, 20)
    warmup = max(0, int(warmup) - PRECURSOR_WINDOWS)
    return [
        ("benign", int(warmup)),
        ("benign_precursor", PRECURSOR_WINDOWS),
        ("port_scan", int(recon_len)),
        (bf_action, int(bf_len)),
        (lm_action, int(lm_len)),
        ("c2_beacon", int(c2_len)),
        ("data_exfiltration", int(exfil_len)),
        ("benign", int(tail)),
    ]


def _sample_for_action(rng: np.random.Generator, action: str, evasive: bool,
                        target_port: int | None, narrow: bool) -> dict:
    if action == "benign":
        return sample_benign_window(rng)
    if action == "benign_precursor":
        return sample_precursor_window(rng, target_port=target_port)
    if action == "port_scan":
        return sample_recon_window(rng, evasive=evasive, target_port=target_port, narrow=narrow)
    if action in BRUTEFORCE_ACTIONS:
        return sample_bruteforce_window(rng, port=PORT_OF_ACTION[action])
    if action in LATERAL_ACTIONS:
        return sample_lateral_movement_window(rng, port=PORT_OF_ACTION[action])
    if action == "c2_beacon":
        return sample_c2_window(rng)
    if action == "data_exfiltration":
        return sample_exfiltration_window(rng)
    raise ValueError(f"unknown action: {action}")


def generate_host_timeline(host_id: str, rng: np.random.Generator, is_attack: bool, evasive: bool = False,
                            benign_len: int = 200, bf_action: str | None = None,
                            lm_action: str | None = None) -> pd.DataFrame:
    rows = []
    if is_attack:
        bf_action = bf_action or rng.choice(BRUTEFORCE_ACTIONS)
        lm_action = lm_action or rng.choice(LATERAL_ACTIONS)
        target_port = PORT_OF_ACTION[bf_action]
        script = _build_attack_script(rng, evasive, bf_action, lm_action)
    else:
        target_port = None
        script = [("benign", benign_len)]

    window_idx = 0
    n_recon_windows = next((n for a, n in script if a == "port_scan"), 0)
    recon_seen = 0
    for action, n in script:
        # true_stage is the raw ground-truth label the simulator scripted --
        # "benign_precursor" is recorded as "benign" since nothing is
        # confirmed yet; only the state-labeling engine (from features alone)
        # is allowed to distinguish it as ambiguous.
        recorded_action = "benign" if action == "benign_precursor" else action
        for _ in range(n):
            narrow = False
            if action == "port_scan":
                narrow = recon_seen >= n_recon_windows * 0.6  # narrows onto target port in the later ~40%
                recon_seen += 1
            feats = _sample_for_action(rng, action, evasive, target_port, narrow)
            row = {"host_id": host_id, "window_idx": window_idx, "true_stage": recorded_action}
            row.update(feats)
            rows.append(row)
            window_idx += 1
    df = pd.DataFrame(rows)
    df["timestamp"] = df["window_idx"] * WINDOW_SECONDS
    return df[["host_id", "window_idx", "timestamp", "true_stage"] + FEATURE_COLUMNS]


def generate_dataset(seed: int = RANDOM_SEED, n_benign_hosts: int = 40, n_attack_hosts: int = 19,
                      benign_len: int = 200) -> pd.DataFrame:
    """Builds the full synthetic dataset: benign-only hosts + attack-progression
    hosts (one of which uses the slow/evasive recon variant). Each attack
    host is independently assigned a brute-force action and a lateral-
    movement action so the dataset covers combinations of the two."""
    master_rng = np.random.default_rng(seed)
    frames = []

    for i in range(n_benign_hosts):
        host_id = f"benign-host-{i:03d}"
        host_rng = np.random.default_rng(master_rng.integers(0, 2**32 - 1))
        frames.append(generate_host_timeline(host_id, host_rng, is_attack=False, benign_len=benign_len))

    for i in range(n_attack_hosts):
        host_id = f"attack-host-{i:03d}"
        host_rng = np.random.default_rng(master_rng.integers(0, 2**32 - 1))
        evasive = (i == 0)  # first attack host runs the slow/evasive recon variant
        if evasive:
            frames.append(generate_host_timeline(host_id, host_rng, is_attack=True, evasive=True,
                                                  bf_action="ssh_bruteforce", lm_action="ssh_lateral_movement"))
        else:
            frames.append(generate_host_timeline(host_id, host_rng, is_attack=True, evasive=False))

    full = pd.concat(frames, ignore_index=True)
    return full


if __name__ == "__main__":
    df = generate_dataset()
    from app.config import SYNTHETIC_CSV
    df.to_csv(SYNTHETIC_CSV, index=False)
    print(f"wrote {len(df)} rows to {SYNTHETIC_CSV}")
    print(df["true_stage"].value_counts())
