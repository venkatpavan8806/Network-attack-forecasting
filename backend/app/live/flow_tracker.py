"""Real-time flow/window aggregation from live-captured packets into the
SAME feature schema (FEATURE_COLUMNS) the model was trained on.

This is a genuine adaptation, not a re-derivation from scratch, and the
vantage point is different from the synthetic dataset's: the synthetic
generator profiled a host's own OUTBOUND behavior (an attacker's NetFlow
export of the connections it initiated). Here we capture on ONE host's own
NIC (non-promiscuous), so we naturally see the opposite vantage point: a
remote peer's INBOUND connection attempts against us. The mapping below is
the honest, documented translation between the two:

  - `host_id` becomes the REMOTE peer's IP address (the potential attacker),
    not "this machine".
  - `unique_dst_ports` = distinct LOCAL ports that remote peer's packets
    targeted (directly analogous: "how many of our ports did they touch").
  - `unique_dst_ips` is fixed at 1.0 -- a non-promiscuous single-host capture
    can only ever see traffic addressed to itself, so "how many hosts did
    the remote contact" is unobservable from here and is not estimated.
  - `new_dst_ip_ratio` is repurposed as "how novel is this remote peer to
    this capture session" (1.0 the first time it's seen, decaying as
    1/times_seen afterward) rather than its original meaning.
  - `dst_port_is_22/445/3389/443` keep their EXACT original meaning: real
    destination port observed in real inbound packets. This is the
    least-adapted, most directly-comparable feature to training.
  - `syn_count_port_*` / `failed_conn_ratio_port_*` also keep their exact
    original meaning: real per-port SYN counts and per-port handshake
    failure rate, computed the same way as the aggregate syn_count/
    failed_conn_ratio below but scoped to one watched port at a time.
  - `failed_conn_ratio`, `ttl_mean/std`, `win_size_mean/std`, `iat_mean/std`
    are computed directly from real captured packet fields.

INITIATION-DIRECTION FILTER (critical, found via live testing): a remote
host merely REPLYING to a connection WE opened (ordinary browsing, cloud
sync, CDN traffic) must never be scored as "their behavior toward us" -- a
browser session naturally uses a fresh ephemeral local port per connection,
which a naive "how many of our ports did they touch" count would misread as
scanning. Every feature below is therefore computed ONLY from flows where
the remote side sent the first packet (a bare SYN to us) -- i.e. connections
THEY initiated. A remote host with no such flow in a window contributes no
feature row at all for that window (see roll_window). This was found and
fixed after an early live run flagged ordinary CDN/cloud IPs as "port scan"
purely because this laptop had browsed to them.
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np

from app.config import FEATURE_COLUMNS, WINDOW_SECONDS, WATCHED_PORTS

PORT_SERVICE_NAMES = {22: "SSH", 445: "SMB", 3389: "RDP", 443: "HTTPS"}

# how many raw packets to keep per remote host for on-demand inspection --
# independent of window rollovers, capped so a chatty host can't grow this
# without bound
PACKET_LOG_MAXLEN = 300


@dataclass
class PacketRecord:
    ts: float
    direction: str          # 'in' (from remote to us) or 'out' (from us to remote)
    local_port: int
    remote_port: int
    flags: str               # scapy TCP flags string, e.g. "S", "SA", "PA", "R"
    ttl: int
    win_size: int
    pkt_len: int


def describe_packet(rec: PacketRecord) -> str:
    """Plain-English description of a single captured packet, for the
    per-host packet drill-down in the UI. Not a model output -- a direct,
    deterministic read of the TCP flags/ports on the actual packet."""
    is_syn, is_ack, is_rst, is_fin, is_psh = ("S" in rec.flags, "A" in rec.flags,
                                               "R" in rec.flags, "F" in rec.flags, "P" in rec.flags)
    watched_port = rec.local_port if rec.direction == "in" else rec.remote_port
    service = PORT_SERVICE_NAMES.get(watched_port)
    port_note = f" [{service}]" if service else ""

    if is_rst:
        base = "Connection reset/refused (RST)"
    elif is_syn and is_ack:
        base = "Remote accepted the connection (SYN-ACK)" if rec.direction == "in" else "We accepted their connection (SYN-ACK)"
    elif is_syn:
        base = "New connection attempt from them (SYN)" if rec.direction == "in" else "We opened a new connection (SYN)"
    elif is_fin:
        base = "Connection closing (FIN)"
    elif is_psh and is_ack:
        base = f"Data transfer, {rec.pkt_len} bytes (PSH-ACK)"
    elif is_ack:
        base = "Acknowledgement on an ongoing connection (ACK)"
    else:
        base = f"TCP segment (flags={rec.flags or 'none'})"

    return base + port_note


@dataclass
class _RemoteBucket:
    records: list = field(default_factory=list)


class FlowTracker:
    """Thread-safe packet ingestion + per-window feature computation for one
    local host's own NIC capture."""

    def __init__(self, local_ip: str):
        self.local_ip = local_ip
        self._lock = threading.Lock()
        self._buckets: dict[str, _RemoteBucket] = defaultdict(_RemoteBucket)
        self._times_seen: dict[str, int] = defaultdict(int)  # persists across windows
        # raw per-packet log for on-demand inspection -- independent of window
        # rollovers (never cleared by roll_window), capped per remote host
        self._packet_log: dict[str, deque] = defaultdict(lambda: deque(maxlen=PACKET_LOG_MAXLEN))

    def ingest_tcp(self, remote_ip: str, direction: str, local_port: int, remote_port: int,
                    flags: str, ttl: int, win_size: int, pkt_len: int, ts: float | None = None):
        rec = PacketRecord(ts=ts if ts is not None else time.time(), direction=direction, local_port=local_port,
                            remote_port=remote_port, flags=flags, ttl=ttl, win_size=win_size, pkt_len=pkt_len)
        with self._lock:
            self._buckets[remote_ip].records.append(rec)
            self._packet_log[remote_ip].append(rec)

    def recent_packets(self, remote_ip: str, limit: int = 100) -> list[dict]:
        """Raw, individual packets captured to/from this remote host, most
        recent first -- for the per-host packet drill-down in the UI. This is
        NOT filtered by the initiation-direction rule the aggregate features
        use; it shows everything actually seen, exactly as captured."""
        with self._lock:
            recs = list(self._packet_log.get(remote_ip, []))
        recs = recs[-limit:][::-1]
        return [
            {
                "timestamp": r.ts,
                "direction": r.direction,
                "local_port": r.local_port,
                "remote_port": r.remote_port,
                "flags": r.flags,
                "ttl": r.ttl,
                "win_size": r.win_size,
                "pkt_len": r.pkt_len,
                "description": describe_packet(r),
            }
            for r in recs
        ]

    def _features_for_bucket(self, remote_ip: str, bucket: _RemoteBucket) -> dict | None:
        recs = bucket.records

        # -- initiation-direction filter -------------------------------
        # Critical distinction: a remote host REPLYING to a connection WE
        # opened (ordinary browsing/cloud traffic) must never be scored as
        # "their behavior toward us" -- only connections THEY initiated to
        # a port on us are evidence about them. Without this, visiting any
        # website makes that server's IP look like it "touched many of our
        # ports", because each outbound browser connection uses a fresh
        # ephemeral local source port -- an artifact of how we contacted
        # them, not of anything they did.
        by_port_all: dict[int, list] = defaultdict(list)
        for r in recs:
            by_port_all[r.local_port].append(r)

        remote_initiated_ports = set()
        for port, port_recs in by_port_all.items():
            first = min(port_recs, key=lambda r: r.ts)
            if first.direction == "in" and "S" in first.flags and "A" not in first.flags:
                remote_initiated_ports.add(port)

        if not remote_initiated_ports:
            # every flow this window was something WE opened; this remote
            # host gave us no evidence of its own -- skip it entirely
            return None

        recs = [r for r in recs if r.local_port in remote_initiated_ports]
        inbound = [r for r in recs if r.direction == "in"]
        outbound = [r for r in recs if r.direction == "out"]

        syn_count = sum(1 for r in inbound if "S" in r.flags and "A" not in r.flags)
        ack_count = sum(1 for r in inbound if "A" in r.flags)
        fin_count = sum(1 for r in inbound if "F" in r.flags)
        rst_count = sum(1 for r in inbound if "R" in r.flags)
        psh_count = sum(1 for r in inbound if "P" in r.flags)

        local_ports_touched = sorted({r.local_port for r in inbound})
        flow_count = max(syn_count, 1) if inbound else 0

        # group by local_port to approximate one "flow" per port touched
        by_port: dict[int, list] = defaultdict(list)
        for r in inbound:
            by_port[r.local_port].append(r)
        durations, byte_totals, pkt_counts, completed = [], [], [], []
        completed_by_port: dict[int, int] = {}
        for port, port_recs in by_port.items():
            ts_list = [r.ts for r in port_recs]
            durations.append(max(ts_list) - min(ts_list) if len(ts_list) > 1 else 0.0)
            byte_totals.append(sum(r.pkt_len for r in port_recs))
            pkt_counts.append(len(port_recs))
            had_syn = any("S" in r.flags and "A" not in r.flags for r in port_recs)
            our_synack = any("S" in r.flags and "A" in r.flags for r in outbound if r.local_port == port)
            is_completed = 1 if (had_syn and our_synack) else 0
            completed.append(is_completed)
            completed_by_port[port] = is_completed

        inbound_bytes = sum(r.pkt_len for r in inbound)
        outbound_bytes = sum(r.pkt_len for r in outbound)

        all_ts = sorted(r.ts for r in recs)
        iats = np.diff(all_ts) if len(all_ts) > 1 else np.array([0.0])

        ttls = [r.ttl for r in inbound] or [0]
        win_sizes = [r.win_size for r in inbound] or [0]
        pkt_lens = [r.pkt_len for r in inbound] or [0]

        self._times_seen[remote_ip] += 1
        new_dst_ip_ratio = 1.0 / self._times_seen[remote_ip]

        n_flows = max(len(by_port), 1)
        failed_conn_ratio = 1.0 - (sum(completed) / n_flows)

        feats = {
            "flow_count": float(flow_count),
            "unique_dst_ports": float(len(local_ports_touched)),
            "unique_dst_ips": 1.0,
            "syn_count": float(syn_count),
            "ack_count": float(ack_count),
            "fin_count": float(fin_count),
            "rst_count": float(rst_count),
            "psh_count": float(psh_count),
            "avg_flow_duration": float(np.mean(durations)) if durations else 0.0,
            "std_flow_duration": float(np.std(durations)) if durations else 0.0,
            "avg_bytes_per_flow": float(np.mean(byte_totals)) if byte_totals else 0.0,
            "std_bytes_per_flow": float(np.std(byte_totals)) if byte_totals else 0.0,
            "avg_pkts_per_flow": float(np.mean(pkt_counts)) if pkt_counts else 0.0,
            "failed_conn_ratio": float(np.clip(failed_conn_ratio, 0.0, 1.0)),
            "inbound_bytes": float(inbound_bytes),
            "outbound_bytes": float(outbound_bytes),
            "bytes_ratio_out_in": float(outbound_bytes / max(inbound_bytes, 1)),
            "new_dst_ip_ratio": float(new_dst_ip_ratio),
            "iat_mean": float(np.mean(iats)),
            "iat_std": float(np.std(iats)),
            "ttl_mean": float(np.mean(ttls)),
            "ttl_std": float(np.std(ttls)),
            "win_size_mean": float(np.mean(win_sizes)),
            "win_size_std": float(np.std(win_sizes)),
            "pkt_len_mean": float(np.mean(pkt_lens)),
            "pkt_len_std": float(np.std(pkt_lens)),
            # Laplace-smoothed ratio: a genuine scan (many ports, ~1 attempt
            # each) still saturates toward 1.0 at volume, but a single
            # isolated connection attempt (unique_dst_ports=1, flow_count=1)
            # no longer trivially scores 1.0 the way a naive ratio would --
            # that edge case is exactly what flagged ordinary low-volume
            # traffic as a "port scan" before this fix.
            "port_scan_score": float(min(1.0, len(local_ports_touched) / (flow_count + 2))),
        }
        for p in WATCHED_PORTS:
            port_recs = by_port.get(p, [])
            feats[f"dst_port_is_{p}"] = 1.0 if port_recs else 0.0
            # per-port INTENSITY -- how much connection volume and how many
            # failures concentrated on THIS specific port, computed the same
            # way as the aggregate syn_count/failed_conn_ratio above but
            # scoped to one port, so a brute-force against one service
            # doesn't look identical to a brief touch of the same port --
            # see config.py:FEATURE_DOCS for why this exists.
            feats[f"syn_count_port_{p}"] = float(sum(1 for r in port_recs if "S" in r.flags and "A" not in r.flags))
            feats[f"failed_conn_ratio_port_{p}"] = float(1.0 - completed_by_port[p]) if port_recs else 0.0

        return {k: feats[k] for k in FEATURE_COLUMNS}

    def roll_window(self) -> dict[str, dict]:
        """Computes features for every remote host that showed at least one
        REMOTE-INITIATED connection attempt this window, then clears the
        window. Hosts whose only activity was replying to connections WE
        opened are skipped -- see _features_for_bucket. Returns
        {remote_ip: feature_dict}."""
        with self._lock:
            buckets = self._buckets
            self._buckets = defaultdict(_RemoteBucket)
        result = {}
        for ip, b in buckets.items():
            if not b.records:
                continue
            feats = self._features_for_bucket(ip, b)
            if feats is not None:
                result[ip] = feats
        return result
