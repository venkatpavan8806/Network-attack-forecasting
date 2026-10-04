"""Offline packet-capture replay: .pcap / .pcapng file -> per-window features.

Runs the uploaded capture through exactly the same code the live capture
agent uses (app/live/flow_tracker.py for the FEATURE_COLUMNS vector,
app/live/tripwire.py for the fast rule-based alerts), but windows are cut by
the packets' own timestamps instead of wall-clock time: every WINDOW_SECONDS
of capture time becomes one window.

Vantage point: like the live agent, features describe what each REMOTE
peer did toward ONE monitored host (`local_ip`). If the caller does not say
which host the capture was taken on, the IP address that appears in the most
TCP packets is used (true for a normal single-machine Wireshark capture).
Only TCP over IPv4/IPv6 is used; other traffic is counted and skipped.
"""
from __future__ import annotations

import io
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field

from app.config import WINDOW_SECONDS
from app.live.flow_tracker import FlowTracker, PacketRecord, describe_packet
from app.live.tripwire import Tripwire

MAX_PACKETS = 3_000_000
PACKETS_KEPT_PER_HOST = 300


class PcapError(ValueError):
    pass


@dataclass
class PcapReplay:
    local_ip: str
    windows: list[dict] = field(default_factory=list)   # {remote_ip, window_start, features}
    packets: list[dict] = field(default_factory=list)   # newest PACKETS_KEPT_PER_HOST per remote
    alerts: list[dict] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


def _tcp_fields(pkt):
    """(src, dst, sport, dport, flags, ttl, win, length) for an IPv4/IPv6 TCP packet, else None."""
    from scapy.layers.inet import IP, TCP
    from scapy.layers.inet6 import IPv6
    if TCP not in pkt:
        return None
    if IP in pkt:
        ip = pkt[IP]
        ttl = int(ip.ttl)
    elif IPv6 in pkt:
        ip = pkt[IPv6]
        ttl = int(ip.hlim)
    else:
        return None
    tcp = pkt[TCP]
    return ip.src, ip.dst, int(tcp.sport), int(tcp.dport), str(tcp.flags), ttl, int(tcp.window), len(pkt)


def _read_packets(data: bytes):
    from scapy.utils import PcapReader, PcapNgReader
    from scapy.error import Scapy_Exception
    magic = data[:4]
    reader_cls = PcapNgReader if magic == b"\x0a\x0d\x0d\x0a" else PcapReader
    try:
        reader = reader_cls(io.BytesIO(data))
    except (Scapy_Exception, ValueError, EOFError) as e:
        raise PcapError(f"not a valid pcap/pcapng file: {e}")
    return reader


def guess_local_ip(data: bytes, sample: int = 50_000) -> str:
    counts: Counter = Counter()
    reader = _read_packets(data)
    try:
        for i, pkt in enumerate(reader):
            if i >= sample:
                break
            f = _tcp_fields(pkt)
            if f:
                counts[f[0]] += 1
                counts[f[1]] += 1
    finally:
        reader.close()
    if not counts:
        raise PcapError("the capture contains no TCP packets")
    return counts.most_common(1)[0][0]


def replay_pcap(data: bytes, local_ip: str | None = None, window_seconds: float = WINDOW_SECONDS) -> PcapReplay:
    if not data:
        raise PcapError("empty file")
    local_ip = local_ip or guess_local_ip(data)
    out = PcapReplay(local_ip=local_ip)
    tracker = FlowTracker(local_ip)
    tripwire = Tripwire()  # memory sink: alerts are collected and stored by the caller
    recent_by_remote: dict[str, deque] = defaultdict(lambda: deque(maxlen=PACKETS_KEPT_PER_HOST))

    n_total = n_tcp = n_used = 0
    first_ts = last_ts = None
    window_start = None

    def roll(start: float):
        for remote_ip, feats in tracker.roll_window().items():
            out.windows.append({"remote_ip": remote_ip, "window_start": start, "features": feats})

    reader = _read_packets(data)
    try:
        for pkt in reader:
            n_total += 1
            if n_total > MAX_PACKETS:
                raise PcapError(f"capture has more than {MAX_PACKETS:,} packets -- please trim it")
            f = _tcp_fields(pkt)
            if f is None:
                continue
            n_tcp += 1
            src, dst, sport, dport, flags, ttl, win, length = f
            if src == local_ip:
                direction, remote_ip, local_port, remote_port = "out", dst, sport, dport
            elif dst == local_ip:
                direction, remote_ip, local_port, remote_port = "in", src, dport, sport
            else:
                continue  # traffic between two other hosts: not from the monitored host's vantage point
            ts = float(pkt.time)
            if first_ts is None:
                first_ts = window_start = ts
            last_ts = ts
            if ts >= window_start + window_seconds:
                roll(window_start)
                # jump over idle gaps instead of rolling thousands of empty windows
                skipped = int((ts - window_start) // window_seconds)
                window_start += skipped * window_seconds
            n_used += 1
            tracker.ingest_tcp(remote_ip=remote_ip, direction=direction, local_port=local_port,
                               remote_port=remote_port, flags=flags, ttl=ttl, win_size=win, pkt_len=length, ts=ts)
            rec = PacketRecord(ts=ts, direction=direction, local_port=local_port, remote_port=remote_port,
                               flags=flags, ttl=ttl, win_size=win, pkt_len=length)
            recent_by_remote[remote_ip].append(rec)
            if direction == "in" and "S" in flags and "A" not in flags:
                out.alerts.extend(tripwire.on_inbound_syn(remote_ip=remote_ip, local_port=local_port, ts=ts))
    finally:
        reader.close()

    if window_start is not None:
        roll(window_start)
    if n_used == 0:
        raise PcapError(f"no TCP packets to or from {local_ip} in this capture "
                        f"({n_total} packets, {n_tcp} TCP) -- check the monitored host IP")

    for remote_ip, recs in recent_by_remote.items():
        for r in recs:
            out.packets.append({"remote_ip": remote_ip, "ts": r.ts, "direction": r.direction,
                                "local_port": r.local_port, "remote_port": r.remote_port, "flags": r.flags,
                                "ttl": r.ttl, "win_size": r.win_size, "pkt_len": r.pkt_len,
                                "description": describe_packet(r)})
    out.stats = {
        "packets_total": n_total, "packets_tcp": n_tcp, "packets_used": n_used,
        "capture_seconds": round((last_ts - first_ts), 3) if first_ts is not None else 0.0,
        "windows": len(out.windows), "remote_hosts": len({w["remote_ip"] for w in out.windows}),
        "remote_hosts_seen": len(recent_by_remote), "alerts": len(out.alerts),
    }
    return out
