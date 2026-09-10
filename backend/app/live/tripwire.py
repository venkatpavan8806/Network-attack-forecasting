"""Fast, rule-based, millisecond-latency alerting.

This is explicitly NOT the ML world model -- it is a simple threshold
heuristic evaluated synchronously inside the packet-capture callback, the
same class of "IDS raises an alert: port scan detected" logic the project's
own pitch describes as the starting point CARIBBEAN builds on top of. It
exists to give an immediate signal (real milliseconds, measured from packet
arrival to alert creation) while the LSTM forecast -- which by design needs
real history across several windows to reason over a trend, not a single
packet -- is still accumulating that history. The two are complementary:
this catches "something just happened", the world model answers "given
what's happened, what's likely next, and how confident should you be".

Two rules, both evaluated per inbound SYN:
  1. watched_port_contact: any SYN to port 22/445/3389/443 (the same four
     services the trained model watches for) fires immediately, with a
     short per-(remote_ip, port) cooldown so a single repeated attempt does
     not flood the alert feed.
  2. rapid_multi_port: N or more DISTINCT local ports touched by the same
     remote_ip within a short rolling time window -- the classic real-time
     port-scan heuristic used by rule-based IDS tools (Snort/Suricata-style
     thresholding), independent of the ML model entirely.
"""
from __future__ import annotations

import threading
from collections import deque, defaultdict
from datetime import datetime, timezone

from app.config import WATCHED_PORTS

PORT_SERVICE_NAMES = {22: "SSH", 445: "SMB", 3389: "RDP", 443: "HTTPS"}

MULTI_PORT_WINDOW_SECONDS = 3.0
MULTI_PORT_THRESHOLD = 4
WATCHED_PORT_COOLDOWN_SECONDS = 5.0


class Tripwire:
    def __init__(self):
        self._lock = threading.Lock()
        self.alerts: deque = deque(maxlen=200)
        self._recent_syn_ports: dict[str, deque] = defaultdict(deque)
        self._last_watched_alert: dict[tuple[str, int], float] = {}
        self._next_id = 1

    def reset(self):
        with self._lock:
            self.alerts.clear()
            self._recent_syn_ports.clear()
            self._last_watched_alert.clear()

    def _raise(self, remote_ip: str, message: str, severity: str, detail: dict) -> dict:
        alert = {
            "id": self._next_id,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "remote_ip": remote_ip,
            "message": message,
            "severity": severity,
            "detail": detail,
        }
        with self._lock:
            self._next_id += 1
            self.alerts.append(alert)
        return alert

    def on_inbound_syn(self, remote_ip: str, local_port: int, ts: float) -> list[dict]:
        """Call synchronously from the packet-capture callback for every
        inbound SYN (no ACK) -- i.e. a new connection attempt someone else
        initiated toward us. Returns any alerts that fired (usually empty)."""
        fired = []

        if local_port in WATCHED_PORTS:
            key = (remote_ip, local_port)
            last = self._last_watched_alert.get(key)  # None means "never alerted" -- ts=0.0 must still fire
            if last is None or ts - last >= WATCHED_PORT_COOLDOWN_SECONDS:
                self._last_watched_alert[key] = ts
                service_name = PORT_SERVICE_NAMES.get(local_port, str(local_port))
                fired.append(self._raise(
                    remote_ip, f"Connection attempt to {service_name} (port {local_port}) from {remote_ip}",
                    severity="warning",
                    detail={"rule": "watched_port_contact", "port": local_port, "service": service_name},
                ))

        with self._lock:
            buf = self._recent_syn_ports[remote_ip]
            buf.append((ts, local_port))
            cutoff = ts - MULTI_PORT_WINDOW_SECONDS
            while buf and buf[0][0] < cutoff:
                buf.popleft()
            n_distinct = len({p for _, p in buf})
            distinct_ports = sorted({p for _, p in buf})

        if n_distinct >= MULTI_PORT_THRESHOLD:
            fired.append(self._raise(
                remote_ip,
                f"Rapid multi-port probing from {remote_ip}: {n_distinct} distinct ports in under {MULTI_PORT_WINDOW_SECONDS:.0f}s",
                severity="critical",
                detail={"rule": "rapid_multi_port", "distinct_ports": distinct_ports, "window_seconds": MULTI_PORT_WINDOW_SECONDS},
            ))

        return fired

    def recent(self, limit: int = 50) -> list[dict]:
        with self._lock:
            items = list(self.alerts)[-limit:]
        return list(reversed(items))


tripwire = Tripwire()
