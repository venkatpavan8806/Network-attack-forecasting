"""Network Attack Forecasting -- capture agent.

Runs on YOUR machine (the one you want to monitor), captures TCP packets to
and from it, turns them into the model's 30-second window features locally
(app/live/flow_tracker.py -- the same code the server uses for pcap
uploads), and sends only those features + a small sample of packet headers
to your account on the website. Packet payloads never leave the machine.

    python nadf_agent.py --token nadf_xxx                # auto-detect interface
    python nadf_agent.py --list-interfaces
    python nadf_agent.py --token nadf_xxx --iface "Wi-Fi"

Requirements: Python 3.9+, `pip install -r requirements.txt` (scapy, numpy),
and packet-capture permission:
  - Windows: install Npcap (https://npcap.com) and run the terminal as Administrator
  - Linux:   sudo python3 nadf_agent.py ...   (or grant cap_net_raw)
  - macOS:   sudo python3 nadf_agent.py ...

Only traffic addressed to/from this machine is read (no promiscuous mode).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict, deque
from datetime import datetime, timezone

AGENT_VERSION = "1.0.0"
DEFAULT_SERVER = "__NADF_SERVER_URL__"  # filled in by the website's download endpoint
PACKETS_PER_HOST_PER_UPLOAD = 100
MAX_BACKLOG_BATCHES = 120          # ~1 hour of windows kept while the server is unreachable
ALERT_FLUSH_SECONDS = 2.0

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def _log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


class Api:
    def __init__(self, server: str, token: str):
        self.server = server.rstrip("/")
        self.token = token

    def post(self, path: str, body: dict, timeout: float = 75.0) -> dict:
        req = urllib.request.Request(
            self.server + path, data=json.dumps(body).encode(), method="POST",
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.token}",
                     "User-Agent": f"nadf-agent/{AGENT_VERSION}"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read() or b"{}")


def _primary_ip() -> str | None:
    """The local IP used for outbound traffic (no packet is actually sent)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except OSError:
        return None


def _interfaces():
    from scapy.all import get_if_list, get_if_addr  # noqa
    out = []
    if platform.system() == "Windows":
        try:
            from scapy.arch.windows import get_windows_if_list
            for i in get_windows_if_list():
                ips = [ip for ip in i.get("ips", []) if "." in ip and not ip.startswith("169.254")]
                if ips:
                    out.append((i.get("name"), ips[0], i.get("description") or ""))
            return out
        except Exception:
            pass
    for name in get_if_list():
        try:
            ip = get_if_addr(name)
        except Exception:
            ip = "0.0.0.0"
        if ip and ip != "0.0.0.0":
            out.append((name, ip, ""))
    return out


def _pick_interface(iface: str | None, local_ip: str | None):
    ifaces = _interfaces()
    if iface:
        match = next((i for i in ifaces if i[0] == iface), None)
        return iface, local_ip or (match[1] if match else None)
    target_ip = local_ip or _primary_ip()
    match = next((i for i in ifaces if i[1] == target_ip), None)
    if match:
        return match[0], match[1]
    if ifaces:
        return ifaces[0][0], local_ip or ifaces[0][1]
    return None, target_ip


class Agent:
    def __init__(self, api: Api, iface: str, local_ip: str, window_seconds: float):
        from app.live.flow_tracker import FlowTracker, PacketRecord, describe_packet
        from app.live.tripwire import Tripwire
        self.api = api
        self.iface = iface
        self.local_ip = local_ip
        self.window_seconds = window_seconds
        self.tracker = FlowTracker(local_ip)
        self._PacketRecord, self._describe = PacketRecord, describe_packet
        self.alerts: deque = deque(maxlen=1000)
        self.tripwire = Tripwire(sink=self._on_alert)
        self.pending_packets: dict[str, deque] = defaultdict(lambda: deque(maxlen=PACKETS_PER_HOST_PER_UPLOAD))
        self.lock = threading.Lock()
        self.backlog: deque = deque(maxlen=MAX_BACKLOG_BATCHES)
        self.packets_seen = 0
        self.error: str | None = None
        self.stop = threading.Event()

    def _on_alert(self, remote_ip, message, severity, detail, ts):
        alert = {"remote_ip": remote_ip, "message": message, "severity": severity, "detail": detail,
                 "timestamp": datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()}
        self.alerts.append(alert)
        _log(f"ALERT [{severity}] {message}")
        return alert

    # -- capture thread ------------------------------------------------------
    def _on_packet(self, pkt):
        from scapy.layers.inet import IP, TCP
        self.packets_seen += 1
        if IP not in pkt or TCP not in pkt:
            return
        ip, tcp = pkt[IP], pkt[TCP]
        if ip.src == self.local_ip:
            direction, remote_ip, local_port, remote_port = "out", ip.dst, int(tcp.sport), int(tcp.dport)
        elif ip.dst == self.local_ip:
            direction, remote_ip, local_port, remote_port = "in", ip.src, int(tcp.dport), int(tcp.sport)
        else:
            return
        flags, now = str(tcp.flags), time.time()
        fields = dict(direction=direction, local_port=local_port, remote_port=remote_port, flags=flags,
                      ttl=int(ip.ttl), win_size=int(tcp.window), pkt_len=len(pkt))
        self.tracker.ingest_tcp(remote_ip=remote_ip, ts=now, **fields)
        rec = self._PacketRecord(ts=now, **fields)
        with self.lock:
            self.pending_packets[remote_ip].append({"remote_ip": remote_ip, "ts": now, **fields,
                                                    "description": self._describe(rec)})
        if direction == "in" and "S" in flags and "A" not in flags:
            self.tripwire.on_inbound_syn(remote_ip=remote_ip, local_port=local_port, ts=now)

    def _sniff(self):
        from scapy.all import sniff
        try:
            sniff(iface=self.iface, filter="tcp", prn=self._on_packet, store=False,
                  stop_filter=lambda p: self.stop.is_set())
        except Exception as e:  # missing Npcap / no admin rights / bad interface
            hint = ("install Npcap (https://npcap.com) and run as Administrator" if platform.system() == "Windows"
                    else "run with sudo (packet capture needs root)")
            self.error = f"capture failed: {e} -- {hint}"
            _log(self.error)
            self.stop.set()

    # -- upload loop ------------------------------------------------------------
    def _send(self, path: str, body: dict) -> bool:
        try:
            self.api.post(path, body)
            return True
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:300]
            _log(f"server rejected {path}: HTTP {e.code} {detail}")
            if e.code == 401:
                _log("the sensor token is invalid or was revoked -- create a new one on the website")
                self.stop.set()
            return e.code < 500  # 4xx: drop the batch, 5xx: keep for retry
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            _log(f"server unreachable ({e}); will retry")
            return False

    def _flush_alerts(self):
        if not self.alerts:
            return
        batch = [self.alerts.popleft() for _ in range(min(len(self.alerts), 200))]
        if not self._send("/agent/alerts", {"alerts": batch}):
            self.alerts.extendleft(reversed(batch))

    def _roll_and_upload(self):
        observed_at = datetime.now(timezone.utc).isoformat()
        feats = self.tracker.roll_window()
        with self.lock:
            packets = [p for q in self.pending_packets.values() for p in q]
            self.pending_packets.clear()
        batch = {"windows": [{"remote_ip": ip, "observed_at": observed_at, "features": f} for ip, f in feats.items()],
                 "packets": packets, "packets_seen": self.packets_seen, "error": self.error}
        self.backlog.append(batch)
        while self.backlog:
            if not self._send("/agent/windows", self.backlog[0]):
                break
            self.backlog.popleft()
        flagged = ", ".join(sorted(feats)) or "none"
        _log(f"window sent: {len(feats)} remote host(s) with inbound activity ({flagged}); "
             f"{self.packets_seen} packets seen so far")

    def run(self):
        threading.Thread(target=self._sniff, daemon=True).start()
        next_window = time.time() + self.window_seconds
        last_alert_flush = 0.0
        while not self.stop.is_set():
            time.sleep(0.5)
            now = time.time()
            if now - last_alert_flush >= ALERT_FLUSH_SECONDS:
                self._flush_alerts()
                last_alert_flush = now
            if now >= next_window:
                self._roll_and_upload()
                next_window += self.window_seconds
        if self.error:
            self._send("/agent/windows", {"windows": [], "packets": [], "packets_seen": self.packets_seen,
                                          "error": self.error})


def main(argv=None):
    p = argparse.ArgumentParser(description="Network Attack Forecasting capture agent")
    p.add_argument("--token", default=os.environ.get("NADF_TOKEN"), help="sensor token from the website (nadf_...)")
    p.add_argument("--server", default=os.environ.get("NADF_SERVER", DEFAULT_SERVER), help="backend API URL")
    p.add_argument("--iface", default=None, help="network interface to capture on (default: auto)")
    p.add_argument("--local-ip", default=None, help="this machine's IP on that interface (default: auto)")
    p.add_argument("--list-interfaces", action="store_true", help="print capture interfaces and exit")
    args = p.parse_args(argv)

    try:
        import scapy  # noqa: F401
        import numpy  # noqa: F401
    except ImportError:
        print("Missing dependencies. Run:  pip install -r requirements.txt", file=sys.stderr)
        return 2

    if args.list_interfaces:
        for name, ip, desc in _interfaces():
            print(f"{name:30s} {ip:16s} {desc}")
        return 0
    if not args.token:
        print("A sensor token is required: create one on the website (Capture -> Add sensor), then run\n"
              "  python nadf_agent.py --token nadf_...", file=sys.stderr)
        return 2
    if not args.server or args.server.startswith("__"):
        print("No server URL: pass --server https://your-backend.onrender.com", file=sys.stderr)
        return 2

    iface, local_ip = _pick_interface(args.iface, args.local_ip)
    if not local_ip:
        print("Could not determine this machine's IP address; pass --iface and --local-ip "
              "(see --list-interfaces).", file=sys.stderr)
        return 2

    api = Api(args.server, args.token)
    _log(f"connecting to {api.server} (first request can take up to a minute if the server was asleep)...")
    try:
        hello = api.post("/agent/hello", {"hostname": socket.gethostname(), "local_ip": local_ip, "iface": iface,
                                          "os": f"{platform.system()} {platform.release()}",
                                          "agent_version": AGENT_VERSION})
    except urllib.error.HTTPError as e:
        print(f"Server refused the agent: HTTP {e.code} {e.read().decode(errors='replace')[:300]}", file=sys.stderr)
        return 1
    except (urllib.error.URLError, OSError) as e:
        print(f"Cannot reach {api.server}: {e}", file=sys.stderr)
        return 1

    window = float(hello.get("window_seconds", 30))
    _log(f"registered as sensor '{hello.get('name')}'. Capturing on {iface} ({local_ip}); "
         f"a window is analysed every {window:.0f}s. Press Ctrl+C to stop.")
    agent = Agent(api, iface, local_ip, window)
    try:
        agent.run()
    except KeyboardInterrupt:
        agent.stop.set()
        _log("stopped.")
    return 1 if agent.error else 0


if __name__ == "__main__":
    sys.exit(main())
