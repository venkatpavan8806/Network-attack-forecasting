"""Network Attack Forecasting -- capture agent.

Run this on the computer you want to monitor and leave it running. It:
  1. tells the website which network interfaces this computer has,
  2. waits for you to press "Start live capture" on the website,
  3. captures TCP packets to/from this computer on the interface you picked,
     turns every 30 seconds of them into the model's window features
     (app/live/flow_tracker.py -- the same code the backend uses), and sends
     those features + packet headers + instant alerts to YOUR account,
  4. stops when you press "Stop capture".
Packet payloads never leave this computer.

    python nadf_agent.py --token nadf_xxxxxxxx

Needs Python 3.9+ and `pip install -r requirements.txt`, plus capture rights:
  Windows: install Npcap (https://npcap.com) and run the terminal as Administrator
  Linux / macOS: run with sudo
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
DEFAULT_SERVER = "__NADF_SERVER_URL__"  # filled in when downloaded from the website
POLL_SECONDS = 3.0                      # how often the agent checks the Start/Stop button
PACKETS_PER_HOST_PER_UPLOAD = 100

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


class Api:
    def __init__(self, server: str, token: str):
        self.server, self.token = server.rstrip("/"), token

    def call(self, method: str, path: str, body: dict | None = None, timeout: float = 75.0) -> dict:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.server + path, data=data, method=method, headers={
            "Content-Type": "application/json", "Authorization": f"Bearer {self.token}",
            "User-Agent": f"nadf-agent/{AGENT_VERSION}"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read() or b"{}")


def list_interfaces() -> list[dict]:
    """[{name, ip, description}] for interfaces with an IPv4 address."""
    out = []
    if platform.system() == "Windows":
        try:
            from scapy.arch.windows import get_windows_if_list
            for i in get_windows_if_list():
                ips = [ip for ip in i.get("ips", []) if "." in ip and not ip.startswith("169.254")]
                if ips:
                    out.append({"name": i.get("name"), "ip": ips[0], "description": i.get("description") or ""})
            return out
        except Exception:
            pass
    from scapy.all import get_if_list, get_if_addr
    for name in get_if_list():
        try:
            ip = get_if_addr(name)
        except Exception:
            continue
        if ip and ip not in ("0.0.0.0",) and not ip.startswith("127."):
            out.append({"name": name, "ip": ip, "description": ""})
    return out


class Capture:
    """One capture session on one interface (started/stopped from the website)."""

    def __init__(self, iface: str, local_ip: str):
        from app.live.flow_tracker import FlowTracker, PacketRecord, describe_packet
        from app.live.tripwire import Tripwire
        self.iface, self.local_ip = iface, local_ip
        self.tracker = FlowTracker(local_ip)
        self.tripwire = Tripwire()  # no database here: fired alerts are uploaded by the agent
        self._PacketRecord, self._describe = PacketRecord, describe_packet
        self.alerts: deque = deque(maxlen=1000)
        self.packets: dict[str, deque] = defaultdict(lambda: deque(maxlen=PACKETS_PER_HOST_PER_UPLOAD))
        self.lock = threading.Lock()
        self.packets_seen = 0
        self.error: str | None = None
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._sniff, daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()

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
            self.packets[remote_ip].append({"remote_ip": remote_ip, "timestamp": now, **fields,
                                            "description": self._describe(rec)})
        if direction == "in" and "S" in flags and "A" not in flags:
            for a in self.tripwire.on_inbound_syn(remote_ip=remote_ip, local_port=local_port, ts=now):
                self.alerts.append({**a, "timestamp": datetime.now(timezone.utc).isoformat()})
                log(f"ALERT [{a['severity']}] {a['message']}")

    def _sniff(self):
        from scapy.all import sniff
        try:
            sniff(iface=self.iface, filter="tcp", prn=self._on_packet, store=False,
                  stop_filter=lambda p: self.stop_event.is_set())
        except Exception as e:
            hint = ("install Npcap (https://npcap.com) and run the terminal as Administrator"
                    if platform.system() == "Windows" else "run the agent with sudo")
            self.error = f"capture failed on '{self.iface}': {e} -- {hint}"
            log(self.error)

    def take_window(self) -> dict:
        feats = self.tracker.roll_window()
        with self.lock:
            packets = [p for q in self.packets.values() for p in q]
            self.packets.clear()
        return {"windows": [{"remote_ip": ip, "features": f} for ip, f in feats.items()],
                "packets": packets, "packets_seen": self.packets_seen}

    def take_alerts(self) -> list[dict]:
        out = []
        while self.alerts and len(out) < 200:
            out.append(self.alerts.popleft())
        return out


def run(api: Api):
    hello = api.call("POST", "/agent/hello", {
        "hostname": socket.gethostname(), "os": f"{platform.system()} {platform.release()}",
        "agent_version": AGENT_VERSION, "interfaces": list_interfaces()})
    window_seconds = float(hello.get("window_seconds", 30))
    log(f"connected as sensor '{hello.get('name')}'. Open the website's Live Capture panel and press "
        f"\"Start live capture\". Leave this window open (Ctrl+C to quit).")

    capture: Capture | None = None
    next_window = 0.0
    backlog: deque = deque(maxlen=120)  # windows kept while the server is unreachable
    while True:
        status = {"capturing": capture is not None and capture.error is None,
                  "packets_seen": capture.packets_seen if capture else 0,
                  "error": capture.error if capture else None,
                  "alerts": capture.take_alerts() if capture else []}
        try:
            ctl = api.call("POST", "/agent/control", status)
        except urllib.error.HTTPError as e:
            if e.code == 401:
                log("this agent's token was revoked on the website -- add a new sensor and restart the agent")
                return 1
            log(f"server error {e.code}; retrying")
            time.sleep(POLL_SECONDS)
            continue
        except (urllib.error.URLError, OSError) as e:
            log(f"server unreachable ({e}); retrying")
            if capture and status["alerts"]:
                capture.alerts.extendleft(reversed(status["alerts"]))
            time.sleep(POLL_SECONDS)
            continue

        want = bool(ctl.get("capture"))
        iface, local_ip = ctl.get("iface"), ctl.get("local_ip")
        if want and capture is not None and (capture.iface != iface or capture.error):
            capture.stop()
            capture = None
        if want and capture is None and iface and local_ip:
            log(f"Start pressed on the website: capturing on {iface} ({local_ip})")
            capture = Capture(iface, local_ip)
            capture.start()
            next_window = time.time() + window_seconds
        elif not want and capture is not None:
            log("Stop pressed on the website: capture stopped")
            capture.stop()
            capture = None

        if capture is not None and time.time() >= next_window:
            backlog.append(capture.take_window())
            next_window += window_seconds
            while backlog:
                try:
                    res = api.call("POST", "/agent/windows", backlog[0])
                except (urllib.error.URLError, OSError):
                    break
                backlog.popleft()
                log(f"window sent: {res.get('accepted', 0)} remote host(s) analysed, "
                    f"{capture.packets_seen if capture else 0} packets seen")
        time.sleep(POLL_SECONDS)


def main(argv=None):
    p = argparse.ArgumentParser(description="Network Attack Forecasting capture agent")
    p.add_argument("--token", default=os.environ.get("NADF_TOKEN"), help="sensor token from the website (nadf_...)")
    p.add_argument("--server", default=os.environ.get("NADF_SERVER", DEFAULT_SERVER), help="backend URL")
    p.add_argument("--list-interfaces", action="store_true", help="print this computer's interfaces and exit")
    args = p.parse_args(argv)
    try:
        import scapy  # noqa: F401
        import numpy  # noqa: F401
    except ImportError:
        print("Missing dependencies -- run:  pip install -r requirements.txt", file=sys.stderr)
        return 2
    if args.list_interfaces:
        for i in list_interfaces():
            print(f"{i['name']:32s} {i['ip']:16s} {i['description']}")
        return 0
    if not args.token:
        print("Pass the token shown on the website:  python nadf_agent.py --token nadf_...", file=sys.stderr)
        return 2
    if not args.server or args.server.startswith("__"):
        print("Pass the backend URL:  --server https://your-backend.onrender.com", file=sys.stderr)
        return 2
    api = Api(args.server, args.token)
    log(f"connecting to {api.server} (can take up to a minute if the server was asleep)...")
    try:
        return run(api) or 0
    except urllib.error.HTTPError as e:
        print(f"Server refused the agent: HTTP {e.code} {e.read().decode(errors='replace')[:300]}", file=sys.stderr)
        return 1
    except (urllib.error.URLError, OSError) as e:
        print(f"Cannot reach {api.server}: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        log("agent stopped")
        return 0


if __name__ == "__main__":
    sys.exit(main())
