"""Live packet capture -> windowed feature extraction -> real inference,
using the exact same trained scaler/LSTM/baseline as the CSV-replay path.

Runs as two background threads inside the FastAPI process:
  - a scapy sniff loop that feeds raw packets into a FlowTracker
  - a "windower" loop that, every WINDOW_SECONDS, rolls the tracker's
    current window into per-remote-host feature vectors, runs real
    inference once enough windows have accumulated for that host, and
    logs the result the same way a CSV-replay forecast would.

This is genuinely live: no synthetic data is involved once capture starts.
See app/live/flow_tracker.py for the documented vantage-point adaptation
(monitoring from one host's own NIC, not the attacker's outbound view the
training data was built from).
"""
from __future__ import annotations

import threading
import time
from collections import deque
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn.functional as F

from app.config import FEATURE_COLUMNS, SEQ_LEN, ROLLOUT_K, WINDOW_SECONDS
from app.live.flow_tracker import FlowTracker
from app.live.tripwire import tripwire
from app.models.lstm_world_model import infiltration_probability
from app.models.attack_mapping import map_stage
from app.explain.attention import explain_prediction
from app import db


class LiveCaptureManager:
    def __init__(self):
        self.running = False
        self.iface: str | None = None
        self.local_ip: str | None = None
        self.tracker: FlowTracker | None = None
        self._sniff_thread: threading.Thread | None = None
        self._window_thread: threading.Thread | None = None
        self.history: dict[str, deque] = {}
        self.window_counter: dict[str, int] = {}
        self.recent_predictions: deque = deque(maxlen=100)
        self.started_at: str | None = None
        self.packets_seen = 0
        self.error: str | None = None

    def status(self) -> dict:
        return {
            "running": self.running,
            "iface": self.iface,
            "local_ip": self.local_ip,
            "started_at": self.started_at,
            "packets_seen": self.packets_seen,
            "hosts_seen": len(self.history),
            "error": self.error,
        }

    def start(self, iface: str, local_ip: str):
        if self.running:
            raise RuntimeError("live capture is already running")
        self.iface = iface
        self.local_ip = local_ip
        self.tracker = FlowTracker(local_ip)
        self.history = {}
        self.window_counter = {}
        self.packets_seen = 0
        self.error = None
        self.recent_predictions.clear()  # a fresh session must not mix in a previous session's log entries
        tripwire.reset()
        self.running = True
        self.started_at = datetime.now(timezone.utc).isoformat()

        self._sniff_thread = threading.Thread(target=self._sniff_loop, daemon=True)
        self._window_thread = threading.Thread(target=self._window_loop, daemon=True)
        self._sniff_thread.start()
        self._window_thread.start()

    def stop(self):
        self.running = False

    def _sniff_loop(self):
        try:
            from scapy.all import sniff, IP, TCP
        except Exception as e:
            self.error = f"scapy unavailable: {e}"
            self.running = False
            return

        def on_packet(pkt):
            if not self.running:
                return
            self.packets_seen += 1
            if IP not in pkt or TCP not in pkt:
                return
            ip_layer = pkt[IP]
            tcp_layer = pkt[TCP]
            src, dst = ip_layer.src, ip_layer.dst
            if src == self.local_ip:
                direction, remote_ip = "out", dst
                local_port, remote_port = tcp_layer.sport, tcp_layer.dport
            elif dst == self.local_ip:
                direction, remote_ip = "in", src
                local_port, remote_port = tcp_layer.dport, tcp_layer.sport
            else:
                return  # not traffic to/from this host -- shouldn't happen in non-promiscuous capture

            flags = str(tcp_layer.flags)
            now = time.time()
            self.tracker.ingest_tcp(
                remote_ip=remote_ip, direction=direction, local_port=int(local_port),
                remote_port=int(remote_port), flags=flags, ttl=int(ip_layer.ttl),
                win_size=int(tcp_layer.window), pkt_len=len(pkt), ts=now,
            )

            # fast tripwire: evaluated synchronously, right here, on every
            # inbound SYN -- real millisecond latency, independent of the
            # WINDOW_SECONDS/SEQ_LEN cadence the ML forecast needs
            if direction == "in" and "S" in flags and "A" not in flags:
                tripwire.on_inbound_syn(remote_ip=remote_ip, local_port=int(local_port), ts=now)

        try:
            sniff(iface=self.iface, filter="tcp", prn=on_packet, store=False,
                  stop_filter=lambda p: not self.running)
        except Exception as e:
            self.error = f"capture error: {e}"
            self.running = False

    def _window_loop(self):
        while self.running:
            time.sleep(WINDOW_SECONDS)
            if not self.running or self.tracker is None:
                return
            try:
                self._process_window()
            except Exception as e:
                self.error = f"window processing error: {e}"

    def _process_window(self):
        from app.inference.service import service
        if not service.ready:
            return

        feats_by_remote = self.tracker.roll_window()
        for remote_ip, feats in feats_by_remote.items():
            row = np.array([feats[c] for c in FEATURE_COLUMNS], dtype=np.float32)
            hist = self.history.setdefault(remote_ip, deque(maxlen=SEQ_LEN))
            hist.append(row)
            self.window_counter[remote_ip] = self.window_counter.get(remote_ip, 0) + 1
            window_idx = self.window_counter[remote_ip]
            host_id = f"live:{remote_ip}"

            entry = {
                "host_id": host_id,
                "window_idx": window_idx,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "raw_features": feats,
                "predicted_stage": None,
                "infiltration_probability_world_model": None,
                "infiltration_probability_baseline": None,
            }

            if len(hist) == SEQ_LEN:
                window_raw = np.stack(list(hist))
                window_scaled = service.scaler.transform(window_raw).astype(np.float32)
                explanation = explain_prediction(service.model, window_scaled)
                baseline_prob = float(service.baseline.predict_proba(window_scaled[-1:])[0, 1])
                predicted_stage = max(explanation["stage_probabilities"].items(), key=lambda kv: kv[1])[0]

                entry["predicted_stage"] = predicted_stage
                entry["attack_mapping"] = map_stage(predicted_stage)
                entry["infiltration_probability_world_model"] = explanation["infiltration_probability"]
                entry["infiltration_probability_baseline"] = round(baseline_prob, 4)
                entry["stage_probabilities"] = explanation["stage_probabilities"]
                entry["explanation"] = {
                    "attention_over_past_windows": explanation["attention_over_past_windows"],
                    "top_contributors": explanation["top_contributors"],
                }

                db.log_inference(host_id, window_idx, "world_model_lstm", predicted_stage,
                                  explanation["infiltration_probability"], None, None, source="live_capture")
                db.log_inference(host_id, window_idx, "baseline_logreg", None,
                                  baseline_prob, None, None, source="live_capture")

            self.recent_predictions.append(entry)


live_capture = LiveCaptureManager()
