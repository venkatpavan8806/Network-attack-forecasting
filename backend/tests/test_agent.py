"""Capture agent flow over HTTP: add a sensor on the website, the agent
connects and reports interfaces, the website's Start/Stop button reaches the
agent, captured windows/packets/alerts land in the user's account."""
from __future__ import annotations

import io
import time
import zipfile

import pytest
from fastapi.testclient import TestClient

from app import db
import app.api.main as main_module
from app.live.flow_tracker import FlowTracker
from tests.test_api import real_service  # noqa: F401  (session-trained service fixture)


@pytest.fixture()
def client(real_service, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setattr(main_module, "service", real_service)
    db.configure(f"sqlite:///{(tmp_path / 'agent.sqlite3').as_posix()}")
    db.init_db()
    real_service._user_cache.clear()
    real_service._report_cache.clear()
    return TestClient(main_module.app)


def _scan_features(remote="10.0.0.66", local="10.0.0.5", ports=range(20, 60)):
    """A real FEATURE_COLUMNS vector from the real FlowTracker for a SYN scan."""
    tr = FlowTracker(local)
    t = time.time()
    for i, p in enumerate(ports):
        tr.ingest_tcp(remote_ip=remote, direction="in", local_port=p, remote_port=40000 + i, flags="S",
                      ttl=52, win_size=1024, pkt_len=60, ts=t + i * 0.01)
        tr.ingest_tcp(remote_ip=remote, direction="out", local_port=p, remote_port=40000 + i, flags="RA",
                      ttl=64, win_size=0, pkt_len=54, ts=t + i * 0.01 + 0.001)
    return tr.roll_window()[remote]


def test_agent_flow_start_capture_stop(client):
    sensor = client.post("/sensors", json={"name": "my-laptop"}).json()
    agent = {"Authorization": f"Bearer {sensor['token']}"}
    assert "token" not in client.get("/sensors").json()[0]  # shown only once

    hello = client.post("/agent/hello", headers=agent, json={
        "hostname": "laptop", "os": "Windows 11", "interfaces": [{"name": "Wi-Fi", "ip": "10.0.0.5", "description": "x"}]})
    assert hello.status_code == 200 and hello.json()["window_seconds"] == 30
    assert client.get("/live/interfaces").json() == [{"name": "Wi-Fi", "description": "x", "ip": "10.0.0.5"}]
    assert client.post("/agent/control", headers=agent, json={}).json()["capture"] is False

    # Start button on the website -> the agent is told to capture on that interface
    st = client.post("/live/start", json={"iface": "Wi-Fi", "local_ip": "10.0.0.5"}).json()
    assert st["running"] is True and st["mode"] == "agent"
    ctl = client.post("/agent/control", headers=agent, json={"capturing": True, "packets_seen": 40, "alerts": [
        {"remote_ip": "10.0.0.66", "message": "Rapid multi-port probing", "severity": "critical",
         "detail": {"rule": "rapid_multi_port"}}]}).json()
    assert ctl == {"capture": True, "iface": "Wi-Fi", "local_ip": "10.0.0.5", "window_seconds": 30}
    assert client.get("/live/alerts").json()[0]["severity"] == "critical"

    # a captured window
    pkt = {"remote_ip": "10.0.0.66", "timestamp": time.time(), "direction": "in", "local_port": 22,
           "remote_port": 40000, "flags": "S", "ttl": 52, "win_size": 1024, "pkt_len": 60, "description": "SYN"}
    r = client.post("/agent/windows", headers=agent, json={
        "windows": [{"remote_ip": "10.0.0.66", "features": _scan_features()}], "packets": [pkt], "packets_seen": 80})
    assert r.json() == {"accepted": 1}
    recent = client.get("/live/recent", params={"limit": 200}).json()
    assert recent[0]["host_id"] == "live:10.0.0.66" and recent[0]["warmup"] is True
    assert "live:10.0.0.66" in client.get("/hosts").json()  # live capture feeds the Forecasts host list
    assert client.get("/live/packets/10.0.0.66").json()[0]["description"] == "SYN"
    assert client.get("/live/status").json()["packets_seen"] == 80

    # Stop button
    assert client.post("/live/stop").json()["running"] is False
    assert client.post("/agent/control", headers=agent, json={}).json()["capture"] is False


def test_agent_rejects_bad_tokens_and_bad_features(client):
    assert client.post("/agent/hello", json={}, headers={"Authorization": "Bearer nadf_nope"}).status_code == 401
    token = client.post("/sensors", json={"name": "x"}).json()["token"]
    feats = _scan_features()
    feats.pop("flow_count")
    r = client.post("/agent/windows", headers={"Authorization": f"Bearer {token}"},
                    json={"windows": [{"remote_ip": "1.2.3.4", "features": feats}]})
    assert r.status_code == 422


def test_removed_sensor_token_stops_working(client):
    s = client.post("/sensors", json={"name": "x"}).json()
    agent = {"Authorization": f"Bearer {s['token']}"}
    assert client.post("/agent/hello", headers=agent, json={}).status_code == 200
    assert client.delete(f"/sensors/{s['id']}").status_code == 200
    assert client.post("/agent/control", headers=agent, json={}).status_code == 401


def test_no_agent_and_no_local_capture_explains_how_to_start(client, monkeypatch):
    monkeypatch.setattr(main_module, "_local_interfaces", lambda: [])  # e.g. backend on a Linux server
    assert client.get("/live/interfaces").json() == []
    r = client.post("/live/start", json={"iface": "eth0", "local_ip": "1.1.1.1"})
    assert r.status_code == 409 and "agent" in r.json()["detail"]


def test_agent_download_contains_the_capture_code_and_this_server(client):
    r = client.get("/agent/download")
    z = zipfile.ZipFile(io.BytesIO(r.content))
    names = set(z.namelist())
    for f in ("nadf_agent.py", "app/live/flow_tracker.py", "app/live/tripwire.py", "app/config.py", "requirements.txt"):
        assert f"nadf-agent/{f}" in names
    src = z.read("nadf-agent/nadf_agent.py").decode()
    assert "http://testserver" in src and "__NADF_SERVER_URL__" not in src
