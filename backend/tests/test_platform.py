"""Multi-user platform: auth, sensors + capture-agent uploads, pcap replay,
sample data, workspace isolation, agent download."""
from __future__ import annotations

import io
import time
import zipfile
from datetime import datetime, timedelta, timezone

import jwt
import pytest
from fastapi import HTTPException

from app import auth
from app.config import FEATURE_COLUMNS
from app.live.flow_tracker import FlowTracker
from tests.conftest import as_user

ALICE = as_user("alice")


# ---------------------------------------------------------------------------
# auth
# ---------------------------------------------------------------------------
def _token(key, alg, sub="user-123", aud="authenticated", exp_minutes=5, kid=None):
    payload = {"sub": sub, "aud": aud, "exp": datetime.now(timezone.utc) + timedelta(minutes=exp_minutes),
               "role": "authenticated"}
    headers = {"kid": kid} if kid else None
    return jwt.encode(payload, key, algorithm=alg, headers=headers)


def test_hs256_supabase_token_is_verified_with_the_project_secret():
    cfg = auth.AuthConfig(mode="supabase", supabase_url="https://x.supabase.co", jwt_secret="s3cret" * 6)
    assert auth.verify_supabase_token(_token(cfg.jwt_secret, "HS256"), cfg)["sub"] == "user-123"
    with pytest.raises(HTTPException) as e:
        auth.verify_supabase_token(_token("wrong-secret" * 4, "HS256"), cfg)
    assert e.value.status_code == 401


def test_expired_or_wrong_audience_tokens_are_rejected():
    cfg = auth.AuthConfig(mode="supabase", supabase_url="https://x.supabase.co", jwt_secret="s3cret" * 6)
    for tok in (_token(cfg.jwt_secret, "HS256", exp_minutes=-1), _token(cfg.jwt_secret, "HS256", aud="anon")):
        with pytest.raises(HTTPException):
            auth.verify_supabase_token(tok, cfg)


def test_es256_token_is_verified_against_the_projects_jwks(monkeypatch):
    from cryptography.hazmat.primitives.asymmetric import ec
    private = ec.generate_private_key(ec.SECP256R1())

    class FakeKey:
        key = private.public_key()

    class FakeJwks:
        def get_signing_key_from_jwt(self, token):
            return FakeKey()

    monkeypatch.setattr(auth, "_jwk_client", lambda url: FakeJwks())
    cfg = auth.AuthConfig(mode="supabase", supabase_url="https://x.supabase.co", jwt_secret=None)
    assert auth.verify_supabase_token(_token(private, "ES256", kid="k1"), cfg)["sub"] == "user-123"
    other = ec.generate_private_key(ec.SECP256R1())
    with pytest.raises(HTTPException):
        auth.verify_supabase_token(_token(other, "ES256", kid="k1"), cfg)


def test_supabase_mode_requires_a_bearer_token(client, monkeypatch):
    monkeypatch.setattr(auth, "CONFIG", auth.AuthConfig(mode="supabase", supabase_url="https://x.supabase.co",
                                                        jwt_secret="s3cret" * 6))
    assert client.get("/hosts").status_code == 401
    assert client.get("/hosts", headers=ALICE).status_code == 401  # dev header is ignored outside dev mode
    good = {"Authorization": f"Bearer {_token('s3cret' * 6, 'HS256', sub='u-1')}"}
    assert client.get("/hosts", headers=good).json() == []


def test_dev_auth_is_refused_on_render(monkeypatch):
    monkeypatch.setenv("RENDER", "true")
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("AUTH_MODE", raising=False)
    assert auth.load_config().problem


# ---------------------------------------------------------------------------
# sensors + agent
# ---------------------------------------------------------------------------
def _scan_features(remote="10.0.0.66", local="10.0.0.5", ports=range(20, 60)):
    """Real FEATURE_COLUMNS vector from the real FlowTracker for a SYN scan."""
    tr = FlowTracker(local)
    t = time.time()
    for i, p in enumerate(ports):
        tr.ingest_tcp(remote_ip=remote, direction="in", local_port=p, remote_port=40000 + i, flags="S",
                      ttl=52, win_size=1024, pkt_len=60, ts=t + i * 0.01)
        tr.ingest_tcp(remote_ip=remote, direction="out", local_port=p, remote_port=40000 + i, flags="RA",
                      ttl=64, win_size=0, pkt_len=54, ts=t + i * 0.01 + 0.001)
    return tr.roll_window()[remote]


def _new_sensor(client, headers=ALICE):
    r = client.post("/sensors", json={"name": "laptop"}, headers=headers)
    assert r.status_code == 200
    return r.json()


def test_agent_flow_end_to_end(client):
    sensor = _new_sensor(client)
    token = sensor["token"]
    agent = {"Authorization": f"Bearer {token}"}
    assert client.get("/sensors", headers=ALICE).json()[0].get("token") is None  # token only shown once

    assert client.post("/agent/hello", json={"hostname": "pc", "local_ip": "10.0.0.5"}, headers=agent).status_code == 200
    feats = _scan_features()
    pkt = {"remote_ip": "10.0.0.66", "ts": time.time(), "direction": "in", "local_port": 22, "remote_port": 40000,
           "flags": "S", "ttl": 52, "win_size": 1024, "pkt_len": 60, "description": "SYN to SSH"}
    for _ in range(3):
        r = client.post("/agent/windows", json={"windows": [{"remote_ip": "10.0.0.66", "features": feats}],
                                                "packets": [pkt], "packets_seen": 120}, headers=agent)
        assert r.status_code == 200, r.text
    r = client.post("/agent/alerts", json={"alerts": [{"remote_ip": "10.0.0.66", "message": "probe",
                                                       "severity": "critical", "detail": {"rule": "x"}}]},
                    headers=agent)
    assert r.status_code == 200

    status = client.get("/live/status", headers=ALICE).json()
    assert status["running"] is True and status["packets_seen"] == 120
    assert "live:10.0.0.66" in client.get("/hosts", headers=ALICE).json()
    recent = client.get("/live/recent", headers=ALICE).json()
    assert len(recent) == 3 and recent[0]["predicted_stage"] in client.get("/stage-classes").json()
    assert recent[-1]["warmup"] is True and recent[0]["history_windows_used"] == 3
    assert recent[0]["explanation"]["attention_over_past_windows"]
    assert client.get("/live/packets/10.0.0.66", headers=ALICE).json()[0]["description"] == "SYN to SSH"
    assert client.get("/live/alerts", headers=ALICE).json()[0]["severity"] == "critical"
    assert client.get("/track/live:10.0.0.66", headers=ALICE).json()["steps"][-1]["history_windows_used"] == 3
    assert client.get("/kpis", headers=ALICE).json()["forecasts_generated_today"] == 6  # 3 windows x 2 models

    # isolation: bob sees none of it and cannot use or revoke alice's sensor
    bob = as_user("bob")
    assert client.get("/hosts", headers=bob).json() == []
    assert client.get("/live/recent", headers=bob).json() == []
    assert client.get("/live/status", headers=bob).json()["running"] is False
    assert client.delete(f"/sensors/{sensor['id']}", headers=bob).status_code == 404


def test_agent_rejects_bad_tokens_and_bad_features(client):
    assert client.post("/agent/hello", json={}, headers={"Authorization": "Bearer nadf_nope"}).status_code == 401
    token = _new_sensor(client)["token"]
    agent = {"Authorization": f"Bearer {token}"}
    feats = _scan_features()
    feats.pop(FEATURE_COLUMNS[0])
    r = client.post("/agent/windows", json={"windows": [{"remote_ip": "1.2.3.4", "features": feats}]}, headers=agent)
    assert r.status_code == 422


def test_revoked_sensor_token_stops_working(client):
    s = _new_sensor(client)
    agent = {"Authorization": f"Bearer {s['token']}"}
    assert client.post("/agent/hello", json={}, headers=agent).status_code == 200
    assert client.delete(f"/sensors/{s['id']}", headers=ALICE).status_code == 200
    assert client.post("/agent/hello", json={}, headers=agent).status_code == 401


def test_agent_download_bundles_the_real_capture_code(client):
    r = client.get("/agent/download")
    assert r.status_code == 200
    z = zipfile.ZipFile(io.BytesIO(r.content))
    names = set(z.namelist())
    for f in ("nadf_agent.py", "app/live/flow_tracker.py", "app/live/tripwire.py", "app/config.py"):
        assert f"nadf-agent/{f}" in names
    agent_src = z.read("nadf-agent/nadf_agent.py").decode()
    assert "__NADF_SERVER_URL__" not in agent_src and "http://testserver" in agent_src
    assert "token" not in z.read("nadf-agent/requirements.txt").decode()


# ---------------------------------------------------------------------------
# pcap upload
# ---------------------------------------------------------------------------
def _scan_pcap(tmp_path, attacker="10.0.0.66", victim="10.0.0.5", seconds=95):
    from scapy.all import Ether, IP, TCP, wrpcap
    pkts, t0 = [], 1_700_000_000.0
    for i in range(seconds * 4):
        port = 20 + (i % 80)
        syn = Ether() / IP(src=attacker, dst=victim, ttl=52) / TCP(sport=40000 + i, dport=port, flags="S", window=1024)
        syn.time = t0 + i * 0.25
        rst = Ether() / IP(src=victim, dst=attacker, ttl=64) / TCP(sport=port, dport=40000 + i, flags="RA")
        rst.time = t0 + i * 0.25 + 0.001
        pkts += [syn, rst]
    # some unrelated traffic the victim initiated (must NOT produce a host)
    out = Ether() / IP(src=victim, dst="93.184.216.34") / TCP(sport=50000, dport=443, flags="S")
    out.time = t0 + 1
    pkts.append(out)
    path = tmp_path / "scan.pcap"
    wrpcap(str(path), pkts)
    return path.read_bytes()


def test_pcap_replay_produces_windows_alerts_and_packets(client, tmp_path):
    data = _scan_pcap(tmp_path)
    r = client.post("/upload/pcap", files={"file": ("scan.pcap", data, "application/octet-stream")}, headers=ALICE)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["local_ip"] == "10.0.0.5"  # auto-detected monitored host
    assert body["stats"]["windows"] == 4   # 95 s of capture -> 4 windows of 30 s
    assert [h["host_id"] for h in body["hosts"]] == ["pcap:10.0.0.66"]
    tl = client.get("/host-timeline/pcap:10.0.0.66", headers=ALICE).json()
    assert len(tl) == 4
    assert client.get("/live/packets/pcap:10.0.0.66", headers=ALICE).json()
    assert any(a["detail"]["rule"] == "rapid_multi_port" for a in client.get("/live/alerts", headers=ALICE).json())
    # re-upload replaces rather than duplicates
    client.post("/upload/pcap", files={"file": ("scan.pcap", data, "application/octet-stream")}, headers=ALICE)
    assert len(client.get("/host-timeline/pcap:10.0.0.66", headers=ALICE).json()) == 4


def test_pcap_upload_rejects_garbage(client):
    r = client.post("/upload/pcap", files={"file": ("x.pcap", b"not a capture at all", "application/octet-stream")},
                    headers=ALICE)
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# sample data + workspace
# ---------------------------------------------------------------------------
def test_sample_data_is_fresh_every_time_and_private(client):
    a = client.post("/workspace/sample", json={"attack_hosts": 1, "benign_hosts": 1}, headers=ALICE).json()
    b = client.post("/workspace/sample", json={"attack_hosts": 1, "benign_hosts": 1}, headers=ALICE).json()
    assert a["seed"] != b["seed"]
    hosts = client.get("/hosts", headers=ALICE).json()
    assert len(hosts) == 4 and all(h.startswith("sample-") for h in hosts)
    assert client.get("/hosts", headers=as_user("bob")).json() == []
    attack = next(h["host_id"] for h in a["hosts"] if "attack" in h["host_id"])
    track = client.get(f"/track/{attack}", headers=ALICE).json()
    assert track["summary"]["accuracy_top1"] is not None  # sample data carries labels for scoring


def test_workspace_reset_clears_only_the_caller(client):
    client.post("/workspace/sample", json={"attack_hosts": 1, "benign_hosts": 0}, headers=ALICE)
    client.post("/workspace/sample", json={"attack_hosts": 1, "benign_hosts": 0}, headers=as_user("bob"))
    assert client.delete("/workspace", headers=ALICE).json()["traffic_windows"] == 0
    assert client.get("/hosts", headers=ALICE).json() == []
    assert len(client.get("/hosts", headers=as_user("bob")).json()) == 1


def test_csv_ingest_stores_windows_for_the_caller(client, real_service):
    host = sorted(real_service.labeled_df["host_id"].unique())[0]
    df = real_service.labeled_df[real_service.labeled_df["host_id"] == host]
    csv = df[["host_id", "window_idx"] + FEATURE_COLUMNS].to_csv(index=False).encode()
    r = client.post("/ingest", files={"file": ("t.csv", csv, "text/csv")}, headers=ALICE)
    assert r.status_code == 200
    assert len(client.get(f"/host-timeline/{host}", headers=ALICE).json()) == len(df)


def test_unreachable_database_is_reported_on_health_instead_of_crashing(monkeypatch, real_service):
    """A wrong DATABASE_URL (e.g. Supabase's IPv6-only direct host) must not
    crash the server into a restart loop: /health explains what to fix."""
    import app.api.main as main_module
    from fastapi.testclient import TestClient
    from app import db

    bad = "postgresql://postgres:pw@db.nonexistent-project.supabase.co:5432/postgres"
    monkeypatch.setenv("DATABASE_URL", bad)
    monkeypatch.setattr(main_module, "service", real_service)
    monkeypatch.setattr(real_service, "ready", False)
    monkeypatch.setattr(real_service, "load", lambda: (db.init_db(), None)[1])
    with TestClient(main_module.app) as c:  # runs the startup hook
        body = c.get("/health").json()
        assert body["status"] == "not_ready"
        assert "database connection failed" in body["problem"]
        assert "Session pooler" in body["problem"]
        assert "pw@" not in body["problem"]  # password never echoed
        assert c.get("/hosts", headers=ALICE).status_code == 503
    db.configure("sqlite://")
