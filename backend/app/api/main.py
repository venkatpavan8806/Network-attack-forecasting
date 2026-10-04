"""FastAPI application.

Multi-user: every endpoint that touches traffic, predictions, alerts or
sensors resolves the caller with `Depends(current_user)` (a verified
Supabase login -- see app/auth.py) and only ever reads/writes that user's
rows (app/db.py). Capture agents authenticate with a per-sensor token
(`Depends(current_sensor)`). Model reports (benchmark, calibration, ...)
describe the trained model itself and are public.

Every response is either a genuine model inference result on the caller's
own data or a report computed by app/train.py from an actual training run;
nothing here is a hard-coded placeholder, and there is no shared demo data.
"""
from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path

import pandas as pd
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field

from app import auth, db
from app.auth import current_sensor, current_user
from app.config import STAGE_CLASSES, WINDOW_SECONDS, BACKEND_DIR
from app.features.extraction import FeatureValidationError
from app.inference.service import service, ArtifactsNotReadyError
from app.models.attack_mapping import all_mappings, map_stage

MAX_UPLOAD_MB = float(os.environ.get("MAX_UPLOAD_MB", "25"))
MAX_AGENT_WINDOWS = 500
MAX_AGENT_PACKETS = 5000
MAX_AGENT_ALERTS = 200

app = FastAPI(title="Network Attack Forecasting API", version="1.0.0")

_origins = [o.strip() for o in os.environ.get("CORS_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_origin_regex=os.environ.get("CORS_ORIGIN_REGEX") or None,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    db.configure()
    try:
        service.load()
    except ArtifactsNotReadyError as e:
        # Server still starts so /health reports the real reason instead of crashing silently.
        print(f"[startup] {e}")
    if auth.CONFIG.problem:
        print(f"[startup] AUTH PROBLEM: {auth.CONFIG.problem}")


def _require_ready():
    if not service.ready:
        raise HTTPException(
            status_code=503,
            detail="Model artifacts not trained yet. Run `python -m app.train` in the backend directory, then restart the API.",
        )


def _not_found_or_bad(e: ValueError):
    msg = str(e)
    return HTTPException(status_code=404 if ("unknown host_id" in msg or "no windows" in msg) else 400, detail=msg)


async def _read_upload(file: UploadFile) -> bytes:
    limit = int(MAX_UPLOAD_MB * 1024 * 1024)
    content = await file.read(limit + 1)
    if len(content) > limit:
        raise HTTPException(status_code=413, detail=f"file is larger than {MAX_UPLOAD_MB:g} MB")
    return content


# ---------------------------------------------------------------------------
# health / identity
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok" if service.ready and not auth.CONFIG.problem else "not_ready",
        "ready": service.ready,
        "auth_mode": auth.CONFIG.mode,
        "database": db.engine().dialect.name,
        "problem": auth.CONFIG.problem,
    }


@app.get("/me")
def me(user: str = Depends(current_user)):
    return {"user_id": user, "auth_mode": auth.CONFIG.mode, "workspace": db.workspace_counts(user)}


# ---------------------------------------------------------------------------
# dashboard
# ---------------------------------------------------------------------------
@app.get("/kpis")
def kpis(user: str = Depends(current_user)):
    _require_ready()
    lead_time = service.lead_time_report() or {}
    return {
        "hosts_monitored": db.count_hosts(user),
        "forecasts_generated_today": db.count_forecasts_today(user),
        "high_risk_trajectories": db.count_high_risk_hosts(user, threshold=0.5),
        "median_lead_time_minutes": lead_time.get("median_lead_time_minutes_all_hosts"),
    }


@app.get("/highest-risk-host")
def highest_risk_host(user: str = Depends(current_user)):
    _require_ready()
    row = db.highest_risk_host(user)
    if row is None:
        return None
    mapping = map_stage(row["predicted_stage"]) if row.get("predicted_stage") else None
    return {**row, "attack_mapping": mapping}


@app.get("/hosts")
def hosts(user: str = Depends(current_user)):
    _require_ready()
    return service.list_hosts(user)


@app.get("/hosts/details")
def hosts_details(user: str = Depends(current_user)):
    return db.list_hosts(user)


@app.delete("/hosts/{host_id}")
def delete_host(host_id: str, user: str = Depends(current_user)):
    n = db.delete_hosts(user, [host_id])
    if n == 0:
        raise HTTPException(status_code=404, detail=f"unknown host_id: {host_id}")
    return {"deleted_windows": n}


@app.get("/live/forecastable-hosts")
def live_forecastable_hosts(user: str = Depends(current_user)):
    """The caller's agent-captured (live) hosts."""
    _require_ready()
    return service.live_hosts_with_predictions(user)


@app.get("/forecast/{host_id}")
def forecast(host_id: str, at_window_idx: int | None = None, user: str = Depends(current_user)):
    _require_ready()
    try:
        return service.forecast_host(user, host_id, at_window_idx=at_window_idx)
    except ValueError as e:
        raise _not_found_or_bad(e)


@app.get("/forecast/{host_id}/branches")
def forecast_branches(host_id: str, at_window_idx: int | None = None, depth: int | None = None,
                      branch_factor: int | None = None, user: str = Depends(current_user)):
    """K-step forecast as a branching attack-path tree, each node MITRE-mapped
    -- see app/inference/service.py:branching_forecast."""
    _require_ready()
    try:
        return service.branching_forecast(user, host_id, at_window_idx=at_window_idx, depth=depth,
                                          branch_factor=branch_factor)
    except ValueError as e:
        raise _not_found_or_bad(e)


@app.get("/track/{host_id}")
def track_attacker(host_id: str, at_window_idx: int | None = None, user: str = Depends(current_user)):
    """Step-by-step attacker tracking: a next-step prediction after EVERY
    window from the host's first one, the attack path recognised so far, and
    the next 1/2/3 moves -- see app/tracking/step_tracker.py."""
    _require_ready()
    try:
        return service.track_attacker(user, host_id, at_window_idx=at_window_idx)
    except ValueError as e:
        raise _not_found_or_bad(e)


@app.get("/host-timeline/{host_id}")
def host_timeline(host_id: str, user: str = Depends(current_user)):
    """Windows + labels (when the data has them) for one of the caller's hosts."""
    _require_ready()
    rows = db.host_timeline(user, host_id)
    if not rows:
        raise HTTPException(status_code=404, detail=f"unknown host_id: {host_id}")
    return rows


@app.get("/mitigations")
def mitigations():
    _require_ready()
    return service.available_mitigations()


@app.get("/counterfactual/{host_id}")
def counterfactual(host_id: str, mitigation_id: str = "isolate_host", at_window_idx: int | None = None,
                   user: str = Depends(current_user)):
    """Digital-twin what-if: compares the world model's predicted
    trajectory with vs. without a named mitigation applied to the cloned
    Digital Twin network state. Safe sandbox simulation -- real_network_touched: false."""
    _require_ready()
    try:
        return service.run_counterfactual(user, host_id, mitigation_id, at_window_idx=at_window_idx)
    except ValueError as e:
        raise _not_found_or_bad(e)


class DigitalTwinSimulateRequest(BaseModel):
    host_id: str
    mitigation_id: str = "isolate_host"
    at_window_idx: int | None = None


@app.get("/digital-twin/state/{host_id}")
def digital_twin_state(host_id: str, user: str = Depends(current_user)):
    """Returns the current logical Digital Twin network state, services, and firewall policy."""
    _require_ready()
    if db.last_window_idx(user, host_id) is None:
        raise HTTPException(status_code=404, detail=f"unknown host_id: {host_id}")
    try:
        from app.simulation.network_twin import build_initial_twin_network
        twin = build_initial_twin_network(host_id, is_live=host_id.startswith(("live:", "pcap:")))
        return twin.to_dict()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/digital-twin/simulate")
def digital_twin_simulate(req: DigitalTwinSimulateRequest, user: str = Depends(current_user)):
    """Runs a safe sandbox simulation on a cloned Digital Twin network state."""
    _require_ready()
    try:
        return service.run_counterfactual(user, req.host_id, req.mitigation_id, at_window_idx=req.at_window_idx)
    except ValueError as e:
        raise _not_found_or_bad(e)


@app.get("/shap/{host_id}")
def shap_explanation(host_id: str, at_window_idx: int | None = None, user: str = Depends(current_user)):
    """SHAP on the baseline next to attention + saliency on the LSTM, for the
    same host/window. See app/explain/shap_baseline.py and app/explain/attention.py."""
    _require_ready()
    try:
        return service.explain_shap(user, host_id, at_window_idx=at_window_idx)
    except ValueError as e:
        raise _not_found_or_bad(e)
    except ArtifactsNotReadyError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/defense/{host_id}")
def defense_advice(host_id: str, at_window_idx: int | None = None, user: str = Depends(current_user)):
    """Ranked defensive recommendation computed from real counterfactual
    rollouts of the trained world model (every mitigation vs. no mitigation),
    plus the ATT&CK-mapped manual playbook for the stage the model expects.
    Decision support only -- nothing is applied to any network."""
    _require_ready()
    try:
        return service.defense_advice(user, host_id, at_window_idx=at_window_idx)
    except ValueError as e:
        raise _not_found_or_bad(e)
    except ArtifactsNotReadyError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/attack-stage-breakdown")
def attack_stage_breakdown(user: str = Depends(current_user)):
    _require_ready()
    return db.stage_breakdown(user)


@app.get("/forecast-log")
def forecast_log(limit: int = 25, user: str = Depends(current_user)):
    _require_ready()
    return db.recent_forecast_log(user, limit=min(limit, 500))


# ---------------------------------------------------------------------------
# data in: CSV, pcap, sample data
# ---------------------------------------------------------------------------
class SandboxTestResponse(BaseModel):
    outcome: str
    errors: list[str]
    rows: int
    hosts: int


@app.post("/sandbox/test", response_model=SandboxTestResponse)
async def sandbox_test(file: UploadFile = File(...), user: str = Depends(current_user)):
    """Validates an uploaded telemetry CSV without storing anything."""
    _require_ready()
    content = await _read_upload(file)
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        return SandboxTestResponse(outcome="failure", errors=[f"could not parse CSV: {e}"], rows=0, hosts=0)
    errors = service.validate_telemetry_csv(df)
    n_hosts = df["host_id"].nunique() if "host_id" in df.columns else 0
    return SandboxTestResponse(outcome="failure" if errors else "success", errors=errors, rows=len(df),
                               hosts=int(n_hosts))


@app.post("/ingest")
async def ingest(file: UploadFile = File(...), user: str = Depends(current_user)):
    """Stores every window of an uploaded telemetry CSV in the caller's
    workspace and returns a real forecast for each host's last window.
    422 on genuinely malformed input."""
    _require_ready()
    content = await _read_upload(file)
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"could not parse CSV: {e}")
    try:
        return service.ingest_csv(user, df)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/upload/pcap")
async def upload_pcap(file: UploadFile = File(...), local_ip: str | None = Form(default=None),
                      user: str = Depends(current_user)):
    """Replays a Wireshark/tcpdump capture (.pcap / .pcapng) through the same
    feature extraction + tripwire as the live agent; each remote host
    becomes `pcap:<ip>` with a prediction after every 30-second window."""
    _require_ready()
    from app.ingest.pcap import PcapError
    content = await _read_upload(file)
    try:
        return service.ingest_pcap(user, content, local_ip=(local_ip or "").strip() or None)
    except PcapError as e:
        raise HTTPException(status_code=422, detail=str(e))


class SampleRequest(BaseModel):
    attack_hosts: int = Field(default=2, ge=0, le=5)
    benign_hosts: int = Field(default=2, ge=0, le=5)


@app.post("/workspace/sample")
def workspace_sample(req: SampleRequest | None = None, user: str = Depends(current_user)):
    """Generates FRESH synthetic traffic (new random seed every call) into
    the caller's workspace -- for trying the dashboard without a sensor."""
    _require_ready()
    req = req or SampleRequest()
    if req.attack_hosts + req.benign_hosts == 0:
        raise HTTPException(status_code=400, detail="ask for at least one host")
    return service.generate_sample_data(user, n_attack=req.attack_hosts, n_benign=req.benign_hosts)


@app.get("/workspace")
def workspace(user: str = Depends(current_user)):
    return db.workspace_counts(user)


@app.delete("/workspace")
def reset_workspace(include_sensors: bool = False, user: str = Depends(current_user)):
    db.reset_workspace(user, include_sensors=include_sensors)
    return db.workspace_counts(user)


# ---------------------------------------------------------------------------
# sensors (capture agents) -- managed by the website user
# ---------------------------------------------------------------------------
class SensorCreate(BaseModel):
    name: str = Field(default="my-computer", min_length=1, max_length=100)


@app.get("/sensors")
def list_sensors(user: str = Depends(current_user)):
    return db.list_sensors(user)


@app.post("/sensors")
def create_sensor(req: SensorCreate, user: str = Depends(current_user)):
    """Creates a sensor and returns its token ONCE (only a hash is stored)."""
    if len(db.list_sensors(user)) >= 10:
        raise HTTPException(status_code=400, detail="at most 10 sensors per account -- revoke one first")
    sensor, token = db.create_sensor(user, req.name.strip())
    return {**sensor, "token": token}


@app.delete("/sensors/{sensor_id}")
def delete_sensor(sensor_id: str, user: str = Depends(current_user)):
    if not db.delete_sensor(user, sensor_id):
        raise HTTPException(status_code=404, detail="unknown sensor")
    return {"deleted": True}


def _public_api_url(request: Request) -> str:
    explicit = os.environ.get("PUBLIC_API_URL")
    if explicit:
        return explicit.rstrip("/")
    proto = request.headers.get("x-forwarded-proto", request.url.scheme)
    host = request.headers.get("x-forwarded-host", request.headers.get("host", request.url.netloc))
    return f"{proto}://{host}"


AGENT_FILES = {
    "nadf_agent.py": "agent/nadf_agent.py",
    "README.txt": "agent/README.txt",
    "requirements.txt": "agent/requirements.txt",
    "app/__init__.py": "app/__init__.py",
    "app/config.py": "app/config.py",
    "app/live/__init__.py": "app/live/__init__.py",
    "app/live/flow_tracker.py": "app/live/flow_tracker.py",
    "app/live/tripwire.py": "app/live/tripwire.py",
}


@app.get("/agent/download")
def agent_download(request: Request):
    """The capture agent as a zip, built from this backend's own source (the
    same flow_tracker.py / tripwire.py the server uses), with this server's
    URL pre-filled. Contains no secrets -- the sensor token is passed on the
    command line."""
    server = _public_api_url(request)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for arcname, rel in AGENT_FILES.items():
            data = (Path(BACKEND_DIR) / rel).read_text(encoding="utf-8")
            if arcname == "nadf_agent.py":
                data = data.replace("__NADF_SERVER_URL__", server)
            z.writestr(f"nadf-agent/{arcname}", data)
    return Response(content=buf.getvalue(), media_type="application/zip",
                    headers={"Content-Disposition": 'attachment; filename="nadf-agent.zip"'})


# ---------------------------------------------------------------------------
# capture agent endpoints -- authenticated by the sensor token
# ---------------------------------------------------------------------------
class AgentHello(BaseModel):
    hostname: str | None = None
    local_ip: str | None = None
    iface: str | None = None
    os: str | None = None
    agent_version: str | None = None


class AgentWindow(BaseModel):
    remote_ip: str = Field(min_length=1, max_length=64)
    observed_at: str | None = None
    features: dict[str, float]


class AgentPacket(BaseModel):
    remote_ip: str = Field(min_length=1, max_length=64)
    ts: float
    direction: str
    local_port: int
    remote_port: int
    flags: str
    ttl: int
    win_size: int
    pkt_len: int
    description: str = ""


class AgentBatch(BaseModel):
    windows: list[AgentWindow] = []
    packets: list[AgentPacket] = []
    packets_seen: int | None = None
    error: str | None = None


class AgentAlert(BaseModel):
    remote_ip: str = Field(min_length=1, max_length=64)
    message: str
    severity: str = "warning"
    detail: dict = {}
    timestamp: str | None = None


class AgentAlerts(BaseModel):
    alerts: list[AgentAlert]


@app.post("/agent/hello")
def agent_hello(req: AgentHello, sensor: dict = Depends(current_sensor)):
    db.touch_sensor(sensor["id"], hostname=req.hostname, local_ip=req.local_ip, iface=req.iface, os=req.os,
                    agent_version=req.agent_version, error=None)
    return {"sensor_id": sensor["id"], "name": sensor["name"], "window_seconds": WINDOW_SECONDS}


@app.post("/agent/windows")
def agent_windows(batch: AgentBatch, sensor: dict = Depends(current_sensor)):
    _require_ready()
    if len(batch.windows) > MAX_AGENT_WINDOWS or len(batch.packets) > MAX_AGENT_PACKETS:
        raise HTTPException(status_code=413, detail="batch too large")
    try:
        results = service.ingest_agent_batch(sensor, [w.model_dump() for w in batch.windows],
                                             [p.model_dump() for p in batch.packets])
    except FeatureValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    db.touch_sensor(sensor["id"], packets_seen=batch.packets_seen, error=batch.error)
    return {"accepted": len(results), "results": results}


@app.post("/agent/alerts")
def agent_alerts(req: AgentAlerts, sensor: dict = Depends(current_sensor)):
    for a in req.alerts[:MAX_AGENT_ALERTS]:
        db.log_tripwire_alert(sensor["user_id"], a.remote_ip, a.message,
                              a.severity if a.severity in ("info", "warning", "critical") else "warning",
                              a.detail, created_at=a.timestamp, source="agent")
    db.touch_sensor(sensor["id"])
    return {"accepted": min(len(req.alerts), MAX_AGENT_ALERTS)}


# ---------------------------------------------------------------------------
# live view (the caller's sensors)
# ---------------------------------------------------------------------------
def _packet_host_id(host: str) -> str:
    return host if host.startswith(("live:", "pcap:", "sample-")) else f"live:{host}"


@app.get("/live/status")
def live_status(user: str = Depends(current_user)):
    sensors = db.list_sensors(user)
    online = [s for s in sensors if s["online"]]
    primary = online[0] if online else (sensors[0] if sensors else None)
    return {
        "running": bool(online),
        "sensors_total": len(sensors),
        "sensors_online": len(online),
        "iface": primary and primary["iface"],
        "local_ip": primary and primary["local_ip"],
        "started_at": primary and primary["created_at"],
        "packets_seen": sum(s["packets_seen"] for s in sensors),
        "hosts_seen": len(service.live_hosts_with_predictions(user)) if service.ready else 0,
        "error": next((s["error"] for s in sensors if s["error"]), None),
    }


@app.get("/live/recent")
def live_recent(limit: int = 50, user: str = Depends(current_user)):
    return db.recent_live_windows(user, limit=min(limit, 500))


@app.get("/live/alerts")
def live_alerts(limit: int = 50, user: str = Depends(current_user)):
    """Fast, rule-based tripwire alerts (from the caller's agents and pcap
    uploads) -- NOT the ML world model. See app/live/tripwire.py."""
    return db.recent_tripwire_alerts(user, limit=min(limit, 500))


@app.get("/live/packets/{host}")
def live_packets(host: str, limit: int = 100, user: str = Depends(current_user)):
    """Recent raw packets (headers only) for one of the caller's hosts."""
    return db.recent_packets(user, _packet_host_id(host), limit=min(limit, 300))


# ---------------------------------------------------------------------------
# model reports -- describe the trained model, not any user's data
# ---------------------------------------------------------------------------
@app.get("/attack-mapping")
def attack_mapping():
    return all_mappings()


@app.get("/stage-classes")
def stage_classes():
    return STAGE_CLASSES


@app.get("/step-tracking-report")
def step_tracking_report():
    """Accuracy of step-by-step tracking + next-1/2/3-move comparison
    (Markov / trigram / LSTM / hybrid). Written by app.evaluate_step_tracking."""
    r = service.step_tracking_report()
    if r is None:
        raise HTTPException(status_code=404, detail="step tracking report not computed yet -- run `python -m app.evaluate_step_tracking`")
    return r


@app.get("/benchmark")
def benchmark():
    report = service.benchmark_report()
    if report is None:
        raise HTTPException(status_code=404, detail="benchmark not yet computed; run `python -m app.train`")
    return report


@app.get("/calibration")
def calibration():
    report = service.calibration_report()
    if report is None:
        raise HTTPException(status_code=404, detail="calibration report not yet computed; run `python -m app.train`")
    return report


@app.get("/lead-time")
def lead_time():
    report = service.lead_time_report()
    if report is None:
        raise HTTPException(status_code=404, detail="lead-time report not yet computed; run `python -m app.train`")
    return report


@app.get("/false-alarms")
def false_alarms():
    return service.false_alarm_examples()


@app.get("/threshold-calibration")
def threshold_calibration():
    """What alert threshold would be needed for a given alerts-per-day
    budget, computed from the model's real score distribution on held-out
    benign traffic -- does not change the 0.5 threshold used elsewhere.
    See app/evaluation/threshold_tuning.py."""
    report = service.threshold_calibration_report()
    if report is None:
        raise HTTPException(status_code=404, detail="threshold calibration not yet computed; run `python -m app.train`")
    return report


@app.get("/robustness-report")
def robustness_report():
    """Variance across several independent training seeds and cross-run
    generalization. See app/evaluate_robustness.py; not run automatically by
    `python -m app.train` (takes several minutes)."""
    report = service.robustness_report()
    if report is None:
        raise HTTPException(status_code=404, detail="robustness report not yet computed; run `python -m app.evaluate_robustness`")
    return report
