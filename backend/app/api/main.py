"""FastAPI application. Fully offline -- no external/cloud calls anywhere in
this module or anything it imports. Every response is either a genuine model
inference result or a report computed by app/train.py from an actual run;
nothing here is a hard-coded placeholder.
"""
from __future__ import annotations

import io
import os
import zipfile

import pandas as pd
# pyrefly: ignore [missing-import]
from fastapi import Depends, FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import Response
# pyrefly: ignore [missing-import]
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.inference.service import service, ArtifactsNotReadyError
from app.models.attack_mapping import all_mappings
from app import auth, db
from app.auth import current_sensor, current_user
from app.config import BACKEND_DIR, FEATURE_COLUMNS, STAGE_CLASSES, WINDOW_SECONDS
from app.features.extraction import FeatureValidationError, validate_feature_vector
from app.live.capture import live_capture
from app.live.tripwire import tripwire

app = FastAPI(title="Network Attack Forecasting API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


STARTUP_PROBLEM: str | None = None


@app.on_event("startup")
def startup():
    global STARTUP_PROBLEM
    target = db.describe_url()
    print(f"[startup] database: {target}")
    try:
        db.configure()
        service.load()
    except ArtifactsNotReadyError as e:
        # Server still starts so /health reports the real reason instead of crashing silently.
        STARTUP_PROBLEM = str(e)
        print(f"[startup] {e}")
    except Exception as e:  # e.g. database unreachable / wrong DATABASE_URL
        first_line = str(e).splitlines()[0][:300] if str(e) else type(e).__name__
        STARTUP_PROBLEM = f"database connection failed: {first_line} [{target}]"
        print(f"[startup] {STARTUP_PROBLEM}")


def _require_ready():
    if not service.ready:
        raise HTTPException(
            status_code=503,
            detail=STARTUP_PROBLEM or "Model artifacts not trained yet. Run `python -m app.train` in the backend directory, then restart the API.",
        )


@app.get("/health")
def health():
    problem = STARTUP_PROBLEM or auth.PROBLEM
    return {"status": "ok" if service.ready and not problem else "not_ready", "ready": service.ready,
            "problem": problem}


@app.get("/kpis")
def kpis(user: str = Depends(current_user)):
    _require_ready()
    reports = service.cached_user_reports(user)  # never blocks; computed in the background on first use
    lead_time = (reports or {}).get("lead_time") or {}
    return {
        "hosts_monitored": len(service.list_demo_hosts(user)),
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
    mapping = None
    try:
        from app.models.attack_mapping import map_stage
        if row.get("predicted_stage"):
            mapping = map_stage(row["predicted_stage"])
    except Exception:
        mapping = None
    return {**row, "attack_mapping": mapping}


@app.get("/hosts")
def hosts(user: str = Depends(current_user)):
    _require_ready()
    return service.list_demo_hosts(user)


@app.get("/live/forecastable-hosts")
def live_forecastable_hosts(user: str = Depends(current_user)):
    """Live-captured hosts with enough window history (>= SEQ_LEN) to be
    forecastable right now -- i.e. eligible for Explainability and Digital
    Twin, not just the Live Capture table's own building-history view."""
    _require_ready()
    return service.live_hosts_with_predictions(user)


@app.get("/forecast/{host_id}")
def forecast(host_id: str, at_window_idx: int | None = None, user: str = Depends(current_user)):
    _require_ready()
    try:
        return service.forecast_demo_host(user, host_id, at_window_idx=at_window_idx)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/forecast/{host_id}/branches")
def forecast_branches(host_id: str, at_window_idx: int | None = None,
                       depth: int | None = None, branch_factor: int | None = None, user: str = Depends(current_user)):
    """K-step forecast as a branching attack-path tree, each node MITRE-mapped
    -- see app/inference/service.py:branching_forecast."""
    _require_ready()
    try:
        return service.branching_forecast(user, host_id, at_window_idx=at_window_idx, depth=depth, branch_factor=branch_factor)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/track/{host_id}")
def track_attacker(host_id: str, at_window_idx: int | None = None, user: str = Depends(current_user)):
    """Step-by-step attacker tracking: a next-step prediction after EVERY
    window from the host's first one, the attack path recognised so far, and
    the next 1/2/3 moves -- see app/tracking/step_tracker.py."""
    _require_ready()
    try:
        return service.track_attacker(user, host_id, at_window_idx=at_window_idx)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


NO_LABELLED_DATA = ("No labelled traffic yet: upload a CSV that includes a true_stage column (Forecasts -> "
                    "CSV Ingestion) to see these results computed on your own data.")


def _user_report(user: str, key: str):
    """Benchmarks are computed from the signed-in user's own labelled uploads
    (app/evaluation/user_reports.py), not from the training run."""
    _require_ready()
    report = service.user_reports(user)[key]
    if report is None:
        raise HTTPException(status_code=404, detail=NO_LABELLED_DATA)
    return report


@app.get("/step-tracking-report")
def step_tracking_report(user: str = Depends(current_user)):
    """Step-by-step tracking accuracy + next-1/2/3-move comparison on the user's data."""
    return _user_report(user, "step_tracking")


@app.get("/host-timeline/{host_id}")
def host_timeline(host_id: str, user: str = Depends(current_user)):
    """Windows + ground-truth action for one host -- lets the UI offer
    'jump to when port_scan/bruteforce/etc started' instead of only 'now'."""
    _require_ready()
    df = service.user_df(user)
    host_df = df[df["host_id"] == host_id].sort_values("window_idx")
    if len(host_df) == 0:
        raise HTTPException(status_code=404, detail=f"unknown host_id: {host_id}")
    return host_df[["window_idx", "true_stage", "state_label"]].to_dict("records")


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
        raise HTTPException(status_code=404 if "unknown host_id" in str(e) else 400, detail=str(e))


class DigitalTwinSimulateRequest(BaseModel):
    host_id: str
    mitigation_id: str = "isolate_host"
    at_window_idx: int | None = None


@app.get("/digital-twin/state/{host_id}")
def digital_twin_state(host_id: str, user: str = Depends(current_user)):
    """Returns the current logical Digital Twin network state, services, and firewall policy."""
    _require_ready()
    try:
        from app.simulation.network_twin import build_initial_twin_network
        twin = build_initial_twin_network(host_id, is_live=host_id.startswith("live:"))
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
        raise HTTPException(status_code=404 if "unknown host_id" in str(e) else 400, detail=str(e))


@app.get("/shap/{host_id}")
def shap_explanation(host_id: str, at_window_idx: int | None = None, user: str = Depends(current_user)):
    """SHAP on the baseline next to attention + saliency on the LSTM, for the
    same host/window. Works for demo hosts and for `live:<ip>` hosts from the
    live capture. See app/explain/shap_baseline.py and app/explain/attention.py."""
    _require_ready()
    try:
        return service.explain_shap(user, host_id, at_window_idx=at_window_idx)
    except ValueError as e:
        raise HTTPException(status_code=404 if "unknown host_id" in str(e) else 400, detail=str(e))
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
        raise HTTPException(status_code=404 if "unknown host_id" in str(e) else 400, detail=str(e))
    except ArtifactsNotReadyError as e:
        raise HTTPException(status_code=503, detail=str(e))


class SandboxTestResponse(BaseModel):
    outcome: str
    errors: list[str]
    rows: int
    hosts: int


@app.post("/sandbox/test", response_model=SandboxTestResponse)
async def sandbox_test(file: UploadFile = File(...), user: str = Depends(current_user)):
    """Genuinely validates the uploaded CSV and can return outcome="failure"
    when the input actually warrants it (malformed columns, too few windows
    per host, non-numeric feature values, etc.) -- see app/inference/service.py:validate_telemetry_csv.
    """
    _require_ready()
    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        return SandboxTestResponse(outcome="failure", errors=[f"could not parse CSV: {e}"], rows=0, hosts=0)

    errors = service.validate_telemetry_csv(df)
    outcome = "failure" if errors else "success"
    n_hosts = df["host_id"].nunique() if "host_id" in df.columns else 0
    return SandboxTestResponse(outcome=outcome, errors=errors, rows=len(df), hosts=int(n_hosts))


@app.post("/ingest")
async def ingest(file: UploadFile = File(...), user: str = Depends(current_user)):
    """Real CSV ingestion: parses the uploaded file, runs actual inference
    for the last window of every host present, logs results, and returns
    them. Raises 422 on genuinely malformed input rather than silently
    returning a fixed dummy result."""
    _require_ready()
    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"could not parse CSV: {e}")
    try:
        results = service.ingest_csv(user, df)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return results


@app.get("/attack-stage-breakdown")
def attack_stage_breakdown(user: str = Depends(current_user)):
    _require_ready()
    return db.stage_breakdown(user)


@app.get("/forecast-log")
def forecast_log(limit: int = 25, user: str = Depends(current_user)):
    _require_ready()
    return db.recent_forecast_log(user, limit=limit)


@app.get("/attack-mapping")
def attack_mapping():
    return all_mappings()


@app.get("/benchmark")
def benchmark(user: str = Depends(current_user)):
    """World model vs. baseline (precision/recall/F1/FPR) on the user's data."""
    return _user_report(user, "benchmark")


@app.get("/calibration")
def calibration(user: str = Depends(current_user)):
    return _user_report(user, "calibration")


@app.get("/lead-time")
def lead_time(user: str = Depends(current_user)):
    return _user_report(user, "lead_time")


@app.get("/false-alarms")
def false_alarms():
    return service.false_alarm_examples()


@app.get("/threshold-calibration")
def threshold_calibration(user: str = Depends(current_user)):
    """What alert threshold a given alerts-per-day budget needs, from the
    model's score distribution on the user's benign hosts -- does not change
    the 0.5 threshold used elsewhere. See app/evaluation/threshold_tuning.py."""
    return _user_report(user, "threshold")


@app.get("/robustness-report")
def robustness_report():
    """Variance across several independent training seeds (mean/std/CI, not
    a single lucky-or-unlucky run) and cross-run generalization (train once,
    evaluate on entirely fresh independently-generated worlds -- this
    project's stand-in for a time-based split). See
    app/evaluate_robustness.py; not run automatically by `python -m app.train`
    (takes several minutes) -- run `python -m app.evaluate_robustness` separately."""
    report = service.robustness_report()
    if report is None:
        raise HTTPException(status_code=404, detail="robustness report not yet computed; run `python -m app.evaluate_robustness`")
    return report


@app.get("/stage-classes")
def stage_classes():
    return STAGE_CLASSES


# ---------------------------------------------------------------------------
# Live capture: real packets, real features, real inference. Off by default;
# only runs when explicitly started, and only ever reads packets addressed
# to/from THIS machine's own interface (no promiscuous mode, no traffic
# between other hosts is captured). See app/live/flow_tracker.py for the
# documented vantage-point adaptation and app/live/capture.py for the loop.
# ---------------------------------------------------------------------------
class LiveStartRequest(BaseModel):
    iface: str = "Wi-Fi"
    local_ip: str


def _online_sensor(user: str) -> dict | None:
    """The user's capture agent that is running right now (most recently seen)."""
    online = [x for x in db.list_sensors(user) if x["online"]]
    return max(online, key=lambda x: x["last_seen_at"] or "") if online else None


def _local_interfaces() -> list[dict]:
    """Interfaces of the machine the BACKEND runs on (only useful when it runs
    on the monitored Windows machine itself, e.g. a local demo)."""
    try:
        # pyrefly: ignore [missing-import]
        from scapy.arch.windows import get_windows_if_list
    except Exception:
        return []
    out = []
    for i in get_windows_if_list():
        ipv4 = [ip for ip in i.get("ips", []) if "." in ip and not ip.startswith("169.254")]
        if ipv4:
            out.append({"name": i.get("name"), "description": i.get("description"), "ip": ipv4[0]})
    return out


@app.get("/live/interfaces")
def live_interfaces(user: str = Depends(current_user)):
    """Network interfaces the user can capture on: those reported by their
    running capture agent, else (local demo) this machine's own. Empty when
    neither is available -- the panel then shows how to set up the agent."""
    sensor = _online_sensor(user)
    if sensor is not None:
        return [{"name": i.get("name"), "description": i.get("description") or "", "ip": i.get("ip")}
                for i in sensor["interfaces"] if i.get("name") and i.get("ip")]
    return _local_interfaces()


@app.post("/live/start")
def live_start(req: LiveStartRequest, user: str = Depends(current_user)):
    """Starts capture on the chosen interface: on the user's capture agent if
    one is running (it picks up the request within a few seconds), otherwise
    on this machine (local demo)."""
    _require_ready()
    sensor = _online_sensor(user)
    if sensor is not None:
        db.update_sensor(sensor["id"], touch=False, capture_requested=True, capture_iface=req.iface,
                         capture_local_ip=req.local_ip, capture_started_at=db._now(), error=None)
        return _live_status_for(user)
    if not _local_interfaces():
        raise HTTPException(status_code=409, detail="no capture agent is running -- start the agent on the "
                                                    "computer you want to monitor (see the instructions above)")
    try:
        live_capture.start(req.iface, req.local_ip, owner=user)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return _live_status_for(user)


def _live_status_for(user: str) -> dict:
    hosts_seen = len({h for h in service.user_df(user)["host_id"] if h.startswith("live:")}) if service.ready else 0
    sensor = _online_sensor(user)
    if sensor is not None:
        return {"running": sensor["capture_requested"], "mode": "agent", "agent_online": True,
                "agent_name": sensor["name"], "agent_hostname": sensor["hostname"],
                "iface": sensor["capture_iface"], "local_ip": sensor["capture_local_ip"],
                "started_at": sensor["capture_started_at"], "packets_seen": sensor["packets_seen"],
                "hosts_seen": hosts_seen, "error": sensor["error"]}
    status = live_capture.status()
    mine = live_capture.owner == user
    return {**status, "mode": "local", "agent_online": False,
            "running": bool(status["running"] and mine),
            "packets_seen": status["packets_seen"] if mine else 0, "hosts_seen": hosts_seen,
            "error": status["error"] if mine else None}


@app.post("/live/stop")
def live_stop(user: str = Depends(current_user)):
    sensor = _online_sensor(user)
    if sensor is not None:
        db.update_sensor(sensor["id"], touch=False, capture_requested=False)
    elif live_capture.running:
        if live_capture.owner != user:
            raise HTTPException(status_code=409, detail="this live capture was started by another user")
        live_capture.stop()
    return _live_status_for(user)


@app.get("/live/status")
def live_status(user: str = Depends(current_user)):
    return _live_status_for(user)


@app.get("/live/recent")
def live_recent(limit: int = 50, user: str = Depends(current_user)):
    """The user's most recent live windows with their predictions (newest first)."""
    return db.recent_live_entries(user, limit=min(limit, 500))


@app.get("/live/alerts")
def live_alerts(limit: int = 50, user: str = Depends(current_user)):
    """Fast, rule-based tripwire alerts -- millisecond latency, NOT the ML
    world model. See app/live/tripwire.py. Complementary to /live/recent,
    which is the slower LSTM forecast that needs real window history."""
    return tripwire.recent(user, limit=limit)


@app.get("/live/packets/{remote_ip}")
def live_packets(remote_ip: str, limit: int = 100, user: str = Depends(current_user)):
    """Raw, individual packets captured to/from one remote host, most
    recent first, each with a plain-English description of what it is
    (SYN/SYN-ACK/RST/FIN/data/ACK)."""
    _require_ready()
    if live_capture.tracker is not None and live_capture.owner == user:
        return live_capture.tracker.recent_packets(remote_ip, limit=limit)
    return db.recent_live_packets(user, remote_ip, limit=min(limit, 300))


# ---------------------------------------------------------------------------
# Capture agent: lets the hosted website capture the real traffic of the
# user's own computer (a browser cannot read packets; a cloud server only
# sees its own). The user adds a sensor, runs the downloaded agent with its
# token, then uses the normal Start/Stop button in the Live Capture panel.
# ---------------------------------------------------------------------------
class SensorCreate(BaseModel):
    name: str = Field(default="my-laptop", min_length=1, max_length=100)


@app.get("/sensors")
def list_sensors(user: str = Depends(current_user)):
    return db.list_sensors(user)


@app.post("/sensors")
def create_sensor(req: SensorCreate, user: str = Depends(current_user)):
    """Creates a sensor and returns its token ONCE (only its hash is stored)."""
    if len(db.list_sensors(user)) >= 10:
        raise HTTPException(status_code=400, detail="at most 10 sensors per account -- remove one first")
    sensor, token = db.create_sensor(user, req.name.strip())
    return {**sensor, "token": token}


@app.delete("/sensors/{sensor_id}")
def delete_sensor(sensor_id: str, user: str = Depends(current_user)):
    if not db.delete_sensor(user, sensor_id):
        raise HTTPException(status_code=404, detail="unknown sensor")
    return {"deleted": True}


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
    """The capture agent as a zip, built from this backend's own capture code
    (flow_tracker.py / tripwire.py), with this server's address filled in.
    Contains no secrets: the sensor token is given on the command line."""
    server = os.environ.get("PUBLIC_API_URL", "").rstrip("/") or (
        f"{request.headers.get('x-forwarded-proto', request.url.scheme)}://"
        f"{request.headers.get('x-forwarded-host', request.headers.get('host', request.url.netloc))}")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for arcname, rel in AGENT_FILES.items():
            text = (BACKEND_DIR / rel).read_text(encoding="utf-8")
            if arcname == "nadf_agent.py":
                text = text.replace("__NADF_SERVER_URL__", server)
            z.writestr(f"nadf-agent/{arcname}", text)
    return Response(content=buf.getvalue(), media_type="application/zip",
                    headers={"Content-Disposition": 'attachment; filename="nadf-agent.zip"'})


class AgentHello(BaseModel):
    hostname: str | None = None
    os: str | None = None
    agent_version: str | None = None
    interfaces: list[dict] = []


class AgentControl(BaseModel):
    capturing: bool = False
    packets_seen: int = 0
    error: str | None = None
    alerts: list[dict] = []


class AgentWindow(BaseModel):
    remote_ip: str = Field(min_length=1, max_length=64)
    features: dict[str, float]


class AgentWindows(BaseModel):
    windows: list[AgentWindow] = []
    packets: list[dict] = []
    packets_seen: int = 0


@app.post("/agent/hello")
def agent_hello(req: AgentHello, sensor: dict = Depends(current_sensor)):
    db.update_sensor(sensor["id"], hostname=req.hostname, os=req.os, agent_version=req.agent_version,
                     interfaces=req.interfaces[:50], capturing=False, error=None)
    return {"sensor_id": sensor["id"], "name": sensor["name"], "window_seconds": WINDOW_SECONDS}


@app.post("/agent/control")
def agent_control(req: AgentControl, sensor: dict = Depends(current_sensor)):
    """Polled every few seconds by the agent: reports its state + instant
    alerts, and receives the Start/Stop request the user made on the website."""
    for a in req.alerts[:200]:
        db.log_tripwire_alert(sensor["user_id"], str(a.get("remote_ip", "?"))[:64], str(a.get("message", "")),
                              a.get("severity") if a.get("severity") in ("warning", "critical") else "warning",
                              a.get("detail") or {})
    db.update_sensor(sensor["id"], capturing=req.capturing, packets_seen=req.packets_seen,
                     error=(req.error or None) and req.error[:500])
    return {"capture": sensor["capture_requested"], "iface": sensor["capture_iface"],
            "local_ip": sensor["capture_local_ip"], "window_seconds": WINDOW_SECONDS}


@app.post("/agent/windows")
def agent_windows(req: AgentWindows, sensor: dict = Depends(current_sensor)):
    """One 30-second window from the agent: each remote host's features are
    validated, saved to the user's hosts and predicted (same path as local
    capture), plus recent packet headers for the per-host packet view."""
    _require_ready()
    if len(req.windows) > 500 or len(req.packets) > 5000:
        raise HTTPException(status_code=413, detail="batch too large")
    user = sensor["user_id"]
    accepted = 0
    for w in req.windows:
        try:
            feats = validate_feature_vector(w.features)
        except FeatureValidationError as e:
            raise HTTPException(status_code=422, detail=str(e))
        service.ingest_live_window(user, w.remote_ip, dict(zip(FEATURE_COLUMNS, feats.tolist())))
        accepted += 1
    db.add_live_packets(user, [p for p in req.packets if p.get("remote_ip")])
    db.update_sensor(sensor["id"], packets_seen=req.packets_seen)
    return {"accepted": accepted}
