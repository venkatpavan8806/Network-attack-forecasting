"""FastAPI application. Fully offline -- no external/cloud calls anywhere in
this module or anything it imports. Every response is either a genuine model
inference result or a report computed by app/train.py from an actual run;
nothing here is a hard-coded placeholder.
"""
from __future__ import annotations

import io

import pandas as pd
# pyrefly: ignore [missing-import]
from fastapi import FastAPI, UploadFile, File, HTTPException
# pyrefly: ignore [missing-import]
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.inference.service import service, ArtifactsNotReadyError
from app.models.attack_mapping import all_mappings
from app import db
from app.config import STAGE_CLASSES
from app.live.capture import live_capture
from app.live.tripwire import tripwire

app = FastAPI(title="Network Attack Forecasting API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    try:
        service.load()
    except ArtifactsNotReadyError as e:
        # Server still starts so /health reports the real reason instead of crashing silently.
        print(f"[startup] {e}")


def _require_ready():
    if not service.ready:
        raise HTTPException(
            status_code=503,
            detail="Model artifacts not trained yet. Run `python -m app.train` in the backend directory, then restart the API.",
        )


@app.get("/health")
def health():
    return {"status": "ok" if service.ready else "not_ready", "ready": service.ready}


@app.get("/kpis")
def kpis():
    _require_ready()
    lead_time = service.lead_time_report() or {}
    return {
        "hosts_monitored": len(service.list_demo_hosts()),
        "forecasts_generated_today": db.count_forecasts_today(),
        "high_risk_trajectories": db.count_high_risk_hosts(threshold=0.5),
        "median_lead_time_minutes": lead_time.get("median_lead_time_minutes_all_hosts"),
    }


@app.get("/highest-risk-host")
def highest_risk_host():
    _require_ready()
    row = db.highest_risk_host()
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
def hosts():
    _require_ready()
    return service.list_demo_hosts()


@app.get("/live/forecastable-hosts")
def live_forecastable_hosts():
    """Live-captured hosts with enough window history (>= SEQ_LEN) to be
    forecastable right now -- i.e. eligible for Explainability and Digital
    Twin, not just the Live Capture table's own building-history view."""
    _require_ready()
    return service.live_hosts_with_predictions()


@app.get("/forecast/{host_id}")
def forecast(host_id: str, at_window_idx: int | None = None):
    _require_ready()
    try:
        return service.forecast_demo_host(host_id, at_window_idx=at_window_idx)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/forecast/{host_id}/branches")
def forecast_branches(host_id: str, at_window_idx: int | None = None,
                       depth: int | None = None, branch_factor: int | None = None):
    """K-step forecast as a branching attack-path tree, each node MITRE-mapped
    -- see app/inference/service.py:branching_forecast."""
    _require_ready()
    try:
        return service.branching_forecast(host_id, at_window_idx=at_window_idx, depth=depth, branch_factor=branch_factor)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/track/{host_id}")
def track_attacker(host_id: str, at_window_idx: int | None = None):
    """Step-by-step attacker tracking: a next-step prediction after EVERY
    window from the host's first one, the attack path recognised so far, and
    the next 1/2/3 moves -- see app/tracking/step_tracker.py."""
    _require_ready()
    try:
        return service.track_attacker(host_id, at_window_idx=at_window_idx)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/step-tracking-report")
def step_tracking_report():
    """Accuracy of step-by-step tracking + next-1/2/3-move comparison
    (Markov / trigram / LSTM / hybrid). Written by app.evaluate_step_tracking."""
    _require_ready()
    r = service.step_tracking_report()
    if r is None:
        raise HTTPException(status_code=404, detail="step tracking report not computed yet -- run `python -m app.evaluate_step_tracking`")
    return r


@app.get("/host-timeline/{host_id}")
def host_timeline(host_id: str):
    """Windows + ground-truth action for one host -- lets the UI offer
    'jump to when port_scan/bruteforce/etc started' instead of only 'now'."""
    _require_ready()
    if service.labeled_df is None:
        raise HTTPException(status_code=404, detail="no dataset loaded")
    host_df = service.labeled_df[service.labeled_df["host_id"] == host_id].sort_values("window_idx")
    if len(host_df) == 0:
        raise HTTPException(status_code=404, detail=f"unknown host_id: {host_id}")
    return host_df[["window_idx", "true_stage", "state_label"]].to_dict("records")


@app.get("/mitigations")
def mitigations():
    _require_ready()
    return service.available_mitigations()


@app.get("/counterfactual/{host_id}")
def counterfactual(host_id: str, mitigation_id: str = "isolate_host", at_window_idx: int | None = None):
    """Digital-twin what-if: compares the world model's predicted
    trajectory with vs. without a named mitigation applied to the cloned
    Digital Twin network state. Safe sandbox simulation -- real_network_touched: false."""
    _require_ready()
    try:
        return service.run_counterfactual(host_id, mitigation_id, at_window_idx=at_window_idx)
    except ValueError as e:
        raise HTTPException(status_code=404 if "unknown host_id" in str(e) else 400, detail=str(e))


class DigitalTwinSimulateRequest(BaseModel):
    host_id: str
    mitigation_id: str = "isolate_host"
    at_window_idx: int | None = None


@app.get("/digital-twin/state/{host_id}")
def digital_twin_state(host_id: str):
    """Returns the current logical Digital Twin network state, services, and firewall policy."""
    _require_ready()
    try:
        from app.simulation.network_twin import build_initial_twin_network
        twin = build_initial_twin_network(host_id, is_live=host_id.startswith("live:"))
        return twin.to_dict()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/digital-twin/simulate")
def digital_twin_simulate(req: DigitalTwinSimulateRequest):
    """Runs a safe sandbox simulation on a cloned Digital Twin network state."""
    _require_ready()
    try:
        return service.run_counterfactual(req.host_id, req.mitigation_id, at_window_idx=req.at_window_idx)
    except ValueError as e:
        raise HTTPException(status_code=404 if "unknown host_id" in str(e) else 400, detail=str(e))


@app.get("/shap/{host_id}")
def shap_explanation(host_id: str, at_window_idx: int | None = None):
    """SHAP on the baseline next to attention + saliency on the LSTM, for the
    same host/window. Works for demo hosts and for `live:<ip>` hosts from the
    live capture. See app/explain/shap_baseline.py and app/explain/attention.py."""
    _require_ready()
    try:
        return service.explain_shap(host_id, at_window_idx=at_window_idx)
    except ValueError as e:
        raise HTTPException(status_code=404 if "unknown host_id" in str(e) else 400, detail=str(e))
    except ArtifactsNotReadyError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/defense/{host_id}")
def defense_advice(host_id: str, at_window_idx: int | None = None):
    """Ranked defensive recommendation computed from real counterfactual
    rollouts of the trained world model (every mitigation vs. no mitigation),
    plus the ATT&CK-mapped manual playbook for the stage the model expects.
    Decision support only -- nothing is applied to any network."""
    _require_ready()
    try:
        return service.defense_advice(host_id, at_window_idx=at_window_idx)
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
async def sandbox_test(file: UploadFile = File(...)):
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
async def ingest(file: UploadFile = File(...)):
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
        results = service.ingest_csv(df)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return results


@app.get("/attack-stage-breakdown")
def attack_stage_breakdown():
    _require_ready()
    return db.stage_breakdown()


@app.get("/forecast-log")
def forecast_log(limit: int = 25):
    _require_ready()
    return db.recent_forecast_log(limit=limit)


@app.get("/attack-mapping")
def attack_mapping():
    return all_mappings()


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


@app.get("/live/interfaces")
def live_interfaces():
    """Real network interfaces on this machine with an IPv4 address, so the
    UI can offer a picker instead of free-text entry."""
    try:
        # pyrefly: ignore [missing-import]
        from scapy.arch.windows import get_windows_if_list
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"scapy unavailable: {e}")
    out = []
    for i in get_windows_if_list():
        ipv4 = [ip for ip in i.get("ips", []) if "." in ip and not ip.startswith("169.254")]
        if ipv4:
            out.append({"name": i.get("name"), "description": i.get("description"), "ip": ipv4[0]})
    return out


@app.post("/live/start")
def live_start(req: LiveStartRequest):
    _require_ready()
    try:
        live_capture.start(req.iface, req.local_ip)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return live_capture.status()


@app.post("/live/stop")
def live_stop():
    live_capture.stop()
    return live_capture.status()


@app.get("/live/status")
def live_status():
    return live_capture.status()


@app.get("/live/recent")
def live_recent(limit: int = 50):
    items = list(live_capture.recent_predictions)[-limit:]
    return list(reversed(items))


@app.get("/live/alerts")
def live_alerts(limit: int = 50):
    """Fast, rule-based tripwire alerts -- millisecond latency, NOT the ML
    world model. See app/live/tripwire.py. Complementary to /live/recent,
    which is the slower LSTM forecast that needs real window history."""
    return tripwire.recent(limit=limit)


@app.get("/live/packets/{remote_ip}")
def live_packets(remote_ip: str, limit: int = 100):
    """Raw, individual packets captured to/from one remote host, most
    recent first, each with a plain-English description of what it is
    (SYN/SYN-ACK/RST/FIN/data/ACK). Independent of the window-based
    aggregate features and of the initiation-direction filter those use --
    this shows everything actually captured for that IP."""
    _require_ready()
    if live_capture.tracker is None:
        raise HTTPException(status_code=404, detail="live capture is not running")
    return live_capture.tracker.recent_packets(remote_ip, limit=limit)
