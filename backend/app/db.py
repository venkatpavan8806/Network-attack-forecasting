"""SQLite-backed inference log. Every row is either seeded from a real
training-time evaluation run (data/recent_forecast_log.json, produced by
app/train.py against actual held-out data) or inserted by a live API call
that actually ran inference. Nothing in this table is fabricated -- the
"Forecasts Generated (today)" KPI and the "Recent Forecast Log" UI table
both read straight from it.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

from app.config import DATA_DIR, FORECAST_LOG_JSON

DB_PATH = DATA_DIR / "app_state.sqlite3"

SCHEMA = """
CREATE TABLE IF NOT EXISTS inference_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    host_id TEXT NOT NULL,
    window_idx INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    model TEXT NOT NULL,
    predicted_stage TEXT,
    infiltration_probability REAL NOT NULL,
    true_stage TEXT,
    state_label TEXT,
    source TEXT NOT NULL DEFAULT 'live'
);
CREATE INDEX IF NOT EXISTS idx_inference_created_at ON inference_log(created_at);
CREATE INDEX IF NOT EXISTS idx_inference_host ON inference_log(host_id);

CREATE TABLE IF NOT EXISTS tripwire_alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    remote_ip TEXT NOT NULL,
    message TEXT NOT NULL,
    severity TEXT NOT NULL,
    detail_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_tripwire_created_at ON tripwire_alerts(created_at);
CREATE INDEX IF NOT EXISTS idx_tripwire_remote_ip ON tripwire_alerts(remote_ip);
"""


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.executescript(SCHEMA)


def log_inference(host_id: str, window_idx: int, model: str, predicted_stage: str | None,
                   infiltration_probability: float, true_stage: str | None = None,
                   state_label: str | None = None, source: str = "live"):
    """Logs one real inference call. Skips the insert if it would be an exact
    repeat of the immediately-preceding row for this (host_id, model) pair --
    a demo host's prediction is deterministic (frozen weights, same input),
    so re-polling it (e.g. the Overview page's 10s refresh) would otherwise
    flood the 'Recent Forecast Log' / 'Explainability Digest' UI with copies
    of the same event instead of genuinely new activity. Scoped to `model`
    too: one forecast call logs both "world_model_lstm" and
    "baseline_logreg" rows for the same window, so comparing only against
    the single most recent row (regardless of model) would never match --
    it would always be the OTHER model's row. A live host's window_idx
    advances every real window, so its rows are never skipped by this."""
    with get_conn() as conn:
        last = conn.execute(
            "SELECT window_idx, predicted_stage, infiltration_probability, source "
            "FROM inference_log WHERE host_id = ? AND model = ? ORDER BY id DESC LIMIT 1",
            (host_id, model),
        ).fetchone()
        if (last is not None and last["window_idx"] == window_idx
                and last["predicted_stage"] == predicted_stage and last["source"] == source
                and abs(last["infiltration_probability"] - float(infiltration_probability)) < 1e-9):
            return
        conn.execute(
            "INSERT INTO inference_log (host_id, window_idx, created_at, model, predicted_stage, "
            "infiltration_probability, true_stage, state_label, source) VALUES (?,?,?,?,?,?,?,?,?)",
            (host_id, window_idx, datetime.now(timezone.utc).isoformat(), model, predicted_stage,
             float(infiltration_probability), true_stage, state_label, source),
        )


def seed_from_training_log(force: bool = False):
    """Loads app/train.py's recent_forecast_log.json (real held-out
    predictions from both models) into the DB so the demo isn't empty on
    first boot. Skipped if the table already has seeded rows, unless force."""
    with get_conn() as conn:
        existing = conn.execute("SELECT COUNT(*) as c FROM inference_log WHERE source='seed'").fetchone()["c"]
        if existing > 0 and not force:
            return 0

    if not FORECAST_LOG_JSON.exists():
        return 0
    with open(FORECAST_LOG_JSON) as f:
        rows = json.load(f)

    count = 0
    for r in rows:
        log_inference(r["host_id"], r["window_idx"], "world_model_lstm", r["predicted_stage"],
                       r["infiltration_probability_world_model"], r["true_stage"], r["state_label"], source="seed")
        log_inference(r["host_id"], r["window_idx"], "baseline_logreg", None,
                       r["infiltration_probability_baseline"], r["true_stage"], r["state_label"], source="seed")
        count += 2
    return count


def count_forecasts_today() -> int:
    today = datetime.now(timezone.utc).date().isoformat()
    with get_conn() as conn:
        row = conn.execute(
            "SELECT COUNT(*) as c FROM inference_log WHERE substr(created_at,1,10) = ?", (today,)
        ).fetchone()
        return int(row["c"])


# "Currently" high-risk means recently high-risk for a CONTINUOUS MONITOR
# -- a live-capture flag from an hour ago must not permanently occupy
# "highest risk right now" just because nothing newer has been logged for
# that host since (a real host stops appearing once its traffic is no
# longer flagged, e.g. after the flow_tracker fix removed a false
# positive -- its last, stale log row must not linger forever). Live
# capture hosts refresh every WINDOW_SECONDS (30s) while running, so 15
# minutes of silence is a reasonable staleness cutoff. This does NOT apply
# to seed/ingest rows (source != 'live_capture') -- those are fixed
# reference results from a specific run (the CSV-replay demo benchmark or
# an explicit /ingest call), not a claim about what is happening right
# now, so they are not expected to keep refreshing and should not age out.
RECENCY_WINDOW_MINUTES = 15


def _recency_cutoff_iso() -> str:
    return (datetime.now(timezone.utc) - timedelta(minutes=RECENCY_WINDOW_MINUTES)).isoformat()


def count_high_risk_hosts(threshold: float = 0.5) -> int:
    """Hosts whose MOST RECENT world-model log row is above threshold --
    live-capture rows older than the recency window are excluded."""
    cutoff = _recency_cutoff_iso()
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT host_id, infiltration_probability, created_at FROM inference_log "
            "WHERE model='world_model_lstm' AND (source != 'live_capture' OR created_at >= ?) "
            "ORDER BY created_at DESC",
            (cutoff,),
        ).fetchall()
    seen = set()
    count = 0
    for r in rows:
        if r["host_id"] in seen:
            continue
        seen.add(r["host_id"])
        if r["infiltration_probability"] >= threshold:
            count += 1
    return count


def highest_risk_host():
    cutoff = _recency_cutoff_iso()
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT host_id, window_idx, predicted_stage, infiltration_probability, created_at FROM inference_log "
            "WHERE model='world_model_lstm' AND (source != 'live_capture' OR created_at >= ?) "
            "ORDER BY created_at DESC",
            (cutoff,),
        ).fetchall()
    latest_per_host = {}
    for r in rows:
        if r["host_id"] not in latest_per_host:
            latest_per_host[r["host_id"]] = r
    if not latest_per_host:
        return None
    best = max(latest_per_host.values(), key=lambda r: r["infiltration_probability"])
    return dict(best)


def recent_forecast_log(limit: int = 25):
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT host_id, window_idx, created_at, model, predicted_stage, infiltration_probability, "
            "true_stage, state_label FROM inference_log ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def stage_breakdown(limit: int = 500):
    """Share of recent world-model windows classified into each stage."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT predicted_stage FROM inference_log WHERE model='world_model_lstm' "
            "AND predicted_stage IS NOT NULL ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    total = len(rows)
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["predicted_stage"]] = counts.get(r["predicted_stage"], 0) + 1
    return {"total": total, "counts": counts}


def log_tripwire_alert(remote_ip: str, message: str, severity: str, detail: dict) -> dict:
    """Persists a fast rule-based tripwire alert (see app/live/tripwire.py).
    Unlike inference_log rows, these are never wiped on capture restart --
    they're a genuine event log, not per-session state."""
    created_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO tripwire_alerts (created_at, remote_ip, message, severity, detail_json) VALUES (?,?,?,?,?)",
            (created_at, remote_ip, message, severity, json.dumps(detail)),
        )
        alert_id = cur.lastrowid
    return {"id": alert_id, "timestamp": created_at, "remote_ip": remote_ip, "message": message, "severity": severity, "detail": detail}


def recent_tripwire_alerts(limit: int = 50) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, created_at, remote_ip, message, severity, detail_json FROM tripwire_alerts "
            "ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [
        {
            "id": r["id"],
            "timestamp": r["created_at"],
            "remote_ip": r["remote_ip"],
            "message": r["message"],
            "severity": r["severity"],
            "detail": json.loads(r["detail_json"]) if r["detail_json"] else {},
        }
        for r in rows
    ]
