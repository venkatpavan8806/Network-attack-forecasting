"""Per-user storage (SQLAlchemy Core).

Every row belongs to one user (`user_id` = the Supabase Auth user id) and
every query filters on it, so each user only ever sees their own data:

  user_windows     the hosts the user uploaded (telemetry CSV); a new
                   user starts with none
  inference_log    every real inference call (KPIs, "Recent Forecast Log")
  tripwire_alerts  fast rule-based alerts from a live capture the user started

Backend: Supabase Postgres when DATABASE_URL is set (production), otherwise
a local SQLite file (development / tests).
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pandas as pd
from sqlalchemy import (
    JSON, BigInteger, Column, DateTime, Float, Index, Integer, MetaData, String, Table,
    UniqueConstraint, create_engine, delete, func, insert, select, text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import Engine

from app.config import DATA_DIR, FEATURE_COLUMNS

_Id = BigInteger().with_variant(Integer, "sqlite")
_Json = JSON().with_variant(JSONB, "postgresql")

metadata = MetaData()

user_windows = Table(
    "user_windows", metadata,
    Column("id", _Id, primary_key=True, autoincrement=True),
    Column("user_id", String(64), nullable=False),
    Column("host_id", String(128), nullable=False),
    Column("window_idx", Integer, nullable=False),
    Column("true_stage", String(40)),
    Column("state_label", String(40)),
    Column("features", _Json, nullable=False),
    UniqueConstraint("user_id", "host_id", "window_idx", name="uq_uw_user_host_window"),
    Index("ix_uw_user", "user_id"),
)

inference_log = Table(
    "inference_log", metadata,
    Column("id", _Id, primary_key=True, autoincrement=True),
    Column("user_id", String(64), nullable=False),
    Column("host_id", String(128), nullable=False),
    Column("window_idx", Integer, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("model", String(32), nullable=False),
    Column("predicted_stage", String(40)),
    Column("infiltration_probability", Float, nullable=False),
    Column("true_stage", String(40)),
    Column("state_label", String(40)),
    Column("source", String(16), nullable=False, server_default="live"),
    Index("ix_il_user_created", "user_id", "created_at"),
    Index("ix_il_user_host", "user_id", "host_id"),
)

tripwire_alerts = Table(
    "tripwire_alerts", metadata,
    Column("id", _Id, primary_key=True, autoincrement=True),
    Column("user_id", String(64), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("remote_ip", String(64), nullable=False),
    Column("message", String(500), nullable=False),
    Column("severity", String(16), nullable=False),
    Column("detail", _Json),
    Index("ix_ta_user", "user_id", "id"),
)

# ---------------------------------------------------------------------------
# engine
# ---------------------------------------------------------------------------
_engine: Engine | None = None


def _normalise_url(url: str) -> str:
    # Supabase gives postgres:// or postgresql:// URLs; SQLAlchemy needs the driver name
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    if url.startswith("postgresql://"):
        url = "postgresql+psycopg://" + url[len("postgresql://"):]
    return url


def default_url() -> str:
    return os.environ.get("DATABASE_URL") or f"sqlite:///{(DATA_DIR / 'app_state.sqlite3').as_posix()}"


def describe_url(url: str | None = None) -> str:
    """Database target for logs/health WITHOUT the password, with hints for
    the usual Supabase connection-string mistakes."""
    from sqlalchemy.engine import make_url
    raw = url or default_url()
    if "[" in raw.split("@")[0]:
        return "DATABASE_URL still contains the [YOUR-PASSWORD] placeholder -- put the real database password in"
    try:
        u = make_url(_normalise_url(raw))
    except Exception:
        return "DATABASE_URL is not a valid URL -- if the password has @ # / : ? % characters, URL-encode them"
    if u.drivername.startswith("sqlite"):
        return f"sqlite database {u.database}"
    host = u.host or ""
    if any(c in host for c in "@#/?%: ") or "@" in (u.username or ""):
        # never echo host/user here: with an unencoded @ they contain part of the password
        return ("DATABASE_URL is malformed: the password probably contains @ # / : ? % characters -- "
                "URL-encode them (@ -> %40, # -> %23, / -> %2F) or use a password with only letters and digits")
    desc = f"postgres host={host} port={u.port} database={u.database} user={u.username}"
    if host.startswith("db.") and host.endswith(".supabase.co"):
        desc += " -- this is Supabase's IPv6-only direct host; use the 'Session pooler' connection string instead"
    return desc


def configure(url: str | None = None) -> Engine:
    """(Re)creates the engine; tests call this to use a throwaway database."""
    global _engine
    url = _normalise_url(url or default_url())
    if url.startswith("sqlite"):
        _engine = create_engine(url, connect_args={"check_same_thread": False})
    else:
        _engine = create_engine(url, pool_pre_ping=True, pool_size=5, max_overflow=5, pool_recycle=300)
    return _engine


def engine() -> Engine:
    if _engine is None:
        configure()
    return _engine


def init_db():
    """Creates the tables if missing. On Postgres also enables Row Level
    Security with no policies, so Supabase's public REST API cannot read
    them -- only this backend (connected as the database owner) can."""
    eng = engine()
    metadata.create_all(eng)
    if eng.dialect.name == "postgresql":
        with eng.begin() as conn:
            for t in metadata.sorted_tables:
                conn.execute(text(f'ALTER TABLE "{t.name}" ENABLE ROW LEVEL SECURITY'))


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _aware(dt):
    if dt is None:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _iso(dt) -> str | None:
    dt = _aware(dt)
    return dt.isoformat() if dt else None


# ---------------------------------------------------------------------------
# the user's own dataset
# ---------------------------------------------------------------------------
def user_has_data(user_id: str) -> bool:
    with engine().connect() as conn:
        return conn.execute(
            select(user_windows.c.id).where(user_windows.c.user_id == user_id).limit(1)
        ).first() is not None


def replace_user_hosts(user_id: str, df: pd.DataFrame):
    """Saves the hosts in df for this user, replacing any earlier upload of
    the same host ids."""
    host_ids = sorted(df["host_id"].astype(str).unique())
    with engine().begin() as conn:
        conn.execute(delete(user_windows).where(user_windows.c.user_id == user_id,
                                                user_windows.c.host_id.in_(host_ids)))
    insert_user_windows(user_id, df)


def insert_user_windows(user_id: str, df: pd.DataFrame):
    """df: host_id, window_idx, true_stage, state_label + FEATURE_COLUMNS."""
    rows = [{
        "user_id": user_id,
        "host_id": str(r["host_id"]),
        "window_idx": int(r["window_idx"]),
        "true_stage": r.get("true_stage"),
        "state_label": r.get("state_label"),
        "features": {c: float(r[c]) for c in FEATURE_COLUMNS},
    } for r in df.to_dict("records")]
    with engine().begin() as conn:
        for i in range(0, len(rows), 500):
            conn.execute(insert(user_windows), rows[i:i + 500])


def user_frame(user_id: str) -> pd.DataFrame:
    """All of the user's windows as a DataFrame shaped like the training
    dataset (host_id, window_idx, true_stage, state_label, FEATURE_COLUMNS)."""
    uw = user_windows
    with engine().connect() as conn:
        rows = conn.execute(
            select(uw.c.host_id, uw.c.window_idx, uw.c.true_stage, uw.c.state_label, uw.c.features)
            .where(uw.c.user_id == user_id).order_by(uw.c.host_id, uw.c.window_idx)
        ).mappings().all()
    records = []
    for r in rows:
        rec = {"host_id": r["host_id"], "window_idx": r["window_idx"],
               "true_stage": r["true_stage"], "state_label": r["state_label"]}
        feats = r["features"] or {}
        for c in FEATURE_COLUMNS:
            rec[c] = float(feats.get(c, 0.0))
        records.append(rec)
    return pd.DataFrame(records, columns=["host_id", "window_idx", "true_stage", "state_label"] + FEATURE_COLUMNS)


# ---------------------------------------------------------------------------
# inference log
# ---------------------------------------------------------------------------
def log_inference(user_id: str, host_id: str, window_idx: int, model: str, predicted_stage: str | None,
                  infiltration_probability: float, true_stage: str | None = None,
                  state_label: str | None = None, source: str = "live"):
    """Logs one real inference call for this user. Skips an exact repeat of
    the previous row for this (user, host, model) -- a demo host's
    prediction is deterministic, so re-polling it (e.g. the Overview page's
    10 s refresh) must not flood the log with copies. A live host's
    window_idx advances every window, so its rows are never skipped."""
    il = inference_log
    with engine().begin() as conn:
        last = conn.execute(
            select(il.c.window_idx, il.c.predicted_stage, il.c.infiltration_probability, il.c.source)
            .where(il.c.user_id == user_id, il.c.host_id == host_id, il.c.model == model)
            .order_by(il.c.id.desc()).limit(1)
        ).mappings().first()
        if (last is not None and last["window_idx"] == window_idx and last["predicted_stage"] == predicted_stage
                and last["source"] == source
                and abs(last["infiltration_probability"] - float(infiltration_probability)) < 1e-9):
            return
        conn.execute(insert(il).values(
            user_id=user_id, host_id=host_id, window_idx=int(window_idx), created_at=_now(), model=model,
            predicted_stage=predicted_stage, infiltration_probability=float(infiltration_probability),
            true_stage=true_stage, state_label=state_label, source=source,
        ))


def count_forecasts_today(user_id: str) -> int:
    start = _now().replace(hour=0, minute=0, second=0, microsecond=0)
    with engine().connect() as conn:
        return int(conn.execute(
            select(func.count()).select_from(inference_log)
            .where(inference_log.c.user_id == user_id, inference_log.c.created_at >= start)
        ).scalar() or 0)


# "Currently" high-risk means recently high-risk for a continuous monitor: a
# live-capture flag older than this no longer counts as "right now". Rows
# from the demo dataset / CSV ingest are fixed reference results and do not
# age out.
RECENCY_WINDOW_MINUTES = 15


def _latest_world_model_rows(user_id: str):
    il = inference_log
    cutoff = _now() - timedelta(minutes=RECENCY_WINDOW_MINUTES)
    with engine().connect() as conn:
        rows = conn.execute(
            select(il.c.host_id, il.c.window_idx, il.c.predicted_stage, il.c.infiltration_probability,
                   il.c.created_at)
            .where(il.c.user_id == user_id, il.c.model == "world_model_lstm",
                   (il.c.source != "live_capture") | (il.c.created_at >= cutoff))
            .order_by(il.c.created_at.desc(), il.c.id.desc())
        ).mappings().all()
    latest = {}
    for r in rows:
        if r["host_id"] not in latest:
            latest[r["host_id"]] = r
    return latest


def count_high_risk_hosts(user_id: str, threshold: float = 0.5) -> int:
    return sum(1 for r in _latest_world_model_rows(user_id).values() if r["infiltration_probability"] >= threshold)


def highest_risk_host(user_id: str):
    latest = _latest_world_model_rows(user_id)
    if not latest:
        return None
    best = max(latest.values(), key=lambda r: r["infiltration_probability"])
    return {**dict(best), "created_at": _iso(best["created_at"])}


def recent_forecast_log(user_id: str, limit: int = 25):
    il = inference_log
    with engine().connect() as conn:
        rows = conn.execute(
            select(il.c.host_id, il.c.window_idx, il.c.created_at, il.c.model, il.c.predicted_stage,
                   il.c.infiltration_probability, il.c.true_stage, il.c.state_label)
            .where(il.c.user_id == user_id).order_by(il.c.id.desc()).limit(limit)
        ).mappings().all()
    return [{**dict(r), "created_at": _iso(r["created_at"])} for r in rows]


def stage_breakdown(user_id: str, limit: int = 500):
    """Share of the user's recent world-model windows classified into each stage."""
    il = inference_log
    with engine().connect() as conn:
        rows = conn.execute(
            select(il.c.predicted_stage).where(
                il.c.user_id == user_id, il.c.model == "world_model_lstm", il.c.predicted_stage.isnot(None))
            .order_by(il.c.id.desc()).limit(limit)
        ).all()
    counts: dict[str, int] = {}
    for (stage,) in rows:
        counts[stage] = counts.get(stage, 0) + 1
    return {"total": len(rows), "counts": counts}


# ---------------------------------------------------------------------------
# tripwire alerts
# ---------------------------------------------------------------------------
def log_tripwire_alert(user_id: str, remote_ip: str, message: str, severity: str, detail: dict) -> dict:
    """Persists a fast rule-based tripwire alert (see app/live/tripwire.py)
    for the user who started the live capture."""
    created_at = _now()
    with engine().begin() as conn:
        res = conn.execute(insert(tripwire_alerts).values(
            user_id=user_id, created_at=created_at, remote_ip=remote_ip, message=message[:500],
            severity=severity, detail=detail,
        ))
        alert_id = res.inserted_primary_key[0]
    return {"id": alert_id, "timestamp": created_at.isoformat(timespec="milliseconds"), "remote_ip": remote_ip,
            "message": message, "severity": severity, "detail": detail}


def recent_tripwire_alerts(user_id: str, limit: int = 50) -> list[dict]:
    ta = tripwire_alerts
    with engine().connect() as conn:
        rows = conn.execute(
            select(ta).where(ta.c.user_id == user_id).order_by(ta.c.id.desc()).limit(limit)
        ).mappings().all()
    return [{"id": r["id"], "timestamp": _aware(r["created_at"]).isoformat(timespec="milliseconds"),
             "remote_ip": r["remote_ip"], "message": r["message"], "severity": r["severity"],
             "detail": r["detail"] or {}} for r in rows]


def delete_user(user_id: str):
    """Removes everything stored for one user (used by tests)."""
    with engine().begin() as conn:
        for t in (user_windows, inference_log, tripwire_alerts):
            conn.execute(delete(t).where(t.c.user_id == user_id))
