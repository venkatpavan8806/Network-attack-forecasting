"""Per-user persistence layer (SQLAlchemy Core).

Every row belongs to exactly one user (`user_id` = the Supabase Auth user's
UUID, or a dev id when running locally without Supabase). Every query in
this module filters on `user_id`, so one account can never see another
account's traffic, predictions, alerts or sensors -- a new account starts
with an empty workspace.

Backends:
  - production: Supabase Postgres, via DATABASE_URL (use the Supabase
    "Session pooler" connection string; see DEPLOY.md)
  - local dev / tests: SQLite file (default when DATABASE_URL is unset)

Tables:
  traffic_windows  one row per (user, host, window): the FEATURE_COLUMNS
                   vector + the model's prediction for the next window
  inference_log    one row per logged model call (KPIs, forecast log)
  tripwire_alerts  fast rule-based alerts sent by capture agents / pcap replay
  packet_log       recent raw packets per host (capped) for drill-down
  sensors          capture agents registered by the user (token stored hashed)

Nothing here is seeded: there is no shared demo data. Users fill their own
workspace via a capture agent, a pcap/CSV upload, or "Generate sample data"
(fresh random synthetic traffic, generated per request).
"""
from __future__ import annotations

import hashlib
import os
import secrets
import uuid
from datetime import datetime, timedelta, timezone

import pandas as pd
from sqlalchemy import (
    JSON, BigInteger, Column, DateTime, Float, Index, Integer, MetaData, String, Table,
    UniqueConstraint, create_engine, delete, func, insert, select, text, update,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import Engine

from app.config import DATA_DIR, FEATURE_COLUMNS

PACKET_LOG_PER_HOST = 300  # newest packets kept per (user, host)
# "Currently" high-risk means recently high-risk for a continuously
# monitored host: an agent (live capture) row older than this stops counting
# as "right now". Upload/sample rows are fixed results from a specific file
# and do not age out.
RECENCY_WINDOW_MINUTES = 15
LIVE_SOURCES = ("agent",)
SENSOR_ONLINE_SECONDS = 90  # a sensor counts as online if it reported within this many seconds

_Id = BigInteger().with_variant(Integer, "sqlite")
_Json = JSON().with_variant(JSONB, "postgresql")

metadata = MetaData()

traffic_windows = Table(
    "traffic_windows", metadata,
    Column("id", _Id, primary_key=True, autoincrement=True),
    Column("user_id", String(64), nullable=False),
    Column("host_id", String(160), nullable=False),
    Column("window_idx", Integer, nullable=False),
    Column("observed_at", DateTime(timezone=True), nullable=False),
    Column("source", String(16), nullable=False),  # agent | pcap | csv | sample
    Column("features", _Json, nullable=False),
    Column("true_stage", String(40)),
    Column("state_label", String(40)),
    Column("prediction", _Json),
    UniqueConstraint("user_id", "host_id", "window_idx", name="uq_tw_user_host_window"),
    Index("ix_tw_user_host", "user_id", "host_id", "window_idx"),
    Index("ix_tw_user_observed", "user_id", "observed_at"),
)

inference_log = Table(
    "inference_log", metadata,
    Column("id", _Id, primary_key=True, autoincrement=True),
    Column("user_id", String(64), nullable=False),
    Column("host_id", String(160), nullable=False),
    Column("window_idx", Integer, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("model", String(32), nullable=False),
    Column("predicted_stage", String(40)),
    Column("infiltration_probability", Float, nullable=False),
    Column("true_stage", String(40)),
    Column("state_label", String(40)),
    Column("source", String(16), nullable=False),
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
    Column("source", String(16), nullable=False, server_default="agent"),
    Index("ix_ta_user_id", "user_id", "id"),
)

packet_log = Table(
    "packet_log", metadata,
    Column("id", _Id, primary_key=True, autoincrement=True),
    Column("user_id", String(64), nullable=False),
    Column("host_id", String(160), nullable=False),
    Column("ts", Float, nullable=False),
    Column("direction", String(4), nullable=False),
    Column("local_port", Integer, nullable=False),
    Column("remote_port", Integer, nullable=False),
    Column("flags", String(16), nullable=False),
    Column("ttl", Integer, nullable=False),
    Column("win_size", Integer, nullable=False),
    Column("pkt_len", Integer, nullable=False),
    Column("description", String(300), nullable=False),
    Index("ix_pl_user_host", "user_id", "host_id", "id"),
)

sensors = Table(
    "sensors", metadata,
    Column("id", String(32), primary_key=True),
    Column("user_id", String(64), nullable=False),
    Column("name", String(100), nullable=False),
    Column("token_hash", String(64), nullable=False, unique=True),
    Column("token_hint", String(12), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("last_seen_at", DateTime(timezone=True)),
    Column("hostname", String(255)),
    Column("local_ip", String(64)),
    Column("iface", String(255)),
    Column("os", String(100)),
    Column("agent_version", String(32)),
    Column("packets_seen", BigInteger, nullable=False, server_default="0"),
    Column("error", String(500)),
    Index("ix_sensors_user", "user_id"),
)

ALL_USER_TABLES = (traffic_windows, inference_log, tripwire_alerts, packet_log)

# ---------------------------------------------------------------------------
# engine
# ---------------------------------------------------------------------------
_engine: Engine | None = None


def _normalise_url(url: str) -> str:
    # Supabase hands out postgres:// / postgresql:// URLs; SQLAlchemy needs the driver name.
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    if url.startswith("postgresql://"):
        url = "postgresql+psycopg://" + url[len("postgresql://"):]
    return url


def default_url() -> str:
    return os.environ.get("DATABASE_URL") or f"sqlite:///{(DATA_DIR / 'app_state.sqlite3').as_posix()}"


def configure(url: str | None = None) -> Engine:
    """(Re)creates the engine. Called once at startup, and by tests to point
    at a throwaway database."""
    global _engine
    url = _normalise_url(url or default_url())
    if url.startswith("sqlite"):
        _engine = create_engine(url, connect_args={"check_same_thread": False})
    else:
        # small pool: Supabase's free pooler allows a limited number of clients
        _engine = create_engine(url, pool_pre_ping=True, pool_size=5, max_overflow=5, pool_recycle=300)
    return _engine


def engine() -> Engine:
    if _engine is None:
        configure()
    return _engine


def init_db():
    """Creates tables if missing. On Postgres also enables Row Level Security
    with NO policies: the Supabase REST API (anon / authenticated keys) can
    then read nothing from these tables, and only this backend -- which
    connects as the database owner and filters every query by user_id --
    can access them."""
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
    if isinstance(dt, str):
        dt = datetime.fromisoformat(dt)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _iso(dt) -> str | None:
    dt = _aware(dt)
    return dt.isoformat() if dt else None


# ---------------------------------------------------------------------------
# traffic windows
# ---------------------------------------------------------------------------
def last_window_idx(user_id: str, host_id: str) -> int | None:
    with engine().connect() as conn:
        return conn.execute(
            select(func.max(traffic_windows.c.window_idx)).where(
                traffic_windows.c.user_id == user_id, traffic_windows.c.host_id == host_id)
        ).scalar()


def insert_windows(user_id: str, rows: list[dict]):
    """rows: dicts with host_id, window_idx, observed_at, source, features,
    and optionally true_stage, state_label, prediction."""
    if not rows:
        return
    payload = [{
        "user_id": user_id,
        "host_id": r["host_id"],
        "window_idx": int(r["window_idx"]),
        "observed_at": _aware(r.get("observed_at")) or _now(),
        "source": r["source"],
        "features": {k: float(r["features"][k]) for k in FEATURE_COLUMNS},
        "true_stage": r.get("true_stage"),
        "state_label": r.get("state_label"),
        "prediction": r.get("prediction"),
    } for r in rows]
    with engine().begin() as conn:
        for i in range(0, len(payload), 500):
            conn.execute(insert(traffic_windows), payload[i:i + 500])


def host_frame(user_id: str, host_id: str, upto_window_idx: int | None = None,
               last_n: int | None = None) -> pd.DataFrame:
    """The host's windows in time order as a DataFrame with host_id,
    window_idx, observed_at, source, true_stage, state_label, prediction and
    one column per FEATURE_COLUMNS entry. Empty DataFrame if unknown."""
    t = traffic_windows
    q = select(t.c.host_id, t.c.window_idx, t.c.observed_at, t.c.source, t.c.features,
               t.c.true_stage, t.c.state_label, t.c.prediction).where(
        t.c.user_id == user_id, t.c.host_id == host_id)
    if upto_window_idx is not None:
        q = q.where(t.c.window_idx <= upto_window_idx)
    if last_n is not None:
        q = q.order_by(t.c.window_idx.desc()).limit(last_n)
    else:
        q = q.order_by(t.c.window_idx)
    with engine().connect() as conn:
        rows = conn.execute(q).mappings().all()
    if not rows:
        return pd.DataFrame(columns=["host_id", "window_idx", "observed_at", "source", "true_stage",
                                     "state_label", "prediction"] + FEATURE_COLUMNS)
    recs = []
    for r in rows:
        rec = {"host_id": r["host_id"], "window_idx": r["window_idx"], "observed_at": _iso(r["observed_at"]),
               "source": r["source"], "true_stage": r["true_stage"], "state_label": r["state_label"],
               "prediction": r["prediction"]}
        feats = r["features"] or {}
        for c in FEATURE_COLUMNS:
            rec[c] = float(feats.get(c, 0.0))
        recs.append(rec)
    df = pd.DataFrame(recs)
    return df.sort_values("window_idx").reset_index(drop=True)


def list_hosts(user_id: str) -> list[dict]:
    t = traffic_windows
    with engine().connect() as conn:
        rows = conn.execute(
            select(t.c.host_id, func.count().label("n_windows"), func.max(t.c.observed_at).label("last_seen"),
                   func.min(t.c.source).label("source"))
            .where(t.c.user_id == user_id).group_by(t.c.host_id).order_by(t.c.host_id)
        ).mappings().all()
    return [{"host_id": r["host_id"], "n_windows": int(r["n_windows"]), "last_seen": _iso(r["last_seen"]),
             "source": r["source"]} for r in rows]


def host_timeline(user_id: str, host_id: str) -> list[dict]:
    t = traffic_windows
    with engine().connect() as conn:
        rows = conn.execute(
            select(t.c.window_idx, t.c.true_stage, t.c.state_label, t.c.observed_at)
            .where(t.c.user_id == user_id, t.c.host_id == host_id).order_by(t.c.window_idx)
        ).mappings().all()
    return [{"window_idx": r["window_idx"], "true_stage": r["true_stage"], "state_label": r["state_label"],
             "observed_at": _iso(r["observed_at"])} for r in rows]


def recent_live_windows(user_id: str, limit: int = 200) -> list[dict]:
    """Most recent agent-captured windows, shaped like the old in-memory
    live-capture log entries the UI already understands."""
    t = traffic_windows
    with engine().connect() as conn:
        rows = conn.execute(
            select(t).where(t.c.user_id == user_id, t.c.source.in_(LIVE_SOURCES))
            .order_by(t.c.id.desc()).limit(limit)
        ).mappings().all()
    out = []
    for r in rows:
        p = r["prediction"] or {}
        out.append({
            "host_id": r["host_id"],
            "window_idx": r["window_idx"],
            "timestamp": _iso(r["observed_at"]),
            "raw_features": r["features"],
            "predicted_stage": p.get("predicted_stage"),
            "infiltration_probability_world_model": p.get("infiltration_probability_world_model"),
            "infiltration_probability_baseline": p.get("infiltration_probability_baseline"),
            "stage_probabilities": p.get("stage_probabilities"),
            "warmup": p.get("warmup"),
            "history_windows_used": p.get("history_windows_used"),
            "explanation": p.get("explanation"),
            "attack_mapping": p.get("attack_mapping"),
        })
    return out


def count_hosts(user_id: str) -> int:
    with engine().connect() as conn:
        return int(conn.execute(
            select(func.count(func.distinct(traffic_windows.c.host_id))).where(traffic_windows.c.user_id == user_id)
        ).scalar() or 0)


# ---------------------------------------------------------------------------
# inference log
# ---------------------------------------------------------------------------
def log_inference(user_id: str, host_id: str, window_idx: int, model: str, predicted_stage: str | None,
                  infiltration_probability: float, true_stage: str | None = None,
                  state_label: str | None = None, source: str = "agent"):
    """Logs one real inference call. Skips an exact repeat of the previous row
    for this (user, host, model) -- re-polling a fixed host (e.g. the
    Overview page's 10 s refresh) must not flood the log with copies."""
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


def _latest_world_model_rows(user_id: str):
    il = inference_log
    cutoff = _now() - timedelta(minutes=RECENCY_WINDOW_MINUTES)
    with engine().connect() as conn:
        rows = conn.execute(
            select(il.c.host_id, il.c.window_idx, il.c.predicted_stage, il.c.infiltration_probability,
                   il.c.created_at)
            .where(il.c.user_id == user_id, il.c.model == "world_model_lstm",
                   (il.c.source.notin_(LIVE_SOURCES)) | (il.c.created_at >= cutoff))
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
                   il.c.infiltration_probability, il.c.true_stage, il.c.state_label, il.c.source)
            .where(il.c.user_id == user_id).order_by(il.c.id.desc()).limit(limit)
        ).mappings().all()
    return [{**dict(r), "created_at": _iso(r["created_at"])} for r in rows]


def stage_breakdown(user_id: str, limit: int = 500):
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
def log_tripwire_alert(user_id: str, remote_ip: str, message: str, severity: str, detail: dict,
                       created_at: datetime | None = None, source: str = "agent") -> dict:
    created_at = _aware(created_at) or _now()
    with engine().begin() as conn:
        res = conn.execute(insert(tripwire_alerts).values(
            user_id=user_id, created_at=created_at, remote_ip=remote_ip, message=message[:500],
            severity=severity, detail=detail, source=source,
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


# ---------------------------------------------------------------------------
# packet log
# ---------------------------------------------------------------------------
def insert_packets(user_id: str, packets: list[dict]):
    """packets: dicts with host_id, ts, direction, local_port, remote_port,
    flags, ttl, win_size, pkt_len, description. Keeps only the newest
    PACKET_LOG_PER_HOST rows per host."""
    if not packets:
        return
    rows = [{
        "user_id": user_id, "host_id": p["host_id"], "ts": float(p["ts"]), "direction": str(p["direction"])[:4],
        "local_port": int(p["local_port"]), "remote_port": int(p["remote_port"]), "flags": str(p["flags"])[:16],
        "ttl": int(p["ttl"]), "win_size": int(p["win_size"]), "pkt_len": int(p["pkt_len"]),
        "description": str(p.get("description", ""))[:300],
    } for p in packets]
    hosts = {r["host_id"] for r in rows}
    pl = packet_log
    with engine().begin() as conn:
        for i in range(0, len(rows), 500):
            conn.execute(insert(pl), rows[i:i + 500])
        for h in hosts:
            cutoff = conn.execute(
                select(pl.c.id).where(pl.c.user_id == user_id, pl.c.host_id == h)
                .order_by(pl.c.id.desc()).offset(PACKET_LOG_PER_HOST - 1).limit(1)
            ).scalar()
            if cutoff is not None:
                conn.execute(delete(pl).where(pl.c.user_id == user_id, pl.c.host_id == h, pl.c.id < cutoff))


def recent_packets(user_id: str, host_id: str, limit: int = 100) -> list[dict]:
    pl = packet_log
    with engine().connect() as conn:
        rows = conn.execute(
            select(pl).where(pl.c.user_id == user_id, pl.c.host_id == host_id).order_by(pl.c.id.desc()).limit(limit)
        ).mappings().all()
    return [{"timestamp": r["ts"], "direction": r["direction"], "local_port": r["local_port"],
             "remote_port": r["remote_port"], "flags": r["flags"], "ttl": r["ttl"], "win_size": r["win_size"],
             "pkt_len": r["pkt_len"], "description": r["description"]} for r in rows]


# ---------------------------------------------------------------------------
# sensors (capture agents)
# ---------------------------------------------------------------------------
TOKEN_PREFIX = "nadf_"


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _sensor_dict(r) -> dict:
    last_seen = _aware(r["last_seen_at"])
    online = bool(last_seen and (_now() - last_seen).total_seconds() <= SENSOR_ONLINE_SECONDS)
    return {"id": r["id"], "name": r["name"], "token_hint": r["token_hint"], "created_at": _iso(r["created_at"]),
            "last_seen_at": _iso(last_seen), "online": online, "hostname": r["hostname"], "local_ip": r["local_ip"],
            "iface": r["iface"], "os": r["os"], "agent_version": r["agent_version"],
            "packets_seen": int(r["packets_seen"] or 0), "error": r["error"]}


def create_sensor(user_id: str, name: str) -> tuple[dict, str]:
    """Returns (sensor, plaintext_token). The token is shown to the user once;
    only its SHA-256 hash is stored."""
    token = TOKEN_PREFIX + secrets.token_urlsafe(32)
    sid = uuid.uuid4().hex
    with engine().begin() as conn:
        conn.execute(insert(sensors).values(
            id=sid, user_id=user_id, name=(name or "sensor")[:100], token_hash=_hash_token(token),
            token_hint=token[-6:], created_at=_now(), packets_seen=0,
        ))
        row = conn.execute(select(sensors).where(sensors.c.id == sid)).mappings().first()
    return _sensor_dict(row), token


def list_sensors(user_id: str) -> list[dict]:
    with engine().connect() as conn:
        rows = conn.execute(
            select(sensors).where(sensors.c.user_id == user_id).order_by(sensors.c.created_at)
        ).mappings().all()
    return [_sensor_dict(r) for r in rows]


def delete_sensor(user_id: str, sensor_id: str) -> bool:
    with engine().begin() as conn:
        res = conn.execute(delete(sensors).where(sensors.c.user_id == user_id, sensors.c.id == sensor_id))
    return res.rowcount > 0


def sensor_by_token(token: str) -> dict | None:
    if not token or not token.startswith(TOKEN_PREFIX):
        return None
    with engine().connect() as conn:
        row = conn.execute(select(sensors).where(sensors.c.token_hash == _hash_token(token))).mappings().first()
    if row is None:
        return None
    return {**_sensor_dict(row), "user_id": row["user_id"]}


def touch_sensor(sensor_id: str, **fields):
    """Marks the sensor as seen now and updates any reported fields.
    `error=None` explicitly clears a previously reported error."""
    allowed = {"hostname", "local_ip", "iface", "os", "agent_version", "packets_seen"}
    values = {k: v for k, v in fields.items() if k in allowed and v is not None}
    if "error" in fields:
        values["error"] = (fields["error"] or None) and str(fields["error"])[:500]
    values["last_seen_at"] = _now()
    with engine().begin() as conn:
        conn.execute(update(sensors).where(sensors.c.id == sensor_id).values(**values))


# ---------------------------------------------------------------------------
# workspace
# ---------------------------------------------------------------------------
def workspace_counts(user_id: str) -> dict:
    out = {}
    with engine().connect() as conn:
        for t in ALL_USER_TABLES + (sensors,):
            out[t.name] = int(conn.execute(
                select(func.count()).select_from(t).where(t.c.user_id == user_id)).scalar() or 0)
    return out


def delete_hosts(user_id: str, host_ids: list[str] | None = None, source: str | None = None) -> int:
    """Deletes windows/logs/packets for the given hosts (or every host with
    the given source). Returns number of windows deleted."""
    tw = traffic_windows
    with engine().begin() as conn:
        if host_ids is None:
            q = select(func.distinct(tw.c.host_id)).where(tw.c.user_id == user_id)
            if source:
                q = q.where(tw.c.source == source)
            host_ids = [h for (h,) in conn.execute(q).all()]
        if not host_ids:
            return 0
        n = conn.execute(delete(tw).where(tw.c.user_id == user_id, tw.c.host_id.in_(host_ids))).rowcount
        conn.execute(delete(inference_log).where(inference_log.c.user_id == user_id,
                                                 inference_log.c.host_id.in_(host_ids)))
        conn.execute(delete(packet_log).where(packet_log.c.user_id == user_id, packet_log.c.host_id.in_(host_ids)))
    return int(n or 0)


def reset_workspace(user_id: str, include_sensors: bool = False):
    with engine().begin() as conn:
        for t in ALL_USER_TABLES:
            conn.execute(delete(t).where(t.c.user_id == user_id))
        if include_sensors:
            conn.execute(delete(sensors).where(sensors.c.user_id == user_id))
