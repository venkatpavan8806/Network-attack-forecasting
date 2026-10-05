from datetime import datetime, timedelta, timezone

import pytest

from app import db


@pytest.fixture()
def isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test_state.sqlite3")
    db.init_db()
    yield db


def _log_at(host_id, prob, minutes_ago, source="live_capture", model="world_model_lstm"):
    with db.get_conn() as conn:
        ts = (datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)).isoformat()
        conn.execute(
            "INSERT INTO inference_log (host_id, window_idx, created_at, model, predicted_stage, "
            "infiltration_probability, true_stage, state_label, source) VALUES (?,?,?,?,?,?,?,?,?)",
            (host_id, 1, ts, model, "port_scan", prob, None, None, source),
        )


def test_stale_live_capture_row_excluded_from_highest_risk(isolated_db):
    _log_at("live:1.2.3.4", 0.99, minutes_ago=20, source="live_capture")  # older than the 15-min window
    assert db.highest_risk_host() is None


def test_fresh_live_capture_row_included_in_highest_risk(isolated_db):
    _log_at("live:1.2.3.4", 0.99, minutes_ago=2, source="live_capture")
    result = db.highest_risk_host()
    assert result is not None
    assert result["host_id"] == "live:1.2.3.4"


def test_stale_seed_row_still_counts_as_a_fixed_reference_result(isolated_db):
    """Seed/ingest rows are one-off reference results, not a continuous
    monitoring claim -- they must not age out the way live captures do."""
    _log_at("attack-host-002", 0.95, minutes_ago=999, source="seed")
    result = db.highest_risk_host()
    assert result is not None
    assert result["host_id"] == "attack-host-002"


def test_stale_live_capture_row_excluded_from_high_risk_count(isolated_db):
    _log_at("live:1.2.3.4", 0.99, minutes_ago=20, source="live_capture")
    assert db.count_high_risk_hosts(threshold=0.5) == 0


def test_fresh_live_capture_row_included_in_high_risk_count(isolated_db):
    _log_at("live:1.2.3.4", 0.99, minutes_ago=1, source="live_capture")
    assert db.count_high_risk_hosts(threshold=0.5) == 1


def test_highest_risk_picks_the_highest_probability_among_fresh_hosts(isolated_db):
    _log_at("live:1.1.1.1", 0.40, minutes_ago=1, source="live_capture")
    _log_at("live:2.2.2.2", 0.85, minutes_ago=1, source="live_capture")
    result = db.highest_risk_host()
    assert result["host_id"] == "live:2.2.2.2"


def _count_rows(host_id: str) -> int:
    with db.get_conn() as conn:
        return conn.execute(
            "SELECT COUNT(*) as c FROM inference_log WHERE host_id = ?", (host_id,)
        ).fetchone()["c"]


def test_log_inference_skips_exact_repeat_for_same_host_and_model(isolated_db):
    """A demo host's prediction is deterministic, so re-polling the same
    window (e.g. Overview's 10s refresh) must not spam the log."""
    for _ in range(3):
        db.log_inference("attack-host-000", 129, "world_model_lstm", "benign", 0.42, source="live")
    assert _count_rows("attack-host-000") == 1


def test_log_inference_does_not_skip_across_different_models(isolated_db):
    """Regression guard: one forecast call logs both world_model_lstm AND
    baseline_logreg rows for the same window. Comparing a new row only
    against the single most recent row (rather than the most recent row for
    the SAME model) would see the other model's row and never match, so
    every poll would still insert a fresh duplicate pair -- this is exactly
    what shipped and was caught by hand in the browser, not by a test."""
    for _ in range(3):
        db.log_inference("attack-host-000", 129, "world_model_lstm", "benign", 0.42, source="live")
        db.log_inference("attack-host-000", 129, "baseline_logreg", None, 0.0, source="live")
    assert _count_rows("attack-host-000") == 2  # one world_model_lstm row, one baseline_logreg row


def test_log_inference_does_not_skip_a_genuinely_new_window(isolated_db):
    db.log_inference("attack-host-000", 74, "world_model_lstm", "data_exfiltration", 0.999, source="live")
    db.log_inference("attack-host-000", 129, "world_model_lstm", "benign", 0.42, source="live")
    assert _count_rows("attack-host-000") == 2
