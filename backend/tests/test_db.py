from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import insert, select, func

from app import db
from tests.conftest import make_test_db

ALICE, BOB = "alice", "bob"


@pytest.fixture()
def isolated_db(tmp_path):
    yield make_test_db(tmp_path)


def _log_at(host_id, prob, minutes_ago, source="agent", model="world_model_lstm", user=ALICE):
    ts = datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)
    with db.engine().begin() as conn:
        conn.execute(insert(db.inference_log).values(
            user_id=user, host_id=host_id, window_idx=1, created_at=ts, model=model, predicted_stage="port_scan",
            infiltration_probability=prob, true_stage=None, state_label=None, source=source))


def _count_rows(host_id: str, user=ALICE) -> int:
    with db.engine().connect() as conn:
        return conn.execute(select(func.count()).select_from(db.inference_log).where(
            db.inference_log.c.host_id == host_id, db.inference_log.c.user_id == user)).scalar()


def test_stale_live_row_excluded_from_highest_risk(isolated_db):
    _log_at("live:1.2.3.4", 0.99, minutes_ago=20)  # older than the 15-min window
    assert db.highest_risk_host(ALICE) is None


def test_fresh_live_row_included_in_highest_risk(isolated_db):
    _log_at("live:1.2.3.4", 0.99, minutes_ago=2)
    result = db.highest_risk_host(ALICE)
    assert result is not None
    assert result["host_id"] == "live:1.2.3.4"


def test_stale_upload_row_still_counts_as_a_fixed_reference_result(isolated_db):
    """Upload/sample rows are one-off results from a file, not a continuous
    monitoring claim -- they must not age out the way live captures do."""
    _log_at("pcap:10.0.0.5", 0.95, minutes_ago=999, source="pcap")
    result = db.highest_risk_host(ALICE)
    assert result is not None
    assert result["host_id"] == "pcap:10.0.0.5"


def test_stale_live_row_excluded_from_high_risk_count(isolated_db):
    _log_at("live:1.2.3.4", 0.99, minutes_ago=20)
    assert db.count_high_risk_hosts(ALICE, threshold=0.5) == 0


def test_fresh_live_row_included_in_high_risk_count(isolated_db):
    _log_at("live:1.2.3.4", 0.99, minutes_ago=1)
    assert db.count_high_risk_hosts(ALICE, threshold=0.5) == 1


def test_highest_risk_picks_the_highest_probability_among_fresh_hosts(isolated_db):
    _log_at("live:1.1.1.1", 0.40, minutes_ago=1)
    _log_at("live:2.2.2.2", 0.85, minutes_ago=1)
    assert db.highest_risk_host(ALICE)["host_id"] == "live:2.2.2.2"


def test_users_never_see_each_others_rows(isolated_db):
    _log_at("live:9.9.9.9", 0.99, minutes_ago=1, user=BOB)
    assert db.highest_risk_host(ALICE) is None
    assert db.count_high_risk_hosts(ALICE) == 0
    assert db.recent_forecast_log(ALICE) == []
    assert db.highest_risk_host(BOB)["host_id"] == "live:9.9.9.9"


def test_log_inference_skips_exact_repeat_for_same_host_and_model(isolated_db):
    """Re-polling the same window (e.g. Overview's 10 s refresh) must not spam the log."""
    for _ in range(3):
        db.log_inference(ALICE, "sample-a", 129, "world_model_lstm", "benign", 0.42, source="sample")
    assert _count_rows("sample-a") == 1


def test_log_inference_does_not_skip_across_different_models(isolated_db):
    """One forecast logs both world_model_lstm AND baseline_logreg rows for
    the same window; dedup must compare against the same model's last row."""
    for _ in range(3):
        db.log_inference(ALICE, "sample-a", 129, "world_model_lstm", "benign", 0.42, source="sample")
        db.log_inference(ALICE, "sample-a", 129, "baseline_logreg", None, 0.0, source="sample")
    assert _count_rows("sample-a") == 2


def test_log_inference_does_not_skip_a_genuinely_new_window(isolated_db):
    db.log_inference(ALICE, "sample-a", 74, "world_model_lstm", "data_exfiltration", 0.999, source="sample")
    db.log_inference(ALICE, "sample-a", 129, "world_model_lstm", "benign", 0.42, source="sample")
    assert _count_rows("sample-a") == 2


def test_log_inference_dedup_is_per_user(isolated_db):
    db.log_inference(ALICE, "sample-a", 1, "world_model_lstm", "benign", 0.1, source="sample")
    db.log_inference(BOB, "sample-a", 1, "world_model_lstm", "benign", 0.1, source="sample")
    assert _count_rows("sample-a", ALICE) == 1
    assert _count_rows("sample-a", BOB) == 1


def test_packet_log_is_capped_per_host(isolated_db):
    pkts = [{"host_id": "live:1.1.1.1", "ts": float(i), "direction": "in", "local_port": 22, "remote_port": 5000,
             "flags": "S", "ttl": 64, "win_size": 1024, "pkt_len": 60, "description": "syn"}
            for i in range(db.PACKET_LOG_PER_HOST + 50)]
    db.insert_packets(ALICE, pkts)
    recent = db.recent_packets(ALICE, "live:1.1.1.1", limit=1000)
    assert len(recent) == db.PACKET_LOG_PER_HOST
    assert recent[0]["timestamp"] == float(db.PACKET_LOG_PER_HOST + 49)  # newest kept, oldest dropped


def test_sensor_token_is_stored_hashed_and_resolves_to_owner(isolated_db):
    sensor, token = db.create_sensor(ALICE, "laptop")
    assert token.startswith(db.TOKEN_PREFIX)
    with db.engine().connect() as conn:
        stored = conn.execute(select(db.sensors.c.token_hash)).scalar()
    assert token not in stored
    found = db.sensor_by_token(token)
    assert found["user_id"] == ALICE and found["id"] == sensor["id"]
    assert db.sensor_by_token(token + "x") is None
    assert db.delete_sensor(BOB, sensor["id"]) is False  # cannot revoke someone else's sensor
    assert db.delete_sensor(ALICE, sensor["id"]) is True
    assert db.sensor_by_token(token) is None


def test_sensor_online_status_follows_last_seen(isolated_db):
    sensor, _ = db.create_sensor(ALICE, "laptop")
    assert db.list_sensors(ALICE)[0]["online"] is False
    db.touch_sensor(sensor["id"], hostname="pc", packets_seen=10)
    s = db.list_sensors(ALICE)[0]
    assert s["online"] is True and s["hostname"] == "pc" and s["packets_seen"] == 10


def test_reset_workspace_only_touches_the_caller(isolated_db):
    _log_at("live:1.1.1.1", 0.9, minutes_ago=1, user=ALICE)
    _log_at("live:2.2.2.2", 0.9, minutes_ago=1, user=BOB)
    db.reset_workspace(ALICE)
    assert db.workspace_counts(ALICE)["inference_log"] == 0
    assert db.workspace_counts(BOB)["inference_log"] == 1
