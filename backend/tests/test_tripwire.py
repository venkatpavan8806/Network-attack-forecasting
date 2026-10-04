import pytest

from app import db
from app.live.tripwire import Tripwire, MULTI_PORT_THRESHOLD, WATCHED_PORT_COOLDOWN_SECONDS


@pytest.fixture()
def isolated_db(tmp_path, monkeypatch):
    db.configure(f"sqlite:///{(tmp_path / 'test_tripwire.sqlite3').as_posix()}")
    db.init_db()
    yield db



def _tw():
    tw = Tripwire()
    tw.owner = "alice"  # alerts belong to the user who started the live capture
    return tw


def test_alerts_are_private_to_the_capture_owner(isolated_db):
    _tw().on_inbound_syn(remote_ip="10.0.0.9", local_port=22, ts=0.0)
    assert len(db.recent_tripwire_alerts("alice")) == 1
    assert db.recent_tripwire_alerts("bob") == []


def test_watched_port_contact_fires_immediately(isolated_db):
    tw = _tw()
    fired = tw.on_inbound_syn(remote_ip="10.0.0.9", local_port=22, ts=0.0)
    assert len(fired) == 1
    assert fired[0]["detail"]["rule"] == "watched_port_contact"
    assert fired[0]["detail"]["service"] == "SSH"


def test_non_watched_port_does_not_fire_watched_alert(isolated_db):
    tw = _tw()
    fired = tw.on_inbound_syn(remote_ip="10.0.0.9", local_port=8080, ts=0.0)
    assert not any(f["detail"]["rule"] == "watched_port_contact" for f in fired)


def test_watched_port_alert_has_cooldown(isolated_db):
    tw = _tw()
    first = tw.on_inbound_syn(remote_ip="10.0.0.9", local_port=22, ts=0.0)
    again_immediately = tw.on_inbound_syn(remote_ip="10.0.0.9", local_port=22, ts=0.1)
    after_cooldown = tw.on_inbound_syn(remote_ip="10.0.0.9", local_port=22, ts=WATCHED_PORT_COOLDOWN_SECONDS + 0.1)
    assert any(f["detail"]["rule"] == "watched_port_contact" for f in first)
    assert not any(f["detail"]["rule"] == "watched_port_contact" for f in again_immediately)
    assert any(f["detail"]["rule"] == "watched_port_contact" for f in after_cooldown)


def test_rapid_multi_port_fires_at_threshold(isolated_db):
    tw = _tw()
    fired_any = []
    for i, port in enumerate(range(9000, 9000 + MULTI_PORT_THRESHOLD)):
        fired_any.extend(tw.on_inbound_syn(remote_ip="10.0.0.9", local_port=port, ts=i * 0.1))
    assert any(f["detail"]["rule"] == "rapid_multi_port" for f in fired_any)


def test_rapid_multi_port_does_not_fire_below_threshold(isolated_db):
    tw = _tw()
    fired_any = []
    for i, port in enumerate(range(9000, 9000 + MULTI_PORT_THRESHOLD - 1)):
        fired_any.extend(tw.on_inbound_syn(remote_ip="10.0.0.9", local_port=port, ts=i * 0.1))
    assert not any(f["detail"]["rule"] == "rapid_multi_port" for f in fired_any)


def test_rapid_multi_port_ignores_old_packets_outside_rolling_window(isolated_db):
    tw = _tw()
    fired_any = []
    for i, port in enumerate(range(9000, 9000 + MULTI_PORT_THRESHOLD)):
        fired_any.extend(tw.on_inbound_syn(remote_ip="10.0.0.9", local_port=port, ts=i * 10.0))
    assert not any(f["detail"]["rule"] == "rapid_multi_port" for f in fired_any)


def test_alerts_have_millisecond_precision_timestamps(isolated_db):
    tw = _tw()
    fired = tw.on_inbound_syn(remote_ip="10.0.0.9", local_port=22, ts=0.0)
    ts = fired[0]["timestamp"]
    assert "." in ts  # ISO timestamp with sub-second precision present


def test_alerts_persist_across_tripwire_instances(isolated_db):
    """Regression test: alerts must survive a live-capture restart (a new
    Tripwire instance), since they're a real event log now, not in-memory
    session state."""
    tw1 = _tw()
    tw1.on_inbound_syn(remote_ip="10.0.0.9", local_port=22, ts=0.0)

    tw2 = _tw()  # simulates a fresh capture session after restart
    assert len(tw2.recent("alice")) == 1


def test_reset_does_not_delete_alert_history(isolated_db):
    """reset() clears in-process rule state (cooldowns) but must NOT wipe
    the persisted alert log -- that's a real gap this was built to fix."""
    tw = _tw()
    tw.on_inbound_syn(remote_ip="10.0.0.9", local_port=22, ts=0.0)
    assert len(tw.recent("alice")) == 1
    tw.reset()
    assert len(tw.recent("alice")) == 1


def test_reset_clears_cooldown_state_allowing_immediate_refire(isolated_db):
    tw = _tw()
    tw.on_inbound_syn(remote_ip="10.0.0.9", local_port=22, ts=0.0)
    tw.reset()
    # immediately after reset, same port/host should fire again despite the cooldown window
    fired = tw.on_inbound_syn(remote_ip="10.0.0.9", local_port=22, ts=0.05)
    assert any(f["detail"]["rule"] == "watched_port_contact" for f in fired)


def test_recent_returns_newest_first(isolated_db):
    tw = _tw()
    tw.on_inbound_syn(remote_ip="10.0.0.9", local_port=22, ts=0.0)
    tw.on_inbound_syn(remote_ip="10.0.0.8", local_port=445, ts=1.0)
    items = tw.recent("alice")
    assert items[0]["remote_ip"] == "10.0.0.8"
    assert items[1]["remote_ip"] == "10.0.0.9"
