import pytest

from app import db
from tests.conftest import make_test_db
from app.live.tripwire import Tripwire, MULTI_PORT_THRESHOLD, WATCHED_PORT_COOLDOWN_SECONDS


@pytest.fixture()
def isolated_db(tmp_path):
    yield make_test_db(tmp_path)


def _db_sink(user_id):
    """The same persistence the backend applies to alerts an agent uploads."""
    def sink(remote_ip, message, severity, detail, ts):
        return db.log_tripwire_alert(user_id, remote_ip, message, severity, detail)
    return sink


def test_watched_port_contact_fires_immediately():
    tw = Tripwire()
    fired = tw.on_inbound_syn(remote_ip="10.0.0.9", local_port=22, ts=0.0)
    assert len(fired) == 1
    assert fired[0]["detail"]["rule"] == "watched_port_contact"
    assert fired[0]["detail"]["service"] == "SSH"


def test_non_watched_port_does_not_fire_watched_alert():
    tw = Tripwire()
    fired = tw.on_inbound_syn(remote_ip="10.0.0.9", local_port=8080, ts=0.0)
    assert not any(f["detail"]["rule"] == "watched_port_contact" for f in fired)


def test_watched_port_alert_has_cooldown():
    tw = Tripwire()
    first = tw.on_inbound_syn(remote_ip="10.0.0.9", local_port=22, ts=0.0)
    again_immediately = tw.on_inbound_syn(remote_ip="10.0.0.9", local_port=22, ts=0.1)
    after_cooldown = tw.on_inbound_syn(remote_ip="10.0.0.9", local_port=22, ts=WATCHED_PORT_COOLDOWN_SECONDS + 0.1)
    assert any(f["detail"]["rule"] == "watched_port_contact" for f in first)
    assert not any(f["detail"]["rule"] == "watched_port_contact" for f in again_immediately)
    assert any(f["detail"]["rule"] == "watched_port_contact" for f in after_cooldown)


def test_rapid_multi_port_fires_at_threshold():
    tw = Tripwire()
    fired_any = []
    for i, port in enumerate(range(9000, 9000 + MULTI_PORT_THRESHOLD)):
        fired_any.extend(tw.on_inbound_syn(remote_ip="10.0.0.9", local_port=port, ts=i * 0.1))
    assert any(f["detail"]["rule"] == "rapid_multi_port" for f in fired_any)


def test_rapid_multi_port_does_not_fire_below_threshold():
    tw = Tripwire()
    fired_any = []
    for i, port in enumerate(range(9000, 9000 + MULTI_PORT_THRESHOLD - 1)):
        fired_any.extend(tw.on_inbound_syn(remote_ip="10.0.0.9", local_port=port, ts=i * 0.1))
    assert not any(f["detail"]["rule"] == "rapid_multi_port" for f in fired_any)


def test_rapid_multi_port_ignores_old_packets_outside_rolling_window():
    tw = Tripwire()
    fired_any = []
    for i, port in enumerate(range(9000, 9000 + MULTI_PORT_THRESHOLD)):
        fired_any.extend(tw.on_inbound_syn(remote_ip="10.0.0.9", local_port=port, ts=i * 10.0))
    assert not any(f["detail"]["rule"] == "rapid_multi_port" for f in fired_any)


def test_alerts_have_millisecond_precision_timestamps():
    tw = Tripwire()
    fired = tw.on_inbound_syn(remote_ip="10.0.0.9", local_port=22, ts=1_700_000_000.123)
    assert "." in fired[0]["timestamp"]


def test_alerts_persist_across_tripwire_instances(isolated_db):
    """Alerts stored through the database sink survive an agent restart (a
    new Tripwire instance) -- they are an event log, not session state."""
    Tripwire(sink=_db_sink("alice")).on_inbound_syn(remote_ip="10.0.0.9", local_port=22, ts=0.0)
    Tripwire(sink=_db_sink("alice"))  # fresh capture session
    assert len(db.recent_tripwire_alerts("alice")) == 1
    assert db.recent_tripwire_alerts("bob") == []


def test_reset_does_not_delete_alert_history(isolated_db):
    tw = Tripwire(sink=_db_sink("alice"))
    tw.on_inbound_syn(remote_ip="10.0.0.9", local_port=22, ts=0.0)
    tw.reset()
    assert len(db.recent_tripwire_alerts("alice")) == 1


def test_reset_clears_cooldown_state_allowing_immediate_refire():
    tw = Tripwire()
    tw.on_inbound_syn(remote_ip="10.0.0.9", local_port=22, ts=0.0)
    tw.reset()
    fired = tw.on_inbound_syn(remote_ip="10.0.0.9", local_port=22, ts=0.05)
    assert any(f["detail"]["rule"] == "watched_port_contact" for f in fired)


def test_recent_returns_newest_first(isolated_db):
    tw = Tripwire(sink=_db_sink("alice"))
    tw.on_inbound_syn(remote_ip="10.0.0.9", local_port=22, ts=0.0)
    tw.on_inbound_syn(remote_ip="10.0.0.8", local_port=445, ts=1.0)
    items = db.recent_tripwire_alerts("alice")
    assert items[0]["remote_ip"] == "10.0.0.8"
    assert items[1]["remote_ip"] == "10.0.0.9"
