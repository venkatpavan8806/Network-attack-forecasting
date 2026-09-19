from app.live.tripwire import Tripwire, MULTI_PORT_THRESHOLD, WATCHED_PORT_COOLDOWN_SECONDS


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
    # spread the same number of distinct ports out over a much longer time span
    for i, port in enumerate(range(9000, 9000 + MULTI_PORT_THRESHOLD)):
        fired_any.extend(tw.on_inbound_syn(remote_ip="10.0.0.9", local_port=port, ts=i * 10.0))
    assert not any(f["detail"]["rule"] == "rapid_multi_port" for f in fired_any)


def test_alerts_have_millisecond_precision_timestamps():
    tw = Tripwire()
    fired = tw.on_inbound_syn(remote_ip="10.0.0.9", local_port=22, ts=0.0)
    ts = fired[0]["timestamp"]
    assert "." in ts  # ISO timestamp with sub-second precision present


def test_reset_clears_state():
    tw = Tripwire()
    tw.on_inbound_syn(remote_ip="10.0.0.9", local_port=22, ts=0.0)
    assert len(tw.recent()) > 0
    tw.reset()
    assert len(tw.recent()) == 0
    # cooldown state also cleared -- same port immediately re-fires
    fired = tw.on_inbound_syn(remote_ip="10.0.0.9", local_port=22, ts=0.05)
    assert any(f["detail"]["rule"] == "watched_port_contact" for f in fired)


def test_recent_returns_newest_first():
    tw = Tripwire()
    tw.on_inbound_syn(remote_ip="10.0.0.9", local_port=22, ts=0.0)
    tw.on_inbound_syn(remote_ip="10.0.0.8", local_port=445, ts=1.0)
    items = tw.recent()
    assert items[0]["remote_ip"] == "10.0.0.8"
    assert items[1]["remote_ip"] == "10.0.0.9"
