from app.live.capture import LiveCaptureManager


def test_start_clears_stale_recent_predictions_from_a_previous_session():
    """Regression test for a real bug found during live testing: stopping and
    restarting capture reset window counters and history, but left old
    prediction-log entries from the PREVIOUS session in place, so a host's
    windows from an earlier session appeared mixed in with its new session's
    windows under the same window_idx numbering."""
    mgr = LiveCaptureManager()
    mgr.recent_predictions.append({"host_id": "live:1.2.3.4", "window_idx": 5, "stale": True})
    assert len(mgr.recent_predictions) == 1

    try:
        mgr.start(iface="nonexistent-iface-for-test", local_ip="0.0.0.0")
        assert len(mgr.recent_predictions) == 0
    finally:
        mgr.stop()


def test_start_resets_history_and_window_counters():
    mgr = LiveCaptureManager()
    mgr.history = {"1.2.3.4": ["stale"]}
    mgr.window_counter = {"1.2.3.4": 5}
    mgr.packets_seen = 999

    try:
        mgr.start(iface="nonexistent-iface-for-test", local_ip="0.0.0.0")
        assert mgr.history == {}
        assert mgr.window_counter == {}
        assert mgr.packets_seen == 0
    finally:
        mgr.stop()


def test_cannot_start_twice_while_running():
    mgr = LiveCaptureManager()
    try:
        mgr.start(iface="nonexistent-iface-for-test", local_ip="0.0.0.0")
        try:
            mgr.start(iface="nonexistent-iface-for-test", local_ip="0.0.0.0")
            assert False, "expected RuntimeError"
        except RuntimeError:
            pass
    finally:
        mgr.stop()
