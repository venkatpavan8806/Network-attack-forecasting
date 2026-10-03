from app.config import FEATURE_COLUMNS
from app.live.flow_tracker import FlowTracker, PacketRecord, describe_packet, PACKET_LOG_MAXLEN


def test_empty_window_returns_no_hosts():
    tracker = FlowTracker(local_ip="10.0.0.5")
    assert tracker.roll_window() == {}


def test_port_scan_shape_produces_high_port_scan_score():
    """A remote host hitting many distinct local ports with few packets each
    (classic scan shape) should score high on port_scan_score and show
    a non-trivial failed_conn_ratio (no completed handshakes)."""
    tracker = FlowTracker(local_ip="10.0.0.5")
    for port in range(20, 40):
        tracker.ingest_tcp(remote_ip="10.0.0.9", direction="in", local_port=port,
                            remote_port=54321, flags="S", ttl=64, win_size=1024, pkt_len=60, ts=port * 0.01)
    result = tracker.roll_window()
    assert "10.0.0.9" in result
    feats = result["10.0.0.9"]
    assert set(feats.keys()) == set(FEATURE_COLUMNS)
    assert feats["unique_dst_ports"] == 20
    assert feats["port_scan_score"] > 0.5
    assert feats["failed_conn_ratio"] == 1.0  # no SYN-ACK replies were ever sent back


def test_completed_handshake_lowers_failed_conn_ratio():
    tracker = FlowTracker(local_ip="10.0.0.5")
    tracker.ingest_tcp(remote_ip="10.0.0.9", direction="in", local_port=22, remote_port=1000,
                        flags="S", ttl=64, win_size=1024, pkt_len=60, ts=0.0)
    tracker.ingest_tcp(remote_ip="10.0.0.9", direction="out", local_port=22, remote_port=1000,
                        flags="SA", ttl=128, win_size=8192, pkt_len=60, ts=0.01)
    tracker.ingest_tcp(remote_ip="10.0.0.9", direction="in", local_port=22, remote_port=1000,
                        flags="A", ttl=64, win_size=1024, pkt_len=60, ts=0.02)
    result = tracker.roll_window()
    feats = result["10.0.0.9"]
    assert feats["failed_conn_ratio"] == 0.0


def test_watched_port_indicators_reflect_real_traffic():
    tracker = FlowTracker(local_ip="10.0.0.5")
    tracker.ingest_tcp(remote_ip="10.0.0.9", direction="in", local_port=22, remote_port=2000,
                        flags="S", ttl=64, win_size=1024, pkt_len=60, ts=0.0)
    result = tracker.roll_window()
    feats = result["10.0.0.9"]
    assert feats["dst_port_is_22"] == 1.0
    assert feats["dst_port_is_445"] == 0.0
    assert feats["dst_port_is_3389"] == 0.0
    assert feats["dst_port_is_443"] == 0.0


def test_unique_dst_ips_is_fixed_at_one_by_design():
    tracker = FlowTracker(local_ip="10.0.0.5")
    tracker.ingest_tcp(remote_ip="10.0.0.9", direction="in", local_port=80, remote_port=3000,
                        flags="S", ttl=64, win_size=1024, pkt_len=60, ts=0.0)
    feats = tracker.roll_window()["10.0.0.9"]
    assert feats["unique_dst_ips"] == 1.0


def test_new_dst_ip_ratio_decays_across_windows_for_same_host():
    tracker = FlowTracker(local_ip="10.0.0.5")
    tracker.ingest_tcp(remote_ip="10.0.0.9", direction="in", local_port=80, remote_port=3000,
                        flags="S", ttl=64, win_size=1024, pkt_len=60, ts=0.0)
    first_ratio = tracker.roll_window()["10.0.0.9"]["new_dst_ip_ratio"]

    tracker.ingest_tcp(remote_ip="10.0.0.9", direction="in", local_port=80, remote_port=3001,
                        flags="S", ttl=64, win_size=1024, pkt_len=60, ts=0.0)
    second_ratio = tracker.roll_window()["10.0.0.9"]["new_dst_ip_ratio"]

    assert first_ratio == 1.0
    assert second_ratio < first_ratio


def test_ttl_and_window_size_pulled_from_real_inbound_packets():
    tracker = FlowTracker(local_ip="10.0.0.5")
    tracker.ingest_tcp(remote_ip="10.0.0.9", direction="in", local_port=80, remote_port=3000,
                        flags="S", ttl=54, win_size=64240, pkt_len=60, ts=0.0)
    feats = tracker.roll_window()["10.0.0.9"]
    assert feats["ttl_mean"] == 54.0
    assert feats["win_size_mean"] == 64240.0


def test_our_own_outbound_browsing_is_never_flagged():
    """Regression test for a real bug found during live testing: visiting a
    website (we send the SYN, they reply) must produce NO feature row for
    that remote host -- it is not evidence of anything they did to us."""
    tracker = FlowTracker(local_ip="10.0.0.5")
    # we open the connection: our outbound SYN first, then their SYN-ACK, then our ACK
    tracker.ingest_tcp(remote_ip="93.184.216.34", direction="out", local_port=51234, remote_port=443,
                        flags="S", ttl=128, win_size=64240, pkt_len=60, ts=1.0)
    tracker.ingest_tcp(remote_ip="93.184.216.34", direction="in", local_port=51234, remote_port=443,
                        flags="SA", ttl=54, win_size=65535, pkt_len=60, ts=1.02)
    tracker.ingest_tcp(remote_ip="93.184.216.34", direction="out", local_port=51234, remote_port=443,
                        flags="A", ttl=128, win_size=64240, pkt_len=60, ts=1.03)
    tracker.ingest_tcp(remote_ip="93.184.216.34", direction="in", local_port=51234, remote_port=443,
                        flags="PA", ttl=54, win_size=65535, pkt_len=1400, ts=1.05)
    result = tracker.roll_window()
    assert "93.184.216.34" not in result


def test_many_ephemeral_ports_from_our_own_browsing_is_not_a_scan():
    """A chatty remote server answering several browser-initiated connections
    (each on a fresh ephemeral local port) must not look like it scanned us,
    even though many distinct local ports are technically touched."""
    tracker = FlowTracker(local_ip="10.0.0.5")
    for i, local_port in enumerate([51000, 51001, 51002, 51003, 51004]):
        ts = i * 0.5
        tracker.ingest_tcp(remote_ip="172.217.0.1", direction="out", local_port=local_port, remote_port=443,
                            flags="S", ttl=128, win_size=64240, pkt_len=60, ts=ts)
        tracker.ingest_tcp(remote_ip="172.217.0.1", direction="in", local_port=local_port, remote_port=443,
                            flags="SA", ttl=54, win_size=65535, pkt_len=60, ts=ts + 0.02)
    result = tracker.roll_window()
    assert "172.217.0.1" not in result


def test_mixed_traffic_only_scores_the_remote_initiated_part():
    """A remote host that BOTH replies to our browsing AND independently
    probes a port on us in the same window should only be scored on the
    genuinely remote-initiated part."""
    tracker = FlowTracker(local_ip="10.0.0.5")
    # our own outbound browsing to this host
    tracker.ingest_tcp(remote_ip="10.0.0.9", direction="out", local_port=52000, remote_port=443,
                        flags="S", ttl=128, win_size=64240, pkt_len=60, ts=1.0)
    tracker.ingest_tcp(remote_ip="10.0.0.9", direction="in", local_port=52000, remote_port=443,
                        flags="SA", ttl=54, win_size=65535, pkt_len=60, ts=1.02)
    # the same host independently probes port 22 on us
    tracker.ingest_tcp(remote_ip="10.0.0.9", direction="in", local_port=22, remote_port=9999,
                        flags="S", ttl=54, win_size=1024, pkt_len=60, ts=2.0)
    result = tracker.roll_window()
    assert "10.0.0.9" in result
    feats = result["10.0.0.9"]
    assert feats["unique_dst_ports"] == 1  # only port 22 counts, not the browsing port
    assert feats["dst_port_is_22"] == 1.0


def test_port_scan_score_not_saturated_by_single_isolated_connection():
    """A single legitimate inbound connection attempt (not a scan) should
    NOT trivially score port_scan_score == 1.0 -- this was the actual bug
    that flagged ordinary low-volume traffic as a port scan."""
    tracker = FlowTracker(local_ip="10.0.0.5")
    tracker.ingest_tcp(remote_ip="10.0.0.9", direction="in", local_port=22, remote_port=1000,
                        flags="S", ttl=64, win_size=1024, pkt_len=60, ts=0.0)
    feats = tracker.roll_window()["10.0.0.9"]
    assert feats["port_scan_score"] < 0.5


def test_zero_timestamp_is_a_valid_timestamp():
    """Regression test: `ts or time.time()` treated ts=0.0 as falsy and
    silently replaced it with the current wall-clock time, corrupting
    chronological ordering. Must use an explicit `is not None` check."""
    tracker = FlowTracker(local_ip="10.0.0.5")
    tracker.ingest_tcp(remote_ip="10.0.0.9", direction="in", local_port=22, remote_port=1000,
                        flags="S", ttl=64, win_size=1024, pkt_len=60, ts=0.0)
    rec = tracker._buckets["10.0.0.9"].records[0]
    assert rec.ts == 0.0


def test_window_clears_after_roll():
    tracker = FlowTracker(local_ip="10.0.0.5")
    tracker.ingest_tcp(remote_ip="10.0.0.9", direction="in", local_port=80, remote_port=3000,
                        flags="S", ttl=64, win_size=1024, pkt_len=60, ts=0.0)
    tracker.roll_window()
    assert tracker.roll_window() == {}


def _rec(**overrides):
    base = dict(ts=0.0, direction="in", local_port=22, remote_port=1000, flags="S", ttl=64, win_size=1024, pkt_len=60)
    base.update(overrides)
    return PacketRecord(**base)


def test_describe_packet_syn():
    desc = describe_packet(_rec(flags="S", direction="in", local_port=22))
    assert "SYN" in desc
    assert "them" in desc.lower()
    assert "[SSH]" in desc


def test_describe_packet_syn_ack():
    desc = describe_packet(_rec(flags="SA"))
    assert "SYN-ACK" in desc


def test_describe_packet_rst():
    desc = describe_packet(_rec(flags="R"))
    assert "reset" in desc.lower() or "refused" in desc.lower()


def test_describe_packet_fin():
    desc = describe_packet(_rec(flags="F"))
    assert "closing" in desc.lower()


def test_describe_packet_data_transfer_includes_byte_count():
    desc = describe_packet(_rec(flags="PA", pkt_len=1400))
    assert "1400 bytes" in desc


def test_describe_packet_unwatched_port_has_no_service_tag():
    desc = describe_packet(_rec(local_port=8080, direction="in"))
    assert "[" not in desc


def test_recent_packets_survives_window_rollover():
    """Unlike the aggregate window features, the raw packet log must NOT be
    cleared when a window rolls over -- it's for independent inspection."""
    tracker = FlowTracker(local_ip="10.0.0.5")
    tracker.ingest_tcp(remote_ip="10.0.0.9", direction="in", local_port=22, remote_port=1000,
                        flags="S", ttl=64, win_size=1024, pkt_len=60, ts=1.0)
    tracker.roll_window()
    packets = tracker.recent_packets("10.0.0.9")
    assert len(packets) == 1
    assert packets[0]["description"]
    assert packets[0]["local_port"] == 22


def test_recent_packets_newest_first():
    tracker = FlowTracker(local_ip="10.0.0.5")
    tracker.ingest_tcp(remote_ip="10.0.0.9", direction="in", local_port=22, remote_port=1000,
                        flags="S", ttl=64, win_size=1024, pkt_len=60, ts=1.0)
    tracker.ingest_tcp(remote_ip="10.0.0.9", direction="in", local_port=445, remote_port=1001,
                        flags="S", ttl=64, win_size=1024, pkt_len=60, ts=2.0)
    packets = tracker.recent_packets("10.0.0.9")
    assert packets[0]["local_port"] == 445
    assert packets[1]["local_port"] == 22


def test_recent_packets_unknown_host_returns_empty():
    tracker = FlowTracker(local_ip="10.0.0.5")
    assert tracker.recent_packets("9.9.9.9") == []


def test_recent_packets_respects_limit():
    tracker = FlowTracker(local_ip="10.0.0.5")
    for i in range(10):
        tracker.ingest_tcp(remote_ip="10.0.0.9", direction="in", local_port=1000 + i, remote_port=2000,
                            flags="S", ttl=64, win_size=1024, pkt_len=60, ts=float(i))
    assert len(tracker.recent_packets("10.0.0.9", limit=3)) == 3


def test_packet_log_is_capped():
    tracker = FlowTracker(local_ip="10.0.0.5")
    for i in range(PACKET_LOG_MAXLEN + 50):
        tracker.ingest_tcp(remote_ip="10.0.0.9", direction="in", local_port=1000, remote_port=2000,
                            flags="A", ttl=64, win_size=1024, pkt_len=60, ts=float(i))
    assert len(tracker.recent_packets("10.0.0.9", limit=PACKET_LOG_MAXLEN + 100)) == PACKET_LOG_MAXLEN
