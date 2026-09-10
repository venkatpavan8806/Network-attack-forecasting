import numpy as np
import pandas as pd

from app.config import FEATURE_COLUMNS, AMBIGUOUS_LOOKBACK
from app.labeling.state_labeler import derive_state_labels, build_transition_pairs


def _make_fixture_timeline():
    """Small hand-built fixture: one host with 10 benign windows, then 4
    precursor-ish windows with elevated port_scan_score/failed_conn_ratio,
    then reconnaissance onset. A second host stays pure benign throughout."""
    rows = []

    def base_row(host, idx, stage, **overrides):
        row = {c: 0.01 for c in FEATURE_COLUMNS}
        row.update({
            "port_scan_score": 0.03, "failed_conn_ratio": 0.02, "new_dst_ip_ratio": 0.05,
            "iat_mean": 0.15, "iat_std": 0.05,
        })
        row.update(overrides)
        row["host_id"] = host
        row["window_idx"] = idx
        row["true_stage"] = stage
        return row

    for i in range(10):
        rows.append(base_row("host-A", i, "benign"))
    for i in range(10, 10 + AMBIGUOUS_LOOKBACK):
        rows.append(base_row("host-A", i, "benign", port_scan_score=0.6, failed_conn_ratio=0.5,
                              new_dst_ip_ratio=0.4, iat_std=0.5))
    for i in range(10 + AMBIGUOUS_LOOKBACK, 10 + AMBIGUOUS_LOOKBACK + 5):
        rows.append(base_row("host-A", i, "port_scan", port_scan_score=0.8, failed_conn_ratio=0.7))

    for i in range(15):
        rows.append(base_row("host-B", i, "benign"))

    return pd.DataFrame(rows)


def test_ambiguous_windows_are_relabeled():
    df = _make_fixture_timeline()
    labeled = derive_state_labels(df)
    host_a = labeled[labeled.host_id == "host-A"].sort_values("window_idx")
    precursor_rows = host_a[(host_a.window_idx >= 10) & (host_a.window_idx < 10 + AMBIGUOUS_LOOKBACK)]
    assert (precursor_rows["state_label"] == "ambiguous_pre_attack").all()


def test_pure_benign_host_never_relabeled():
    df = _make_fixture_timeline()
    labeled = derive_state_labels(df)
    host_b = labeled[labeled.host_id == "host-B"]
    assert (host_b["state_label"] == "benign").all()


def test_early_benign_windows_far_from_onset_stay_benign():
    df = _make_fixture_timeline()
    labeled = derive_state_labels(df)
    host_a = labeled[labeled.host_id == "host-A"].sort_values("window_idx")
    early = host_a[host_a.window_idx < 10]
    assert (early["state_label"] == "benign").all()


def test_hard_malicious_labels_pass_through_unchanged():
    df = _make_fixture_timeline()
    labeled = derive_state_labels(df)
    host_a = labeled[labeled.host_id == "host-A"].sort_values("window_idx")
    recon_rows = host_a[host_a.true_stage == "port_scan"]
    assert (recon_rows["state_label"] == "port_scan").all()


def test_build_transition_pairs_shapes():
    df = _make_fixture_timeline()
    labeled = derive_state_labels(df)
    pairs = build_transition_pairs(labeled)
    # one fewer transition per host than rows per host
    n_rows_a = (labeled.host_id == "host-A").sum()
    n_rows_b = (labeled.host_id == "host-B").sum()
    assert len(pairs) == (n_rows_a - 1) + (n_rows_b - 1)
    for c in FEATURE_COLUMNS:
        assert c in pairs.columns
        assert f"next_{c}" in pairs.columns
    assert "next_state_label" in pairs.columns
