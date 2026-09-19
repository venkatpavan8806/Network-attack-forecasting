import numpy as np
import pandas as pd

from app.config import FEATURE_COLUMNS
from app.data_gen.generator import generate_dataset, generate_host_timeline


def test_generate_dataset_shape_and_columns():
    df = generate_dataset(seed=1, n_benign_hosts=3, n_attack_hosts=2, benign_len=20)
    assert len(df) > 0
    expected_cols = {"host_id", "window_idx", "timestamp", "true_stage"} | set(FEATURE_COLUMNS)
    assert expected_cols.issubset(set(df.columns))
    assert df[FEATURE_COLUMNS].isna().sum().sum() == 0


def test_no_stage_leakage_into_feature_columns():
    """The ground-truth stage must never be encoded (one-hot or otherwise)
    inside the feature columns the model trains on."""
    df = generate_dataset(seed=2, n_benign_hosts=2, n_attack_hosts=1, benign_len=15)
    for col in FEATURE_COLUMNS:
        assert "stage" not in col.lower()
    # true_stage must be a distinct column, not derivable via an exact 1:1 encoded feature
    assert "true_stage" not in FEATURE_COLUMNS


def test_attack_hosts_cover_full_progression():
    from app.data_gen.generator import BRUTEFORCE_ACTIONS, LATERAL_ACTIONS

    df = generate_dataset(seed=3, n_benign_hosts=1, n_attack_hosts=2, benign_len=10)
    attack_actions = set(df[df.host_id.str.startswith("attack")]["true_stage"].unique())
    assert "port_scan" in attack_actions
    assert attack_actions & set(BRUTEFORCE_ACTIONS)
    assert attack_actions & set(LATERAL_ACTIONS)
    assert "c2_beacon" in attack_actions
    assert "data_exfiltration" in attack_actions


def test_evasive_recon_has_lower_flow_volume_than_fast_recon():
    rng_evasive = np.random.default_rng(10)
    rng_fast = np.random.default_rng(11)
    evasive_df = generate_host_timeline("h-evasive", rng_evasive, is_attack=True, evasive=True)
    fast_df = generate_host_timeline("h-fast", rng_fast, is_attack=True, evasive=False)

    evasive_recon = evasive_df[evasive_df.true_stage == "port_scan"]
    fast_recon = fast_df[fast_df.true_stage == "port_scan"]
    assert evasive_recon["flow_count"].mean() < fast_recon["flow_count"].mean()
    # evasive recon should span many more windows (spread thin) than fast recon
    assert len(evasive_recon) > len(fast_recon)


def test_benign_traffic_has_low_port_scan_score():
    df = generate_dataset(seed=4, n_benign_hosts=3, n_attack_hosts=0, benign_len=50)
    assert df["port_scan_score"].mean() < 0.2
