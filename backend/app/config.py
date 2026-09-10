"""Central configuration: actions, feature list, paths, hyperparameters.

This is the single source of truth for the attack-action vocabulary and the
feature schema. Every other module imports from here so the feature set used
by the generator, the labeling engine, the LSTM, and the baseline can never
drift apart.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BACKEND_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BACKEND_DIR / "data"
MODELS_DIR = BACKEND_DIR / "models_store"
DATA_DIR.mkdir(exist_ok=True, parents=True)
MODELS_DIR.mkdir(exist_ok=True, parents=True)

SYNTHETIC_CSV = DATA_DIR / "synthetic_telemetry.csv"
BENCHMARK_JSON = DATA_DIR / "benchmark_report.json"
CALIBRATION_JSON = DATA_DIR / "calibration_report.json"
LEAD_TIME_JSON = DATA_DIR / "lead_time_report.json"
FALSE_ALARM_JSON = DATA_DIR / "false_alarm_examples.json"
FORECAST_LOG_JSON = DATA_DIR / "recent_forecast_log.json"

LSTM_WEIGHTS = MODELS_DIR / "lstm_world_model.pt"
LSTM_META = MODELS_DIR / "lstm_world_model_meta.json"
BASELINE_WEIGHTS = MODELS_DIR / "baseline_logreg.joblib"
SCALER_WEIGHTS = MODELS_DIR / "feature_scaler.joblib"

# ---------------------------------------------------------------------------
# Window / simulation parameters
# ---------------------------------------------------------------------------
WINDOW_SECONDS = 30          # duration of one time window
SEQ_LEN = 8                  # number of past windows the LSTM conditions on
ROLLOUT_K = 6                # how many windows to roll forward for the forecast curve
AMBIGUOUS_LOOKBACK = 4       # windows before a hard attack-action onset considered for re-labeling
AMBIGUOUS_SCORE_THRESHOLD = 0.35  # precursor score above which a benign-labeled window becomes "ambiguous_pre_attack"

RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Action vocabulary
# ---------------------------------------------------------------------------
# Hard ground-truth actions produced directly by the simulator's attack
# script. Each is a SPECIFIC technique tied to a real destination port/
# service, not a broad tactic bucket -- this is what lets a forecast say
# "SSH brute-force is the most likely next action" rather than just
# "initial access".
HARD_ACTIONS = [
    "benign",
    "port_scan",
    "ssh_bruteforce",
    "rdp_bruteforce",
    "smb_bruteforce",
    "ssh_lateral_movement",
    "rdp_lateral_movement",
    "smb_lateral_movement",
    "c2_beacon",
    "data_exfiltration",
]

# Extended vocabulary used for LSTM next-action classification: adds the
# "ambiguous_pre_attack" class derived by the state-labeling engine itself
# (NOT inherited from the simulator's own labels) for windows immediately
# preceding a hard attack onset that already show measurable precursor signal.
ACTION_CLASSES = [
    "benign",
    "ambiguous_pre_attack",
    "port_scan",
    "ssh_bruteforce",
    "rdp_bruteforce",
    "smb_bruteforce",
    "ssh_lateral_movement",
    "rdp_lateral_movement",
    "smb_lateral_movement",
    "c2_beacon",
    "data_exfiltration",
]
ACTION_TO_IDX = {s: i for i, s in enumerate(ACTION_CLASSES)}
IDX_TO_ACTION = {i: s for s, i in ACTION_TO_IDX.items()}

MALICIOUS_HARD_ACTIONS = {
    "port_scan", "ssh_bruteforce", "rdp_bruteforce", "smb_bruteforce",
    "ssh_lateral_movement", "rdp_lateral_movement", "smb_lateral_movement",
    "c2_beacon", "data_exfiltration",
}

# backward-compatible aliases used by a few older call sites / tests
STAGE_CLASSES = ACTION_CLASSES
STAGE_TO_IDX = ACTION_TO_IDX
IDX_TO_STAGE = IDX_TO_ACTION
HARD_STAGES = HARD_ACTIONS
MALICIOUS_HARD_STAGES = MALICIOUS_HARD_ACTIONS

# Destination port each action targets (used by the generator to set the
# per-window port-indicator features, and by the digital-twin mitigations to
# know which action a mitigation like "rate-limit SSH" actually suppresses).
# port_scan has no single port -- it touches many.
PORT_OF_ACTION = {
    "ssh_bruteforce": 22, "ssh_lateral_movement": 22,
    "rdp_bruteforce": 3389, "rdp_lateral_movement": 3389,
    "smb_bruteforce": 445, "smb_lateral_movement": 445,
    "c2_beacon": 443, "data_exfiltration": 443,
}
WATCHED_PORTS = [22, 445, 3389, 443]

# ---------------------------------------------------------------------------
# MITRE ATT&CK mapping (documented, project-specific -- see README)
# ---------------------------------------------------------------------------
ATTACK_STAGE_MAP = {
    "benign": {"technique_id": None, "technique_name": "N/A", "tactic": "N/A"},
    "ambiguous_pre_attack": {
        "technique_id": "TA0043-pre",
        "technique_name": "Pre-Attack Precursor Activity",
        "tactic": "Reconnaissance (Pre-Confirmation)",
    },
    "port_scan": {
        "technique_id": "T1595",
        "technique_name": "Active Scanning",
        "tactic": "Reconnaissance",
    },
    "ssh_bruteforce": {
        "technique_id": "T1110",
        "technique_name": "Brute Force (SSH)",
        "tactic": "Credential Access / Initial Access",
    },
    "rdp_bruteforce": {
        "technique_id": "T1110",
        "technique_name": "Brute Force (RDP)",
        "tactic": "Credential Access / Initial Access",
    },
    "smb_bruteforce": {
        "technique_id": "T1110",
        "technique_name": "Brute Force (SMB)",
        "tactic": "Credential Access / Initial Access",
    },
    "ssh_lateral_movement": {
        "technique_id": "T1021.004",
        "technique_name": "Remote Services: SSH",
        "tactic": "Lateral Movement",
    },
    "rdp_lateral_movement": {
        "technique_id": "T1021.001",
        "technique_name": "Remote Services: Remote Desktop Protocol",
        "tactic": "Lateral Movement",
    },
    "smb_lateral_movement": {
        "technique_id": "T1021.002",
        "technique_name": "Remote Services: SMB/Windows Admin Shares",
        "tactic": "Lateral Movement",
    },
    "c2_beacon": {
        "technique_id": "T1071.001",
        "technique_name": "Application Layer Protocol: Web Protocols",
        "tactic": "Command and Control",
    },
    "data_exfiltration": {
        "technique_id": "T1041",
        "technique_name": "Exfiltration Over C2 Channel",
        "tactic": "Exfiltration",
    },
}

# ---------------------------------------------------------------------------
# Feature schema (traffic-shaped only -- NEVER includes the ground-truth action)
# ---------------------------------------------------------------------------
FEATURE_COLUMNS = [
    # flow-level
    "flow_count",
    "unique_dst_ports",
    "unique_dst_ips",
    "syn_count",
    "ack_count",
    "fin_count",
    "rst_count",
    "psh_count",
    "avg_flow_duration",
    "std_flow_duration",
    "avg_bytes_per_flow",
    "std_bytes_per_flow",
    "avg_pkts_per_flow",
    "failed_conn_ratio",
    "inbound_bytes",
    "outbound_bytes",
    "bytes_ratio_out_in",
    "new_dst_ip_ratio",
    # packet-level
    "iat_mean",
    "iat_std",
    "ttl_mean",
    "ttl_std",
    "win_size_mean",
    "win_size_std",
    "pkt_len_mean",
    "pkt_len_std",
    "port_scan_score",
    # per-port destination indicators -- legitimate observable NetFlow-style
    # signal (which service traffic in this window was headed to), NOT a
    # one-hot encoding of the ground-truth action. Real traffic genuinely
    # correlates with destination port; that correlation is what any
    # detector (ours or a human analyst) is allowed to use.
    "dst_port_is_22",
    "dst_port_is_445",
    "dst_port_is_3389",
    "dst_port_is_443",
]

N_FEATURES = len(FEATURE_COLUMNS)
