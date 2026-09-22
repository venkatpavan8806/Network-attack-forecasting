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
STAGE_MEAN_VECTORS_JSON = DATA_DIR / "stage_mean_vectors.json"

LSTM_WEIGHTS = MODELS_DIR / "lstm_world_model.pt"
LSTM_META = MODELS_DIR / "lstm_world_model_meta.json"
BASELINE_WEIGHTS = MODELS_DIR / "baseline_logreg.joblib"
SCALER_WEIGHTS = MODELS_DIR / "feature_scaler.joblib"

# ---------------------------------------------------------------------------
# Window / simulation parameters
# ---------------------------------------------------------------------------
# TRAFFIC WINDOW semantics -- VERIFIED against the actual implementation in
# app/data_gen/generator.py, app/live/flow_tracker.py, and
# app/live/capture.py (none of which were modified to make this doc fit):
#
#   - One "state" S_t is the FEATURE_COLUMNS vector computed for a single
#     host/remote-peer for ONE window.
#
#   - WINDOW_SECONDS: for LIVE capture, this is a genuine wall-clock
#     duration -- app/live/capture.py's `_window_loop` literally does
#     `time.sleep(WINDOW_SECONDS)` between calls to `FlowTracker.roll_window()`,
#     which computes features from, and then clears, whatever packets were
#     ingested during that real sleep interval. For SYNTHETIC/offline-CSV
#     data there is no real-time process at all: `generate_host_timeline()`
#     samples one row per `window_idx` directly (no per-packet simulation),
#     and WINDOW_SECONDS is used only to convert `window_idx` into the
#     `timestamp` column (`window_idx * WINDOW_SECONDS`) -- a nominal
#     per-window duration convention, not an actually-elapsed wall-clock
#     interval.
#
#   - WINDOW_STEP: windows are currently NON-OVERLAPPING in both paths --
#     synthetic increments `window_idx` by exactly 1 per row; live clears
#     the tracker's buckets on every `roll_window()` call. WINDOW_STEP is a
#     documentation/future-proofing constant only -- no code in this
#     repository currently reads `WINDOW_STEP`; it exists so a future
#     change to overlapping windows only has to change one value, not an
#     enforced parameter today.
#
#   - timestamp semantics differ by path and are NOT the same type:
#     - Synthetic/offline: `timestamp = window_idx * WINDOW_SECONDS`, a
#       plain float, seconds since that host's own timeline start (not
#       wall-clock/epoch time).
#     - Live: `app/live/capture.py`'s `_process_window()` stores
#       `timestamp = datetime.now(timezone.utc).isoformat()` in its
#       in-memory `recent_predictions`/log entries -- a real wall-clock
#       ISO-8601 UTC string, not a `window_idx * WINDOW_SECONDS` float.
#       (Individual packets inside `FlowTracker` do carry raw epoch-second
#       floats via `time.time()`, but that's an internal detail -- it is
#       not what ends up in the `timestamp` field consumers see.)
#     These two `timestamp` representations should not be assumed
#     interchangeable by anything that reads both paths.
#
#   - host_id semantics also differ by path:
#     - Synthetic/offline: the profiled host's own outbound behavior
#       (e.g. "attack-host-003").
#     - Live, as returned by `FlowTracker.roll_window()`: keyed by the bare
#       REMOTE peer IP address (a single-NIC capture sees inbound
#       connection attempts FROM that peer -- see flow_tracker.py's module
#       docstring for the full vantage-point mapping). By the time
#       `app/live/capture.py` records a window, it stores this as
#       `host_id = f"live:{remote_ip}"` (prefixed), not the bare IP --
#       so the exact string differs between `FlowTracker`'s own return
#       value and what ends up in the live inference log/API.
#
#   - CAUSALITY -- precisely scoped to where it's actually enforced, not
#     claimed project-wide:
#     1. LIVE feature computation is genuinely causal: `FlowTracker`
#        computes a window's features only from packets it has already
#        ingested (via `ingest_tcp`) by the time `roll_window()` is called;
#        it cannot see packets that haven't arrived yet.
#     2. The offline precursor-score ROW-LOCAL FORMULA is causal in the
#        sense documented in detail in app/labeling/state_labeler.py's
#        module docstring ("CAUSALITY, PRECISELY STATED") -- see that file
#        for the full three-layer breakdown (row-local formula vs.
#        dataset-level normalization reference vs. offline candidate
#        selection).
#     3. This does NOT extend to a claim that the SYNTHETIC GENERATOR's
#        per-window feature sampling is blind to future-stage information.
#        `generate_host_timeline()` decides a host's entire attack script
#        up front -- including exactly which service it will later
#        brute-force (`target_port`) -- before generating any window rows,
#        and by design (see generator.py's own module docstring) the later
#        ~40% of the reconnaissance phase's simulated port-scan focus is
#        deliberately conditioned on that future brute-force target. This
#        is an intentional, documented, offline scenario-scripting choice
#        (the "port scan reveals SSH open -> SSH brute-force follows"
#        foreshadowing signal the world model is meant to learn), not a
#        live/real-time process reading ahead into arriving raw traffic --
#        but it does mean "no code anywhere reads ahead" is NOT an accurate
#        description of the synthetic generator specifically, and this
#        comment does not claim otherwise.
WINDOW_SECONDS = 30          # duration of one time window
WINDOW_STEP = WINDOW_SECONDS  # stride between windows; == WINDOW_SECONDS (non-overlapping); not read by any code today
SEQ_LEN = 8                  # number of past windows the LSTM conditions on
ROLLOUT_K = 6                # how many windows to roll forward for the forecast curve

# Branching K-step forecast (attack-path tree). Forks the linear rollout()
# above into the BRANCH_FACTOR most probable next-actions at every step,
# instead of only ever following the single argmax continuation, so the UI
# can show "what are the plausible next moves" rather than one committed
# guess. See models/lstm_world_model.py:branching_rollout for the full
# design writeup, incl. how sibling branches are made to diverge.
BRANCH_FACTOR = 3            # candidate next-actions forked at each step
BRANCH_DEPTH = 4             # tree depth in windows (kept < ROLLOUT_K -- node
                              # count grows ~BRANCH_FACTOR**depth)
BRANCH_MIN_PATH_PROB = 0.03  # a branch is pruned once its cumulative path probability drops below this
BRANCH_STATE_BLEND = 0.5     # 0 = every branch continues from the same model-regressed
                              # state (branches would only ever differ in their label, not
                              # in what happens after); 1 = every branch continues purely
                              # from its class's mean training-data feature vector

# ---------------------------------------------------------------------------
# Pre-detection / precursor labeling parameters
# ---------------------------------------------------------------------------
# Canonical names (use these in new code). `AMBIGUOUS_LOOKBACK` /
# `AMBIGUOUS_SCORE_THRESHOLD` are kept below as backward-compatible aliases
# since existing call sites and tests already reference them.
PRECURSOR_LOOKBACK = 4            # windows before a hard attack-action onset considered as pre-attack candidates
PRECURSOR_THRESHOLD = 0.35        # precursor score >= this makes a candidate window "ambiguous_pre_attack"

# Weight of each individual, feature-derived precursor SIGNAL in the combined
# precursor SCORE (see app/labeling/state_labeler.py). Weights are a single,
# authoritative, configurable source -- not scattered/hardcoded inline in the
# labeling logic. Must sum to 1.0. Current values reproduce the project's
# original (already-tuned) formula; kept unchanged here, not re-tuned.
PRECURSOR_WEIGHTS = {
    "port_scan": 0.35,          # elevated port_scan_score -> reconnaissance-like behavior
    "failed_connections": 0.25,  # elevated failed_conn_ratio -> probing/brute-force-like behavior
    "destination_novelty": 0.20,  # elevated new_dst_ip_ratio -> unusual spread of destinations
    "timing_irregularity": 0.20,  # elevated iat_std/iat_mean -> irregular, non-human/non-routine timing
}
assert abs(sum(PRECURSOR_WEIGHTS.values()) - 1.0) < 1e-9, "PRECURSOR_WEIGHTS must sum to 1.0"

# Backward-compatible aliases (existing code/tests use these names).
AMBIGUOUS_LOOKBACK = PRECURSOR_LOOKBACK       # windows before a hard attack-action onset considered for re-labeling
AMBIGUOUS_SCORE_THRESHOLD = PRECURSOR_THRESHOLD  # precursor score above which a benign-labeled window becomes "ambiguous_pre_attack"

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
# Tools & likely-consequence reference table -- curated security-analyst
# knowledge, NOT a model output. The LSTM predicts the ACTION CATEGORY from
# traffic features alone (flag counts, timing, ports); it never sees tool
# fingerprints or payload content, so it has no basis to infer which
# specific tool an attacker is running. This table attaches typical tooling
# and likely system impact to whichever action the model predicts, the same
# way ATTACK_STAGE_MAP attaches a MITRE technique ID -- an enrichment layer
# on top of the prediction, not part of the prediction itself.
# ---------------------------------------------------------------------------
TOOLS_AND_IMPACT_MAP = {
    "benign": {
        "likely_tools": None,
        "likely_system_state": "Normal operation -- no attacker activity detected.",
    },
    "ambiguous_pre_attack": {
        "likely_tools": "Unconfirmed -- this is an early precursor signal, not a specific technique yet.",
        "likely_system_state": "No confirmed compromise. Elevated precursor signal only -- treat as a watch condition, not an incident.",
    },
    "port_scan": {
        "likely_tools": "Nmap, Masscan, ZMap, or a custom SYN-scan script",
        "likely_system_state": "Attacker has enumerated open ports/services. No host compromised yet, but the attack surface is now known to the attacker.",
    },
    "ssh_bruteforce": {
        "likely_tools": "Hydra, Medusa, Ncrack, or a custom credential-stuffing script",
        "likely_system_state": "If successful: attacker obtains valid SSH credentials and gains interactive shell access to the host.",
    },
    "rdp_bruteforce": {
        "likely_tools": "Hydra, Crowbar, NLBrute, or a custom RDP credential-spraying tool",
        "likely_system_state": "If successful: attacker obtains valid RDP credentials and gains full remote-desktop control of the host.",
    },
    "smb_bruteforce": {
        "likely_tools": "Hydra, CrackMapExec, Responder (relay), or a custom SMB credential-spraying tool",
        "likely_system_state": "If successful: attacker obtains valid SMB/Windows credentials and gains access to file shares and admin endpoints.",
    },
    "ssh_lateral_movement": {
        "likely_tools": "Stolen SSH keys/credentials with scp/rsync, or SSH tunneling/ProxyJump pivoting",
        "likely_system_state": "Attacker uses compromised SSH access to reach additional internal hosts -- network segmentation has been breached.",
    },
    "rdp_lateral_movement": {
        "likely_tools": "Stolen RDP credentials, PsExec, or RDP session hijacking tools",
        "likely_system_state": "Attacker pivots to additional internal Windows hosts via RDP -- network segmentation has been breached.",
    },
    "smb_lateral_movement": {
        "likely_tools": "PsExec, WMIC, Impacket (psexec.py / wmiexec.py), or PowerShell remoting",
        "likely_system_state": "Attacker moves laterally using stolen SMB/Windows credentials -- often paired with credential dumping (e.g. Mimikatz) on newly reached hosts.",
    },
    "c2_beacon": {
        "likely_tools": "Cobalt Strike, Metasploit, Sliver, or a custom C2 framework beaconing over HTTPS/DNS",
        "likely_system_state": "A persistent command-and-control channel is established -- the attacker maintains remote control and can issue further commands at will.",
    },
    "data_exfiltration": {
        "likely_tools": "rclone, curl/scp, DNS tunneling tools, or abused cloud-sync utilities",
        "likely_system_state": "Sensitive data is leaving the network. If not stopped, this represents a completed data breach.",
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

# ---------------------------------------------------------------------------
# Feature documentation
# ---------------------------------------------------------------------------
# One entry per FEATURE_COLUMNS name. This is the authoritative, machine-
# readable documentation for the feature schema: meaning, unit/range,
# how it's computed, why it's useful for forecasting, and whether it is a
# directly-observed quantity or an engineered/derived one. Kept in sync with
# FEATURE_COLUMNS by `validate_feature_vector` (see app/features/extraction.py),
# which asserts the two have identical key sets.
FEATURE_DOCS = {
    "flow_count": {
        "meaning": "Number of distinct flows observed in the window.",
        "unit_or_range": "count, >= 0",
        "calculation": "One flow per distinct (remote peer, local/dst port) pair with traffic in the window.",
        "purpose": "Overall traffic volume/activity; scans and brute-force bursts raise it sharply.",
        "kind": "observed",
    },
    "unique_dst_ports": {
        "meaning": "Number of distinct destination ports touched in the window.",
        "unit_or_range": "count, >= 0",
        "calculation": "Count of distinct destination ports across the window's flows.",
        "purpose": "Core port-scan signal: many ports touched with little else in common is scan-shaped.",
        "kind": "observed",
    },
    "unique_dst_ips": {
        "meaning": "Number of distinct destination IPs contacted in the window.",
        "unit_or_range": "count, >= 0 (fixed at 1.0 in single-host live capture -- see flow_tracker.py)",
        "calculation": "Count of distinct destination IPs across the window's flows.",
        "purpose": "Spread of activity across hosts; relevant to lateral movement / scanning breadth.",
        "kind": "observed",
    },
    "syn_count": {
        "meaning": "Number of TCP SYN packets (connection attempts) in the window.",
        "unit_or_range": "count, >= 0",
        "calculation": "Count of packets with the SYN flag set and ACK not set.",
        "purpose": "Raw connection-attempt volume; central to scan/brute-force detection.",
        "kind": "observed",
    },
    "ack_count": {
        "meaning": "Number of TCP ACK packets in the window.",
        "unit_or_range": "count, >= 0",
        "calculation": "Count of packets with the ACK flag set.",
        "purpose": "Compared against syn_count to estimate how many attempts actually progressed.",
        "kind": "observed",
    },
    "fin_count": {
        "meaning": "Number of TCP FIN packets (graceful connection close) in the window.",
        "unit_or_range": "count, >= 0",
        "calculation": "Count of packets with the FIN flag set.",
        "purpose": "Normal connections close gracefully; its absence/ratio helps characterize traffic shape.",
        "kind": "observed",
    },
    "rst_count": {
        "meaning": "Number of TCP RST packets (abrupt connection reset) in the window.",
        "unit_or_range": "count, >= 0",
        "calculation": "Count of packets with the RST flag set.",
        "purpose": "Elevated resets indicate refused/failed connection attempts (scanning, brute-force).",
        "kind": "observed",
    },
    "psh_count": {
        "meaning": "Number of TCP PSH packets (application data pushed) in the window.",
        "unit_or_range": "count, >= 0",
        "calculation": "Count of packets with the PSH flag set.",
        "purpose": "Proxy for actual data-carrying traffic, as opposed to bare handshake packets.",
        "kind": "observed",
    },
    "avg_flow_duration": {
        "meaning": "Mean duration of flows in the window.",
        "unit_or_range": "seconds, >= 0",
        "calculation": "Mean of (last_packet_ts - first_packet_ts) per flow.",
        "purpose": "Short-lived flows suggest scanning/probing; long flows suggest sustained sessions/exfiltration.",
        "kind": "engineered",
    },
    "std_flow_duration": {
        "meaning": "Standard deviation of flow duration in the window.",
        "unit_or_range": "seconds, >= 0",
        "calculation": "Sample std-dev of per-flow durations.",
        "purpose": "Captures homogeneity of flow shape -- automated behavior tends to be either very uniform or erratic.",
        "kind": "engineered",
    },
    "avg_bytes_per_flow": {
        "meaning": "Mean total bytes transferred per flow.",
        "unit_or_range": "bytes, >= 0",
        "calculation": "Mean of per-flow (inbound + outbound) byte totals.",
        "purpose": "Distinguishes bulk transfer (exfiltration) from thin probing traffic.",
        "kind": "engineered",
    },
    "std_bytes_per_flow": {
        "meaning": "Standard deviation of total bytes per flow.",
        "unit_or_range": "bytes, >= 0",
        "calculation": "Sample std-dev of per-flow byte totals.",
        "purpose": "Flags mixed traffic shapes within one window.",
        "kind": "engineered",
    },
    "avg_pkts_per_flow": {
        "meaning": "Mean packet count per flow.",
        "unit_or_range": "count, >= 0",
        "calculation": "Mean of per-flow packet counts.",
        "purpose": "Complements avg_bytes_per_flow; low packets/high bytes vs. high packets/low bytes differ meaningfully.",
        "kind": "engineered",
    },
    "failed_conn_ratio": {
        "meaning": "Fraction of connection attempts that did not complete a handshake.",
        "unit_or_range": "ratio, [0, 1]",
        "calculation": "1 - (completed handshakes / relevant connection attempts), clipped to [0, 1].",
        "purpose": "Core scanning/brute-force signal: a real attacker's probes mostly fail or are refused.",
        "kind": "engineered",
    },
    "inbound_bytes": {
        "meaning": "Total bytes received in the window.",
        "unit_or_range": "bytes, >= 0",
        "calculation": "Sum of packet lengths for inbound-direction packets.",
        "purpose": "Raw volume signal; used with outbound_bytes for direction-of-flow analysis.",
        "kind": "observed",
    },
    "outbound_bytes": {
        "meaning": "Total bytes sent in the window.",
        "unit_or_range": "bytes, >= 0",
        "calculation": "Sum of packet lengths for outbound-direction packets.",
        "purpose": "Large outbound volume relative to inbound is a classic exfiltration indicator.",
        "kind": "observed",
    },
    "bytes_ratio_out_in": {
        "meaning": "Ratio of outbound to inbound byte volume.",
        "unit_or_range": "ratio, >= 0 (1.0 roughly balanced)",
        "calculation": "outbound_bytes / max(inbound_bytes, 1).",
        "purpose": "Directly targets data exfiltration (large, sustained outbound skew).",
        "kind": "engineered",
    },
    "new_dst_ip_ratio": {
        "meaning": "How novel the destination(s) contacted this window are.",
        "unit_or_range": "ratio, [0, 1]",
        "calculation": (
            "Offline/synthetic: fraction of this window's destination IPs not seen before for this host. "
            "Live (flow_tracker.py): repurposed as 1 / times_this_remote_peer_seen, i.e. novelty of the "
            "REMOTE PEER to this capture session -- documented deviation, see flow_tracker.py docstring."
        ),
        "purpose": "Reconnaissance and lateral movement both tend to touch new destinations.",
        "kind": "engineered",
    },
    "iat_mean": {
        "meaning": "Mean inter-arrival time between packets in the window.",
        "unit_or_range": "seconds, >= 0",
        "calculation": "Mean of consecutive packet timestamp differences.",
        "purpose": "Automated tooling (scanners, beacons) has a distinctive pacing vs. human-driven traffic.",
        "kind": "engineered",
    },
    "iat_std": {
        "meaning": "Standard deviation of inter-arrival time between packets.",
        "unit_or_range": "seconds, >= 0",
        "calculation": "Sample std-dev of consecutive packet timestamp differences.",
        "purpose": (
            "Timing irregularity/regularity signal: unusually high std suggests jittered/evasive probing; "
            "unusually low std (relative to mean) suggests a suspiciously regular automated beacon."
        ),
        "kind": "engineered",
    },
    "ttl_mean": {
        "meaning": "Mean IP TTL (time-to-live) value observed in the window.",
        "unit_or_range": "hops, typically 0-255",
        "calculation": "Mean of per-packet TTL field.",
        "purpose": "TTL correlates with OS/network-path; shifts can indicate a different actual source than expected.",
        "kind": "observed",
    },
    "ttl_std": {
        "meaning": "Standard deviation of IP TTL in the window.",
        "unit_or_range": "hops, >= 0",
        "calculation": "Sample std-dev of per-packet TTL field.",
        "purpose": "High variance suggests traffic mixed from multiple sources/paths within one window.",
        "kind": "observed",
    },
    "win_size_mean": {
        "meaning": "Mean TCP receive-window size advertised in the window.",
        "unit_or_range": "bytes, >= 0",
        "calculation": "Mean of per-packet TCP window-size field.",
        "purpose": "Stack/tooling fingerprint; some scanning/brute-force tools use non-default window sizes.",
        "kind": "observed",
    },
    "win_size_std": {
        "meaning": "Standard deviation of TCP window size in the window.",
        "unit_or_range": "bytes, >= 0",
        "calculation": "Sample std-dev of per-packet TCP window-size field.",
        "purpose": "Homogeneity of stack/tooling fingerprint across the window's packets.",
        "kind": "observed",
    },
    "pkt_len_mean": {
        "meaning": "Mean packet length in the window.",
        "unit_or_range": "bytes, >= 0",
        "calculation": "Mean of per-packet total length.",
        "purpose": "Bulk transfer uses large packets; bare probes/handshakes use small ones.",
        "kind": "observed",
    },
    "pkt_len_std": {
        "meaning": "Standard deviation of packet length in the window.",
        "unit_or_range": "bytes, >= 0",
        "calculation": "Sample std-dev of per-packet total length.",
        "purpose": "Mixed small/large packets within a window can indicate mixed control+data traffic.",
        "kind": "observed",
    },
    "port_scan_score": {
        "meaning": "Laplace-smoothed ratio of distinct ports touched to flow volume.",
        "unit_or_range": "ratio, [0, 1]",
        "calculation": "min(1.0, unique_dst_ports / (flow_count + 2)) (offline generator uses an analogous shaped distribution).",
        "purpose": "Single scalar recon summary: high when many ports are touched with few attempts each.",
        "kind": "engineered",
    },
    "dst_port_is_22": {
        "meaning": "Whether SSH-service traffic (port 22) was observed in the window.",
        "unit_or_range": "indicator, {0.0, 1.0}",
        "calculation": "1.0 if any flow in the window targets destination port 22, else 0.0.",
        "purpose": "Legitimate, directly-observable destination-port telemetry -- NOT a one-hot of the attack label.",
        "kind": "observed",
    },
    "dst_port_is_445": {
        "meaning": "Whether SMB-service traffic (port 445) was observed in the window.",
        "unit_or_range": "indicator, {0.0, 1.0}",
        "calculation": "1.0 if any flow in the window targets destination port 445, else 0.0.",
        "purpose": "Legitimate, directly-observable destination-port telemetry -- NOT a one-hot of the attack label.",
        "kind": "observed",
    },
    "dst_port_is_3389": {
        "meaning": "Whether RDP-service traffic (port 3389) was observed in the window.",
        "unit_or_range": "indicator, {0.0, 1.0}",
        "calculation": "1.0 if any flow in the window targets destination port 3389, else 0.0.",
        "purpose": "Legitimate, directly-observable destination-port telemetry -- NOT a one-hot of the attack label.",
        "kind": "observed",
    },
    "dst_port_is_443": {
        "meaning": "Whether HTTPS-service traffic (port 443) was observed in the window.",
        "unit_or_range": "indicator, {0.0, 1.0}",
        "calculation": "1.0 if any flow in the window targets destination port 443, else 0.0.",
        "purpose": "Legitimate, directly-observable destination-port telemetry -- NOT a one-hot of the attack label.",
        "kind": "observed",
    },
}
assert set(FEATURE_DOCS.keys()) == set(FEATURE_COLUMNS), "FEATURE_DOCS must document exactly the FEATURE_COLUMNS set"
