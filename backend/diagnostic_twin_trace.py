"""Diagnostic script: traces one complete sandbox run with full detail.

Captures exact BEFORE / AFTER topology, attacker response, path changes,
and the feature vector pipeline for the LSTM.
"""
import json, numpy as np
from app.inference.service import service
from app.config import PORT_OF_ACTION, WATCHED_PORTS, FEATURE_COLUMNS, SEQ_LEN
from app.simulation.network_twin import build_initial_twin_network, _PORT_SERVICE_NAME
from app.simulation.sandbox import DigitalTwinSandbox, _MITIGATION_TO_PORT
from app.simulation.attacker import GenericAttacker, AttackState
from app.simulation.path_predictor import PathPredictor
from app.simulation.mitigations import get_mitigation_fn

service.load()
print("=" * 80)
print("DIAGNOSTIC: Full pipeline trace for quarantine_exfiltration @ window 107")
print("=" * 80)

host_id = "attack-host-000"
mitigation_id = "quarantine_exfiltration"
window_idx = 107

# Get the raw seed data
host_df = service.labeled_df[service.labeled_df["host_id"] == host_id].sort_values("window_idx").reset_index(drop=True)
host_df = host_df[host_df["window_idx"] <= window_idx].reset_index(drop=True)
end_pos = len(host_df) - 1
seed_raw = host_df[FEATURE_COLUMNS].values[end_pos - SEQ_LEN + 1: end_pos + 1].astype(np.float32)
true_stage = host_df.iloc[end_pos]["true_stage"]

print(f"\n[INPUT] host_id={host_id}, window_idx={window_idx}, true_stage={true_stage}")
print(f"[INPUT] seed_raw.shape = {seed_raw.shape}")
print(f"[INPUT] Last window raw features (first 8):")
for i, col in enumerate(FEATURE_COLUMNS[:8]):
    print(f"  {col}: {seed_raw[-1][i]:.4f}")

# ---- STEP 1: Build BEFORE topology ----
print("\n" + "=" * 80)
print("STEP 1: BEFORE topology (initial twin)")
print("=" * 80)
initial_twin = build_initial_twin_network(host_id)

print(f"\nHosts ({len(initial_twin.hosts)}):")
for hid, h in initial_twin.hosts.items():
    print(f"  {hid} @ {h.ip_address} | isolated={h.is_isolated}")
    for port, svc in h.services.items():
        print(f"    port {port} ({svc['name']}): status={svc['status']}")

print(f"\nSubnets: {initial_twin.subnets}")
print(f"Firewall rules: {len(initial_twin.firewall.rules)} (none yet)")
print(f"real_network_touched: {initial_twin.real_network_touched}")

# ---- Path prediction BEFORE mitigation ----
print("\nPath prediction BEFORE mitigation:")
path_before = PathPredictor(initial_twin).calculate_attack_paths("external_attacker", host_id, true_stage)
for p in path_before["all_paths"]:
    status = "BLOCKED" if p["is_blocked"] else "OPEN"
    print(f"  [{status}] {p['stage']} -> port {p['target_port']} on {p['target_host']} ({p['tactic']})")
print(f"  Summary: {path_before['summary']}")

# ---- STEP 2: Clone and apply mitigation ----
print("\n" + "=" * 80)
print("STEP 2: Clone + apply mitigation")
print("=" * 80)
cloned = initial_twin.clone()
print(f"Cloned twin created. real_network_touched: {cloned.real_network_touched}")
print(f"Mitigation: {mitigation_id}")
print(f"Dynamic port lookup: _MITIGATION_TO_PORT = {_MITIGATION_TO_PORT}")

# Apply mitigation
sandbox = DigitalTwinSandbox(service.model, service.scaler)
sandbox._apply_mitigation_to_twin_state(cloned, host_id, mitigation_id)

print(f"\nAFTER mitigation:")
for hid, h in cloned.hosts.items():
    if hid == host_id:
        print(f"  {hid} @ {h.ip_address} | isolated={h.is_isolated}")
        for port, svc in h.services.items():
            changed = " <-- CHANGED" if svc['status'] != 'open' else ""
            print(f"    port {port} ({svc['name']}): status={svc['status']}{changed}")

print(f"\nFirewall rules ({len(cloned.firewall.rules)}):")
for r in cloned.firewall.rules:
    print(f"  [{r.action}] {r.rule_id}: {r.source} -> {r.destination}:{r.port} | {r.description}")

# ---- Path prediction AFTER mitigation ----
print("\nPath prediction AFTER mitigation:")
path_after = PathPredictor(cloned).calculate_attack_paths("external_attacker", host_id, true_stage)
for p in path_after["all_paths"]:
    status = "BLOCKED" if p["is_blocked"] else "OPEN"
    print(f"  [{status}] {p['stage']} -> port {p['target_port']} on {p['target_host']} ({p['tactic']})")
print(f"  Summary: {path_after['summary']}")

# ---- STEP 3: Attacker response ----
print("\n" + "=" * 80)
print("STEP 3: Attacker response")
print("=" * 80)
attack_state = AttackState(
    current_stage=true_stage,
    target_host_id=host_id,
    observed_features=seed_raw[-1].copy(),
)
print(f"AttackState: stage={attack_state.current_stage}, port={attack_state.target_port} (from PORT_OF_ACTION)")
response = GenericAttacker.step(cloned, attack_state, mitigation_id=mitigation_id)
print(f"Outcome: {response.outcome}")
print(f"Reason: {response.reason}")
print(f"Affected port: {response.affected_port}")
print(f"Has simulated features: {response.simulated_features is not None}")

if response.simulated_features is not None:
    print(f"\nSimulated features vs original (first 8):")
    for i, col in enumerate(FEATURE_COLUMNS[:8]):
        orig = seed_raw[-1][i]
        sim = response.simulated_features[i]
        delta = sim - orig
        print(f"  {col}: {orig:.4f} -> {sim:.4f} (delta={delta:+.4f})")

# ---- STEP 4: Feature sequence to LSTM ----
print("\n" + "=" * 80)
print("STEP 4: Feature sequence pipeline to LSTM")
print("=" * 80)
lstm_seed_raw = seed_raw.copy()
print(f"lstm_seed_raw[-1] BEFORE blend: first 4 = {lstm_seed_raw[-1][:4]}")
if response.simulated_features is not None:
    lstm_seed_raw[-1] = response.simulated_features
    print(f"lstm_seed_raw[-1] AFTER blend:  first 4 = {lstm_seed_raw[-1][:4]}")
else:
    print("No simulated features -> lstm_seed_raw[-1] unchanged")

print(f"\nFull temporal sequence shape: {lstm_seed_raw.shape}")
print(f"Windows 0-6: ORIGINAL observed features (unmodified)")
print(f"Window 7 (last): {'SIMULATED features from attacker response' if response.simulated_features is not None else 'ORIGINAL (no simulation output)'}")

print(f"\nThen rollout_counterfactual() additionally applies mitigation_fn to this window")
print(f"  -> counterfactual.py L71-72: mitigated_raw[-1] = mitigation_fn(mitigated_raw[-1])")
print(f"  -> Then scaler.transform() -> LSTM -> for k steps: inverse_transform -> clamp -> mitigation_fn -> transform -> append")

# ---- STEP 5: Data origin audit ----
print("\n" + "=" * 80)
print("STEP 5: Data origin audit (live vs synthetic/configured)")
print("=" * 80)
print("""
HOSTS:
  Source: build_initial_twin_network() in network_twin.py L162-193
  Origin: SYNTHETIC/CONFIGURED. For demo hosts:
    - target host_id comes from the labeled_df (CSV dataset)
    - IP address is computed: f"10.0.1.{10 + (idx % 200)}"  (line 180)
    - Neighbor IDs are computed: f"attack-host-{(idx+1)%10:03d}" (line 186-187)
  For live hosts:
    - host_id is "live:<ip>" from the live capture
    - IP is the actual remote IP if parseable, else "192.168.1.50"
    - Neighbors are SYNTHETIC: "live:gateway" 192.168.1.1, "live:internal-server" 192.168.1.100

VLANS:
  Source: TwinNetwork.__init__() line 134
  Origin: HARDCODED. Two VLANs: "Workstation VLAN 10" and "Server VLAN 20"
  Assignment: by name heuristic (line 141: "server" in host_id -> Server VLAN)

SERVICES:
  Source: TwinHost.__init__() lines 50-54
  Origin: CONFIGURED from app.config.WATCHED_PORTS = [22, 445, 3389, 443]
  Names derived from PORT_OF_ACTION (line 25-29), e.g. 22 -> "SSH"
  All start with status="open" -- NOT observed from a real scan

FIREWALL RULES:
  Source: _apply_mitigation_to_twin_state() in sandbox.py L151-226
  Origin: GENERATED per-simulation. Zero rules at start.
  Rules are added only when a mitigation is selected.
  Port numbers derived from _MITIGATION_TO_PORT (lines 30-41),
  which reverses PORT_OF_ACTION.

TOPOLOGY EDGES:
  Source: PathPredictor._get_reachable_hosts() + calculate_attack_paths()
  Origin: INFERRED from the hosts in the TwinNetwork.
  Full mesh assumed: every host can reach every other host
  unless firewall DENY or host isolation blocks it.
  No explicit edge list -- reachability is computed by firewall.evaluate().

FEATURE VECTORS:
  Source: service._raw_seed() -> labeled_df CSV (demo) or live_capture (live)
  Origin: REAL OBSERVED for demo (from synthetic_telemetry.csv training data)
          REAL OBSERVED for live (from actual packet capture)
""")

print("DONE")
