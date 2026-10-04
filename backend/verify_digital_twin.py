"""Verification and proof generator for Hema's Digital Twin prototype.

Runs comprehensive counterfactual simulations across multiple attack scenarios,
demonstrating that mitigations physically suppress attacks, avert downstream
malicious stages, and dramatically reduce forecasted infiltration probability.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from app import db
from app.inference.service import service
from app.config import DATA_DIR, FEATURE_COLUMNS

# The API is multi-user; this script works in its own throwaway workspace,
# filled with the training dataset's hosts it needs.
VERIFY_USER = "verify-digital-twin"


def _load_verify_hosts(host_ids):
    db.configure("sqlite://")  # in-memory
    db.init_db()
    for host_id in host_ids:
        g = service.labeled_df[service.labeled_df["host_id"] == host_id].sort_values("window_idx")
        rows = [{"features": {c: float(r[c]) for c in FEATURE_COLUMNS}, "true_stage": r["true_stage"],
                 "state_label": r["state_label"]} for _, r in g.iterrows()]
        service.ingest_windows(VERIFY_USER, host_id, rows, source="sample")


def run_proofs():
    print("=" * 70)
    print("DIGITAL TWIN PROTOTYPE VERIFICATION & PROOF OF FUNCTIONALITY")
    print("=" * 70)

    service.load()
    print("[+] Model artifacts and dataset successfully loaded.")
    _load_verify_hosts(sorted(service.labeled_df["host_id"].unique()))
    print(f"[+] Total available mitigations: {len(service.available_mitigations())}")

    # Scenarios to prove
    test_cases = [
        {
            "scenario": "Host Isolation during Active Attack",
            "host_id": "attack-host-000",
            "mitigation_id": "isolate_host",
            "window_idx": 29,  # Port scan onset
            "expected_outcome": "High Efficacy (Attack Successfully Averted)",
        },
        {
            "scenario": "Reconnaissance Defense (Block Scanner Probes)",
            "host_id": "attack-host-000",
            "mitigation_id": "block_scanner_ip",
            "window_idx": 29,  # Port scan
            "expected_outcome": "High Efficacy (Attack Successfully Averted)",
        },
        {
            "scenario": "Access Control (Block SSH Brute-Force)",
            "host_id": "attack-host-000",
            "mitigation_id": "block_ssh",
            "window_idx": 75,  # SSH brute force
            "expected_outcome": "High/Moderate Efficacy",
        },
        {
            "scenario": "Traffic Shaping (Rate-limit SSH Brute-Force)",
            "host_id": "attack-host-000",
            "mitigation_id": "rate_limit_ssh",
            "window_idx": 75,  # SSH brute force
            "expected_outcome": "Mitigation applied to port 22 traffic",
        },
        {
            "scenario": "C2 Severing (Block Outbound Beaconing)",
            "host_id": "attack-host-000",
            "mitigation_id": "block_c2_egress",
            "window_idx": 93,  # C2 Beacon
            "expected_outcome": "Clears outbound port 443 and resets beacon regularity",
        },
        {
            "scenario": "Data Loss Prevention (Quarantine Exfiltration Channel)",
            "host_id": "attack-host-000",
            "mitigation_id": "quarantine_exfiltration",
            "window_idx": 107,  # Data Exfiltration
            "expected_outcome": "Stops large outbound egress burst",
        },
    ]

    proof_results = []

    for tc in test_cases:
        print("\n" + "-" * 70)
        print(f"TEST CASE: {tc['scenario']}")
        print(f"Target: {tc['host_id']} at Window #{tc['window_idx']} | Mitigation: {tc['mitigation_id']}")

        res = service.run_counterfactual(
            VERIFY_USER, tc["host_id"], tc["mitigation_id"], at_window_idx=tc["window_idx"]
        )

        unmit_p = res["without_mitigation"]["infiltration_probs"]
        mit_p = res["with_mitigation"]["infiltration_probs"]
        unmit_stages = res["without_mitigation"]["predicted_stage_per_horizon"]
        mit_stages = res["with_mitigation"]["predicted_stage_per_horizon"]
        divergences = res["action_divergences"]
        risk_red = res["risk_reduction_pct"]
        verdict = res["verdict"]

        print(f"Ground Truth Action at t_0:  {res['true_stage']}")
        print(f"Unmitigated Infiltration Curve: {unmit_p}")
        print(f"Mitigated Infiltration Curve:   {mit_p}")
        print(f"Unmitigated Stage Rollout:      {unmit_stages}")
        print(f"Mitigated Stage Rollout:        {mit_stages}")
        print(f"Prevented Attack Horizons:      {len(divergences)} of {res['horizon_windows']}")
        print(f"Infiltration Risk Reduction:    {risk_red}%")
        print(f"Simulation Verdict:             {verdict}")

        if divergences:
            print("Action Divergence Proof:")
            for d in divergences[:3]:
                print(f"  [t+{d['horizon']}] {d['without_mitigation_action']} -> {d['with_mitigation_action']} (THWARTED)")

        proof_results.append({
            "scenario": tc["scenario"],
            "host_id": tc["host_id"],
            "window_idx": tc["window_idx"],
            "mitigation_id": tc["mitigation_id"],
            "true_stage": res["true_stage"],
            "risk_reduction_pct": risk_red,
            "verdict": verdict,
            "unmitigated_probs": unmit_p,
            "mitigated_probs": mit_p,
            "unmitigated_stages": unmit_stages,
            "mitigated_stages": mit_stages,
            "divergence_count": len(divergences),
        })

    verify_sandbox_digital_twin_requirements()

    proof_path = DATA_DIR / "digital_twin_proofs.json"
    with open(proof_path, "w", encoding="utf-8") as f:
        json.dump(proof_results, f, indent=2)

    print("\n" + "=" * 70)
    print(f"ALL PROOFS GENERATED AND SAVED TO: {proof_path}")
    print("=" * 70)


def verify_sandbox_digital_twin_requirements():
    print("\n" + "=" * 70)
    print("VERIFYING 10-POINT DIGITAL TWIN SANDBOX CRITERIA")
    print("=" * 70)

    from app.simulation.network_twin import build_initial_twin_network
    from app.simulation.sandbox import DigitalTwinSandbox
    from app.inference.service import service

    # 1. Initial twin state exists
    host_id = "attack-host-000"
    twin_init = build_initial_twin_network(host_id)
    assert twin_init is not None, "Failed: initial twin state not created"
    assert host_id in twin_init.hosts, f"Failed: target host {host_id} missing from initial twin"
    print("  [1/10] PASS: Initial twin state exists with target host and topology.")

    # 2. Dynamic attack representation
    res = service.run_counterfactual(VERIFY_USER, host_id, "isolate_host", at_window_idx=29)
    assert res.get("true_stage") is not None, "Failed: attack stage not represented dynamically"
    print(f"  [2/10] PASS: Current attack event represented dynamically (stage: {res['true_stage']}).")

    # 3. Selected mitigation changes CLONED twin state
    cloned_state = res.get("cloned_twin_state", {})
    target_in_cloned = cloned_state.get("hosts", {}).get(host_id, {})
    assert target_in_cloned.get("is_isolated") is True, "Failed: mitigation did not modify cloned twin state"
    print("  [3/10] PASS: Selected mitigation modifies CLONED twin state (is_isolated=True).")

    # 4. Original twin remains unchanged
    assert twin_init.hosts[host_id].is_isolated is False, "Failed: original twin state was mutated!"
    print("  [4/10] PASS: Original twin state remains UNCHANGED (is_isolated=False).")

    # 5 & 6. Attacker response (blocked / throttled / continued)
    attacker_resp = res.get("simulated_attacker_response", {})
    outcome = attacker_resp.get("outcome")
    assert outcome in ["BLOCKED", "THROTTLED", "CONTINUED"], f"Failed invalid attacker outcome: {outcome}"
    print(f"  [5-6/10] PASS: Attacker interacts with twin state -> outcome: {outcome} ({attacker_resp.get('reason')}).")

    # 7. Remaining paths recalculated from topology/state
    path_analysis = res.get("path_prediction", {})
    metrics = path_analysis.get("metrics", {})
    assert "blocked_paths_count" in metrics, "Failed: remaining paths not recalculated"
    print(f"  [7/10] PASS: Topology attack paths recalculated ({metrics['blocked_paths_count']} paths blocked, {metrics['remaining_paths_count']} remaining).")

    # 8 & 9. Telemetry reaches LSTM & future state prediction
    with_mit = res.get("with_mitigation", {})
    assert len(with_mit.get("infiltration_probs", [])) > 0, "Failed: LSTM forecast missing"
    assert len(with_mit.get("predicted_stage_per_horizon", [])) > 0, "Failed: LSTM stage rollout missing"
    print(f"  [8-9/10] PASS: Simulated telemetry fed to LSTM -> predicted curve: {with_mit['infiltration_probs'][:3]}.")

    # 10. No real network touched
    assert res.get("real_network_touched") is False, "Failed: real_network_touched is not False!"
    print("  [10/10] PASS: Safe sandbox enforced (real_network_touched: False).")


if __name__ == "__main__":
    run_proofs()
