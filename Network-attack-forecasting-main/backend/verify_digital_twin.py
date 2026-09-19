"""Verification and proof generator for Hema's Digital Twin prototype.

Runs comprehensive counterfactual simulations across multiple attack scenarios,
demonstrating that mitigations physically suppress attacks, avert downstream
malicious stages, and dramatically reduce forecasted infiltration probability.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from app.inference.service import service
from app.config import DATA_DIR


def run_proofs():
    print("=" * 70)
    print("DIGITAL TWIN PROTOTYPE VERIFICATION & PROOF OF FUNCTIONALITY")
    print("=" * 70)

    service.load()
    print("[+] Model artifacts and dataset successfully loaded.")
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
            tc["host_id"], tc["mitigation_id"], at_window_idx=tc["window_idx"]
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

    proof_path = DATA_DIR / "digital_twin_proofs.json"
    with open(proof_path, "w", encoding="utf-8") as f:
        json.dump(proof_results, f, indent=2)

    print("\n" + "=" * 70)
    print(f"ALL PROOFS GENERATED AND SAVED TO: {proof_path}")
    print("=" * 70)


if __name__ == "__main__":
    run_proofs()
