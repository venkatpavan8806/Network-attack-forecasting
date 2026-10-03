"""Defense advisor: turns the digital-twin counterfactual rollouts into a
ranked, explained "what should the defender do" recommendation.

Nothing here is a lookup keyed on the attack name alone. The ranking is
computed from real model output every time it is called:

    for every mitigation:  rollout(world model, mitigation applied)
                           vs.  rollout(world model, no mitigation)
    -> how much does each one lower the predicted infiltration curve?

The only static content is (a) the ATT&CK-mapped manual playbook text below
and (b) a coarse "operational disruption" tier per mitigation (a judgment
call, documented in DISRUPTION). Both are labelled as such in the API output.

This module is pure Python/NumPy on purpose (no torch import) so it can be
unit-tested without a trained model. The service layer
(app/inference/service.py:defense_advice) is what feeds it real rollouts.

DECISION SUPPORT ONLY -- nothing here applies a mitigation to a network.
That is a deliberate project boundary (see README, "closed-loop automatic
defensive action" is out of scope).
"""
from __future__ import annotations

from statistics import mean

from app.config import ACTION_CLASSES, ATTACK_STAGE_MAP

# Thresholds reuse the cut-offs the rest of the dashboard already uses:
#   0.5 -> "high-risk trajectory" KPI (api/main.py:/kpis)
#   0.2 -> "watch" cut-off in the Explainability Digest (ExplainabilityDigest.tsx)
ACT_THRESHOLD = 0.5
WATCH_THRESHOLD = 0.2

# A mitigation only counts as "effective" if it lowers the MEAN predicted
# infiltration probability over the horizon by at least this much (absolute).
MIN_MEAN_REDUCTION = 0.02

# Recommend the LEAST disruptive mitigation that still achieves at least this
# fraction of the best reduction any mitigation achieved.
GOOD_ENOUGH_FRACTION = 0.8

# Coarse operational-disruption tier: 1 = barely noticeable to legitimate
# users, 3 = takes the host off the network. A defensible engineering
# judgment, NOT a measurement -- it is surfaced as such in the response.
DISRUPTION = {
    "rate_limit_ssh": 1, "rate_limit_rdp": 1, "rate_limit_smb": 1,
    "block_ssh": 2, "block_rdp": 2, "block_smb": 2,
    "isolate_host": 3,
}
DISRUPTION_LABEL = {1: "low", 2: "medium", 3: "high"}

# ATT&CK mitigation IDs (attack.mitre.org). `id: None` = good practice that is
# not an official ATT&CK mitigation entry for that technique.
# Verified against attack.mitre.org: T1110 (M1036, M1032, M1027, M1018) and
# T1021.002 (M1037, M1035, M1027, M1026, M1018). The remaining rows are from
# the ATT&CK knowledge base as recalled -- re-check them on the technique's
# page before presenting.
PLAYBOOK: dict[str, dict] = {
    "ambiguous_pre_attack": {
        "summary": "Early precursor signal only. Nothing is confirmed, so prefer observation over blocking.",
        "mitigations": [
            {"id": "M1056", "name": "Pre-compromise", "action": "Not preventable with controls; focus on visibility."},
        ],
        "analyst_steps": [
            "Raise flow/packet logging detail for this source for the next few windows.",
            "Check whether the source is a known scanner, monitoring tool or new legitimate device.",
            "Do NOT block yet -- this is the stage where false alarms live (see the False-Alarm panel).",
        ],
    },
    "port_scan": {
        "summary": "Active scanning (reconnaissance). Attacker is mapping which services exist.",
        "mitigations": [
            {"id": "M1056", "name": "Pre-compromise", "action": "Scanning itself is hard to prevent; reduce what a scan can find."},
            {"id": None, "name": "Filter inbound traffic", "action": "Deny-by-default firewall policy so closed ports do not answer."},
        ],
        "analyst_steps": [
            "Confirm the scan source and whether it is internal or external.",
            "Verify only intended services (22/445/3389/443) are reachable from that source.",
            "Prepare the port-specific mitigations below -- brute force typically follows a scan.",
        ],
    },
    "ssh_bruteforce": {
        "summary": "Credential guessing against SSH (T1110).",
        "mitigations": [
            {"id": "M1036", "name": "Account Use Policies", "action": "Account lockout after repeated failures."},
            {"id": "M1032", "name": "Multi-factor Authentication", "action": "Require MFA / key-based auth for SSH."},
            {"id": "M1027", "name": "Password Policies", "action": "Enforce strong, unique passwords."},
            {"id": "M1018", "name": "User Account Management", "action": "Reset accounts targeted by the attempts."},
        ],
        "analyst_steps": [
            "Review SSH auth logs for the target accounts; look for any success after many failures.",
            "Disable password auth for SSH where key-based auth is available.",
            "If any login succeeded, treat the host as compromised and escalate.",
        ],
    },
    "rdp_bruteforce": {
        "summary": "Credential guessing against RDP (T1110).",
        "mitigations": [
            {"id": "M1036", "name": "Account Use Policies", "action": "Account lockout / conditional access for RDP logons."},
            {"id": "M1032", "name": "Multi-factor Authentication", "action": "Require MFA for RDP."},
            {"id": "M1027", "name": "Password Policies", "action": "Enforce strong, unique passwords."},
            {"id": "M1018", "name": "User Account Management", "action": "Reset accounts targeted by the attempts."},
        ],
        "analyst_steps": [
            "Review Windows logon events (failed then successful) for the target accounts.",
            "Put RDP behind a VPN / gateway instead of exposing it directly.",
            "If any logon succeeded, treat the host as compromised and escalate.",
        ],
    },
    "smb_bruteforce": {
        "summary": "Credential guessing against SMB (T1110).",
        "mitigations": [
            {"id": "M1036", "name": "Account Use Policies", "action": "Account lockout after repeated failures."},
            {"id": "M1032", "name": "Multi-factor Authentication", "action": "Require MFA where the service supports it."},
            {"id": "M1027", "name": "Password Policies", "action": "Enforce strong, unique passwords; no shared local admin passwords."},
            {"id": "M1018", "name": "User Account Management", "action": "Reset accounts targeted by the attempts."},
        ],
        "analyst_steps": [
            "Check SMB/Windows security logs for failed-then-successful authentication.",
            "Restrict SMB to the hosts that legitimately need it.",
            "If any logon succeeded, treat the host as compromised and escalate.",
        ],
    },
    "ssh_lateral_movement": {
        "summary": "Attacker with valid access is moving between hosts over SSH (T1021.004).",
        "mitigations": [
            {"id": "M1035", "name": "Limit Access to Resource Over Network", "action": "Restrict which hosts may SSH to which."},
            {"id": "M1032", "name": "Multi-factor Authentication", "action": "Require MFA for SSH logins."},
            {"id": "M1026", "name": "Privileged Account Management", "action": "Limit and audit privileged SSH accounts."},
            {"id": "M1042", "name": "Disable or Remove Feature or Program", "action": "Disable SSH where it is not needed."},
        ],
        "analyst_steps": [
            "Identify which credentials were used and rotate them.",
            "Map which hosts this source reached; isolate the ones with unexpected SSH sessions.",
            "Escalate to incident response -- lateral movement means access was already gained.",
        ],
    },
    "rdp_lateral_movement": {
        "summary": "Attacker with valid access is moving between hosts over RDP (T1021.001).",
        "mitigations": [
            {"id": "M1035", "name": "Limit Access to Resource Over Network", "action": "Restrict RDP to jump hosts / management VLAN."},
            {"id": "M1032", "name": "Multi-factor Authentication", "action": "Require MFA for RDP."},
            {"id": "M1030", "name": "Network Segmentation", "action": "Segment workstations from each other and from servers."},
            {"id": "M1026", "name": "Privileged Account Management", "action": "Limit and audit privileged RDP accounts."},
        ],
        "analyst_steps": [
            "Identify which credentials were used and rotate them.",
            "Map which hosts this source reached over RDP; isolate unexpected ones.",
            "Escalate to incident response -- lateral movement means access was already gained.",
        ],
    },
    "smb_lateral_movement": {
        "summary": "Attacker with valid access is moving between hosts over SMB / admin shares (T1021.002).",
        "mitigations": [
            {"id": "M1037", "name": "Filter Network Traffic", "action": "Host firewall rules to block workstation-to-workstation SMB."},
            {"id": "M1035", "name": "Limit Access to Resource Over Network", "action": "Disable Windows administrative shares where not needed."},
            {"id": "M1027", "name": "Password Policies", "action": "No reused local administrator passwords."},
            {"id": "M1026", "name": "Privileged Account Management", "action": "Tiered admin model; limit who has local admin."},
        ],
        "analyst_steps": [
            "Identify which credentials were used and rotate them.",
            "Check which shares/hosts were accessed; isolate unexpected ones.",
            "Escalate to incident response -- lateral movement means access was already gained.",
        ],
    },
    "c2_beacon": {
        "summary": "Regular outbound check-ins to an attacker server (T1071.001). Host is likely compromised.",
        "mitigations": [
            {"id": "M1031", "name": "Network Intrusion Prevention", "action": "Block/alert on the beacon destination and pattern."},
            {"id": None, "name": "Egress filtering", "action": "Restrict outbound 443 to approved destinations / proxy."},
        ],
        "analyst_steps": [
            "Identify the beacon destination and block it at the egress firewall/proxy.",
            "Isolate the host from the network for forensic triage.",
            "Preserve memory/disk evidence before reimaging.",
        ],
    },
    "data_exfiltration": {
        "summary": "Large outbound transfer over the C2 channel (T1041). Data may already be leaving.",
        "mitigations": [
            {"id": "M1031", "name": "Network Intrusion Prevention", "action": "Block the exfiltration destination and channel."},
            {"id": None, "name": "Egress filtering / DLP", "action": "Cap outbound volume and inspect large uploads."},
        ],
        "analyst_steps": [
            "Cut the outbound channel immediately (block destination; isolate the host).",
            "Quantify what left: destination, bytes out, time range.",
            "Start incident response and breach-notification assessment.",
        ],
    },
}

GENERIC_PLAYBOOK = {
    "summary": "No specific attack stage expected within the forecast horizon.",
    "mitigations": [],
    "analyst_steps": ["Keep monitoring; no defensive action is indicated by the forecast."],
}


def _stage_severity(stage: str) -> int:
    """Position in the kill-chain-ordered ACTION_CLASSES list (benign=0)."""
    return ACTION_CLASSES.index(stage) if stage in ACTION_CLASSES else 0


def furthest_expected_stage(predicted_stage_per_horizon: list[str]) -> str | None:
    """The most advanced attack stage the model expects anywhere in the
    horizon (None if it only ever expects 'benign')."""
    non_benign = [s for s in predicted_stage_per_horizon if s != "benign"]
    if not non_benign:
        return None
    return max(non_benign, key=_stage_severity)


def risk_level(unmitigated_probs: list[float]) -> str:
    peak = max(unmitigated_probs) if unmitigated_probs else 0.0
    if peak >= ACT_THRESHOLD:
        return "act_now"
    if peak >= WATCH_THRESHOLD:
        return "watch"
    return "monitor"


def score_mitigation(mitigation_id: str, meta: dict, unmitigated: dict, mitigated: dict) -> dict:
    """Compare one mitigated rollout against the unmitigated one."""
    p0, p1 = unmitigated["infiltration_probs"], mitigated["infiltration_probs"]
    mean_before, mean_after = mean(p0), mean(p1)
    mean_reduction = mean_before - mean_after
    tier = DISRUPTION.get(mitigation_id, 2)
    return {
        "id": mitigation_id,
        "label": meta.get("label", mitigation_id),
        "description": meta.get("description", ""),
        "mean_probability_without": round(mean_before, 4),
        "mean_probability_with": round(mean_after, 4),
        "mean_reduction": round(mean_reduction, 4),
        "relative_reduction": round(mean_reduction / mean_before, 4) if mean_before > 1e-9 else 0.0,
        "peak_reduction": round(max(p0) - max(p1), 4),
        "stages_changed": sum(1 for a, b in zip(unmitigated["predicted_stage"], mitigated["predicted_stage"]) if a != b),
        "disruption_tier": tier,
        "disruption": DISRUPTION_LABEL[tier],
        "effective": mean_reduction >= MIN_MEAN_REDUCTION,
    }


def pick_recommendation(scored: list[dict]) -> dict | None:
    """Least-disruptive mitigation that still achieves >= GOOD_ENOUGH_FRACTION
    of the best reduction. Ties on disruption go to the bigger reduction."""
    effective = [s for s in scored if s["effective"]]
    if not effective:
        return None
    best = max(s["mean_reduction"] for s in effective)
    good_enough = [s for s in effective if s["mean_reduction"] >= GOOD_ENOUGH_FRACTION * best]
    return min(good_enough, key=lambda s: (s["disruption_tier"], -s["mean_reduction"]))


def build_advice(host_id: str, window_idx: int, unmitigated: dict,
                 mitigated_by_id: dict[str, dict], mitigations_meta: dict[str, dict]) -> dict:
    """unmitigated / mitigated_by_id[...] are rollout dicts with at least
    `infiltration_probs` and `predicted_stage` (what
    app.models.lstm_world_model.rollout and
    app.simulation.counterfactual.rollout_counterfactual return)."""
    level = risk_level(unmitigated["infiltration_probs"])
    expected_stage = furthest_expected_stage(unmitigated["predicted_stage"])

    scored = [
        score_mitigation(mid, mitigations_meta.get(mid, {}), unmitigated, roll)
        for mid, roll in mitigated_by_id.items()
    ]
    scored.sort(key=lambda s: s["mean_reduction"], reverse=True)

    recommended, reason = None, None
    if level == "monitor":
        reason = (f"Peak predicted infiltration probability is below {WATCH_THRESHOLD:.0%} across the "
                  f"forecast horizon, so no defensive action is indicated.")
    else:
        recommended = pick_recommendation(scored)
        if recommended is None:
            reason = ("None of the modelled mitigations lowers the forecast by a meaningful amount "
                      f"(>= {MIN_MEAN_REDUCTION:.0%} mean reduction). Follow the manual playbook steps.")
        else:
            best = max(s["mean_reduction"] for s in scored)
            reason = (f"'{recommended['label']}' lowers the mean predicted infiltration probability from "
                      f"{recommended['mean_probability_without']:.0%} to {recommended['mean_probability_with']:.0%} "
                      f"with {recommended['disruption']} operational disruption"
                      + ("" if recommended["mean_reduction"] >= best - 1e-9 else
                         f" (achieves {recommended['mean_reduction'] / best:.0%} of the best possible reduction "
                         f"while being less disruptive)")
                      + ".")

    playbook = dict(PLAYBOOK.get(expected_stage, GENERIC_PLAYBOOK)) if expected_stage else dict(GENERIC_PLAYBOOK)
    if expected_stage:
        playbook["attack_mapping"] = ATTACK_STAGE_MAP.get(expected_stage)

    return {
        "host_id": host_id,
        "window_idx": window_idx,
        "risk_level": level,
        "peak_probability": round(max(unmitigated["infiltration_probs"]), 4),
        "expected_stage": expected_stage,
        "horizon_windows": len(unmitigated["infiltration_probs"]),
        "recommended": recommended,
        "recommendation_reason": reason,
        "evidence": scored,
        "playbook": playbook,
        "method": {
            "computed": "Each mitigation is applied inside the trained world model and the K-step rollout is compared with the unmitigated rollout.",
            "static": "Playbook text (ATT&CK mapping) and the disruption tier per mitigation are fixed reference content, not model output.",
            "limits": "Mitigation effects are modelled as documented transformations of the observed features, not measured on a real network. Decision support only; nothing is applied automatically.",
            "thresholds": {
                "act_now": ACT_THRESHOLD, "watch": WATCH_THRESHOLD,
                "min_mean_reduction": MIN_MEAN_REDUCTION, "good_enough_fraction": GOOD_ENOUGH_FRACTION,
            },
        },
    }
