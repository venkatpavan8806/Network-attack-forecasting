from app.defense import advisor


def _roll(probs, stages):
    return {"infiltration_probs": probs, "predicted_stage": stages}


META = {
    "rate_limit_ssh": {"label": "Rate-limit SSH", "description": ""},
    "block_ssh": {"label": "Block SSH", "description": ""},
    "isolate_host": {"label": "Isolate host", "description": ""},
}
BAD = _roll([0.9, 0.9, 0.9, 0.9], ["ssh_bruteforce"] * 4)


def test_benign_host_gets_no_recommendation():
    calm = _roll([0.01, 0.02, 0.02, 0.03], ["benign"] * 4)
    out = advisor.build_advice("h", 10, calm, {"isolate_host": calm}, META)
    assert out["risk_level"] == "monitor"
    assert out["recommended"] is None
    assert out["expected_stage"] is None


def test_prefers_least_disruptive_mitigation_that_is_good_enough():
    mitigated = {
        "rate_limit_ssh": _roll([0.48] * 4, ["ssh_bruteforce"] * 4),   # -0.42
        "block_ssh": _roll([0.42] * 4, ["ssh_bruteforce"] * 4),        # -0.48
        "isolate_host": _roll([0.40] * 4, ["benign"] * 4),             # -0.50 (best, but most disruptive)
    }
    out = advisor.build_advice("h", 10, BAD, mitigated, META)
    assert out["risk_level"] == "act_now"
    assert out["recommended"]["id"] == "rate_limit_ssh"   # 84% of the best, lowest disruption
    assert out["evidence"][0]["id"] == "isolate_host"     # evidence is sorted by raw effect


def test_falls_back_to_bigger_effect_when_light_options_are_too_weak():
    mitigated = {
        "rate_limit_ssh": _roll([0.85] * 4, ["ssh_bruteforce"] * 4),   # -0.05, far below 80% of best
        "isolate_host": _roll([0.20] * 4, ["benign"] * 4),             # -0.70
    }
    out = advisor.build_advice("h", 10, BAD, mitigated, META)
    assert out["recommended"]["id"] == "isolate_host"


def test_no_effective_mitigation_says_so():
    mitigated = {"block_ssh": _roll([0.89] * 4, ["ssh_bruteforce"] * 4)}
    out = advisor.build_advice("h", 10, BAD, mitigated, META)
    assert out["recommended"] is None
    assert "None of the modelled mitigations" in out["recommendation_reason"]


def test_furthest_expected_stage_follows_kill_chain_order():
    stages = ["benign", "port_scan", "ssh_bruteforce", "c2_beacon", "ssh_bruteforce"]
    assert advisor.furthest_expected_stage(stages) == "c2_beacon"
    assert advisor.furthest_expected_stage(["benign", "benign"]) is None


def test_playbook_is_attack_mapped_and_covers_every_malicious_stage():
    from app.config import MALICIOUS_HARD_ACTIONS
    for stage in MALICIOUS_HARD_ACTIONS | {"ambiguous_pre_attack"}:
        assert stage in advisor.PLAYBOOK, f"no playbook for {stage}"
        assert advisor.PLAYBOOK[stage]["analyst_steps"]
    out = advisor.build_advice("h", 1, BAD, {"block_ssh": _roll([0.3] * 4, ["benign"] * 4)}, META)
    ids = {m["id"] for m in out["playbook"]["mitigations"]}
    assert "M1032" in ids and out["playbook"]["attack_mapping"]["technique_id"] == "T1110"
