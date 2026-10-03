from app.config import STAGE_CLASSES
from app.inference.service import _build_branching_forecast


def _uniform_probs():
    n = len(STAGE_CLASSES)
    return [1.0 / n] * n


def test_returns_one_entry_per_horizon():
    probs_per_horizon = [_uniform_probs() for _ in range(6)]
    result = _build_branching_forecast(probs_per_horizon)
    assert len(result) == 6
    assert [r["horizon"] for r in result] == [1, 2, 3, 4, 5, 6]


def test_respects_top_k():
    probs_per_horizon = [_uniform_probs()]
    result = _build_branching_forecast(probs_per_horizon, top_k=3)
    assert len(result[0]["candidates"]) == 3


def test_candidates_sorted_by_probability_descending():
    probs = [0.0] * len(STAGE_CLASSES)
    probs[STAGE_CLASSES.index("port_scan")] = 0.6
    probs[STAGE_CLASSES.index("ssh_bruteforce")] = 0.3
    probs[STAGE_CLASSES.index("benign")] = 0.1
    result = _build_branching_forecast([probs], top_k=3)
    actions = [c["action"] for c in result[0]["candidates"]]
    assert actions == ["port_scan", "ssh_bruteforce", "benign"]


def test_each_candidate_has_real_mitre_mapping():
    probs = [0.0] * len(STAGE_CLASSES)
    probs[STAGE_CLASSES.index("ssh_lateral_movement")] = 1.0
    result = _build_branching_forecast([probs], top_k=1)
    c = result[0]["candidates"][0]
    assert c["technique_id"] == "T1021.004"
    assert c["tactic"] == "Lateral Movement"


def test_probabilities_are_rounded_floats():
    probs = [1.0 / len(STAGE_CLASSES)] * len(STAGE_CLASSES)
    result = _build_branching_forecast([probs], top_k=1)
    p = result[0]["candidates"][0]["probability"]
    assert isinstance(p, float)
    assert p == round(1.0 / len(STAGE_CLASSES), 4)
