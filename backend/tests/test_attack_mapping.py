import pytest

from app.config import STAGE_CLASSES
from app.models.attack_mapping import map_stage, all_mappings


def test_all_stage_classes_have_a_mapping():
    mappings = all_mappings()
    mapped_labels = {m["state_label"] for m in mappings}
    assert mapped_labels == set(STAGE_CLASSES)


def test_map_stage_returns_expected_fields():
    m = map_stage("port_scan")
    assert m["technique_id"] == "T1595"
    assert m["tactic"] == "Reconnaissance"


def test_lateral_movement_actions_use_real_attck_subtechniques():
    assert map_stage("ssh_lateral_movement")["technique_id"] == "T1021.004"
    assert map_stage("rdp_lateral_movement")["technique_id"] == "T1021.001"
    assert map_stage("smb_lateral_movement")["technique_id"] == "T1021.002"


def test_ambiguous_pre_attack_maps_to_pre_confirmation_tactic():
    m = map_stage("ambiguous_pre_attack")
    assert "Pre-Confirmation" in m["tactic"] or "pre" in (m["technique_id"] or "").lower()


def test_unknown_stage_raises():
    with pytest.raises(ValueError):
        map_stage("not_a_real_stage")


def test_every_malicious_action_has_tools_and_system_state():
    from app.config import MALICIOUS_HARD_ACTIONS
    for action in MALICIOUS_HARD_ACTIONS:
        m = map_stage(action)
        assert m["likely_tools"], f"{action} missing likely_tools"
        assert m["likely_system_state"], f"{action} missing likely_system_state"


def test_benign_has_no_tools():
    m = map_stage("benign")
    assert m["likely_tools"] is None


def test_ssh_bruteforce_tools_mention_known_tools():
    m = map_stage("ssh_bruteforce")
    assert "Hydra" in m["likely_tools"]


def test_system_state_describes_consequence_not_just_label():
    """The system-state text should describe an actual consequence, not
    just restate the action name."""
    m = map_stage("c2_beacon")
    assert "command-and-control" in m["likely_system_state"].lower() or "control" in m["likely_system_state"].lower()
