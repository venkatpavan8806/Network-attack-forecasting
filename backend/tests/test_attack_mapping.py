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
