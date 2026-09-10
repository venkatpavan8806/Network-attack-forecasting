"""MITRE ATT&CK stage lookup. Deliberately a small, honestly-documented
table tied to this project's specific six-stage vocabulary (see README for
the mapping rationale) -- not a claim of general-purpose ATT&CK coverage.
"""
from app.config import ATTACK_STAGE_MAP, STAGE_CLASSES


def map_stage(state_label: str) -> dict:
    if state_label not in ATTACK_STAGE_MAP:
        raise ValueError(f"unknown state label: {state_label}")
    return {"state_label": state_label, **ATTACK_STAGE_MAP[state_label]}


def all_mappings() -> list[dict]:
    return [map_stage(s) for s in STAGE_CLASSES]
