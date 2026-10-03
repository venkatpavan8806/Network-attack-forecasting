"""MITRE ATT&CK stage lookup, plus tools/likely-system-state reference
context. Deliberately a small, honestly-documented table tied to this
project's specific action vocabulary (see README for the mapping
rationale) -- not a claim of general-purpose ATT&CK coverage. The
tools/system-state fields are curated reference knowledge attached to
whichever action the model predicts -- see TOOLS_AND_IMPACT_MAP's docstring
in app/config.py for why this is an enrichment layer, not a model output.
"""
from app.config import ATTACK_STAGE_MAP, TOOLS_AND_IMPACT_MAP, STAGE_CLASSES


def map_stage(state_label: str) -> dict:
    if state_label not in ATTACK_STAGE_MAP:
        raise ValueError(f"unknown state label: {state_label}")
    return {
        "state_label": state_label,
        **ATTACK_STAGE_MAP[state_label],
        **TOOLS_AND_IMPACT_MAP.get(state_label, {"likely_tools": None, "likely_system_state": None}),
    }


def all_mappings() -> list[dict]:
    return [map_stage(s) for s in STAGE_CLASSES]
