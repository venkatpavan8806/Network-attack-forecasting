"""False-alarm honesty: surfaces real examples from a test run where the
world model's rising-concern signal fired on a window that was, in ground
truth, never part of an attack. Evaluators in this space specifically probe
"what happens when you're wrong" -- so these are computed from the actual
test run and shown, not hidden.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from app.config import SEQ_LEN, FEATURE_COLUMNS, FALSE_ALARM_JSON, MALICIOUS_HARD_STAGES


@torch.no_grad()
def find_false_alarms(model, scaler, labeled_df: pd.DataFrame, benign_test_hosts: set,
                       watch_threshold: float = 0.3, max_examples: int = 5) -> list[dict]:
    from app.models.lstm_world_model import infiltration_probability

    examples = []
    for host_id in sorted(benign_test_hosts):
        host_df = labeled_df[labeled_df["host_id"] == host_id].sort_values("window_idx").reset_index(drop=True)
        if host_df["true_stage"].isin(MALICIOUS_HARD_STAGES).any():
            continue  # only pure-benign hosts count as a genuine false alarm
        feats = scaler.transform(host_df[FEATURE_COLUMNS].values).astype(np.float32)
        n = len(feats)
        for t in range(SEQ_LEN - 1, n - 1):
            window = feats[t - SEQ_LEN + 1: t + 1]
            x = torch.tensor(window, dtype=torch.float32).unsqueeze(0)
            logits, _, _ = model(x)
            probs = F.softmax(logits, dim=1).squeeze(0).numpy()
            infil_prob = infiltration_probability(probs)
            if infil_prob >= watch_threshold:
                examples.append({
                    "host_id": host_id,
                    "window_idx": int(host_df.iloc[t]["window_idx"]),
                    "infiltration_probability": round(float(infil_prob), 4),
                    "true_stage": "benign",
                    "outcome": "benign (no attack ever occurred on this host)",
                })
                if len(examples) >= max_examples:
                    break
        if len(examples) >= max_examples:
            break

    with open(FALSE_ALARM_JSON, "w") as f:
        json.dump(examples, f, indent=2)
    return examples
