"""Step-by-step attacker tracking: a next-step prediction after EVERY window,
starting from the very first one.

The existing forecast (forecast_demo_host / live capture) only produces a
prediction once SEQ_LEN = 8 windows of history exist, and then only for the
latest window. This module instead walks a host's timeline from window 1:

    after window 1 -> predict window 2
    after window 2 -> predict window 3
    ...

and keeps a running track of what the attacker has done so far and what
they are likely to do next.

Warm-up (the first SEQ_LEN - 1 windows): the trained LSTM expects an
8-window input. When fewer real windows exist, the input is left-padded by
repeating the earliest real window ("edge padding") -- i.e. the model is
told "assume the host looked like its first observed window before we
started watching". No extra training is involved; the warm-up predictions
are flagged (`warmup: true`) and their accuracy is reported separately in
app/evaluate_step_tracking.py so their quality is measured, not assumed.

Each step reports:
  - predicted_next_action : argmax of the LSTM's softmax P(action at t+1 | windows 1..t)
  - confidence            : that argmax probability (the metric the decision is based on)
  - top_candidates        : top-3 actions with probabilities
  - infiltration_probability : 1 - P(benign)
  - attack_path_so_far    : the attacker's distinct moves recognised so far
  - next_moves            : next 1 / 2 / 3 MOVES (n-gram + LSTM hybrid), if the n-gram model is available
  - actual_* / correct    : ground truth, only when the data has labels
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from app.config import SEQ_LEN, IDX_TO_STAGE, STAGE_CLASSES
from app.models.attack_mapping import map_stage
from app.models.lstm_world_model import infiltration_probability
from app.models.ngram_move_model import (
    NGramMoveModel, NON_MOVE_ACTIONS, KILL_CHAIN_PHASE, hybrid_next_move_distribution, labels_to_moves,
)

MOVE_CONFIRM_WINDOWS = 2  # a new move enters the tracked path once predicted in this many consecutive windows


def padded_window(feats_scaled: np.ndarray, t: int, seq_len: int = SEQ_LEN) -> tuple[np.ndarray, int]:
    """Input window ending at position t (inclusive). Returns (window of shape
    (seq_len, n_features), number of REAL windows in it). Positions before the
    first observed window are filled with a copy of the first observed window."""
    start = max(0, t - seq_len + 1)
    real = feats_scaled[start:t + 1]
    n_real = len(real)
    if n_real < seq_len:
        pad = np.repeat(real[:1], seq_len - n_real, axis=0)
        real = np.concatenate([pad, real], axis=0)
    return real.astype(np.float32), n_real


@torch.no_grad()
def batch_next_step_probs(model, feats_scaled: np.ndarray, seq_len: int = SEQ_LEN) -> tuple[np.ndarray, np.ndarray]:
    """Runs the LSTM once per window position (batched). Returns
    (probs: (n, n_classes) = P(action at t+1 | up to t), n_real: (n,))."""
    model.eval()
    windows, n_real = [], []
    for t in range(len(feats_scaled)):
        w, r = padded_window(feats_scaled, t, seq_len)
        windows.append(w)
        n_real.append(r)
    if not windows:
        return np.zeros((0, len(STAGE_CLASSES))), np.zeros(0, dtype=int)
    x = torch.tensor(np.stack(windows), dtype=torch.float32)
    logits, _, _ = model(x)
    return F.softmax(logits, dim=1).numpy(), np.array(n_real)


class _PathRecogniser:
    """Keeps the running 'attack path so far' from the model's own per-window
    predictions (no ground truth needed, so it also works on live traffic).

    Rules (plain heuristics, not learned):
      - a move is accepted once it has been the top prediction for
        `confirm` consecutive windows (filters one-window flickers);
      - same kill-chain phase, different variant (e.g. rdp_bruteforce then
        ssh_bruteforce) -> the last move is REVISED, not appended: later
        windows carry more traffic evidence about which service is really
        targeted;
      - a move from an earlier phase than the last accepted one is ignored
        (an attack path is tracked forward along the kill chain);
      - skipping ahead more than one phase needs twice the confirmation,
        so a short burst of e.g. "exfiltration" during a long scan does
        not lock the path at the end of the kill chain."""

    def __init__(self, confirm: int | None = None):
        self.confirm = MOVE_CONFIRM_WINDOWS if confirm is None else confirm
        self.moves: list[str] = []
        self._candidate: str | None = None
        self._streak = 0

    def update(self, action: str) -> list[str]:
        if action == self._candidate:
            self._streak += 1
        else:
            self._candidate, self._streak = action, 1
        if self._streak >= self.confirm and action not in NON_MOVE_ACTIONS:
            if not self.moves:
                self.moves.append(action)
            elif self.moves[-1] != action:
                last_phase = KILL_CHAIN_PHASE.get(self.moves[-1], 0)
                phase = KILL_CHAIN_PHASE.get(action, 0)
                if phase == last_phase:
                    self.moves[-1] = action
                elif phase == last_phase + 1 or (phase > last_phase and self._streak >= 2 * self.confirm):
                    self.moves.append(action)
        return list(self.moves)


def next_moves_block(ngram: NGramMoveModel | None, path: list[str], stage_probs, k: int = 3) -> dict | None:
    """Next 1/2/3 moves for the current path (n-gram grammar + LSTM evidence
    for the first move). None if no n-gram model or no attack activity yet."""
    if ngram is None or not path:
        return None
    first = hybrid_next_move_distribution(ngram, path, stage_probs)
    return {
        "method": "ngram(order=%d) + LSTM evidence" % ngram.order,
        "per_step": ngram.per_step_top(path, k=k, top=3, first_step_dist=first),
        "top_sequences": ngram.predict_sequences(path, k=k, beam=3, first_step_dist=first),
    }


def track_host(model, feats_scaled: np.ndarray, window_idxs: list[int],
               labels: list[str] | None = None, ngram: NGramMoveModel | None = None,
               seq_len: int = SEQ_LEN) -> dict:
    """Step-by-step track for one host. `feats_scaled`: (n, n_features),
    rows in time order; `labels`: optional per-window ground-truth action."""
    probs_all, n_real_all = batch_next_step_probs(model, feats_scaled, seq_len)
    recogniser = _PathRecogniser()
    true_moves_full = labels_to_moves(labels) if labels is not None else None

    steps = []
    for t in range(len(feats_scaled)):
        probs = probs_all[t]
        order = np.argsort(-probs)
        pred = IDX_TO_STAGE[int(order[0])]
        path = recogniser.update(pred)
        step = {
            "step": t + 1,
            "window_idx": int(window_idxs[t]),
            "history_windows_used": int(n_real_all[t]),
            "warmup": bool(n_real_all[t] < seq_len),
            "predicted_next_action": pred,
            "confidence": round(float(probs[order[0]]), 4),
            "top_candidates": [
                {"action": IDX_TO_STAGE[int(i)], "probability": round(float(probs[i]), 4)} for i in order[:3]
            ],
            "infiltration_probability": round(infiltration_probability(probs), 4),
            "attack_mapping": map_stage(pred),
            "attack_path_so_far": path,
            "next_moves": next_moves_block(ngram, path, probs),
        }
        if labels is not None:
            step["actual_current_action"] = labels[t]
            if t + 1 < len(labels):
                actual_next = labels[t + 1]
                step["actual_next_action"] = actual_next
                step["correct"] = pred == actual_next
                step["correct_top3"] = actual_next in [c["action"] for c in step["top_candidates"]]
                step["is_transition"] = actual_next != labels[t]
            else:
                step["actual_next_action"] = None
                step["correct"] = None
        steps.append(step)

    return {"steps": steps, "summary": summarise_track(steps, true_moves_full)}


def summarise_track(steps: list[dict], true_moves: list[str] | None = None) -> dict:
    scored = [s for s in steps if s.get("correct") is not None]
    out = {
        "n_windows": len(steps),
        "n_warmup_windows": sum(1 for s in steps if s["warmup"]),
        "first_prediction_after_window": 1 if steps else None,
        "old_first_prediction_after_window": SEQ_LEN,
        "first_attack_alert_step": next(
            (s["step"] for s in steps if s["predicted_next_action"] not in NON_MOVE_ACTIONS), None),
        "attack_path_recognised": steps[-1]["attack_path_so_far"] if steps else [],
        "decision_rule": "predicted_next_action = argmax_a P(a | windows so far) from the LSTM softmax; "
                         "confidence = that probability; infiltration = 1 - P(benign)",
    }
    if scored:
        def acc(rows, key="correct"):
            return round(float(np.mean([r[key] for r in rows])), 4) if rows else None
        warm = [s for s in scored if s["warmup"]]
        full = [s for s in scored if not s["warmup"]]
        trans = [s for s in scored if s["is_transition"]]
        out.update({
            "accuracy_top1": acc(scored),
            "accuracy_top3": acc(scored, "correct_top3"),
            "accuracy_warmup_windows": acc(warm),
            "accuracy_full_history_windows": acc(full),
            "n_transitions": len(trans),
            "accuracy_on_transitions": acc(trans),
            "attack_path_actual": true_moves,
        })
    return out
