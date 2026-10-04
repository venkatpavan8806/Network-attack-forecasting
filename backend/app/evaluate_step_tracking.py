"""Evaluation for step-by-step attacker tracking and multi-step (next 1/2/3
move) prediction. Uses the ALREADY-TRAINED LSTM + scaler from app.train;
does not retrain or modify them.

Run with:  python -m app.evaluate_step_tracking      (after python -m app.train)

Produces:
  models_store/ngram_move_model.json   trigram move model (train hosts only)
  data/step_tracking_report.json       all numbers below

Sections of the report:
  1. window_level_tracking  -- a prediction after EVERY window from window 1
     (incl. the first SEQ_LEN-1 "warm-up" windows the old pipeline skipped):
     top-1 / top-3 accuracy, macro-F1, accuracy on phase transitions,
     warm-up vs full-history accuracy. Held-out (val+test) hosts only.
  2. cold_start             -- the harder warm-up case: the timeline starts
     AT the attack's first window (the attacker is seen for the first time,
     like a new IP in live capture), so all warm-up windows are attack windows.
  3. path_recognition       -- does the tracked "attack path so far" match
     the real one, at tactic (kill-chain phase) and technique level.
  4. multi_step_moves       -- next 1 / 2 / 3 MOVES ("words"): Markov
     chain, trigram, LSTM-only, and trigram + LSTM hybrid, compared.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from app.config import DATA_DIR, FEATURE_COLUMNS, SEQ_LEN, STAGE_CLASSES, STAGE_TO_IDX
from app.features.extraction import load_scaler, host_split
from app.models.lstm_world_model import load_model
from app.models.ngram_move_model import (
    NGramMoveModel, END, labels_to_moves, lstm_next_move_evidence, hybrid_next_move_distribution,
    NGRAM_MODEL_JSON,
)
from app.tracking.step_tracker import track_host, batch_next_step_probs, KILL_CHAIN_PHASE

STEP_TRACKING_JSON = DATA_DIR / "step_tracking_report.json"


def _host_arrays(labeled: pd.DataFrame, scaler, host_id: str):
    g = labeled[labeled["host_id"] == host_id].sort_values("window_idx")
    return scaler.transform(g[FEATURE_COLUMNS].values).astype(np.float32), g["window_idx"].tolist(), g["state_label"].tolist()


def _acc(x):
    return round(float(np.mean(x)), 4) if len(x) else None


# ---------------------------------------------------------------------------
# 1 + 3: window-level tracking and path recognition
# ---------------------------------------------------------------------------
def evaluate_window_tracking(model, scaler, labeled, hosts: set, attack_hosts=None) -> tuple[dict, dict]:
    is_attack = (lambda h: h in attack_hosts) if attack_hosts is not None else (lambda h: h.startswith("attack-host"))
    y_true, y_pred, top3, warm, trans = [], [], [], [], []
    path_rows = []
    for h in sorted(hosts):
        feats, widx, labels = _host_arrays(labeled, scaler, h)
        tr = track_host(model, feats, widx, labels, ngram=None)
        for s in tr["steps"]:
            if s.get("correct") is None:
                continue
            y_true.append(s["actual_next_action"])
            y_pred.append(s["predicted_next_action"])
            top3.append(s["correct_top3"])
            warm.append(s["warmup"])
            trans.append(s["is_transition"])
        if is_attack(h):
            got = tr["summary"]["attack_path_recognised"]
            real = tr["summary"]["attack_path_actual"]
            path_rows.append({
                "host_id": h,
                "recognised": got,
                "actual": real,
                "technique_level_match": got == real,
                "tactic_level_match": [KILL_CHAIN_PHASE[m] for m in got if m != "ambiguous_pre_attack"]
                                      == [KILL_CHAIN_PHASE[m] for m in real if m != "ambiguous_pre_attack"],
            })
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    correct = y_true == y_pred
    warm, trans, top3 = np.array(warm), np.array(trans), np.array(top3)
    attack_mask = y_true != "benign"
    window = {
        "hosts_evaluated": len(hosts),
        "n_predictions": int(len(y_true)),
        "n_warmup_predictions": int(warm.sum()),
        "accuracy_top1": _acc(correct),
        "accuracy_top3": _acc(top3),
        "macro_f1": round(float(f1_score(y_true, y_pred, labels=STAGE_CLASSES, average="macro", zero_division=0)), 4),
        "accuracy_warmup_windows (1..%d)" % (SEQ_LEN - 1): _acc(correct[warm]),
        "accuracy_full_history_windows": _acc(correct[~warm]),
        "accuracy_when_next_window_is_attack": _acc(correct[attack_mask]),
        "n_transitions": int(trans.sum()),
        "accuracy_on_transitions": _acc(correct[trans]),
        "top3_on_transitions": _acc(top3[trans]),
        "per_class_recall": {
            c: _acc(correct[y_true == c]) for c in STAGE_CLASSES if (y_true == c).any()
        },
    }
    paths = {
        "rule": "move accepted after 2 consecutive windows with the same top prediction; same-phase variant "
                "revises the last move; earlier-phase moves ignored; skipping >1 phase needs 2x confirmation (see tracking/step_tracker.py)",
        "n_attack_hosts": len(path_rows),
        "tactic_level_exact_match": _acc([r["tactic_level_match"] for r in path_rows]),
        "technique_level_exact_match": _acc([r["technique_level_match"] for r in path_rows]),
        "hosts": path_rows,
    }
    return window, paths


# ---------------------------------------------------------------------------
# 2: cold start -- attacker seen for the first time at window 1
# ---------------------------------------------------------------------------
def evaluate_cold_start(model, scaler, labeled, hosts: set, attack_hosts=None) -> dict:
    is_attack = (lambda h: h in attack_hosts) if attack_hosts is not None else (lambda h: h.startswith("attack-host"))
    correct_by_step = {k: [] for k in range(1, SEQ_LEN)}
    alert_after = []
    for h in sorted(hosts):
        if not is_attack(h):
            continue
        feats, widx, labels = _host_arrays(labeled, scaler, h)
        onset = next(i for i, l in enumerate(labels) if l != "benign")
        f, lab = feats[onset:], labels[onset:]
        probs, _ = batch_next_step_probs(model, f)
        pred = [STAGE_CLASSES[i] for i in probs.argmax(1)]
        for t in range(min(SEQ_LEN - 1, len(lab) - 1)):
            correct_by_step[t + 1].append(pred[t] == lab[t + 1])
        alert_after.append(next((t + 1 for t, p in enumerate(pred) if p != "benign"), None))
    return {
        "setup": "timeline starts at the first non-benign window, so every warm-up window is attack traffic "
                 "(same situation as a brand-new attacker IP in live capture)",
        "accuracy_by_step": {f"after_window_{k}": _acc(v) for k, v in correct_by_step.items()},
        "accuracy_warmup_overall": _acc(sum(correct_by_step.values(), [])),
        "first_non_benign_prediction_after_window": alert_after,
        "old_pipeline_first_prediction_after_window": SEQ_LEN,
    }


# ---------------------------------------------------------------------------
# 4: next 1 / 2 / 3 moves
# ---------------------------------------------------------------------------
def _move_eval_points(labels: list[str], probs: np.ndarray | None):
    """One evaluation point per attack window: (true move history incl. the
    current move, future moves incl. <END>, LSTM probs at that window)."""
    moves_full = labels_to_moves(labels, add_end=True)
    pts = []
    history: list[str] = []
    for t, lab in enumerate(labels):
        if lab == "benign":
            if history:  # attack over
                break
            continue
        if not history or history[-1] != lab:
            history.append(lab)
        future = moves_full[len(history):]
        pts.append((list(history), future, None if probs is None else probs[t]))
    return pts


def _phase(m):
    return "END" if m == END else KILL_CHAIN_PHASE.get(m)


def _score(method_preds: list[tuple[list[list[str]], list[str]]], max_k: int = 3) -> dict:
    """method_preds: list of (ranked candidate sequences (best first), true future)."""
    out = {}
    for k in range(1, max_k + 1):
        rows = [(c, f) for c, f in method_preds if len(f) >= k]
        if not rows:
            continue
        exact = [c[0][:k] == f[:k] for c, f in rows]
        tactic = [[_phase(x) for x in c[0][:k]] == [_phase(x) for x in f[:k]] for c, f in rows]
        out[f"next_{k}"] = {
            "n": len(rows),
            "exact_match": _acc(exact),
            "tactic_level_match": _acc(tactic),
        }
        if k == 1:
            out["next_1"]["top3_accuracy"] = _acc([f[0] in [s[0] for s in c[:3]] for c, f in rows])
    return out


def _ranked_from_ngram(ng: NGramMoveModel, hist, first=None, k=3):
    seqs = ng.predict_sequences(hist, k=k, beam=10, first_step_dist=first)
    # candidate list for next-1 top-3: marginal first-step ranking
    first_rank = ng.per_step_top(hist, k=1, top=3, first_step_dist=first)[0]
    best = seqs[0]["moves"]
    return [best] + [[c["move"]] for c in first_rank if c["move"] != best[0]]


def evaluate_multistep(model, scaler, labeled, train_hosts, heldout_hosts, attack_hosts=None,
                       trained_ngram: NGramMoveModel | None = None, save: bool = True) -> dict:
    """trained_ngram: evaluate an already-trained move model (e.g. on a user's
    uploaded traffic) instead of fitting one on train_hosts; save=False never
    touches the stored model file."""
    if attack_hosts is None:
        attack_hosts = sorted(h for h in labeled["host_id"].unique() if h.startswith("attack-host"))
    attack_hosts = sorted(attack_hosts)
    seqs = {h: labels_to_moves(_host_arrays(labeled, scaler, h)[2], add_end=True) for h in attack_hosts}

    # (a) n-gram family, leave-one-attack-host-out over ALL attack hosts (no LSTM involved)
    loho: dict[str, list] = {"unigram_most_frequent": [], "markov_order1": [], "trigram_order2": []}
    for h in attack_hosts:
        others = [seqs[o] for o in attack_hosts if o != h]
        models = {
            "unigram_most_frequent": NGramMoveModel(order=1).fit(others),
            "markov_order1": NGramMoveModel(order=2).fit(others),
            "trigram_order2": NGramMoveModel(order=3).fit(others),
        }
        labels = _host_arrays(labeled, scaler, h)[2]
        for hist, fut, _ in _move_eval_points(labels, None):
            for name, ng in models.items():
                loho[name].append((_ranked_from_ngram(ng, hist), fut))

    # (b) head-to-head on held-out attack hosts (the LSTM never trained on them);
    # n-gram trained on TRAIN attack hosts only -- same split as the LSTM.
    if trained_ngram is not None:
        ng_train = trained_ngram
        markov_train = NGramMoveModel(order=2)  # first-order view of the same trained counts
        markov_train.counts[1], markov_train.counts[2] = ng_train.counts[1], ng_train.counts[2]
    else:
        train_seqs = [seqs[h] for h in attack_hosts if h in train_hosts]
        ng_train = NGramMoveModel(order=3).fit(train_seqs)
        markov_train = NGramMoveModel(order=2).fit(train_seqs)
    h2h: dict[str, list] = {"markov_order1": [], "trigram_order2": [], "lstm_only": [], "hybrid_trigram_plus_lstm": []}
    for h in sorted(x for x in heldout_hosts if x in attack_hosts):
        feats, _, labels = _host_arrays(labeled, scaler, h)
        probs, _ = batch_next_step_probs(model, feats)
        for hist, fut, p in _move_eval_points(labels, probs):
            h2h["markov_order1"].append((_ranked_from_ngram(markov_train, hist), fut))
            h2h["trigram_order2"].append((_ranked_from_ngram(ng_train, hist), fut))
            ev = lstm_next_move_evidence(p, hist[-1])
            ranked = sorted(ev, key=lambda m: -ev[m]) or ["?"]
            h2h["lstm_only"].append(([[m] for m in ranked], fut))
            hyb = hybrid_next_move_distribution(ng_train, hist, p)
            h2h["hybrid_trigram_plus_lstm"].append((_ranked_from_ngram(ng_train, hist, first=hyb), fut))

    if save:
        ng_train.save(NGRAM_MODEL_JSON)
    return {
        "unit": "a MOVE = a distinct attacker action (consecutive identical windows merged, benign dropped, "
                "<END> = attack finished). One evaluation point per attack window; history = true moves so far.",
        "leave_one_host_out_all_attack_hosts": {
            name: _score(v) for name, v in loho.items()
        },
        "heldout_hosts_vs_lstm": {
            "hosts": sorted(x for x in heldout_hosts if x in attack_hosts),
            "note": "lstm_only only predicts the next single move (it is a next-WINDOW model); "
                    "the hybrid uses LSTM evidence for move 1 and trigram grammar for moves 2-3.",
            **{name: _score(v, max_k=1 if name == "lstm_only" else 3) for name, v in h2h.items()},
        },
        "saved_model": str(NGRAM_MODEL_JSON.name),
    }


def main():
    labeled = pd.read_csv(DATA_DIR / "labeled_states.csv")
    model = load_model()
    scaler = load_scaler()
    train_hosts, val_hosts, test_hosts = host_split(labeled)
    heldout = val_hosts | test_hosts

    print("== window-level step-by-step tracking (held-out hosts) ==")
    window, paths = evaluate_window_tracking(model, scaler, labeled, heldout)
    print(json.dumps(window, indent=2))
    print("\n== cold start (attacker first seen at window 1) ==")
    cold = evaluate_cold_start(model, scaler, labeled, heldout)
    print(json.dumps(cold, indent=2))
    print("\n== attack path recognition (all attack hosts) ==")
    all_attack = {h for h in labeled["host_id"].unique() if h.startswith("attack-host")}
    _, paths_all = evaluate_window_tracking(model, scaler, labeled, all_attack)
    paths_all["heldout_only"] = {k: paths[k] for k in ("n_attack_hosts", "tactic_level_exact_match", "technique_level_exact_match")}
    print(json.dumps({k: v for k, v in paths_all.items() if k != "hosts"}, indent=2))
    print("\n== next 1 / 2 / 3 moves ==")
    multi = evaluate_multistep(model, scaler, labeled, train_hosts, heldout)
    print(json.dumps(multi, indent=2))

    report = {
        "decision_metric": {
            "next_step_rule": "argmax over the LSTM softmax: predicted = argmax_a P(action at t+1 = a | windows 1..t)",
            "confidence": "the winning softmax probability P(predicted action)",
            "infiltration_probability": "1 - P(benign), alert when >= 0.5 (same threshold as the benchmark)",
            "next_moves": "trigram P(m_{i+1} | m_{i-1}, m_i) x LSTM evidence for the first move, chained "
                          "(beam search) for moves 2 and 3",
            "evaluation_metrics": "top-1 / top-3 accuracy, macro-F1, accuracy on transitions (next window "
                                  "differs from current), exact-sequence match for next-k moves",
        },
        "window_level_tracking": window,
        "cold_start": cold,
        "path_recognition": paths_all,
        "multi_step_moves": multi,
    }
    with open(STEP_TRACKING_JSON, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {STEP_TRACKING_JSON}")


if __name__ == "__main__":
    main()
