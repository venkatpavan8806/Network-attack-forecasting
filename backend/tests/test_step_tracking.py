import numpy as np
import pytest

from app.config import N_FEATURES, SEQ_LEN
from app.models.lstm_world_model import LSTMWorldModel
from app.models.ngram_move_model import NGramMoveModel, END, labels_to_moves, hybrid_next_move_distribution
from app.tracking.step_tracker import padded_window, track_host, _PathRecogniser


def test_padded_window_left_pads_with_first_window():
    feats = np.arange(3 * N_FEATURES, dtype=np.float32).reshape(3, N_FEATURES)
    w, n_real = padded_window(feats, 1)
    assert w.shape == (SEQ_LEN, N_FEATURES)
    assert n_real == 2
    assert np.array_equal(w[-1], feats[1]) and np.array_equal(w[-2], feats[0])
    assert all(np.array_equal(r, feats[0]) for r in w[:-2])


def test_padded_window_full_history_is_unpadded():
    feats = np.random.default_rng(0).normal(size=(20, N_FEATURES)).astype(np.float32)
    w, n_real = padded_window(feats, 15)
    assert n_real == SEQ_LEN
    assert np.array_equal(w, feats[15 - SEQ_LEN + 1:16])


def test_track_host_predicts_from_first_window():
    model = LSTMWorldModel()
    model.eval()
    n = 12
    feats = np.random.default_rng(1).normal(size=(n, N_FEATURES)).astype(np.float32)
    labels = ["benign"] * 3 + ["port_scan"] * 5 + ["ssh_bruteforce"] * 4
    out = track_host(model, feats, list(range(n)), labels=labels)
    steps = out["steps"]
    assert len(steps) == n  # one prediction per window, starting at window 1
    assert steps[0]["warmup"] and steps[0]["history_windows_used"] == 1
    assert not steps[SEQ_LEN - 1]["warmup"]
    assert steps[-1]["correct"] is None  # no next window to score against
    assert out["summary"]["n_warmup_windows"] == SEQ_LEN - 1
    assert out["summary"]["attack_path_actual"] == ["port_scan", "ssh_bruteforce"]


def test_labels_to_moves_merges_and_drops_benign():
    assert labels_to_moves(["benign", "port_scan", "port_scan", "ssh_bruteforce", "benign"], add_end=True) == \
        ["port_scan", "ssh_bruteforce", END]


def test_ngram_predicts_learned_sequence_three_steps_ahead():
    seq = ["port_scan", "ssh_bruteforce", "ssh_lateral_movement", "c2_beacon", "data_exfiltration", END]
    ng = NGramMoveModel(order=3).fit([seq] * 5)
    best = ng.predict_sequences(["port_scan"], k=3)[0]["moves"]
    assert best == ["ssh_bruteforce", "ssh_lateral_movement", "c2_beacon"]
    dist = ng.next_distribution(["port_scan", "ssh_bruteforce"])
    assert abs(sum(dist.values()) - 1) < 1e-9
    assert dist["ssh_bruteforce"] == 0.0  # a move never immediately repeats


def test_ngram_roundtrip_serialisation():
    ng = NGramMoveModel(order=3).fit([["port_scan", "rdp_bruteforce", END]])
    ng2 = NGramMoveModel.from_dict(ng.to_dict())
    assert ng2.next_distribution(["port_scan"]) == pytest.approx(ng.next_distribution(["port_scan"]))


def test_hybrid_uses_lstm_evidence_to_pick_variant():
    from app.config import STAGE_TO_IDX, STAGE_CLASSES
    seqs = [["port_scan", b, END] for b in ("ssh_bruteforce", "rdp_bruteforce", "smb_bruteforce")]
    ng = NGramMoveModel(order=3).fit(seqs)
    probs = np.full(len(STAGE_CLASSES), 0.01)
    probs[STAGE_TO_IDX["port_scan"]] = 0.6
    probs[STAGE_TO_IDX["smb_bruteforce"]] = 0.3
    hyb = hybrid_next_move_distribution(ng, ["port_scan"], probs / probs.sum())
    assert max(hyb, key=hyb.get) == "smb_bruteforce"


def test_path_recogniser_revises_same_phase_and_ignores_backwards():
    r = _PathRecogniser(confirm=2)
    for a in ["port_scan"] * 2 + ["rdp_bruteforce"] * 2 + ["ssh_bruteforce"] * 2 + ["port_scan"] * 3:
        path = r.update(a)
    assert path == ["port_scan", "ssh_bruteforce"]
