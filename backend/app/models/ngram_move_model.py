"""N-gram (variable-order Markov) model over the attacker's MOVE sequence --
the "next word / next 2 words / next 3 words" predictor.

Why a second model next to the LSTM:
  The LSTM world model works at WINDOW level: "what will the next 30 s
  window of traffic look like". An attack phase lasts many windows (a port
  scan can run 7-46 windows), so the LSTM's next-window answer is usually
  "more of the same". The question "what is the attacker's NEXT STEP, and
  the one after that" is a question about the sequence of distinct moves:

      ambiguous_pre_attack -> port_scan -> ssh_bruteforce
        -> ssh_lateral_movement -> c2_beacon -> data_exfiltration -> <END>

  That is exactly the shape of next-word prediction in language modelling,
  where each move is a "word" and a whole attack is a "sentence". An n-gram
  model is the classic, transparent algorithm for it:

      P(m_{i+1} | m_{i-1}, m_i)       (order 3 = trigram)

  estimated by counting transitions in training attacks, interpolated with
  the bigram and unigram estimates (Jelinek-Mercer smoothing) so an unseen
  context still gets a sensible answer. Predicting 2 or 3 moves ahead is a
  beam search over the chained conditional probabilities.

What it cannot do on its own:
  It only sees move NAMES, not traffic. After port_scan, the n-gram knows
  the next move is "some brute-force" but it cannot know which service --
  that choice is random per attacker. The LSTM CAN, because the late part
  of the scan narrows onto the target port (dst_port_is_* features). So
  `hybrid_next_move_distribution` combines the two: n-gram = attack
  grammar, LSTM = traffic evidence. See app/evaluate_step_tracking.py for
  the measured comparison.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict

import numpy as np

from app.config import STAGE_CLASSES, MODELS_DIR

END = "<END>"
NON_MOVE_ACTIONS = {"benign"}  # benign windows are "no move", not a word in the attack sentence
MOVE_VOCAB = [s for s in STAGE_CLASSES if s not in NON_MOVE_ACTIONS] + [END]
NGRAM_MODEL_JSON = MODELS_DIR / "ngram_move_model.json"

# kill-chain phase of each move (same-phase moves are variants of one step)
KILL_CHAIN_PHASE = {
    "ambiguous_pre_attack": 0, "port_scan": 1,
    "ssh_bruteforce": 2, "rdp_bruteforce": 2, "smb_bruteforce": 2,
    "ssh_lateral_movement": 3, "rdp_lateral_movement": 3, "smb_lateral_movement": 3,
    "c2_beacon": 4, "data_exfiltration": 5,
}


def labels_to_moves(labels, add_end: bool = False) -> list[str]:
    """Collapses a per-window action sequence into the attacker's move
    sequence: consecutive duplicates merged, benign windows dropped.
    ['benign','benign','port_scan','port_scan','ssh_bruteforce'] -> ['port_scan','ssh_bruteforce']"""
    moves: list[str] = []
    for lab in labels:
        if lab in NON_MOVE_ACTIONS:
            continue
        if not moves or moves[-1] != lab:
            moves.append(lab)
    if add_end and moves:
        moves.append(END)
    return moves


class NGramMoveModel:
    def __init__(self, order: int = 3, lambdas: tuple[float, ...] | None = None, add_k: float = 0.1):
        if order < 1:
            raise ValueError("order must be >= 1")
        self.order = order
        # interpolation weights, highest order first; must sum to 1
        if lambdas is None:
            lambdas = {1: (1.0,), 2: (0.85, 0.15), 3: (0.6, 0.3, 0.1)}.get(order) or tuple(
                [0.5] + [0.5 / (order - 1)] * (order - 1))
        if len(lambdas) != order or abs(sum(lambdas) - 1.0) > 1e-6:
            raise ValueError("lambdas must have `order` entries summing to 1")
        self.lambdas = tuple(lambdas)
        self.add_k = add_k
        self.vocab = list(MOVE_VOCAB)
        # counts[n][context_tuple][next_move]; context length = n - 1
        self.counts: dict[int, dict[tuple, Counter]] = {n: defaultdict(Counter) for n in range(1, order + 1)}

    # -- training ---------------------------------------------------------
    def fit(self, move_sequences: list[list[str]]) -> "NGramMoveModel":
        for seq in move_sequences:
            for i, nxt in enumerate(seq):
                for n in range(1, self.order + 1):
                    ctx_len = n - 1
                    if i - ctx_len < 0:
                        continue
                    ctx = tuple(seq[i - ctx_len:i])
                    self.counts[n][ctx][nxt] += 1
        return self

    # -- inference --------------------------------------------------------
    def _order_dist(self, n: int, history: list[str]) -> np.ndarray | None:
        ctx_len = n - 1
        if len(history) < ctx_len:
            return None
        ctx = tuple(history[len(history) - ctx_len:]) if ctx_len else ()
        c = self.counts[n].get(ctx)
        if not c:
            return None
        v = np.array([c.get(m, 0) for m in self.vocab], dtype=np.float64) + self.add_k
        return v / v.sum()

    def next_distribution(self, history: list[str]) -> dict[str, float]:
        """P(next move | history), interpolated over all orders that have
        data for this context (weights of missing orders are redistributed)."""
        mix = np.zeros(len(self.vocab))
        wsum = 0.0
        for lam, n in zip(self.lambdas, range(self.order, 0, -1)):
            d = self._order_dist(n, history)
            if d is not None:
                mix += lam * d
                wsum += lam
        if wsum == 0:
            mix = np.ones(len(self.vocab))
            wsum = mix.sum()
        mix /= wsum
        # a move never immediately repeats itself (duplicates were merged)
        if history:
            last = history[-1]
            if last in self.vocab:
                mix[self.vocab.index(last)] = 0.0
                mix /= mix.sum()
        return {m: float(p) for m, p in zip(self.vocab, mix)}

    def predict_sequences(self, history: list[str], k: int = 3, beam: int = 5,
                          first_step_dist: dict[str, float] | None = None) -> list[dict]:
        """Beam search for the `beam` most probable next-k-move continuations.
        `first_step_dist` optionally overrides the distribution for the first
        step (used by the hybrid model to inject LSTM traffic evidence)."""
        beams = [([], 1.0)]
        for step in range(k):
            cand = []
            for seq, p in beams:
                if seq and seq[-1] == END:
                    cand.append((seq, p))
                    continue
                dist = first_step_dist if (step == 0 and first_step_dist) else self.next_distribution(history + seq)
                for m, q in dist.items():
                    if q > 0:
                        cand.append((seq + [m], p * q))
            cand.sort(key=lambda x: -x[1])
            beams = cand[:beam]
        return [{"moves": s, "probability": round(p, 4)} for s, p in beams]

    def per_step_top(self, history: list[str], k: int = 3, top: int = 3,
                     first_step_dist: dict[str, float] | None = None) -> list[list[dict]]:
        """Marginal top-`top` candidates for each of the next k moves
        (marginalised over all paths, not only the best beam)."""
        frontier = {(): 1.0}
        out = []
        for step in range(k):
            marg: Counter = Counter()
            new_frontier: dict[tuple, float] = defaultdict(float)
            for seq, p in frontier.items():
                if seq and seq[-1] == END:
                    marg[END] += p
                    new_frontier[seq] += p
                    continue
                dist = first_step_dist if (step == 0 and first_step_dist) else self.next_distribution(history + list(seq))
                for m, q in dist.items():
                    if q <= 0:
                        continue
                    marg[m] += p * q
                    new_frontier[seq + (m,)] += p * q
            # keep the frontier small
            frontier = dict(sorted(new_frontier.items(), key=lambda kv: -kv[1])[:50])
            out.append([{"move": m, "probability": round(p, 4)} for m, p in marg.most_common(top)])
        return out

    # -- persistence ------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "order": self.order,
            "lambdas": list(self.lambdas),
            "add_k": self.add_k,
            "vocab": self.vocab,
            "counts": {
                str(n): [[list(ctx), dict(c)] for ctx, c in table.items()]
                for n, table in self.counts.items()
            },
        }

    @classmethod
    def from_dict(cls, d: dict) -> "NGramMoveModel":
        m = cls(order=d["order"], lambdas=tuple(d["lambdas"]), add_k=d["add_k"])
        m.vocab = d["vocab"]
        for n, rows in d["counts"].items():
            for ctx, c in rows:
                m.counts[int(n)][tuple(ctx)] = Counter(c)
        return m

    def save(self, path=NGRAM_MODEL_JSON):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path=NGRAM_MODEL_JSON) -> "NGramMoveModel":
        with open(path) as f:
            return cls.from_dict(json.load(f))


def lstm_next_move_evidence(stage_probs: np.ndarray | list[float], current_move: str | None) -> dict[str, float]:
    """Turns the LSTM's next-window class distribution into evidence about
    the NEXT DISTINCT move: drop benign, the move already in progress, and
    moves from an EARLIER kill-chain phase (lingering traffic of a finished
    step is not a next step); renormalise what is left. Returns {} if no
    mass is left."""
    probs = np.asarray(stage_probs, dtype=np.float64)
    cur_phase = KILL_CHAIN_PHASE.get(current_move, -1)
    ev = {}
    for i, s in enumerate(STAGE_CLASSES):
        if s in NON_MOVE_ACTIONS or s == current_move or KILL_CHAIN_PHASE.get(s, 99) < cur_phase:
            continue
        ev[s] = float(probs[i])
    tot = sum(ev.values())
    if tot <= 0:
        return {}
    return {s: p / tot for s, p in ev.items()}


def hybrid_next_move_distribution(ngram: NGramMoveModel, history: list[str],
                                  stage_probs, beta: float = 1.0, eps: float = 0.05) -> dict[str, float]:
    """Product-of-experts for the next move:
        P(m) ∝ P_ngram(m | history) * (eps + P_lstm(m | traffic))^beta
    n-gram supplies the attack grammar (which kind of move can come next),
    the LSTM supplies traffic evidence (which specific variant). `eps` keeps
    a move the LSTM has not seen evidence for yet from being ruled out."""
    ng = ngram.next_distribution(history)
    ev = lstm_next_move_evidence(stage_probs, history[-1] if history else None)
    out = {}
    for m, p in ng.items():
        out[m] = p * (eps + ev.get(m, 0.0)) ** beta
    tot = sum(out.values())
    if tot <= 0:
        return ng
    return {m: p / tot for m, p in out.items()}
