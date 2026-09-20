# AI-Based Network Attack Forecasting from Network Traffic Data

SIH26153 (Theme: Cybersecurity, NCIIPC — Critical Information Infrastructure
Protection). Team CARIBBEAN.

Governing principle:

```
TRAFFIC -> EVOLVING PROBABILITY -> EXPLAINED TREND -> DEFENDER ACTS EARLY
```

Never: `TRAFFIC -> AI -> "ATTACK CONFIRMED"`. Every prediction in this system
is a calibrated probability with an explanation attached, not a verdict.

## What this actually is (novelty, stated honestly)

Temporal modeling + ATT&CK-stage mapping + explainability, as a combination,
is **not new** — prior systems already do next-attack-step prediction once an
attack has been flagged, at very high F1. This project's actual contribution
is narrower and specific:

> Forecasts infiltration likelihood from raw, pre-detection flow and packet
> telemetry — before any window is labeled malicious — by deriving
> state-transition labels for the ambiguous pre-attack windows itself
> (rather than inheriting a dataset's existing "Normal" label), and
> reporting a calibrated, multi-step confidence trajectory with measured
> lead-time over a static baseline.

This is an output-format and evaluation contribution built on a standard
LSTM sequence model — not a claim of inventing world models, LSTMs, or GNNs
for network security, and not a claim that no related work exists.

## Repository layout

```
backend/
  app/
    config.py                 stage vocabulary, feature schema, ATT&CK map, all hyperparameters
    data_gen/generator.py     synthetic flow+packet telemetry generator
    labeling/state_labeler.py state-labeling engine (incl. ambiguous_pre_attack derivation)
    features/extraction.py    scaler, host-level train/val/test split, sequence building
    models/
      lstm_world_model.py     LSTM world model (attention + rollout + branching rollout + saliency)
      baseline_lr.py          logistic regression baseline
      attack_mapping.py       MITRE ATT&CK stage lookup
    explain/
      attention.py            attention + input-gradient explanation for the LSTM
      shap_baseline.py        SHAP, validated only on the baseline
    evaluation/
      metrics.py               F1 / precision / recall / FPR benchmark
      calibration.py           reliability diagram at a fixed rollout horizon
      lead_time.py             lead-time metric vs. the baseline
      false_alarms.py          real false-alarm examples from held-out benign hosts
    inference/service.py      shared inference layer used by the API
    api/main.py                FastAPI app
    db.py                      SQLite-backed inference log (KPIs, forecast log table)
    train.py                   end-to-end training/evaluation entry point
  tests/                      pytest suite (75 tests)
  data/                       generated CSVs + JSON reports (all produced by train.py)
  models_store/               saved LSTM weights + baseline + scaler
frontend/
  src/                        React + TypeScript + Recharts dashboard
```

## Running it

Backend (Python 3.12):

```bash
cd backend
python -m venv venv
venv\Scripts\pip install -r requirements.txt
venv\Scripts\python -m app.train        # generates data, trains both models, computes all reports (~30s on CPU)
venv\Scripts\python -m pytest -q        # 75 tests
venv\Scripts\python -m uvicorn app.api.main:app --port 8000
```

Frontend (Node 22):

```bash
cd frontend
npm install
npm run dev        # proxies /api/* to http://127.0.0.1:8000, open the printed localhost URL
```

Everything runs fully offline — no cloud API calls anywhere in the pipeline
or the UI.

## Feature schema and anti-leakage rule

`app/config.py:FEATURE_COLUMNS` lists the 27 traffic-shaped features (flow
counts, TCP flag counts, flow duration stats, byte/packet stats, inter-
arrival-time stats, TTL stats, TCP window-size stats, packet-length stats, a
port-scan score, and new-destination-IP ratio). **The ground-truth attack
stage is never one-hot-encoded or otherwise embedded in this feature
vector** — it is generated and stored as a separate `true_stage` /
`state_label` column, and the LSTM and baseline only ever see
`FEATURE_COLUMNS`. This was flagged in the project brief as a bug in an
earlier prototype skeleton; `tests/test_generator.py::test_no_stage_leakage_into_feature_columns`
guards against it regressing.

## Synthetic data generator

`app/data_gen/generator.py` scripts each host through a stage sequence
(`benign -> reconnaissance -> initial_access -> lateral_movement ->
command_and_control -> exfiltration -> benign`) and samples that window's
27 features from a stage-specific distribution (e.g. reconnaissance has a
high port-scan score and failed-connection ratio; C2 has a long, suspiciously
*regular* beacon interval; exfiltration has a large outbound/inbound byte
ratio). 40 hosts are pure benign background traffic; 6 hosts run the full
attack progression. One of the six (`attack-host-000`) runs a **slow/evasive
reconnaissance variant**: low per-window flow count (mean ≈ 10.7 vs ≈ 118 for
the fast variant) spread over 35–55 windows with heavily jittered timing, so
a naive flow-count threshold would miss it — it is only visible through the
packet-level features (IAT jitter, TTL variance, cumulative port-scan
score). This host is pinned to the test split so the reported benchmark
demonstrates generalization to it, not just to the fast/obvious attacks.

This is a synthetic, hand-tuned generator, not a capture of real traffic —
see **Known limitations** below and **Swapping in a real dataset**.

## State-labeling engine and the ambiguous-window judgment call

`app/labeling/state_labeler.py` is the project's actual point of novelty.
For every window immediately preceding a hard attack-stage onset (the
`AMBIGUOUS_LOOKBACK = 4` windows before it), it computes a **precursor
score** from that window's own features only (no lookahead):

```
score = 0.35 * norm(port_scan_score)
      + 0.25 * norm(failed_conn_ratio)
      + 0.20 * norm(new_dst_ip_ratio)
      + 0.20 * norm(iat_std / iat_mean)      # timing irregularity
```

Each term is min-max normalized against the 5th/95th percentile range of
that feature over the benign population, so terms of very different scale
are comparable. A window scoring ≥ `AMBIGUOUS_SCORE_THRESHOLD = 0.35` is
relabeled from the simulator's own "benign" to a new class,
`ambiguous_pre_attack`, that the simulator itself never assigns. This is a
heuristic engineering judgment call — **there is no ground truth to validate
it against**, since real networks do not come with "this window was 35% an
attack precursor" labels either. The weights, threshold, and lookback are
plain constants in `app/config.py`, not another learned black box, so the
judgment stays inspectable. On the current synthetic run this relabels 24 of
24 designed precursor windows across the 6 attack hosts.

## The LSTM world model

`app/models/lstm_world_model.py`: a single-layer LSTM (hidden size 64) with
additive (Bahdanau-style) attention pooling over the last `SEQ_LEN = 8`
windows, feeding two heads:

- **next_stage_head** — classifies the *next* window's state label (7
  classes: `benign`, `ambiguous_pre_attack`, and the 5 hard attack stages).
- **next_state_head** — regresses the next window's normalized 27-feature
  vector. This is what makes K-step rollout possible: `rollout()` feeds the
  model's own predicted next state back in as the new most-recent
  observation and repeats for `ROLLOUT_K = 6` steps, producing a genuine
  multi-step probability curve rather than a single point score.

Infiltration probability at any horizon = `1 - P(benign)` (the
`ambiguous_pre_attack` class counts as partial evidence, at whatever weight
the model itself assigned it — no extra hand-tuning on top).

Training: combined cross-entropy (class-balanced, since attack windows are
rare) + MSE loss, Adam, early stopping on a held-out validation split (by
host, never by row — see below).

## Branching K-step forecast + MITRE ATT&CK kill-chain

`rollout()` above produces one committed path: at every step it follows only
the model's own argmax continuation. `app/models/lstm_world_model.py:branching_rollout`
forks that into an actual **attack-forecast tree**: at every step it keeps the
`BRANCH_FACTOR` (default 3) most probable next actions instead of just one,
prunes any branch whose cumulative path probability drops below
`BRANCH_MIN_PATH_PROB`, and stops at `BRANCH_DEPTH` (default 4) steps. Every
node carries the MITRE ATT&CK mapping for its action
(`app/models/attack_mapping.py`), so the tree reads directly as a set of
plausible, ranked ATT&CK kill-chain continuations — "70% chance of
`T1595 Active Scanning` next, forking into a 55% continuation toward
`T1110 Brute Force (SSH)` and a 20% continuation toward `T1110 Brute Force
(RDP)`" — rather than a single infiltration-probability number.

**Making sibling branches actually diverge:** `next_state_head` regresses one
continuous feature vector per input regardless of which action a branch
picks, so naively feeding every branch that same vector back in would make
sibling branches differ only in their label for exactly one step, then
collapse back onto an identical continuation. To avoid that,
`app/train.py:build_stage_mean_vectors` computes, once per training run and
from **train hosts only** (same split the scaler is fit on — no test/val
leakage), the mean normalized feature vector observed for windows of each
action class, saved to `data/stage_mean_vectors.json`. Each branch's
continued state is then a blend — `BRANCH_STATE_BLEND` (default 0.5) — of the
model's own regressed `next_state` and that branch's class-mean vector: "if
the world actually goes down this branch, also nudge the rolling assumption
of the traffic itself toward what that action typically looks like." This is
a documented heuristic layered on top of a real trained model, not a second
learned model, and it only affects which branches get shown — it is never
used in training or in any reported benchmark/calibration/lead-time metric.

Exposed via `GET /forecast/{host_id}/branches`
(`app/inference/service.py:branching_forecast`) and rendered in the
"Branching Attack-Path Forecast" card on the Forecasts tab
(`frontend/src/components/BranchingForecastTree.tsx`), alongside the
single highest-path-probability continuation and the continuation with the
highest final infiltration probability, each shown as an ordered MITRE
kill-chain. `backend/tests/test_lstm.py` covers tree shape/depth/pruning and
that branches genuinely condition on the chosen action (not just cosmetically
labeled).

## Explainability

**Primary (LSTM):** the attention weights above (which past real windows the
forecast leaned on) combined with input-gradient saliency (gradient of the
infiltration probability w.r.t. the actual input, on the actual trained
model) — `app/explain/attention.py`. Both are computed fresh for every
prediction; neither is a static importance table.

**Baseline only:** SHAP (`shap.LinearExplainer`) — reserved for the logistic
regression baseline because SHAP on a recurrent sequence model is fragile,
per the project brief.

## Baseline and the benchmark comparison

`app/models/baseline_lr.py` is a plain, class-balanced logistic regression —
the "traditional tool" the problem statement contrasts the world model
against. It classifies a single window using only that window's own
features, no history.

`app/evaluation/metrics.py` scores both models on the same held-out,
by-host test split, against the same ground truth (`true_stage(w)` in the
five confirmed hard attack stages), but each in the mode it actually
operates in: the baseline classifies window `w` using window `w`'s own
(already-happened) features; the world model forecasts window `w` using only
history through window `w-1` (available a full window before `w` occurs).
That timing gap is also what the lead-time metric below measures.

## Host-level splitting (why it matters)

`app/features/extraction.py:host_split` splits by **host**, never by row —
splitting by row would leak adjacent-window information from the same
timeline across train/test. The evasive-recon host is pinned to test. The
scaler is fit on train hosts only.

## Results from the actual run committed in this repo

(`backend/data/benchmark_report.json`, `lead_time_report.json`,
`calibration_report.json` — regenerate with `python -m app.train`; nothing
below is hard-coded.)

**Benchmark (held-out test split, threshold 0.5):**

| | Precision | Recall | F1 | FPR | n |
|---|---|---|---|---|---|
| Baseline (logistic regression) | 1.000 | 1.000 | 1.000 | 0.000 | 1325 |
| World model (LSTM, one-step-ahead) | 0.975 | 0.987 | 0.981 | 0.0017 | 1269 |

**Lead time:** median 0.5 windows (0.25 min) across all 6 attack hosts;
median 0.0 windows on the held-out-only evasive host. All 6 hosts were
correctly flagged by both models.

**Calibration @ t+3 rollout horizon:** Brier score 0.0424 over 2455 held-out
rollout points. The two extreme bins (predicted 0.8–0.9 and 0.9–1.0) track
observed frequency closely (0.85→0.09 observed is the one weak point;
0.99→0.93 observed is good); the mid-range bins (0.1–0.8) currently show
observed frequency 0.0, i.e. the model is not yet well-calibrated when it
hedges — see limitations below.

## Known limitations (stated plainly, not hidden)

- **The synthetic attack signatures are cleanly separable by design**, which
  is why both models score very highly (the baseline hits a perfect 1.000
  F1 on this test split). This is a property of the current hand-tuned
  generator, not a general claim about either model's performance on real
  traffic — this is exactly the gap a real CIC-IDS-2018/CTU-13 run would
  close (see below).
- **Lead time is currently small** (median 0.25–0.5 minutes) because the
  synthetic recon onset is fairly sharp — the world model's advantage comes
  almost entirely from the designed ambiguous-precursor windows, and there
  are only 4 of them per host. A less abrupt/longer precursor design, a
  lower alert threshold, or real traffic with a genuinely gradual onset
  would likely widen this gap; it should not be presented as larger than
  what is actually measured here.
- **Mid-range calibration is currently poor** (observed frequency 0.0 in
  every 0.1–0.8 predicted bin at the t+3 horizon) — the model tends to be
  either very confident or wrong when it hedges, likely a consequence of
  limited ambiguous-window training data (24 examples) and the sharp
  synthetic separability above. The reliability diagram in the Benchmarks
  tab shows this as-is, not smoothed over.
- The MITRE ATT&CK mapping (`app/models/attack_mapping.py`) is a small,
  6-row table tied specifically to this project's six-stage vocabulary — not
  a general-purpose ATT&CK coverage claim.

## Swapping in a real dataset (CIC-IDS-2018 / CTU-13)

The data-ingestion layer is dataset-agnostic by construction. The exact swap
point: replace `app/data_gen/generator.py:generate_dataset()` with a loader
that reads a real CSV/PCAP and emits a DataFrame with the same columns
(`host_id`, `window_idx`, `timestamp`, `true_stage`, and the 27
`FEATURE_COLUMNS` from `app/config.py`) — every downstream module
(`state_labeler.py`, `extraction.py`, `train.py`, the API) consumes only that
schema and does not know or care where the rows came from. `true_stage`
would come from the dataset's own attack-type labels, mapped into the six
hard stages in `app/config.py:HARD_STAGES`; the `ambiguous_pre_attack`
derivation in `state_labeler.py` runs unchanged since it only depends on
`FEATURE_COLUMNS` and `true_stage`. This ~50GB dataset was deliberately not
downloaded for this pass, per project scope.

## What was deferred (Priority 3, out of scope for this pass)

- Digital twin / Mininet-GNS3 live network simulation, live DDoS/SSH
  injection demo. The frontend's "Digital Twin" nav tab is present (matching
  the reference design's tab row) but disabled with an explicit "deferred"
  message — it is not wired to any backend functionality.
- Any closed-loop automatic defensive action. Nothing in this codebase
  modifies live network policy based on a forecast; the system is decision
  support only.
- A real CIC-IDS-2018/CTU-13 loader (see swap point above).
- GNN/Transformer world-model variants (LSTM only, as specified).

## Tests

`backend/tests/` — 75 tests, `pytest -q` from `backend/`:

- `test_generator.py` — output shape/columns, no stage leakage, attack hosts
  cover the full progression, evasive recon has lower flow volume than fast
  recon, benign traffic has a low port-scan score.
- `test_labeling.py` — a hand-built fixture timeline verifies ambiguous
  windows are correctly relabeled, a pure-benign host is never relabeled,
  early benign windows far from any onset stay benign, hard malicious labels
  pass through unchanged, transition-pair shapes are correct.
- `test_features.py` — host split is disjoint and covers all hosts, the
  evasive host is pinned to test, sequence/single-window table shapes.
- `test_lstm.py` — forward-pass shapes, attention weights sum to 1,
  infiltration-probability range, rollout output shapes/ranges, saliency
  shape and finiteness, branching-rollout tree shape/depth/pruning and that
  sibling branches genuinely diverge (not cosmetically labeled).
- `test_baseline.py` — baseline trains and separates a synthetic separable
  case, metrics dict has the expected keys.
- `test_attack_mapping.py` — every stage class has a mapping, unknown stage
  raises.
- `test_e2e.py` — a small synthetic dataset run through the full pipeline
  (generate → label → split → scale → train tiny models → benchmark →
  explain → rollout), checking real, sane shapes and ranges throughout.

## API

FastAPI app (`app/api/main.py`), fully offline:

- `GET /health`, `GET /kpis`, `GET /highest-risk-host`, `GET /hosts`
- `GET /forecast/{host_id}` — real one-step forecast + K-step rollout +
  explanation for a demo host
- `GET /forecast/{host_id}/branches` — K-step forecast as a branching
  MITRE-mapped attack-path tree (see "Branching K-step forecast" above)
- `POST /sandbox/test` — genuinely validates an uploaded CSV and returns
  `outcome: "failure"` with specific reasons when the input actually is
  malformed (missing columns, non-numeric values, too few windows per host)
- `POST /ingest` — parses an uploaded CSV and runs real inference on it
- `GET /attack-stage-breakdown`, `GET /forecast-log`, `GET /attack-mapping`
- `GET /benchmark`, `GET /calibration`, `GET /lead-time`, `GET /false-alarms`
  — serve the JSON reports produced by `app/train.py`

All KPIs and the "Recent Forecast Log" table are backed by a small SQLite
log (`app/db.py`) seeded from the real training-run predictions and appended
to by every live `/forecast` or `/ingest` call — nothing is a fixed
placeholder.
