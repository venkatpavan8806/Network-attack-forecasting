"""Inference service layer shared by the API. Loads real trained artifacts
once at startup and exposes functions that run genuine model inference --
no fixed/dummy outputs regardless of input, per the project's anti-stub rule.

Multi-user: the trained models are shared, the data is not. Every host-level
method takes the calling user's id and reads only that user's windows from
the database (app/db.py).
"""
from __future__ import annotations

import json
import secrets
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from app.config import (
    FEATURE_COLUMNS, SEQ_LEN, ROLLOUT_K, STAGE_CLASSES, IDX_TO_STAGE,
    SYNTHETIC_CSV, DATA_DIR, BENCHMARK_JSON, CALIBRATION_JSON, LEAD_TIME_JSON,
    FALSE_ALARM_JSON, LSTM_WEIGHTS, BASELINE_WEIGHTS, SCALER_WEIGHTS, RANDOM_SEED, THRESHOLD_CALIBRATION_JSON,
    ROBUSTNESS_JSON, STAGE_MEAN_VECTORS_JSON, WINDOW_SECONDS,
)
from app.labeling.state_labeler import derive_state_labels
from app.features.extraction import load_scaler, validate_feature_vector
from app.models.lstm_world_model import (
    load_model, rollout as lstm_rollout,
    branching_rollout as lstm_branching_rollout, enumerate_paths,
)
from app.models.baseline_lr import load_baseline
from app.models.attack_mapping import map_stage
from app.explain.attention import explain_prediction
from app.defense.advisor import build_advice
from app.simulation.mitigations import get_mitigation_fn, list_mitigations
from app.simulation.counterfactual import rollout_counterfactual
from app.simulation.sandbox import DigitalTwinSandbox
from app import db
from app.models.ngram_move_model import NGramMoveModel, NGRAM_MODEL_JSON, labels_to_moves
from app.tracking.step_tracker import track_host, batch_next_step_probs, padded_window

SHAP_BACKGROUND_SIZE = 200  # normal-traffic windows used as SHAP's reference point
MAX_TRACK_WINDOWS = 2000     # step tracker replays at most this many most-recent windows
MAX_ALERTS_PER_UPLOAD = 500  # tripwire alerts stored per pcap upload
BENIGN_IDX = STAGE_CLASSES.index("benign")


class ArtifactsNotReadyError(RuntimeError):
    pass


def _build_branching_forecast(stage_probs_per_horizon: list[list[float]], top_k: int = 3) -> list[dict]:
    """Turns the model's raw per-horizon class distributions (already
    computed inside rollout(), just never exposed before) into a compact
    top-K structure with real MITRE mapping attached to each candidate --
    the "branching" next-action forecast: not just the single most-likely
    action at each future step, but the other plausible ones too."""
    out = []
    for h, probs in enumerate(stage_probs_per_horizon, start=1):
        ranked = sorted(zip(STAGE_CLASSES, probs), key=lambda kv: kv[1], reverse=True)[:top_k]
        candidates = []
        for action, p in ranked:
            m = map_stage(action)
            candidates.append({
                "action": action,
                "probability": round(float(p), 4),
                "technique_id": m["technique_id"],
                "technique_name": m["technique_name"],
                "tactic": m["tactic"],
            })
        out.append({"horizon": h, "candidates": candidates})
    return out


def _clean(v):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else v


class InferenceService:
    """Model inference over ONE USER'S traffic windows.

    `labeled_df` is the training dataset baked into the deployment by
    `python -m app.train`; it is used only as model reference data (SHAP's
    normal-traffic background, the n-gram fallback), never shown to users as
    if it were their traffic.
    """

    def __init__(self):
        self.model = None
        self.baseline = None
        self.scaler = None
        self.labeled_df: pd.DataFrame | None = None
        self.stage_mean_vectors: dict | None = None
        self.shap_explainer = None  # built lazily: importing shap costs ~190 MB of RAM
        self.ngram: NGramMoveModel | None = None
        self.ready = False

    def load(self):
        missing = [p for p in [LSTM_WEIGHTS, BASELINE_WEIGHTS, SCALER_WEIGHTS] if not p.exists()]
        if missing:
            raise ArtifactsNotReadyError(
                f"model artifacts missing: {[str(p) for p in missing]}. Run `python -m app.train` first."
            )
        self.model = load_model()
        self.baseline = load_baseline()
        self.scaler = load_scaler()

        labeled_path = DATA_DIR / "labeled_states.csv"
        if labeled_path.exists():
            self.labeled_df = pd.read_csv(labeled_path)
        elif SYNTHETIC_CSV.exists():
            raw = pd.read_csv(SYNTHETIC_CSV)
            self.labeled_df = derive_state_labels(raw)
        else:
            self.labeled_df = None

        if STAGE_MEAN_VECTORS_JSON.exists():
            with open(STAGE_MEAN_VECTORS_JSON) as f:
                self.stage_mean_vectors = json.load(f)
        else:
            self.stage_mean_vectors = None

        self.ngram = self._load_ngram()

        db.init_db()
        self.ready = True

    def _build_shap_explainer(self):
        """SHAP needs a reference point. We use real NORMAL windows from the
        training dataset (scaled exactly like the model input) so every SHAP
        value reads as 'how much did this feature move the score away from normal'."""
        if self.labeled_df is None:
            return None
        normal = self.labeled_df[self.labeled_df["true_stage"] == "benign"]
        if len(normal) == 0:
            return None
        sample = normal.sample(n=min(SHAP_BACKGROUND_SIZE, len(normal)), random_state=RANDOM_SEED)
        background = self.scaler.transform(sample[FEATURE_COLUMNS].values)
        from app.explain.shap_baseline import BaselineShapExplainer
        return BaselineShapExplainer(self.baseline, background)

    def _load_ngram(self) -> NGramMoveModel | None:
        """Next-1/2/3-move model. Written by app.evaluate_step_tracking; if it
        is missing, fit it on the training dataset's attack hosts so the step
        tracker still works (it is a cheap counting model)."""
        if NGRAM_MODEL_JSON.exists():
            return NGramMoveModel.load()
        if self.labeled_df is None:
            return None
        seqs = [labels_to_moves(g.sort_values("window_idx")["state_label"], add_end=True)
                for _, g in self.labeled_df.groupby("host_id")]
        seqs = [s for s in seqs if s]
        return NGramMoveModel(order=3).fit(seqs) if seqs else None

    # -- validation -----------------------------------------------------
    def validate_telemetry_csv(self, df: pd.DataFrame) -> list[str]:
        errors = []
        required = {"host_id", "window_idx"} | set(FEATURE_COLUMNS)
        missing_cols = required - set(df.columns)
        if missing_cols:
            errors.append(f"missing required columns: {sorted(missing_cols)}")
            return errors  # can't check further without the columns
        if len(df) == 0:
            errors.append("uploaded file has zero rows")
            return errors
        for col in FEATURE_COLUMNS:
            non_numeric = pd.to_numeric(df[col], errors="coerce").isna() & df[col].notna()
            if non_numeric.any():
                errors.append(f"column '{col}' contains non-numeric values in {int(non_numeric.sum())} row(s)")
        if df[FEATURE_COLUMNS].isna().all(axis=None):
            errors.append("all feature values are null")
        elif df[FEATURE_COLUMNS].isna().any(axis=None):
            errors.append("some feature values are empty -- every window needs all feature columns")
        return errors

    # -- per-user host data -------------------------------------------------
    def _frame(self, user_id: str, host_id: str, at_window_idx: int | None = None,
               last_n: int | None = None) -> pd.DataFrame:
        df = db.host_frame(user_id, host_id, upto_window_idx=at_window_idx, last_n=last_n)
        if len(df) == 0:
            if at_window_idx is not None and db.last_window_idx(user_id, host_id) is not None:
                raise ValueError(f"host '{host_id}' has no windows at/before window {at_window_idx}")
            raise ValueError(f"unknown host_id: {host_id}")
        return df

    @staticmethod
    def _padded_raw(frame: pd.DataFrame) -> tuple[np.ndarray, int]:
        """(SEQ_LEN, n_features) RAW window ending at the frame's last row,
        left-padded with its first row when fewer than SEQ_LEN windows exist
        (same warm-up rule as app/tracking/step_tracker.py)."""
        raw = frame[FEATURE_COLUMNS].values.astype(np.float32)
        return padded_window(raw, len(raw) - 1)

    def _seed(self, user_id: str, host_id: str, at_window_idx: int | None = None):
        frame = self._frame(user_id, host_id, at_window_idx, last_n=SEQ_LEN)
        seed_raw, n_real = self._padded_raw(frame)
        return seed_raw, n_real, frame.iloc[-1]

    def list_hosts(self, user_id: str) -> list[str]:
        return [h["host_id"] for h in db.list_hosts(user_id)]

    def live_hosts_with_predictions(self, user_id: str) -> list[str]:
        """Agent-captured (live) hosts -- forecastable from their first window."""
        return [h for h in self.list_hosts(user_id) if h.startswith("live:")]

    # -- core inference ---------------------------------------------------
    def _forecast_from_seed(self, host_id: str, seed_raw: np.ndarray, n_real: int, last_row) -> dict:
        window = self.scaler.transform(seed_raw).astype(np.float32)
        explanation = explain_prediction(self.model, window)
        roll = lstm_rollout(self.model, window, k=ROLLOUT_K)
        baseline_prob = float(self.baseline.predict_proba(window[-1:])[0, 1])
        predicted_stage = max(explanation["stage_probabilities"].items(), key=lambda kv: kv[1])[0]
        return {
            "host_id": host_id,
            "window_idx": int(last_row["window_idx"]),
            "predicted_stage": predicted_stage,
            "attack_mapping": map_stage(predicted_stage),
            "infiltration_probability_world_model": explanation["infiltration_probability"],
            "infiltration_probability_baseline": round(baseline_prob, 4),
            "stage_probabilities": explanation["stage_probabilities"],
            "explanation": {
                "attention_over_past_windows": explanation["attention_over_past_windows"],
                "top_contributors": explanation["top_contributors"],
            },
            "rollout": {
                "horizon_windows": ROLLOUT_K,
                "infiltration_probs_world_model": [round(p, 4) for p in roll["infiltration_probs"]],
                "predicted_stage_per_horizon": roll["predicted_stage"],
                "branching_forecast": _build_branching_forecast(roll["stage_probs"]),
            },
            "true_stage": _clean(last_row.get("true_stage")),
            "state_label": _clean(last_row.get("state_label")),
            "warmup": bool(n_real < SEQ_LEN),
            "history_windows_used": int(n_real),
            "source": _clean(last_row.get("source")),
        }

    def forecast_host_from_dataframe(self, host_id: str, host_df: pd.DataFrame, user_id: str | None = None,
                                     log_source: str = "csv"):
        """One-step forecast + K-step rollout + explanation from the LAST row
        of `host_df` (rows for ONE host with FEATURE_COLUMNS). Logs both
        models' outputs to `user_id`'s inference log when a user is given."""
        host_df = host_df.sort_values("window_idx").reset_index(drop=True)
        if len(host_df) == 0:
            raise ValueError(f"host '{host_id}' has no windows")
        seed_raw, n_real = self._padded_raw(host_df.tail(SEQ_LEN))
        result = self._forecast_from_seed(host_id, seed_raw, n_real, host_df.iloc[-1])
        if user_id is not None:
            self._log(user_id, result, log_source)
        return result

    def _log(self, user_id: str, result: dict, source: str):
        db.log_inference(user_id, result["host_id"], result["window_idx"], "world_model_lstm",
                         result["predicted_stage"], result["infiltration_probability_world_model"],
                         result.get("true_stage"), result.get("state_label"), source=source)
        db.log_inference(user_id, result["host_id"], result["window_idx"], "baseline_logreg", None,
                         result["infiltration_probability_baseline"], result.get("true_stage"),
                         result.get("state_label"), source=source)

    def forecast_host(self, user_id: str, host_id: str, at_window_idx: int | None = None, log: bool = True):
        seed_raw, n_real, last_row = self._seed(user_id, host_id, at_window_idx)
        result = self._forecast_from_seed(host_id, seed_raw, n_real, last_row)
        if log:
            self._log(user_id, result, result["source"] or "csv")
        return result

    def branching_forecast(self, user_id: str, host_id: str, at_window_idx: int | None = None,
                           depth: int | None = None, branch_factor: int | None = None):
        """K-step forecast as a branching attack-path TREE (see
        app/models/lstm_world_model.py:branching_rollout), every node
        MITRE ATT&CK-mapped: the distinct plausible attack-technique
        continuations, each with its own probability, ranked."""
        seed_raw, n_real, last_row = self._seed(user_id, host_id, at_window_idx)
        window = self.scaler.transform(seed_raw).astype(np.float32)
        kwargs = {}
        if depth is not None:
            kwargs["depth"] = depth
        if branch_factor is not None:
            kwargs["branch_factor"] = branch_factor
        tree = lstm_branching_rollout(self.model, window, stage_mean_vectors=self.stage_mean_vectors, **kwargs)
        paths = enumerate_paths(tree["root"])
        most_likely = paths[0] if paths else None
        highest_risk = max(paths, key=lambda p: p["final_infiltration_probability"]) if paths else None
        return {
            "host_id": host_id,
            "window_idx": int(last_row["window_idx"]),
            "depth": tree["depth"],
            "branch_factor": tree["branch_factor"],
            "tree": tree["root"],
            "paths": paths,
            "most_likely_path": most_likely,
            "highest_risk_path": highest_risk,
            "true_stage": _clean(last_row.get("true_stage")),
            "state_label": _clean(last_row.get("state_label")),
            "warmup": bool(n_real < SEQ_LEN),
        }

    # -- ingestion: every path (agent, pcap, csv, sample) ends here ----------
    def ingest_windows(self, user_id: str, host_id: str, rows: list[dict], source: str,
                       log_all: bool = False, explain_last: bool = False) -> dict:
        """Appends `rows` (time-ordered dicts with `features` and optional
        observed_at / true_stage / state_label) after the host's existing
        windows, predicts the NEXT window after every new one (the same
        prediction the step tracker makes), and stores windows + predictions.
        Logs both models for every new window (`log_all`, live agents) or
        only the last one (bulk uploads)."""
        if not rows:
            return {"host_id": host_id, "windows_added": 0}
        prev = db.host_frame(user_id, host_id, last_n=SEQ_LEN - 1)
        start_idx = int(prev["window_idx"].max()) + 1 if len(prev) else 0
        raw_new = np.array([[float(r["features"][c]) for c in FEATURE_COLUMNS] for r in rows], dtype=np.float32)
        raw_all = np.concatenate([prev[FEATURE_COLUMNS].values.astype(np.float32), raw_new]) if len(prev) else raw_new
        scaled_all = self.scaler.transform(raw_all).astype(np.float32)
        probs_all, n_real_all = batch_next_step_probs(self.model, scaled_all)
        offset = len(prev)
        baseline_new = self.baseline.predict_proba(scaled_all[offset:])[:, 1]

        records = []
        for i, r in enumerate(rows):
            probs = probs_all[offset + i]
            stage = IDX_TO_STAGE[int(np.argmax(probs))]
            n_real = int(n_real_all[offset + i])
            prediction = {
                "predicted_stage": stage,
                "infiltration_probability_world_model": round(float(1.0 - probs[BENIGN_IDX]), 4),
                "infiltration_probability_baseline": round(float(baseline_new[i]), 4),
                "stage_probabilities": {s: round(float(p), 4) for s, p in zip(STAGE_CLASSES, probs)},
                "warmup": n_real < SEQ_LEN,
                "history_windows_used": n_real,
                "attack_mapping": map_stage(stage),
            }
            records.append({
                "host_id": host_id, "window_idx": start_idx + i, "observed_at": r.get("observed_at"),
                "source": source, "features": r["features"],
                "true_stage": r.get("true_stage"), "state_label": r.get("state_label"),
                "prediction": prediction,
            })
        if explain_last:
            window, _ = padded_window(scaled_all, len(scaled_all) - 1)
            expl = explain_prediction(self.model, window)
            records[-1]["prediction"]["explanation"] = {
                "attention_over_past_windows": expl["attention_over_past_windows"],
                "top_contributors": expl["top_contributors"],
            }
        db.insert_windows(user_id, records)

        for rec in (records if log_all else records[-1:]):
            p = rec["prediction"]
            db.log_inference(user_id, host_id, rec["window_idx"], "world_model_lstm", p["predicted_stage"],
                             p["infiltration_probability_world_model"], rec["true_stage"], rec["state_label"], source)
            db.log_inference(user_id, host_id, rec["window_idx"], "baseline_logreg", None,
                             p["infiltration_probability_baseline"], rec["true_stage"], rec["state_label"], source)
        last = records[-1]
        return {"host_id": host_id, "windows_added": len(records), "first_window_idx": start_idx,
                "last_window_idx": last["window_idx"], **{k: last["prediction"][k] for k in (
                    "predicted_stage", "infiltration_probability_world_model", "infiltration_probability_baseline")}}

    def ingest_csv(self, user_id: str, df: pd.DataFrame):
        """Stores every window of an uploaded telemetry CSV in the user's
        workspace (appended after any existing windows of the same host) and
        returns a full forecast for the last window of each host."""
        errors = self.validate_telemetry_csv(df)
        if errors:
            raise ValueError("; ".join(errors))
        has_true, has_state = "true_stage" in df.columns, "state_label" in df.columns
        results = []
        for host_id, host_df in df.groupby("host_id", sort=False):
            host_df = host_df.sort_values("window_idx")
            rows = [{
                "features": {c: float(r[c]) for c in FEATURE_COLUMNS},
                "true_stage": r["true_stage"] if has_true and pd.notna(r["true_stage"]) else None,
                "state_label": r["state_label"] if has_state and pd.notna(r["state_label"]) else None,
            } for _, r in host_df.iterrows()]
            self.ingest_windows(user_id, str(host_id), rows, source="csv")
            results.append(self.forecast_host(user_id, str(host_id), log=False))
        return results

    def ingest_pcap(self, user_id: str, data: bytes, local_ip: str | None = None) -> dict:
        """Replays a pcap/pcapng through the live agent's feature + tripwire
        code (app/ingest/pcap.py) and stores the result as hosts
        `pcap:<remote ip>`. Re-uploading replaces those hosts."""
        from app.ingest.pcap import replay_pcap
        rep = replay_pcap(data, local_ip=local_ip)
        by_remote: dict[str, list] = {}
        for w in rep.windows:
            by_remote.setdefault(w["remote_ip"], []).append(w)
        host_ids = {ip: f"pcap:{ip}" for ip in by_remote}
        db.delete_hosts(user_id, list(host_ids.values()))
        hosts = []
        for ip, ws in by_remote.items():
            rows = [{"features": w["features"],
                     "observed_at": datetime.fromtimestamp(w["window_start"], tz=timezone.utc)} for w in ws]
            hosts.append(self.ingest_windows(user_id, host_ids[ip], rows, source="pcap"))
        db.insert_packets(user_id, [{**p, "host_id": host_ids[p["remote_ip"]]}
                                    for p in rep.packets if p["remote_ip"] in host_ids])
        for a in rep.alerts[:MAX_ALERTS_PER_UPLOAD]:
            db.log_tripwire_alert(user_id, a["remote_ip"], a["message"], a["severity"], a["detail"],
                                  created_at=a["timestamp"], source="pcap")
        hosts.sort(key=lambda h: -(h.get("infiltration_probability_world_model") or 0))
        return {"local_ip": rep.local_ip, "stats": rep.stats, "hosts": hosts}

    def ingest_agent_batch(self, sensor: dict, windows: list[dict], packets: list[dict]) -> list[dict]:
        """One upload from a capture agent: per-remote-host feature windows
        (each validated against FEATURE_COLUMNS -- NaN/Inf/missing keys are
        rejected) + recent raw packets. Each window is predicted and logged
        immediately (live behaviour)."""
        user_id = sensor["user_id"]
        results = []
        for w in windows:
            feats = validate_feature_vector(w["features"])
            row = {"features": dict(zip(FEATURE_COLUMNS, feats.tolist())), "observed_at": w.get("observed_at")}
            results.append(self.ingest_windows(user_id, f"live:{w['remote_ip']}", [row], source="agent",
                                               log_all=True, explain_last=True))
        if packets:
            db.insert_packets(user_id, [{**p, "host_id": f"live:{p['remote_ip']}"} for p in packets])
        return results

    def generate_sample_data(self, user_id: str, n_attack: int = 2, n_benign: int = 2) -> dict:
        """Fresh synthetic traffic for THIS user, generated now from a random
        seed (different every time -- not a stored demo dataset) by the same
        simulator the model was trained on, labeled by the state-labeling
        engine, and ingested like any upload."""
        from app.data_gen.generator import generate_host_timeline
        seed = secrets.randbits(32)
        rng = np.random.default_rng(seed)
        tag = secrets.token_hex(2)
        frames = [generate_host_timeline(f"sample-{tag}-attack-{i + 1}", rng, is_attack=True)
                  for i in range(n_attack)]
        frames += [generate_host_timeline(f"sample-{tag}-benign-{i + 1}", rng, is_attack=False, benign_len=80)
                   for i in range(n_benign)]
        labeled = derive_state_labels(pd.concat(frames, ignore_index=True))
        span = WINDOW_SECONDS * int(labeled["window_idx"].max() + 1)
        start = datetime.now(timezone.utc) - timedelta(seconds=span)
        hosts = []
        for host_id, g in labeled.groupby("host_id", sort=False):
            g = g.sort_values("window_idx")
            rows = [{"features": {c: float(r[c]) for c in FEATURE_COLUMNS},
                     "observed_at": start + timedelta(seconds=WINDOW_SECONDS * int(r["window_idx"])),
                     "true_stage": r["true_stage"], "state_label": r["state_label"]} for _, r in g.iterrows()]
            hosts.append(self.ingest_windows(user_id, str(host_id), rows, source="sample"))
        return {"seed": seed, "hosts": hosts}

    # -- digital twin: model-based counterfactual + network state sandbox -------------
    def _counterfactual_result(self, host_id: str, window_idx: int, mitigation_id: str,
                               seed_raw: np.ndarray, true_stage=None, state_label=None, is_live=False):
        sandbox = DigitalTwinSandbox(self.model, self.scaler)
        return sandbox.run_sandbox_simulation(
            host_id=host_id,
            mitigation_id=mitigation_id,
            seed_raw=seed_raw,
            at_window_idx=window_idx,
            true_stage=true_stage,
            state_label=state_label,
            is_live=is_live,
        )

    def run_counterfactual(self, user_id: str, host_id: str, mitigation_id: str, at_window_idx: int | None = None):
        """Compares the world model's predicted infiltration trajectory with
        and without a named mitigation applied to the cloned Digital Twin network
        state, starting from the host's real most-recent observed window (or
        `at_window_idx`). Safe sandbox simulation -- real_network_touched: false."""
        seed_raw, _, last_row = self._seed(user_id, host_id, at_window_idx)
        return self._counterfactual_result(
            host_id, int(last_row["window_idx"]), mitigation_id, seed_raw,
            true_stage=_clean(last_row.get("true_stage")), state_label=_clean(last_row.get("state_label")),
            is_live=host_id.startswith(("live:", "pcap:")),
        )

    def available_mitigations(self):
        return list_mitigations()

    def _raw_seed(self, user_id: str, host_id: str, at_window_idx: int | None = None):
        """(seed_raw, window_idx, true_stage, state_label) -- RAW (unscaled)
        (SEQ_LEN, n_features) window ending at the host's latest window (or
        `at_window_idx`), padded during warm-up."""
        seed_raw, _, last_row = self._seed(user_id, host_id, at_window_idx)
        return (seed_raw, int(last_row["window_idx"]), _clean(last_row.get("true_stage")),
                _clean(last_row.get("state_label")))

    # -- SHAP (baseline) next to attention + saliency (LSTM) --------------------
    def explain_shap(self, user_id: str, host_id: str, at_window_idx: int | None = None, top_k: int = 8):
        """Two explanations of the SAME moment, side by side:
          * SHAP on the logistic-regression baseline (sees only the current window)
          * attention x input-gradient saliency on the LSTM (sees the last SEQ_LEN windows)
        plus how much their top features agree."""
        if self.shap_explainer is None:
            self.shap_explainer = self._build_shap_explainer()
        if self.shap_explainer is None:
            raise ArtifactsNotReadyError("SHAP explainer unavailable: no normal-traffic reference data loaded")
        seed_raw, window_idx, true_stage, state_label = self._raw_seed(user_id, host_id, at_window_idx)
        seed_scaled = self.scaler.transform(seed_raw).astype(np.float32)

        shap_out = self.shap_explainer.explain(seed_scaled[-1], seed_raw[-1], top_k=top_k)
        model_prob = float(self.baseline.predict_proba(seed_scaled[-1:])[0, 1])
        lstm = explain_prediction(self.model, seed_scaled, top_k=top_k)

        shap_top = [c["feature"] for c in shap_out["contributions"][:5]]
        lstm_top = list(dict.fromkeys(c["feature"] for c in lstm["top_contributors"]))[:5]
        shared = sorted(set(shap_top) & set(lstm_top))
        union = set(shap_top) | set(lstm_top)

        return {
            "host_id": host_id,
            "window_idx": window_idx,
            "true_stage": true_stage,
            "state_label": state_label,
            "shap": {
                **shap_out,
                "model_probability": round(model_prob, 4),  # baseline.predict_proba -- must match baseline_probability
            },
            "lstm": {
                "infiltration_probability": lstm["infiltration_probability"],
                "attention_over_past_windows": lstm["attention_over_past_windows"],
                "top_contributors": lstm["top_contributors"],
            },
            "agreement": {
                "shap_top_features": shap_top,
                "lstm_top_features": lstm_top,
                "shared_features": shared,
                "jaccard": round(len(shared) / len(union), 3) if union else 0.0,
            },
        }

    # -- defense: rank every mitigation by what the world model says it achieves ---
    def defense_advice(self, user_id: str, host_id: str, at_window_idx: int | None = None):
        seed_raw, window_idx, true_stage, state_label = self._raw_seed(user_id, host_id, at_window_idx)
        seed_scaled = self.scaler.transform(seed_raw).astype(np.float32)

        unmitigated = lstm_rollout(self.model, seed_scaled, k=ROLLOUT_K)
        meta = {m["id"]: m for m in list_mitigations()}
        mitigated = {
            mid: rollout_counterfactual(self.model, self.scaler, seed_raw, get_mitigation_fn(mid), k=ROLLOUT_K)
            for mid in meta if mid != "no_mitigation"
        }

        advice = build_advice(host_id, window_idx, unmitigated, mitigated, meta)
        advice["true_stage"] = true_stage
        advice["state_label"] = state_label
        advice["trajectory_without"] = [round(p, 4) for p in unmitigated["infiltration_probs"]]
        rec = advice["recommended"]
        advice["trajectory_with_recommended"] = (
            [round(p, 4) for p in mitigated[rec["id"]]["infiltration_probs"]] if rec else None
        )
        return advice

    # -- step-by-step attacker tracking (prediction after EVERY window) ----
    def track_attacker(self, user_id: str, host_id: str, at_window_idx: int | None = None):
        """Walks the host's timeline from its FIRST window and predicts the
        attacker's next step after each one -- no SEQ_LEN warm-up gap (see
        app/tracking/step_tracker.py). Ground-truth columns are used for
        scoring only when the host has them (sample data / labeled CSV)."""
        frame = self._frame(user_id, host_id, at_window_idx).tail(MAX_TRACK_WINDOWS)
        feats = self.scaler.transform(frame[FEATURE_COLUMNS].values).astype(np.float32)
        labels = frame["state_label"].tolist() if frame["state_label"].notna().all() else None
        result = track_host(self.model, feats, frame["window_idx"].astype(int).tolist(), labels=labels, ngram=self.ngram)
        result["host_id"] = host_id
        return result

    def step_tracking_report(self):
        from app.evaluate_step_tracking import STEP_TRACKING_JSON
        return self.load_report(STEP_TRACKING_JSON)

    # -- precomputed model reports (global: about the model, not a user) -----
    def load_report(self, path):
        if not path.exists():
            return None
        with open(path) as f:
            return json.load(f)

    def benchmark_report(self):
        return self.load_report(BENCHMARK_JSON)

    def calibration_report(self):
        return self.load_report(CALIBRATION_JSON)

    def lead_time_report(self):
        return self.load_report(LEAD_TIME_JSON)

    def false_alarm_examples(self):
        return self.load_report(FALSE_ALARM_JSON) or []

    def threshold_calibration_report(self):
        return self.load_report(THRESHOLD_CALIBRATION_JSON)

    def robustness_report(self):
        return self.load_report(ROBUSTNESS_JSON)


service = InferenceService()
