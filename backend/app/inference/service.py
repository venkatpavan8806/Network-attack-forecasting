"""Inference service layer shared by the API. Loads real trained artifacts
once at startup and exposes functions that run genuine model inference --
no fixed/dummy outputs regardless of input, per the project's anti-stub rule.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from app.config import (
    FEATURE_COLUMNS, SEQ_LEN, ROLLOUT_K, STAGE_CLASSES, IDX_TO_STAGE,
    SYNTHETIC_CSV, DATA_DIR, BENCHMARK_JSON, CALIBRATION_JSON, LEAD_TIME_JSON,
    FALSE_ALARM_JSON, LSTM_WEIGHTS, BASELINE_WEIGHTS, SCALER_WEIGHTS, RANDOM_SEED,
    STAGE_MEAN_VECTORS_JSON, BRANCH_DEPTH, BRANCH_FACTOR,
)
from app.labeling.state_labeler import derive_state_labels
from app.features.extraction import load_scaler
from app.models.lstm_world_model import (
    load_model, rollout as lstm_rollout, one_step_forecast,
    branching_rollout as lstm_branching_rollout, enumerate_paths,
)
from app.models.baseline_lr import load_baseline
from app.models.attack_mapping import map_stage
from app.explain.attention import explain_prediction
from app.explain.shap_baseline import BaselineShapExplainer
from app.defense.advisor import build_advice
from app.simulation.mitigations import get_mitigation_fn, list_mitigations
from app.simulation.counterfactual import compare_with_and_without, rollout_counterfactual
from app import db

SHAP_BACKGROUND_SIZE = 200  # normal-traffic windows used as SHAP's reference point


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


class InferenceService:
    def __init__(self):
        self.model = None
        self.baseline = None
        self.scaler = None
        self.labeled_df: pd.DataFrame | None = None
        self.stage_mean_vectors: dict | None = None
        self.shap_explainer: BaselineShapExplainer | None = None
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

        self.shap_explainer = self._build_shap_explainer()

        db.init_db()
        db.seed_from_training_log()
        self.ready = True

    def _build_shap_explainer(self) -> BaselineShapExplainer | None:
        """SHAP needs a reference point. We use real NORMAL windows from the
        dataset (scaled exactly like the model input) so every SHAP value
        reads as 'how much did this feature move the score away from normal'."""
        if self.labeled_df is None:
            return None
        normal = self.labeled_df[self.labeled_df["true_stage"] == "benign"]
        if len(normal) == 0:
            return None
        sample = normal.sample(n=min(SHAP_BACKGROUND_SIZE, len(normal)), random_state=RANDOM_SEED)
        background = self.scaler.transform(sample[FEATURE_COLUMNS].values)
        return BaselineShapExplainer(self.baseline, background)

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
        for host_id, host_df in df.groupby("host_id"):
            if len(host_df) < SEQ_LEN:
                errors.append(
                    f"host '{host_id}' has only {len(host_df)} window(s); at least {SEQ_LEN} are required "
                    f"to build one inference sequence"
                )
        return errors

    # -- core inference ---------------------------------------------------
    def _window_sequence(self, host_df: pd.DataFrame, end_pos: int) -> np.ndarray:
        feats = self.scaler.transform(host_df[FEATURE_COLUMNS].values).astype(np.float32)
        return feats[end_pos - SEQ_LEN + 1: end_pos + 1]

    def forecast_host_from_dataframe(self, host_id: str, host_df: pd.DataFrame, log_source: str = "live"):
        """host_df: rows for ONE host, sorted by window_idx ascending, with
        FEATURE_COLUMNS present. Runs one-step forecast + K-step rollout +
        explainability from the LAST available window, and the baseline's
        current-window score. Logs both models' outputs to the inference log."""
        host_df = host_df.sort_values("window_idx").reset_index(drop=True)
        if len(host_df) < SEQ_LEN:
            raise ValueError(f"host '{host_id}' needs at least {SEQ_LEN} windows, has {len(host_df)}")

        end_pos = len(host_df) - 1
        window = self._window_sequence(host_df, end_pos)
        explanation = explain_prediction(self.model, window)
        roll = lstm_rollout(self.model, window, k=ROLLOUT_K)

        cur_feats_scaled = self.scaler.transform(host_df[FEATURE_COLUMNS].values).astype(np.float32)
        baseline_prob = float(self.baseline.predict_proba(cur_feats_scaled[end_pos:end_pos + 1])[0, 1])

        last_row = host_df.iloc[end_pos]
        true_stage = last_row.get("true_stage") if "true_stage" in host_df.columns else None
        state_label = last_row.get("state_label") if "state_label" in host_df.columns else None
        window_idx = int(last_row["window_idx"])

        predicted_stage = max(explanation["stage_probabilities"].items(), key=lambda kv: kv[1])[0]
        stage_mapping = map_stage(predicted_stage)

        db.log_inference(host_id, window_idx, "world_model_lstm", predicted_stage,
                          explanation["infiltration_probability"], true_stage, state_label, source=log_source)
        db.log_inference(host_id, window_idx, "baseline_logreg", None,
                          baseline_prob, true_stage, state_label, source=log_source)

        return {
            "host_id": host_id,
            "window_idx": window_idx,
            "predicted_stage": predicted_stage,
            "attack_mapping": stage_mapping,
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
            "true_stage": true_stage,
            "state_label": state_label,
        }

    def forecast_demo_host(self, host_id: str, at_window_idx: int | None = None):
        if host_id.startswith("live:"):
            if at_window_idx is not None:
                raise ValueError("at_window_idx is not supported for live hosts -- live capture only keeps the most recent 8 windows, there's no history to rewind to")
            return self.forecast_live_host(host_id[len("live:"):])
        if self.labeled_df is None:
            raise ArtifactsNotReadyError("no dataset loaded; run app.train first")
        host_df = self.labeled_df[self.labeled_df["host_id"] == host_id]
        if len(host_df) == 0:
            raise ValueError(f"unknown host_id: {host_id}")
        if at_window_idx is not None:
            host_df = host_df[host_df["window_idx"] <= at_window_idx]
            if len(host_df) < SEQ_LEN:
                raise ValueError(f"host '{host_id}' has fewer than {SEQ_LEN} windows at/before window {at_window_idx}")
        return self.forecast_host_from_dataframe(host_id, host_df, log_source="live")

    # -- live-capture hosts: same inference, sourced from the live capture's
    # own rolling raw-feature history instead of the synthetic demo dataset --
    def _live_history_raw(self, remote_ip: str) -> tuple[np.ndarray, int]:
        """Returns (raw_window_array, window_idx) for a live-captured remote
        host, or raises ValueError with a clear reason if it's not ready."""
        from app.live.capture import live_capture
        if not live_capture.running:
            raise ValueError(f"live capture is not running -- cannot forecast live host '{remote_ip}'")
        hist = live_capture.history.get(remote_ip)
        if hist is None or len(hist) < SEQ_LEN:
            have = 0 if hist is None else len(hist)
            raise ValueError(f"live host '{remote_ip}' has {have}/{SEQ_LEN} windows of history -- not enough yet")
        window_idx = live_capture.window_counter.get(remote_ip, len(hist))
        return np.stack(list(hist)).astype(np.float32), window_idx

    def forecast_live_host(self, remote_ip: str):
        """Same real inference as forecast_host_from_dataframe (attention +
        saliency explanation, K-step rollout, branching forecast), sourced
        from live capture's raw window history instead of the demo dataset.
        Does not re-log to the inference table -- that already happened once
        when the window was first processed by app/live/capture.py; this is
        an on-demand recomputation for the UI, not a new observation."""
        window_raw, window_idx = self._live_history_raw(remote_ip)
        window_scaled = self.scaler.transform(window_raw).astype(np.float32)
        explanation = explain_prediction(self.model, window_scaled)
        roll = lstm_rollout(self.model, window_scaled, k=ROLLOUT_K)
        baseline_prob = float(self.baseline.predict_proba(window_scaled[-1:])[0, 1])
        predicted_stage = max(explanation["stage_probabilities"].items(), key=lambda kv: kv[1])[0]
        stage_mapping = map_stage(predicted_stage)

        return {
            "host_id": f"live:{remote_ip}",
            "window_idx": window_idx,
            "predicted_stage": predicted_stage,
            "attack_mapping": stage_mapping,
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
            "true_stage": None,
            "state_label": None,
        }

    def live_hosts_with_predictions(self) -> list[str]:
        """Live-captured hosts that have accumulated enough windows to be
        forecastable/explainable right now (i.e. eligible for Explainability
        and Digital Twin, not just the Live Capture table's own view)."""
        from app.live.capture import live_capture
        if not live_capture.running:
            return []
        return [f"live:{ip}" for ip, hist in live_capture.history.items() if len(hist) >= SEQ_LEN]

    def branching_forecast(self, host_id: str, at_window_idx: int | None = None,
                            depth: int | None = None, branch_factor: int | None = None):
        """K-step forecast as a branching attack-path TREE (see
        app/models/lstm_world_model.py:branching_rollout) instead of the
        single linear path `forecast_demo_host`'s `rollout` block returns.
        Every node in the tree carries its MITRE ATT&CK mapping, so this is
        the "K-step + branching + MITRE" forecast: not just "infiltration
        probability rises over the next K windows" but "here are the
        distinct plausible attack-technique continuations, each with its
        own probability, ranked". Demo-dataset hosts only -- live hosts only
        keep SEQ_LEN windows of raw history, which the linear rollout's own
        `branching_forecast` field (see _build_branching_forecast above)
        already covers per-horizon."""
        if self.labeled_df is None:
            raise ArtifactsNotReadyError("no dataset loaded; run app.train first")
        host_df = self.labeled_df[self.labeled_df["host_id"] == host_id]
        if len(host_df) == 0:
            raise ValueError(f"unknown host_id: {host_id}")
        if at_window_idx is not None:
            host_df = host_df[host_df["window_idx"] <= at_window_idx]
        host_df = host_df.sort_values("window_idx").reset_index(drop=True)
        if len(host_df) < SEQ_LEN:
            raise ValueError(f"host '{host_id}' needs at least {SEQ_LEN} windows, has {len(host_df)}")

        end_pos = len(host_df) - 1
        window = self._window_sequence(host_df, end_pos)
        kwargs = {}
        if depth is not None:
            kwargs["depth"] = depth
        if branch_factor is not None:
            kwargs["branch_factor"] = branch_factor
        tree = lstm_branching_rollout(self.model, window, stage_mean_vectors=self.stage_mean_vectors, **kwargs)
        paths = enumerate_paths(tree["root"])

        last_row = host_df.iloc[end_pos]
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
            "true_stage": last_row.get("true_stage"),
            "state_label": last_row.get("state_label"),
        }

    def ingest_csv(self, df: pd.DataFrame):
        """Parses an uploaded CSV of synthetic-telemetry-shaped rows and runs
        real inference for the LAST window of every host present in it."""
        errors = self.validate_telemetry_csv(df)
        if errors:
            raise ValueError("; ".join(errors))
        results = []
        for host_id, host_df in df.groupby("host_id"):
            results.append(self.forecast_host_from_dataframe(str(host_id), host_df, log_source="ingest"))
        return results

    # -- digital twin: model-based counterfactual "what if" -------------
    def _counterfactual_result(self, host_id: str, window_idx: int, mitigation_id: str,
                                seed_raw: np.ndarray, true_stage=None, state_label=None):
        mitigation_fn = get_mitigation_fn(mitigation_id)
        baseline_roll, mitigated_roll, metrics = compare_with_and_without(self.model, self.scaler, seed_raw, mitigation_fn, k=ROLLOUT_K)
        mitigations_meta = {m["id"]: m for m in list_mitigations()}

        divergences = []
        for h, (a, b) in enumerate(zip(baseline_roll["predicted_stage"], mitigated_roll["predicted_stage"]), start=1):
            if a != b:
                divergences.append({
                    "horizon": h,
                    "without_mitigation_action": a,
                    "with_mitigation_action": b,
                })

        return {
            "host_id": host_id,
            "window_idx": window_idx,
            "mitigation": mitigations_meta[mitigation_id],
            "horizon_windows": ROLLOUT_K,
            "without_mitigation": {
                "infiltration_probs": [round(p, 4) for p in baseline_roll["infiltration_probs"]],
                "predicted_stage_per_horizon": baseline_roll["predicted_stage"],
            },
            "with_mitigation": {
                "infiltration_probs": [round(p, 4) for p in mitigated_roll["infiltration_probs"]],
                "predicted_stage_per_horizon": mitigated_roll["predicted_stage"],
            },
            "action_divergences": divergences,
            "metrics": metrics,
            "true_stage": true_stage,
            "state_label": state_label,
        }

    def run_counterfactual(self, host_id: str, mitigation_id: str, at_window_idx: int | None = None):
        """Compares the world model's predicted infiltration trajectory with
        and without a named mitigation applied, starting from the host's
        real most-recent observed window (or `at_window_idx`, to replay a
        specific moment such as "right when brute-force was detected"). Not
        a live network simulation -- see app/simulation/counterfactual.py."""
        if host_id.startswith("live:"):
            if at_window_idx is not None:
                raise ValueError("at_window_idx is not supported for live hosts -- live capture only keeps the most recent 8 windows, there's no history to rewind to")
            remote_ip = host_id[len("live:"):]
            seed_raw, window_idx = self._live_history_raw(remote_ip)
            return self._counterfactual_result(host_id, window_idx, mitigation_id, seed_raw)

        if self.labeled_df is None:
            raise ArtifactsNotReadyError("no dataset loaded; run app.train first")
        host_df = self.labeled_df[self.labeled_df["host_id"] == host_id].sort_values("window_idx").reset_index(drop=True)
        if len(host_df) == 0:
            raise ValueError(f"unknown host_id: {host_id}")
        if at_window_idx is not None:
            host_df = host_df[host_df["window_idx"] <= at_window_idx].reset_index(drop=True)
        if len(host_df) < SEQ_LEN:
            raise ValueError(f"host '{host_id}' needs at least {SEQ_LEN} windows, has {len(host_df)}")

        end_pos = len(host_df) - 1
        seed_raw = host_df[FEATURE_COLUMNS].values[end_pos - SEQ_LEN + 1: end_pos + 1].astype(np.float32)
        last_row = host_df.iloc[end_pos]
        return self._counterfactual_result(
            host_id, int(last_row["window_idx"]), mitigation_id, seed_raw,
            true_stage=last_row.get("true_stage"), state_label=last_row.get("state_label"),
        )

    def available_mitigations(self):
        return list_mitigations()

    def list_demo_hosts(self):
        if self.labeled_df is None:
            return []
        return sorted(self.labeled_df["host_id"].unique().tolist())

    # -- shared: raw seed window for a demo host OR a live-capture host --------
    def _raw_seed(self, host_id: str, at_window_idx: int | None = None):
        """Returns (seed_raw, window_idx, true_stage, state_label) where seed_raw is the
        RAW (unscaled) (SEQ_LEN, n_features) window ending at the host's latest window
        (or `at_window_idx`). Works for CSV/demo hosts and for `live:<ip>` hosts,
        whose history is whatever the live packet capture has really accumulated."""
        if host_id.startswith("live:"):
            from app.live.capture import live_capture  # local import: avoids a circular import at load time
            remote_ip = host_id[len("live:"):]
            hist = live_capture.history.get(remote_ip)
            if hist is None:
                raise ValueError(f"unknown host_id: {host_id} (not seen by the live capture)")
            rows = list(hist)
            if len(rows) < SEQ_LEN:
                raise ValueError(f"live host '{host_id}' has {len(rows)}/{SEQ_LEN} windows of history so far")
            return (np.stack(rows).astype(np.float32),
                    int(live_capture.window_counter.get(remote_ip, len(rows))), None, None)

        if self.labeled_df is None:
            raise ArtifactsNotReadyError("no dataset loaded; run app.train first")
        host_df = self.labeled_df[self.labeled_df["host_id"] == host_id].sort_values("window_idx").reset_index(drop=True)
        if len(host_df) == 0:
            raise ValueError(f"unknown host_id: {host_id}")
        if at_window_idx is not None:
            host_df = host_df[host_df["window_idx"] <= at_window_idx].reset_index(drop=True)
        if len(host_df) < SEQ_LEN:
            raise ValueError(f"host '{host_id}' needs at least {SEQ_LEN} windows, has {len(host_df)}")
        end_pos = len(host_df) - 1
        seed_raw = host_df[FEATURE_COLUMNS].values[end_pos - SEQ_LEN + 1: end_pos + 1].astype(np.float32)
        last_row = host_df.iloc[end_pos]
        return seed_raw, int(last_row["window_idx"]), last_row.get("true_stage"), last_row.get("state_label")

    # -- SHAP (baseline) next to attention + saliency (LSTM) --------------------
    def explain_shap(self, host_id: str, at_window_idx: int | None = None, top_k: int = 8):
        """Two explanations of the SAME moment, side by side:
          * SHAP on the logistic-regression baseline (sees only the current window)
          * attention x input-gradient saliency on the LSTM (sees the last SEQ_LEN windows)
        plus how much their top features agree."""
        if self.shap_explainer is None:
            raise ArtifactsNotReadyError("SHAP explainer unavailable: no normal-traffic reference data loaded")
        seed_raw, window_idx, true_stage, state_label = self._raw_seed(host_id, at_window_idx)
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
    def defense_advice(self, host_id: str, at_window_idx: int | None = None):
        seed_raw, window_idx, true_stage, state_label = self._raw_seed(host_id, at_window_idx)
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

    # -- precomputed reports ---------------------------------------------
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


service = InferenceService()
