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
    FALSE_ALARM_JSON, LSTM_WEIGHTS, BASELINE_WEIGHTS, SCALER_WEIGHTS,
)
from app.labeling.state_labeler import derive_state_labels
from app.features.extraction import load_scaler
from app.models.lstm_world_model import load_model, rollout as lstm_rollout, one_step_forecast
from app.models.baseline_lr import load_baseline
from app.models.attack_mapping import map_stage
from app.explain.attention import explain_prediction
from app.simulation.mitigations import get_mitigation_fn, list_mitigations
from app.simulation.counterfactual import compare_with_and_without
from app import db


class ArtifactsNotReadyError(RuntimeError):
    pass


class InferenceService:
    def __init__(self):
        self.model = None
        self.baseline = None
        self.scaler = None
        self.labeled_df: pd.DataFrame | None = None
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

        db.init_db()
        db.seed_from_training_log()
        self.ready = True

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
            },
            "true_stage": true_stage,
            "state_label": state_label,
        }

    def forecast_demo_host(self, host_id: str, at_window_idx: int | None = None):
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
    def run_counterfactual(self, host_id: str, mitigation_id: str, at_window_idx: int | None = None):
        """Compares the world model's predicted infiltration trajectory with
        and without a named mitigation applied, starting from the host's
        real most-recent observed window (or `at_window_idx`, to replay a
        specific moment such as "right when brute-force was detected"). Not
        a live network simulation -- see app/simulation/counterfactual.py."""
        if self.labeled_df is None:
            raise ArtifactsNotReadyError("no dataset loaded; run app.train first")
        host_df = self.labeled_df[self.labeled_df["host_id"] == host_id].sort_values("window_idx").reset_index(drop=True)
        if len(host_df) == 0:
            raise ValueError(f"unknown host_id: {host_id}")
        if at_window_idx is not None:
            host_df = host_df[host_df["window_idx"] <= at_window_idx].reset_index(drop=True)
        if len(host_df) < SEQ_LEN:
            raise ValueError(f"host '{host_id}' needs at least {SEQ_LEN} windows, has {len(host_df)}")

        mitigation_fn = get_mitigation_fn(mitigation_id)
        end_pos = len(host_df) - 1
        seed_raw = host_df[FEATURE_COLUMNS].values[end_pos - SEQ_LEN + 1: end_pos + 1].astype(np.float32)

        baseline_roll, mitigated_roll = compare_with_and_without(self.model, self.scaler, seed_raw, mitigation_fn, k=ROLLOUT_K)

        last_row = host_df.iloc[end_pos]
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
            "window_idx": int(last_row["window_idx"]),
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
            "true_stage": last_row.get("true_stage"),
            "state_label": last_row.get("state_label"),
        }

    def available_mitigations(self):
        return list_mitigations()

    def list_demo_hosts(self):
        if self.labeled_df is None:
            return []
        return sorted(self.labeled_df["host_id"].unique().tolist())

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
