"""Robustness evaluation: is the benchmark_report.json headline number
trustworthy, or does it swing wildly between runs?

app/train.py trains and evaluates exactly ONCE, on ONE random seed (with 4
held-out attack hosts). That single number is what gets reported everywhere
else in this project. This script answers two separate honesty questions
that a single run cannot:

  1. VARIANCE: train+evaluate N independent times (fresh synthetic world,
     fresh host split, fresh model, each on its own seed) and report
     mean +/- std and a 95% confidence interval per metric, instead of one
     number that might have gotten lucky or unlucky on its particular
     4-host test split.

  2. CROSS-RUN GENERALIZATION (a synthetic stand-in for a time-based split):
     this project's data has no real timestamps to split on -- every host's
     "day" is the same nominal clock. What we CAN test is whether a model
     trained on ONE generated world still works on a COMPLETELY independent
     one it has never seen a single row of (fresh RNG draws for every host's
     behavior) -- the closest honest analogue to "train on earlier days,
     test on later ones" available here. A big gap between a model's score
     on its OWN held-out test hosts vs. on a fresh world means the model
     partly memorized this run's particular synthetic quirks.

Does NOT touch models_store/ or overwrite the live benchmark_report.json --
purely a reporting run. Trains small, throwaway models per seed; nothing
here is ever loaded by the API.

Run with:  python -m app.evaluate_robustness [--seeds N] [--fast]
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from app.config import MALICIOUS_HARD_STAGES, ROBUSTNESS_JSON
from app.data_gen.generator import generate_dataset
from app.labeling.state_labeler import derive_state_labels
from app.features.extraction import host_split, fit_scaler, build_sequences, build_single_window_table
from app.models.baseline_lr import train_baseline
from app.evaluation.metrics import run_full_benchmark
from app.train import train_lstm

DEFAULT_SEEDS = [1, 2, 3, 4, 5]


def _train_and_benchmark(seed: int, epochs: int = 60, dataset_kwargs: dict | None = None):
    """One complete, independent run: fresh synthetic world -> fresh host
    split -> fresh scaler/model -> benchmark on ITS OWN held-out test hosts.
    Returns (report_dict, trained_artifacts) so the caller can optionally
    reuse the trained model against a DIFFERENT seed's data. `dataset_kwargs`
    overrides generate_dataset's host counts/length -- tests use a tiny
    dataset so this stays fast; the CLI uses the real production size."""
    raw = generate_dataset(seed=seed, **(dataset_kwargs or {}))
    labeled = derive_state_labels(raw)
    train_hosts, val_hosts, test_hosts = host_split(labeled, seed=seed)

    scaler = fit_scaler(labeled, train_hosts)
    X_train, y_stage_train, y_next_train, _, _ = build_sequences(labeled, scaler, train_hosts)
    X_val, y_stage_val, y_next_val, _, _ = build_sequences(labeled, scaler, val_hosts)
    X_test, y_stage_test, y_next_test, _, meta_test = build_sequences(labeled, scaler, test_hosts)

    lstm_model, _, _ = train_lstm(X_train, y_stage_train, y_next_train, X_val, y_stage_val, y_next_val,
                                   epochs=epochs)

    X_base_train, y_base_train, _ = build_single_window_table(labeled, scaler, train_hosts)
    X_base_test, y_base_test, _ = build_single_window_table(labeled, scaler, test_hosts)
    baseline_clf = train_baseline(X_base_train, y_base_train)

    labeled_idx = labeled.set_index(["host_id", "window_idx"])
    y_true_hard_at_w = np.array([
        1 if labeled_idx.loc[(m["host_id"], m["window_idx_next"]), "true_stage"] in MALICIOUS_HARD_STAGES else 0
        for m in meta_test
    ])
    report = run_full_benchmark(baseline_clf, lstm_model, X_base_test, y_base_test, X_test, y_true_hard_at_w,
                                 write=False)
    artifacts = {"model": lstm_model, "baseline": baseline_clf, "scaler": scaler,
                 "n_train_hosts": len(train_hosts), "n_val_hosts": len(val_hosts), "n_test_hosts": len(test_hosts)}
    return report, artifacts


def _ci95(values: list[float]) -> tuple[float, float]:
    """Normal-approximation 95% CI (n is small here, but this project makes
    no claim beyond 'a rough sense of spread' -- see the report's own note)."""
    arr = np.asarray(values, dtype=np.float64)
    mean = float(arr.mean())
    se = float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return mean - 1.96 * se, mean + 1.96 * se


def _aggregate(model_key: str, per_seed_reports: list[dict], metric_keys: list[str]) -> dict:
    out = {}
    for key in metric_keys:
        values = [r[model_key][key] for r in per_seed_reports]
        lo, hi = _ci95(values)
        out[key] = {
            "values_by_seed": [round(v, 4) for v in values],
            "mean": round(float(np.mean(values)), 4),
            "std": round(float(np.std(values, ddof=1)), 4) if len(values) > 1 else 0.0,
            "ci95_low": round(lo, 4),
            "ci95_high": round(hi, 4),
        }
    return out


def variance_across_seeds(seeds: list[int], epochs: int = 60, dataset_kwargs: dict | None = None) -> dict:
    metric_keys = ["precision", "recall", "f1", "false_positive_rate"]
    per_seed = []
    for seed in seeds:
        print(f"  seed {seed}: generating + training + evaluating...")
        t0 = time.time()
        report, _ = _train_and_benchmark(seed, epochs=epochs, dataset_kwargs=dataset_kwargs)
        print(f"    world_model F1={report['world_model_lstm']['f1']:.4f}  "
              f"baseline F1={report['baseline_logistic_regression']['f1']:.4f}  ({time.time()-t0:.1f}s)")
        per_seed.append(report)
    return {
        "seeds": seeds,
        "world_model_lstm": _aggregate("world_model_lstm", per_seed, metric_keys),
        "baseline_logistic_regression": _aggregate("baseline_logistic_regression", per_seed, metric_keys),
        "note": (
            "Each seed is a fully independent run: its own synthetic world, its own host "
            "split, its own trained model, evaluated on its own held-out test hosts. "
            "ci95_low/high is a normal-approximation 95% confidence interval over the "
            "per-seed values -- with only a handful of seeds this is a rough spread "
            "indicator, not a rigorous interval; report the per-seed values alongside it."
        ),
    }


def cross_run_generalization(train_seed: int, eval_seeds: list[int], epochs: int = 60,
                              dataset_kwargs: dict | None = None) -> dict:
    print(f"  training once on seed {train_seed}...")
    own_report, artifacts = _train_and_benchmark(train_seed, epochs=epochs, dataset_kwargs=dataset_kwargs)
    model, scaler = artifacts["model"], artifacts["scaler"]

    fresh_results = []
    for seed in eval_seeds:
        print(f"  evaluating that SAME model on a fresh, independently-generated world (seed {seed})...")
        raw = generate_dataset(seed=seed, **(dataset_kwargs or {}))
        labeled = derive_state_labels(raw)
        # every host in this fresh world is unseen -- use ALL of them, not just a held-out slice
        all_hosts = set(labeled["host_id"].unique())
        X, y_stage, y_next, _, meta = build_sequences(labeled, scaler, all_hosts)

        labeled_idx = labeled.set_index(["host_id", "window_idx"])
        y_true_hard_at_w = np.array([
            1 if labeled_idx.loc[(m["host_id"], m["window_idx_next"]), "true_stage"] in MALICIOUS_HARD_STAGES else 0
            for m in meta
        ])
        from app.evaluation.metrics import evaluate_lstm, _binary_metrics
        preds, _ = evaluate_lstm(model, X, None, threshold=0.5)
        metrics = _binary_metrics(y_true_hard_at_w, preds)
        print(f"    fresh-world F1={metrics['f1']:.4f} (own-held-out-test F1 was {own_report['world_model_lstm']['f1']:.4f})")
        fresh_results.append({"seed": seed, **metrics})

    own_f1 = own_report["world_model_lstm"]["f1"]
    fresh_f1s = [r["f1"] for r in fresh_results]
    return {
        "trained_on_seed": train_seed,
        "own_held_out_test_f1": round(own_f1, 4),
        "fresh_world_results": fresh_results,
        "mean_fresh_world_f1": round(float(np.mean(fresh_f1s)), 4),
        "generalization_gap": round(own_f1 - float(np.mean(fresh_f1s)), 4),
        "note": (
            "generalization_gap = own_held_out_test_f1 - mean_fresh_world_f1. A large positive "
            "gap means the model does notably worse on traffic generated by an ENTIRELY "
            "independent run (fresh RNG draws for every host, never seen in any form during "
            "training) than on its own held-out test hosts from the SAME generation run -- "
            "i.e. it partly learned this run's particular synthetic quirks rather than the "
            "underlying attack pattern. This project has no real timestamps to split on, so "
            "this cross-run test stands in for a time-based split (train on 'earlier' "
            "traffic, test on traffic the model has truly never been near)."
        ),
    }


def main(seeds: list[int] | None = None, epochs: int = 60, dataset_kwargs: dict | None = None, write: bool = True):
    t0 = time.time()
    seeds = seeds or DEFAULT_SEEDS
    print(f"== variance across {len(seeds)} independent seeds: {seeds} ==")
    variance = variance_across_seeds(seeds, epochs=epochs, dataset_kwargs=dataset_kwargs)

    print(f"\n== cross-run generalization (train on seed {seeds[0]}, test on the rest) ==")
    cross_run = cross_run_generalization(seeds[0], seeds[1:], epochs=epochs, dataset_kwargs=dataset_kwargs)

    report = {"variance_across_seeds": variance, "cross_run_generalization": cross_run}
    if write:
        with open(ROBUSTNESS_JSON, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nwrote {ROBUSTNESS_JSON}")
    print(f"Total robustness evaluation time: {time.time() - t0:.1f}s")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=len(DEFAULT_SEEDS), help="number of independent seeds to run")
    parser.add_argument("--fast", action="store_true", help="fewer training epochs, for a quick smoke test")
    args = parser.parse_args()
    seed_list = list(range(1, args.seeds + 1))
    main(seeds=seed_list, epochs=15 if args.fast else 60)
