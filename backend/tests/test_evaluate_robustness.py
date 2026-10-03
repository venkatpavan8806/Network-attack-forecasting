"""Fast tests for the multi-seed robustness evaluator. Uses a tiny synthetic
dataset (few hosts, short timelines) and few training epochs so this stays
quick -- the real numbers only come from the full CLI run
(`python -m app.evaluate_robustness`), same relationship test_e2e.py has to
a real `python -m app.train`."""
import numpy as np

from app.evaluate_robustness import (
    _ci95, _aggregate, variance_across_seeds, cross_run_generalization, main,
)

TINY = {"n_benign_hosts": 6, "n_attack_hosts": 3, "benign_len": 30}
FAST_EPOCHS = 3


def test_ci95_widens_with_more_spread():
    tight = _ci95([0.90, 0.91, 0.90, 0.91])
    wide = _ci95([0.5, 0.95, 0.6, 0.99])
    assert (tight[1] - tight[0]) < (wide[1] - wide[0])


def test_ci95_single_value_has_zero_width():
    lo, hi = _ci95([0.8])
    assert lo == hi == 0.8


def test_aggregate_reports_one_entry_per_seed():
    reports = [
        {"world_model_lstm": {"f1": 0.9, "precision": 0.8}},
        {"world_model_lstm": {"f1": 0.7, "precision": 0.6}},
    ]
    agg = _aggregate("world_model_lstm", reports, ["f1", "precision"])
    assert agg["f1"]["values_by_seed"] == [0.9, 0.7]
    assert agg["f1"]["mean"] == 0.8


def test_variance_across_seeds_runs_two_independent_tiny_seeds():
    result = variance_across_seeds([101, 102], epochs=FAST_EPOCHS, dataset_kwargs=TINY)
    assert result["seeds"] == [101, 102]
    for model_key in ["world_model_lstm", "baseline_logistic_regression"]:
        assert len(result[model_key]["f1"]["values_by_seed"]) == 2
        assert 0.0 <= result[model_key]["f1"]["mean"] <= 1.0


def test_cross_run_generalization_reports_a_real_gap():
    result = cross_run_generalization(201, [202], epochs=FAST_EPOCHS, dataset_kwargs=TINY)
    assert result["trained_on_seed"] == 201
    assert len(result["fresh_world_results"]) == 1
    assert 0.0 <= result["own_held_out_test_f1"] <= 1.0
    assert 0.0 <= result["mean_fresh_world_f1"] <= 1.0
    # gap is just own - fresh, can legitimately be positive, negative or zero on tiny data --
    # just check it's a real, finite, consistent computation
    assert np.isfinite(result["generalization_gap"])
    # gap is computed from full-precision F1s then rounded once; re-deriving it from the
    # already-rounded display values can differ by a rounding ulp, so allow a small tolerance
    expected = result["own_held_out_test_f1"] - result["mean_fresh_world_f1"]
    assert abs(expected - result["generalization_gap"]) < 1e-3


def test_main_does_not_write_when_write_is_false(tmp_path, monkeypatch):
    import app.evaluate_robustness as mod
    fake_path = tmp_path / "robustness_report.json"
    monkeypatch.setattr(mod, "ROBUSTNESS_JSON", fake_path)
    report = main(seeds=[301, 302], epochs=FAST_EPOCHS, dataset_kwargs=TINY, write=False)
    assert not fake_path.exists()
    assert "variance_across_seeds" in report
    assert "cross_run_generalization" in report
