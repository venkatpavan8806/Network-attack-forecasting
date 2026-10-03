"""Tests for the model-versioning/rollback gate in app/train.py.

Only the pure decision function (should_promote) and the file-reading helper
(_load_live_lstm_f1) are tested here -- not a real training run, which would
take minutes. See the module docstring on should_promote for why this gate
exists: a real regression (the CNN+BiLSTM architecture scoring F1 0.970 vs.
the prior LSTM's 0.994) shipped as the live model in this project without
anyone noticing until a benchmark report was read by hand.
"""
import json

import pytest

from app.train import should_promote, _load_live_lstm_f1
import app.train as train_module


def test_promotes_when_nothing_to_compare_against():
    assert should_promote(new_f1=0.5, live_f1=None) is True


def test_promotes_an_improvement():
    assert should_promote(new_f1=0.99, live_f1=0.97, tolerance=0.02) is True


def test_promotes_within_tolerance():
    assert should_promote(new_f1=0.951, live_f1=0.97, tolerance=0.02) is True


def test_holds_back_a_regression_beyond_tolerance():
    """The exact scenario that shipped: 0.994 -> 0.970 is a 0.024 drop,
    bigger than the default 0.02 tolerance."""
    assert should_promote(new_f1=0.970, live_f1=0.994, tolerance=0.02) is False


def test_force_overrides_a_regression():
    assert should_promote(new_f1=0.5, live_f1=0.99, force=True) is True


@pytest.fixture()
def isolated_paths(tmp_path, monkeypatch):
    weights = tmp_path / "lstm_world_model.pt"
    benchmark = tmp_path / "benchmark_report.json"
    monkeypatch.setattr(train_module, "LSTM_WEIGHTS", weights)
    monkeypatch.setattr(train_module, "BENCHMARK_JSON", benchmark)
    return weights, benchmark


def test_load_live_f1_returns_none_on_fresh_checkout(isolated_paths):
    assert _load_live_lstm_f1() is None


def test_load_live_f1_returns_none_if_only_one_file_exists(isolated_paths):
    weights, benchmark = isolated_paths
    weights.write_bytes(b"fake weights")
    # benchmark.json missing -> still None
    assert _load_live_lstm_f1() is None


def test_load_live_f1_reads_the_real_value(isolated_paths):
    weights, benchmark = isolated_paths
    weights.write_bytes(b"fake weights")
    benchmark.write_text(json.dumps({"world_model_lstm": {"f1": 0.8842}}))
    assert _load_live_lstm_f1() == 0.8842


def test_load_live_f1_returns_none_on_malformed_report(isolated_paths):
    weights, benchmark = isolated_paths
    weights.write_bytes(b"fake weights")
    benchmark.write_text("not valid json {{{")
    assert _load_live_lstm_f1() is None
