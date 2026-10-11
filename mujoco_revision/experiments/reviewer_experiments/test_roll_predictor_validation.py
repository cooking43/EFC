"""Numerical and causal contract tests for roll predictor validation."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from roll_predictor_validation import (  # noqa: E402
    causal_history_prediction, finite_difference_predictions,
)


def test_second_order_is_exact_for_constant_acceleration_sequence():
    dt = 0.02
    acceleration = 0.7
    times = np.asarray([-2 * dt, -dt, 0.0])
    values = 0.3 + 0.2 * times + 0.5 * acceleration * times**2
    result = finite_difference_predictions(values, 0.1, dt)
    expected = 0.3 + 0.2 * 0.1 + 0.5 * acceleration * 0.1**2
    assert np.isclose(result["Second-Order"], expected)
    assert not np.isclose(result["First-Order"], expected)


def test_horizon_and_order_change_actual_outputs():
    values = np.asarray([0.0, 0.01, 0.03])
    short = finite_difference_predictions(values, 0.05)
    long = finite_difference_predictions(values, 0.20)
    assert short["Persistence"] == long["Persistence"]
    assert short["First-Order"] != long["First-Order"]
    assert long["Second-Order"] != long["First-Order"]


def test_causal_history_does_not_read_future_samples():
    history = np.sin(np.linspace(0.0, 0.5, 10))
    first = causal_history_prediction(history, 0.1)
    unrelated_future = np.asarray([999.0, -999.0])
    second = causal_history_prediction(np.concatenate((history, unrelated_future))[:-2], 0.1)
    assert first == second
