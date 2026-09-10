import json

import numpy as np
import pytest

from stoa.io import load_spec
from stoa.models import PortfolioSpec, Regime, StoaValidationError
from stoa.optimizer import RobustAllocator


def make_spec(**overrides):
    values = {
        "assets": ("A", "B"),
        "regimes": (
            Regime(
                name="base",
                probability=0.7,
                expected_returns=np.array([0.08, 0.03]),
                covariance=np.array([[0.04, -0.002], [-0.002, 0.01]]),
            ),
            Regime(
                name="stress",
                probability=0.3,
                expected_returns=np.array([-0.1, 0.06]),
                covariance=np.array([[0.09, -0.004], [-0.004, 0.016]]),
            ),
        ),
        "current_weights": np.array([0.5, 0.5]),
        "minimum_weights": np.array([0.0, 0.0]),
        "maximum_weights": np.array([0.8, 0.8]),
        "iterations": 1000,
    }
    values.update(overrides)
    return PortfolioSpec(**values)


def test_regime_probabilities_must_sum_to_one():
    with pytest.raises(StoaValidationError):
        PortfolioSpec(
            assets=("A",),
            regimes=(Regime("base", 0.8, np.array([0.05]), np.array([[0.01]])),),
        )


def test_optimizer_respects_budget_and_bounds():
    result = RobustAllocator(make_spec()).optimize()
    assert np.isclose(result.weights.sum(), 1.0, atol=1e-8)
    assert (result.weights >= -1e-8).all()
    assert (result.weights <= 0.8 + 1e-8).all()
    assert result.volatility >= 0


def test_regime_dispersion_increases_total_variance():
    spec = make_spec(risk_aversion=0.0, uncertainty_radius=0.0, turnover_penalty=0.0)
    allocator = RobustAllocator(spec)
    weights = np.array([0.5, 0.5])
    state = allocator._state(weights)
    within = (
        0.7 * weights @ spec.regimes[0].covariance @ weights
        + 0.3 * weights @ spec.regimes[1].covariance @ weights
    )
    assert state.variance >= within


def test_transaction_cost_is_reported():
    spec = make_spec(transaction_costs=np.array([0.01, 0.02]))
    result = RobustAllocator(spec).optimize()
    expected = spec.transaction_costs @ np.abs(result.weights - spec.current_weights)
    assert np.isclose(result.transaction_cost, expected)


def test_turnover_limit_is_respected():
    spec = make_spec(current_weights=np.array([0.5, 0.5]), max_turnover=0.2)
    result = RobustAllocator(spec).optimize()
    assert result.turnover <= 0.2 + 1e-7


def test_configuration_loader(tmp_path):
    payload = {
        "assets": ["A", "B"],
        "regimes": [
            {
                "name": "base",
                "probability": 1.0,
                "expected_returns": [0.05, 0.02],
                "covariance": [[0.01, 0.0], [0.0, 0.02]],
            }
        ],
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    spec = load_spec(path)
    assert spec.assets == ("A", "B")
    assert spec.regimes[0].name == "base"
