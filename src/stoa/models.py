from dataclasses import dataclass
from typing import Any

import numpy as np


class StoaValidationError(ValueError):
    pass


@dataclass(frozen=True)
class Regime:
    name: str
    probability: float
    expected_returns: np.ndarray
    covariance: np.ndarray

    def __post_init__(self) -> None:
        returns = np.asarray(self.expected_returns, dtype=float)
        covariance = np.asarray(self.covariance, dtype=float)
        if returns.ndim != 1:
            raise StoaValidationError(
                "expected_returns must be a one-dimensional array"
            )
        if covariance.shape != (returns.size, returns.size):
            raise StoaValidationError(
                "covariance must be square with one row per asset"
            )
        if not np.isfinite(returns).all() or not np.isfinite(covariance).all():
            raise StoaValidationError("regime inputs must be finite")
        if self.probability < 0 or not np.isfinite(self.probability):
            raise StoaValidationError(
                "regime probability must be non-negative and finite"
            )
        if not np.allclose(covariance, covariance.T, atol=1e-10):
            raise StoaValidationError("covariance must be symmetric")
        eigenvalues = np.linalg.eigvalsh(covariance)
        if eigenvalues.min() < -1e-9:
            raise StoaValidationError("covariance must be positive semidefinite")
        object.__setattr__(self, "expected_returns", returns)
        object.__setattr__(self, "covariance", covariance)


@dataclass(frozen=True)
class PortfolioSpec:
    assets: tuple[str, ...]
    regimes: tuple[Regime, ...]
    risk_aversion: float = 4.0
    uncertainty_radius: float = 0.02
    transaction_costs: np.ndarray | None = None
    current_weights: np.ndarray | None = None
    minimum_weights: np.ndarray | None = None
    maximum_weights: np.ndarray | None = None
    budget: float = 1.0
    turnover_penalty: float = 0.05
    max_turnover: float | None = None
    iterations: int = 1500
    step_size: float = 0.05

    def __post_init__(self) -> None:
        assets = tuple(self.assets)
        if not assets or len(set(assets)) != len(assets):
            raise StoaValidationError("assets must be non-empty and unique")
        if not self.regimes:
            raise StoaValidationError("at least one regime is required")
        n = len(assets)
        probabilities = np.array(
            [regime.probability for regime in self.regimes], dtype=float
        )
        if not np.isclose(probabilities.sum(), 1.0, atol=1e-8):
            raise StoaValidationError("regime probabilities must sum to one")
        for regime in self.regimes:
            if regime.expected_returns.size != n:
                raise StoaValidationError(
                    "every regime must contain one expected return per asset"
                )
        if (
            self.risk_aversion < 0
            or self.uncertainty_radius < 0
            or self.turnover_penalty < 0
        ):
            raise StoaValidationError(
                "risk and penalty parameters must be non-negative"
            )
        if self.budget <= 0:
            raise StoaValidationError("budget must be positive")
        if self.iterations < 1 or self.step_size <= 0:
            raise StoaValidationError("iterations and step_size must be positive")
        costs = self._vector_or_default(
            self.transaction_costs, np.zeros(n), n, "transaction_costs"
        )
        current = self._vector_or_default(
            self.current_weights, np.zeros(n), n, "current_weights"
        )
        minimum = self._vector_or_default(
            self.minimum_weights, np.zeros(n), n, "minimum_weights"
        )
        maximum = self._vector_or_default(
            self.maximum_weights, np.full(n, self.budget), n, "maximum_weights"
        )
        if (costs < 0).any():
            raise StoaValidationError("transaction costs must be non-negative")
        if (minimum > maximum).any():
            raise StoaValidationError("minimum_weights cannot exceed maximum_weights")
        if minimum.sum() > self.budget + 1e-10 or maximum.sum() < self.budget - 1e-10:
            raise StoaValidationError("weight bounds cannot satisfy the budget")
        if self.max_turnover is not None and self.max_turnover < 0:
            raise StoaValidationError("max_turnover must be non-negative")
        if np.abs(current).sum() > 0 and not np.isclose(
            current.sum(), self.budget, atol=1e-8
        ):
            raise StoaValidationError(
                "current_weights must sum to budget when supplied"
            )
        object.__setattr__(self, "assets", assets)
        object.__setattr__(self, "transaction_costs", costs)
        object.__setattr__(self, "current_weights", current)
        object.__setattr__(self, "minimum_weights", minimum)
        object.__setattr__(self, "maximum_weights", maximum)

    @staticmethod
    def _vector_or_default(
        value: np.ndarray | None, default: np.ndarray, size: int, name: str
    ) -> np.ndarray:
        result = default if value is None else np.asarray(value, dtype=float)
        if result.shape != (size,):
            raise StoaValidationError(f"{name} must contain one value per asset")
        if not np.isfinite(result).all():
            raise StoaValidationError(f"{name} must be finite")
        return result


@dataclass(frozen=True)
class OptimizationResult:
    assets: tuple[str, ...]
    weights: np.ndarray
    expected_return: float
    robust_return: float
    variance: float
    volatility: float
    transaction_cost: float
    turnover: float
    objective: float
    iterations: int
    converged: bool
    regime_returns: dict[str, float]
    regime_probabilities: dict[str, float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "assets": list(self.assets),
            "weights": {
                asset: float(weight)
                for asset, weight in zip(self.assets, self.weights, strict=True)
            },
            "expected_return": self.expected_return,
            "robust_return": self.robust_return,
            "variance": self.variance,
            "volatility": self.volatility,
            "transaction_cost": self.transaction_cost,
            "turnover": self.turnover,
            "objective": self.objective,
            "iterations": self.iterations,
            "converged": self.converged,
            "regime_returns": self.regime_returns,
            "regime_probabilities": self.regime_probabilities,
        }
