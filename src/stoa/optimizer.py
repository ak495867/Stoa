from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import OptimizationResult, PortfolioSpec


@dataclass(frozen=True)
class _State:
    expected_return: float
    robust_return: float
    variance: float
    transaction_cost: float
    turnover: float
    objective: float


class RobustAllocator:
    def __init__(self, spec: PortfolioSpec) -> None:
        self.spec = spec
        self.probabilities = np.array([regime.probability for regime in spec.regimes])
        self.returns = np.vstack([regime.expected_returns for regime in spec.regimes])
        self.covariances = np.stack([regime.covariance for regime in spec.regimes])
        self.mean_return = self.probabilities @ self.returns

    def optimize(self) -> OptimizationResult:
        weights = self._project(self.spec.current_weights.copy())
        state = self._state(weights)
        converged = False
        completed_iterations = 0
        for iteration in range(1, self.spec.iterations + 1):
            gradient = self._gradient(weights)
            if np.linalg.norm(gradient, ord=np.inf) < 1e-9:
                converged = True
                completed_iterations = iteration
                break
            step = self.spec.step_size
            accepted = False
            while step > 1e-10:
                candidate = self._project(weights + step * gradient)
                candidate_state = self._state(candidate)
                if candidate_state.objective >= state.objective - 1e-12:
                    accepted = True
                    break
                step *= 0.5
            if not accepted:
                converged = True
                completed_iterations = iteration
                break
            movement = np.linalg.norm(candidate - weights, ord=np.inf)
            weights = candidate
            state = candidate_state
            completed_iterations = iteration
            if movement < 1e-9:
                converged = True
                break
        return self._result(weights, state, completed_iterations, converged)

    def _state(self, weights: np.ndarray) -> _State:
        regime_returns = self.returns @ weights
        expected_return = float(self.probabilities @ regime_returns)
        centered = regime_returns - expected_return
        regime_variances = np.einsum("i,rij,j->r", weights, self.covariances, weights)
        within_variance = float(np.sum(self.probabilities * regime_variances))
        between_variance = float(np.sum(self.probabilities * centered**2))
        variance = max(within_variance + between_variance, 0.0)
        robust_return = expected_return - self.spec.uncertainty_radius * float(np.linalg.norm(weights))
        turnover = float(np.abs(weights - self.spec.current_weights).sum())
        transaction_cost = float(self.spec.transaction_costs @ np.abs(weights - self.spec.current_weights))
        risk_charge = 0.5 * self.spec.risk_aversion * variance
        turnover_charge = self.spec.turnover_penalty * turnover**2
        objective = robust_return - risk_charge - transaction_cost - turnover_charge
        return _State(expected_return, robust_return, variance, transaction_cost, turnover, objective)

    def _gradient(self, weights: np.ndarray) -> np.ndarray:
        regime_returns = self.returns @ weights
        expected_return = float(self.probabilities @ regime_returns)
        covariance_gradient = 2.0 * np.einsum("r,rij,j->ri", self.probabilities, self.covariances, weights).sum(axis=0)
        regime_deviation = (regime_returns - expected_return)[:, None]
        return_deviation = self.returns - self.mean_return
        between_gradient = 2.0 * np.sum(self.probabilities[:, None] * regime_deviation * return_deviation, axis=0)
        risk_gradient = covariance_gradient + between_gradient
        norm = max(float(np.linalg.norm(weights)), 1e-12)
        ambiguity_gradient = self.spec.uncertainty_radius * weights / norm
        turnover_difference = weights - self.spec.current_weights
        cost_gradient = self.spec.transaction_costs * np.sign(turnover_difference)
        turnover_gradient = 2.0 * self.spec.turnover_penalty * turnover_difference
        risk_charge_gradient = 0.5 * self.spec.risk_aversion * risk_gradient
        return self.mean_return - ambiguity_gradient - risk_charge_gradient - cost_gradient - turnover_gradient

    def _project(self, weights: np.ndarray) -> np.ndarray:
        lower = self.spec.minimum_weights
        upper = self.spec.maximum_weights
        candidate = np.clip(weights, lower, upper)
        if np.isclose(candidate.sum(), self.spec.budget, atol=1e-10, rtol=0.0):
            return candidate
        left = float(np.min(candidate - upper)) - self.spec.budget - 1.0
        right = float(np.max(candidate - lower)) + self.spec.budget + 1.0
        for _ in range(100):
            threshold = 0.5 * (left + right)
            projected = np.clip(weights - threshold, lower, upper)
            if projected.sum() > self.spec.budget:
                left = threshold
            else:
                right = threshold
        projected = np.clip(weights - 0.5 * (left + right), lower, upper)
        residual = self.spec.budget - projected.sum()
        if abs(residual) > 1e-8:
            slack = np.maximum(upper - projected, 0.0) if residual > 0 else np.maximum(projected - lower, 0.0)
            if slack.sum() > 0:
                projected += residual * slack / slack.sum()
        if self.spec.max_turnover is not None:
            projected = self._project_turnover(projected)
        return projected

    def _project_turnover(self, weights: np.ndarray) -> np.ndarray:
        difference = weights - self.spec.current_weights
        turnover = float(np.abs(difference).sum())
        if turnover <= self.spec.max_turnover + 1e-10:
            return weights
        scale = self.spec.max_turnover / turnover
        return self._project_without_turnover(self.spec.current_weights + scale * difference)

    def _project_without_turnover(self, weights: np.ndarray) -> np.ndarray:
        lower = self.spec.minimum_weights
        upper = self.spec.maximum_weights
        candidate = np.clip(weights, lower, upper)
        residual = self.spec.budget - candidate.sum()
        if abs(residual) <= 1e-10:
            return candidate
        slack = np.maximum(upper - candidate, 0.0) if residual > 0 else np.maximum(candidate - lower, 0.0)
        if slack.sum() == 0:
            return candidate
        return np.clip(candidate + residual * slack / slack.sum(), lower, upper)

    def _result(self, weights: np.ndarray, state: _State, iterations: int, converged: bool) -> OptimizationResult:
        regime_returns = self.returns @ weights
        return OptimizationResult(
            assets=self.spec.assets,
            weights=weights,
            expected_return=state.expected_return,
            robust_return=state.robust_return,
            variance=state.variance,
            volatility=float(np.sqrt(state.variance)),
            transaction_cost=state.transaction_cost,
            turnover=state.turnover,
            objective=state.objective,
            iterations=iterations,
            converged=converged,
            regime_returns={
                regime.name: float(value)
                for regime, value in zip(self.spec.regimes, regime_returns, strict=True)
            },
            regime_probabilities={regime.name: float(regime.probability) for regime in self.spec.regimes},
        )
