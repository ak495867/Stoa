# Stoa

Stoa is a Python command-line tool for portfolio construction under model uncertainty. It allocates across multiple return and covariance regimes instead of relying on one estimated forecast. The allocator combines probability-weighted expected returns, within-regime covariance, between-regime dispersion, ambiguity protection, portfolio bounds, turnover limits, and transaction costs.

> Stoa is a research and engineering toolkit. It does not provide personalized financial advice.

## Features

| Capability | Description |
| --- | --- |
| Regime-aware allocation | Supports any finite set of named market regimes with probabilities, return vectors, and covariance matrices. |
| Model uncertainty control | Applies an ambiguity penalty based on the Euclidean size of the portfolio weight vector. |
| Risk aggregation | Combines expected within-regime variance with between-regime return dispersion. |
| Trading frictions | Supports asset-level transaction costs and quadratic turnover penalties relative to current holdings. |
| Portfolio constraints | Supports a budget, minimum and maximum weights, and a maximum turnover limit. |
| Reproducible CLI | Reads and writes deterministic JSON files with no external data source required. |
| Python API | Exposes validated models and the `RobustAllocator` class for library usage. |

## Installation

Stoa requires Python 3.10 or newer.

```bash
git clone https://github.com/ak495867/Stoa.git
cd Stoa
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Quick start

Validate the included example configuration:

```bash
stoa validate examples/basic_config.json
```

Optimize the example portfolio and write the result to a JSON file:

```bash
stoa optimize examples/basic_config.json --output output.json
```

To print the result directly to standard output:

```bash
stoa optimize examples/basic_config.json
```

## Configuration format

A configuration contains an ordered list of assets and one or more regimes. Regime probabilities must sum to one. Returns and covariance values should use consistent units, such as decimal annualized returns and decimal annualized covariance.

```json
{
  "assets": ["US_EQ", "GOV_BOND", "GOLD"],
  "regimes": [
    {
      "name": "expansion",
      "probability": 0.6,
      "expected_returns": [0.09, 0.035, 0.045],
      "covariance": [
        [0.0324, -0.0020, 0.0040],
        [-0.0020, 0.0064, 0.0010],
        [0.0040, 0.0010, 0.0144]
      ]
    },
    {
      "name": "stress",
      "probability": 0.4,
      "expected_returns": [-0.12, 0.075, 0.11],
      "covariance": [
        [0.0784, -0.0060, 0.0100],
        [-0.0060, 0.0121, 0.0020],
        [0.0100, 0.0020, 0.0361]
      ]
    }
  ],
  "risk_aversion": 4.0,
  "uncertainty_radius": 0.02,
  "transaction_costs": [0.001, 0.0002, 0.0008],
  "current_weights": [0.5, 0.4, 0.1],
  "minimum_weights": [0.0, 0.0, 0.0],
  "maximum_weights": [0.8, 0.8, 0.5],
  "budget": 1.0,
  "turnover_penalty": 0.05,
  "max_turnover": 0.8,
  "iterations": 1500,
  "step_size": 0.05
}
```

The optional fields default to a unit budget, zero current holdings, zero transaction costs, long-only lower bounds, a full-budget upper bound per asset, four units of risk aversion, 0.02 ambiguity radius, 0.05 turnover penalty, 1,500 iterations, and a step size of 0.05.

## Objective

For portfolio weights \(w\), Stoa calculates the probability-weighted return \(\mu(w)\), a total variance that includes within-regime and between-regime components, an ambiguity penalty \(\rho\lVert w\rVert_2\), transaction costs, and a quadratic turnover penalty.

The optimized objective is:

\[
U(w) = \mu(w) - \rho\lVert w\rVert_2 - \frac{\gamma}{2}V(w) - c^T|w-w_0| - \lambda\lVert w-w_0\rVert_1^2
\]

The optimizer uses projected gradient ascent with backtracking. Projection enforces the budget and asset bounds. If a turnover limit is supplied, the candidate is contracted toward the current portfolio before the final budget projection.

## Python API

```python
import numpy as np

from stoa import PortfolioSpec, Regime, RobustAllocator

spec = PortfolioSpec(
    assets=("US_EQ", "BOND"),
    regimes=(
        Regime(
            name="base",
            probability=1.0,
            expected_returns=np.array([0.08, 0.03]),
            covariance=np.array([[0.04, -0.002], [-0.002, 0.01]]),
        ),
    ),
)
result = RobustAllocator(spec).optimize()
print(result.as_dict())
```

## Development

Run the test suite and static checks from the repository root:

```bash
python -m pytest
ruff check .
```

## License

Stoa is released under the MIT License. See `LICENSE`.
