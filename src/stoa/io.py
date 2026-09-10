from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .models import PortfolioSpec, Regime


def load_spec(path: str | Path) -> PortfolioSpec:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"configuration file not found: {source}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"configuration file is not valid JSON: {exc.msg}") from exc
    return spec_from_dict(payload)


def spec_from_dict(payload: dict[str, Any]) -> PortfolioSpec:
    if not isinstance(payload, dict):
        raise ValueError("configuration root must be an object")
    regimes = tuple(
        Regime(
            name=str(item["name"]),
            probability=float(item["probability"]),
            expected_returns=np.array(item["expected_returns"], dtype=float),
            covariance=np.array(item["covariance"], dtype=float),
        )
        for item in payload["regimes"]
    )
    values = {
        "assets": tuple(str(asset) for asset in payload["assets"]),
        "regimes": regimes,
    }
    optional_fields = (
        "risk_aversion",
        "uncertainty_radius",
        "transaction_costs",
        "current_weights",
        "minimum_weights",
        "maximum_weights",
        "budget",
        "turnover_penalty",
        "max_turnover",
        "iterations",
        "step_size",
    )
    for field in optional_fields:
        if field in payload:
            vector_fields = {
                "transaction_costs",
                "current_weights",
                "minimum_weights",
                "maximum_weights",
            }
            values[field] = (
                np.array(payload[field], dtype=float)
                if field in vector_fields
                else payload[field]
            )
    return PortfolioSpec(**values)


def write_result(path: str | Path, result: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
