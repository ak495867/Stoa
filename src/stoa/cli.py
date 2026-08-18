from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .io import load_spec, write_result
from .optimizer import RobustAllocator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="stoa", description="Robust portfolio allocation under model uncertainty")
    subparsers = parser.add_subparsers(dest="command", required=True)
    optimize = subparsers.add_parser("optimize", help="optimize a portfolio from a JSON configuration")
    optimize.add_argument("config", type=Path)
    optimize.add_argument("--output", type=Path)
    validate = subparsers.add_parser("validate", help="validate a portfolio configuration")
    validate.add_argument("config", type=Path)
    return parser


def run(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        spec = load_spec(args.config)
        if args.command == "validate":
            validation = {
                "valid": True,
                "assets": list(spec.assets),
                "regimes": [regime.name for regime in spec.regimes],
            }
            print(json.dumps(validation, indent=2))
            return 0
        result = RobustAllocator(spec).optimize().as_dict()
        if args.output:
            write_result(args.output, result)
        else:
            print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (KeyError, TypeError, ValueError, OSError) as exc:
        print(f"stoa: {exc}", file=sys.stderr)
        return 2


def main() -> None:
    raise SystemExit(run())
