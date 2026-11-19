#!/usr/bin/env python3
"""Background (2026-11): validate/export shared evaluation inputs, not new metric definitions.

Usage: python scripts/evaluate.py runs/experiment
       python scripts/evaluate.py runs/mock_experiment --allow-simulated
Metrics are intentionally pending the existing server evaluator and reference protocol.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evaluation.inputs import export_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--allow-simulated", action="store_true")
    parser.add_argument("--allow-failures", action="store_true", help="explicitly export an incomplete subset; not a fair main-table comparison")
    args = parser.parse_args()
    try:
        print(export_inputs(args.run_dir, args.allow_simulated, args.allow_failures))
    except (ValueError, KeyError, OSError) as exc:
        print(f"Evaluation input validation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
