#!/usr/bin/env python3
"""Background (2026-11): persistent isolated algorithm execution for comparisons.

Usage: /environment/bin/python src/worker.py < requests.jsonl
Normally launched by runner with worker_command; --help describes the protocol.
"""
import argparse
import json
import sys
import traceback
from contextlib import redirect_stdout
from dataclasses import asdict
from pathlib import Path

from runner import make_backend


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    backend = None
    initial_config = None
    try:
        for line in sys.stdin:
            try:
                request = json.loads(line)
                cfg = request["config"]
                with redirect_stdout(sys.stderr):
                    if backend is None:
                        initial_config = cfg
                        backend = make_backend(cfg, Path(__file__).resolve().parents[1])
                    if cfg != initial_config:
                        raise ValueError("Worker config cannot change within a process")
                    result = backend.infer(request["sample"], cfg, request["out_dir"])
                    result.provenance = {**(result.provenance or {}),
                                         "worker_python": sys.version,
                                         "worker_executable": sys.executable}
                response = {"result": asdict(result)}
            except Exception as exc:
                traceback.print_exc(file=sys.stderr)
                response = {"error": str(exc)}
            print(json.dumps(response), flush=True)
    finally:
        if backend is not None:
            backend.close()


if __name__ == "__main__":
    main()
