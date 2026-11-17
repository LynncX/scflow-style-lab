#!/usr/bin/env python3
"""Check Python, CUDA, GPU and core runtime dependencies.

Reason: fail fast before GPU-server baseline runs.
Created: 2026-11-17
Goal: print a compact environment report and exit non-zero on blockers.
Usage:
    python scripts/check_gpu.py
    python scripts/check_gpu.py -h
"""

import argparse
import importlib
import sys

MODULES = ["torch", "torchvision", "lightning", "omegaconf", "clip", "PIL"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.parse_args()

    print(f"python: {sys.version.split()[0]}")
    block = False
    for name in MODULES:
        try:
            mod = importlib.import_module(name)
            version = getattr(mod, "__version__", "installed")
            print(f"{name}: {version}")
        except Exception as exc:
            block = True
            print(f"{name}: MISSING ({exc})")

    try:
        import torch
        print(f"cuda_available: {torch.cuda.is_available()}")
        print(f"cuda_version: {torch.version.cuda}")
        print(f"gpu_count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"gpu[{i}]: {name}, {total:.1f} GiB")
        if not torch.cuda.is_available():
            block = True
    except Exception as exc:
        block = True
        print(f"torch cuda check failed: {exc}")

    print("status: " + ("BLOCKED" if block else "OK"))
    sys.exit(1 if block else 0)


if __name__ == "__main__":
    main()
