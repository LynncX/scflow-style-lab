#!/usr/bin/env python3
"""Background (2026-11): prepare pinned full sources without installing environments.

Usage: python scripts/bootstrap.py --check
       python scripts/bootstrap.py --prepare --local-root /path/to/existing/clones
Without --local-root, --prepare clones from the pinned upstream URL (network required).
Never overwrites existing dirty checkouts. No weights or dependencies are downloaded.
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sources import specifications, validate_source


def prepare(root, method, spec, local_root=None):
    target = root / "third_party" / spec["directory"]
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        source = str(Path(local_root).resolve() / spec["directory"]) if local_root else spec["url"]
        subprocess.run(["git", "clone", "--no-hardlinks", "--no-checkout", source, str(target)], check=True)
        subprocess.run(["git", "-C", str(target), "checkout", "--detach", spec["commit"]], check=True)
    try:
        return validate_source(root, method, target)
    except ValueError:
        # Apply only to a clean checkout at the exact revision; never reset or clean user changes.
        commit = subprocess.check_output(["git", "-C", str(target), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", str(target), "status", "--porcelain"], text=True).strip()
        if commit != spec["commit"] or dirty or not spec.get("patch"):
            raise
        patch = str(root / spec["patch"])
        subprocess.run(["git", "-C", str(target), "apply", "--check", patch], check=True)
        subprocess.run(["git", "-C", str(target), "apply", patch], check=True)
        return validate_source(root, method, target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="read-only checks (default)")
    mode.add_argument("--prepare", action="store_true", help="explicitly clone pinned sources/apply patches")
    parser.add_argument("--local-root", type=Path)
    parser.add_argument("--method", action="append", choices=list(specifications(ROOT)))
    args = parser.parse_args()
    failed = False
    for method, spec in specifications(ROOT).items():
        if args.method and method not in args.method:
            continue
        try:
            if args.prepare:
                prepare(ROOT, method, spec, args.local_root)
            else:
                validate_source(ROOT, method, ROOT / "third_party" / spec["directory"])
            print(f"[ok] {method}")
        except (ValueError, OSError, subprocess.CalledProcessError) as exc:
            print(f"[missing/invalid] {method}: {exc}", file=sys.stderr)
            failed = True
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
