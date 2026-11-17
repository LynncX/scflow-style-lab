#!/usr/bin/env python3
"""Download SCFlow checkpoints and dataset files.

Reason: prepare GPU-server resources without committing large binaries.
Created: 2026-11-17
Goal: resumable downloads with optional SHA256 verification.
Usage:
    python scripts/download_assets.py --group ckpt
    python scripts/download_assets.py --group data
    python scripts/download_assets.py --group all
    python scripts/download_assets.py --verify-only --group all
    python scripts/download_assets.py -h
"""

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

BASE = "https://huggingface.co/CompVis/SCFlow/resolve/main"
ASSETS = [
    {"name": "scflow", "group": "ckpt", "url": f"{BASE}/scflow_last.ckpt", "path": "ckpts/scflow_last.ckpt", "sha256": None, "min_bytes": 1_000_000},
    {"name": "unclip", "group": "ckpt", "url": f"{BASE}/sd21-unclip-l.ckpt", "path": "ckpts/sd21-unclip-l.ckpt", "sha256": None, "min_bytes": 1_000_000},
    {"name": "train.h5", "group": "data", "url": f"{BASE}/train.h5", "path": "dataset/train.h5", "sha256": None, "min_bytes": 1_000_000},
    {"name": "test.h5", "group": "data", "url": f"{BASE}/test.h5", "path": "dataset/test.h5", "sha256": None, "min_bytes": 1_000_000},
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(asset: dict, root: Path) -> bool:
    path = root / asset["path"]
    if not path.exists() or path.stat().st_size < asset.get("min_bytes", 1):
        return False
    expected = asset.get("sha256")
    return bool(expected) == False or sha256(path) == expected


def download(asset: dict, root: Path):
    path = root / asset["path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    if verify(asset, root):
        print(f"[skip] {asset['name']}: {path}")
        return

    partial = path.with_suffix(path.suffix + ".part")
    size = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={size}-"} if size else {}
    req = urllib.request.Request(asset["url"], headers=headers)
    print(f"[download] {asset['name']} -> {path} (resume from {size} bytes)")
    with urllib.request.urlopen(req) as resp, partial.open("ab") as out:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    if not verify(asset, root) and not sha256_ok_or_size(asset, partial):
        raise RuntimeError(f"Verification failed for {asset['name']}")
    partial.replace(path)
    print(f"[ok] {asset['name']}: {path}")


def sha256_ok_or_size(asset: dict, path: Path) -> bool:
    expected = asset.get("sha256")
    if expected:
        return sha256(path) == expected
    return path.stat().st_size >= asset.get("min_bytes", 1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--group", choices=["ckpt", "data", "all"], default="all")
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--verify-only", action="store_true")
    args = p.parse_args()

    selected = [a for a in ASSETS if args.group == "all" or a["group"] == args.group]
    ok = True
    for asset in selected:
        path = args.root / asset["path"]
        if args.verify_only:
            valid = verify(asset, args.root)
            print(f"{'ok' if valid else 'missing'} {asset['name']}: {path}")
            ok = ok and valid
        else:
            download(asset, args.root)
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
