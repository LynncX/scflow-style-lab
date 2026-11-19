"""Pinned source validation shared by preflight, bootstrap and algorithm workers."""
import hashlib
import json
import subprocess
from pathlib import Path


def specifications(root):
    return json.loads((root / "manifests/upstreams.json").read_text())


def validate_source(root, method, source):
    source = Path(source)
    if not (source / ".git").exists():
        raise ValueError(f"Missing Git checkout: {source}")
    spec = specifications(root)[method]
    def git(*args):
        return subprocess.check_output(["git", "-C", str(source), *args], text=True).strip()
    commit = git("rev-parse", "HEAD")
    if commit != spec["commit"]:
        raise ValueError(f"{method}: unexpected source commit {commit}")
    expected = spec.get("patched_files_sha256", {})
    # Reject untracked, staged and unrelated edits, including edits outside the permitted patch.
    if git("ls-files", "--others", "--exclude-standard") or git("diff", "--cached", "--name-only"):
        raise ValueError(f"{method}: upstream has untracked or staged changes")
    changed = set(git("diff", "--name-only").splitlines())
    if changed != set(expected):
        raise ValueError(f"{method}: source changes do not match required patch; run bootstrap --prepare")
    for name, digest in expected.items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"{method}: unexpected contents in {name}")
    return {"commit": commit, "patch": spec.get("patch"), "patched_files_sha256": expected}
