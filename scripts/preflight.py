#!/usr/bin/env python3
"""Background (2026-11): read-only experiment checks before costly GPU execution.

Usage: python scripts/preflight.py configs/experiments/adain.example.json
Reports missing inputs, pinned sources, weights and worker interpreter dependencies.
Does not install, download, import models or claim CUDA/model compatibility.
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from runner import load_config, apply_overrides, resolve_project_path
from experiment_inputs import input_fields, input_paths, integration_options, resolve_sample_inputs
from sources import validate_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()
    cfg = apply_overrides(load_config(args.config), args.set)
    failures = []
    def check(label, operation):
        try:
            operation()
            print(f"[ok] {label}")
        except (ValueError, KeyError, OSError, subprocess.SubprocessError) as exc:
            failures.append(label)
            print(f"[missing/invalid] {label}: {exc}")
    def file(value):
        path = Path(resolve_project_path(value, ROOT))
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    def inputs():
        manifest = Path(resolve_project_path(cfg["manifest"], ROOT))
        samples = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
        if not samples:
            raise ValueError("empty manifest")
        data_root = Path(resolve_project_path(cfg.get("data_root", "."), ROOT))
        for sample in samples:
            resolve_sample_inputs(sample, cfg, data_root)
            for key in input_fields(sample):
                for path in input_paths(sample, key):
                    file(path)
    check("manifest and input files", inputs)
    backend = cfg["backend"]
    if cfg.get("method", backend) == "scflow":
        check("SCFlow integration options", lambda: integration_options(cfg))
    if backend in {"adain", "stytr2", "styleid"}:
        options = cfg[backend]
        check("pinned source and patches", lambda: validate_source(ROOT, backend, ROOT / options["source"]))
        keys = {"adain": ["vgg", "decoder"], "stytr2": ["vgg", "decoder", "transformer", "embedding"],
                "styleid": ["ckpt", "model_config"]}[backend]
        for key in keys:
            check(f"weight {key}", lambda key=key: file(options[key]))
        if backend == "styleid":
            from backends.styleid import make_options
            check("StyleID fixed-step/cache/options contract", lambda: make_options(ROOT, cfg, {
                "sample_id": "preflight", "content_path": "content.png", "style_path": "style.png"}, ROOT))
            for name in ("config.json", "vocab.json", "merges.txt"):
                check(f"local CLIP {name}", lambda name=name: file(str(Path(options["text_encoder"]) / name)))
            def clip_weights():
                directory = ROOT / options["text_encoder"]
                if not any((directory / name).is_file() for name in ("pytorch_model.bin", "model.safetensors", "pytorch_model.bin.index.json", "model.safetensors.index.json")):
                    raise FileNotFoundError(f"CLIP model weights: {directory}")
            check("local CLIP weights", clip_weights)
    elif backend == "scflow":
        options = cfg["scflow"]
        for key in ("ckpt", "unclip_ckpt"):
            check(f"weight {key}", lambda key=key: file(options[key]))
        check("SCFlow model config", lambda: file(options.get("config", "third_party/scflow_pkg/inference.yaml")))
    elif backend != "mock":
        failures.append("backend")
        print(f"[not implemented] preflight for {backend}; this does not imply its runtime is unavailable")
    def interpreter():
        if backend in {"adain", "stytr2", "styleid"} and not cfg.get("worker_command"):
            raise ValueError(f"{backend} requires worker_command for environment isolation")
        command = cfg.get("worker_command", [sys.executable])
        if not isinstance(command, list) or not command or not all(isinstance(s, str) for s in command):
            raise ValueError("worker_command must be an argv list")
        executable = command[0]
        if "/" in executable:
            executable = str(ROOT / executable)
        elif not shutil.which(executable):
            raise FileNotFoundError(executable)
        modules = [] if backend == "mock" else ["torch", "torchvision", "PIL", "numpy"]
        if backend == "stytr2":
            modules += ["scipy", "einops"]
        if backend == "styleid":
            modules += ["omegaconf", "pytorch_lightning", "einops", "transformers", "kornia", "clip", "taming", "tqdm"]
        if backend == "scflow":
            modules += ["omegaconf", "pytorch_lightning", "torchdiffeq", "clip"]
        code = "import importlib.util,json; print(json.dumps([m for m in " + repr(modules) + " if importlib.util.find_spec(m) is None]))"
        result = subprocess.run([executable, *command[1:], "-c", code], cwd=ROOT, check=True,
                                capture_output=True, text=True, timeout=30)
        missing = json.loads(result.stdout)
        if missing:
            raise ValueError("missing modules: " + ", ".join(missing))
    check("worker interpreter and module availability", interpreter)
    print(f"[info] free disk: {shutil.disk_usage(ROOT).free / 2**30:.1f} GiB; CUDA/import/VRAM compatibility not tested")
    return int(bool(failures))


if __name__ == "__main__":
    sys.exit(main())
