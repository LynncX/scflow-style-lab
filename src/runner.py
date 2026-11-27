import argparse
import importlib
import json
import os
import sys
import time
import traceback
import hashlib
import re
import subprocess
from contextlib import redirect_stdout, closing
from datetime import datetime, timezone
from pathlib import Path
from experiment_inputs import input_fields, input_paths, integration_options, resolve_sample_inputs

ENV_OVERRIDES = {
    "SCFLOW_MANIFEST": "manifest",
    "SCFLOW_OUT_ROOT": "out_root",
    "SCFLOW_CKPT": "scflow.ckpt",
    "SCFLOW_UNCLIP_CKPT": "scflow.unclip_ckpt",
}


def load_config(path: Path) -> dict:
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("Need PyYAML for YAML configs, or use a JSON config") from exc
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def set_nested(cfg: dict, dotted: str, value):
    node = cfg
    parts = dotted.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    for env, key in ENV_OVERRIDES.items():
        if os.environ.get(env):
            set_nested(cfg, key, os.environ[env])
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid --set {item!r}; expected key=value")
        key, value = item.split("=", 1)
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass
        set_nested(cfg, key, value)
    return cfg


def resolve_project_path(value: str, project_root: Path) -> str:
    p = Path(value)
    return str(p if p.is_absolute() else project_root / p)


def make_backend(cfg: dict, project_root: Path, out_dir: Path | None = None, in_worker: bool = False):
    if cfg.get("worker_command"):
        from backends.worker import WorkerBackend
        return WorkerBackend(cfg, project_root, out_dir)
    name = cfg["backend"]
    if name in {"adain", "stytr2", "styleid"} and not in_worker:
        raise ValueError(f"{name} requires worker_command pointing to its isolated Python environment")
    if name == "mock":
        from backends.mock import MockBackend
        return MockBackend()
    module = importlib.import_module(f"backends.{name}")
    cls = getattr(module, {"scflow": "SCFlowBackend", "adain": "AdaINBackend", "stytr2": "StyTr2Backend", "styleid": "StyleIDBackend"}[name])
    if name in {"scflow", "adain", "stytr2", "styleid"}:
        return cls(project_root)
    return cls()


def load_records(out_dir: Path) -> list[dict]:
    journal = out_dir / "records.jsonl"
    if journal.exists():
        records = []
        lines = journal.read_text(encoding="utf-8").splitlines(keepends=True)
        for i, line in enumerate(lines):
            try:
                records.append(json.loads(line))
                if not line.endswith("\n"):
                    with journal.open("a", encoding="utf-8") as f:
                        f.write("\n")
            except json.JSONDecodeError:
                if i != len(lines) - 1 or line.endswith("\n"):
                    raise
                # A killed process may leave an incomplete final append.
                journal.write_text("".join(lines[:i]), encoding="utf-8")
        return records
    path = out_dir / "run.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("samples", [])


def sample_complete(record: dict, out_dir: Path) -> bool:
    return record.get("status") == "ok" and bool(record.get("artifacts")) and all(
        (out_dir / name).is_file() and (out_dir / name).stat().st_size > 0
        for name in record["artifacts"]
    )


def write_json(path: Path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)


def code_revision(project_root: Path) -> dict:
    try:
        def git(*args):
            return subprocess.check_output(["git", "-C", str(project_root), *args], text=True).strip()
        return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def file_sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def run(cfg_path: Path, overrides: list[str] | None = None, out_dir_override: str | None = None, resume: bool = False) -> Path:
    project_root = Path(__file__).resolve().parents[1]
    cfg = apply_overrides(load_config(cfg_path), overrides or [])
    manifest_path = Path(resolve_project_path(cfg["manifest"], project_root))
    samples = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    method = cfg.get("method", cfg["backend"])
    if method not in {"mock", "scflow", "adain", "stytr2", "styleid"}:
        raise ValueError(f"Unknown method: {method}")
    if cfg["backend"] != "mock" and method != cfg["backend"]:
        raise ValueError("method must match the real backend")
    if method == "scflow":
        integration_options(cfg)
    seen = set()
    data_root = Path(resolve_project_path(cfg.get("data_root", "."), project_root))
    for sample in samples:
        sid = sample["sample_id"]
        if not isinstance(sid, str) or not re.fullmatch(r"[\w-][\w.-]*", sid) or sid in seen:
            raise ValueError(f"Unsafe or duplicate sample_id: {sid!r}")
        seen.add(sid)
        resolve_sample_inputs(sample, cfg, data_root)
    identity = {"config": cfg, "samples": samples,
                "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                "inputs_sha256": {p: file_sha256(Path(p))
                    for p in {p for s in samples for k in input_fields(s) for p in input_paths(s, k)}}}
    if cfg["backend"] in {"adain", "stytr2", "styleid"}:
        options = cfg[cfg["backend"]]
        from sources import validate_source
        identity["source"] = validate_source(project_root, cfg["backend"], project_root / options["source"])
        keys = {"adain": ["vgg", "decoder"], "stytr2": ["vgg", "decoder", "transformer", "embedding"],
                "styleid": ["ckpt", "model_config"]}[cfg["backend"]]
        identity["weights_sha256"] = {key: file_sha256(project_root / options[key]) for key in keys}
        if cfg["backend"] == "styleid":
            text_encoder = project_root / options["text_encoder"]
            if not text_encoder.is_dir():
                raise FileNotFoundError(f"Local CLIP text encoder required: {text_encoder}")
            identity["text_encoder_sha256"] = {str(p.relative_to(text_encoder)): file_sha256(p)
                                               for p in sorted(text_encoder.rglob("*")) if p.is_file()}
            if not identity["text_encoder_sha256"]:
                raise ValueError("Local CLIP text encoder directory is empty")
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()

    if out_dir_override:
        out_dir = Path(out_dir_override)
        if not out_dir.is_absolute():
            out_dir = project_root / out_dir
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = project_root / cfg.get("out_root", "runs") / f"{cfg['run_name']}_{stamp}"
    if out_dir.exists() and any(out_dir.iterdir()):
        snapshot = out_dir / "config.json"
        if not resume or not snapshot.exists():
            raise ValueError("Nonempty output directory; use --resume with a compatible config.json, or a new directory")
        if json.loads(snapshot.read_text())["fingerprint"] != fingerprint:
            raise ValueError("Configuration, manifest or inputs changed; use a new output directory")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "config.json", {"fingerprint": fingerprint, **identity})
    provenance_path = out_dir / "provenance.json"
    if not provenance_path.exists():
        write_json(provenance_path, {"code": code_revision(project_root),
                   "python": sys.version, "simulated": cfg["backend"] == "mock",
                   "note": "Minimal controller provenance; upstream/environment/weight verification not implemented."})
    preds = out_dir / "predictions"
    preds.mkdir(parents=True, exist_ok=True)

    records = load_records(out_dir) if resume else []
    latest = {r["sample_id"]: r for r in records}
    done = {sid for sid, record in latest.items() if sample_complete(record, out_dir)}
    errors_path = out_dir / "errors.jsonl"

    log_path = out_dir / "log.txt"
    log_file = log_path.open("a", encoding="utf-8")

    def emit(msg: str):
        print(msg)
        print(msg, file=log_file, flush=True)

    with log_file, redirect_stdout(sys.stdout), closing(make_backend(cfg, project_root, out_dir)) as backend:
        emit(f"run_dir={out_dir}")
        emit(f"backend={cfg['backend']} samples={len(samples)} resume={resume}")

        for sample in samples:
            sid = sample["sample_id"]
            if sid in done:
                emit(f"[skip] {sid}")
                continue
            sample_dir = preds / sid
            sample_dir.mkdir(parents=True, exist_ok=True)
            started = time.perf_counter()
            try:
                result = backend.infer(sample, cfg, str(sample_dir))
                artifacts = []
                for key, value in result.embeddings.items():
                    if not isinstance(value, list):
                        value = [float(v) for v in value.tolist()]
                    (sample_dir / f"clip_pred_{key}.json").write_text(json.dumps(value), encoding="utf-8")
                    artifacts.append(str((sample_dir / f"clip_pred_{key}.json").relative_to(out_dir)))
                if cfg.get("save_images", True):
                    if not result.image_path or not Path(result.image_path).is_file():
                        raise ValueError("Backend did not produce the required image")
                    artifacts.append(str(Path(result.image_path).relative_to(out_dir)))
                if not artifacts:
                    raise ValueError("Backend produced no artifacts")
                record = {
                    "sample_id": sid,
                    "backend": cfg["backend"],
                    **{k: str(v) if isinstance(v, Path) else v for k, v in sample.items()},
                    "nfe": result.nfe,
                    "timing_ms": result.timing_ms,
                    "peak_vram_mb": result.peak_vram_mb,
                    "image_path": result.image_path,
                    "total_ms": (time.perf_counter() - started) * 1000,
                    "status": "ok",
                    "method": method,
                    "simulated": cfg["backend"] == "mock",
                    "artifacts": artifacts,
                    "provenance": result.provenance,
                }
                records.append(record)
                emit(f"[ok] {sid} {record['total_ms']:.1f}ms")
            except Exception as exc:
                record = {
                    "sample_id": sid,
                    "backend": cfg["backend"],
                    **{k: str(v) if isinstance(v, Path) else v for k, v in sample.items()},
                    "status": "error",
                    "method": method,
                    "simulated": cfg["backend"] == "mock",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "total_ms": (time.perf_counter() - started) * 1000,
                }
                records.append(record)
                with errors_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                emit(f"[error] {sid}: {exc}")

            with (out_dir / "records.jsonl").open("a", encoding="utf-8") as journal:
                journal.write(json.dumps(record, ensure_ascii=False) + "\n")
                journal.flush()
                os.fsync(journal.fileno())

        summary = {
            "run_name": cfg["run_name"],
            "backend": cfg["backend"],
            "config": cfg,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "simulated": cfg["backend"] == "mock",
            "samples": list({r["sample_id"]: r for r in records}.values()),
        }
        write_json(out_dir / "run.json", summary)
        emit(f"summary={out_dir / 'run.json'}")
    return out_dir


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run SCFlow-style-lab baseline")
    p.add_argument("config", type=Path)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override config value")
    p.add_argument("--out-dir", default=None, help="fixed output directory; enables stable resume target")
    p.add_argument("--resume", action="store_true", help="skip completed samples in --out-dir")
    args = p.parse_args()
    output = run(args.config, args.set, args.out_dir, args.resume)
    print(output)
    summary = json.loads((output / "run.json").read_text())
    sys.exit(1 if any(r["status"] != "ok" for r in summary["samples"]) else 0)
