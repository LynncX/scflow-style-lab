import argparse
import importlib
import json
import os
import sys
import time
import traceback
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

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
        set_nested(cfg, key, value)
    return cfg


def resolve_project_path(value: str, project_root: Path) -> str:
    p = Path(value)
    return str(p if p.is_absolute() else project_root / p)


def make_backend(cfg: dict, project_root: Path):
    name = cfg["backend"]
    module = importlib.import_module(f"backends.{name}")
    cls = getattr(module, {"mock": "MockBackend", "scflow": "SCFlowBackend"}[name])
    if name == "scflow":
        return cls(project_root)
    return cls()


def load_records(out_dir: Path) -> list[dict]:
    path = out_dir / "run.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("samples", [])


def sample_complete(sample_dir: Path, sample_id: str) -> bool:
    required = ["clip_pred_0.json", "clip_pred_1.json", "clip_pred_mean.json"]
    return all((sample_dir / name).exists() for name in required) and (sample_dir / f"{sample_id}.png").exists()


def run(cfg_path: Path, overrides: list[str] | None = None, out_dir_override: str | None = None, resume: bool = False) -> Path:
    project_root = Path(__file__).resolve().parents[1]
    cfg = apply_overrides(load_config(cfg_path), overrides or [])
    manifest_path = Path(resolve_project_path(cfg["manifest"], project_root))
    samples = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    if out_dir_override:
        out_dir = Path(out_dir_override)
        if not out_dir.is_absolute():
            out_dir = project_root / out_dir
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = project_root / cfg.get("out_root", "runs") / f"{cfg['run_name']}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    preds = out_dir / "predictions"
    preds.mkdir(parents=True, exist_ok=True)

    records = load_records(out_dir) if resume else []
    done = {r["sample_id"] for r in records}
    errors_path = out_dir / "errors.jsonl"

    log_path = out_dir / "log.txt"
    log_file = log_path.open("a", encoding="utf-8")

    def emit(msg: str):
        print(msg)
        print(msg, file=log_file, flush=True)

    with redirect_stdout(sys.stdout):
        emit(f"run_dir={out_dir}")
        emit(f"backend={cfg['backend']} samples={len(samples)} resume={resume}")
        backend = make_backend(cfg, project_root)

        for sample in samples:
            sid = sample["sample_id"]
            if sid in done:
                emit(f"[skip] {sid}")
                continue
            sample_dir = preds / sid
            sample_dir.mkdir(parents=True, exist_ok=True)
            if resume and sample_complete(sample_dir, sid):
                emit(f"[skip-complete] {sid}")
                continue
            started = time.perf_counter()
            try:
                result = backend.infer(sample, cfg, str(sample_dir))
                for key, value in result.embeddings.items():
                    if not isinstance(value, list):
                        value = [float(v) for v in value.tolist()]
                    (sample_dir / f"clip_pred_{key}.json").write_text(json.dumps(value), encoding="utf-8")
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
                }
                records.append(record)
                emit(f"[ok] {sid} {record['total_ms']:.1f}ms")
            except Exception as exc:
                record = {
                    "sample_id": sid,
                    "backend": cfg["backend"],
                    **{k: str(v) if isinstance(v, Path) else v for k, v in sample.items()},
                    "status": "error",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "total_ms": (time.perf_counter() - started) * 1000,
                }
                records.append(record)
                with errors_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                emit(f"[error] {sid}: {exc}")

        summary = {
            "run_name": cfg["run_name"],
            "backend": cfg["backend"],
            "config": cfg,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "samples": records,
        }
        (out_dir / "run.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        emit(f"summary={out_dir / 'run.json'}")
    log_file.close()
    return out_dir


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run SCFlow-style-lab baseline")
    p.add_argument("config", type=Path)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override config value")
    p.add_argument("--out-dir", default=None, help="fixed output directory; enables stable resume target")
    p.add_argument("--resume", action="store_true", help="skip completed samples in --out-dir")
    args = p.parse_args()
    print(run(args.config, args.set, args.out_dir, args.resume))
