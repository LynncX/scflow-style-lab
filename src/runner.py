import importlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

def load_config(path: Path) -> dict:
    cfg = json.loads(path.read_text(encoding="utf-8")) if path.suffix == ".json" else None
    if cfg is not None:
        return cfg
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("Need PyYAML for YAML configs, or use a JSON config") from exc
    return yaml.safe_load(path.read_text(encoding="utf-8"))

def make_backend(cfg: dict, project_root: Path):
    name = cfg["backend"]
    module = importlib.import_module(f"backends.{name}")
    cls = getattr(module, {"mock": "MockBackend", "scflow": "SCFlowBackend"}[name])
    if name == "scflow":
        return cls(Path(cfg["scflow"]["repo"]))
    return cls()

def run(cfg_path: Path) -> Path:
    project_root = Path(__file__).resolve().parents[1]
    cfg = load_config(cfg_path)
    manifest_path = Path(cfg["manifest"])
    if not manifest_path.is_absolute():
        manifest_path = project_root / manifest_path
    samples = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    run_name = cfg["run_name"]
    out_dir = project_root / cfg.get("out_root", "runs") / f"{run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    preds = out_dir / "predictions"
    preds.mkdir(parents=True, exist_ok=True)

    backend = make_backend(cfg, project_root)
    records = []
    for sample in samples:
        sample_dir = preds / sample["sample_id"]
        sample_dir.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        result = backend.infer(sample, cfg, str(sample_dir))
        for key, value in result.embeddings.items():
            if not isinstance(value, list):
                value = [float(v) for v in value.tolist()]
            (sample_dir / f"clip_pred_{key}.json").write_text(json.dumps(value), encoding="utf-8")
        record = {
            "sample_id": sample["sample_id"],
            "backend": cfg["backend"],
            **{k: str(v) if isinstance(v, Path) else v for k, v in sample.items()},
            "nfe": result.nfe,
            "timing_ms": result.timing_ms,
            "peak_vram_mb": result.peak_vram_mb,
            "image_path": result.image_path,
            "total_ms": (time.perf_counter() - started) * 1000,
        }
        records.append(record)

    summary = {
        "run_name": run_name,
        "backend": cfg["backend"],
        "config": cfg,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "samples": records,
    }
    (out_dir / "run.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_dir

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Run SCFlow-style-lab baseline")
    p.add_argument("config", type=Path)
    args = p.parse_args()
    print(run(args.config))
