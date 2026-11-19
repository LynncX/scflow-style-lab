"""Strictly prepare ordered evaluation pairs; never computes substitute research metrics."""
import hashlib
import json
import struct
from pathlib import Path

from runner import file_sha256, write_json


def prepare_inputs(run_dir, allow_simulated=False, allow_failures=False):
    root = Path(run_dir).resolve()
    snapshot = json.loads((root / "config.json").read_text())
    fingerprint = snapshot.pop("fingerprint")
    if hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode()).hexdigest() != fingerprint:
        raise ValueError("Run snapshot fingerprint does not match contents")
    cfg = snapshot["config"]
    simulated = cfg["backend"] == "mock"
    if simulated and not allow_simulated:
        raise ValueError("Mock outputs require explicit --allow-simulated; not research results")
    records_path = root / "records.jsonl"
    records = [json.loads(line) for line in records_path.read_text().splitlines() if line.strip()]
    latest = {record["sample_id"]: record for record in records}
    samples = snapshot["samples"]
    ids = [sample["sample_id"] for sample in samples]
    if len(set(ids)) != len(ids) or set(latest) - set(ids):
        raise ValueError("Duplicate snapshot IDs or unexpected record IDs")
    method = cfg.get("method", cfg["backend"])
    pairs, excluded = [], []
    for sample in samples:
        sid = sample["sample_id"]
        record = latest.get(sid)
        if not record or record.get("status") != "ok":
            excluded.append({"sample_id": sid, "reason": "missing" if not record else record.get("error", "failed")})
            continue
        if record.get("method") != method or record.get("simulated") != simulated:
            raise ValueError(f"Method/simulation mismatch: {sid}")
        for key in ("content_path", "style_path"):
            path = sample[key]
            if record.get(key) != path or file_sha256(Path(path)) != snapshot["inputs_sha256"][path]:
                raise ValueError(f"Input pair or image content changed: {sid}/{key}")
        if not isinstance(record.get("image_path"), str) or not record["image_path"]:
            raise ValueError(f"Evaluation requires PNG image predictions: {sid}")
        image = Path(record["image_path"])
        image = (root / image).resolve()
        if not image.is_relative_to(root):
            raise ValueError(f"Prediction outside run directory: {sid}")
        with image.open("rb") as source:
            header = source.read(24)
            source.seek(-12, 2)
            end = source.read()
        if len(header) != 24 or header[:12] != b"\x89PNG\r\n\x1a\n\x00\x00\x00\r" or header[12:16] != b"IHDR" or end != b"\x00\x00\x00\x00IEND\xaeB`\x82":
            raise ValueError(f"Prediction is not a complete PNG container: {sid}")
        size = int(cfg.get("image_size", 512))
        if struct.unpack(">II", header[16:24]) != (size, size):
            raise ValueError(f"Wrong prediction dimensions: {sid}")
        pairs.append({"sample_id": sid, "content_path": sample["content_path"], "style_path": sample["style_path"],
                      "prediction_path": str(image), "prediction_sha256": file_sha256(image)})
    if excluded and not allow_failures:
        raise ValueError(f"Run has {len(excluded)} failed/missing samples; --allow-failures explicitly exports a non-comparable subset")
    if not pairs:
        raise ValueError("No successful predictions available")
    return {"schema_version": 1, "kind": "evaluation_inputs_only", "simulated": simulated, "method": method,
            "run_fingerprint": fingerprint, "manifest_sha256": snapshot["manifest_sha256"],
            "total_samples": len(samples), "accepted_samples": len(pairs), "complete": not excluded,
            "excluded": excluded, "pairs": pairs,
            "warning": "No metrics computed. PNG container checks do not prove pixel-level decodability. Incomplete runs are not fair main-table comparisons."}


def export_inputs(run_dir, allow_simulated=False, allow_failures=False):
    report = prepare_inputs(run_dir, allow_simulated, allow_failures)
    directory = Path(run_dir) / "evaluation"
    directory.mkdir(exist_ok=True)
    path = directory / "inputs.json"
    write_json(path, report)
    return path
