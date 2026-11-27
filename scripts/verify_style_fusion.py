"""Validate a real multi-reference SCFlow run and export ordered fusion weights."""

import argparse
import csv
import json
import math
from pathlib import Path
import sys

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from backends.style_fusion import style_fusion_settings
from evaluation.inputs import prepare_inputs


def verify(run_dir, expected_count):
    report = prepare_inputs(run_dir)
    summary = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    config = summary["config"]
    mode, temperature = style_fusion_settings(config)
    if (report["method"] != "scflow" or mode not in ("mean", "weighted")
            or config.get("num_steps") != 1 or config.get("seed") != 2025 or config.get("image_size") != 512):
        raise ValueError("Expected the fixed one-step SCFlow multi-reference protocol")
    if report["accepted_samples"] != expected_count:
        raise ValueError(f"Expected {expected_count} completed samples")
    records = {record["sample_id"]: record for record in summary["samples"]}
    weight_rows = []
    for pair in report["pairs"]:
        sid = pair["sample_id"]
        record = records[sid]
        fusion = record["provenance"]["style_fusion"]
        paths, weights = fusion["references"], fusion["weights"]
        if (record["nfe"] != 1 or fusion["mode"] != mode or paths != pair["style_paths"]
                or fusion["temperature"] != (temperature if mode == "weighted" else None)):
            raise ValueError(f"Incorrect fusion or integration protocol: {sid}")
        if (len(weights) != 3 or not all(math.isfinite(w) and w >= 0 for w in weights)
                or not math.isclose(sum(weights), 1, abs_tol=1e-6)):
            raise ValueError(f"Invalid fusion weights: {sid}")
        if mode == "mean" and not all(math.isclose(w, 1 / 3, abs_tol=1e-6) for w in weights):
            raise ValueError(f"Mean fusion must have uniform weights: {sid}")
        image = Path(pair["prediction_path"])
        with Image.open(image) as picture:
            picture.load()
            if picture.mode != "RGB":
                raise ValueError(f"Expected RGB output: {sid}")
        for branch in ("0", "1", "mean"):
            path = image.parent / f"clip_pred_{branch}.json"
            vector = json.loads(path.read_text(encoding="utf-8"))
            if len(vector) != 768 or not all(math.isfinite(v) for v in vector):
                raise ValueError(f"Invalid CLIP output: {sid}/{branch}")
        for reference_index, (path, weight) in enumerate(zip(paths, weights), start=1):
            weight_rows.append((sid, mode, reference_index, path, weight))

    target = run_dir / "evaluation" / "style_weights.csv"
    target.parent.mkdir(exist_ok=True)
    temporary = target.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(("sample_id", "fusion", "reference_index", "reference_path", "weight"))
        writer.writerows(weight_rows)
    temporary.replace(target)
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--expected-count", type=int, required=True)
    args = parser.parse_args()
    print(verify(args.run_dir, args.expected_count))
