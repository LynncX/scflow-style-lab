"""Manifest input contract shared by the runner, preflight and evaluation."""
import math
from pathlib import Path


def input_fields(sample):
    return ("content_path", "style_paths" if "style_paths" in sample else "style_path")


def input_paths(sample, key):
    value = sample[key]
    paths = value if key == "style_paths" else [value]
    if not isinstance(paths, list) or not paths or any(
        not isinstance(path, str) or not path.strip() for path in paths
    ):
        raise ValueError(f"Invalid {key}: expected nonempty image path(s)")
    return paths


def style_references(sample, cfg):
    key = input_fields(sample)[1]
    paths = input_paths(sample, key)
    mode = cfg.get("scflow", {}).get("fusion", "mean")
    if mode not in ("mean", "weighted"):
        raise ValueError("scflow.fusion must be mean or weighted")
    if key == "style_paths" and "style_path" in sample:
        raise ValueError("Use either style_path or style_paths, not both")
    if "style_weights" in sample or (mode == "weighted" and key == "style_paths"):
        if key != "style_paths":
            raise ValueError("style_weights requires style_paths")
        weights = sample.get("style_weights")
        if not isinstance(weights, list) or len(weights) != len(paths) or any(
            type(w) not in (int, float) or w < 0 or (type(w) is float and not math.isfinite(w))
            for w in weights
        ) or not max(weights):
            raise ValueError("style_weights must match style_paths and be finite, nonnegative with positive total")
        if mode == "weighted":
            scale = max(weights)
            scaled = [w / scale for w in weights]
            total = sum(scaled)
            return paths, [w / total for w in scaled]
    return paths, [1 / len(paths)] * len(paths)


def integration_options(cfg):
    steps = cfg.get("num_steps", 1)
    if type(steps) is not int or steps < 1:
        raise ValueError("num_steps must be a positive integer")
    options = cfg.get("scflow", {})
    mode = options.get("integration", "fixed")
    if mode == "fixed":
        return steps, None
    if mode != "adaptive" or steps != 1:
        raise ValueError("Use fixed integration or adaptive integration with num_steps=1")
    threshold = options.get("adaptive_threshold")
    if type(threshold) not in (int, float) or not math.isfinite(threshold) or threshold < 0:
        raise ValueError("adaptive integration requires a finite nonnegative scflow.adaptive_threshold")
    return steps, threshold


def resolve_sample_inputs(sample, cfg, data_root):
    method = cfg.get("method", cfg["backend"])
    if method == "scflow":
        style_references(sample, cfg)
    elif "style_paths" in sample or "style_weights" in sample:
        raise ValueError("Multi-reference inputs are only supported by SCFlow")
    for key in input_fields(sample):
        paths = [str(data_root / path) for path in input_paths(sample, key)]
        for path in paths:
            if not Path(path).is_file():
                raise FileNotFoundError(path)
        sample[key] = paths if key == "style_paths" else paths[0]
