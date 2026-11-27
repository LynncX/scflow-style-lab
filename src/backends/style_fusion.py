"""Select ordered reference images and fuse their CLIP embeddings."""

import math
from pathlib import Path


DEFAULT_TEMPERATURE = 0.1


def style_fusion_settings(cfg):
    if cfg["backend"] != "scflow":
        return "single", DEFAULT_TEMPERATURE
    options = cfg.get("scflow", {})
    mode = options.get("style_fusion", "single")
    if mode not in ("single", "mean", "weighted"):
        raise ValueError("scflow.style_fusion must be single, mean or weighted")
    temperature = options.get("style_temperature", DEFAULT_TEMPERATURE)
    if mode == "weighted":
        check_temperature(temperature)
    return mode, temperature


def check_temperature(temperature):
    if (isinstance(temperature, bool) or not isinstance(temperature, (float, int))
            or not math.isfinite(temperature) or temperature <= 0):
        raise ValueError("Style fusion temperature must be positive and finite")


def style_reference_paths(sample, mode):
    if mode == "single":
        return [sample["style_path"]]
    paths = sample.get("style_paths")
    sample_id = sample["sample_id"]
    if not isinstance(paths, list) or len(paths) != 3 or not all(isinstance(p, str) and p for p in paths):
        raise ValueError(f"{sample_id}: style_paths must contain exactly three reference paths")
    if Path(paths[0]) != Path(sample["style_path"]):
        raise ValueError(f"{sample_id}: first style_paths entry must match style_path")
    if len({Path(p) for p in paths}) != 3:
        raise ValueError(f"{sample_id}: style_paths must contain three distinct references")
    return paths


def fuse_style_features(references, mode, temperature=DEFAULT_TEMPERATURE):
    """Return a [1, D] raw CLIP vector and weights in input order."""
    import torch
    from torch.nn.functional import cosine_similarity

    if mode not in ("mean", "weighted") or references.ndim != 2 or references.size(0) != 3:
        raise ValueError("Mean and weighted fusion require three reference feature rows")
    features = references.float()
    if not torch.isfinite(features).all():
        raise ValueError("Reference features must be finite")
    center = features.mean(dim=0, keepdim=True)
    if mode == "mean":
        return center, features.new_full((3,), 1 / 3)
    check_temperature(temperature)
    similarities = cosine_similarity(features, center, dim=1)
    # Subtract the maximum before scaling so sharp temperatures remain stable.
    logits = (similarities - similarities.max()).double() / temperature
    weights = torch.softmax(logits, dim=0).to(features.dtype)
    return (features * weights[:, None]).sum(dim=0, keepdim=True), weights
