import math
import os
import sys
import time
from pathlib import Path

from .base import Backend, BackendResult
from experiment_inputs import integration_options, style_references


def fuse_styles(paths, weights, encode):
    if len(paths) == 1:
        return encode(paths[0])
    return sum(encode(path).float() * weight for path, weight in zip(paths, weights))


def endpoint_gap(pred):
    a, b = pred[..., :768].float(), pred[..., 768:].float()
    return (a - b).square().mean().sqrt().item()


def integrate(module, x, steps, threshold):
    """Use upstream Euler stepping and count actual vector-field evaluations."""
    nfe = 0

    def count(*_):
        nonlocal nfe
        nfe += 1

    hook = module.model.register_forward_hook(count)
    original_steps = module.val_step_num
    try:
        module.val_step_num = steps
        pred = module.predict_forward(x)
        gap = None
        if threshold is not None:
            gap = endpoint_gap(pred)
            if not math.isfinite(gap):
                raise ValueError("Non-finite one-step SCFlow endpoint gap")
            if gap > threshold:
                module.val_step_num = 4
                pred = module.predict_forward(x)
        selected_steps = module.val_step_num
    finally:
        hook.remove()
        module.val_step_num = original_steps
    return pred, nfe, {
        "integration": "fixed" if threshold is None else "adaptive",
        "one_step_endpoint_rmse": gap,
        "adaptive_threshold": threshold,
        "fallback": selected_steps != steps,
        "selected_steps": selected_steps,
    }


class CLIPEncoder:
    def __init__(self, device):
        import clip
        import torch

        self.device = device
        self.model, self.preprocess = clip.load("ViT-L/14", device=str(device))
        self.model.eval()

    def encode_image(self, image_path):
        import torch
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        x = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.model.encode_image(x)


class SCFlowBackend(Backend):
    def __init__(self, project_root: Path):
        self.project_root = project_root.resolve()
        self.pkg_root = self.project_root / "third_party" / "scflow_pkg"
        self._loaded = None

    def _prepare_unclip_config(self) -> Path:
        config_dir = self.pkg_root / "configs"
        config_dir.mkdir(exist_ok=True)
        for name in ("v2-1-stable-unclip-l-inference.yaml", "ViT-L-14_stats.th"):
            (config_dir / name).write_bytes((self.pkg_root / name).read_bytes())
        return self.pkg_root

    def infer(self, sample: dict, cfg: dict, out_dir: str) -> BackendResult:
        # The bundled Stable Diffusion modules also use absolute ldm imports.
        original_path = sys.path[:]
        sys.path[:0] = [str(self.pkg_root), str(self.pkg_root / "scflow")]
        try:
            return self._infer(sample, cfg, out_dir)
        except Exception:
            self._loaded = None
            raise
        finally:
            sys.path[:] = original_path

    def _infer(self, sample: dict, cfg: dict, out_dir: str) -> BackendResult:
        import torch
        from omegaconf import OmegaConf

        from scflow.trainer_module import TrainerModuleSCFlow

        sc = cfg["scflow"]
        ckpt = self.project_root / sc["ckpt"]
        unclip = self.project_root / sc["unclip_ckpt"]
        for label, path in [("checkpoint", ckpt), ("unclip", unclip)]:
            if not path.exists():
                raise FileNotFoundError(f"Missing {label}: {path}")

        device = (
            torch.device("cuda", torch.cuda.current_device())
            if torch.cuda.is_available() else torch.device("cpu")
        )
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        seed = cfg.get("seed")
        if seed is not None:
            seed = int(seed)
            torch.manual_seed(seed)

        config_path = self.project_root / sc.get(
            "config", "third_party/scflow_pkg/inference.yaml"
        )
        steps, threshold = integration_options(cfg)
        paths, weights = style_references(sample, cfg)
        save_images = bool(cfg.get("save_images", True))
        model_key = (config_path, ckpt, unclip, steps, save_images, seed)
        model_load_ms = 0.0
        if self._loaded is None or self._loaded[0] != model_key:
            self._loaded = None
            started = time.perf_counter()
            infer_cfg = OmegaConf.load(config_path)
            config_dir = self._prepare_unclip_config()
            cwd = os.getcwd()
            os.chdir(config_dir)
            try:
                module = TrainerModuleSCFlow.load_from_checkpoint(
                    checkpoint_path=str(ckpt),
                    fm_cfg=infer_cfg.model.fm,
                    test_vis=save_images,
                    unclip_ckpt=str(unclip),
                    val_step_num=steps,
                    n_intermediates=0,
                    reverse_inference=False,
                    strict=False,
                    map_location="cpu",
                )
            finally:
                os.chdir(cwd)
            module.eval()
            self._loaded = (model_key, module)
            model_load_ms = (time.perf_counter() - started) * 1000
        else:
            _, module = self._loaded

        module.model.to(device)

        encoder = CLIPEncoder(device)
        c = encoder.encode_image(self.project_root / sample["content_path"]).to(device=device, dtype=module.dtype)
        s = fuse_styles(paths, weights, lambda path: encoder.encode_image(self.project_root / path))
        s = s.to(device=device, dtype=module.dtype)
        x = torch.cat([c, s], dim=-1).unsqueeze(1)
        del encoder

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        with torch.no_grad():
            pred, nfe, provenance = integrate(module, x, steps, threshold)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        feature_ms = (time.perf_counter() - start) * 1000
        p0, p1 = pred[:, :, :768], pred[:, :, 768:]
        emb_tensor = {
            "0": p0,
            "1": p1,
            "mean": (p0 + p1) / 2.0,
        }
        emb = {k: v.squeeze().detach().cpu().tolist() for k, v in emb_tensor.items()}

        image_path = None
        decode_ms = 0.0
        if save_images:
            image_path = str(Path(out_dir) / f"{sample['sample_id']}.png")
            module.model.cpu()
            if seed is not None:
                torch.manual_seed(seed)
            output_size = cfg.get("image_size")
            start = time.perf_counter()
            module.get_sample_to_vis(
                emb_tensor["mean"], path=image_path, device=device,
                output_size=int(output_size) if output_size is not None else None,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            decode_ms = (time.perf_counter() - start) * 1000

        peak = None
        if device.type == "cuda":
            peak = torch.cuda.max_memory_allocated() / (1024 * 1024)

        return BackendResult(
            embeddings=emb,
            image_path=image_path,
            nfe=nfe,
            timing_ms={"model_load_ms": model_load_ms, "feature_ms": feature_ms, "decode_ms": decode_ms},
            peak_vram_mb=peak,
            provenance={**provenance, "fusion": sc.get("fusion", "mean"), "style_weights": weights},
        )
