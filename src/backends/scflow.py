import os
import sys
import time
from pathlib import Path

from PIL import Image

from .base import Backend, BackendResult
from .style_fusion import fuse_style_features, style_fusion_settings, style_reference_paths


class CLIPEncoder:
    def __init__(self, device):
        import clip

        self.device = device
        cache_root = os.environ.get("XDG_CACHE_HOME")
        download_root = str(Path(cache_root) / "clip") if cache_root else None
        self.model, self.preprocess = clip.load("ViT-L/14", device=str(device), download_root=download_root)
        self.model.eval()

    def encode_image(self, image_path):
        import torch

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
        fusion_mode, temperature = style_fusion_settings(cfg)
        style_paths = style_reference_paths(sample, fusion_mode)
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
        steps = int(cfg.get("num_steps", 1))
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
        reference_features = [encoder.encode_image(self.project_root / path) for path in style_paths]
        del encoder

        fusion_ms = 0.0
        weights = [1.0]
        if fusion_mode == "single":
            s = reference_features[0]
        else:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            fusion_start = time.perf_counter()
            s, reference_weights = fuse_style_features(torch.cat(reference_features), fusion_mode, temperature)
            weights = reference_weights.tolist()
            fusion_ms = (time.perf_counter() - fusion_start) * 1000
        s = s.to(device=device, dtype=module.dtype)
        x = torch.cat([c, s], dim=-1).unsqueeze(1)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        with torch.no_grad():
            pred = module.predict_forward(x)
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
            nfe=steps,
            timing_ms={
                "model_load_ms": model_load_ms, "style_fusion_ms": fusion_ms,
                "feature_ms": feature_ms, "decode_ms": decode_ms,
            },
            peak_vram_mb=peak,
            provenance={"style_fusion": {
                "mode": fusion_mode, "references": style_paths, "weights": weights,
                "temperature": temperature if fusion_mode == "weighted" else None,
            }},
        )
