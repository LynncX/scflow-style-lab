import os
import sys
import time
from pathlib import Path

from PIL import Image

from .base import Backend, BackendResult


class CLIPEncoder:
    def __init__(self, device):
        import clip
        import torch

        self.device = device
        self.model, self.preprocess = clip.load("ViT-L/14", device=str(device))
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

    def _prepare_unclip_config(self) -> Path:
        src = self.pkg_root / "v2-1-stable-unclip-l-inference.yaml"
        dst = self.pkg_root / "configs" / "v2-1-stable-unclip-l-inference.yaml"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())
        return dst.parent

    def infer(self, sample: dict, cfg: dict, out_dir: str) -> BackendResult:
        sc = cfg["scflow"]
        ckpt = self.project_root / sc["ckpt"]
        unclip = self.project_root / sc["unclip_ckpt"]
        for label, path in [("checkpoint", ckpt), ("unclip", unclip)]:
            if not path.exists():
                raise FileNotFoundError(f"Missing {label}: {path}")

        sys.path.insert(0, str(self.pkg_root))
        try:
            import torch
            from omegaconf import OmegaConf

            from scflow.trainer_module import TrainerModuleSCFlow
        finally:
            sys.path.pop(0)

        infer_cfg = OmegaConf.load(self.pkg_root / "inference.yaml")
        infer_cfg.val_step_num = int(cfg.get("num_steps", 1))
        infer_cfg.unclip_ckpt = str(unclip)
        infer_cfg.model_type = sc.get("model_type", "scflow")

        config_dir = self._prepare_unclip_config()
        cwd = os.getcwd()
        os.chdir(config_dir)
        try:
            module = TrainerModuleSCFlow.load_from_checkpoint(
                checkpoint_path=str(ckpt),
                fm_cfg=infer_cfg.model.fm,
                test_vis=bool(cfg.get("save_images", True)),
                unclip_ckpt=str(unclip),
                val_step_num=infer_cfg.val_step_num,
                reverse_inference=False,
                strict=False,
                map_location="cpu",
            )
        finally:
            os.chdir(cwd)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        module = module.to(device)
        module.eval()

        encoder = CLIPEncoder(device)
        c = encoder.encode_image(self.project_root / sample["content_path"]).to(device=device, dtype=module.dtype)
        s = encoder.encode_image(self.project_root / sample["style_path"]).to(device=device, dtype=module.dtype)
        x = torch.cat([c, s], dim=2)

        start = time.perf_counter()
        with torch.no_grad():
            pred = module.predict_forward(x)
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
        if cfg.get("save_images", True):
            start = time.perf_counter()
            image_path = str(Path(out_dir) / f"{sample['sample_id']}.png")
            module.get_sample_to_vis(emb_tensor["mean"], path=image_path)
            decode_ms = (time.perf_counter() - start) * 1000

        peak = None
        if device.type == "cuda":
            peak = torch.cuda.max_memory_allocated() / (1024 * 1024)

        return BackendResult(
            embeddings=emb,
            image_path=image_path,
            nfe=int(cfg.get("num_steps", 1)),
            timing_ms={"feature_ms": feature_ms, "decode_ms": decode_ms},
            peak_vram_mb=peak,
        )
