"""Thin adapter to patched original StyleID, without persistent inversion caches."""
import importlib
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from .base import Backend, BackendResult
from sources import validate_source


def make_options(root, cfg, sample, out_dir):
    settings = cfg["styleid"]
    supported = {"source", "model_config", "ckpt", "text_encoder", "cache", "precision", "T", "gamma"}
    if set(settings) - supported or "num_steps" in cfg:
        raise ValueError("Unsupported StyleID parameters; original adapter fixes inversion/sampling at 50 steps")
    if settings.get("cache", False):
        raise ValueError("StyleID disk inversion cache is disabled in this adapter")
    if int(cfg.get("image_size", 512)) != 512 or not cfg.get("save_images", True):
        raise ValueError("StyleID requires image_size=512 and save_images=true")
    precision = settings.get("precision", "autocast")
    if precision not in {"autocast", "full"}:
        raise ValueError("StyleID precision must be autocast or full")
    content, style = Path(sample["content_path"]), Path(sample["style_path"])
    # The upstream feature indexing/merging assumes exactly 50 steps.
    return SimpleNamespace(
        cnt=str(content.parent), sty=str(style.parent), content_file=content.name, style_file=style.name,
        sample_id=sample["sample_id"], output_path=str(out_dir), precomputed="",
        model_config=str(root / settings["model_config"]), ckpt=str(root / settings["ckpt"]),
        seed=int(cfg.get("seed", 22)), ddim_inv_steps=50, save_feat_steps=50, start_step=49,
        ddim_eta=0.0, H=512, W=512, C=4, f=8, T=float(settings.get("T", 1.5)),
        gamma=float(settings.get("gamma", 0.75)), attn_layer="6,7,8,9,10,11", precision=precision,
        without_init_adain=False, without_attn_injection=False,
    )


class StyleIDBackend(Backend):
    def __init__(self, project_root):
        self.root = project_root
        self.runtime = {}
        self.upstream = None

    def infer(self, sample, cfg, out_dir):
        options = make_options(self.root, cfg, sample, out_dir)
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("Original StyleID requires CUDA; CPU fallback is not supported")
        device = torch.device(cfg.get("device", "cuda"))
        if device.type != "cuda":
            raise ValueError("Original StyleID adapter only supports CUDA")
        torch.cuda.set_device(device.index if device.index is not None else torch.cuda.current_device())
        model_load_ms = 0.0
        if self.upstream is None:
            start = time.perf_counter()
            self.source_info = validate_source(self.root, "styleid", self.root / cfg["styleid"]["source"])
            sys.path.insert(0, str(self.root / cfg["styleid"]["source"]))
            upstream = importlib.import_module("run_styleid")
            upstream.seed_everything(options.seed)
            text_encoder = self.root / cfg["styleid"]["text_encoder"]
            if not text_encoder.is_dir():
                raise FileNotFoundError(f"Local CLIP text encoder required: {text_encoder}")
            model_config = upstream.OmegaConf.load(options.model_config)
            model_config.model.params.cond_stage_config.params = {"version": str(text_encoder)}
            model = upstream.load_model_from_config(model_config, options.ckpt)
            torch.cuda.synchronize()
            self.runtime["model"] = model
            self.upstream = upstream
            model_load_ms = (time.perf_counter() - start) * 1000
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        calls = 0
        def count_calls(module, args):
            nonlocal calls
            calls += 1
        hook = self.runtime["model"].model.diffusion_model.register_forward_pre_hook(count_calls)
        try:
            # Upstream places only sampling under no_grad; also avoid inversion autograd graphs.
            with torch.no_grad():
                self.upstream.main(options, self.runtime)
            torch.cuda.synchronize()
        finally:
            hook.remove()
        elapsed = (time.perf_counter() - start) * 1000
        return BackendResult(
            {}, str(Path(out_dir) / f"{sample['sample_id']}.png"), calls,
            {"model_load_ms": model_load_ms, "inversion_sampling_save_ms": elapsed},
            torch.cuda.max_memory_allocated() / 2**20,
            {"source": self.source_info, "effective_options": vars(options), "torch": torch.__version__,
             "device": str(torch.cuda.current_device()), "cache": "disabled", "nfe_definition": "actual UNet forward calls including inversion"},
        )
