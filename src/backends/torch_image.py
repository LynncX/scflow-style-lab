"""Shared image IO/timing only; imports remain inside the algorithm worker."""
import hashlib
import time
from pathlib import Path

from .base import Backend, BackendResult
from sources import validate_source


class TorchImageBackend(Backend):
    def __init__(self, project_root):
        self.root = project_root
        self.loaded = False
        self.weights = {}

    def infer(self, sample, cfg, out_dir):
        import torch
        from PIL import Image
        from torchvision import transforms
        from torchvision.utils import save_image

        if not cfg.get("save_images", True):
            raise ValueError("Image-only algorithms require save_images=true")

        if not self.loaded:
            start = time.perf_counter()
            self.options = cfg[self.method]
            self.device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
            torch.manual_seed(int(cfg.get("seed", 2025)))
            self.source = self.root / self.options["source"]
            self.source_info = validate_source(self.root, self.method, self.source)
            self.build(torch)
            self.sync(torch)
            load_ms = (time.perf_counter() - start) * 1000
            self.loaded = True
        else:
            load_ms = 0.0
        size = int(cfg.get("image_size", 512))
        transform = transforms.Compose([transforms.Resize(size), transforms.CenterCrop(size), transforms.ToTensor()])
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        self.sync(torch)
        start = time.perf_counter()
        def read(path):
            with Image.open(path) as image:
                return transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        content, style = read(sample["content_path"]), read(sample["style_path"])
        with torch.no_grad():
            output = self.forward(content, style)
        self.sync(torch)
        inference_ms = (time.perf_counter() - start) * 1000
        image_path = Path(out_dir) / f"{sample['sample_id']}.png"
        save_image(output.cpu(), str(image_path))
        return BackendResult({}, str(image_path), None,
                             {"model_load_ms": load_ms, "preprocess_inference_ms": inference_ms},
                             torch.cuda.max_memory_allocated(self.device) / 2**20 if self.device.type == "cuda" else None,
                             {"source": self.source_info, "weights_sha256": self.weights,
                              "preprocessing": "RGB, torchvision Resize(short edge), CenterCrop, ToTensor",
                              "device": str(self.device), "torch": torch.__version__})

    def sync(self, torch):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def weight(self, torch, key):
        path = self.root / self.options[key]
        with path.open("rb") as source:
            self.weights[key] = hashlib.file_digest(source, "sha256").hexdigest()
        # These upstream files contain plain state dictionaries. Never allow arbitrary pickle execution.
        return torch.load(path, map_location="cpu", weights_only=True)
