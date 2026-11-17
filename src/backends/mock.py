import base64
import json
import time
from pathlib import Path

from .base import Backend, BackendResult

PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADUlEQVR4nGNgYGD4DwABBAEAX+XDSwAAAABJRU5ErkJggg=="
)

class MockBackend(Backend):
    def infer(self, sample: dict, cfg: dict, out_dir: str) -> BackendResult:
        start = time.perf_counter()
        sid = sample["sample_id"]
        digest = __import__("hashlib").sha256(f"{sid}:{cfg.get('seed', 0)}".encode()).digest()
        vec = [(digest[i % len(digest)] / 255.0) * 2 - 1 for i in range(768)]
        image_path = None
        if cfg.get("save_images", True):
            p = Path(out_dir) / f"{sid}.png"
            p.write_bytes(PNG_1X1)
            image_path = str(p)
        elapsed = (time.perf_counter() - start) * 1000
        return BackendResult(
            embeddings={"0": vec, "1": vec, "mean": vec},
            image_path=image_path,
            nfe=int(cfg.get("num_steps", 1)),
            timing_ms={"feature_ms": elapsed, "decode_ms": 0.0},
            peak_vram_mb=None,
        )
