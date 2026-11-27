import time
import struct
import zlib
from pathlib import Path

from .base import Backend, BackendResult

def mock_png(size: int) -> bytes:
    """Solid RGB fixture, not an algorithm prediction."""
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress((b"\0" + b"\x40\x80\xc0" * size) * size)) + chunk(b"IEND", b""))

class MockBackend(Backend):
    def infer(self, sample: dict, cfg: dict, out_dir: str) -> BackendResult:
        if sample.get("mock_fail"):
            raise RuntimeError("Requested mock failure")
        start = time.perf_counter()
        sid = sample["sample_id"]
        digest = __import__("hashlib").sha256(f"{sid}:{cfg.get('seed', 0)}".encode()).digest()
        vec = [(digest[i % len(digest)] / 255.0) * 2 - 1 for i in range(768)]
        image_path = None
        if cfg.get("save_images", True):
            p = Path(out_dir) / f"{sid}.png"
            p.write_bytes(mock_png(int(cfg.get("image_size", 64))))
            image_path = str(p)
        elapsed = (time.perf_counter() - start) * 1000
        is_scflow = cfg.get("method", "scflow") in {"mock", "scflow"}
        return BackendResult(
            embeddings={"0": vec, "1": vec, "mean": vec} if is_scflow else {},
            image_path=image_path,
            nfe=int(cfg.get("num_steps", 1)) if is_scflow and cfg.get("scflow", {}).get("integration") != "adaptive" else None,
            timing_ms={"feature_ms": elapsed, "decode_ms": 0.0},
            peak_vram_mb=None,
        )
