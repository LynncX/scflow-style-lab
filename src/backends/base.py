from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class BackendResult:
    embeddings: dict[str, list[float]]
    image_path: str | None
    nfe: int
    timing_ms: dict[str, float]
    peak_vram_mb: float | None = None

class Backend(ABC):
    @abstractmethod
    def infer(self, sample: dict, cfg: dict, out_dir: str) -> BackendResult:
        raise NotImplementedError
