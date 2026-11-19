from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class BackendResult:
    embeddings: dict[str, list[float]]
    image_path: str | None
    nfe: int | None
    timing_ms: dict[str, float]
    peak_vram_mb: float | None = None
    provenance: dict | None = None

class Backend(ABC):
    def close(self):
        """Release optional worker resources."""

    @abstractmethod
    def infer(self, sample: dict, cfg: dict, out_dir: str) -> BackendResult:
        raise NotImplementedError
