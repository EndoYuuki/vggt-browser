"""Adapter interface that normalizes different VGGT-family models to SceneResult."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

from shared.model_config import ModelEntry
from shared.scene import SceneResult

ProgressCb = Callable[[str, float], None]  # (stage, fraction 0..1)


class ReconstructionAdapter(ABC):
    """Base class for a swappable reconstruction model.

    Lifecycle: construct(entry) -> load(device,dtype) -> run(image_paths, ...) -> unload().
    `run` returns a normalized SceneResult so api/frontend never see model-specific
    output schemas.
    """

    def __init__(self, entry: ModelEntry) -> None:
        self.entry = entry
        self.name = entry.name
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @abstractmethod
    def load(self, device: str, dtype: str) -> None:
        """Load weights and move to device. Idempotent."""

    @abstractmethod
    def unload(self) -> None:
        """Free GPU memory (del model + empty_cache). Idempotent."""

    @abstractmethod
    def run(
        self,
        image_paths: list[str],
        *,
        resolution: int,
        conf_threshold: float = 0.0,
        progress_cb: ProgressCb | None = None,
        extra: dict[str, Any] | None = None,
    ) -> SceneResult:
        """Full inference: preprocess -> forward -> normalize to SceneResult."""

    # convenience no-op progress
    @staticmethod
    def _progress(cb: ProgressCb | None, stage: str, frac: float) -> None:
        if cb is not None:
            cb(stage, frac)
