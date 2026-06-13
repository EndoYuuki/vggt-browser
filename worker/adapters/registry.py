"""Maps models.yaml `adapter:` strings to adapter classes and manages the
single resident model (one model on the GPU at a time)."""

from __future__ import annotations

from shared.model_config import ModelsConfig, load_models_config

from .base import ReconstructionAdapter
from .vggt_adapter import VGGTAdapter

ADAPTER_CLASSES: dict[str, type[ReconstructionAdapter]] = {
    "vggt": VGGTAdapter,
}

try:  # optional: only available when the vggt-omega package is installed.
    from .vggt_omega_adapter import VGGTOmegaAdapter

    ADAPTER_CLASSES["vggt_omega"] = VGGTOmegaAdapter
except Exception:  # pragma: no cover
    pass


class ModelRegistry:
    """Holds the config and the currently-resident adapter. Single-GPU: only one
    adapter is loaded at a time; switching unloads the previous."""

    def __init__(self, config: ModelsConfig | None = None, device: str = "cuda") -> None:
        self.config = config or load_models_config()
        self.device = device
        self._current: ReconstructionAdapter | None = None

    def _build(self, name: str) -> ReconstructionAdapter:
        entry = self.config.get(name)
        cls = ADAPTER_CLASSES.get(entry.adapter)
        if cls is None:
            raise KeyError(f"no adapter class for {entry.adapter!r}")
        return cls(entry)

    def get(self, name: str | None) -> ReconstructionAdapter:
        """Return a loaded adapter for `name`, swapping the resident model if needed."""
        entry = self.config.get(name)
        target = entry.name
        if self._current is not None and self._current.name == target and self._current.is_loaded:
            return self._current

        if self._current is not None:
            print(f"[registry] unloading {self._current.name} to swap in {target}")
            self._current.unload()
            self._current = None

        print(f"[registry] loading model: {target}")
        adapter = self._build(target)
        adapter.load(self.device, entry.dtype)
        self._current = adapter
        print(f"[registry] resident model: {target}")
        return adapter

    def warm(self, name: str | None = None) -> ReconstructionAdapter:
        return self.get(name or self.config.default)

    @property
    def current_name(self) -> str | None:
        return self._current.name if self._current else None
