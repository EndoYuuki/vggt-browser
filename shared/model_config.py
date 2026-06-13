"""Loader for config/models.yaml. torch-free; used by both api and worker."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = os.environ.get("MODELS_CONFIG", "/config/models.yaml")


@dataclass(frozen=True)
class Caps:
    max_frames: int
    default_frames: int
    recommended_resolution: int
    max_resolution: int

    def clamp_frames(self, n: int) -> int:
        return max(1, min(int(n), self.max_frames))

    def clamp_resolution(self, r: int) -> int:
        return max(64, min(int(r), self.max_resolution))


@dataclass(frozen=True)
class ModelEntry:
    name: str
    adapter: str
    dtype: str
    caps: Caps
    display_name: str
    description: str
    gated: bool
    # adapter-specific knobs (hf_repo, checkpoint, ...)
    options: dict[str, Any]

    def public_dict(self) -> dict[str, Any]:
        """JSON-safe view for the api /models endpoint (no secrets)."""
        return {
            "name": self.name,
            "adapter": self.adapter,
            "display_name": self.display_name,
            "description": self.description,
            "caps": {
                "max_frames": self.caps.max_frames,
                "default_frames": self.caps.default_frames,
                "recommended_resolution": self.caps.recommended_resolution,
                "max_resolution": self.caps.max_resolution,
            },
        }


@dataclass(frozen=True)
class ModelsConfig:
    default: str
    models: dict[str, ModelEntry]

    def get(self, name: str | None) -> ModelEntry:
        key = name or self.default
        if key not in self.models:
            raise KeyError(f"unknown model: {key!r} (have {list(self.models)})")
        return self.models[key]


def _parse(raw: dict[str, Any]) -> ModelsConfig:
    reserved = {
        "adapter", "dtype", "caps", "display_name", "description", "gated",
    }
    entries: dict[str, ModelEntry] = {}
    for name, cfg in raw["models"].items():
        caps_cfg = cfg["caps"]
        caps = Caps(
            max_frames=int(caps_cfg["max_frames"]),
            default_frames=int(caps_cfg["default_frames"]),
            recommended_resolution=int(caps_cfg["recommended_resolution"]),
            max_resolution=int(caps_cfg["max_resolution"]),
        )
        options = {k: v for k, v in cfg.items() if k not in reserved}
        entries[name] = ModelEntry(
            name=name,
            adapter=cfg["adapter"],
            dtype=cfg.get("dtype", "bfloat16"),
            caps=caps,
            display_name=cfg.get("display_name", name),
            description=cfg.get("description", ""),
            gated=bool(cfg.get("gated", False)),
            options=options,
        )
    default = raw.get("default") or next(iter(entries))
    return ModelsConfig(default=default, models=entries)


@lru_cache(maxsize=None)
def load_models_config(path: str = DEFAULT_CONFIG_PATH) -> ModelsConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return _parse(raw)
