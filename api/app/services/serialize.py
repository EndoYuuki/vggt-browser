"""Read worker-written npz and serialize for the frontend (torch-free).

- points  -> packed binary (shared.wire_format)
- cameras -> JSON
- depth/conf per-frame -> PNG (colormap), generated lazily and cached.
"""

from __future__ import annotations

import io
import os

import numpy as np
from PIL import Image

from shared.scene import SceneResult
from shared.wire_format import pack_points

# Headless matplotlib for colormaps only.
import matplotlib

matplotlib.use("Agg")
from matplotlib import cm  # noqa: E402


def _npz_path(results_dir: str, job_id: str) -> str:
    return os.path.join(results_dir, job_id, "arrays.npz")


def load_scene(results_dir: str, job_id: str) -> SceneResult | None:
    path = _npz_path(results_dir, job_id)
    if not os.path.exists(path):
        return None
    return SceneResult.load_npz(path)


def points_binary(scene: SceneResult) -> bytes:
    return pack_points(scene.to_point_blob())


def cameras_json(scene: SceneResult) -> dict:
    return scene.cameras_json()


def _to_png(arr_rgb_u8: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr_rgb_u8, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


def depth_png(scene: SceneResult, frame: int) -> bytes:
    d = scene.depth[frame].astype(np.float32)
    finite = np.isfinite(d)
    lo = np.percentile(d[finite], 2) if finite.any() else 0.0
    hi = np.percentile(d[finite], 98) if finite.any() else 1.0
    norm = np.clip((d - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    rgb = (cm.get_cmap("turbo")(norm)[..., :3] * 255).astype(np.uint8)
    return _to_png(rgb)


def conf_png(scene: SceneResult, frame: int) -> bytes:
    c = np.clip(scene.depth_conf[frame].astype(np.float32), 0.0, 1.0)
    rgb = (cm.get_cmap("viridis")(c)[..., :3] * 255).astype(np.uint8)
    return _to_png(rgb)


def cache_png(results_dir: str, job_id: str, name: str, data: bytes) -> str:
    cache_dir = os.path.join(results_dir, job_id, "png")
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, name)
    with open(path, "wb") as f:
        f.write(data)
    return path
