"""SceneResult: the single normalized output contract every adapter must satisfy.

torch-free (numpy only) so both worker and api can import it. The worker produces
a SceneResult and writes it to npz; the api reads the npz back into a SceneResult
to serialize endpoints.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import wire_format as wf


@dataclass
class SceneResult:
    points_xyz: np.ndarray   # (N, 3) float32, world coords (OpenCV convention)
    points_rgb: np.ndarray   # (N, 3) uint8
    points_conf: np.ndarray  # (N,)   float32, normalized 0..1
    point_frame: np.ndarray  # (N,)   uint16, source frame index
    depth: np.ndarray        # (S, H, W) float32
    depth_conf: np.ndarray   # (S, H, W) float32
    extrinsics: np.ndarray   # (S, 3, 4) float32, world->cam
    intrinsics: np.ndarray   # (S, 3, 3) float32
    image_size: tuple[int, int]  # (H, W)
    frame_count: int
    meta: dict[str, Any] = field(default_factory=dict)

    # -- validation ---------------------------------------------------------
    def validate(self) -> None:
        n = self.points_xyz.shape[0]
        s = self.frame_count
        h, w = self.image_size
        checks = {
            "points_xyz": (self.points_xyz.shape, (n, 3)),
            "points_rgb": (self.points_rgb.shape, (n, 3)),
            "points_conf": (self.points_conf.shape, (n,)),
            "point_frame": (self.point_frame.shape, (n,)),
            "depth": (self.depth.shape, (s, h, w)),
            "depth_conf": (self.depth_conf.shape, (s, h, w)),
            "extrinsics": (self.extrinsics.shape, (s, 3, 4)),
            "intrinsics": (self.intrinsics.shape, (s, 3, 3)),
        }
        for name, (got, want) in checks.items():
            if got != want:
                raise ValueError(f"{name} shape {got} != expected {want}")

    # -- point blob for the /points binary endpoint -------------------------
    def to_point_blob(self) -> wf.PointBlob:
        return wf.PointBlob(
            positions=self.points_xyz.astype(np.float32),
            colors=self.points_rgb.astype(np.uint8),
            conf=self.points_conf.astype(np.float32),
            frame_idx=self.point_frame.astype(np.uint16),
        )

    # -- cameras JSON -------------------------------------------------------
    def cameras_json(self) -> dict[str, Any]:
        h, w = self.image_size
        return {
            "frame_count": self.frame_count,
            "image_size": {"height": h, "width": w},
            "cameras": [
                {
                    "frame": i,
                    "extrinsic": self.extrinsics[i].tolist(),  # 3x4 world->cam
                    "intrinsic": self.intrinsics[i].tolist(),  # 3x3
                }
                for i in range(self.frame_count)
            ],
        }

    # -- npz persistence ----------------------------------------------------
    def save_npz(self, path: str) -> None:
        self.validate()
        np.savez_compressed(
            path,
            **{
                wf.NPZ_POINTS_XYZ: self.points_xyz.astype(np.float32),
                wf.NPZ_POINTS_RGB: self.points_rgb.astype(np.uint8),
                wf.NPZ_POINTS_CONF: self.points_conf.astype(np.float32),
                wf.NPZ_POINT_FRAME: self.point_frame.astype(np.uint16),
                wf.NPZ_DEPTH: self.depth.astype(np.float32),
                wf.NPZ_DEPTH_CONF: self.depth_conf.astype(np.float32),
                wf.NPZ_EXTRINSICS: self.extrinsics.astype(np.float32),
                wf.NPZ_INTRINSICS: self.intrinsics.astype(np.float32),
                wf.NPZ_IMAGE_SIZE: np.array(self.image_size, dtype=np.int32),
                wf.NPZ_FRAME_COUNT: np.array(self.frame_count, dtype=np.int32),
            },
        )
        # meta as a sibling json so it's human-readable
        meta_path = os.path.splitext(path)[0] + ".meta.json"
        with open(meta_path, "w") as f:
            json.dump(self.meta, f, indent=2, default=str)

    @classmethod
    def load_npz(cls, path: str) -> "SceneResult":
        z = np.load(path)
        h, w = (int(x) for x in z[wf.NPZ_IMAGE_SIZE])
        meta: dict[str, Any] = {}
        meta_path = os.path.splitext(path)[0] + ".meta.json"
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
        return cls(
            points_xyz=z[wf.NPZ_POINTS_XYZ],
            points_rgb=z[wf.NPZ_POINTS_RGB],
            points_conf=z[wf.NPZ_POINTS_CONF],
            point_frame=z[wf.NPZ_POINT_FRAME],
            depth=z[wf.NPZ_DEPTH],
            depth_conf=z[wf.NPZ_DEPTH_CONF],
            extrinsics=z[wf.NPZ_EXTRINSICS],
            intrinsics=z[wf.NPZ_INTRINSICS],
            image_size=(h, w),
            frame_count=int(z[wf.NPZ_FRAME_COUNT]),
            meta=meta,
        )
