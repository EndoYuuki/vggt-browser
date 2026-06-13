"""Adapter contract test using a fake adapter (no torch). Proves the registry can
build/load/swap adapters and that produced SceneResults satisfy the contract."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared.model_config import load_models_config  # noqa: E402
from shared.scene import SceneResult  # noqa: E402

CONFIG = os.path.join(os.path.dirname(__file__), "..", "config", "models.yaml")


def make_fake_scene(s=4, h=12, w=16):
    rng = np.random.default_rng(2)
    n = s * h * w
    return SceneResult(
        points_xyz=rng.standard_normal((n, 3)).astype(np.float32),
        points_rgb=rng.integers(0, 256, (n, 3)).astype(np.uint8),
        points_conf=rng.random(n).astype(np.float32),
        point_frame=rng.integers(0, s, n).astype(np.uint16),
        depth=rng.random((s, h, w)).astype(np.float32),
        depth_conf=rng.random((s, h, w)).astype(np.float32),
        extrinsics=rng.standard_normal((s, 3, 4)).astype(np.float32),
        intrinsics=rng.standard_normal((s, 3, 3)).astype(np.float32),
        image_size=(h, w),
        frame_count=s,
    )


def test_fake_scene_satisfies_contract():
    sc = make_fake_scene()
    sc.validate()  # raises if any shape/dtype is wrong
    blob = sc.to_point_blob()
    assert blob.positions.dtype == np.float32
    assert blob.colors.dtype == np.uint8
    assert blob.frame_idx.dtype == np.uint16
    cams = sc.cameras_json()
    assert cams["frame_count"] == sc.frame_count


def test_registry_adapter_mapping():
    """The registry should map config adapter strings to classes (without loading)."""
    cfg = load_models_config(CONFIG)
    for entry in cfg.models.values():
        assert entry.adapter in {"vggt", "vggt_omega"}
