"""SceneResult validation + npz round-trip + cameras json."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared.scene import SceneResult  # noqa: E402


def _scene(s=3, h=8, w=10, n=50):
    rng = np.random.default_rng(1)
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
        meta={"model": "test"},
    )


def test_validate_ok():
    _scene().validate()


def test_validate_rejects_bad_shape():
    sc = _scene()
    sc.extrinsics = np.zeros((sc.frame_count, 4, 4), np.float32)
    try:
        sc.validate()
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_npz_roundtrip(tmp_path):
    sc = _scene()
    p = str(tmp_path / "arrays.npz")
    sc.save_npz(p)
    back = SceneResult.load_npz(p)
    np.testing.assert_array_equal(back.points_xyz, sc.points_xyz)
    np.testing.assert_array_equal(back.extrinsics, sc.extrinsics)
    assert back.image_size == sc.image_size
    assert back.frame_count == sc.frame_count
    assert back.meta["model"] == "test"


def test_cameras_json_shape():
    sc = _scene()
    cj = sc.cameras_json()
    assert cj["frame_count"] == sc.frame_count
    assert len(cj["cameras"]) == sc.frame_count
    assert len(cj["cameras"][0]["extrinsic"]) == 3
    assert len(cj["cameras"][0]["extrinsic"][0]) == 4
