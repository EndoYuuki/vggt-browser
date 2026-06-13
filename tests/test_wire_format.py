"""Round-trip the binary point container. Mirrors the JS parser in web/src/api/client.ts."""

import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared import wire_format as wf  # noqa: E402


def _sample(n=1000):
    rng = np.random.default_rng(0)
    return wf.PointBlob(
        positions=rng.standard_normal((n, 3)).astype(np.float32),
        colors=rng.integers(0, 256, (n, 3)).astype(np.uint8),
        conf=rng.random(n).astype(np.float32),
        frame_idx=rng.integers(0, 30, n).astype(np.uint16),
    )


def test_pack_unpack_roundtrip():
    blob = _sample()
    data = wf.pack_points(blob)
    back = wf.unpack_points(data)
    assert back.count == blob.count
    np.testing.assert_array_equal(back.positions, blob.positions)
    np.testing.assert_array_equal(back.colors, blob.colors)
    np.testing.assert_allclose(back.conf, blob.conf)
    np.testing.assert_array_equal(back.frame_idx, blob.frame_idx)


def test_header_layout_matches_js_parser():
    """Verify the exact byte offsets the JS DataView parser reads."""
    blob = _sample(n=10)
    data = wf.pack_points(blob)
    assert data[:4] == wf.MAGIC
    # JS reads magic as big-endian int and compares to 0x56474742
    magic_be = struct.unpack(">I", data[:4])[0]
    assert magic_be == 0x56474742
    n = struct.unpack_from("<I", data, 8)[0]
    assert n == 10
    pos_off = struct.unpack_from("<I", data, 12)[0]
    assert pos_off == wf.HEADER_SIZE


def test_empty_cloud():
    blob = wf.PointBlob(
        positions=np.zeros((0, 3), np.float32),
        colors=np.zeros((0, 3), np.uint8),
        conf=np.zeros((0,), np.float32),
        frame_idx=np.zeros((0,), np.uint16),
    )
    back = wf.unpack_points(wf.pack_points(blob))
    assert back.count == 0
