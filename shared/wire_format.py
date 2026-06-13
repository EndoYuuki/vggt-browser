"""Binary wire format shared between api and worker.

This module is intentionally dependency-light (numpy only, no torch) so it can be
vendored into both the `api` image (torch-free) and the `worker` image. It defines:

- The on-disk npz key names that the worker writes and the api reads.
- The packed-binary container layout that the api serves to the browser for the
  point-cloud endpoint, and helpers to pack/unpack it.

The browser reads the binary blob with a tiny header parser and attaches the
position section directly to a three.js BufferGeometry as a Float32Array.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# npz key names (worker writes -> api reads). Keep in sync with SceneResult.
# ---------------------------------------------------------------------------
NPZ_POINTS_XYZ = "points_xyz"   # (N, 3) float32, world coords (OpenCV)
NPZ_POINTS_RGB = "points_rgb"   # (N, 3) uint8
NPZ_POINTS_CONF = "points_conf"  # (N,)  float32, normalized 0..1
NPZ_POINT_FRAME = "point_frame"  # (N,)  uint16, source frame index
NPZ_DEPTH = "depth"             # (S, H, W) float32
NPZ_DEPTH_CONF = "depth_conf"   # (S, H, W) float32
NPZ_EXTRINSICS = "extrinsics"   # (S, 3, 4) float32, world->cam (OpenCV)
NPZ_INTRINSICS = "intrinsics"   # (S, 3, 3) float32
NPZ_IMAGE_SIZE = "image_size"   # (2,) int32 -> (H, W)
NPZ_FRAME_COUNT = "frame_count"  # scalar int32


# ---------------------------------------------------------------------------
# Packed-binary container for the /points endpoint.
#
# Layout (all little-endian):
#   [ header: 64 bytes ]
#   [ positions: N * 3 * float32 ]
#   [ colors:    N * 3 * uint8   ]
#   [ conf:      N     * float32 ]
#   [ frameIdx:  N     * uint16  ]
#
# Header (64 bytes):
#   offset size field
#   0      4    magic  = b"VGGB"
#   4      2    version (uint16) = 1
#   6      2    reserved (uint16)
#   8      4    point_count N (uint32)
#   12     4    pos_offset  (uint32)  byte offset of positions section
#   16     4    rgb_offset  (uint32)
#   20     4    conf_offset (uint32)
#   24     4    frame_offset(uint32)
#   28     36   reserved/padding (zeros)
# ---------------------------------------------------------------------------
MAGIC = b"VGGB"
VERSION = 1
HEADER_SIZE = 64
_HEADER_STRUCT = struct.Struct("<4sHHIIIII")  # 4s,H,H,I,I,I,I,I = 28 bytes used


@dataclass
class PointBlob:
    positions: np.ndarray  # (N, 3) float32
    colors: np.ndarray     # (N, 3) uint8
    conf: np.ndarray       # (N,)   float32
    frame_idx: np.ndarray  # (N,)   uint16

    @property
    def count(self) -> int:
        return int(self.positions.shape[0])


def pack_points(blob: PointBlob) -> bytes:
    """Pack a PointBlob into the binary container served to the browser."""
    positions = np.ascontiguousarray(blob.positions, dtype="<f4")
    colors = np.ascontiguousarray(blob.colors, dtype=np.uint8)
    conf = np.ascontiguousarray(blob.conf, dtype="<f4")
    frame_idx = np.ascontiguousarray(blob.frame_idx, dtype="<u2")

    n = positions.shape[0]
    if positions.shape != (n, 3):
        raise ValueError(f"positions must be (N,3), got {positions.shape}")
    if colors.shape != (n, 3):
        raise ValueError(f"colors must be (N,3), got {colors.shape}")
    if conf.shape != (n,):
        raise ValueError(f"conf must be (N,), got {conf.shape}")
    if frame_idx.shape != (n,):
        raise ValueError(f"frame_idx must be (N,), got {frame_idx.shape}")

    pos_bytes = positions.tobytes()
    rgb_bytes = colors.tobytes()
    conf_bytes = conf.tobytes()
    frame_bytes = frame_idx.tobytes()

    pos_off = HEADER_SIZE
    rgb_off = pos_off + len(pos_bytes)
    conf_off = rgb_off + len(rgb_bytes)
    frame_off = conf_off + len(conf_bytes)

    header = bytearray(HEADER_SIZE)
    _HEADER_STRUCT.pack_into(
        header, 0, MAGIC, VERSION, 0, n, pos_off, rgb_off, conf_off, frame_off
    )

    return bytes(header) + pos_bytes + rgb_bytes + conf_bytes + frame_bytes


def unpack_points(data: bytes) -> PointBlob:
    """Inverse of pack_points. Used by round-trip tests mirroring the JS parser."""
    magic, version, _reserved, n, pos_off, rgb_off, conf_off, frame_off = (
        _HEADER_STRUCT.unpack_from(data, 0)
    )
    if magic != MAGIC:
        raise ValueError(f"bad magic: {magic!r}")
    if version != VERSION:
        raise ValueError(f"unsupported version: {version}")

    positions = np.frombuffer(data, dtype="<f4", count=n * 3, offset=pos_off).reshape(n, 3)
    colors = np.frombuffer(data, dtype=np.uint8, count=n * 3, offset=rgb_off).reshape(n, 3)
    conf = np.frombuffer(data, dtype="<f4", count=n, offset=conf_off)
    frame_idx = np.frombuffer(data, dtype="<u2", count=n, offset=frame_off)

    return PointBlob(
        positions=positions.copy(),
        colors=colors.copy(),
        conf=conf.copy(),
        frame_idx=frame_idx.copy(),
    )
