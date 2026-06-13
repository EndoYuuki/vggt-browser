"""Adapter for VGGT-Omega.

Omega's output schema differs from VGGT: it returns pose_enc + depth + depth_conf
(no point_map). Cameras are decoded via encoding_to_camera and depth is unprojected
to a world-space point map here, then normalized to the same SceneResult contract.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from shared.scene import SceneResult

from .base import ProgressCb, ReconstructionAdapter
from .vggt_adapter import VGGTAdapter  # reuse _normalize_conf


def _unproject(depth: np.ndarray, extr: np.ndarray, intr: np.ndarray) -> np.ndarray:
    """depth (H,W) + extrinsic (3,4) world->cam + intrinsic (3,3) -> world points (H,W,3)."""
    h, w = depth.shape
    ys, xs = np.mgrid[0:h, 0:w]
    fx, fy = intr[0, 0], intr[1, 1]
    cx, cy = intr[0, 2], intr[1, 2]
    x_cam = (xs - cx) / fx * depth
    y_cam = (ys - cy) / fy * depth
    z_cam = depth
    pts_cam = np.stack([x_cam, y_cam, z_cam], axis=-1).reshape(-1, 3)  # (HW,3)
    # world->cam is [R|t]; cam->world: X_w = R^T (X_c - t)
    R = extr[:, :3]
    t = extr[:, 3]
    pts_world = (pts_cam - t) @ R  # (HW,3) since (R^T x) == x @ R for row vecs
    return pts_world.reshape(h, w, 3)


class VGGTOmegaAdapter(ReconstructionAdapter):
    def __init__(self, entry) -> None:
        super().__init__(entry)
        self._model = None
        self._device = "cuda"
        self._torch_dtype = None
        self._resolution = int(entry.options.get("image_resolution", 512))

    def load(self, device: str, dtype: str) -> None:
        if self._loaded:
            return
        import torch
        from vggt_omega.models import VGGTOmega

        self._device = device
        self._torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }.get(dtype, torch.bfloat16)

        model = VGGTOmega().to(device).eval()

        # Resolve checkpoint: explicit local path wins, else download from the
        # gated HF repo (facebook/VGGT-Omega) by filename.
        ckpt_path = self.entry.options.get("checkpoint_path") or os.environ.get(
            "OMEGA_CHECKPOINT_PATH"
        )
        if not (ckpt_path and os.path.exists(ckpt_path)):
            repo = self.entry.options.get("hf_repo", "facebook/VGGT-Omega")
            filename = self.entry.options.get("checkpoint_filename")
            if not filename:
                raise RuntimeError("Omega: set checkpoint_filename in models.yaml")
            from huggingface_hub import hf_hub_download

            ckpt_path = hf_hub_download(repo_id=repo, filename=filename)

        state = torch.load(ckpt_path, map_location="cpu")
        # checkpoints may wrap weights under common keys
        for key in ("model", "state_dict", "ema"):
            if isinstance(state, dict) and key in state and isinstance(state[key], dict):
                state = state[key]
                break
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"[omega] {len(missing)} missing keys (e.g. {missing[:3]})")
        if unexpected:
            print(f"[omega] {len(unexpected)} unexpected keys (e.g. {unexpected[:3]})")
        self._model = model
        self._loaded = True

    def unload(self) -> None:
        if not self._loaded:
            return
        import torch

        del self._model
        self._model = None
        self._loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def run(
        self,
        image_paths: list[str],
        *,
        resolution: int,
        conf_threshold: float = 0.0,
        progress_cb: ProgressCb | None = None,
        extra: dict[str, Any] | None = None,
    ) -> SceneResult:
        import torch
        from vggt_omega.utils.load_fn import load_and_preprocess_images
        from vggt_omega.utils.pose_enc import encoding_to_camera

        assert self._model is not None, "model not loaded"
        self._progress(progress_cb, "preprocess", 0.05)
        res = resolution or self._resolution
        images = load_and_preprocess_images(image_paths, image_resolution=res).to(
            self._device
        )

        self._progress(progress_cb, "forward", 0.2)
        with torch.inference_mode():
            with torch.autocast(
                "cuda", dtype=self._torch_dtype, enabled=self._device.startswith("cuda")
            ):
                preds = self._model(images)

        self._progress(progress_cb, "normalize", 0.7)
        extr, intr = encoding_to_camera(preds["pose_enc"], preds["images"].shape[-2:])
        result = self._normalize(preds, extr, intr, conf_threshold)
        self._progress(progress_cb, "normalize", 0.95)
        return result

    def _normalize(self, preds, extr, intr, conf_threshold: float) -> SceneResult:
        import torch

        def np_(x):
            if isinstance(x, torch.Tensor):
                return x.detach().float().cpu().numpy()
            return np.asarray(x)

        def sq(a):
            return a[0] if (a.ndim >= 1 and a.shape[0] == 1) else a

        extrinsic = sq(np_(extr))   # (S,3,4)
        intrinsic = sq(np_(intr))   # (S,3,3)
        depth = sq(np_(preds["depth"]))
        if depth.ndim == 4 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        depth_conf = sq(np_(preds["depth_conf"]))
        imgs = sq(np_(preds["images"]))  # (S,3,H,W)

        s, h, w = depth.shape
        rgb = (np.clip(np.transpose(imgs, (0, 2, 3, 1)), 0, 1) * 255).astype(np.uint8)

        # Unproject all frames -> world points, flatten.
        pts = np.concatenate(
            [_unproject(depth[i], extrinsic[i], intrinsic[i]).reshape(-1, 3) for i in range(s)]
        )
        cols = rgb.reshape(-1, 3)
        conf_raw = depth_conf.reshape(-1).astype(np.float32)
        frame_ids = np.repeat(np.arange(s, dtype=np.uint16), h * w)

        finite = np.isfinite(pts).all(axis=1) & np.isfinite(conf_raw)
        # conf_threshold is a quantile fraction: drop lowest X% (see VGGT adapter).
        keep_quantile = float(np.clip(conf_threshold, 0.0, 0.99))
        if finite.any() and keep_quantile > 0:
            cutoff = np.quantile(conf_raw[finite], keep_quantile)
        else:
            cutoff = -np.inf
        mask = finite & (conf_raw >= cutoff)

        flat_conf = VGGTAdapter._rescale01(conf_raw[mask])
        depth_conf_disp = np.stack(
            [VGGTAdapter._rescale01(depth_conf[i].reshape(-1)).reshape(h, w) for i in range(s)]
        ).astype(np.float32)

        return SceneResult(
            points_xyz=pts[mask].astype(np.float32),
            points_rgb=cols[mask].astype(np.uint8),
            points_conf=flat_conf.astype(np.float32),
            point_frame=frame_ids[mask].astype(np.uint16),
            depth=depth.astype(np.float32),
            depth_conf=depth_conf_disp,
            extrinsics=extrinsic.astype(np.float32),
            intrinsics=intrinsic.astype(np.float32),
            image_size=(h, w),
            frame_count=s,
            meta={
                "model": self.name,
                "adapter": "vggt_omega",
                "raw_points": int(s * h * w),
                "kept_points": int(mask.sum()),
                "conf_quantile": keep_quantile,
            },
        )
