"""Adapter for facebook/VGGT-1B."""

from __future__ import annotations

from typing import Any

import numpy as np

from shared.scene import SceneResult

from .base import ProgressCb, ReconstructionAdapter

_DTYPE_MAP = {}  # lazily filled with torch dtypes inside load()


class VGGTAdapter(ReconstructionAdapter):
    def __init__(self, entry) -> None:
        super().__init__(entry)
        self._model = None
        self._device = "cuda"
        self._torch_dtype = None

    # -- lifecycle ----------------------------------------------------------
    def load(self, device: str, dtype: str) -> None:
        if self._loaded:
            return
        import torch
        from vggt.models.vggt import VGGT

        self._device = device
        self._torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }.get(dtype, torch.bfloat16)

        hf_repo = self.entry.options.get("hf_repo", "facebook/VGGT-1B")
        model = VGGT.from_pretrained(hf_repo)
        model = model.to(device).eval()
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

    # -- inference ----------------------------------------------------------
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
        from vggt.utils.load_fn import load_and_preprocess_images
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        assert self._model is not None, "model not loaded"
        # Point source: "depth" (unproject depth+camera, paper-recommended, higher
        # accuracy) or "pointmap" (the model's direct world_points head).
        point_source = (extra or {}).get(
            "point_source", self.entry.options.get("point_source", "depth")
        )
        self._progress(progress_cb, "preprocess", 0.05)

        # (S, 3, H, W) on device
        images = load_and_preprocess_images(image_paths).to(self._device)

        self._progress(progress_cb, "forward", 0.2)
        use_amp = self._device.startswith("cuda")
        with torch.no_grad():
            with torch.autocast(
                "cuda", dtype=self._torch_dtype, enabled=use_amp
            ):
                preds = self._model(images)  # batched internally; returns dict

            # Decode cameras from the pose encoding (needs image H,W).
            # preds["images"] is (1,S,3,H,W); pass last two dims.
            hw = preds["images"].shape[-2:]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(preds["pose_enc"], hw)

        self._progress(progress_cb, "normalize", 0.7)
        result = self._normalize(preds, extrinsic, intrinsic, conf_threshold, point_source)
        self._progress(progress_cb, "normalize", 0.95)
        return result

    # -- normalization to SceneResult --------------------------------------
    def _normalize(
        self, preds: dict, extrinsic, intrinsic, conf_threshold: float, point_source: str = "depth"
    ) -> SceneResult:
        import torch

        def np_(x):
            if isinstance(x, torch.Tensor):
                return x.detach().float().cpu().numpy()
            return np.asarray(x)

        # VGGT outputs carry a leading batch dim of 1; squeeze it.
        def squeeze_batch(a: np.ndarray) -> np.ndarray:
            if a.ndim >= 1 and a.shape[0] == 1:
                return a[0]
            return a

        extrinsic = squeeze_batch(np_(extrinsic))  # (S, 3, 4)
        intrinsic = squeeze_batch(np_(intrinsic))  # (S, 3, 3)
        depth_raw = squeeze_batch(np_(preds["depth"]))      # (S, H, W, 1) or (S,H,W)
        depth_conf = squeeze_batch(np_(preds["depth_conf"]))  # (S, H, W)

        # 2D-display depth with the channel dim stripped.
        depth = depth_raw[..., 0] if (depth_raw.ndim == 4 and depth_raw.shape[-1] == 1) else depth_raw

        if point_source == "depth":
            # Paper-recommended: unproject depth + camera. Uses VGGT's official
            # geometry util for correctness. Confidence comes from depth_conf.
            # The util expects depth_map (S,H,W,1) — keep the channel dim.
            from vggt.utils.geometry import unproject_depth_map_to_point_map

            depth_in = depth_raw if depth_raw.ndim == 4 else depth_raw[..., None]
            point_map = np.asarray(
                unproject_depth_map_to_point_map(depth_in, extrinsic, intrinsic)
            )
            point_conf = depth_conf
        else:  # "pointmap": the model's direct world_points head
            point_map = squeeze_batch(np_(preds["world_points"]))      # (S, H, W, 3)
            point_conf = squeeze_batch(np_(preds["world_points_conf"]))  # (S, H, W)

        s, h, w = point_map.shape[0], point_map.shape[1], point_map.shape[2]

        # RGB from the preprocessed images: preds["images"] is (1,S,3,H,W) in [0,1]
        imgs = np_(preds["images"])
        if imgs.ndim == 5:
            imgs = imgs[0]
        # resize-match guard: images H/W should equal point_map H/W
        rgb = np.transpose(imgs, (0, 2, 3, 1))  # (S,H,W,3)
        rgb = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)

        # Flatten per-pixel points
        pts = point_map.reshape(-1, 3)
        conf_raw = point_conf.reshape(-1).astype(np.float32)
        cols = rgb.reshape(-1, 3)
        frame_ids = np.repeat(np.arange(s, dtype=np.uint16), h * w)

        # Also drop non-finite points (NaN/Inf depth can sneak in).
        finite = np.isfinite(pts).all(axis=1) & np.isfinite(conf_raw)

        # conf_threshold is a QUANTILE fraction (0..1): drop the lowest X% by
        # confidence. Robust regardless of the raw score range (VGGT conf is
        # an unbounded score, ~1..30, not 0..1). This matches VGGT's demo.
        keep_quantile = float(np.clip(conf_threshold, 0.0, 0.99))
        if finite.any() and keep_quantile > 0:
            cutoff = np.quantile(conf_raw[finite], keep_quantile)
        else:
            cutoff = -np.inf
        mask = finite & (conf_raw >= cutoff)

        flat_pts = pts[mask].astype(np.float32)
        flat_cols = cols[mask].astype(np.uint8)
        flat_frame = frame_ids[mask].astype(np.uint16)
        # Rescale kept confidence to 0..1 for frontend coloring / live filter.
        flat_conf = self._rescale01(conf_raw[mask]).astype(np.float32)

        # depth_conf for the 2D panels, rescaled to 0..1 per-frame for display.
        depth_conf_disp = np.stack(
            [self._rescale01(depth_conf[i].reshape(-1)).reshape(h, w) for i in range(s)]
        ).astype(np.float32)

        return SceneResult(
            points_xyz=flat_pts,
            points_rgb=flat_cols,
            points_conf=flat_conf,
            point_frame=flat_frame,
            depth=depth.astype(np.float32),
            depth_conf=depth_conf_disp,
            extrinsics=extrinsic.astype(np.float32),
            intrinsics=intrinsic.astype(np.float32),
            image_size=(h, w),
            frame_count=s,
            meta={
                "model": self.name,
                "adapter": "vggt",
                "point_source": point_source,
                "raw_points": int(s * h * w),
                "kept_points": int(flat_pts.shape[0]),
                "conf_quantile": keep_quantile,
            },
        )

    @staticmethod
    def _rescale01(conf: np.ndarray) -> np.ndarray:
        """Rescale a confidence score array to 0..1 (percentile 2..98) for display."""
        c = conf.astype(np.float32)
        if c.size == 0:
            return c
        lo = np.percentile(c, 2)
        hi = np.percentile(c, 98)
        if hi <= lo:
            return np.zeros_like(c)
        return np.clip((c - lo) / (hi - lo), 0.0, 1.0)

    # backward-compat alias used by the Omega adapter
    _normalize_conf = _rescale01
