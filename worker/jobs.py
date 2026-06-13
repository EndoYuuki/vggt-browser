"""The inference job executed by the RQ worker. Runs in the GPU process where the
model registry is resident."""

from __future__ import annotations

import json
import os
import time
import traceback
from typing import Any

import redis

from shared.scene import SceneResult

RESULTS_DIR = os.environ.get("RESULTS_DIR", "/results")
REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379/0")
RESULT_TTL_SECONDS = int(os.environ.get("RESULT_TTL_SECONDS", str(24 * 3600)))

# Single global registry, populated by run.py at worker boot (model stays resident).
REGISTRY = None  # type: ignore


def _redis() -> "redis.Redis":
    return redis.Redis.from_url(REDIS_URL)


def _publish(r: "redis.Redis", job_id: str, event: dict[str, Any]) -> None:
    """Update job hash + publish a progress event for the WS bridge."""
    event = {**event, "job_id": job_id, "ts": time.time()}
    r.hset(f"job:{job_id}", mapping={k: json.dumps(v) for k, v in event.items()})
    r.publish(f"job:{job_id}:events", json.dumps(event))


def _set_registry(registry) -> None:
    global REGISTRY
    REGISTRY = registry


def run_inference_job(
    job_id: str,
    image_paths: list[str],
    model_name: str,
    resolution: int,
    conf_threshold: float = 0.1,
    point_source: str = "depth",
) -> dict[str, Any]:
    """Enqueued by the api. Returns a small dict; full output is written to disk."""
    r = _redis()
    sweep_expired_results()
    out_dir = os.path.join(RESULTS_DIR, job_id)
    os.makedirs(out_dir, exist_ok=True)

    def progress(stage: str, frac: float) -> None:
        _publish(r, job_id, {"status": "running", "stage": stage, "progress": frac})

    try:
        _publish(r, job_id, {"status": "running", "stage": "load_model", "progress": 0.02})
        assert REGISTRY is not None, "registry not initialized in worker"
        adapter = REGISTRY.get(model_name)

        # Inference with OOM auto-downgrade: on CUDA OOM, halve the resolution
        # (down to a floor) and retry. Keeps the model resident; never wedges GPU.
        res = int(resolution)
        attempt = 0
        min_res = 128
        # Fault injection for testing the OOM path without exhausting VRAM:
        # FAULT_OOM_UNTIL_RES=N raises OOM until resolution drops to <= N.
        fault_until = int(os.environ.get("FAULT_OOM_UNTIL_RES", "0"))
        result: SceneResult | None = None
        while result is None:
            try:
                if fault_until and res > fault_until:
                    raise RuntimeError("CUDA out of memory (injected fault)")
                result = adapter.run(
                    image_paths,
                    resolution=res,
                    conf_threshold=conf_threshold,
                    progress_cb=progress,
                    extra={"point_source": point_source},
                )
            except Exception as exc:  # noqa: BLE001
                if not _is_oom(exc) or res <= min_res:
                    raise
                _free_cuda()
                new_res = max(min_res, res // 2)
                attempt += 1
                _publish(
                    r,
                    job_id,
                    {
                        "status": "running",
                        "stage": f"oom_retry (res {res}->{new_res})",
                        "progress": 0.05,
                    },
                )
                res = new_res

        _publish(r, job_id, {"status": "running", "stage": "serialize", "progress": 0.95})
        npz_path = os.path.join(out_dir, "arrays.npz")
        result.save_npz(npz_path)
        _write_glb(result, os.path.join(out_dir, "scene.glb"))

        meta = {
            "status": "done",
            "stage": "done",
            "progress": 1.0,
            "model": model_name,
            "frame_count": result.frame_count,
            "point_count": int(result.points_xyz.shape[0]),
            "image_size": list(result.image_size),
            "resolution": res,
            "downgraded": attempt > 0,
        }
        _publish(r, job_id, meta)
        return meta

    except Exception as exc:  # noqa: BLE001
        is_oom = _is_oom(exc)
        _free_cuda()
        _publish(
            r,
            job_id,
            {
                "status": "failed_oom" if is_oom else "failed",
                "stage": "error",
                "progress": 0.0,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise


def _is_oom(exc: Exception) -> bool:
    return (
        exc.__class__.__name__ == "OutOfMemoryError"
        or "out of memory" in str(exc).lower()
        or "CUDA_ERROR_OUT_OF_MEMORY" in str(exc)
    )


def _free_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass  # best-effort cleanup; never raise from here


def _write_glb(result: SceneResult, path: str) -> None:
    """Export the point cloud as a downloadable GLB."""
    try:
        import numpy as np
        import trimesh

        if result.points_xyz.shape[0] == 0:
            return  # nothing to export

        cloud = trimesh.PointCloud(
            vertices=result.points_xyz.astype("float64"),
            colors=np.concatenate(
                [result.points_rgb, np.full((result.points_rgb.shape[0], 1), 255, np.uint8)],
                axis=1,
            ),
        )
        scene = trimesh.Scene()
        scene.add_geometry(cloud)
        scene.export(path)
    except Exception:
        # GLB is a non-critical download artifact; never fail the job over it.
        traceback.print_exc()


def sweep_expired_results() -> int:
    """Delete result dirs older than RESULT_TTL_SECONDS. Returns count removed."""
    import shutil
    import time

    if not os.path.isdir(RESULTS_DIR):
        return 0
    now = time.time()
    removed = 0
    for name in os.listdir(RESULTS_DIR):
        path = os.path.join(RESULTS_DIR, name)
        if not os.path.isdir(path):
            continue
        try:
            age = now - os.path.getmtime(path)
            if age > RESULT_TTL_SECONDS:
                shutil.rmtree(path, ignore_errors=True)
                removed += 1
        except OSError:
            continue
    if removed:
        print(f"[cleanup] removed {removed} expired result dir(s)")
    return removed
