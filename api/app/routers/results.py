from __future__ import annotations

import os

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response

from ..config import RESULTS_DIR
from ..services import serialize

router = APIRouter(prefix="/api", tags=["results"])


def _scene_or_404(job_id: str):
    scene = serialize.load_scene(RESULTS_DIR, job_id)
    if scene is None:
        raise HTTPException(404, "result not ready or unknown job")
    return scene


@router.get("/jobs/{job_id}/points")
def get_points(job_id: str) -> Response:
    scene = _scene_or_404(job_id)
    return Response(
        content=serialize.points_binary(scene),
        media_type="application/octet-stream",
    )


@router.get("/jobs/{job_id}/cameras")
def get_cameras(job_id: str) -> JSONResponse:
    scene = _scene_or_404(job_id)
    return JSONResponse(serialize.cameras_json(scene))


@router.get("/jobs/{job_id}/depth/{frame}")
def get_depth(job_id: str, frame: int) -> Response:
    return _png_for(job_id, frame, "depth")


@router.get("/jobs/{job_id}/conf/{frame}")
def get_conf(job_id: str, frame: int) -> Response:
    return _png_for(job_id, frame, "conf")


def _png_for(job_id: str, frame: int, kind: str) -> Response:
    cached = os.path.join(RESULTS_DIR, job_id, "png", f"{kind}_{frame}.png")
    if os.path.exists(cached):
        return FileResponse(cached, media_type="image/png")
    scene = _scene_or_404(job_id)
    if frame < 0 or frame >= scene.frame_count:
        raise HTTPException(404, "frame out of range")
    data = serialize.depth_png(scene, frame) if kind == "depth" else serialize.conf_png(scene, frame)
    serialize.cache_png(RESULTS_DIR, job_id, f"{kind}_{frame}.png", data)
    return Response(content=data, media_type="image/png")


@router.get("/jobs/{job_id}/glb")
def get_glb(job_id: str) -> FileResponse:
    path = os.path.join(RESULTS_DIR, job_id, "scene.glb")
    if not os.path.exists(path):
        raise HTTPException(404, "glb not available")
    return FileResponse(path, media_type="model/gltf-binary", filename=f"{job_id}.glb")
