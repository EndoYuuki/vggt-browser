from __future__ import annotations

import os
import uuid

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from ..config import MAX_UPLOAD_MB, UPLOADS_DIR, models
from ..services import ingest, queue

router = APIRouter(prefix="/api", tags=["jobs"])


@router.post("/jobs")
async def create_job(
    files: list[UploadFile] = File(...),
    model: str = Form(None),
    frames: int = Form(None),
    resolution: int = Form(None),
    fps: float = Form(2.0),
    conf_threshold: float = Form(0.1),
    point_source: str = Form("depth"),
) -> dict:
    cfg = models()
    try:
        entry = cfg.get(model)
    except KeyError as e:
        raise HTTPException(400, str(e))

    caps = entry.caps
    max_frames = caps.clamp_frames(frames) if frames else caps.default_frames
    res = caps.clamp_resolution(resolution) if resolution else caps.recommended_resolution

    job_id = uuid.uuid4().hex
    job_dir = os.path.join(UPLOADS_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    # Save uploads to disk (shared volume the worker reads).
    saved: list[str] = []
    limit_bytes = MAX_UPLOAD_MB * 1024 * 1024
    total = 0
    for uf in files:
        dest = os.path.join(job_dir, os.path.basename(uf.filename or "file"))
        with open(dest, "wb") as out:
            while chunk := await uf.read(1024 * 1024):
                total += len(chunk)
                if total > limit_bytes:
                    raise HTTPException(413, f"upload exceeds {MAX_UPLOAD_MB} MB")
                out.write(chunk)
        saved.append(dest)

    try:
        ingested = ingest.prepare_inputs(
            job_dir, saved, fps=fps, max_frames=max_frames
        )
    except ValueError as e:
        raise HTTPException(400, str(e))

    ps = point_source if point_source in ("depth", "pointmap") else "depth"
    queue.enqueue_inference(
        job_id=job_id,
        image_paths=ingested.image_paths,
        model_name=entry.name,
        resolution=res,
        conf_threshold=conf_threshold,
        point_source=ps,
    )

    return {
        "job_id": job_id,
        "model": entry.name,
        "frame_count": len(ingested.image_paths),
        "resolution": res,
        "source_kind": ingested.source_kind,
        "point_source": ps,
    }


@router.get("/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    state = queue.get_job_state(job_id)
    if state is None:
        raise HTTPException(404, "unknown job")
    return {"job_id": job_id, **state}


@router.delete("/jobs/{job_id}")
def delete_job(job_id: str) -> dict:
    ok = queue.cancel_job(job_id)
    return {"job_id": job_id, "cancelled": ok}
