"""RQ enqueue wrappers + job state access (torch-free)."""

from __future__ import annotations

import json
from typing import Any

import redis
from rq import Queue

from ..config import JOB_TIMEOUT_SECONDS, REDIS_URL, RESULT_TTL_SECONDS, RQ_QUEUE

_conn = redis.Redis.from_url(REDIS_URL)
_queue = Queue(RQ_QUEUE, connection=_conn)

# The worker imports this same dotted path; api references it by string so it
# never imports the worker module (which pulls torch).
JOB_FUNC = "worker.jobs.run_inference_job"


def enqueue_inference(
    job_id: str,
    image_paths: list[str],
    model_name: str,
    resolution: int,
    conf_threshold: float,
    point_source: str = "depth",
) -> str:
    job = _queue.enqueue(
        JOB_FUNC,
        kwargs=dict(
            job_id=job_id,
            image_paths=image_paths,
            model_name=model_name,
            resolution=resolution,
            conf_threshold=conf_threshold,
            point_source=point_source,
        ),
        job_id=job_id,
        job_timeout=JOB_TIMEOUT_SECONDS,
        result_ttl=RESULT_TTL_SECONDS,
        failure_ttl=RESULT_TTL_SECONDS,
    )
    set_job_state(job_id, {"status": "queued", "stage": "queued", "progress": 0.0})
    return job.id


def set_job_state(job_id: str, state: dict[str, Any]) -> None:
    _conn.hset(f"job:{job_id}", mapping={k: json.dumps(v) for k, v in state.items()})


def get_job_state(job_id: str) -> dict[str, Any] | None:
    raw = _conn.hgetall(f"job:{job_id}")
    if not raw:
        return None
    return {k.decode(): json.loads(v) for k, v in raw.items()}


def cancel_job(job_id: str) -> bool:
    from rq.job import Job

    try:
        job = Job.fetch(job_id, connection=_conn)
        job.cancel()
        set_job_state(job_id, {"status": "cancelled", "stage": "cancelled", "progress": 0.0})
        return True
    except Exception:
        # job already finished, gone, or unknown id — nothing to cancel
        return False


def redis_conn() -> "redis.Redis":
    return _conn
