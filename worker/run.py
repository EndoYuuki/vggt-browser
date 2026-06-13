"""RQ worker entrypoint. Warms the default model (resident load) then processes
inference jobs one at a time, serializing GPU access."""

from __future__ import annotations

import os

import redis
from rq import Queue, SimpleWorker

from worker import jobs
from worker.adapters.registry import ModelRegistry

REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379/0")
QUEUE_NAME = os.environ.get("RQ_QUEUE", "inference")


def main() -> None:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[worker] device={device}")
    if device == "cuda":
        print(f"[worker] gpu={torch.cuda.get_device_name(0)}")

    registry = ModelRegistry(device=device)
    warm = os.environ.get("WARM_DEFAULT_MODEL", "1") == "1"
    if warm:
        print(f"[worker] warming default model: {registry.config.default}")
        registry.warm()
        print(f"[worker] resident model: {registry.current_name}")
    jobs._set_registry(registry)
    jobs.sweep_expired_results()  # clean stale results on boot

    conn = redis.Redis.from_url(REDIS_URL)
    queue = Queue(QUEUE_NAME, connection=conn)
    # SimpleWorker runs jobs in-process (no fork) so the resident CUDA model and
    # its memory are reused across jobs. Single worker => serialized GPU access.
    worker = SimpleWorker([queue], connection=conn)
    print(f"[worker] listening on queue '{QUEUE_NAME}'")
    worker.work(with_scheduler=False)


if __name__ == "__main__":
    main()
