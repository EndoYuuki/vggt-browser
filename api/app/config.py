"""api settings + model config access (torch-free)."""

from __future__ import annotations

import os

from shared.model_config import ModelsConfig, load_models_config

REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379/0")
RQ_QUEUE = os.environ.get("RQ_QUEUE", "inference")
UPLOADS_DIR = os.environ.get("UPLOADS_DIR", "/uploads")
RESULTS_DIR = os.environ.get("RESULTS_DIR", "/results")
RESULT_TTL_SECONDS = int(os.environ.get("RESULT_TTL_SECONDS", str(24 * 3600)))
JOB_TIMEOUT_SECONDS = int(os.environ.get("JOB_TIMEOUT_SECONDS", "1800"))
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "1024"))


def models() -> ModelsConfig:
    return load_models_config()
