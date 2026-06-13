# GPU inference worker: torch + CUDA + VGGT (+ VGGT-Omega later).
# Base image ships CUDA-enabled torch matching modern GPUs (RTX 4090 = sm_89).
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# System libs needed by opencv-headless / trimesh
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps first (better layer caching)
COPY worker/requirements.txt /app/worker/requirements.txt
RUN pip install -r /app/worker/requirements.txt

# Install VGGT (no deps so it doesn't downgrade the base torch)
RUN pip install --no-deps "git+https://github.com/facebookresearch/vggt.git"

# Install VGGT-Omega (no deps; checkpoints are gated and fetched at runtime)
RUN pip install --no-deps "git+https://github.com/facebookresearch/vggt-omega.git"

# Project code
COPY shared /app/shared
COPY worker /app/worker
COPY samples /app/samples

ENV PYTHONPATH=/app \
    MODELS_CONFIG=/config/models.yaml \
    SAMPLE_DIR=/app/samples

# Default: run the RQ worker (warms the model at boot). Override for smoke test.
CMD ["python", "-m", "worker.run"]
