# api: FastAPI + ffmpeg. torch-free (numpy only).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY api/requirements.txt /app/api/requirements.txt
RUN pip install -r /app/api/requirements.txt

COPY shared /app/shared
COPY api /app/api

ENV PYTHONPATH=/app MODELS_CONFIG=/config/models.yaml

CMD ["uvicorn", "api.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
