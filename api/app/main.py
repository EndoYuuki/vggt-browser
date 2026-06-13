from __future__ import annotations

import os

from fastapi import FastAPI

from .config import RESULTS_DIR, UPLOADS_DIR
from .routers import jobs, models, results, ws


def create_app() -> FastAPI:
    os.makedirs(UPLOADS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    app = FastAPI(title="vggt-browser api")

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok"}

    app.include_router(models.router)
    app.include_router(jobs.router)
    app.include_router(results.router)
    app.include_router(ws.router)
    return app


app = create_app()
