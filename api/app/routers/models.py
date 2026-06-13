from __future__ import annotations

from fastapi import APIRouter

from ..config import models

router = APIRouter(prefix="/api", tags=["models"])


@router.get("/models")
def list_models() -> dict:
    cfg = models()
    return {
        "default": cfg.default,
        "models": [entry.public_dict() for entry in cfg.models.values()],
    }
