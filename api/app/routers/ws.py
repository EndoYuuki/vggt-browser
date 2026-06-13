from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..services import queue

router = APIRouter(tags=["ws"])


@router.websocket("/ws/jobs/{job_id}")
async def job_events(ws: WebSocket, job_id: str) -> None:
    await ws.accept()
    conn = queue.redis_conn()

    # Send current state immediately so late subscribers aren't stuck waiting.
    state = queue.get_job_state(job_id)
    if state:
        await ws.send_json({"job_id": job_id, **state})
        if state.get("status") in {"done", "failed", "failed_oom", "cancelled"}:
            await ws.close()
            return

    pubsub = conn.pubsub()
    pubsub.subscribe(f"job:{job_id}:events")
    try:
        while True:
            msg = pubsub.get_message(ignore_subscribe_messages=True, timeout=0.0)
            if msg is not None:
                payload = json.loads(msg["data"])
                await ws.send_json(payload)
                if payload.get("status") in {"done", "failed", "failed_oom", "cancelled"}:
                    break
            await asyncio.sleep(0.2)
    except WebSocketDisconnect:
        pass
    finally:
        pubsub.close()
        try:
            await ws.close()
        except Exception:
            pass  # client may already be gone
