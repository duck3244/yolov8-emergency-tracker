"""WebSocket — counts/fps/alert push (2Hz throttle)."""
from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from app.auth import _expected_token

logger = logging.getLogger(__name__)
router = APIRouter()


@router.websocket("/ws/state")
async def ws_state(websocket: WebSocket, token: str | None = Query(default=None)):
    expected = _expected_token()
    if expected and token != expected:
        await websocket.close(code=1008)
        return

    await websocket.accept()
    service = websocket.app.state.video_service
    try:
        while True:
            state = service.get_state()
            await websocket.send_json(state)
            await asyncio.sleep(0.5)  # 2Hz
    except WebSocketDisconnect:
        return
    except Exception:
        logger.exception("WebSocket loop failed")
        try:
            await websocket.close()
        except Exception:
            pass
