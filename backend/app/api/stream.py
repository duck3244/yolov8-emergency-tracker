"""MJPEG 스트림 + 단일 스냅샷."""
from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import StreamingResponse

from app.auth import require_token

logger = logging.getLogger(__name__)
router = APIRouter()

BOUNDARY = "frame"


@router.get("/stream", dependencies=[Depends(require_token)])
async def mjpeg_stream(request: Request):
    """MJPEG (multipart/x-mixed-replace). 브라우저 <img src=/stream> 으로 표시 가능."""
    service = request.app.state.video_service

    async def generator():
        while True:
            if await request.is_disconnected():
                break
            jpeg = service.get_latest_jpeg()
            if jpeg is not None:
                chunk = (
                    f"--{BOUNDARY}\r\n"
                    "Content-Type: image/jpeg\r\n"
                    f"Content-Length: {len(jpeg)}\r\n\r\n"
                ).encode("ascii") + jpeg + b"\r\n"
                yield chunk
            await asyncio.sleep(1 / 15)

    return StreamingResponse(
        generator(),
        media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}",
    )


@router.get("/api/snapshot", dependencies=[Depends(require_token)])
async def snapshot(request: Request):
    """현재 프레임 1장만 (영역 편집기 용)."""
    service = request.app.state.video_service
    jpeg = service.get_latest_jpeg()
    if jpeg is None:
        raise HTTPException(status_code=404, detail="No frame available yet")
    return Response(content=jpeg, media_type="image/jpeg")
