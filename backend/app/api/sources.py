"""영상 소스 목록·선택."""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request

from app.auth import require_token
from app.schemas import (
    ActionResponse,
    SelectSourceRequest,
    SourceItem,
    SourcesResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def _dataset_dir() -> Path:
    return Path(__file__).resolve().parent.parent.parent / "dataset"


@router.get("/api/sources", response_model=SourcesResponse,
            dependencies=[Depends(require_token)])
async def list_sources():
    items = [SourceItem(id="webcam", label="🎥 Webcam", kind="webcam")]
    d = _dataset_dir()
    if d.exists():
        for p in sorted(d.iterdir()):
            if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
                items.append(
                    SourceItem(
                        id=p.name,
                        label=p.name,
                        kind="file",
                        size_bytes=p.stat().st_size,
                    )
                )
    return SourcesResponse(items=items)


@router.post("/api/sources/select", response_model=ActionResponse,
             dependencies=[Depends(require_token)])
async def select_source(request: Request, payload: SelectSourceRequest):
    service = request.app.state.video_service
    try:
        service.start(payload.source_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Failed to start source")
        raise HTTPException(status_code=500, detail=str(e))
    return ActionResponse(ok=True, message=f"Started: {payload.source_id}")
