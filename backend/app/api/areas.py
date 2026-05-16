"""/api/areas — entrance/exit 폴리곤 업데이트."""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request

from app.auth import require_token
from app.schemas import ActionResponse, AreaModel, PutAreasRequest

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/api/areas", response_model=AreaModel,
            dependencies=[Depends(require_token)])
async def get_areas(request: Request):
    cm = request.app.state.config_manager
    return AreaModel(
        entrance=cm.counting.entrance_area,
        exit=cm.counting.exit_area,
    )


@router.put("/api/areas", response_model=ActionResponse,
            dependencies=[Depends(require_token)])
async def put_areas(request: Request, payload: PutAreasRequest):
    if payload.entrance is None and payload.exit is None:
        raise HTTPException(
            status_code=400,
            detail="Provide at least one of entrance or exit",
        )
    if payload.entrance is not None and len(payload.entrance) < 3:
        raise HTTPException(
            status_code=400,
            detail="entrance polygon requires at least 3 points",
        )
    if payload.exit is not None and len(payload.exit) < 3:
        raise HTTPException(
            status_code=400,
            detail="exit polygon requires at least 3 points",
        )
    svc = request.app.state.video_service
    try:
        svc.apply_areas(
            entrance=payload.entrance,
            exit_area=payload.exit,
            reset_counts=payload.reset_counts,
        )
    except Exception as e:
        logger.exception("Failed to apply areas")
        raise HTTPException(status_code=500, detail=str(e))
    if payload.reset_counts:
        svc.reset_counts()

    updated = []
    if payload.entrance is not None:
        updated.append("entrance")
    if payload.exit is not None:
        updated.append("exit")
    return ActionResponse(ok=True, message=f"Updated: {', '.join(updated)}")
