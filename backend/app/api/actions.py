"""/api/actions/* — 명령형 액션."""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, Request

from app.auth import require_token
from app.schemas import ActionResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/api/actions/reset_counts", response_model=ActionResponse,
             dependencies=[Depends(require_token)])
async def reset_counts(request: Request):
    request.app.state.video_service.reset_counts()
    return ActionResponse(ok=True, message="Counts reset")


@router.post("/api/actions/stop", response_model=ActionResponse,
             dependencies=[Depends(require_token)])
async def stop(request: Request):
    request.app.state.video_service.stop()
    return ActionResponse(ok=True, message="Stopped")


@router.post("/api/actions/save_state", response_model=ActionResponse,
             dependencies=[Depends(require_token)])
async def save_state(request: Request):
    svc = request.app.state.video_service
    cm = request.app.state.config_manager
    try:
        svc.save_state_snapshot()
        out = Path(cm.video.output_path)
        out.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(out / f"session_{stamp}.json", "w", encoding="utf-8") as f:
            json.dump(svc.get_state(), f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.exception("save_state failed")
        return ActionResponse(ok=False, message=str(e))
    return ActionResponse(ok=True, message="State saved")


@router.post("/api/actions/restore_state", response_model=ActionResponse,
             dependencies=[Depends(require_token)])
async def restore_state(request: Request):
    ok = request.app.state.video_service.restore_state_snapshot()
    return ActionResponse(ok=ok, message="Restored" if ok else "No snapshot found")


@router.post("/api/actions/send_test_alert", response_model=ActionResponse,
             dependencies=[Depends(require_token)])
async def send_test_alert(request: Request):
    svc = request.app.state.video_service
    cm = request.app.state.config_manager
    if svc.notification_manager is None:
        return ActionResponse(ok=False, message="Notification manager not ready")
    location = {
        "name": cm.location.name,
        "lat": cm.location.latitude,
        "lon": cm.location.longitude,
    }
    inside = svc.get_state()["counts"]["current_inside"]
    sent = svc.notification_manager.check_and_send_alerts(
        max(inside, cm.alert.overcrowding_threshold), location, "수동 테스트 알림"
    )
    return ActionResponse(ok=bool(sent), message="Test alert dispatched" if sent else "Suppressed or unconfigured")
