"""/api/config — 부분 업데이트."""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Request

from app.auth import require_token
from app.schemas import ConfigPatch, ConfigSummary

logger = logging.getLogger(__name__)
router = APIRouter()


def _summary(cm) -> ConfigSummary:
    return ConfigSummary(
        confidence_threshold=cm.model.confidence_threshold,
        iou_threshold=cm.model.iou_threshold,
        device=cm.model.device,
        frame_skip=cm.video.frame_skip,
        frame_width=cm.video.frame_width,
        frame_height=cm.video.frame_height,
        distance_threshold=cm.tracking.distance_threshold,
        max_disappeared=cm.tracking.max_disappeared,
        min_residence_time=cm.counting.min_residence_time,
        area_name=cm.counting.area_name,
        overcrowding_threshold=cm.alert.overcrowding_threshold,
        warning_threshold=cm.alert.warning_threshold,
        notification_interval=cm.alert.notification_interval,
        enable_email=cm.alert.enable_email,
        emergency_contacts=cm.alert.emergency_contacts,
        email_configured=cm.email.is_configured(),
        location_name=cm.location.name,
    )


@router.get("/api/config", response_model=ConfigSummary,
            dependencies=[Depends(require_token)])
async def get_config(request: Request):
    cm = request.app.state.config_manager
    return _summary(cm)


@router.put("/api/config", response_model=ConfigSummary,
            dependencies=[Depends(require_token)])
async def update_config(request: Request, patch: ConfigPatch):
    cm = request.app.state.config_manager
    svc = request.app.state.video_service

    data = patch.model_dump(exclude_unset=True)
    if "confidence_threshold" in data:
        cm.model.confidence_threshold = data["confidence_threshold"]
        if svc.detector is not None:
            svc.detector.confidence_threshold = data["confidence_threshold"]
    if "iou_threshold" in data:
        cm.model.iou_threshold = data["iou_threshold"]
        if svc.detector is not None:
            svc.detector.iou_threshold = data["iou_threshold"]
    if "device" in data:
        cm.model.device = data["device"]
    if "frame_skip" in data:
        cm.video.frame_skip = data["frame_skip"]
    if "distance_threshold" in data:
        cm.tracking.distance_threshold = data["distance_threshold"]
        if svc.tracker is not None:
            svc.tracker.distance_threshold = data["distance_threshold"]
    if "max_disappeared" in data:
        cm.tracking.max_disappeared = data["max_disappeared"]
        if svc.tracker is not None:
            svc.tracker.max_disappeared = data["max_disappeared"]
    if "min_residence_time" in data:
        cm.counting.min_residence_time = data["min_residence_time"]
        if svc.counter is not None:
            svc.counter.min_residence_time = data["min_residence_time"]
    if "overcrowding_threshold" in data:
        cm.alert.overcrowding_threshold = data["overcrowding_threshold"]
    if "warning_threshold" in data:
        cm.alert.warning_threshold = data["warning_threshold"]
    if "notification_interval" in data:
        cm.alert.notification_interval = data["notification_interval"]
    if "enable_email" in data:
        cm.alert.enable_email = data["enable_email"]
    if "emergency_contacts" in data:
        cm.alert.emergency_contacts = data["emergency_contacts"]

    svc.apply_thresholds()
    cm.save_config()
    return _summary(cm)
