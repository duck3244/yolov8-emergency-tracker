"""API 요청·응답 Pydantic 스키마."""
from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


# --- 공통 ---


class HealthResponse(BaseModel):
    status: str
    video_running: bool
    source: Optional[str] = None
    session_id: Optional[str] = None
    device: Optional[str] = None
    uptime_seconds: float
    model_ready: bool


# --- 상태 (WS / GET /api/state) ---


class CountsModel(BaseModel):
    entered: int = 0
    exited: int = 0
    current_inside: int = 0
    area_name: str = "Building"
    max_inside_seen: int = 0


class StateModel(BaseModel):
    running: bool = False
    source: Optional[str] = None
    session_id: Optional[str] = None
    frame_count: int = 0
    fps: float = 0.0
    counts: CountsModel = Field(default_factory=CountsModel)
    alert_status: str = "normal"  # normal | warning | emergency
    last_error: Optional[str] = None


# --- 영상 소스 ---


class SourceItem(BaseModel):
    id: str
    label: str
    kind: str  # "webcam" | "file"
    size_bytes: Optional[int] = None


class SourcesResponse(BaseModel):
    items: List[SourceItem]


class SelectSourceRequest(BaseModel):
    source_id: str = Field(..., description='Either "webcam" or a dataset filename')


# --- 영역 ---


class AreaModel(BaseModel):
    entrance: List[List[int]]
    exit: List[List[int]]


class PutAreasRequest(BaseModel):
    """둘 다 보내면 동시에 갱신, 한 쪽만 보내면 그 쪽만 갱신한다.

    클라이언트가 부분 업데이트를 원할 때(예: exit 위치만 조정) 다른 폴리곤을
    먼저 GET 해와서 다시 보내야 하는 번거로움을 없애기 위함.
    """

    entrance: Optional[List[List[int]]] = None
    exit: Optional[List[List[int]]] = None
    reset_counts: bool = True


# --- 설정 ---


class ConfigPatch(BaseModel):
    """`/api/config` PUT — 부분 업데이트. 변경하지 않을 키는 생략."""

    confidence_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    iou_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    device: Optional[str] = None
    frame_skip: Optional[int] = Field(None, ge=1)
    distance_threshold: Optional[int] = Field(None, ge=1)
    max_disappeared: Optional[int] = Field(None, ge=1)
    overcrowding_threshold: Optional[int] = Field(None, ge=1)
    warning_threshold: Optional[int] = Field(None, ge=1)
    notification_interval: Optional[int] = Field(None, ge=1)
    enable_email: Optional[bool] = None
    emergency_contacts: Optional[List[str]] = None
    min_residence_time: Optional[float] = Field(None, ge=0.0)


class ConfigSummary(BaseModel):
    confidence_threshold: float
    iou_threshold: float
    device: str
    frame_skip: int
    frame_width: int
    frame_height: int
    distance_threshold: int
    max_disappeared: int
    min_residence_time: float
    area_name: str
    overcrowding_threshold: int
    warning_threshold: int
    notification_interval: int
    enable_email: bool
    emergency_contacts: List[str]
    email_configured: bool  # password가 메모리에 있는지
    location_name: str


# --- 액션 ---


class ActionResponse(BaseModel):
    ok: bool
    message: Optional[str] = None
