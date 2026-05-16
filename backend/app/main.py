"""FastAPI 진입점.

실행:
    cd backend
    uvicorn app.main:app --host 127.0.0.1 --port 8000

운영 워커는 단일 프로세스(--workers 1)로 유지해야 한다.
VideoService가 프로세스 내 상태를 보유하므로 멀티 워커는 불가.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

# backend 디렉토리를 import path에 추가 — uvicorn을 backend/ 디렉토리에서 실행하지 않을 때 안전장치
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import actions, areas, config as config_router, sessions, sources, stream, ws
from app.auth import _expected_token
from app.schemas import HealthResponse
from app.services.video_service import VideoService
from config import ConfigManager

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    if logging.getLogger().handlers:
        return
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    _configure_logging()
    started_at = time.time()

    cm = ConfigManager("config.json")
    app.state.config_manager = cm

    if not _expected_token():
        logger.warning(
            "APP_TOKEN is not set — API is unauthenticated. "
            "Bind to 127.0.0.1 only or set APP_TOKEN before exposing on a network."
        )

    svc = VideoService(cm)
    app.state.video_service = svc
    app.state.started_at = started_at

    if os.environ.get("RESTORE_LAST_SESSION", "").lower() in {"1", "true", "yes"}:
        if svc.restore_state_snapshot():
            logger.info("Previous session restored")

    svc.start_persistence_loop(interval_seconds=300)

    try:
        yield
    finally:
        logger.info("Shutting down VideoService...")
        svc.shutdown()


app = FastAPI(title="Emergency Tracker API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(stream.router)
app.include_router(ws.router)
app.include_router(sources.router)
app.include_router(config_router.router)
app.include_router(areas.router)
app.include_router(actions.router)
app.include_router(sessions.router)


@app.get("/healthz", response_model=HealthResponse)
async def healthz():
    svc: VideoService = app.state.video_service
    state = svc.get_state()
    return HealthResponse(
        status="ok",
        video_running=state["running"],
        source=state["source"],
        session_id=state["session_id"],
        device=svc.detector.device if svc.detector else None,
        uptime_seconds=round(time.time() - app.state.started_at, 1),
        model_ready=svc.model_ready,
    )


@app.exception_handler(Exception)
async def unhandled_exception(_request, exc: Exception):
    logger.exception("Unhandled exception in request")
    return JSONResponse(status_code=500, content={"detail": str(exc)})


# 정적 서빙 — frontend/dist 가 빌드되어 있으면 SPA fallback 으로 제공
# (위의 모든 API 라우트 이후에 등록되어야 캐치-올로 동작하면서도 가려지지 않음)
_DIST_DIR = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"
if _DIST_DIR.exists():
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    # 자산은 정확한 경로로 서빙
    app.mount("/assets", StaticFiles(directory=str(_DIST_DIR / "assets")), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa_fallback(full_path: str):
        """SPA fallback — API 미매칭 경로는 index.html을 반환해 클라이언트 라우터가 처리."""
        # 정적 파일이 실제로 존재하면 그것을 반환
        candidate = _DIST_DIR / full_path
        if full_path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(_DIST_DIR / "index.html")

    logger.info("Mounted static frontend at %s", _DIST_DIR)
