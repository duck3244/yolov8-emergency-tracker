"""/api/sessions — 저장된 세션 JSON 목록·다운로드."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.auth import require_token

logger = logging.getLogger(__name__)
router = APIRouter()


class SessionEntry(BaseModel):
    filename: str
    kind: str  # "session" | "statistics" | "state" | "other"
    size_bytes: int
    modified_at: str  # ISO 8601


class SessionsResponse(BaseModel):
    items: List[SessionEntry]


def _classify(name: str) -> str:
    n = name.lower()
    if n.startswith("session_"):
        return "session"
    if n.startswith("statistics_"):
        return "statistics"
    if n == "session_state.json":
        return "state"
    return "other"


def _output_dir(request: Request) -> Path:
    cm = request.app.state.config_manager
    return (Path(__file__).resolve().parent.parent.parent / cm.video.output_path).resolve()


@router.get("/api/sessions", response_model=SessionsResponse,
            dependencies=[Depends(require_token)])
async def list_sessions(request: Request):
    out = _output_dir(request)
    items: List[SessionEntry] = []
    if out.exists():
        for p in sorted(out.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
            if not p.is_file() or p.suffix.lower() not in {".json", ".mp4", ".html"}:
                continue
            st = p.stat()
            items.append(
                SessionEntry(
                    filename=p.name,
                    kind=_classify(p.name),
                    size_bytes=st.st_size,
                    modified_at=__import__("datetime").datetime.fromtimestamp(
                        st.st_mtime
                    ).isoformat(),
                )
            )
    return SessionsResponse(items=items)


@router.get("/api/sessions/{filename}", dependencies=[Depends(require_token)])
async def download_session(request: Request, filename: str):
    out = _output_dir(request)
    target = (out / filename).resolve()
    try:
        target.relative_to(out)
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid path")
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(
        path=str(target),
        filename=filename,
        media_type="application/octet-stream",
    )
