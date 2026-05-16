"""APP_TOKEN 기반 단순 인증.

- 토큰이 설정되어 있으면 모든 보호 엔드포인트가 Bearer 토큰을 요구한다.
- WebSocket은 ?token=... 쿼리 파라미터로 검증한다.
- 토큰이 비어있으면(개발 모드) 검증을 스킵하지만, 운영 시작 시 LAN 노출이면 경고를 띄운다.
"""
from __future__ import annotations

import os
import secrets
from typing import Optional

from fastapi import Header, HTTPException, Query, WebSocket, status


def _expected_token() -> str:
    return os.environ.get("APP_TOKEN", "").strip()


def require_token(authorization: Optional[str] = Header(default=None)) -> None:
    """REST 보호용 의존성. Bearer 토큰 검증."""
    expected = _expected_token()
    if not expected:
        return  # 토큰 미설정 = 보호 비활성 (127.0.0.1 바인딩 가정)

    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Bearer token required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    provided = authorization.split(" ", 1)[1].strip()
    if not secrets.compare_digest(provided, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def ws_authenticate(websocket: WebSocket, token: Optional[str] = Query(default=None)) -> None:
    """WebSocket 핸드셰이크 검증. 실패 시 1008 close."""
    expected = _expected_token()
    if not expected:
        return
    if not token or not secrets.compare_digest(token, expected):
        await websocket.close(code=1008)
        raise HTTPException(status_code=403, detail="Invalid token")
