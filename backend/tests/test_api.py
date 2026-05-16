"""FastAPI 엔드포인트 통합 테스트.

비싼 모델 로드를 피하기 위해 VideoService 초기화 단계를 monkey-patch한다.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# VideoService 의존성(detector/tracker/counter) 만 stub 처리
import app.services.video_service as vs_mod
from counter import AreaCounter
from tracker import Tracker


def _stub_init_components(self):
    self.detector = None  # 추론은 안 한다
    self.tracker = Tracker(
        distance_threshold=self.config.tracking.distance_threshold,
        max_disappeared=self.config.tracking.max_disappeared,
    )
    self.counter = AreaCounter(
        entrance_area=self.config.counting.entrance_area,
        exit_area=self.config.counting.exit_area,
        area_name=self.config.counting.area_name,
        min_residence_time=self.config.counting.min_residence_time,
    )
    self.notification_manager = None
    self.model_ready = True


vs_mod.VideoService._init_components = _stub_init_components
vs_mod.VideoService._warmup = lambda self: None


@pytest.fixture(scope="module")
def client(tmp_path_factory, monkeypatch_module):
    cfg_dir = tmp_path_factory.mktemp("backend")
    monkeypatch_module.chdir(cfg_dir)
    # APP_TOKEN 비활성 — 인증 우회 테스트 케이스도 별도로 검증
    monkeypatch_module.delenv("APP_TOKEN", raising=False)

    from app.main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def monkeypatch_module():
    """module-scope monkeypatch (pytest 기본은 function-scope)."""
    from _pytest.monkeypatch import MonkeyPatch

    mp = MonkeyPatch()
    yield mp
    mp.undo()


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_ready"] is True
    assert body["video_running"] is False


def test_config_get_and_patch(client):
    r = client.get("/api/config")
    assert r.status_code == 200
    initial = r.json()
    assert "confidence_threshold" in initial

    r = client.put("/api/config", json={"warning_threshold": 33})
    assert r.status_code == 200
    assert r.json()["warning_threshold"] == 33


def test_config_validates_bounds(client):
    r = client.put("/api/config", json={"confidence_threshold": 1.5})
    assert r.status_code == 422


def test_sources_list(client):
    r = client.get("/api/sources")
    assert r.status_code == 200
    items = r.json()["items"]
    # 최소 webcam은 있어야 함
    ids = {i["id"] for i in items}
    assert "webcam" in ids


def test_areas_get_and_put(client):
    r = client.get("/api/areas")
    assert r.status_code == 200
    areas = r.json()
    assert "entrance" in areas and "exit" in areas

    new = {
        "entrance": [[10, 10], [20, 10], [20, 20], [10, 20]],
        "exit": [[100, 100], [200, 100], [200, 200], [100, 200]],
        "reset_counts": True,
    }
    r = client.put("/api/areas", json=new)
    assert r.status_code == 200
    assert r.json()["ok"] is True

    # 폴리곤 점이 부족하면 400
    r = client.put("/api/areas", json={
        "entrance": [[0, 0], [1, 1]],
        "exit": [[2, 2], [3, 3]],
    })
    assert r.status_code == 400


def test_areas_partial_update_exit_only(client):
    """entrance 생략 — exit만 변경, entrance는 직전 값 유지."""
    # 베이스라인 설정
    base = {
        "entrance": [[0, 0], [10, 0], [10, 10], [0, 10]],
        "exit": [[100, 100], [110, 100], [110, 110], [100, 110]],
    }
    assert client.put("/api/areas", json=base).status_code == 200
    before = client.get("/api/areas").json()

    # exit만 변경
    only_exit = {"exit": [[200, 200], [210, 200], [210, 210], [200, 210]],
                 "reset_counts": False}
    r = client.put("/api/areas", json=only_exit)
    assert r.status_code == 200
    assert "exit" in r.json()["message"]
    assert "entrance" not in r.json()["message"]

    after = client.get("/api/areas").json()
    assert after["entrance"] == before["entrance"]  # 유지
    assert after["exit"] == only_exit["exit"]       # 갱신됨


def test_areas_partial_update_entrance_only(client):
    only_entry = {"entrance": [[300, 300], [310, 300], [310, 310], [300, 310]],
                  "reset_counts": False}
    before = client.get("/api/areas").json()
    r = client.put("/api/areas", json=only_entry)
    assert r.status_code == 200

    after = client.get("/api/areas").json()
    assert after["exit"] == before["exit"]
    assert after["entrance"] == only_entry["entrance"]


def test_areas_empty_payload_400(client):
    r = client.put("/api/areas", json={"reset_counts": False})
    assert r.status_code == 400


def test_actions_reset_and_stop(client):
    assert client.post("/api/actions/reset_counts").status_code == 200
    assert client.post("/api/actions/stop").status_code == 200


def test_token_enforcement(monkeypatch):
    """APP_TOKEN 활성 시 보호 엔드포인트는 401."""
    monkeypatch.setenv("APP_TOKEN", "s3cret")
    from importlib import reload
    import app.auth as auth_mod

    reload(auth_mod)
    from app.main import app as fresh_app

    with TestClient(fresh_app) as c:
        # /healthz는 보호 안 함
        assert c.get("/healthz").status_code == 200
        # /api/config 는 토큰 없으면 401
        assert c.get("/api/config").status_code == 401
        # 토큰 있으면 200
        r = c.get("/api/config", headers={"Authorization": "Bearer s3cret"})
        assert r.status_code == 200


def test_snapshot_returns_404_when_no_frame_yet(client):
    r = client.get("/api/snapshot")
    assert r.status_code == 404
