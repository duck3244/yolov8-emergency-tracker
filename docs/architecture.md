# Architecture — YOLOv8 Emergency Tracker

YOLOv8 기반 사람 카운팅·과밀 알림 시스템. 단일 호스트(워크스테이션)에서 실행되는 FastAPI 백엔드와 React/Vite 프론트엔드로 구성된다.

---

## 1. 전체 개요

```
┌──────────────────────────────────────────────────────────────────┐
│                          Browser (SPA)                            │
│   React 18 + Vite + Tailwind + shadcn/ui (Radix primitives)       │
│   ─ <img src=/stream>  : MJPEG live preview                        │
│   ─ WebSocket /ws/state : 2 Hz counts / fps / alert push           │
│   ─ REST /api/*        : config, areas, sources, actions, sessions │
└──────────────────────────────┬───────────────────────────────────┘
                               │  HTTP / WS  (single host, 127.0.0.1:8000 by default)
┌──────────────────────────────▼───────────────────────────────────┐
│                       FastAPI (single worker)                     │
│  app/main.py    : lifespan, CORS, SPA fallback, /healthz          │
│  app/auth.py    : Bearer APP_TOKEN (REST)  + ?token=… (WS)        │
│  app/api/*      : stream / ws / sources / config / areas /        │
│                   actions / sessions  (Pydantic-validated)        │
│  app/schemas.py : 요청·응답 모델 (frontend types.ts와 미러)         │
└──────────────────────────────┬───────────────────────────────────┘
                               │  in-process call (thread-safe state)
┌──────────────────────────────▼───────────────────────────────────┐
│                    VideoService (singleton)                       │
│  ─ 백그라운드 캡처 스레드 ("video-worker") 1개                       │
│  ─ 주기적 상태 스냅샷 스레드 ("state-persist", 5분 간격)             │
│  ─ latest-only JPEG buffer + state dict, 모두 Lock 보호             │
│                                                                   │
│   ┌────────────┐  ┌────────────┐  ┌─────────────┐  ┌─────────────┐│
│   │YOLODetector│→ │  Tracker   │→ │ AreaCounter │→ │Notification ││
│   │ (yolov8s)  │  │ centroid + │  │ entrance /  │  │ Manager     ││
│   │            │  │ disappear  │  │ exit polys  │  │ email/slack ││
│   └────────────┘  └────────────┘  └─────────────┘  └─────────────┘│
│         ▲                                                          │
│         └── ConfigManager (config.json + .env) — 핫리로드 setters    │
└──────────────────────────────┬───────────────────────────────────┘
                               │
                       OpenCV VideoCapture
                               │
            ┌──────────────────┴──────────────────┐
            │                                     │
        Webcam (0)                       backend/dataset/*.mp4
```

핵심 설계 원칙:
- **단일 워커**: `VideoService`가 프로세스 내 상태(카운트, JPEG 버퍼)를 보유하므로 `uvicorn --workers 1` 고정.
- **GUI 비의존**: 백엔드는 `cv2.imshow` 등 GUI 호출을 하지 않는다. 프레임은 JPEG로 인코딩되어 MJPEG/스냅샷으로만 노출.
- **Latest-only 전달**: 큐를 쌓지 않고 가장 최근 JPEG/state만 보관. 클라이언트는 폴링/구독 방식으로 가져간다.
- **단일 사용자 가정**: APP_TOKEN 한 개로 보호하고 기본은 127.0.0.1 바인딩. 본격 멀티유저 인증은 비-범위.

---

## 2. 디렉토리 레이아웃

```
yolov8-emergency-tracker/
├── README.md
├── LICENSE
├── backend/
│   ├── app/                       # FastAPI 애플리케이션 패키지
│   │   ├── main.py                # 진입점, lifespan, 라우터 등록, SPA fallback
│   │   ├── auth.py                # APP_TOKEN Bearer / WS 토큰
│   │   ├── schemas.py             # Pydantic 요청·응답 모델
│   │   ├── api/
│   │   │   ├── stream.py          # GET /stream (MJPEG), /api/snapshot
│   │   │   ├── ws.py              # WS /ws/state (2Hz push)
│   │   │   ├── sources.py         # GET /api/sources, POST /api/sources/select
│   │   │   ├── config.py          # GET/PUT /api/config (부분 업데이트)
│   │   │   ├── areas.py           # GET/PUT /api/areas (entrance/exit 폴리곤)
│   │   │   ├── actions.py         # reset_counts / stop / save_state / restore_state / send_test_alert
│   │   │   └── sessions.py        # 저장된 세션 JSON 목록 / 다운로드
│   │   └── services/
│   │       └── video_service.py   # 캡처 워커 스레드 + 도메인 객체 소유
│   ├── config.py                  # ConfigManager + dataclass 섹션 (model/video/...)
│   ├── config.json                # 영속 설정
│   ├── .env.example               # APP_TOKEN, EMAIL_PASSWORD 등 비밀값
│   ├── detector.py                # YOLODetector / BatchDetector
│   ├── tracker.py                 # Tracker (centroid) / MultiClassTracker
│   ├── counter.py                 # AreaCounter / MultiAreaCounter / DirectionCounter
│   ├── notification.py            # EmailNotifier / Slack / Discord / Webhook / Manager
│   ├── visualization.py           # 오프라인용 시각화 (서비스 경로에서는 미사용)
│   ├── area_setup.py              # 영역 폴리곤 보조 도구
│   ├── main.py                    # CLI 진입점 (FastAPI와 별개의 standalone runner)
│   ├── requirements.txt
│   ├── dataset/                   # 입력 비디오 (mp4 등) — 화이트리스트 경로
│   ├── output/                    # 세션 스냅샷·통계 JSON 저장
│   ├── yolov8s.pt                 # 사전학습 모델 가중치
│   └── tests/                     # pytest (conftest + api/config/counter/tracker/notification)
└── frontend/
    ├── index.html
    ├── package.json               # React 18, Radix, Tailwind, openapi-typescript
    ├── vite.config.ts
    └── src/
        ├── main.tsx
        ├── App.tsx                # 탭(monitor / areas / settings / history) 구성
        ├── api/
        │   ├── client.ts          # fetch 래퍼 + WebSocket 연결 헬퍼
        │   └── types.ts           # 백엔드 schemas.py와 미러된 TS 타입
        ├── hooks/
        │   ├── useLiveState.ts    # /ws/state 구독, 지수 백오프 재연결
        │   ├── useAlertSound.ts   # alert_status 변화 시 사운드 트리거
        │   └── useDebouncedCallback.ts
        └── components/
            ├── VideoPanel.tsx     # <img src=/stream>
            ├── CountsPanel.tsx
            ├── SourceSelect.tsx
            ├── SettingsForm.tsx
            ├── AreaEditor.tsx     # /api/snapshot 위에 폴리곤 편집
            ├── AlertBanner.tsx
            ├── HistoryPanel.tsx   # /api/sessions 목록·다운로드
            └── ui/                # shadcn/ui (button, card, dialog, ...)
```

---

## 3. 백엔드 모듈 책임

| 모듈 | 책임 | 외부 의존 |
|------|------|----------|
| `app/main.py` | FastAPI 인스턴스, lifespan으로 ConfigManager·VideoService 구동, 라우터 등록, SPA 정적 서빙 | fastapi, uvicorn |
| `app/auth.py` | `APP_TOKEN` 환경변수 기반 Bearer 인증. WS는 `?token=` 쿼리 검증. 미설정 시 개발모드 | secrets |
| `app/schemas.py` | Pydantic 요청·응답 모델 (Health/State/Counts/Areas/ConfigPatch/...) | pydantic |
| `app/services/video_service.py` | 싱글톤 워커. 캡처 스레드·persistence 스레드 관리, 영역/임계값 핫리로드, 상태 스냅샷 영속화 | cv2, threading |
| `config.py` | `ConfigManager` + dataclass 섹션. `config.json` 로드·저장, `.env`에서 비밀값 머지 | dataclasses, json |
| `detector.py` | `YOLODetector.detect_persons_only(frame) → boxes`, 디바이스/half precision 처리 | ultralytics, torch |
| `tracker.py` | Centroid 기반 ID 부여. `update(boxes) → [(x1,y1,x2,y2,id), ...]` | numpy |
| `counter.py` | 진입/퇴장 폴리곤 점-내포 판정, `current_inside` 계산, 영역 시각화 | cv2 |
| `notification.py` | Email/Slack/Discord/Webhook 알림. quiet_hours·notification_interval로 throttle | smtplib, requests |

---

## 4. 데이터 흐름 (런타임)

### 4.1 비디오 처리 파이프라인 (단일 사이클, ~30 Hz)

1. `cv2.VideoCapture.read()` — 웹캠 또는 dataset 파일에서 BGR 프레임.
2. 크기 정규화 (config의 `frame_width × frame_height`).
3. `frame_skip` 간격마다:
   - `YOLODetector.detect_persons_only(frame)` → person 박스 리스트.
   - `Tracker.update(boxes)` → ID가 부여된 tracked 객체.
   - `AreaCounter.update(tracked)` → 진입/퇴장 상태 갱신, `entered_ids`/`exited_ids` 집합 누적.
4. `_overlay(frame)` — counter가 영역 폴리곤을, 서비스가 박스/ID를 그림.
5. JPEG 인코딩 (~15 Hz 쓰로틀) → `_latest_jpeg` 갱신.
6. 상태 락 안에서 `state["counts"]`, `fps`, `max_inside_seen`, `alert_status` 업데이트.
7. `NotificationManager.check_and_send_alerts(inside, location)` — 임계값 초과 시 메일/웹훅, 단 quiet_hours·interval로 억제.

### 4.2 클라이언트 ↔ 서버

- **MJPEG**: 브라우저 `<img>`가 `/stream`에 connect → 서버는 `multipart/x-mixed-replace` 응답을 `latest_jpeg`로 ~15 Hz 송출.
- **WebSocket**: `/ws/state` 핸드셰이크 후 0.5 s마다 `get_state()` JSON 송신. 클라이언트는 `useLiveState` 훅에서 지수 백오프로 재연결.
- **REST**: 토큰 헤더 `Authorization: Bearer <APP_TOKEN>` (설정 시). 모든 요청은 `app.state.video_service` / `app.state.config_manager`를 통해 워커에 영향.

---

## 5. 상태 & 동시성

- `VideoService`는 두 종류의 락을 가진다:
  - `_jpeg_lock` — `_latest_jpeg` (단일 latest-only 버퍼).
  - `_state_lock` — `_state` dict (frame_count/fps/counts/alert_status).
- 워커 스레드는 락 보유 시간을 짧게 유지하고, 외부 핸들러는 락 안에서 deep-copy(JSON round-trip)해서 반환.
- 영역/임계값 변경(`apply_areas`, `apply_thresholds`)은 메인 이벤트 루프에서 호출되며 워커 스레드가 다음 사이클에 새 객체를 사용한다 (재할당이므로 부분 상태 문제 없음).
- 종료 절차: lifespan 종료 → `shutdown()` → `stop_event.set()` → `_thread.join(3s)` → cap.release → persistence 스레드 join.

---

## 6. 설정 & 비밀값

- `backend/config.json` — 모델·비디오·트래킹·카운팅·알림·이메일·위치·UI 섹션. `ConfigManager.save_config()`로 부분 업데이트가 디스크에 즉시 반영.
- `backend/.env` — `APP_TOKEN`, 이메일 비밀번호 등 git 비커밋. `.env.example` 참고.
- 환경변수:
  - `APP_TOKEN` — 비어있으면 인증 미적용 (개발모드, 경고 로그 출력).
  - `LOG_LEVEL` — 기본 `INFO`.
  - `RESTORE_LAST_SESSION` — `1/true/yes`이면 lifespan에서 `session_state.json` 복원.

---

## 7. 영속화

- `backend/output/session_state.json` — 5분 주기 또는 `/api/actions/save_state`로 작성. 카운트, `entered_ids`, `exited_ids`, `max_inside_seen` 포함.
- `backend/output/session_<timestamp>.json` — 수동 저장 시 타임스탬프 파일을 추가로 남김. `/api/sessions`로 목록·다운로드.

---

## 8. 인증·네트워크 경계

- 기본은 `127.0.0.1:8000` 바인딩이라는 운영 가정. 외부 노출 시 `APP_TOKEN` 필수.
- CORS는 `localhost:5173 / 127.0.0.1:5173`만 허용 (개발 Vite).
- MJPEG `/stream`은 헤더 인증이 까다로워 토큰 쿼리를 받지 않음. 운영 시 reverse-proxy 인증 + 동일 호스트 가정.

---

## 9. 테스트

- `backend/tests/` — pytest. 다음을 단위 테스트:
  - `test_config.py` — ConfigManager 로드/저장.
  - `test_tracker.py` — centroid id 할당·disappeared.
  - `test_counter.py` — 영역 진입/퇴장 카운트.
  - `test_notification.py` — quiet hours / 억제 로직.
  - `test_api.py` — FastAPI TestClient로 라우터 응답.
- `conftest.py`에 공통 fixture (임시 config, 더미 frame 등).

---

## 10. 향후 확장 지점 (참고)

- WebRTC 또는 HLS 전환으로 MJPEG 대비 대역폭 절감.
- Multi-camera는 `VideoService`를 `dict[source_id, VideoService]`로 확장 + 라우팅 키 도입.
- 알림 채널은 `NotificationManager`가 이미 plug-in 구조(`add_custom_webhook`)이므로 라우터·UI만 추가하면 됨.
- 모델은 `detector.py`의 `_load_model`을 교체해 더 작은 모델(`yolov8n`)이나 ONNX/TensorRT 백엔드로 스왑 가능.
