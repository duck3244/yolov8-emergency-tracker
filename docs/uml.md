# UML Diagrams — YOLOv8 Emergency Tracker

Mermaid 기반 UML 모음. GitHub/Obsidian/VS Code Markdown Preview 등에서 자동 렌더된다.

---

## 1. 컴포넌트 다이어그램 (시스템 토폴로지)

```mermaid
flowchart LR
    subgraph Browser["Browser (SPA)"]
        UI[App.tsx<br/>Tabs: monitor/areas/settings/history]
        VP[VideoPanel<br/>&lt;img src=/stream&gt;]
        AE[AreaEditor<br/>polygon overlay]
        Hook[useLiveState<br/>WebSocket client]
        ApiC[api/client.ts<br/>fetch wrapper]
    end

    subgraph FastAPI["FastAPI Process (uvicorn, workers=1)"]
        Main[app/main.py<br/>lifespan + SPA fallback]
        Auth[app/auth.py<br/>Bearer + WS token]
        subgraph Routers["app/api/*"]
            R1[stream.py]
            R2[ws.py]
            R3[sources.py]
            R4[config.py]
            R5[areas.py]
            R6[actions.py]
            R7[sessions.py]
        end
        VS[(VideoService<br/>singleton)]
        CM[(ConfigManager)]
    end

    subgraph Worker["video-worker thread"]
        Det[YOLODetector]
        Trk[Tracker]
        Cnt[AreaCounter]
        Ntf[NotificationManager]
    end

    Cam[(Webcam / mp4)]
    FS[(backend/output/<br/>session_*.json)]
    Mail[/SMTP / Webhook/]

    VP -- "GET /stream (MJPEG)" --> R1
    AE -- "GET /api/snapshot" --> R1
    Hook -- "WS /ws/state" --> R2
    ApiC -- "REST" --> R3 & R4 & R5 & R6 & R7

    R1 & R2 & R3 & R4 & R5 & R6 & R7 --> Auth
    R1 & R2 & R3 & R4 & R5 & R6 & R7 --> VS
    Main --> VS
    Main --> CM
    VS --> CM
    VS --> Worker
    Worker --> Det --> Trk --> Cnt --> Ntf
    Worker -. "cv2.VideoCapture" .-> Cam
    VS -. "snapshot 5min" .-> FS
    Ntf -. "alerts" .-> Mail
```

---

## 2. 클래스 다이어그램 — 백엔드 도메인

```mermaid
classDiagram
    class ConfigManager {
        +ModelConfig model
        +VideoConfig video
        +TrackingConfig tracking
        +CountingConfig counting
        +AlertConfig alert
        +EmailConfig email
        +LocationConfig location
        +UIConfig ui
        +load_config(path)
        +save_config()
    }

    class ModelConfig {
        +str model_path
        +float confidence_threshold
        +float iou_threshold
        +str device
        +int image_size
        +int max_det
        +bool half_precision
    }
    class VideoConfig {
        +str output_path
        +int frame_width
        +int frame_height
        +int fps_limit
        +int frame_skip
    }
    class TrackingConfig {
        +int distance_threshold
        +int max_disappeared
    }
    class CountingConfig {
        +list entrance_area
        +list exit_area
        +float min_residence_time
        +str area_name
    }
    class AlertConfig {
        +int overcrowding_threshold
        +int warning_threshold
        +int notification_interval
        +list[str] emergency_contacts
        +bool enable_email
        +tuple quiet_hours
    }
    class EmailConfig {
        +str smtp_server
        +int smtp_port
        +str sender_email
        +str sender_password
        +bool is_configured()
    }
    class LocationConfig {
        +str name
        +float latitude
        +float longitude
    }

    class VideoService {
        -ConfigManager config
        -YOLODetector detector
        -Tracker tracker
        -AreaCounter counter
        -NotificationManager notification_manager
        -cv2.VideoCapture _cap
        -Thread _thread
        -Event _stop_event
        -Lock _jpeg_lock
        -Lock _state_lock
        -bytes _latest_jpeg
        -dict _state
        +bool model_ready
        +start(source)
        +stop()
        +shutdown()
        +get_latest_jpeg() bytes
        +get_state() dict
        +apply_areas(entrance, exit_area, reset_counts)
        +apply_thresholds()
        +reset_counts()
        +save_state_snapshot()
        +restore_state_snapshot() bool
        +start_persistence_loop(interval)
        -_loop()
        -_process(frame)
        -_overlay(frame)
        -_alert_status(inside) str
    }

    class YOLODetector {
        +str device
        +YOLO model
        +detect(frame, target_classes)
        +detect_persons_only(frame) list
        +detect_with_nms(frame, iou)
        -_setup_device(device)
        -_load_model()
        -_predict(frame)
    }

    class Tracker {
        +dict center_points
        +dict disappeared
        +int distance_threshold
        +int max_disappeared
        +update(objects_rect) list
        +get_current_objects()
        +reset()
        +get_object_trajectory(id, max_points)
    }

    class AreaCounter {
        +list entrance_area
        +list exit_area
        +str area_name
        +set entered_ids
        +set exited_ids
        +float min_residence_time
        +update(tracked_objects)
        +get_counts() dict
        +get_hourly_stats() dict
        +reset_counts()
        +draw_areas(frame, show_counts)
        +save_history(filename)
        -_point_in(poly, x, y) bool
        -_record_transition(id, from, to, t)
    }

    class NotificationManager {
        +EmailNotifier email_notifier
        +SlackNotifier slack
        +DiscordNotifier discord
        +WebhookNotifier webhook
        +int overcrowding_threshold
        +int warning_threshold
        +int notification_interval
        +tuple quiet_hours
        +configure_email(...)
        +set_alert_rules(**rules)
        +check_and_send_alerts(count, location)
        +send_daily_report(data)
        +get_alert_history(limit)
        -_determine_alert_type(count) str
        -_should_suppress_alert(type, t) bool
        -_is_quiet_time(t) bool
    }

    class EmailNotifier {
        +str smtp_server
        +int smtp_port
        +configure(email, password)
        +send_emergency_alert(recipients, count, location)
        +send_periodic_report(recipients, data)
    }
    class SlackNotifier {
        +str webhook_url
        +send_emergency_alert(count, location)
    }
    class DiscordNotifier {
        +str webhook_url
        +send_emergency_alert(count, location)
    }
    class WebhookNotifier {
        +dict webhooks
        +add_webhook(name, url, headers)
        +send_to_all_webhooks(data)
    }

    ConfigManager *-- ModelConfig
    ConfigManager *-- VideoConfig
    ConfigManager *-- TrackingConfig
    ConfigManager *-- CountingConfig
    ConfigManager *-- AlertConfig
    ConfigManager *-- EmailConfig
    ConfigManager *-- LocationConfig

    VideoService o-- ConfigManager
    VideoService *-- YOLODetector
    VideoService *-- Tracker
    VideoService *-- AreaCounter
    VideoService *-- NotificationManager

    NotificationManager *-- EmailNotifier
    NotificationManager *-- SlackNotifier
    NotificationManager *-- DiscordNotifier
    NotificationManager *-- WebhookNotifier
```

---

## 3. 클래스 다이어그램 — FastAPI 레이어 (Pydantic 스키마)

```mermaid
classDiagram
    class HealthResponse {
        +str status
        +bool video_running
        +str? source
        +str? session_id
        +str? device
        +float uptime_seconds
        +bool model_ready
    }
    class StateModel {
        +bool running
        +str? source
        +str? session_id
        +int frame_count
        +float fps
        +CountsModel counts
        +str alert_status
        +str? last_error
    }
    class CountsModel {
        +int entered
        +int exited
        +int current_inside
        +str area_name
        +int max_inside_seen
    }
    class SourceItem {
        +str id
        +str label
        +str kind
        +int? size_bytes
    }
    class SourcesResponse {
        +list[SourceItem] items
    }
    class SelectSourceRequest {
        +str source_id
    }
    class AreaModel {
        +list[list[int]] entrance
        +list[list[int]] exit
    }
    class PutAreasRequest {
        +list[list[int]]? entrance
        +list[list[int]]? exit
        +bool reset_counts
    }
    class ConfigPatch {
        +float? confidence_threshold
        +float? iou_threshold
        +str? device
        +int? frame_skip
        +int? distance_threshold
        +int? max_disappeared
        +int? overcrowding_threshold
        +int? warning_threshold
        +int? notification_interval
        +bool? enable_email
        +list[str]? emergency_contacts
        +float? min_residence_time
    }
    class ConfigSummary {
        +float confidence_threshold
        +float iou_threshold
        +str device
        +int frame_skip
        +int frame_width
        +int frame_height
        +int distance_threshold
        +int max_disappeared
        +float min_residence_time
        +str area_name
        +int overcrowding_threshold
        +int warning_threshold
        +int notification_interval
        +bool enable_email
        +list[str] emergency_contacts
        +bool email_configured
        +str location_name
    }
    class ActionResponse {
        +bool ok
        +str? message
    }

    StateModel *-- CountsModel
    SourcesResponse *-- SourceItem
```

---

## 4. 시퀀스 — 비디오 소스 선택 및 라이브 스트림 시작

```mermaid
sequenceDiagram
    actor U as User (Browser)
    participant FE as Frontend (React)
    participant API as FastAPI Router
    participant SVC as VideoService
    participant W as video-worker Thread
    participant CAM as cv2.VideoCapture

    U->>FE: 소스 선택 (예: webcam)
    FE->>API: POST /api/sources/select {source_id:"webcam"}
    API->>SVC: start("webcam")
    SVC->>SVC: stop() 이전 워커 정리
    SVC->>CAM: VideoCapture(0)
    CAM-->>SVC: opened
    SVC->>W: Thread.start(_loop)
    SVC-->>API: ok
    API-->>FE: ActionResponse{ok:true}

    par MJPEG live preview
        FE->>API: GET /stream
        loop ~15 Hz
            API->>SVC: get_latest_jpeg()
            SVC-->>API: bytes
            API-->>FE: multipart frame
        end
    and 상태 푸시
        FE->>API: WS /ws/state
        loop 2 Hz
            API->>SVC: get_state()
            SVC-->>API: dict
            API-->>FE: JSON state
        end
    and 워커 처리 루프
        loop until stop_event
            W->>CAM: read()
            CAM-->>W: frame
            W->>W: YOLO → Tracker → Counter
            W->>SVC: update _latest_jpeg / _state
        end
    end
```

---

## 5. 시퀀스 — 영역 폴리곤 편집

```mermaid
sequenceDiagram
    actor U as User
    participant AE as AreaEditor.tsx
    participant API as FastAPI
    participant SVC as VideoService
    participant CFG as ConfigManager

    U->>AE: "areas" 탭 진입
    AE->>API: GET /api/snapshot
    API->>SVC: get_latest_jpeg()
    SVC-->>API: JPEG
    API-->>AE: image/jpeg
    AE->>API: GET /api/areas
    API->>CFG: counting.entrance/exit
    API-->>AE: AreaModel
    U->>AE: 폴리곤 정점 드래그 / 추가
    AE->>API: PUT /api/areas {entrance:[...], exit:[...], reset_counts:true}
    API->>SVC: apply_areas(...)
    SVC->>CFG: 갱신 + save_config()
    SVC->>SVC: new AreaCounter(...)
    SVC-->>API: ok
    API-->>AE: ActionResponse{ok:true}
```

---

## 6. 시퀀스 — 과밀 경보 트리거

```mermaid
sequenceDiagram
    participant W as video-worker
    participant C as AreaCounter
    participant SVC as VideoService
    participant NM as NotificationManager
    participant E as EmailNotifier
    participant FE as Browser (WS subscriber)

    loop 매 처리 사이클
        W->>C: update(tracked)
        C-->>W: counts
        W->>SVC: alert_status = _alert_status(inside)
        Note over SVC: inside ≥ overcrowding → "emergency"<br/>≥ warning → "warning"<br/>else → "normal"
        W->>NM: check_and_send_alerts(inside, location)
        alt 임계값 초과 AND quiet_hours 외 AND interval 경과
            NM->>E: send_emergency_alert(contacts, inside, location)
            E-->>NM: ok
            NM->>NM: 마지막 알림 시각 기록
        else 억제
            NM-->>W: skip
        end
    end

    Note over FE: 0.5 s 후 WS push
    FE->>SVC: (WS) get_state()
    SVC-->>FE: alert_status="emergency", counts.current_inside=N
    FE->>FE: AlertBanner 표시 + useAlertSound 트리거
```

---

## 7. 상태 다이어그램 — VideoService 수명주기

```mermaid
stateDiagram-v2
    [*] --> Initialized: lifespan startup\n_init_components()
    Initialized --> Idle: model_ready=true
    Idle --> Running: start(source)
    Running --> Running: worker loop\n(read → detect → track → count)
    Running --> Idle: stop() / EOF / error
    Idle --> ShuttingDown: lifespan shutdown
    Running --> ShuttingDown: lifespan shutdown
    ShuttingDown --> [*]: threads joined, cap released

    state Running {
        [*] --> Capturing
        Capturing --> Processing: every frame_skip
        Processing --> Encoding: overlay → JPEG
        Encoding --> Notifying: state update
        Notifying --> Capturing
    }
```

---

## 8. 상태 다이어그램 — Alert 상태머신

```mermaid
stateDiagram-v2
    [*] --> normal
    normal --> warning: inside ≥ warning_threshold
    warning --> emergency: inside ≥ overcrowding_threshold
    warning --> normal: inside &lt; warning_threshold
    emergency --> warning: inside &lt; overcrowding_threshold AND ≥ warning_threshold
    emergency --> normal: inside &lt; warning_threshold

    note right of emergency
        NotificationManager가
        interval/quiet_hours 통과 시
        이메일·웹훅 전송
    end note
```

---

## 9. 활동 다이어그램 — 워커 루프 한 사이클

```mermaid
flowchart TD
    A[stop_event 검사] -->|set| Z([루프 종료])
    A -->|clear| B[VideoCapture.read]
    B -->|webcam, ret=false| B
    B -->|file EOF| Z
    B -->|ret=true| C{프레임 크기 일치?}
    C -->|no| D[resize] --> E
    C -->|yes| E{frame_count % frame_skip == 0}
    E -->|no| H[last_processed_frame 사용]
    E -->|yes| F[YOLODetector.detect_persons_only]
    F --> G[Tracker.update]
    G --> I[AreaCounter.update]
    I --> J[_overlay: 폴리곤 + 박스 + ID]
    J --> H
    H --> K{15Hz 경과?}
    K -->|yes| L[cv2.imencode JPEG → _latest_jpeg]
    K -->|no| M
    L --> M[state 갱신: fps, counts, alert_status, max_inside_seen]
    M --> N[NotificationManager.check_and_send_alerts]
    N --> A
```

---

## 10. 배포 다이어그램

```mermaid
flowchart TB
    subgraph Host["Workstation / Edge PC (Linux/Windows/macOS)"]
        subgraph Py["Python 3.10+ venv"]
            UVI[uvicorn app.main:app<br/>--workers 1<br/>--host 127.0.0.1 --port 8000]
        end
        subgraph Static["frontend/dist (Vite build)"]
            DIST[index.html + assets/*]
        end
        DISK[(config.json<br/>output/*.json<br/>dataset/*.mp4<br/>yolov8s.pt)]
        ENV[(.env<br/>APP_TOKEN, EMAIL_PASSWORD)]
        CAMERA[(USB Webcam / RTSP)]
        GPU{{Optional GPU<br/>CUDA / MPS}}
    end

    Browser[(Browser<br/>same host or LAN)]
    SMTP[/SMTP Server/]
    SlackEP[/Slack Webhook/]

    Browser -- "127.0.0.1:8000<br/>HTTP / WS" --> UVI
    UVI -- "static fallback" --> DIST
    UVI -- "read/write" --> DISK
    UVI -- "read" --> ENV
    UVI -- "VideoCapture" --> CAMERA
    UVI -. "torch device" .-> GPU
    UVI -- "alerts" --> SMTP & SlackEP
```

---

## 11. 프론트엔드 컴포넌트 트리

```mermaid
flowchart TD
    App[App.tsx]
    App --> AlertBanner
    App --> Nav[Tab Nav]
    App --> M{tab}
    M -->|monitor| Mon[VideoPanel + CountsPanel + SourceSelect + Session info]
    M -->|areas| Areas[AreaEditor + CountsPanel]
    M -->|settings| Set[SettingsForm + Location card]
    M -->|history| Hist[HistoryPanel]

    App -. "hook" .-> H1[useLiveState<br/>→ openStateSocket]
    App -. "hook" .-> H2[useAlertSound]
    Mon -. "hook" .-> H3[useDebouncedCallback]

    subgraph UI["components/ui (shadcn)"]
        button
        card
        dialog
        input
        label
        select
        slider
        switch
        badge
    end

    Mon --- UI
    Areas --- UI
    Set --- UI
    Hist --- UI
```

---

## 12. API 엔드포인트 매트릭스 (요약)

| Method | Path | 인증 | Request | Response | Router |
|--------|------|------|---------|----------|--------|
| GET | `/healthz` | — | — | `HealthResponse` | main |
| GET | `/stream` | Bearer | — | `multipart/x-mixed-replace` | stream |
| GET | `/api/snapshot` | Bearer | — | `image/jpeg` | stream |
| WS | `/ws/state` | `?token=` | — | `StateModel` JSON @2Hz | ws |
| GET | `/api/sources` | Bearer | — | `SourcesResponse` | sources |
| POST | `/api/sources/select` | Bearer | `SelectSourceRequest` | `ActionResponse` | sources |
| GET | `/api/config` | Bearer | — | `ConfigSummary` | config |
| PUT | `/api/config` | Bearer | `ConfigPatch` | `ConfigSummary` | config |
| GET | `/api/areas` | Bearer | — | `AreaModel` | areas |
| PUT | `/api/areas` | Bearer | `PutAreasRequest` | `ActionResponse` | areas |
| POST | `/api/actions/reset_counts` | Bearer | — | `ActionResponse` | actions |
| POST | `/api/actions/stop` | Bearer | — | `ActionResponse` | actions |
| POST | `/api/actions/save_state` | Bearer | — | `ActionResponse` | actions |
| POST | `/api/actions/restore_state` | Bearer | — | `ActionResponse` | actions |
| POST | `/api/actions/send_test_alert` | Bearer | — | `ActionResponse` | actions |
| GET | `/api/sessions` | Bearer | — | `SessionsResponse` | sessions |
| GET | `/api/sessions/{filename}` | Bearer | — | `application/json` | sessions |
