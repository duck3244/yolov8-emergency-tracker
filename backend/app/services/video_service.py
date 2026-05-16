"""VideoService — 카메라/모델/트래커/카운터를 소유하는 백그라운드 워커.

설계:
- 싱글톤(FastAPI 앱 lifespan에서 1개만 생성).
- `start(source)` 호출 시 별도 스레드에서 캡처 루프를 돈다.
- 이벤트 루프와 격리되어 동작하므로 모든 공유 상태는 락으로 보호한다.
- `latest_jpeg`/`state`는 polling 방식으로 핸들러가 읽는다 (latest-only).
- imshow 같은 GUI 호출은 절대 하지 않는다.
"""
from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np

from config import ConfigManager
from counter import AreaCounter
from detector import YOLODetector
from notification import NotificationManager
from tracker import Tracker

logger = logging.getLogger(__name__)


class VideoService:
    """싱글 카메라/모델 인스턴스를 보유한 백그라운드 워커."""

    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.started_at = time.time()

        self.detector: Optional[YOLODetector] = None
        self.tracker: Optional[Tracker] = None
        self.counter: Optional[AreaCounter] = None
        self.notification_manager: Optional[NotificationManager] = None

        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        self._jpeg_lock = threading.Lock()
        self._latest_jpeg: Optional[bytes] = None

        self._state_lock = threading.Lock()
        self._state: Dict[str, Any] = self._empty_state()

        self._persistence_thread: Optional[threading.Thread] = None
        self._persistence_stop = threading.Event()
        self._output_dir = Path(self.config.video.output_path)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._state_file = self._output_dir / "session_state.json"

        self.model_ready = False
        self._init_components()

    # ------------------------------------------------------------------
    # 초기화

    def _empty_state(self) -> Dict[str, Any]:
        return {
            "running": False,
            "source": None,
            "session_id": None,
            "frame_count": 0,
            "fps": 0.0,
            "counts": {
                "entered": 0,
                "exited": 0,
                "current_inside": 0,
                "area_name": self.config.counting.area_name,
                "max_inside_seen": 0,
            },
            "alert_status": "normal",
            "last_error": None,
        }

    def _init_components(self) -> None:
        try:
            logger.info("Initializing YOLO detector...")
            self.detector = YOLODetector(
                model_path=self.config.model.model_path,
                confidence_threshold=self.config.model.confidence_threshold,
                device=self.config.model.device,
                iou_threshold=self.config.model.iou_threshold,
                image_size=self.config.model.image_size,
                max_det=self.config.model.max_det,
                half_precision=self.config.model.half_precision,
            )
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
            self.notification_manager = NotificationManager()
            if self.config.email.is_configured():
                self.notification_manager.configure_email(
                    sender_email=self.config.email.sender_email,
                    sender_password=self.config.email.sender_password,
                    emergency_contacts=self.config.alert.emergency_contacts,
                )
            self.notification_manager.set_alert_rules(
                overcrowding_threshold=self.config.alert.overcrowding_threshold,
                warning_threshold=self.config.alert.warning_threshold,
                notification_interval=self.config.alert.notification_interval,
                quiet_hours=self.config.alert.quiet_hours,
            )
            self._warmup()
            self.model_ready = True
            logger.info("YOLO components ready")
        except Exception:
            logger.exception("Failed to initialize components")
            raise

    def _warmup(self) -> None:
        """첫 추론 지연을 줄이기 위해 dummy 프레임으로 모델 워밍업."""
        try:
            dummy = np.zeros(
                (self.config.video.frame_height, self.config.video.frame_width, 3),
                dtype=np.uint8,
            )
            self.detector.detect_persons_only(dummy)
        except Exception:
            logger.exception("Warmup failed (non-fatal)")

    # ------------------------------------------------------------------
    # 외부 API

    def start(self, source: str) -> None:
        """워커 스레드 시작. 이미 동작 중이면 먼저 중지한다."""
        self.stop()

        cap_target: Any
        if source == "webcam":
            cap_target = 0
        else:
            # 경로 검증: dataset 디렉토리 내부만 허용
            dataset_dir = Path(self.config.video.output_path).parent / "dataset"
            full = (dataset_dir / source).resolve()
            try:
                full.relative_to(dataset_dir.resolve())
            except ValueError as e:
                raise ValueError(f"source path outside dataset/: {source}") from e
            if not full.exists():
                raise FileNotFoundError(str(full))
            cap_target = str(full)

        cap = cv2.VideoCapture(cap_target)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open source: {source}")

        if source == "webcam":
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.video.frame_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.video.frame_height)
            cap.set(cv2.CAP_PROP_FPS, self.config.video.fps_limit)

        self._cap = cap
        self._stop_event.clear()
        session_id = uuid.uuid4().hex[:8]
        with self._state_lock:
            self._state = self._empty_state()
            self._state["running"] = True
            self._state["source"] = source
            self._state["session_id"] = session_id

        self._thread = threading.Thread(
            target=self._loop, name="video-worker", daemon=True
        )
        self._thread.start()
        logger.info("VideoService started: source=%s session=%s", source, session_id)

    def stop(self) -> None:
        if self._thread and self._thread.is_alive():
            self._stop_event.set()
            self._thread.join(timeout=3.0)
        self._thread = None

        if self._cap is not None:
            self._cap.release()
            self._cap = None

        with self._state_lock:
            self._state["running"] = False

    def shutdown(self) -> None:
        """앱 종료 시 호출 — 모든 스레드/리소스 정리."""
        self.stop()
        self._persistence_stop.set()
        if self._persistence_thread and self._persistence_thread.is_alive():
            self._persistence_thread.join(timeout=2.0)

    def start_persistence_loop(self, interval_seconds: int = 300) -> None:
        if self._persistence_thread and self._persistence_thread.is_alive():
            return
        self._persistence_stop.clear()
        self._persistence_thread = threading.Thread(
            target=self._persistence_loop,
            args=(interval_seconds,),
            name="state-persist",
            daemon=True,
        )
        self._persistence_thread.start()

    # ------------------------------------------------------------------
    # Snapshot / state getters

    def get_latest_jpeg(self) -> Optional[bytes]:
        with self._jpeg_lock:
            return self._latest_jpeg

    def get_state(self) -> Dict[str, Any]:
        with self._state_lock:
            # 카운터에서 직접 가져와 최신성 보장
            if self.counter is not None:
                counts = self.counter.get_counts()
                self._state["counts"]["entered"] = counts["entered"]
                self._state["counts"]["exited"] = counts["exited"]
                self._state["counts"]["current_inside"] = counts["current_inside"]
                self._state["counts"]["area_name"] = counts["area_name"]
            return json.loads(json.dumps(self._state))  # deep copy via JSON

    # ------------------------------------------------------------------
    # 핵심 캡처 루프

    def _loop(self) -> None:
        assert self._cap is not None
        frame_skip = max(1, int(self.config.video.frame_skip or 1))
        last_jpeg_ts = 0.0
        last_processed_frame = None
        processing_times: list = []
        frame_count = 0

        # 파일 소스는 원본 FPS로 재생 throttle (웹캠은 cap이 자체 throttle)
        is_file = self._state.get("source") not in (None, "webcam")
        src_fps = self._cap.get(cv2.CAP_PROP_FPS) or 0.0
        frame_interval = 1.0 / src_fps if (is_file and src_fps > 0.5) else 0.0
        next_read_at = time.monotonic()

        try:
            while not self._stop_event.is_set():
                if frame_interval > 0:
                    sleep = next_read_at - time.monotonic()
                    if sleep > 0:
                        # stop_event를 짧은 간격으로 확인하기 위해 wait 사용
                        if self._stop_event.wait(timeout=min(sleep, 0.1)):
                            break
                        continue
                    next_read_at += frame_interval

                ret, frame = self._cap.read()
                if not ret:
                    if self._state.get("source") != "webcam":
                        logger.info("Video source EOF")
                        break
                    time.sleep(0.01)
                    continue

                frame_count += 1

                if frame.shape[:2] != (
                    self.config.video.frame_height,
                    self.config.video.frame_width,
                ):
                    frame = cv2.resize(
                        frame,
                        (
                            self.config.video.frame_width,
                            self.config.video.frame_height,
                        ),
                    )

                if frame_count % frame_skip == 0:
                    t0 = time.time()
                    self._process(frame)
                    processing_times.append(time.time() - t0)
                    if len(processing_times) > 30:
                        processing_times = processing_times[-30:]
                    last_processed_frame = self._overlay(frame)

                output = last_processed_frame if last_processed_frame is not None else frame

                # JPEG 인코딩은 ~15Hz로 throttle
                if time.time() - last_jpeg_ts >= 1 / 15:
                    ok, buf = cv2.imencode(
                        ".jpg", output, [cv2.IMWRITE_JPEG_QUALITY, 70]
                    )
                    if ok:
                        with self._jpeg_lock:
                            self._latest_jpeg = buf.tobytes()
                        last_jpeg_ts = time.time()

                if processing_times:
                    fps = 1.0 / (sum(processing_times) / len(processing_times))
                else:
                    fps = 0.0

                with self._state_lock:
                    self._state["frame_count"] = frame_count
                    self._state["fps"] = round(fps, 1)
                    inside = self.counter.get_counts()["current_inside"]
                    self._state["counts"]["max_inside_seen"] = max(
                        self._state["counts"]["max_inside_seen"], inside
                    )
                    self._state["alert_status"] = self._alert_status(inside)

                # 알림 전송
                location = {
                    "name": self.config.location.name,
                    "lat": self.config.location.latitude,
                    "lon": self.config.location.longitude,
                }
                try:
                    self.notification_manager.check_and_send_alerts(inside, location)
                except Exception:
                    logger.exception("Notification dispatch failed (non-fatal)")
        except Exception as e:
            logger.exception("Video worker crashed")
            with self._state_lock:
                self._state["last_error"] = str(e)
        finally:
            if self._cap is not None:
                self._cap.release()
                self._cap = None
            with self._state_lock:
                self._state["running"] = False
            logger.info("VideoService worker stopped")

    def _process(self, frame) -> None:
        boxes = self.detector.detect_persons_only(frame)
        tracked = self.tracker.update(boxes)
        self.counter.update(tracked)
        self._last_tracked = tracked  # for overlay

    def _overlay(self, frame):
        out = self.counter.draw_areas(frame, show_counts=False)
        for obj in getattr(self, "_last_tracked", []):
            x1, y1, x2, y2, oid = obj
            cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(
                out,
                f"ID:{oid}",
                (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
        return out

    def _alert_status(self, inside: int) -> str:
        if inside >= self.config.alert.overcrowding_threshold:
            return "emergency"
        if inside >= self.config.alert.warning_threshold:
            return "warning"
        return "normal"

    # ------------------------------------------------------------------
    # 변경 반영

    def apply_areas(self, entrance=None, exit_area=None, reset_counts: bool = True) -> None:
        """영역 갱신 (둘 중 하나만 보내면 그 쪽만 갱신).

        polygon 캐시 재빌드를 위해 counter를 새로 만든다.
        """
        if entrance is not None:
            self.config.counting.entrance_area = entrance
        if exit_area is not None:
            self.config.counting.exit_area = exit_area

        self.config.save_config()
        self.counter = AreaCounter(
            entrance_area=self.config.counting.entrance_area,
            exit_area=self.config.counting.exit_area,
            area_name=self.config.counting.area_name,
            min_residence_time=self.config.counting.min_residence_time,
        )
        if reset_counts:
            with self._state_lock:
                self._state["counts"]["max_inside_seen"] = 0

    def reset_counts(self) -> None:
        if self.counter is not None:
            self.counter.reset_counts()
        if self.tracker is not None:
            self.tracker.center_points.clear()
            self.tracker.disappeared.clear()
        with self._state_lock:
            self._state["counts"]["max_inside_seen"] = 0

    def apply_thresholds(self) -> None:
        if self.notification_manager is not None:
            self.notification_manager.set_alert_rules(
                overcrowding_threshold=self.config.alert.overcrowding_threshold,
                warning_threshold=self.config.alert.warning_threshold,
                notification_interval=self.config.alert.notification_interval,
                quiet_hours=self.config.alert.quiet_hours,
            )

    # ------------------------------------------------------------------
    # Persistence

    def _persistence_loop(self, interval: int) -> None:
        while not self._persistence_stop.wait(interval):
            try:
                self.save_state_snapshot()
            except Exception:
                logger.exception("Periodic state snapshot failed")

    def save_state_snapshot(self) -> None:
        state = self.get_state()
        # ID 컬렉션도 보존
        if self.counter is not None:
            state["_entered_ids"] = sorted(self.counter.entered_ids)
            state["_exited_ids"] = sorted(self.counter.exited_ids)
        state["_saved_at"] = datetime.utcnow().isoformat() + "Z"
        with open(self._state_file, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def restore_state_snapshot(self) -> bool:
        if not self._state_file.exists():
            return False
        try:
            data = json.loads(self._state_file.read_text(encoding="utf-8"))
            if self.counter is not None:
                self.counter.entered_ids = set(data.get("_entered_ids", []))
                self.counter.exited_ids = set(data.get("_exited_ids", []))
            with self._state_lock:
                self._state["counts"]["max_inside_seen"] = (
                    data.get("counts", {}).get("max_inside_seen", 0)
                )
            return True
        except Exception:
            logger.exception("State restore failed")
            return False
