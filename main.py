"""
main.py - 메인 실행 모듈
YOLOv8 기반 긴급상황 객체 추적 시스템의 메인 실행 파일
"""

import cv2
import numpy as np
import argparse
import time
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple

# 커스텀 모듈들 임포트
from detector import YOLODetector, draw_detections
from tracker import Tracker
from counter import AreaCounter
from notification import NotificationManager
from visualization import MapVisualizer, DashboardVisualizer
from config import ConfigManager


class EmergencyTracker:
    """긴급상황 추적 시스템 메인 클래스"""
    
    def __init__(self, config_path: str = "config.json"):
        """
        시스템 초기화
        
        Args:
            config_path (str): 설정 파일 경로
        """
        # 설정 로드
        self.config = ConfigManager(config_path)
        
        # 컴포넌트 초기화
        self.detector = None
        self.tracker = None
        self.counter = None
        self.notification_manager = None
        self.map_visualizer = None
        self.dashboard_visualizer = None
        
        # 시스템 상태
        self.is_running = False
        self.frame_count = 0
        self.start_time = None
        
        # 통계 데이터
        self.statistics = {
            'total_frames': 0,
            'total_detections': 0,
            'processing_times': [],
            'fps_history': []
        }
        
        # 시그널 핸들러 설정
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # 컴포넌트 초기화 실행
        self._initialize_components()
    
    def _initialize_components(self):
        """모든 컴포넌트 초기화"""
        try:
            print("🤖 YOLOv8 탐지기 초기화 중...")
            self.detector = YOLODetector(
                model_path=self.config.model.model_path,
                confidence_threshold=self.config.model.confidence_threshold,
                device=self.config.model.device
            )
            
            print("🔍 객체 추적기 초기화 중...")
            self.tracker = Tracker(
                distance_threshold=self.config.tracking.distance_threshold
            )
            
            print("📊 카운터 초기화 중...")
            self.counter = AreaCounter(
                entrance_area=self.config.counting.entrance_area,
                exit_area=self.config.counting.exit_area,
                area_name=self.config.counting.area_name
            )
            
            print("🔔 알림 관리자 초기화 중...")
            self.notification_manager = NotificationManager()
            
            # 이메일 설정이 있으면 설정
            if self.config.email.is_configured():
                self.notification_manager.configure_email(
                    sender_email=self.config.email.sender_email,
                    sender_password=self.config.email.sender_password,
                    emergency_contacts=self.config.alert.emergency_contacts
                )
            
            # 알림 규칙 설정
            self.notification_manager.set_alert_rules(
                overcrowding_threshold=self.config.alert.overcrowding_threshold,
                warning_threshold=self.config.alert.warning_threshold,
                notification_interval=self.config.alert.notification_interval,
                quiet_hours=self.config.alert.quiet_hours
            )
            
            print("🗺️  시각화 도구 초기화 중...")
            self.map_visualizer = MapVisualizer()
            self.dashboard_visualizer = DashboardVisualizer()
            
            print("✅ 모든 컴포넌트 초기화 완료")
            
        except Exception as e:
            print(f"❌ 컴포넌트 초기화 실패: {e}")
            sys.exit(1)
    
    def _signal_handler(self, signum, frame):
        """시그널 핸들러"""
        print(f"\n⚠️  시그널 {signum} 수신, 시스템 종료 중...")
        self.stop()
    
    def _setup_video_capture(self) -> cv2.VideoCapture:
        """비디오 캡처 설정"""
        video_source = self.config.video.input_source
        
        if not video_source:
            print("📹 웹캠 연결 중...")
            cap = cv2.VideoCapture(0)
        else:
            print(f"📹 비디오 파일 로드 중: {video_source}")
            cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            raise RuntimeError("비디오 소스를 열 수 없습니다.")
        
        # 비디오 속성 설정
        if not video_source:  # 웹캠인 경우
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.video.frame_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.video.frame_height)
            cap.set(cv2.CAP_PROP_FPS, self.config.video.fps_limit)
        
        return cap
    
    def _process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """단일 프레임 처리"""
        start_time = time.time()
        
        # 1. 객체 탐지
        person_boxes = self.detector.detect_persons_only(frame)
        
        # 2. 객체 추적
        tracked_objects = self.tracker.update(person_boxes)
        
        # 3. 입출입 카운팅
        self.counter.update(tracked_objects)
        counts = self.counter.get_counts()
        
        # 4. 프레임에 결과 그리기
        result_frame = self._draw_results(frame, tracked_objects, counts)
        
        # 5. 처리 시간 기록
        processing_time = time.time() - start_time
        self.statistics['processing_times'].append(processing_time)
        self.statistics['total_detections'] += len(person_boxes)
        
        return result_frame, counts
    
    def _draw_results(self, frame: np.ndarray, tracked_objects: List, 
                     counts: Dict) -> np.ndarray:
        """프레임에 결과 그리기"""
        result_frame = frame.copy()
        
        # 추적 영역 그리기
        result_frame = self.counter.draw_areas(result_frame, show_counts=False)
        
        # 추적된 객체 그리기
        for obj in tracked_objects:
            x1, y1, x2, y2, obj_id = obj
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            
            # 바운딩 박스
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), 
                         self.config.ui.colors['person_box'], 2)
            
            # 중심점
            cv2.circle(result_frame, (center_x, center_y), 4, 
                      self.config.ui.colors['person_center'], -1)
            
            # ID 라벨
            cv2.putText(result_frame, f'ID: {obj_id}', 
                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                       self.config.ui.font_scale, 
                       self.config.ui.colors['text_normal'], 2)
        
        # 정보 패널 그리기
        self._draw_info_panel(result_frame, counts)
        
        return result_frame
    
    def _draw_info_panel(self, frame: np.ndarray, counts: Dict):
        """정보 패널 그리기"""
        panel_height = 200
        panel_width = 400
        
        # 위치 계산
        if self.config.ui.info_panel_position == "top_right":
            panel_x = frame.shape[1] - panel_width - 10
            panel_y = 10
        elif self.config.ui.info_panel_position == "top_left":
            panel_x = 10
            panel_y = 10
        elif self.config.ui.info_panel_position == "bottom_right":
            panel_x = frame.shape[1] - panel_width - 10
            panel_y = frame.shape[0] - panel_height - 10
        else:  # bottom_left
            panel_x = 10
            panel_y = frame.shape[0] - panel_height - 10
        
        # 패널 배경
        cv2.rectangle(frame, 
                     (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height),
                     self.config.ui.colors['background'], -1)
        cv2.rectangle(frame, 
                     (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height),
                     self.config.ui.colors['text_normal'], 2)
        
        # 정보 텍스트
        y_offset = 30
        info_lines = [
            f"Location: {self.config.location.name}",
            f"Entered: {counts.get('entered', 0)}",
            f"Exited: {counts.get('exited', 0)}",
            f"Current Inside: {counts.get('current_inside', 0)}",
            f"Time: {datetime.now().strftime('%H:%M:%S')}"
        ]
        
        # FPS 정보 추가
        if self.config.ui.show_fps and self.statistics['fps_history']:
            current_fps = self.statistics['fps_history'][-1]
            info_lines.append(f"FPS: {current_fps:.1f}")
        
        # 상태 정보
        current_inside = counts.get('current_inside', 0)
        if current_inside >= self.config.alert.overcrowding_threshold:
            status = "EMERGENCY"
            status_color = self.config.ui.colors['text_danger']
        elif current_inside >= self.config.alert.warning_threshold:
            status = "WARNING"
            status_color = self.config.ui.colors['text_warning']
        else:
            status = "NORMAL"
            status_color = self.config.ui.colors['text_normal']
        
        info_lines.append(f"Status: {status}")
        
        # 텍스트 그리기
        for i, line in enumerate(info_lines):
            y_pos = panel_y + y_offset + i * 25
            
            # 상태 라인에는 특별한 색상 적용
            if "Status:" in line:
                color = status_color
            else:
                color = self.config.ui.colors['text_normal']
            
            cv2.putText(frame, line, (panel_x + 20, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       self.config.ui.font_scale, color, 1)
    
    def _calculate_fps(self) -> float:
        """FPS 계산"""
        if len(self.statistics['processing_times']) < 10:
            return 0.0
        
        recent_times = self.statistics['processing_times'][-10:]
        avg_time = sum(recent_times) / len(recent_times)
        return 1.0 / avg_time if avg_time > 0 else 0.0
    
    def run(self):
        """메인 실행 루프"""
        print("🚀 긴급상황 추적 시스템 시작")
        print("키보드 단축키:")
        print("  q: 종료")
        print("  e: 긴급상황 이메일 발송")
        print("  m: 지도 생성")
        print("  r: 카운터 리셋")
        print("  s: 통계 저장")
        print()
        
        # 비디오 캡처 설정
        try:
            cap = self._setup_video_capture()
        except Exception as e:
            print(f"❌ 비디오 캡처 설정 실패: {e}")
            return
        
        self.is_running = True
        self.start_time = time.time()
        
        # 비디오 저장 설정 (선택적)
        video_writer = None
        if self.config.video.save_video:
            fourcc = cv2.VideoWriter_fourcc(*self.config.video.video_codec)
            output_path = Path(self.config.video.output_path) / f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            print(f"📹 비디오 저장 경로: {output_path}")
        
        try:
            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    if self.config.video.input_source:  # 비디오 파일인 경우
                        print("📹 비디오 파일 재생 완료")
                        break
                    else:
                        continue
                
                self.frame_count += 1
                self.statistics['total_frames'] += 1
                
                # 프레임 스킵 처리
                if self.frame_count % self.config.video.frame_skip != 0:
                    continue
                
                # 프레임 크기 조정
                if frame.shape[:2] != (self.config.video.frame_height, self.config.video.frame_width):
                    frame = cv2.resize(frame, (self.config.video.frame_width, self.config.video.frame_height))
                
                # 프레임 처리
                result_frame, counts = self._process_frame(frame)
                
                # FPS 계산 및 기록
                current_fps = self._calculate_fps()
                if current_fps > 0:
                    self.statistics['fps_history'].append(current_fps)
                    if len(self.statistics['fps_history']) > 100:
                        self.statistics['fps_history'] = self.statistics['fps_history'][-100:]
                
                # 알림 확인 및 발송
                current_inside = counts.get('current_inside', 0)
                location_info = {
                    'name': self.config.location.name,
                    'lat': self.config.location.latitude,
                    'lon': self.config.location.longitude
                }
                
                self.notification_manager.check_and_send_alerts(
                    current_inside, location_info
                )
                
                # 결과 표시
                cv2.imshow(self.config.ui.window_name, result_frame)
                
                # 비디오 저장
                if video_writer:
                    video_writer.write(result_frame)
                
                # 키보드 입력 처리
                key = cv2.waitKey(1) & 0xFF
                if not self._handle_keyboard_input(key, counts, location_info):
                    break
                
                # 주기적 상태 출력
                if self.frame_count % 300 == 0:  # 약 10초마다 (30fps 기준)
                    self._print_status(counts)
        
        except KeyboardInterrupt:
            print("\n⚠️  사용자에 의해 중단되었습니다.")
        except Exception as e:
            print(f"❌ 실행 중 오류 발생: {e}")
        finally:
            # 리소스 정리
            cap.release()
            if video_writer:
                video_writer.release()
            cv2.destroyAllWindows()
            
            # 최종 통계 출력
            self._print_final_statistics()
            
            # 히스토리 저장
            self._save_session_data()
    
    def _handle_keyboard_input(self, key: int, counts: Dict, location_info: Dict) -> bool:
        """
        키보드 입력 처리
        
        Returns:
            bool: 계속 실행할지 여부
        """
        if key == ord('q'):  # 종료
            print("📴 시스템 종료 중...")
            return False
        
        elif key == ord('e'):  # 긴급상황 이메일 발송
            print("📧 긴급상황 이메일 발송 중...")
            success = self.notification_manager.check_and_send_alerts(
                counts.get('current_inside', 0),
                location_info,
                "수동으로 긴급상황 알림을 발송합니다."
            )
            if success:
                print("✅ 이메일 발송 완료")
            else:
                print("❌ 이메일 발송 실패")
        
        elif key == ord('m'):  # 지도 생성
            print("🗺️  지도 생성 중...")
            self._create_emergency_map(counts, location_info)
        
        elif key == ord('r'):  # 카운터 리셋
            print("🔄 카운터 리셋")
            self.counter.reset_counts()
            self.tracker.reset()
        
        elif key == ord('s'):  # 통계 저장
            print("💾 통계 저장 중...")
            self._save_statistics()
        
        elif key == ord('d'):  # 대시보드 생성
            print("📊 대시보드 생성 중...")
            self._create_dashboard(counts)
        
        elif key == ord('c'):  # 설정 출력
            print("⚙️  현재 설정:")
            self.config.print_config_summary()
        
        return True
    
    def _create_emergency_map(self, counts: Dict, location_info: Dict):
        """긴급상황 지도 생성"""
        try:
            locations = [{
                'name': location_info['name'],
                'lat': location_info['lat'],
                'lon': location_info['lon'],
                'people_count': counts.get('current_inside', 0),
                'status': self._get_alert_status(counts.get('current_inside', 0)),
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }]
            
            emergency_map = self.map_visualizer.create_emergency_map(locations)
            
            filename = f"emergency_map_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            self.map_visualizer.save_map(emergency_map, filename, auto_open=True)
            print(f"✅ 지도가 생성되었습니다: {filename}")
            
        except Exception as e:
            print(f"❌ 지도 생성 실패: {e}")
    
    def _create_dashboard(self, counts: Dict):
        """대시보드 생성"""
        try:
            # 현재 데이터 준비
            current_data = {
                'buildings': {
                    self.config.location.name: counts.get('current_inside', 0)
                },
                'max_capacity': self.config.alert.overcrowding_threshold * 2,
                'alert_distribution': self._calculate_alert_distribution(),
                'daily_summary': {
                    '총 입장': counts.get('entered', 0),
                    '총 퇴장': counts.get('exited', 0),
                    '현재 인원': counts.get('current_inside', 0),
                    '최대 동시 인원': self._get_max_occupancy(),
                    '평균 FPS': f"{np.mean(self.statistics['fps_history']):.1f}" if self.statistics['fps_history'] else "0.0"
                }
            }
            
            dashboard_fig = self.dashboard_visualizer.create_realtime_dashboard(current_data)
            
            filename = f"dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            self.dashboard_visualizer.save_dashboard(filename, auto_open=True)
            print(f"✅ 대시보드가 생성되었습니다: {filename}")
            
        except Exception as e:
            print(f"❌ 대시보드 생성 실패: {e}")
    
    def _get_alert_status(self, people_count: int) -> str:
        """알림 상태 반환"""
        if people_count >= self.config.alert.overcrowding_threshold:
            return 'emergency'
        elif people_count >= self.config.alert.warning_threshold:
            return 'warning'
        else:
            return 'normal'
    
    def _calculate_alert_distribution(self) -> Dict:
        """알림 레벨 분포 계산"""
        # 실제 구현시에는 과거 데이터를 기반으로 계산
        return {'정상': 70, '주의': 20, '위험': 10}
    
    def _get_max_occupancy(self) -> int:
        """최대 동시 수용 인원 반환"""
        # 실제 구현시에는 세션 중 최대값 추적
        return max(20, self.counter.get_counts().get('current_inside', 0))
    
    def _print_status(self, counts: Dict):
        """현재 상태 출력"""
        runtime = time.time() - self.start_time
        runtime_str = f"{int(runtime//3600):02d}:{int((runtime%3600)//60):02d}:{int(runtime%60):02d}"
        
        avg_fps = np.mean(self.statistics['fps_history']) if self.statistics['fps_history'] else 0
        
        print(f"⏰ {runtime_str} | "
              f"👥 {counts.get('current_inside', 0)}명 | "
              f"📈 입장: {counts.get('entered', 0)} | "
              f"📉 퇴장: {counts.get('exited', 0)} | "
              f"🎯 FPS: {avg_fps:.1f}")
    
    def _print_final_statistics(self):
        """최종 통계 출력"""
        if not self.start_time:
            return
        
        runtime = time.time() - self.start_time
        counts = self.counter.get_counts()
        
        print("\n" + "="*60)
        print("📊 세션 종료 - 최종 통계")
        print("="*60)
        print(f"⏰ 총 실행 시간: {runtime:.1f}초")
        print(f"🎬 처리된 프레임: {self.statistics['total_frames']}")
        print(f"🎯 총 탐지 수: {self.statistics['total_detections']}")
        print(f"👥 최종 현재 인원: {counts.get('current_inside', 0)}명")
        print(f"📈 총 입장자: {counts.get('entered', 0)}명")
        print(f"📉 총 퇴장자: {counts.get('exited', 0)}명")
        
        if self.statistics['processing_times']:
            avg_processing_time = np.mean(self.statistics['processing_times'])
            print(f"⚡ 평균 처리 시간: {avg_processing_time:.4f}초")
            print(f"📺 평균 FPS: {1/avg_processing_time:.1f}")
        
        if self.statistics['fps_history']:
            print(f"📊 최대 FPS: {max(self.statistics['fps_history']):.1f}")
            print(f"📊 최소 FPS: {min(self.statistics['fps_history']):.1f}")
        
        print("="*60)
    
    def _save_statistics(self):
        """통계 데이터 저장"""
        try:
            stats_data = {
                'session_info': {
                    'start_time': self.start_time,
                    'end_time': time.time(),
                    'location': self.config.location.name,
                    'model': self.config.model.model_path
                },
                'statistics': self.statistics,
                'final_counts': self.counter.get_counts(),
                'config_summary': self.config.get_all_configs()
            }
            
            # 출력 디렉토리 생성
            output_dir = Path(self.config.video.output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 통계 파일 저장
            stats_file = output_dir / f"statistics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            import json
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats_data, f, indent=2, ensure_ascii=False, default=str)
            
            print(f"✅ 통계가 저장되었습니다: {stats_file}")
            
        except Exception as e:
            print(f"❌ 통계 저장 실패: {e}")
    
    def _save_session_data(self):
        """세션 데이터 저장"""
        try:
            # 카운터 히스토리 저장
            history_file = Path(self.config.video.output_path) / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            history_file.parent.mkdir(parents=True, exist_ok=True)
            self.counter.save_history(str(history_file))
            
            # 자동 통계 저장
            self._save_statistics()
            
        except Exception as e:
            print(f"❌ 세션 데이터 저장 실패: {e}")
    
    def stop(self):
        """시스템 중지"""
        self.is_running = False


def parse_arguments():
    """명령행 인수 파싱"""
    parser = argparse.ArgumentParser(description="YOLOv8 긴급상황 추적 시스템")
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config.json",
        help="설정 파일 경로 (기본: config.json)"
    )
    
    parser.add_argument(
        "--video", "-v",
        type=str,
        help="비디오 파일 경로 (지정하지 않으면 웹캠 사용)"
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        help="YOLOv8 모델 경로"
    )
    
    parser.add_argument(
        "--confidence", "-conf",
        type=float,
        help="신뢰도 임계값 (0.0-1.0)"
    )
    
    parser.add_argument(
        "--device", "-d",
        type=str,
        choices=["auto", "cpu", "cuda", "mps"],
        help="사용할 디바이스"
    )
    
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="결과 비디오 저장"
    )
    
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="화면 표시 비활성화 (서버 모드)"
    )
    
    return parser.parse_args()


def main():
    """메인 함수"""
    print("🚨 YOLOv8 긴급상황 추적 시스템")
    print("=" * 50)
    
    # 명령행 인수 파싱
    args = parse_arguments()
    
    # 시스템 초기화
    try:
        tracker_system = EmergencyTracker(args.config)
    except Exception as e:
        print(f"❌ 시스템 초기화 실패: {e}")
        sys.exit(1)
    
    # 명령행 인수로 설정 오버라이드
    if args.video:
        tracker_system.config.video.input_source = args.video
    
    if args.model:
        tracker_system.config.model.model_path = args.model
    
    if args.confidence:
        tracker_system.config.model.confidence_threshold = args.confidence
    
    if args.device:
        tracker_system.config.model.device = args.device
    
    if args.save_video:
        tracker_system.config.video.save_video = True
    
    # 설정 검증
    config_errors = tracker_system.config.validate_configs()
    if config_errors:
        print("⚠️  설정 오류가 발견되었습니다:")
        for error in config_errors:
            print(f"  - {error}")
        
        response = input("계속 진행하시겠습니까? (y/N): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # 시스템 실행
    if args.no_display:
        print("🖥️  서버 모드로 실행 (화면 표시 비활성화)")
        # 서버 모드 구현 (추후 확장 가능)
    
    try:
        tracker_system.run()
    except Exception as e:
        print(f"❌ 실행 중 오류: {e}")
        sys.exit(1)
    
    print("👋 시스템이 정상적으로 종료되었습니다.")


if __name__ == "__main__":
    main()