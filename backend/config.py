"""
config.py - 설정 관리 모듈
시스템 전체의 설정을 중앙 집중식으로 관리
"""

import os
import json
import logging
import yaml
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

try:
    from dotenv import load_dotenv
    # 프로젝트 루트의 .env 자동 로드 (이미 설정된 환경변수는 덮어쓰지 않음)
    load_dotenv()
except ImportError:
    # python-dotenv 미설치 시에도 동작은 하도록 — 환경변수만 사용
    pass

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """YOLOv8 모델 설정"""
    model_path: str = "yolov8s.pt"
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.4
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"
    image_size: int = 640
    max_det: int = 300
    half_precision: bool = False  # FP16 사용 여부
    
    def __post_init__(self):
        """설정 검증"""
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold는 0.0과 1.0 사이여야 합니다.")
        if not 0.0 <= self.iou_threshold <= 1.0:
            raise ValueError("iou_threshold는 0.0과 1.0 사이여야 합니다.")


@dataclass
class VideoConfig:
    """비디오 처리 설정"""
    input_source: str = ""  # 빈 문자열이면 웹캠, 경로 지정시 파일
    output_path: str = "output"
    frame_width: int = 800
    frame_height: int = 600
    fps_limit: int = 30
    frame_skip: int = 3  # 성능 향상을 위한 프레임 스킵
    save_video: bool = False
    video_codec: str = "mp4v"
    
    def __post_init__(self):
        """설정 검증"""
        if self.frame_width <= 0 or self.frame_height <= 0:
            raise ValueError("frame_width와 frame_height는 양수여야 합니다.")
        if self.fps_limit <= 0:
            raise ValueError("fps_limit는 양수여야 합니다.")


@dataclass
class TrackingConfig:
    """객체 추적 설정"""
    distance_threshold: int = 35  # 픽셀 단위
    max_disappeared: int = 10  # 객체가 사라진 것으로 판단할 프레임 수
    min_hits: int = 3  # 추적 시작을 위한 최소 히트 수
    max_age: int = 50  # 최대 추적 나이
    trajectory_length: int = 50  # 저장할 궤적 점 수
    
    def __post_init__(self):
        """설정 검증"""
        if self.distance_threshold <= 0:
            raise ValueError("distance_threshold는 양수여야 합니다.")


@dataclass
class CountingConfig:
    """카운팅 설정"""
    entrance_area: List[List[int]] = None
    exit_area: List[List[int]] = None
    counting_lines: List[Dict] = None
    min_residence_time: float = 0.5  # 영역 내 최소 머무는 시간 (초)
    area_name: str = "Building"
    
    def __post_init__(self):
        """기본 영역 설정"""
        if self.entrance_area is None:
            self.entrance_area = [[100, 300], [200, 300], [200, 400], [100, 400]]
        if self.exit_area is None:
            self.exit_area = [[300, 300], [400, 300], [400, 400], [300, 400]]
        if self.counting_lines is None:
            self.counting_lines = []


@dataclass
class AlertConfig:
    """알림 설정"""
    overcrowding_threshold: int = 50
    warning_threshold: int = 20
    notification_interval: int = 300  # 5분 (초)
    quiet_hours: tuple = (22, 6)  # 조용한 시간 (시작, 끝)
    emergency_contacts: List[str] = None
    enable_email: bool = True
    enable_slack: bool = False
    enable_discord: bool = False
    enable_webhook: bool = False
    
    def __post_init__(self):
        """기본값 설정"""
        if self.emergency_contacts is None:
            self.emergency_contacts = []


@dataclass
class EmailConfig:
    """이메일 설정"""
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    sender_email: str = ""
    sender_password: str = ""
    use_tls: bool = True
    timeout: int = 30
    
    def is_configured(self) -> bool:
        """이메일 설정이 완료되었는지 확인"""
        return bool(self.sender_email and self.sender_password)


@dataclass
class LocationConfig:
    """위치 정보 설정"""
    name: str = "Emergency Building"
    latitude: float = 37.5665
    longitude: float = 126.9780
    address: str = ""
    timezone: str = "Asia/Seoul"
    
    def __post_init__(self):
        """좌표 검증"""
        if not -90 <= self.latitude <= 90:
            raise ValueError("latitude는 -90과 90 사이여야 합니다.")
        if not -180 <= self.longitude <= 180:
            raise ValueError("longitude는 -180과 180 사이여야 합니다.")


@dataclass
class UIConfig:
    """UI 설정"""
    window_name: str = "Emergency Tracker"
    show_fps: bool = True
    show_count_info: bool = True
    show_trajectories: bool = False
    info_panel_position: str = "top_right"  # "top_left", "top_right", "bottom_left", "bottom_right"
    font_scale: float = 0.6
    line_thickness: int = 2
    
    # 색상 설정 (BGR 형식)
    colors: Dict[str, tuple] = None
    
    def __post_init__(self):
        """기본 색상 설정"""
        if self.colors is None:
            self.colors = {
                'entrance_area': (0, 255, 0),    # 녹색
                'exit_area': (0, 0, 255),        # 빨간색
                'counting_line': (255, 0, 255),  # 마젠타
                'person_box': (255, 255, 0),     # 청록색
                'person_center': (0, 255, 255),  # 노란색
                'trajectory': (255, 0, 0),       # 파란색
                'text_normal': (255, 255, 255),  # 흰색
                'text_warning': (0, 255, 255),   # 노란색
                'text_danger': (0, 0, 255),      # 빨간색
                'background': (0, 0, 0)          # 검은색
            }


class ConfigManager:
    """설정 관리자 클래스"""
    
    def __init__(self, config_path: str = "config.json"):
        """
        설정 관리자 초기화
        
        Args:
            config_path (str): 설정 파일 경로
        """
        self.config_path = Path(config_path)
        self.config_dir = self.config_path.parent
        
        # 기본 설정 인스턴스들
        self.model = ModelConfig()
        self.video = VideoConfig()
        self.tracking = TrackingConfig()
        self.counting = CountingConfig()
        self.alert = AlertConfig()
        self.email = EmailConfig()
        self.location = LocationConfig()
        self.ui = UIConfig()
        
        # 설정 파일이 존재하면 로드
        if self.config_path.exists():
            self.load_config()
        else:
            # 기본 설정으로 파일 생성
            self.save_config()

        # 환경변수(.env 포함)로 비밀/오버라이드 값 주입
        self._apply_env_overrides()
    
    def load_config(self, config_path: Optional[str] = None):
        """
        설정 파일 로드
        
        Args:
            config_path (Optional[str]): 설정 파일 경로 (None이면 기본 경로 사용)
        """
        if config_path:
            self.config_path = Path(config_path)
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                if self.config_path.suffix.lower() in ['.yaml', '.yml']:
                    config_data = yaml.safe_load(f)
                else:
                    config_data = json.load(f)
            
            # 각 섹션별로 설정 업데이트
            if 'model' in config_data:
                self.model = ModelConfig(**config_data['model'])
            if 'video' in config_data:
                self.video = VideoConfig(**config_data['video'])
            if 'tracking' in config_data:
                self.tracking = TrackingConfig(**config_data['tracking'])
            if 'counting' in config_data:
                self.counting = CountingConfig(**config_data['counting'])
            if 'alert' in config_data:
                self.alert = AlertConfig(**config_data['alert'])
            if 'email' in config_data:
                self.email = EmailConfig(**config_data['email'])
                if self.email.sender_password:
                    logger.warning(
                        "config.json에 sender_password가 평문으로 저장되어 있습니다. "
                        ".env 또는 환경변수(SENDER_PASSWORD) 사용을 권장합니다."
                    )
            if 'location' in config_data:
                self.location = LocationConfig(**config_data['location'])
            if 'ui' in config_data:
                self.ui = UIConfig(**config_data['ui'])
            
            print(f"설정 파일을 로드했습니다: {self.config_path}")
            
        except Exception:
            logger.exception("설정 파일 로드 실패 — 기본 설정을 사용합니다")
    
    def save_config(self, config_path: Optional[str] = None):
        """
        설정 파일 저장
        
        Args:
            config_path (Optional[str]): 설정 파일 경로 (None이면 기본 경로 사용)
        """
        if config_path:
            self.config_path = Path(config_path)
        
        # 디렉토리 생성
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        email_dict = asdict(self.email)
        # 비밀 정보는 디스크에 저장하지 않는다 — .env 또는 환경변수에서 주입
        email_dict['sender_password'] = ""

        config_data = {
            'model': asdict(self.model),
            'video': asdict(self.video),
            'tracking': asdict(self.tracking),
            'counting': asdict(self.counting),
            'alert': asdict(self.alert),
            'email': email_dict,
            'location': asdict(self.location),
            'ui': asdict(self.ui)
        }
        
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                if self.config_path.suffix.lower() in ['.yaml', '.yml']:
                    yaml.dump(config_data, f, default_flow_style=False, 
                             allow_unicode=True, indent=2)
                else:
                    json.dump(config_data, f, indent=2, ensure_ascii=False)
            
            print(f"설정 파일을 저장했습니다: {self.config_path}")

        except Exception:
            logger.exception("설정 파일 저장 실패")
    
    def _apply_env_overrides(self):
        """환경변수(.env 포함)로 비밀/오버라이드 값 주입.

        디스크에 저장하지 않을 비밀번호 같은 값은 이 단계에서만 메모리에 채워진다.
        """
        env = os.environ

        # 이메일 비밀번호 — 절대 디스크에 저장하지 않음
        if env.get('SENDER_PASSWORD'):
            self.email.sender_password = env['SENDER_PASSWORD']
        if env.get('SENDER_EMAIL'):
            self.email.sender_email = env['SENDER_EMAIL']
        if env.get('SMTP_SERVER'):
            self.email.smtp_server = env['SMTP_SERVER']
        if env.get('SMTP_PORT'):
            try:
                self.email.smtp_port = int(env['SMTP_PORT'])
            except ValueError:
                logger.warning("SMTP_PORT 환경변수가 정수가 아님: %s", env['SMTP_PORT'])

        # 외부 비밀번호 기반 알림 채널은 env에서만 가져오기
        if env.get('SLACK_WEBHOOK_URL'):
            # 알림 매니저에서 직접 참조하도록 별도 보관
            self.alert.enable_slack = True
        if env.get('DISCORD_WEBHOOK_URL'):
            self.alert.enable_discord = True

        # 모델/디바이스 오버라이드
        if env.get('YOLO_MODEL_PATH'):
            self.model.model_path = env['YOLO_MODEL_PATH']
        if env.get('DEVICE'):
            self.model.device = env['DEVICE']

    def update_model_config(self, **kwargs):
        """모델 설정 업데이트"""
        for key, value in kwargs.items():
            if hasattr(self.model, key):
                setattr(self.model, key, value)
            else:
                print(f"알 수 없는 모델 설정: {key}")
    
    def update_video_config(self, **kwargs):
        """비디오 설정 업데이트"""
        for key, value in kwargs.items():
            if hasattr(self.video, key):
                setattr(self.video, key, value)
            else:
                print(f"알 수 없는 비디오 설정: {key}")
    
    def update_counting_areas(self, entrance_area: List[List[int]] = None,
                            exit_area: List[List[int]] = None):
        """카운팅 영역 업데이트"""
        if entrance_area:
            self.counting.entrance_area = entrance_area
        if exit_area:
            self.counting.exit_area = exit_area
    
    def update_alert_config(self, **kwargs):
        """알림 설정 업데이트"""
        for key, value in kwargs.items():
            if hasattr(self.alert, key):
                setattr(self.alert, key, value)
            else:
                print(f"알 수 없는 알림 설정: {key}")
    
    def update_email_config(self, **kwargs):
        """이메일 설정 업데이트"""
        for key, value in kwargs.items():
            if hasattr(self.email, key):
                setattr(self.email, key, value)
            else:
                print(f"알 수 없는 이메일 설정: {key}")
    
    def update_location_config(self, **kwargs):
        """위치 설정 업데이트"""
        for key, value in kwargs.items():
            if hasattr(self.location, key):
                setattr(self.location, key, value)
            else:
                print(f"알 수 없는 위치 설정: {key}")
    
    def get_all_configs(self) -> Dict[str, Any]:
        """모든 설정을 딕셔너리로 반환"""
        return {
            'model': asdict(self.model),
            'video': asdict(self.video),
            'tracking': asdict(self.tracking),
            'counting': asdict(self.counting),
            'alert': asdict(self.alert),
            'email': asdict(self.email),
            'location': asdict(self.location),
            'ui': asdict(self.ui)
        }
    
    def validate_configs(self) -> List[str]:
        """설정 검증 및 오류 목록 반환"""
        errors = []
        
        # 모델 설정 검증
        if not os.path.exists(self.model.model_path) and not self.model.model_path.startswith('yolo'):
            errors.append(f"모델 파일을 찾을 수 없습니다: {self.model.model_path}")
        
        # 비디오 설정 검증
        if self.video.input_source and not os.path.exists(self.video.input_source):
            errors.append(f"비디오 파일을 찾을 수 없습니다: {self.video.input_source}")
        
        # 이메일 설정 검증
        if self.alert.enable_email and not self.email.is_configured():
            errors.append("이메일 알림이 활성화되었지만 이메일 설정이 불완전합니다.")
        
        # 카운팅 영역 검증
        if len(self.counting.entrance_area) < 3:
            errors.append("입구 영역은 최소 3개의 점이 필요합니다.")
        if len(self.counting.exit_area) < 3:
            errors.append("출구 영역은 최소 3개의 점이 필요합니다.")
        
        return errors
    
    def create_backup(self, backup_suffix: str = None):
        """설정 파일 백업 생성"""
        if not self.config_path.exists():
            print("백업할 설정 파일이 없습니다.")
            return
        
        if backup_suffix is None:
            from datetime import datetime
            backup_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        backup_path = self.config_path.with_name(
            f"{self.config_path.stem}_backup_{backup_suffix}{self.config_path.suffix}"
        )
        
        try:
            import shutil
            shutil.copy2(self.config_path, backup_path)
            print(f"설정 파일 백업 생성: {backup_path}")
        except Exception as e:
            print(f"백업 생성 실패: {e}")
    
    def restore_from_backup(self, backup_path: str):
        """백업에서 설정 복원"""
        backup_file = Path(backup_path)
        if not backup_file.exists():
            print(f"백업 파일을 찾을 수 없습니다: {backup_path}")
            return
        
        try:
            import shutil
            shutil.copy2(backup_file, self.config_path)
            self.load_config()
            print(f"백업에서 설정 복원 완료: {backup_path}")
        except Exception as e:
            print(f"백업 복원 실패: {e}")
    
    def export_template(self, template_path: str = "config_template.json"):
        """설정 템플릿 파일 내보내기"""
        template_config = ConfigManager()
        template_config.save_config(template_path)
        print(f"설정 템플릿을 내보냈습니다: {template_path}")
    
    def print_config_summary(self):
        """설정 요약 출력"""
        print("=" * 50)
        print("현재 설정 요약")
        print("=" * 50)
        
        print(f"📹 비디오 소스: {self.video.input_source or '웹캠'}")
        print(f"🤖 모델: {self.model.model_path}")
        print(f"🎯 신뢰도 임계값: {self.model.confidence_threshold}")
        print(f"🔍 추적 거리 임계값: {self.tracking.distance_threshold}픽셀")
        print(f"📍 위치: {self.location.name} ({self.location.latitude}, {self.location.longitude})")
        print(f"🚨 과밀집 임계값: {self.alert.overcrowding_threshold}명")
        print(f"⚠️  경고 임계값: {self.alert.warning_threshold}명")
        print(f"📧 이메일 알림: {'활성화' if self.alert.enable_email else '비활성화'}")
        print(f"🎨 UI 창 이름: {self.ui.window_name}")
        
        errors = self.validate_configs()
        if errors:
            print("\n❌ 설정 오류:")
            for error in errors:
                print(f"  - {error}")
        else:
            print("\n✅ 모든 설정이 유효합니다.")


class EnvironmentConfig:
    """환경변수 기반 설정 관리"""
    
    @staticmethod
    def load_from_env() -> ConfigManager:
        """환경변수에서 설정 로드"""
        config_manager = ConfigManager()
        
        # 모델 설정
        if os.getenv('YOLO_MODEL_PATH'):
            config_manager.model.model_path = os.getenv('YOLO_MODEL_PATH')
        if os.getenv('CONFIDENCE_THRESHOLD'):
            config_manager.model.confidence_threshold = float(os.getenv('CONFIDENCE_THRESHOLD'))
        if os.getenv('DEVICE'):
            config_manager.model.device = os.getenv('DEVICE')
        
        # 비디오 설정
        if os.getenv('VIDEO_SOURCE'):
            config_manager.video.input_source = os.getenv('VIDEO_SOURCE')
        if os.getenv('FRAME_WIDTH'):
            config_manager.video.frame_width = int(os.getenv('FRAME_WIDTH'))
        if os.getenv('FRAME_HEIGHT'):
            config_manager.video.frame_height = int(os.getenv('FRAME_HEIGHT'))
        
        # 알림 설정
        if os.getenv('OVERCROWDING_THRESHOLD'):
            config_manager.alert.overcrowding_threshold = int(os.getenv('OVERCROWDING_THRESHOLD'))
        if os.getenv('WARNING_THRESHOLD'):
            config_manager.alert.warning_threshold = int(os.getenv('WARNING_THRESHOLD'))
        
        # 이메일 설정
        if os.getenv('SMTP_SERVER'):
            config_manager.email.smtp_server = os.getenv('SMTP_SERVER')
        if os.getenv('SMTP_PORT'):
            config_manager.email.smtp_port = int(os.getenv('SMTP_PORT'))
        if os.getenv('SENDER_EMAIL'):
            config_manager.email.sender_email = os.getenv('SENDER_EMAIL')
        if os.getenv('SENDER_PASSWORD'):
            config_manager.email.sender_password = os.getenv('SENDER_PASSWORD')
        
        # 위치 설정
        if os.getenv('LOCATION_NAME'):
            config_manager.location.name = os.getenv('LOCATION_NAME')
        if os.getenv('LATITUDE'):
            config_manager.location.latitude = float(os.getenv('LATITUDE'))
        if os.getenv('LONGITUDE'):
            config_manager.location.longitude = float(os.getenv('LONGITUDE'))
        
        return config_manager


def create_default_config_file(config_path: str = "config.json"):
    """기본 설정 파일 생성"""
    config_manager = ConfigManager()
    
    # 일부 예시 설정
    config_manager.counting.entrance_area = [
        [215, 655], [284, 886], [165, 900], [137, 655]
    ]
    config_manager.counting.exit_area = [
        [129, 654], [158, 902], [75, 911], [51, 658]
    ]
    
    config_manager.location.name = "Emergency Building"
    config_manager.location.latitude = 37.5665
    config_manager.location.longitude = 126.9780
    
    config_manager.alert.emergency_contacts = [
        "emergency@company.com",
        "security@company.com"
    ]
    
    config_manager.save_config(config_path)
    print(f"기본 설정 파일이 생성되었습니다: {config_path}")


if __name__ == "__main__":
    # 테스트 코드
    print("=== 설정 관리 모듈 테스트 ===")
    
    # 1. 기본 설정 관리자 생성
    print("\n1. 기본 설정 관리자 생성")
    config_manager = ConfigManager("test_config.json")
    
    # 2. 설정 업데이트
    print("\n2. 설정 업데이트 테스트")
    config_manager.update_model_config(
        confidence_threshold=0.6,
        device="cpu"
    )
    config_manager.update_alert_config(
        overcrowding_threshold=40,
        warning_threshold=25
    )
    config_manager.update_location_config(
        name="Test Building",
        latitude=37.5665,
        longitude=126.9780
    )
    
    # 3. 카운팅 영역 설정
    print("\n3. 카운팅 영역 설정")
    entrance_area = [[100, 200], [200, 200], [200, 300], [100, 300]]
    exit_area = [[300, 200], [400, 200], [400, 300], [300, 300]]
    config_manager.update_counting_areas(entrance_area, exit_area)
    
    # 4. 설정 저장
    print("\n4. 설정 저장")
    config_manager.save_config()
    
    # 5. 설정 요약 출력
    print("\n5. 설정 요약")
    config_manager.print_config_summary()
    
    # 6. 설정 검증
    print("\n6. 설정 검증")
    errors = config_manager.validate_configs()
    if errors:
        print("설정 오류:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("✅ 모든 설정이 유효합니다.")
    
    # 7. 백업 생성
    print("\n7. 백업 생성")
    config_manager.create_backup()
    
    # 8. YAML 형식으로 저장
    print("\n8. YAML 형식으로 저장")
    config_manager.save_config("test_config.yaml")
    
    # 9. 환경변수 기반 설정 테스트
    print("\n9. 환경변수 기반 설정 테스트")
    os.environ['CONFIDENCE_THRESHOLD'] = '0.7'
    os.environ['OVERCROWDING_THRESHOLD'] = '35'
    env_config = EnvironmentConfig.load_from_env()
    print(f"환경변수에서 로드된 신뢰도 임계값: {env_config.model.confidence_threshold}")
    print(f"환경변수에서 로드된 과밀집 임계값: {env_config.alert.overcrowding_threshold}")
    
    # 10. 기본 설정 파일 생성 테스트
    print("\n10. 기본 설정 파일 생성")
    create_default_config_file("default_config.json")
    
    print("\n✅ 모든 테스트 완료!")
    print("생성된 파일들:")
    print("- test_config.json")
    print("- test_config.yaml") 
    print("- default_config.json")
    print("- test_config_backup_*.json (백업 파일)")
