"""
area_setup.py - 카운팅 영역 대화형 설정 도구
마우스로 클릭하여 입구/출구 영역을 쉽게 설정할 수 있습니다.
"""

import cv2
import logging
import numpy as np
import json
import argparse
from pathlib import Path

logger = logging.getLogger(__name__)


class AreaSetupTool:
    """카운팅 영역 설정 도구"""
    
    def __init__(self, video_path=None, config_path="config.json"):
        self.video_path = video_path
        self.config_path = config_path
        
        # 영역 설정
        self.entrance_points = []
        self.exit_points = []
        self.current_mode = "entrance"  # "entrance" or "exit"
        
        # UI 상태
        self.frame = None
        self.display_frame = None
        
        # 색상 설정
        self.colors = {
            'entrance': (0, 255, 0),      # 녹색
            'exit': (0, 0, 255),          # 빨간색
            'entrance_fill': (0, 255, 0, 50),  # 반투명 녹색
            'exit_fill': (0, 0, 255, 50),       # 반투명 빨간색
            'point': (255, 255, 0),       # 노란색
            'text': (255, 255, 255),      # 흰색
            'background': (0, 0, 0)       # 검은색
        }
    
    def load_frame(self):
        """비디오에서 첫 번째 프레임 로드"""
        if self.video_path:
            cap = cv2.VideoCapture(self.video_path)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                # 화면 크기에 맞게 조정
                height, width = frame.shape[:2]
                if width > 1200:
                    scale = 1200 / width
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    frame = cv2.resize(frame, (new_width, new_height))

                self.frame = frame
                return True
            else:
                logger.error("비디오에서 프레임을 읽을 수 없습니다: %s", self.video_path)
                return False
        else:
            # 웹캠에서 프레임 캡처
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            cap.release()

            if ret:
                self.frame = frame
                return True
            else:
                logger.error("웹캠에서 프레임을 읽을 수 없습니다.")
                return False
    
    def mouse_callback(self, event, x, y, flags, param):
        """마우스 클릭 이벤트 처리"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.current_mode == "entrance":
                self.entrance_points.append((x, y))
                print(f"입구 영역 점 추가: ({x}, {y}) - 총 {len(self.entrance_points)}개")
            elif self.current_mode == "exit":
                self.exit_points.append((x, y))
                print(f"출구 영역 점 추가: ({x}, {y}) - 총 {len(self.exit_points)}개")
            
            self.update_display()
    
    def update_display(self):
        """화면 업데이트"""
        self.display_frame = self.frame.copy()
        
        # 입구 영역 그리기
        if len(self.entrance_points) > 0:
            # 점들 그리기
            for point in self.entrance_points:
                cv2.circle(self.display_frame, point, 5, self.colors['point'], -1)
            
            # 영역 연결선 그리기
            if len(self.entrance_points) > 1:
                pts = np.array(self.entrance_points, np.int32)
                cv2.polylines(self.display_frame, [pts], False, self.colors['entrance'], 2)
            
            # 영역이 완성된 경우 (4개 점 이상)
            if len(self.entrance_points) >= 3:
                pts = np.array(self.entrance_points, np.int32)
                cv2.polylines(self.display_frame, [pts], True, self.colors['entrance'], 2)
                
                # 반투명 채우기
                overlay = self.display_frame.copy()
                cv2.fillPoly(overlay, [pts], self.colors['entrance'])
                cv2.addWeighted(overlay, 0.3, self.display_frame, 0.7, 0, self.display_frame)
        
        # 출구 영역 그리기
        if len(self.exit_points) > 0:
            # 점들 그리기
            for point in self.exit_points:
                cv2.circle(self.display_frame, point, 5, self.colors['point'], -1)
            
            # 영역 연결선 그리기
            if len(self.exit_points) > 1:
                pts = np.array(self.exit_points, np.int32)
                cv2.polylines(self.display_frame, [pts], False, self.colors['exit'], 2)
            
            # 영역이 완성된 경우
            if len(self.exit_points) >= 3:
                pts = np.array(self.exit_points, np.int32)
                cv2.polylines(self.display_frame, [pts], True, self.colors['exit'], 2)
                
                # 반투명 채우기
                overlay = self.display_frame.copy()
                cv2.fillPoly(overlay, [pts], self.colors['exit'])
                cv2.addWeighted(overlay, 0.3, self.display_frame, 0.7, 0, self.display_frame)
        
        # 정보 패널 그리기
        self.draw_info_panel()
    
    def draw_info_panel(self):
        """정보 패널 그리기"""
        # 패널 배경
        panel_height = 150
        cv2.rectangle(self.display_frame, (10, 10), (400, panel_height), 
                     self.colors['background'], -1)
        cv2.rectangle(self.display_frame, (10, 10), (400, panel_height), 
                     self.colors['text'], 2)
        
        # 현재 모드 표시
        mode_text = f"현재 모드: {'입구 영역' if self.current_mode == 'entrance' else '출구 영역'}"
        mode_color = self.colors['entrance'] if self.current_mode == 'entrance' else self.colors['exit']
        cv2.putText(self.display_frame, mode_text, (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)
        
        # 점 개수 표시
        entrance_count = len(self.entrance_points)
        exit_count = len(self.exit_points)
        cv2.putText(self.display_frame, f"입구 영역 점: {entrance_count}개", 
                   (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['entrance'], 1)
        cv2.putText(self.display_frame, f"출구 영역 점: {exit_count}개", 
                   (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['exit'], 1)
        
        # 키보드 단축키 안내
        cv2.putText(self.display_frame, "키보드 단축키:", (20, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
        cv2.putText(self.display_frame, "E: 입구모드, X: 출구모드, R: 리셋, S: 저장, Q: 종료", 
                   (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text'], 1)
    
    def save_config(self):
        """설정을 config.json에 저장"""
        try:
            # 기존 설정 파일 로드
            if Path(self.config_path).exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
            else:
                config = {}
            
            # counting 섹션이 없으면 생성
            if 'counting' not in config:
                config['counting'] = {}
            
            # 영역 정보 저장
            if len(self.entrance_points) >= 3:
                config['counting']['entrance_area'] = self.entrance_points
                print(f"✅ 입구 영역 저장됨: {len(self.entrance_points)}개 점")
            
            if len(self.exit_points) >= 3:
                config['counting']['exit_area'] = self.exit_points
                print(f"✅ 출구 영역 저장됨: {len(self.exit_points)}개 점")
            
            # 파일에 저장
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            print(f"✅ 설정이 저장되었습니다: {self.config_path}")
            return True
            
        except Exception:
            logger.exception("설정 저장 실패")
            return False
    
    def run(self):
        """메인 실행 루프"""
        print("🎯 카운팅 영역 설정 도구")
        print("=" * 50)
        
        # 프레임 로드
        if not self.load_frame():
            return
        
        # 초기 화면 업데이트
        self.update_display()
        
        # 윈도우 생성 및 마우스 콜백 설정
        cv2.namedWindow('Area Setup Tool', cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback('Area Setup Tool', self.mouse_callback)
        
        print("\n📝 사용법:")
        print("1. 마우스 클릭으로 영역의 모서리 점들을 선택하세요")
        print("2. 최소 3개 이상의 점을 선택하면 영역이 완성됩니다")
        print("3. 'E' 키로 입구 모드, 'X' 키로 출구 모드 전환")
        print("4. 'R' 키로 현재 영역 리셋, 'S' 키로 저장")
        print("5. 'Q' 키로 종료")
        print()
        
        while True:
            cv2.imshow('Area Setup Tool', self.display_frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("👋 프로그램을 종료합니다.")
                break
            elif key == ord('e'):
                self.current_mode = "entrance"
                print("🟢 입구 영역 설정 모드로 전환")
                self.update_display()
            elif key == ord('x'):
                self.current_mode = "exit"
                print("🔴 출구 영역 설정 모드로 전환")
                self.update_display()
            elif key == ord('r'):
                if self.current_mode == "entrance":
                    self.entrance_points = []
                    print("🔄 입구 영역 리셋")
                else:
                    self.exit_points = []
                    print("🔄 출구 영역 리셋")
                self.update_display()
            elif key == ord('s'):
                if self.save_config():
                    print("💾 설정 저장 완료!")
                else:
                    print("❌ 설정 저장 실패!")
        
        cv2.destroyAllWindows()
        
        # 최종 결과 출력
        print("\n" + "=" * 50)
        print("📊 최종 설정 결과:")
        print(f"입구 영역: {len(self.entrance_points)}개 점 - {self.entrance_points}")
        print(f"출구 영역: {len(self.exit_points)}개 점 - {self.exit_points}")


def main():
    """메인 함수"""
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="카운팅 영역 설정 도구")
    parser.add_argument("--video", "-v", type=str, help="비디오 파일 경로 (없으면 웹캠 사용)")
    parser.add_argument("--config", "-c", type=str, default="config.json", help="설정 파일 경로")
    
    args = parser.parse_args()
    
    # 도구 실행
    tool = AreaSetupTool(args.video, args.config)
    tool.run()


if __name__ == "__main__":
    main()
