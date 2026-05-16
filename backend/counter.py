"""
counter.py - 입출입 카운팅 모듈
지정된 영역을 통과하는 객체들의 입장/퇴장을 카운팅하는 모듈
"""

import cv2
import numpy as np
from collections import defaultdict
from datetime import datetime
import json


class AreaCounter:
    """영역 기반 입출입 카운터"""
    
    def __init__(self, entrance_area=None, exit_area=None, area_name="Building",
                 min_residence_time: float = 0.5):
        """
        카운터 초기화

        Args:
            entrance_area (list): 입구 영역 좌표 [(x1, y1), (x2, y2), ...]
            exit_area (list): 출구 영역 좌표 [(x1, y1), (x2, y2), ...]
            area_name (str): 영역 이름
            min_residence_time (float): FROM 영역에서 머물러야 하는 최소 시간(초).
                노이즈/순간 잘못된 매칭을 거르기 위한 임계값.
        """
        self.entrance_area = entrance_area or []
        self.exit_area = exit_area or []
        self.area_name = area_name

        # 각 객체의 현재 영역과 진입 시각
        # current_area[obj_id] ∈ {'entrance', 'exit'}
        self.current_area: dict = {}
        self.area_enter_time: dict = {}

        # 카운터 — O(1) 조회를 위해 set 사용
        self.entered_ids: set = set()
        self.exited_ids: set = set()

        # 통계
        self.entry_history = []
        self.exit_history = []

        self.min_residence_time = min_residence_time

        self._entrance_poly = (
            np.array(self.entrance_area, np.int32) if self.entrance_area else None
        )
        self._exit_poly = (
            np.array(self.exit_area, np.int32) if self.exit_area else None
        )

    def _point_in(self, poly, x, y) -> bool:
        if poly is None:
            return False
        return cv2.pointPolygonTest(poly, (x, y), False) >= 0

    def update(self, tracked_objects):
        """
        추적된 객체들을 state machine 방식으로 카운팅.

        규칙:
            entrance → exit 전이 (FROM 영역에서 min_residence_time 충족): 입장(+1)
            exit → entrance 전이 (FROM 영역에서 min_residence_time 충족,
                                  이미 입장한 객체에 한함): 퇴장(+1)
            영역 밖이거나 두 영역에 동시에 걸친 상태는 직전 상태를 유지한다.

        Args:
            tracked_objects (list): [[x1, y1, x2, y2, id], ...]
        """
        current_time = datetime.now()

        for obj in tracked_objects:
            x1, y1, x2, y2, obj_id = obj
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            in_entrance = self._point_in(self._entrance_poly, cx, cy)
            in_exit = self._point_in(self._exit_poly, cx, cy)

            if in_entrance and not in_exit:
                new_area = 'entrance'
            elif in_exit and not in_entrance:
                new_area = 'exit'
            else:
                # 영역 밖이거나 모호한 위치(겹침/없음) — 직전 상태 유지
                continue

            previous_area = self.current_area.get(obj_id)
            if previous_area == new_area:
                # 같은 영역에서 계속 머무는 중 — 상태 변화 없음
                continue

            # 전이 발생
            if previous_area is not None:
                duration = (current_time - self.area_enter_time[obj_id]).total_seconds()
                if duration >= self.min_residence_time:
                    self._record_transition(obj_id, previous_area, new_area, current_time)

            self.current_area[obj_id] = new_area
            self.area_enter_time[obj_id] = current_time

    def _record_transition(self, obj_id, from_area, to_area, current_time):
        """영역 전이를 입장/퇴장으로 기록."""
        if from_area == 'entrance' and to_area == 'exit':
            if obj_id not in self.entered_ids:
                self.entered_ids.add(obj_id)
                self.entry_history.append((obj_id, current_time))
                print(f"[입장] ID {obj_id} - {current_time.strftime('%H:%M:%S')}")
        elif from_area == 'exit' and to_area == 'entrance':
            if obj_id in self.entered_ids and obj_id not in self.exited_ids:
                self.exited_ids.add(obj_id)
                self.exit_history.append((obj_id, current_time))
                print(f"[퇴장] ID {obj_id} - {current_time.strftime('%H:%M:%S')}")
    
    def get_counts(self):
        """현재 카운팅 결과 반환"""
        return {
            'entered': len(self.entered_ids),
            'exited': len(self.exited_ids),
            'current_inside': len(self.entered_ids) - len(self.exited_ids),
            'area_name': self.area_name
        }
    
    def get_hourly_stats(self):
        """시간별 통계 반환"""
        hourly_entries = defaultdict(int)
        hourly_exits = defaultdict(int)
        
        for _, timestamp in self.entry_history:
            hour = timestamp.hour
            hourly_entries[hour] += 1
        
        for _, timestamp in self.exit_history:
            hour = timestamp.hour
            hourly_exits[hour] += 1
        
        return {
            'hourly_entries': dict(hourly_entries),
            'hourly_exits': dict(hourly_exits)
        }
    
    def reset_counts(self):
        """카운트 리셋"""
        self.entered_ids = set()
        self.exited_ids = set()
        self.current_area = {}
        self.area_enter_time = {}
        self.entry_history = []
        self.exit_history = []
    
    def save_history(self, filename):
        """기록을 파일로 저장"""
        history_data = {
            'area_name': self.area_name,
            'entry_history': [
                (obj_id, timestamp.isoformat()) 
                for obj_id, timestamp in self.entry_history
            ],
            'exit_history': [
                (obj_id, timestamp.isoformat()) 
                for obj_id, timestamp in self.exit_history
            ],
            'final_counts': self.get_counts(),
            'hourly_stats': self.get_hourly_stats()
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(history_data, f, indent=2, ensure_ascii=False)
    
    def draw_areas(self, frame, show_counts=True):
        """프레임에 영역과 카운트 정보 그리기"""
        result_frame = frame.copy()
        
        # 입구 영역 그리기 (녹색)
        if self.entrance_area:
            cv2.polylines(
                result_frame,
                [np.array(self.entrance_area, np.int32)],
                True,
                (0, 255, 0),  # 녹색
                2
            )
            cv2.putText(
                result_frame,
                "ENTRANCE",
                tuple(self.entrance_area[0]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
        
        # 출구 영역 그리기 (빨간색)
        if self.exit_area:
            cv2.polylines(
                result_frame,
                [np.array(self.exit_area, np.int32)],
                True,
                (0, 0, 255),  # 빨간색
                2
            )
            cv2.putText(
                result_frame,
                "EXIT",
                tuple(self.exit_area[0]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )
        
        # 카운트 정보 표시
        if show_counts:
            counts = self.get_counts()
            y_offset = 30
            
            # 배경 사각형
            cv2.rectangle(result_frame, (10, 10), (350, 120), (0, 0, 0), -1)
            cv2.rectangle(result_frame, (10, 10), (350, 120), (255, 255, 255), 2)
            
            # 텍스트 정보
            info_texts = [
                f"Area: {counts['area_name']}",
                f"Entered: {counts['entered']}",
                f"Exited: {counts['exited']}",
                f"Current Inside: {counts['current_inside']}"
            ]
            
            for i, text in enumerate(info_texts):
                cv2.putText(
                    result_frame,
                    text,
                    (20, y_offset + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1
                )
        
        return result_frame


class MultiAreaCounter:
    """다중 영역 카운터"""
    
    def __init__(self):
        self.areas = {}  # {area_name: AreaCounter}
    
    def add_area(self, area_name, entrance_area, exit_area):
        """새로운 카운팅 영역 추가"""
        self.areas[area_name] = AreaCounter(entrance_area, exit_area, area_name)
    
    def update_all(self, tracked_objects):
        """모든 영역 업데이트"""
        for area_counter in self.areas.values():
            area_counter.update(tracked_objects)
    
    def get_total_counts(self):
        """전체 영역의 합계 반환"""
        total_entered = 0
        total_exited = 0
        total_inside = 0
        
        for area_counter in self.areas.values():
            counts = area_counter.get_counts()
            total_entered += counts['entered']
            total_exited += counts['exited']
            total_inside += counts['current_inside']
        
        return {
            'total_entered': total_entered,
            'total_exited': total_exited,
            'total_inside': total_inside
        }
    
    def get_area_counts(self, area_name):
        """특정 영역의 카운트 반환"""
        if area_name in self.areas:
            return self.areas[area_name].get_counts()
        return None


class DirectionCounter:
    """방향 기반 카운터 (라인 통과 감지)"""
    
    def __init__(self, counting_line, line_name="Counting Line"):
        """
        방향 카운터 초기화
        
        Args:
            counting_line (tuple): 카운팅 라인 좌표 ((x1, y1), (x2, y2))
            line_name (str): 라인 이름
        """
        self.counting_line = counting_line
        self.line_name = line_name
        
        # 객체 추적
        self.object_positions = {}  # {id: [(x, y, timestamp), ...]}
        
        # 카운터
        self.upward_count = 0    # 위쪽으로 통과
        self.downward_count = 0  # 아래쪽으로 통과
        self.leftward_count = 0  # 왼쪽으로 통과
        self.rightward_count = 0 # 오른쪽으로 통과
        
        # 설정
        self.position_history_length = 5  # 위치 이력 저장 개수
    
    def update(self, tracked_objects):
        """추적된 객체들로 카운팅 업데이트"""
        current_time = datetime.now()
        
        for obj in tracked_objects:
            x1, y1, x2, y2, obj_id = obj
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # 위치 이력 업데이트
            if obj_id not in self.object_positions:
                self.object_positions[obj_id] = []
            
            self.object_positions[obj_id].append((center_x, center_y, current_time))
            
            # 이력 길이 제한
            if len(self.object_positions[obj_id]) > self.position_history_length:
                self.object_positions[obj_id] = self.object_positions[obj_id][-self.position_history_length:]
            
            # 라인 통과 확인
            if len(self.object_positions[obj_id]) >= 2:
                self._check_line_crossing(obj_id)
    
    def _check_line_crossing(self, obj_id):
        """라인 통과 확인"""
        positions = self.object_positions[obj_id]
        if len(positions) < 2:
            return
        
        prev_pos = positions[-2]
        curr_pos = positions[-1]
        
        # 라인 통과 확인
        line_crossed, direction = self._line_intersection(
            prev_pos[:2], curr_pos[:2], self.counting_line
        )
        
        if line_crossed:
            if direction == 'up':
                self.upward_count += 1
            elif direction == 'down':
                self.downward_count += 1
            elif direction == 'left':
                self.leftward_count += 1
            elif direction == 'right':
                self.rightward_count += 1
            
            print(f"[라인 통과] ID {obj_id} - 방향: {direction}")
    
    def _line_intersection(self, p1, p2, line):
        """두 선분의 교차 확인 및 방향 판단"""
        (x1, y1), (x2, y2) = p1, p2
        (lx1, ly1), (lx2, ly2) = line
        
        # 선분 교차 확인
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
        
        if ccw(p1, line[0], line[1]) != ccw(p2, line[0], line[1]) and \
           ccw(p1, p2, line[0]) != ccw(p1, p2, line[1]):
            
            # 방향 판단
            if abs(lx2 - lx1) > abs(ly2 - ly1):  # 수평선에 가까움
                direction = 'up' if y1 > y2 else 'down'
            else:  # 수직선에 가까움
                direction = 'left' if x1 > x2 else 'right'
            
            return True, direction
        
        return False, None
    
    def get_counts(self):
        """카운팅 결과 반환"""
        return {
            'upward': self.upward_count,
            'downward': self.downward_count,
            'leftward': self.leftward_count,
            'rightward': self.rightward_count,
            'total': self.upward_count + self.downward_count + self.leftward_count + self.rightward_count,
            'net_vertical': self.upward_count - self.downward_count,
            'net_horizontal': self.rightward_count - self.leftward_count
        }
    
    def draw_line(self, frame, show_counts=True):
        """프레임에 카운팅 라인과 결과 그리기"""
        result_frame = frame.copy()
        
        # 카운팅 라인 그리기
        cv2.line(result_frame, self.counting_line[0], self.counting_line[1], (255, 0, 255), 3)
        
        # 라인 이름 표시
        mid_x = (self.counting_line[0][0] + self.counting_line[1][0]) // 2
        mid_y = (self.counting_line[0][1] + self.counting_line[1][1]) // 2
        cv2.putText(
            result_frame,
            self.line_name,
            (mid_x - 50, mid_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 255),
            2
        )
        
        # 카운트 정보 표시
        if show_counts:
            counts = self.get_counts()
            info_text = f"Up:{counts['upward']} Down:{counts['downward']} Left:{counts['leftward']} Right:{counts['rightward']}"
            cv2.putText(
                result_frame,
                info_text,
                (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1
            )
        
        return result_frame


if __name__ == "__main__":
    # 테스트 코드
    import random
    
    # 테스트용 영역 설정
    entrance_area = [(100, 300), (200, 300), (200, 400), (100, 400)]
    exit_area = [(300, 300), (400, 300), (400, 400), (300, 400)]
    
    counter = AreaCounter(entrance_area, exit_area, "Test Building")
    
    # 가상의 추적 데이터로 테스트
    print("영역 카운터 테스트 시작...")
    
    for frame_num in range(100):
        # 가상의 추적 객체들 생성
        tracked_objects = []
        for i in range(random.randint(0, 5)):
            x = random.randint(50, 450)
            y = random.randint(250, 450)
            tracked_objects.append([x-25, y-25, x+25, y+25, i])
        
        counter.update(tracked_objects)
        
        if frame_num % 20 == 0:
            counts = counter.get_counts()
            print(f"프레임 {frame_num}: 입장 {counts['entered']}, 퇴장 {counts['exited']}, 현재 {counts['current_inside']}")
    
    # 최종 결과
    final_counts = counter.get_counts()
    print(f"\n최종 결과: {final_counts}")
    
    # 시간별 통계
    hourly_stats = counter.get_hourly_stats()
    print(f"시간별 통계: {hourly_stats}")
    
    # 기록 저장 테스트
    counter.save_history("test_counter_history.json")
    print("기록이 'test_counter_history.json'에 저장되었습니다.")