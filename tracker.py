"""
tracker.py - 객체 추적 모듈
YOLOv8 탐지 결과를 기반으로 객체의 고유 ID를 추적하는 모듈
"""

import math
import numpy as np


class Tracker:
    """객체 추적을 위한 클래스"""
    
    def __init__(self, distance_threshold=35):
        """
        추적기 초기화
        
        Args:
            distance_threshold (int): 동일 객체로 판단할 거리 임계값 (픽셀)
        """
        self.center_points = {}  # 객체의 중심점 저장 {id: (x, y)}
        self.id_count = 0        # 고유 ID 카운터
        self.distance_threshold = distance_threshold
        self.max_disappeared = 10  # 객체가 사라진 것으로 판단할 프레임 수
        self.disappeared = {}     # 사라진 객체 카운터 {id: frame_count}
    
    def update(self, objects_rect):
        """
        새로운 객체들을 기존 객체와 매칭하여 추적
        
        Args:
            objects_rect (list): 탐지된 객체들의 바운딩 박스 [[x1, y1, x2, y2], ...]
            
        Returns:
            list: 추적된 객체들의 정보 [[x1, y1, x2, y2, id], ...]
        """
        objects_bbs_ids = []
        
        # 입력이 비어있는 경우 처리
        if len(objects_rect) == 0:
            # 모든 기존 객체의 사라진 카운터 증가
            disappeared_ids = []
            for obj_id in self.disappeared.keys():
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    disappeared_ids.append(obj_id)
            
            # 너무 오랫동안 사라진 객체들 제거
            for obj_id in disappeared_ids:
                del self.center_points[obj_id]
                del self.disappeared[obj_id]
            
            return objects_bbs_ids
        
        # 새 객체들의 중심점 계산
        input_centroids = []
        for rect in objects_rect:
            x1, y1, x2, y2 = rect
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            input_centroids.append((cx, cy))
        
        # 기존 추적 중인 객체가 없는 경우
        if len(self.center_points) == 0:
            for i, rect in enumerate(objects_rect):
                x1, y1, x2, y2 = rect
                self.center_points[self.id_count] = input_centroids[i]
                objects_bbs_ids.append([x1, y1, x2, y2, self.id_count])
                self.id_count += 1
        else:
            # 기존 객체와 새 객체 간 거리 행렬 계산
            object_ids = list(self.center_points.keys())
            object_centroids = list(self.center_points.values())
            
            # 헝가리안 알고리즘의 간단한 버전 - 최소 거리 매칭
            used_ids = set()
            used_inputs = set()
            
            # 각 입력 객체에 대해 가장 가까운 기존 객체 찾기
            for input_idx, input_centroid in enumerate(input_centroids):
                min_distance = float('inf')
                min_id = None
                
                for obj_id, obj_centroid in zip(object_ids, object_centroids):
                    if obj_id in used_ids:
                        continue
                    
                    distance = math.hypot(
                        input_centroid[0] - obj_centroid[0],
                        input_centroid[1] - obj_centroid[1]
                    )
                    
                    if distance < min_distance and distance < self.distance_threshold:
                        min_distance = distance
                        min_id = obj_id
                
                # 매칭된 객체가 있는 경우
                if min_id is not None:
                    x1, y1, x2, y2 = objects_rect[input_idx]
                    self.center_points[min_id] = input_centroid
                    objects_bbs_ids.append([x1, y1, x2, y2, min_id])
                    used_ids.add(min_id)
                    used_inputs.add(input_idx)
                    
                    # 사라진 카운터에서 제거
                    if min_id in self.disappeared:
                        del self.disappeared[min_id]
            
            # 매칭되지 않은 입력 객체들은 새로운 객체로 등록
            for input_idx, rect in enumerate(objects_rect):
                if input_idx not in used_inputs:
                    x1, y1, x2, y2 = rect
                    self.center_points[self.id_count] = input_centroids[input_idx]
                    objects_bbs_ids.append([x1, y1, x2, y2, self.id_count])
                    self.id_count += 1
            
            # 매칭되지 않은 기존 객체들은 사라진 것으로 처리
            for obj_id in object_ids:
                if obj_id not in used_ids:
                    if obj_id not in self.disappeared:
                        self.disappeared[obj_id] = 1
                    else:
                        self.disappeared[obj_id] += 1
        
        # 너무 오랫동안 사라진 객체들 정리
        disappeared_ids = []
        for obj_id, count in self.disappeared.items():
            if count > self.max_disappeared:
                disappeared_ids.append(obj_id)
        
        for obj_id in disappeared_ids:
            if obj_id in self.center_points:
                del self.center_points[obj_id]
            del self.disappeared[obj_id]
        
        return objects_bbs_ids
    
    def get_current_objects(self):
        """현재 추적 중인 객체 수 반환"""
        return len(self.center_points)
    
    def reset(self):
        """추적기 상태 초기화"""
        self.center_points = {}
        self.disappeared = {}
        self.id_count = 0
    
    def get_object_trajectory(self, obj_id, max_points=50):
        """
        특정 객체의 이동 경로 반환
        
        Args:
            obj_id (int): 객체 ID
            max_points (int): 저장할 최대 점 수
            
        Returns:
            list: 이동 경로 점들 [(x, y), ...]
        """
        # 실제 구현시에는 각 객체별로 이동 경로를 저장하는 기능 추가 가능
        if not hasattr(self, 'trajectories'):
            self.trajectories = {}
        
        if obj_id in self.center_points:
            if obj_id not in self.trajectories:
                self.trajectories[obj_id] = []
            
            self.trajectories[obj_id].append(self.center_points[obj_id])
            
            # 최대 점 수 제한
            if len(self.trajectories[obj_id]) > max_points:
                self.trajectories[obj_id] = self.trajectories[obj_id][-max_points:]
            
            return self.trajectories[obj_id].copy()
        
        return []


class MultiClassTracker:
    """다중 클래스 객체 추적기"""
    
    def __init__(self, classes=None, distance_threshold=35):
        """
        다중 클래스 추적기 초기화
        
        Args:
            classes (list): 추적할 클래스 이름 목록
            distance_threshold (int): 거리 임계값
        """
        self.classes = classes or ['person']
        self.trackers = {cls: Tracker(distance_threshold) for cls in self.classes}
    
    def update(self, detections):
        """
        클래스별로 객체 추적 업데이트
        
        Args:
            detections (dict): {class_name: [[x1, y1, x2, y2], ...]}
            
        Returns:
            dict: {class_name: [[x1, y1, x2, y2, id], ...]}
        """
        results = {}
        
        for class_name in self.classes:
            if class_name in detections:
                results[class_name] = self.trackers[class_name].update(detections[class_name])
            else:
                results[class_name] = self.trackers[class_name].update([])
        
        return results
    
    def get_total_objects(self):
        """모든 클래스의 총 객체 수 반환"""
        total = 0
        for tracker in self.trackers.values():
            total += tracker.get_current_objects()
        return total
    
    def reset_all(self):
        """모든 추적기 초기화"""
        for tracker in self.trackers.values():
            tracker.reset()


if __name__ == "__main__":
    # 간단한 테스트 코드
    tracker = Tracker()
    
    # 테스트 데이터
    frame1_objects = [[100, 100, 150, 200], [300, 300, 350, 400]]
    frame2_objects = [[105, 105, 155, 205], [295, 295, 345, 395]]
    
    print("프레임 1:")
    result1 = tracker.update(frame1_objects)
    for obj in result1:
        print(f"  객체 ID {obj[4]}: ({obj[0]}, {obj[1]}) - ({obj[2]}, {obj[3]})")
    
    print("\n프레임 2:")
    result2 = tracker.update(frame2_objects)
    for obj in result2:
        print(f"  객체 ID {obj[4]}: ({obj[0]}, {obj[1]}) - ({obj[2]}, {obj[3]})")
    
    print(f"\n현재 추적 중인 객체 수: {tracker.get_current_objects()}")