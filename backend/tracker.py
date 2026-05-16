"""
tracker.py - 객체 추적 모듈
YOLOv8 탐지 결과를 기반으로 객체의 고유 ID를 추적하는 모듈
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


class Tracker:
    """객체 추적을 위한 클래스 (Hungarian 매칭 기반)"""

    def __init__(self, distance_threshold=35, max_disappeared: int = 10):
        """
        추적기 초기화

        Args:
            distance_threshold (int): 동일 객체로 판단할 거리 임계값 (픽셀)
            max_disappeared (int): 이 프레임 수 이상 탐지 안 되면 ID 폐기
        """
        self.center_points = {}            # {id: (x, y)}
        self.id_count = 0
        self.distance_threshold = distance_threshold
        self.max_disappeared = max_disappeared
        self.disappeared = {}              # {id: frame_count}

    def _prune_disappeared(self):
        """max_disappeared를 초과한 객체 제거."""
        stale = [oid for oid, c in self.disappeared.items() if c > self.max_disappeared]
        for oid in stale:
            self.center_points.pop(oid, None)
            self.disappeared.pop(oid, None)

    def update(self, objects_rect):
        """
        새로운 검출들을 기존 트랙과 Hungarian으로 매칭한다.

        Args:
            objects_rect (list): [[x1, y1, x2, y2], ...]

        Returns:
            list: [[x1, y1, x2, y2, id], ...]
        """
        # 검출이 없으면 모든 트랙의 disappeared 카운터를 증가
        if len(objects_rect) == 0:
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
            for oid in list(self.center_points.keys()):
                if oid not in self.disappeared:
                    self.disappeared[oid] = 1
            self._prune_disappeared()
            return []

        # 새 검출들의 중심점
        input_centroids = np.array(
            [((r[0] + r[2]) // 2, (r[1] + r[3]) // 2) for r in objects_rect],
            dtype=np.int32,
        )

        # 기존 트랙이 없으면 모두 신규 등록
        if len(self.center_points) == 0:
            results = []
            for i, rect in enumerate(objects_rect):
                self.center_points[self.id_count] = tuple(input_centroids[i].tolist())
                self.disappeared[self.id_count] = 0
                results.append([rect[0], rect[1], rect[2], rect[3], self.id_count])
                self.id_count += 1
            return results

        # 거리 행렬 (M tracks x N detections)
        track_ids = list(self.center_points.keys())
        track_centroids = np.array(
            [self.center_points[oid] for oid in track_ids], dtype=np.float32
        )
        det_centroids = input_centroids.astype(np.float32)

        diff = track_centroids[:, None, :] - det_centroids[None, :, :]
        dist = np.linalg.norm(diff, axis=2)

        # Hungarian: 임계값 초과 매칭을 막기 위해 큰 값으로 패딩
        BIG = 1e6
        cost = np.where(dist <= self.distance_threshold, dist, BIG)

        row_ind, col_ind = linear_sum_assignment(cost)

        matched_tracks = set()
        matched_detections = set()
        results = []

        for r, c in zip(row_ind, col_ind):
            if cost[r, c] >= BIG:
                continue  # 임계값 초과 매칭은 버림
            oid = track_ids[r]
            self.center_points[oid] = tuple(input_centroids[c].tolist())
            self.disappeared[oid] = 0
            x1, y1, x2, y2 = objects_rect[c]
            results.append([x1, y1, x2, y2, oid])
            matched_tracks.add(oid)
            matched_detections.add(c)

        # 매칭 안 된 검출 → 신규 ID
        for c in range(len(objects_rect)):
            if c in matched_detections:
                continue
            x1, y1, x2, y2 = objects_rect[c]
            self.center_points[self.id_count] = tuple(input_centroids[c].tolist())
            self.disappeared[self.id_count] = 0
            results.append([x1, y1, x2, y2, self.id_count])
            self.id_count += 1

        # 매칭 안 된 트랙 → disappeared 카운트 증가
        for oid in track_ids:
            if oid not in matched_tracks:
                self.disappeared[oid] = self.disappeared.get(oid, 0) + 1

        self._prune_disappeared()
        return results
    
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