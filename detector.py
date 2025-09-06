"""
detector.py - YOLOv8 객체 탐지 모듈
YOLOv8 모델을 사용한 객체 탐지 기능을 제공
"""

import cv2
import numpy as np
from ultralytics import YOLO
import torch
from pathlib import Path


class YOLODetector:
    """YOLOv8 기반 객체 탐지기"""
    
    def __init__(self, model_path="yolov8s.pt", confidence_threshold=0.5, device="auto"):
        """
        YOLOv8 탐지기 초기화
        
        Args:
            model_path (str): YOLOv8 모델 파일 경로
            confidence_threshold (float): 신뢰도 임계값
            device (str): 사용할 디바이스 ("auto", "cpu", "cuda", "mps")
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.device = self._setup_device(device)
        
        # YOLOv8 모델 로드
        self.model = self._load_model()
        
        # COCO 클래스 이름
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
    
    def _setup_device(self, device):
        """디바이스 설정"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    def _load_model(self):
        """YOLOv8 모델 로드"""
        try:
            model = YOLO(self.model_path)
            print(f"YOLOv8 모델 로드 성공: {self.model_path}")
            print(f"사용 디바이스: {self.device}")
            return model
        except Exception as e:
            print(f"모델 로딩 실패: {e}")
            print("기본 YOLOv8s 모델을 다운로드합니다...")
            model = YOLO('yolov8s.pt')
            return model
    
    def detect(self, frame, target_classes=None, return_confidence=False):
        """
        프레임에서 객체 탐지 수행
        
        Args:
            frame (np.ndarray): 입력 이미지/프레임
            target_classes (list): 탐지할 클래스 이름 목록 (None이면 모든 클래스)
            return_confidence (bool): 신뢰도 점수 포함 여부
            
        Returns:
            list: 탐지 결과 [[x1, y1, x2, y2, class_name, confidence], ...] 또는
                  [[x1, y1, x2, y2, class_name], ...]
        """
        if target_classes is None:
            target_classes = self.class_names
        
        # YOLOv8으로 탐지 수행
        results = self.model.predict(frame, verbose=False, conf=self.confidence_threshold)
        
        detections = []
        
        if len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()  # 바운딩 박스 좌표
            scores = results[0].boxes.conf.cpu().numpy()  # 신뢰도 점수
            classes = results[0].boxes.cls.cpu().numpy()  # 클래스 인덱스
            
            for box, score, cls_idx in zip(boxes, scores, classes):
                cls_name = self.class_names[int(cls_idx)]
                
                # 지정된 클래스만 필터링
                if cls_name in target_classes:
                    x1, y1, x2, y2 = map(int, box)
                    
                    if return_confidence:
                        detections.append([x1, y1, x2, y2, cls_name, float(score)])
                    else:
                        detections.append([x1, y1, x2, y2, cls_name])
        
        return detections
    
    def detect_persons_only(self, frame):
        """
        사람만 탐지하는 특화 함수
        
        Args:
            frame (np.ndarray): 입력 프레임
            
        Returns:
            list: 사람 바운딩 박스 [[x1, y1, x2, y2], ...]
        """
        detections = self.detect(frame, target_classes=['person'])
        person_boxes = []
        
        for detection in detections:
            x1, y1, x2, y2, class_name = detection
            if class_name == 'person':
                person_boxes.append([x1, y1, x2, y2])
        
        return person_boxes
    
    def detect_with_nms(self, frame, iou_threshold=0.4):
        """
        Non-Maximum Suppression을 적용한 탐지
        
        Args:
            frame (np.ndarray): 입력 프레임
            iou_threshold (float): IoU 임계값
            
        Returns:
            list: NMS가 적용된 탐지 결과
        """
        # YOLOv8는 기본적으로 NMS가 적용되어 있음
        # 추가적인 후처리가 필요한 경우 사용
        results = self.model.predict(frame, verbose=False, conf=self.confidence_threshold)
        
        if len(results[0].boxes) == 0:
            return []
        
        boxes = results[0].boxes.xyxy.cpu().numpy()
        scores = results[0].boxes.conf.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()
        
        # OpenCV NMS 적용
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(), 
            scores.tolist(), 
            self.confidence_threshold, 
            iou_threshold
        )
        
        detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                x1, y1, x2, y2 = map(int, boxes[i])
                class_name = self.class_names[int(classes[i])]
                confidence = float(scores[i])
                detections.append([x1, y1, x2, y2, class_name, confidence])
        
        return detections
    
    def benchmark(self, frame, num_runs=10):
        """
        탐지 성능 벤치마크
        
        Args:
            frame (np.ndarray): 테스트 프레임
            num_runs (int): 실행 횟수
            
        Returns:
            dict: 성능 통계
        """
        import time
        
        # 워밍업
        for _ in range(3):
            self.model.predict(frame, verbose=False)
        
        # 벤치마크 실행
        times = []
        for _ in range(num_runs):
            start_time = time.time()
            results = self.model.predict(frame, verbose=False)
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        fps = 1.0 / avg_time
        
        return {
            'avg_time': avg_time,
            'std_time': std_time,
            'min_time': min_time,
            'max_time': max_time,
            'fps': fps,
            'device': self.device
        }


class BatchDetector:
    """배치 처리용 탐지기"""
    
    def __init__(self, model_path="yolov8s.pt", batch_size=4):
        """
        배치 탐지기 초기화
        
        Args:
            model_path (str): 모델 경로
            batch_size (int): 배치 크기
        """
        self.detector = YOLODetector(model_path)
        self.batch_size = batch_size
    
    def detect_batch(self, frames, target_classes=['person']):
        """
        여러 프레임을 배치로 처리
        
        Args:
            frames (list): 프레임 리스트
            target_classes (list): 탐지할 클래스
            
        Returns:
            list: 각 프레임별 탐지 결과
        """
        all_results = []
        
        for i in range(0, len(frames), self.batch_size):
            batch_frames = frames[i:i+self.batch_size]
            batch_results = []
            
            for frame in batch_frames:
                detections = self.detector.detect(frame, target_classes)
                batch_results.append(detections)
            
            all_results.extend(batch_results)
        
        return all_results


def draw_detections(frame, detections, colors=None, show_confidence=True):
    """
    탐지 결과를 프레임에 그리기
    
    Args:
        frame (np.ndarray): 입력 프레임
        detections (list): 탐지 결과
        colors (dict): 클래스별 색상 {class_name: (b, g, r)}
        show_confidence (bool): 신뢰도 표시 여부
        
    Returns:
        np.ndarray: 결과가 그려진 프레임
    """
    if colors is None:
        colors = {
            'person': (0, 255, 0),      # 녹색
            'car': (255, 0, 0),         # 파란색
            'bicycle': (0, 0, 255),     # 빨간색
            'motorcycle': (255, 255, 0), # 청록색
        }
    
    result_frame = frame.copy()
    
    for detection in detections:
        if len(detection) >= 5:
            x1, y1, x2, y2, class_name = detection[:5]
            confidence = detection[5] if len(detection) > 5 else None
            
            # 색상 선택
            color = colors.get(class_name, (128, 128, 128))  # 기본 회색
            
            # 바운딩 박스 그리기
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)
            
            # 라벨 텍스트 준비
            label = class_name
            if confidence is not None and show_confidence:
                label += f" {confidence:.2f}"
            
            # 라벨 배경
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            cv2.rectangle(
                result_frame,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width, y1),
                color,
                -1
            )
            
            # 라벨 텍스트
            cv2.putText(
                result_frame,
                label,
                (x1, y1 - baseline - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )
    
    return result_frame


if __name__ == "__main__":
    # 테스트 코드
    detector = YOLODetector()
    
    # 웹캠 테스트
    cap = cv2.VideoCapture(0)
    
    print("YOLOv8 탐지 테스트 시작... 'q'를 눌러 종료")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 사람만 탐지
        detections = detector.detect(frame, target_classes=['person'], return_confidence=True)
        
        # 결과 그리기
        result_frame = draw_detections(frame, detections)
        
        # 탐지 개수 표시
        cv2.putText(
            result_frame,
            f"Detected: {len(detections)} persons",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        
        cv2.imshow("YOLOv8 Detection Test", result_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # 성능 벤치마크
    if cap.isOpened():
        ret, test_frame = cap.read()
        if ret:
            print("\n성능 벤치마크 실행 중...")
            benchmark_results = detector.benchmark(test_frame)
            print(f"평균 처리 시간: {benchmark_results['avg_time']:.4f}초")
            print(f"FPS: {benchmark_results['fps']:.1f}")
            print(f"디바이스: {benchmark_results['device']}")
