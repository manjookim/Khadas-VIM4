"""
트래커 구현 (CPU 기반 - VIM4용)

NPU 패키지는 추론만 NPU로 하고, 트래킹은 CPU에서 수행합니다.
간단한 트래커 구현 또는 기존 CPU 트래커 라이브러리 사용 가능.
"""
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import sys, os 

from .inference_backend import InferenceBackend

sys.path.append(os.path.join(os.getcwd(), 'ByteTrack'))
from ByteTrack.byte_tracker import BYTETracker

class TrackerArgs:
    def __init__(self):
        self.track_thresh = 0.25  # 탐지 신뢰도 임계값
        self.track_buffer = 60    # 물체를 놓쳤을 때 유지할 프레임 수
        self.match_thresh = 0.7   # 매칭 임계값
        self.mot20 = False

class TrackerImpl:
    """CPU 기반 트래커 (VIM4용)"""
    
    def __init__(self, inference_backend: InferenceBackend):
        """
        Args:
            inference_backend: 추론 백엔드 (NPU)
        """
        self.backend = inference_backend
        # TODO: 간단한 트래커 구현 또는 라이브러리 초기화
        # 예: Kalman 필터 기반 트래커, IOU 트래커 등
        self.next_track_id = 1
        self.tracks = {}  # {track_id: last_bbox}
        self.max_iou_distance = 0.7

        self.tracker = BYTETracker(TrackerArgs())
    
    def track(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float, int, Optional[int]]]:
        """
        프레임에서 차량 검출 및 트래킹
        
        Args:
            frame: BGR 형식의 numpy 배열 (H, W, 3)
        
        Returns:
            List[(x1, y1, x2, y2, conf, class_id, track_id)]
            track_id는 None일 수 있음 (새로운 객체)
        """
        # NPU 추론으로 검출
        detections = self.backend.detect(frame)
        
        # TODO: 트래킹 로직 구현
        # 1. IOU 기반 매칭 또는 Kalman 필터 기반 매칭
        # 2. 기존 트랙과 매칭
        # 3. 새 트랙 생성
        # 4. 오래된 트랙 제거

        if not detections:
            return []
        
        h,w = frame.shape[:2]

        """ROI 내의 차량만 필터링"""
        
        filtered_inputs = []
        filtered_detections = []
        for det in detections:
            filtered_inputs.append(det["tracker_box"])
            filtered_detections.append(det)
        if not filtered_inputs:
            return []
        tracker_inputs = np.array(filtered_inputs)
        
        
        #tracker_inputs = np.array([r["tracker_box"] for r in detections])
        online_targets = self.tracker.update(tracker_inputs, [640, 640], (640, 640))

        tracks = []
        for t in online_targets:
            tx1, ty1, tx2, ty2 = t.tlbr

            matched_cls = 999  # 기본값
            for det in detections:
                # tracker_box: [x1, y1, x2, y2, score]
                dx1, dy1, dx2, dy2, _ = det["tracker_box"]
                
                if abs(tx1 - dx1) < 2.0 and abs(ty1 - dy1) < 2.0:
                    matched_cls = det["class_id"]
                    break

            # 백엔드에 저장된 ratio와 pad 정보를 꺼내서 계산
            rx1 = int(max(0, min(w, (tx1 - self.backend.pad_left) / self.backend.ratio)))
            ry1 = int(max(0, min(h, (ty1 - self.backend.pad_top) / self.backend.ratio)))
            rx2 = int(max(0, min(w, (tx2 - self.backend.pad_left) / self.backend.ratio)))
            ry2 = int(max(0, min(h, (ty2 - self.backend.pad_top) / self.backend.ratio)))
            
            # 요구된 반환 형식: (x1, y1, x2, y2, conf, class_id, track_id)
            tracks.append((rx1, ry1, rx2, ry2, float(t.score), matched_cls, int(t.track_id)))

        #print(tracks)
        return tracks


    """
        tracks = []
        for x1, y1, x2, y2, conf, class_id in detections:
            # 간단한 구현 예시 (실제로는 더 정교한 트래커 필요)
            track_id = self._assign_track_id((x1, y1, x2, y2))
            tracks.append((x1, y1, x2, y2, conf, class_id, track_id))
        
        return tracks
    """
        
    
    """
    def _assign_track_id(self, bbox: Tuple[int, int, int, int]) -> Optional[int]:
        #IOU 기반 간단한 트랙 ID 할당 (스켈레톤 구현)
        # TODO: 실제 트래킹 알고리즘 구현
        # 예: IOU 기반 매칭, Kalman 필터 등
        track_id = self.next_track_id
        self.next_track_id += 1
        return track_id

    """
