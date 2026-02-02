"""
역주행 검지 핵심 로직 (ROI/방향/이벤트 판정)
"""
import json
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import numpy as np
import cv2

from .types import WrongWayEvent
from .tracker_impl import TrackerImpl


class WrongWayDetectorCore:
    """Core wrong-way detection logic without any GUI dependencies."""

    def __init__(
        self,
        tracker_impl: TrackerImpl,
        wrongway_threshold_angle: float = 90.0,
        wrongway_start_frames: int = 5,
        wrongway_end_frames: int = 5,
        trajectory_length: int = 10,
        min_movement_distance: float = 10.0,
        snapshot_dir: Optional[Path] = None,
    ) -> None:
        self.tracker = tracker_impl

        self.vehicle_class_ids = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

        self.roi_points: List[Tuple[int, int]] = []
        self.roi_set = False
        self.reference_direction: Optional[np.ndarray] = None
        self.reference_start_point: Optional[Tuple[int, int]] = None
        self.reference_end_point: Optional[Tuple[int, int]] = None

        self.wrongway_threshold_angle = wrongway_threshold_angle
        self.wrongway_start_frames = wrongway_start_frames
        self.wrongway_end_frames = wrongway_end_frames
        self.trajectory_length = trajectory_length
        self.min_movement_distance = min_movement_distance

        self.snapshot_dir = snapshot_dir
        if self.snapshot_dir:
            self.snapshot_dir.mkdir(parents=True, exist_ok=True)

        self.track_states: Dict[int, Dict] = {}

    # ----- geometry helpers ----- #

    @staticmethod
    def normalize_vector(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec
        return vec / norm

    @staticmethod
    def get_absolute_angle(pt1, pt2) -> float:
        x1, y1 = float(pt1[0]), float(pt1[1])
        x2, y2 = float(pt2[0]), float(pt2[1])
        dx, dy = x2 - x1, y2 - y1
        angle = np.degrees(np.arctan2(-dy, dx))
        return (angle + 360.0) % 360.0

    @staticmethod
    def calculate_angle_difference(angle1: float, angle2: float) -> float:
        diff = abs(angle1 - angle2) % 360.0
        return diff if diff <= 180.0 else 360.0 - diff

    @staticmethod
    def bbox_points9(x1: int, y1: int, x2: int, y2: int) -> List[Tuple[int, int]]:
        mid_x = (x1 + x2) // 2
        mid_y = (y1 + y2) // 2
        return [
            (x1, y1),
            (mid_x, y1),
            (x2, y1),
            (x1, mid_y),
            (mid_x, mid_y),
            (x2, mid_y),
            (x1, y2),
            (mid_x, y2),
            (x2, y2),
        ]

    # ----- configuration ----- #

    def load_config(self, config_path: Path) -> None:
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        roi_points = data.get("roi_points")
        if not roi_points or len(roi_points) < 3:
            raise ValueError("ROI 설정이 올바르지 않습니다 (최소 3개 점 필요)")
        self.roi_points = [tuple(map(int, pt)) for pt in roi_points]
        self.roi_set = True

        ref_dir = data.get("reference_direction")
        if ref_dir is None:
            raise ValueError("reference_direction 이(가) 필요합니다")
        vec = np.array(ref_dir, dtype=np.float32)
        if np.linalg.norm(vec) == 0:
            raise ValueError("reference_direction 벡터 길이가 0입니다")
        self.reference_direction = self.normalize_vector(vec)

        self.reference_start_point = None
        self.reference_end_point = None
        start_pt = data.get("reference_start_point")
        end_pt = data.get("reference_end_point")
        if start_pt and end_pt:
            self.reference_start_point = tuple(map(int, start_pt))
            self.reference_end_point = tuple(map(int, end_pt))

        # fallback if start/end missing: use ROI centroid & direction vector length 100px
        if self.reference_start_point is None:
            roi_np = np.array(self.roi_points, dtype=np.float32)
            centroid = roi_np.mean(axis=0)
            self.reference_start_point = (int(centroid[0]), int(centroid[1]))
        if self.reference_end_point is None:
            base = np.array(self.reference_start_point, dtype=np.float32)
            end = base + self.reference_direction * 100.0
            self.reference_end_point = (int(end[0]), int(end[1]))

    # ----- processing ----- #

    def process_frame(self, frame: np.ndarray, frame_index: int, timestamp: float) -> List[WrongWayEvent]:
        if not self.roi_set or self.reference_direction is None:
            return []

        # 트래커를 통해 검출 및 트래킹
        tracks = self.tracker.track(frame)

        roi_polygon = np.array(self.roi_points, dtype=np.int32)
        events: List[WrongWayEvent] = []
        current_time = timestamp

        for x1, y1, x2, y2, conf, cls_id, track_id in tracks:
            if track_id is None:
                continue  # skip untracked

            state = self.track_states.setdefault(
                track_id,
                {
                    "center_history": [],
                    "wrongway_frames": 0,
                    "forward_frames": 0,
                    "event_start_time": None,
                    "event_end_time": None,
                    "last_detected_time": None,
                    "wrongway_active": False,
                },
            )

            state["last_detected_time"] = current_time
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            history = state["center_history"]
            history.append(center)
            if len(history) > self.trajectory_length:
                history[:] = history[-self.trajectory_length :]

            bbox_points = self.bbox_points9(x1, y1, x2, y2)
            inside_count = sum(cv2.pointPolygonTest(roi_polygon, pt, False) >= 0 for pt in bbox_points)
            outside_majority = (len(bbox_points) - inside_count) >= 7
            in_roi = inside_count >= 3

            is_wrongway = False
            if len(history) >= 5:
                start_section = history[: max(1, len(history) // 3)]
                end_section = history[-max(1, len(history) // 3) :]
                start_avg = np.mean(start_section, axis=0)
                end_avg = np.mean(end_section, axis=0)
                movement_distance = np.linalg.norm(end_avg - start_avg)

                if movement_distance >= self.min_movement_distance:
                    veh_angle = self.get_absolute_angle(start_avg, end_avg)
                    ref_angle = self.get_absolute_angle((0, 0), self.reference_direction)
                    angle_diff = self.calculate_angle_difference(ref_angle, veh_angle)
                    is_wrongway = angle_diff >= self.wrongway_threshold_angle

            if is_wrongway:
                state["wrongway_frames"] += 1
                state["forward_frames"] = 0
            else:
                state["forward_frames"] += 1
                state["wrongway_frames"] = 0

            # Start event
            if (
                state["wrongway_frames"] >= self.wrongway_start_frames
                and not state["wrongway_active"]
                and in_roi
            ):
                state["wrongway_active"] = True
                state["event_start_time"] = current_time
                snapshot_path = None
                if self.snapshot_dir is not None:
                    snapshot_path = self._save_snapshot(frame, (x1, y1, x2, y2), track_id, frame_index, timestamp)
                events.append(WrongWayEvent("start", track_id, current_time, frame_index, snapshot_path))

            # End event when majority of bbox outside ROI
            if state["wrongway_active"] and outside_majority:
                state["wrongway_active"] = False
                state["event_end_time"] = current_time
                events.append(WrongWayEvent("end", track_id, current_time, frame_index))

            # Additional end condition: recovered forward direction for prolonged frames
            if state["wrongway_active"] and state["forward_frames"] >= self.wrongway_end_frames:
                state["wrongway_active"] = False
                state["event_end_time"] = current_time
                events.append(WrongWayEvent("end", track_id, current_time, frame_index))

        # Cleanup stale tracks
        obsolete = [tid for tid, st in self.track_states.items() if current_time - st.get("last_detected_time", current_time) > 3.0]
        for tid in obsolete:
            del self.track_states[tid]

        return events

    def _save_snapshot(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        track_id: int,
        frame_index: int,
        timestamp: float,
    ) -> Optional[str]:
        if self.snapshot_dir is None:
            return None

        x1, y1, x2, y2 = bbox
        snapshot_frame = frame.copy()
        cv2.rectangle(snapshot_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
        label = f"ID:{track_id}"
        cv2.putText(snapshot_frame, label, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        ts_ms = int(timestamp * 1000)
        filename = f"wrongway_{track_id}_{frame_index}_{ts_ms}.jpg"
        filepath = self.snapshot_dir / filename
        cv2.imwrite(str(filepath), snapshot_frame)
        return str(filepath)
