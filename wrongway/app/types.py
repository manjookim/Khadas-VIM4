"""
타입 정의
"""
from dataclasses import dataclass
from typing import Optional, Dict


@dataclass
class WrongWayEvent:
    """역주행 이벤트"""
    event: str  # "start" or "end"
    track_id: int
    timestamp: float
    frame_index: int
    snapshot_path: Optional[str] = None

    def to_dict(self) -> Dict:
        payload = {
            "event": self.event,
            "track_id": self.track_id,
            "timestamp": self.timestamp,
            "frame_index": self.frame_index,
        }
        if self.snapshot_path:
            payload["snapshot_path"] = self.snapshot_path
        return payload
