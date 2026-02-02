"""
CLI-oriented wrong-way detection pipeline for VIM4 (TIM-VX).
"""
from __future__ import annotations

import argparse
import json
import queue
import threading
from pathlib import Path
from typing import Optional
import time
import cv2

from .wrongway_core import WrongWayDetectorCore
from .tracker_impl import TrackerImpl
from .inference_backend import InferenceBackend
from .types import WrongWayEvent


class FrameGrabber(threading.Thread):
    def __init__(self, cap: cv2.VideoCapture, frame_queue: queue.Queue, stop_event: threading.Event):
        super().__init__(daemon=True)
        self.cap = cap
        self.frame_queue = frame_queue
        self.stop_event = stop_event
        self.frame_index = 0
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        if self.fps <= 0:
            self.fps = 30.0

    def run(self):
        while not self.stop_event.is_set():
            frame_input_start = time.time()
            ret, frame = self.cap.read()
            framgrabber_time = (time.time() - frame_input_start)*1000 
            if not ret:
                self.frame_queue.put(None)
                break
            timestamp = self.frame_index / self.fps
            self.frame_queue.put((self.frame_index, timestamp, frame, framgrabber_time))
            self.frame_index += 1
        self.stop_event.set()


class WrongWayRunner:
    def __init__(
        self,
        source: str,
        config_path: Path,
        model_path: str,
        library: str,
        device: str,
        conf: float,
        output_path: Optional[Path],
        latency_path: Optional[Path],
        queue_size: int = 4,
        snapshot_dir: Optional[Path] = None,
        mode : str = "full",
    ) -> None:
        # 카메라 인덱스는 정수로 처리 (OpenCV 안정성)
        if source.isdigit():
            source_int = int(source)
            cap = cv2.VideoCapture(source_int)
        else:
            cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            raise ValueError(f"비디오 소스를 열 수 없습니다: {source}")

        self.cap = cap
        
        # 추론 백엔드 초기화 (NPU)
        inference_backend = InferenceBackend(
            model_path=model_path,
            library=library,
            device=device,
            conf_threshold=conf,
        )
        
        # 트래커 구현 초기화
        tracker_impl = TrackerImpl(inference_backend)
        
        # 핵심 검지기 초기화
        self.detector = WrongWayDetectorCore(
            tracker_impl=tracker_impl,
            snapshot_dir=snapshot_dir,
        )
        self.detector.load_config(config_path)

        self.frame_queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self.stop_event = threading.Event()
        self.grabber = FrameGrabber(cap, self.frame_queue, self.stop_event)

        self.output_path = output_path
        self.output_file = open(output_path, "w", encoding="utf-8") if output_path else None

        self.latency_path = latency_path
        self.latency_file = open(latency_path, "w", encoding="utf-8") if latency_path else None

        self._stopped = False

        self.total_frame_ms = 0
        self.total_proc_ms = 0
        self.total_count = 0

        self.mode = mode

    def _emit(self, event: WrongWayEvent) -> None:
        '''
        if not self.output_file:
            return

        # 경로가 있을 때만 이 무거운 작업들을 수행함
        payload = event.to_dict()
        line = json.dumps(payload, ensure_ascii=False)
        
        self.output_file.write(line + "\n")
        self.output_file.flush()
        '''
        
        payload = event.to_dict()
        line = json.dumps(payload, ensure_ascii=False)
        if self.output_file:
            self.output_file.write(line + "\n")
            self.output_file.flush()
        else:
            print(line, flush=True)
        

    def _record_latency(self, frame_idx : int, latency_ms: float, frame_ms: float) -> None:
        if self.latency_file:
            line = f'{{"frame_index" : {frame_idx}, "frame_latency" : {frame_ms:.2f}ms , "latency" : {latency_ms:.2f}ms}} \n'
            self.latency_file.write(line)


    def run(self) -> None:
        self.grabber.start()
        mode = self.mode

        try:
            while not self.stop_event.is_set():
                item = None
                try:
                    item = self.frame_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                if item is None:
                    break

                frame_idx, timestamp, frame , frame_ms = item

                if mode == "io":
                    time.sleep(0.065)
                    self.total_frame_ms += frame_ms
                    self.total_count += 1
                    continue

                start = time.time()
                events = []
                
                if mode == "detect":
                    _ = self.detector.tracker.backend.detect(frame)

                else:
                    events = self.detector.process_frame(frame, frame_idx, timestamp)

                proc_ms = (time.time() - start ) *1000
                self._record_latency(frame_idx, proc_ms, frame_ms)

                self.total_frame_ms += frame_ms
                self.total_proc_ms += proc_ms
                self.total_count += 1 

                for evt in events:
                    self._emit(evt)

        finally:
            self.stop()

    def stop(self) -> None:
        if self._stopped:
            return 
        self._stopped = True

        self.stop_event.set()

        backend = self.detector.tracker.backend
        if backend.det_count > 0:
            avg_det_time = backend.total_det_time / backend.det_count
            print(f"Detection FPS : {1/avg_det_time:.2f}")
            print(f"Detection Latency : {avg_det_time*1000:.2f} ms")

        if self.total_count > 0:
            avg_frame_ms = self.total_frame_ms / self.total_count
            avg_proc_ms = self.total_proc_ms / self.total_count 
            print(f"Avg frame_ms : {avg_frame_ms:.2f} ms")
            print(f"Avg proc ms : {avg_proc_ms:.2f} ms")

        if self.grabber.is_alive():
            self.grabber.join(timeout=1.0)
        if self.cap.isOpened():
            self.cap.release()
        if self.output_file:
            self.output_file.close()
        if self.latency_file:
            self.latency_file.close()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ultra-light wrong-way detection CLI (VIM4 TIM-VX)")
    parser.add_argument("--source", required=True, help="비디오 파일 경로 또는 카메라 인덱스 (예: 0)")
    parser.add_argument("--config", required=True, help="ROI/기준방향 설정 파일 (JSON)")
    parser.add_argument("--model", default="models/model.timvx", help="TIM-VX 모델 파일 경로")
    parser.add_argument("--device", default="npu", help="디바이스 (일반적으로 npu)")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--output", help="이벤트 로그 파일 (미입력시 stdout)")
    parser.add_argument("--latency", help="latency 로그 파일 (미입력시 stdout)")
    parser.add_argument("--queue", type=int, default=4, help="프레임 큐 크기")
    parser.add_argument("--snapshot-dir", help="역주행 이벤트 스틸컷 저장 디렉터리")
    parser.add_argument("--library", required=True)
    parser.add_argument("--mode", default="full", choices=["io", "detect", "full"], help="io: 프레임 받아오기, detect: 추론까지만, full: 트래킹 포함 전체")
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    source = args.source
    config_path = Path(args.config)
    if not config_path.exists():
        parser.error(f"설정 파일을 찾을 수 없습니다: {config_path}")

    output_path = Path(args.output) if args.output else None
    snapshot_dir = Path(args.snapshot_dir) if args.snapshot_dir else None
    latency_path = Path(args.latency) if args.latency else None

    start = time.time()
    
    runner = WrongWayRunner(
        source=source,
        config_path=config_path,
        model_path=args.model,
        device=args.device,
        conf=args.conf,
        output_path=output_path,
        latency_path=latency_path,
        queue_size=args.queue,
        snapshot_dir=snapshot_dir,
        library=args.library,
        mode=args.mode,
    )

    try:
        runner.run()
    except KeyboardInterrupt:
        pass
    finally:
        runner.stop()

    end = time.time()
    backend = runner.detector.tracker.backend
    print(f"Total Inference Time : {(end-start):.2f}s")
    print(f"Total FPS : {backend.det_count/(end-start):.2f}")
    #print(f"Total Latency : {((end-start)/backend.det_count)*1000:.2f} ms")

if __name__ == "__main__":
    main()
