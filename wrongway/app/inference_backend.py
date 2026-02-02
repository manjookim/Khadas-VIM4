"""
추론 백엔드 (VIM4 TIM-VX/Khadas NPU SDK)

CONTRACT: detect(frame_bgr) -> List[Det]
  - frame_bgr: BGR 형식의 numpy 배열 (H, W, 3), dtype=uint8
  - 반환: List[Det] where Det = (x1, y1, x2, y2, conf, class_id)
  - 조건:
    * NMS 완료된 검출 결과만 반환
    * 좌표는 원본 프레임 좌표계 (letterbox 등 전처리 복원 완료)
    * 차량 클래스만 반환 (class_id: 2=car, 3=motorcycle, 5=bus, 7=truck)
    * 최소 박스 크기 필터링 완료 (w>=6, h>=6, area>=100)
    * 프레임 경계 필터링 완료

TODO(VIM4): TIM-VX/Khadas NPU SDK 연동 구현
  1. Khadas VIM4 NPU SDK 설치 및 참조
     - 문서: https://docs.khadas.com/products/sbc/vim4/npu/npu-sdk
  2. 모델 변환 (YOLO -> TIM-VX 형식)
     - 변환 도구 및 절차는 Khadas 문서 참조
  3. 전처리 구현
     - 입력 해상도: YOLO 모델 입력 크기 (예: 640x640)
     - Letterbox 처리 (비율 유지, 패딩 추가)
     - BGR -> RGB 변환 (필요시)
     - 정규화 (필요시)
  4. 추론 실행
     - TIM-VX 엔진으로 NPU 추론
     - 출력 텐서 파싱
  5. 후처리 구현
     - NMS (Non-Maximum Suppression)
     - 좌표 복원 (letterbox 역변환, 원본 프레임 좌표로 변환)
     - 클래스 필터링 (차량만)
     - 크기/경계 필터링
"""
from pathlib import Path
from typing import List, Tuple
import numpy as np
import os
import argparse
import cv2 as cv
from ksnn.api import KSNN
import time
import sys


# Det 타입 정의 (x1, y1, x2, y2, conf, class_id)
Det = Tuple[int, int, int, int, float, int]


class InferenceBackend:
    """VIM4 TIM-VX 기반 추론 백엔드 (스캐폴딩)"""
    
    def __init__(self, model_path: str, library: str, device: str = "npu", conf_threshold: float = 0.45):
        """
        Args:
            model_path: TIM-VX 모델 파일 경로 (변환된 모델)
            device: 디바이스 (일반적으로 "npu")
            conf_threshold: 신뢰도 임계값
        """
        self.model_path = model_path
        self.device = device
        self.conf_threshold = conf_threshold
        self.vehicle_class_ids = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
        
        # TODO(VIM4): TIM-VX 엔진 초기화
        # 예:
        # from khadas_npu import TimVXEngine
        # self.engine = TimVXEngine(model_path)
        # self.input_shape = self.engine.get_input_shape()  # 예: (640, 640, 3)

        self.library_path = library
        self.yolov8 = KSNN('VIM4')
        self.yolov8.nn_init(library=self.library_path, model=self.model_path, level=0)
        
        self.NUM_CLS = 80
        self.NMS_THRESH = 0.45
        self.OBJ_THRESH = 0.003
        self.constant_matrix = np.array([[i for i in range(16)]]).T
        self.mean = [0, 0, 0]
        self.var = [255]

        self.total_det_time = 0
        self.det_count = 0
        
        print(f"[{self.device.upper()}] InferenceBackend initialized with {model_path}")
        #raise NotImplementedError("TODO(VIM4): TIM-VX/Khadas NPU SDK 연동 구현 필요")
    
    def sigmoid(self, x): return 1 / (1 + np.exp(-x))

    def softmax(self, x, axis=0):
        x = np.exp(x)
        return x / x.sum(axis=axis, keepdims=True)

    def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114)):
        #print(img.shape)
        shape = img.shape[:2] # [height, width]
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    
        dw /= 2 # 양쪽 여백 분할
        dh /= 2
    
        if shape[::-1] != new_unpad:
            #img = cv.resize(img, new_unpad, interpolation=cv.INTER_LINEAR)
            img = cv.resize(img, new_unpad, interpolation=cv.INTER_NEAREST)
    
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    
        img = cv.copyMakeBorder(img, top, bottom, left, right, cv.BORDER_CONSTANT, value=color)
        return img, r, (left, top)

    def process(self, input):
        grid_h, grid_w = map(int, input.shape[0:2])
        box_class_probs = self.sigmoid(input[..., :self.NUM_CLS])
        box_0 = self.softmax(input[..., self.NUM_CLS: self.NUM_CLS + 16], -1)
        box_1 = self.softmax(input[..., self.NUM_CLS + 16:self.NUM_CLS + 32], -1)
        box_2 = self.softmax(input[..., self.NUM_CLS + 32:self.NUM_CLS + 48], -1)
        box_3 = self.softmax(input[..., self.NUM_CLS + 48:self.NUM_CLS + 64], -1)
        
        result = np.zeros((grid_h, grid_w, 1, 4))
        result[..., 0] = np.dot(box_0, self.constant_matrix)[..., 0]
        result[..., 1] = np.dot(box_1, self.constant_matrix)[..., 0]
        result[..., 2] = np.dot(box_2, self.constant_matrix)[..., 0]
        result[..., 3] = np.dot(box_3, self.constant_matrix)[..., 0]

        col = np.tile(np.arange(0, grid_w), grid_w).reshape(-1, grid_w).reshape(grid_h, grid_w, 1, 1)
        row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_h).reshape(grid_h, grid_w, 1, 1)
        grid = np.concatenate((col, row), axis=-1)

        # 표준 YOLOv8 수식 적용
        result[..., 0:2] = (grid + 0.5 - result[..., 0:2]) / (grid_w, grid_h)
        result[..., 2:4] = (grid + 0.5 + result[..., 2:4]) / (grid_w, grid_h)
        return result, box_class_probs
    
    def filter_boxes(self, boxes, box_class_probs):
        box_classes = np.argmax(box_class_probs, axis=-1)
        box_class_scores = np.max(box_class_probs, axis=-1)
        pos = np.where(box_class_scores >= self.OBJ_THRESH)
        return boxes[pos], box_classes[pos], box_class_scores[pos]
    
    def nms_boxes(self, boxes, scores):
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1, yy1 = np.maximum(x1[i], x1[order[1:]]), np.maximum(y1[i], y1[order[1:]])
            xx2, yy2 = np.minimum(x2[i], x2[order[1:]]), np.minimum(y2[i], y2[order[1:]])
            w, h = np.maximum(0.0, xx2 - xx1 + 1e-5), np.maximum(0.0, yy2 - yy1 + 1e-5)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= self.NMS_THRESH)[0]
            order = order[inds + 1]
        return np.array(keep)
    
    def yolov8_post_process(self, input_data):
        boxes, classes, scores = [], [], []
        for i in range(3):
            res, conf = self.process(input_data[i])
            b, c, s = self.filter_boxes(res, conf)
            if len(b) > 0:
                boxes.append(b); classes.append(c); scores.append(s)
        if not boxes: return None, None, None
        boxes = np.concatenate(boxes); classes = np.concatenate(classes); scores = np.concatenate(scores)

        nboxes, nclasses, nscores = [], [], []
        for c in set(classes):
            if int(c) not in self.vehicle_class_ids: continue
            
            inds = np.where(classes == c)
            b, s = boxes[inds], scores[inds]
            keep = self.nms_boxes(b, s)
            nboxes.append(b[keep]); nclasses.append(classes[inds][keep]); nscores.append(s[keep])
        
        if not nclasses: return None, None, None
        return np.concatenate(nboxes), np.concatenate(nscores), np.concatenate(nclasses)



    def detect(self, frame: np.ndarray) -> List[Det]:
        """
        프레임에서 차량 검출 (NMS 완료, 원본 좌표)
        
        CONTRACT:
          - 입력: frame_bgr (H, W, 3), dtype=uint8, BGR 형식
          - 출력: List[(x1, y1, x2, y2, conf, class_id)]
          - 조건: NMS 완료, 원본 좌표, 차량만, 필터링 완료
        
        Args:
            frame: BGR 형식의 numpy 배열 (H, W, 3), dtype=uint8
        
        Returns:
            List[(x1, y1, x2, y2, conf, class_id)]
        """
        # TODO(VIM4): 전처리
        # 1. Letterbox 처리 (비율 유지, 640x640로 리사이즈)
        # 2. BGR -> RGB 변환 (필요시)
        # 3. 정규화 (필요시)
        # 예:
        # processed_frame, scale, pad = self._letterbox(frame, target_size=(640, 640))
        # rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        # normalized = rgb_frame.astype(np.float32) / 255.0
        
        # TODO(VIM4): NPU 추론
        # 예:
        # output_tensors = self.engine.inference(normalized)
        # boxes, scores, class_ids = self._parse_output(output_tensors)
        
        # TODO(VIM4): 후처리
        # 1. NMS 수행
        # 2. 좌표 복원 (letterbox 역변환: scale, pad 적용)
        # 3. 클래스 필터링 (차량만)
        # 4. 크기/경계 필터링
        # 예:
        # nms_indices = self._nms(boxes, scores, iou_threshold=0.7)
        # detections = []
        # for idx in nms_indices:
        #     x1, y1, x2, y2 = self._restore_coords(boxes[idx], scale, pad, frame.shape)
        #     class_id = int(class_ids[idx])
        #     if class_id in self.vehicle_class_ids:
        #         if self._is_valid_box(x1, y1, x2, y2, frame.shape):
        #             detections.append((x1, y1, x2, y2, float(scores[idx]), class_id))
        # return detections

        #print("Inference Process Start!")

        h, w = frame.shape[:2]

        # 1. 전처리 (Letterbox)
        self.img_pad, self.ratio, (self.pad_left, self.pad_top) = self.letterbox(frame, (640, 640))
        self.img_rgb = cv.cvtColor(self.img_pad, cv.COLOR_BGR2RGB)
        img = self.img_rgb
        #img = self.img_rgb.astype(np.float32)
        #img[:, :, 0] -= self.mean[0]; img[:, :, 1] -= self.mean[1]; img[:, :, 2] -= self.mean[2]
        #img /= self.var[0]

        # 2. NPU 추론
        start_time = time.time()

        data = self.yolov8.nn_inference(img, input_shape=(640, 640, 3), input_type="RGB", 
                                        output_shape=[(40, 40, 144), (80, 80, 144), (20, 20, 144)], 
                                        output_type="FLOAT")
        
        end_time = time.time()

        self.total_det_time += (end_time-start_time)
        self.det_count +=1

        # 3. 텐서 후처리 로직 (yolov8_post_process 내용)
        input_data = [np.expand_dims(data[2], 2), np.expand_dims(data[0], 2), np.expand_dims(data[1], 2)]
        boxes, scores, classes = self.yolov8_post_process(input_data)
        
        detections = []
        if boxes is not None:
            for box, score, cl in zip(boxes, scores, classes):
            # 640x640 공간 좌표 -> 원본 좌표 복원 (사용자님 코드 로직)
                x1, y1 = box[0] * 640, box[1] * 640
                x2, y2 = box[2] * 640, box[3] * 640

                        # 경계 처리 및 정수 변환
                left   = int(max(0, min(w, (x1 - self.pad_left) / self.ratio)))
                top    = int(max(0, min(h, (y1 - self.pad_top) / self.ratio)))
                right  = int(max(0, min(w, (x2 - self.pad_left) / self.ratio)))
                bottom = int(max(0, min(h, (y2 - self.pad_top) / self.ratio)))

                width_box  = right - left
                height_box = bottom - top

                if width_box <=0 or height_box <=0:
                    continue
                
                bw, bh = x2 - x1, y2 - y1
                if bw>=6 and bh>=6 and (bw * bh)>=100:
                    detections.append({
                        "tracker_box": [x1,y1,x2,y2,float(score)],
                        "display_box": [left,top,right,bottom],
                        "score" : float(score),
                        "class_id" : int(cl)
                        })


        #print("Inference Process Finish!")
        #print(detections)
        return detections

        
        #raise NotImplementedError("TODO(VIM4): detect() 메서드 구현 필요")
