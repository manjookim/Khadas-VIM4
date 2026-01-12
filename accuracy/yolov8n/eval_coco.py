import numpy as np
import os
import argparse
import json
import cv2 as cv
from ksnn.api import KSNN
import time

OBJ_THRESH = 0.001 
NMS_THRESH = 0.45
mean = [0, 0, 0]
var = [255]
NUM_CLS = 80

constant_martix = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]]).T

COCO_ID_MAP = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
    35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
    64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90
]

def sigmoid(x): return 1 / (1 + np.exp(-x))

def softmax(x, axis=0):
    x = np.exp(x)
    return x / x.sum(axis=axis, keepdims=True)

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    # 원본 비율을 유지하며 리사이즈하고 남는 부분을 색상으로 채움
    shape = img.shape[:2] # [height, width]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    
    dw /= 2 # 양쪽 여백 분할
    dh /= 2
    
    if shape[::-1] != new_unpad:
        img = cv.resize(img, new_unpad, interpolation=cv.INTER_LINEAR)
    
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    
    img = cv.copyMakeBorder(img, top, bottom, left, right, cv.BORDER_CONSTANT, value=color)
    return img, r, (left, top)

def process(input):
    grid_h, grid_w = map(int, input.shape[0:2])
    box_class_probs = sigmoid(input[..., :NUM_CLS])
    box_0 = softmax(input[..., NUM_CLS: NUM_CLS + 16], -1)
    box_1 = softmax(input[..., NUM_CLS + 16:NUM_CLS + 32], -1)
    box_2 = softmax(input[..., NUM_CLS + 32:NUM_CLS + 48], -1)
    box_3 = softmax(input[..., NUM_CLS + 48:NUM_CLS + 64], -1)
    
    result = np.zeros((grid_h, grid_w, 1, 4))
    result[..., 0] = np.dot(box_0, constant_martix)[..., 0]
    result[..., 1] = np.dot(box_1, constant_martix)[..., 0]
    result[..., 2] = np.dot(box_2, constant_martix)[..., 0]
    result[..., 3] = np.dot(box_3, constant_martix)[..., 0]

    col = np.tile(np.arange(0, grid_w), grid_w).reshape(-1, grid_w).reshape(grid_h, grid_w, 1, 1)
    row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_h).reshape(grid_h, grid_w, 1, 1)
    grid = np.concatenate((col, row), axis=-1)

    # 표준 YOLOv8 수식 적용
    result[..., 0:2] = (grid + 0.5 - result[..., 0:2]) / (grid_w, grid_h)
    result[..., 2:4] = (grid + 0.5 + result[..., 2:4]) / (grid_w, grid_h)
    return result, box_class_probs

# [filter_boxes, nms_boxes, yolov8_post_process 함수는 기존과 동일]
def filter_boxes(boxes, box_class_probs):
    box_classes = np.argmax(box_class_probs, axis=-1)
    box_class_scores = np.max(box_class_probs, axis=-1)
    pos = np.where(box_class_scores >= OBJ_THRESH)
    return boxes[pos], box_classes[pos], box_class_scores[pos]

def nms_boxes(boxes, scores):
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
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    return np.array(keep)

def yolov8_post_process(input_data):
    boxes, classes, scores = [], [], []
    for i in range(3):
        res, conf = process(input_data[i])
        b, c, s = filter_boxes(res, conf)
        boxes.append(b); classes.append(c); scores.append(s)

    boxes = np.concatenate(boxes); classes = np.concatenate(classes); scores = np.concatenate(scores)
    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b, s = boxes[inds], scores[inds]
        keep = nms_boxes(b, s)
        nboxes.append(b[keep]); nclasses.append(classes[inds][keep]); nscores.append(s[keep])
    
    if not nclasses: return None, None, None
    return np.concatenate(nboxes), np.concatenate(nscores), np.concatenate(nclasses)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()

    yolov8 = KSNN('VIM4')
    yolov8.nn_init(library=args.library, model=args.model, level=0)

    results_json = []
    img_list = [f for f in os.listdir(args.dataset) if f.endswith('.jpg')]

    total = 0

    for idx, img_name in enumerate(img_list):
        picture = os.path.join(args.dataset, img_name)
        image_id = int(img_name.split('.')[0])

        orig_img = cv.imread(picture, cv.IMREAD_COLOR)
        h, w = orig_img.shape[:2]
        # --- [Letterbox 적용] ---
        img_pad, ratio, (pad_left, pad_top) = letterbox(orig_img, (640, 640))
        img = img_pad.astype(np.float32)
        img[:, :, 0] -= mean[0]; img[:, :, 1] -= mean[1]; img[:, :, 2] -= mean[2]
        img /= var[0]

        start = time.time()
        
        data = yolov8.nn_inference(img, input_shape=(640, 640, 3), input_type="RAW", 
                                   output_shape=[(40, 40, 144), (80, 80, 144), (20, 20, 144)], 
                                   output_type="FLOAT")

        end =  time.time()

        total += (end - start)

        input_data = [np.expand_dims(data[2], 2), np.expand_dims(data[0], 2), np.expand_dims(data[1], 2)]
        boxes, scores, classes = yolov8_post_process(input_data)

        if boxes is not None:
            for box, score, cl in zip(boxes, scores, classes):
                # 640x640 공간에서의 픽셀 좌표로 먼저 변환
                x1 = box[0] * 640; y1 = box[1] * 640
                x2 = box[2] * 640; y2 = box[3] * 640

                # --- [여백 제거 및 원본 크기로 복원] ---
                left = (x1 - pad_left) / ratio
                top = (y1 - pad_top) / ratio
                right = (x2 - pad_left) / ratio
                bottom = (y2 - pad_top) / ratio

                left = max(0, min(w, left))
                top = max(0, min(h, top))
                right = max(0, min(w, right))
                bottom = max(0, min(h, bottom))

                width_box = right - left
                height_box = bottom - top

                if width_box <= 0 or height_box <= 0:
                    continue

                results_json.append({
                    "image_id": image_id,
                    "category_id": COCO_ID_MAP[cl],
                    "bbox": [round(left, 2), round(top, 2), round(width_box, 2), round(height_box, 2)],
                    "score": round(float(score), 4)
                })

        if idx % 100 == 0: print(f"Progress: {idx}/{len(img_list)}")

    with open("detections.json", "w") as f:
        json.dump(results_json, f)
    
    print("Avg Inference time per image", total / len(img_list))

    yolov8.nn_destory_network()
(venv) khadas2@Khadas2:~/accuracy$ cat eval_coco.py 
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

annFile = "/home/khadas2/dataset/COCO/annotations/instances_val2017.json"
resFile = "detections.json"

cocoGt = COCO(annFile)
cocoDt = cocoGt.loadRes(resFile)

cocoEval = COCOeval(cocoGt, cocoDt, "bbox")  # segm / keypoints 가능
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()
