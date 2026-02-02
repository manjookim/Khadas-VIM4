# -*- coding: utf-8 -*-

import os
import time
import glob
import numpy as np
from PIL import Image
import onnxruntime as ort # ONNX 런타임 라이브러리

# 필요한 라이브러리 설치:
# pip install onnxruntime numpy Pillow

num_runs = 5

onnx_path = "../model/yolov8n.onnx"
image_dir = "../dataset/COCO/val2017_1000_sample"
num_images = 100

image_paths = sorted(glob.glob(f"{image_dir}/*.jpg"))[:num_images]
if not image_paths:
    raise FileNotFoundError(f"No JPG images found in {image_dir}")

if not os.path.exists(onnx_path):
    raise FileNotFoundError(f"ONNX file not found at: {onnx_path}")

# ONNX 런타임 세션 생성 (CPU 사용)
session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

# 모델의 입력 정보 가져오기
input_name = session.get_inputs()[0].name
_, _, h, w = session.get_inputs()[0].shape  # (batch_size, channels, height, width)

print("Input info:")
print(f"Name: {input_name}")
print(f"Shape: (1, 3, {h}, {w})") # ONNX는 보통 NCHW 형식을 사용합니다.


image_data_list = []
print(f"\nPreprocessing {len(image_paths)} images...")
for img_path in image_paths:
    img = Image.open(img_path).convert("RGB").resize((w, h))
    img_np = np.array(img, dtype=np.float32)
    # 이미지 전처리: HWC -> CHW 및 0-1 정규화
    img_np = img_np / 255.0
    img_np = img_np.transpose(2, 0, 1)  # (height, width, channels) -> (channels, height, width)
    img_np = np.expand_dims(img_np, axis=0) # 배치 차원 추가 (1, C, H, W)
    image_data_list.append(img_np)
print("Preprocessing complete.")

total_inference_time_sum = 0.0

print("\nCPU inference is about to start. Press Ctrl-C on this terminal to stop.")
print("Ready... Starting inference in 3 seconds.")
time.sleep(1)
print("2...")
time.sleep(1)
print("1...")
time.sleep(1)

print("\n" + "="*30)
print(">>> INFERENCE START! <<<")
print("="*30 + "\n")

for run_count in range(num_runs):
    print(f"\n--- Starting inference run {run_count+1}/{num_runs} ---")
    
    processing_times = []

    # Infer each image individually with batch size 1
    num_processed = 0
    for img_np in image_data_list:
        start_time = time.time()
        
        _ = session.run(None, {input_name: img_np})
        num_processed += 1
        
        end_time = time.time()
        duration = end_time - start_time
        processing_times.append(duration)
    
    current_run_time = sum(processing_times)
    current_run_std = np.std(processing_times)
    total_inference_time_sum += current_run_time

    if num_processed > 0:
        avg_time_per_image = current_run_time / num_processed
        fps = 1 / avg_time_per_image
        print(f"Run {run_count+1}: Total images processed: {num_processed}")
        print(f"Run {run_count+1}: Total inference time: {current_run_time * 1000:.2f} ms")
        print(f"Run {run_count+1}: Average inference time per image: {avg_time_per_image * 1000:.2f} ms")
        print(f"Run {run_count+1}: std : {current_run_std * 1000:.2f}ms")
        print(f"Run {run_count+1}: Average FPS: {fps:.2f}")
        print("------------------------------------------")

# Calculate and print overall average time
if num_runs > 0:
    overall_avg_total_time = total_inference_time_sum / num_runs
    overall_avg_time_per_image = overall_avg_total_time / num_images
    overall_avg_fps = 1 / overall_avg_time_per_image

    print("\n" + "=" * 50)
    print(f"Overall Results for {num_runs} runs:")
    print(f"Average Total Inference Time for {num_images} images: {overall_avg_total_time * 1000:.2f} ms")
    print(f"Average Inference Time per Image: {overall_avg_time_per_image * 1000:.2f} ms")
    print(f"Overall Average FPS: {overall_avg_fps:.2f}")
