from ksnn.api import KSNN
import cv2
import numpy as np
import time

model = KSNN('VIM4')

version = model.get_nn_version()
print('KSNN version is' , version)

model.nn_init(library='/home/khadas2/model/libnn_yolov8n.so', model='/home/khadas2/model/yolov8n_int8.adla', level=2)

picture = '/home/khadas2/dataset/val2017/000000147725.jpg'

orig_img = cv2.imread(picture, cv2.IMREAD_COLOR)
img = cv2.resize(orig_img, (640, 640)).astype(np.float32)
img[:, :, 0] = img[:, :, 0] - 0
img[:, :, 1] = img[:, :, 1] - 0
img[:, :, 2] = img[:, :, 2] - 0
img = img / 255

print('Start inference')
start_time = time.time()

#Step by Step Code.

#model.nn_set_inputs(img, input_shape=(640, 640, 3), input_type="RAW")
#model.nn_run(output_type="RAW")
#outputs = model.nn_get_outputs(output_shape=[(80,80,144), (40,40,144), (20,20,144)])

#One Touch Code.
outputs = model.nn_inference(img, input_shape = (640, 640, 3), input_type="RAW", output_shape=[(80, 80, 144), (40,40,144), (20,20,144)], output_type="RAW")

end_time = time.time()
print(f'Done. Inference time : {(end_time - start_time)*1000:.2f} ms')

#Check Result
print(f"Output 0 shape: {outputs[0].shape}")
print(f"Output 1 shape: {outputs[1].shape}")
print(f"Output 2 shape: {outputs[2].shape}")
print(f"First 10 values of output 0: {outputs[0].flatten()[:10]}")

#Release NPU
model.nn_destory_network()
