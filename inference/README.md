## Khadas new VIM4 추론 

- `test.py` : 기본 API 사용한 심플한 추론 코드    
- `tracking_video.py` : yolov8 모델을 이용한 비디오 추론 코드    
- `cpu_infer.py` : cpu를 이용한 추론    
- `npu_infer.py` : npu를 이용한 추론   



### 전처리 방식 2가지

- 전처리 방식에 따라 [ksnn_args.txt](https://github.com/manjookim/Khadas-VIM4/blob/main/compile/yolov8n/ksnn_args.txt) 수정 필요 

1. RAW : cpu 전처리 코드 별도 필요 
```
orig_img = cv.imread(picture, cv.IMREAD_COLOR)
        h, w = orig_img.shape[:2]
        # --- [Letterbox 적용] ---
        img_pad, ratio, (pad_left, pad_top) = letterbox(orig_img, (640, 640))

        img_rgb = cv.cvtColor(img_pad, cv.COLOR_BGR2RGB)
        img = np.ascontiguousarray(img_rgb, dtype=np.uint8)
        img_rgb = img_pad[:, :, ::-1]
        img = np.ascontiguousarray(img_rgb)
        #img = img_rgb.astype(np.float32)
        
        img[:, :, 0] -= mean[0]; img[:, :, 1] -= mean[1]; img[:, :, 2] -= mean[2]
        img /= var[0]

        start = time.time()
        
        data = yolov8.nn_inference(img_rgb, input_shape=(640, 640, 3), input_type="RAW", 
                                   output_shape=[(40, 40, 144), (80, 80, 144), (20, 20, 144)], 
                                   output_type="RAW")
```
2. RGB : npu 내부에서 자동으로 수행     
```
orig_img = cv.imread(picture, cv.IMREAD_COLOR)
        h, w = orig_img.shape[:2]
        # --- [Letterbox 적용] ---
        img_pad, ratio, (pad_left, pad_top) = letterbox(orig_img, (640, 640))

        img_rgb = cv.cvtColor(img_pad, cv.COLOR_BGR2RGB)

        start = time.time()
        
        data = yolov8.nn_inference(img_rgb, input_shape=(640, 640, 3), input_type="RGB", 
                                   output_shape=[(40, 40, 144), (80, 80, 144), (20, 20, 144)], 
                                   output_type="FLOAT")
```
