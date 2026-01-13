### Accuracy 측정
1. 결과 json 파일 추출
```
python3 detection.py --model /home/khadas2/model/t3/yolov8n_int8.adla --library /home/khadas2/model/t3/libnn_yolov8n.so --dataset /home/khadas2/dataset/COCO/val2017_1000_sample/
```

[yolov8n/letterbox.py](https://github.com/manjookim/Khadas-VIM4/blob/main/accuracy/yolov8n/letterbox.py)    
+ letterbox 형식으로 resize (공식 코드 수정)
+ [accuracy] 예측 결과 json 파일 출력
+ [inference time] per second 출력     

2. accuracy 측정
```
python3 eval_coco.py
```


## Reference    
[https://github.com/khadas/ksnn-vim4](https://github.com/khadas/ksnn-vim4)
