## Compile
1. pt -> onnx 변환    
```
yolo mode=export model=yolov8n.pt format=onnx
```

2. ksnn_args.txt 수정 (커스텀)
```
nano ksnn_args.txt
```
- 전처리 포함한 추론 함수 사용하려면
- `--inference-input-type`
- `--inference-output-type`
- 파라미터 모두 삭제 후 컴파일 

3. 모델 변환 스크립트 실행    
```
bash convert-in-docker.sh ksnn
```




## Reference
