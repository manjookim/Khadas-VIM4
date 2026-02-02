## ByteTrack 설정

1. 라이브러리 설치    
```
sudo apt-get update
sudo apt-get install python3-dev build-essential
pip install lap xonsh
```
2. cython bbox 설치    
```
git clone https://github.com/samson-wang/cython_bbox.git
cd cython_bbox
pip install -e .
```
+ `.so` 파일을 ByteTrack 디렉토리 (상위 디렉토리)로 복사

3. ByteTrack 디렉토리 수정    
   a) `__init__.py` 생성
   ```
   touch ~/accuracy/ByteTrack/__init__.py
   ```
   b) `byte_tracker.py` import 부분 수정
   ```
   from .kalman_filter import KalmanFilter
   from . import matching
   from .basetrack import BaseTrack, TrackState
   ```
   c) `matching.py` import 부분 수정
   ```
   from cython_bbox import bbox_overlaps as bbox_ious
   #from yolox.tracker import kalman_filter
   from .kalman_filter import chi2inv95
   ```




### Reference
[https://github.com/FoundationVision/ByteTrack](https://github.com/FoundationVision/ByteTrack)    
[https://github.com/samson-wang/cython_bbox](https://github.com/samson-wang/cython_bbox)    
