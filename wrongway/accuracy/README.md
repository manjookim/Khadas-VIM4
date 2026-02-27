# Tracking by Detection 정확도 측정

### 정답 데이터 (GT) 생성

[CVAT](https://www.cvat.ai/) 회원가입 후 Track box 그려서 직접 라벨링

### 추론 데이터 생성

`wrongway_core.py` : tracking result 를 txt 파일로 저장   

### 정확도 계산 

```
pip install motmetrics --break-system-packages 
pip install "numpy<2.0" --break-system-packages
```
```
python3 eval.py
```

`eval.py` : MOTA, MOTP, IDF1, num_switches, precision, recall 값 계산 
- MOTA :
- MOTP :
- IDF1 :
- num_switches :
- precision :
- recall :
