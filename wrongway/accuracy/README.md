# Tracking by Detection 정확도 측정

### 정답 데이터 (GT) 생성

[CVAT](https://www.cvat.ai/) 회원가입 후 Track box 그려서 직접 라벨링

### 추론 데이터 생성

`wrongway_core.py` : tracking result 를 txt 파일로 저장   

### 정확도 계산 

```
pip install motmetrics  
pip install "numpy<2.0" 
```
```
python3 eval.py
```

`eval.py` : MOTA, MOTP, IDF1, num_switches, precision, recall 값 계산 
- MOTA  :  종합적인 정확도
- MOTP  :  IoU의 평균값 
- IDF1  :  객체의 동일한 ID 부여 일관성
- num_switches  :  한 객체의 ID 번호 바뀐 횟수
- precision  :  정밀도 
- recall  :  재현율 
