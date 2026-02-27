import motmetrics as mm
import pandas as pd
import numpy as np

# 1. 데이터 로드
def load_mot_file(filepath):
    # pandas로 로드하여 프레임 인덱스 확인 및 보정
    df = pd.read_csv(filepath, header=None)
    if len(df.columns) == 9:
        df.columns = ['frame', 'id', 'x', 'y', 'w', 'h', 'conf', 'class', 'vis']
    elif len(df.columns) == 10:
        df.columns = ['frame', 'id', 'x', 'y', 'w', 'h', 'conf', 'class', 'vis', 'dummy']
    return df

gt_df = load_mot_file('k1-1_gt.txt')
ts_df = load_mot_file('results.txt')


# 2. 누적기(Accumulator) 생성
acc = mm.MOTAccumulator(auto_id=True)

# 3. 프레임별로 데이터 비교
for frame in gt_df['frame'].unique():
    gt_frame = gt_df[gt_df['frame'] == frame]
    ts_frame = ts_df[ts_df['frame'] == frame]
    
    # 각 프레임의 박스 좌표 추출 [[x, y, w, h], ...]
    gt_boxes = gt_frame[['x', 'y', 'w', 'h']].values
    ts_boxes = ts_frame[['x', 'y', 'w', 'h']].values
    
    # 객체 ID 추출
    gt_ids = gt_frame['id'].values
    ts_ids = ts_frame['id'].values
    
    # IoU 기반 거리 행렬 계산 (1 - IoU)
    iou_distance = mm.distances.iou_matrix(gt_boxes, ts_boxes, max_iou=0.5)
    
    # 누적기에 업데이트
    acc.update(gt_ids, ts_ids, iou_distance)

# 5. 메트릭 계산 및 출력
mh = mm.metrics.create()
summary = mh.compute(acc, metrics=['mota', 'motp', 'idf1', 'num_switches', 'precision', 'recall'], name='VIM4_YOLO_Tracker')

print("\n--- 트래킹 정확도 측정 결과 ---")
print(summary)
