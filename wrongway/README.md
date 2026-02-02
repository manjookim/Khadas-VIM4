# 역주행 검지 시스템 - VIM4 TIM-VX 패키지

## 개요

이 패키지는 Khadas VIM4 보드의 NPU (TIM-VX/Khadas NPU SDK)를 사용한 역주행 검지 시스템입니다. 
**스캐폴딩 패키지**로 제공되며, `inference_backend.py`만 구현하면 전체 시스템이 동작합니다.

### 패키지 목적

- ✅ **스캐폴딩 제공**: 업체가 NPU 연동 부분만 구현하면 전체 시스템 동작
- ✅ **I/O 계약 명확화**: `detect(frame_bgr) -> List[Det]` 인터페이스 정의
- ✅ **CPU 레퍼런스와 동일한 출력**: ROI/방향/이벤트/스냅샷/출력 JSON 스키마 일치

### 범위

- ✅ 지원 기능: 비디오 파일/카메라 입력, ROI 기반 역주행 검지, JSON 이벤트 출력

## 시스템 요구사항

### 하드웨어

- Khadas VIM4 보드
- 카메라 입력 (V4L2 지원)

### 운영체제

- Ubuntu 20.04 LTS 이상 (Khadas 공식 이미지 권장)

### 소프트웨어

- Python 3.8 이상
- Khadas NPU SDK (별도 설치 필요)
- V4L2 지원

## 설치

### 1. 패키지 다운로드 및 압축 해제

```bash
cd altech_wrongway_vim4_timvx
```

### 2. Khadas NPU SDK 설치

**중요**: 먼저 Khadas NPU SDK를 설치해야 합니다.

- 문서: https://github.com/manjookim/Khadas-VIM4/tree/main/
- 설치 방법 및 요구사항은 공식 문서 참조

### 3. venv 환경 설정

```bash
# 실행 권한 부여
chmod +x scripts/setup_venv.sh

# venv 생성 및 패키지 설치
./scripts/setup_venv.sh
```

### 4. venv 활성화

```bash
source venv/bin/activate
```

### 5. 모델 변환 (YOLO -> ADLA)

YOLO 모델을 ADLA 형식으로 변환해야 합니다.

- 변환 방법: Khadas 문서 참조
  - https://github.com/manjookim/Khadas-VIM4/tree/main/compile
- 변환된 모델 파일을 `models/` 디렉터리에 배치

## 설정

### ROI 설정 파일 (JSON)

[sample.roi.json](https://github.com/manjookim/Khadas-VIM4/blob/main/wrongway/configs/sample.roi.json) 을 참고하세요.

## 구현 가이드


#### CONTRACT (계약)

```python
def detect(self, frame: np.ndarray) -> List[Det]:
    """
    CONTRACT:
      - 입력: frame_bgr (H, W, 3), dtype=uint8, BGR 형식
      - 출력: List[(x1, y1, x2, y2, conf, class_id)]
      - 조건: 
        * NMS 완료
        * 원본 좌표 (letterbox 역변환 완료)
        * 차량만 (class_id: 2=car, 3=motorcycle, 5=bus, 7=truck)
        * 필터링 완료 (크기, 경계)
    """
```

#### 구현 단계

1. **NPU 엔진 초기화** (`__init__` 메서드)
   - 모델 파일 로드
   - 입력/출력 텐서 정보 확인

2. **전처리** (`detect` 메서드 내)
   - Letterbox 처리 (비율 유지, 640x640로 리사이즈)
   - BGR -> RGB 변환 
   - 정규화 
   - 스케일 및 패딩 정보 저장 (좌표 복원용)

3. **NPU 추론**
   - KSNN 엔진으로 추론 실행
   - 출력 텐서 파싱 (박스, 점수, 클래스)

4. **후처리**
   - NMS (Non-Maximum Suppression)
   - 좌표 복원 (letterbox 역변환)
   - 클래스 필터링 (차량만)
   - 크기/경계 필터링

#### 전처리/후처리 규약

**입력 해상도**: 640x640 (YOLO 표준)
**Letterbox 규칙**: 
- 원본 비율 유지
- 짧은 변을 640에 맞춤
- 긴 변은 중앙 정렬, 패딩 추가 (회색 또는 검정)
- 스케일 및 패딩 정보 저장 (좌표 복원용)

**좌표 복원 규칙**:
- 추론 결과 좌표를 letterbox 역변환
- 스케일 적용: `original_x = (detected_x - pad_x) / scale`
- 원본 프레임 좌표계로 변환

## 실행

### 기본 실행

```bash
# 실행 권한 부여 (최초 1회)
chmod +x scripts/run.sh

# 비디오 파일 입력
./scripts/run.sh --source video.mp4 --config configs/sample.roi.json --model models/model.adla --output events.log
```

### 명령줄 인자

| 인자 | 필수 | 기본값 | 설명 |
|------|------|--------|------|
| `--source` | ✅ | - | 비디오 파일 경로 또는 카메라 인덱스 |
| `--config` | ✅ | - | ROI 설정 파일 경로 (JSON) |
| `--model` | ❌ | `models/model.adla` | ADLA 모델 파일 경로 |
| `--library` | ✅ | - | ADLA 모델 라이브러리 경로 |
| `--device` | ❌ | `npu` | 디바이스 (일반적으로 npu) |
| `--conf` | ❌ | `0.35` | 신뢰도 임계값 |
| `--output` | ❌ | stdout | 이벤트 로그 파일 |
| `--queue` | ❌ | `4` | 프레임 큐 크기 |
| `--snapshot-dir` | ❌ | - | 스냅샷 저장 디렉터리 |
| `--mode` | ❌ | `full` | 추론 모드 선택 ('io', 'detect', 'full') |

## 출력

CPU 레퍼런스 패키지와 동일한 JSON Lines 형식 이벤트 로그를 출력합니다.

## NPU 연동 참조

### Khadas VIM4 NPU 문서

- NPU SDK: https://docs.khadas.com/products/sbc/vim4/npu/npu-sdk
- 모델 변환: https://docs.khadas.com/products/sbc/vim4/npu/npu-convert
- 애플리케이션 예제: Khadas 공식 예제 참조

### 모델 변환 체크리스트

- [ ] YOLO 모델 파일 준비 (예: `yolov8m.pt`)
- [ ] 변환 도구 설치
- [ ] 입력 해상도 확인 (예: 640x640)
- [ ] 출력 레이어 이름 확인
- [ ] ADLA 모델 파일 생성
- [ ] 모델 파일을 `models/` 디렉터리에 배치

## 트러블슈팅

### NPU SDK 관련

- Khadas 공식 문서 및 포럼 참조
- SDK 버전 호환성 확인
- 모델 변환 오류: 변환 도구 로그 확인

### 성능 최적화

- 프레임 큐 크기 조정 (`--queue`)
- 배치 처리 고려 (여러 프레임 동시 추론)
- 메모리 할당 최적화

## 검증

샘플 비디오/설정으로 실행하여 이벤트 로그가 생성되는지 확인하세요.

## 문의

기술 지원이 필요한 경우 문의해 주세요.










