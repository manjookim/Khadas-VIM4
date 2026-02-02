#!/bin/bash
# venv 설정 스크립트 (VIM4)

set -e

echo "=========================================="
echo "역주행 검지 시스템 - venv 설치 (VIM4)"
echo "=========================================="
echo ""

# Python 버전 확인
echo "[1/4] Python 버전 확인..."
python3 --version || {
    echo "ERROR: Python 3이 설치되어 있지 않습니다."
    exit 1
}
echo ""

# venv 생성
echo "[2/4] venv 생성..."
python3 -m venv venv
echo "venv 생성 완료"
echo ""

# venv 활성화 및 기본 패키지 설치
echo "[3/4] 기본 패키지 설치..."
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo ""

# Khadas NPU SDK 설치 안내
echo "[4/4] Khadas NPU SDK 설치 안내"
echo ""
echo "=========================================="
echo "Khadas NPU SDK 설치가 필요합니다."
echo "=========================================="
echo ""
echo "설치 방법은 다음 문서를 참조하세요:"
echo "  https://docs.khadas.com/products/sbc/vim4/npu/npu-sdk"
echo ""
echo "설치 후 inference_backend.py의 TODO를 구현하세요."
echo ""

echo "=========================================="
echo "설치 완료!"
echo "=========================================="
echo ""
echo "다음 명령으로 venv를 활성화하세요:"
echo "  source venv/bin/activate"
echo ""
