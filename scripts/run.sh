#!/bin/bash
# 실행 스크립트 (VIM4)

set -e

# 스크립트 디렉토리로 이동
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PACKAGE_DIR"

# venv 활성화
if [ ! -d "venv" ]; then
    echo "ERROR: venv가 없습니다. 먼저 scripts/setup_venv.sh를 실행하세요."
    exit 1
fi

source venv/bin/activate

# PYTHONPATH 설정
export PYTHONPATH="$PACKAGE_DIR:$PYTHONPATH"

# 실행
python -m app.wrongway_cli "$@"
