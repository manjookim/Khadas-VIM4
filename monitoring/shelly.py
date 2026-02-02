import requests
import time
import csv
from datetime import datetime

# 1. Shelly의 IP 주소를 입력하세요 (아까 찾은 172.20.10.x 대역)
SHELLY_IP = "172.20.10.5" 
OUTPUT_FILE = "shelly_power_data.csv"

print(f"{SHELLY_IP}에서 데이터 수집을 시작합니다. 중단하려면 Ctrl+C를 누르세요.")

# CSV 파일 헤더 생성
with open(OUTPUT_FILE, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Timestamp", "Power (W)", "Voltage (V)"])

try:
    while True:
        # Shelly Plus 모델의 상태를 가져오는 RPC 호출
        url = f"http://{SHELLY_IP}/rpc/Switch.GetStatus?id=0"
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            power = data["apower"]    # 현재 전력 (W)
            voltage = data["voltage"]  # 현재 전압 (V)
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # 화면 출력 및 파일 저장
            print(f"[{now}] Power: {power}W, Voltage: {voltage}V")
            with open(OUTPUT_FILE, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([now, power, voltage])
        
        time.sleep(1)

except KeyboardInterrupt:
    print("\n수집을 중단합니다. 파일이 저장되었습니다.")
except Exception as e:
    print(f"에러 발생: {e}")
