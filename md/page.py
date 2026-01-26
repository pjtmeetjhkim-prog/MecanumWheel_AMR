import sys
import os
import time
import math
import cv2
import torch
import torch.nn as nn
from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit
from unittest.mock import MagicMock

# 1. 라이브러리 에러 방지 (Mocking)
sys.modules["pandas"] = MagicMock()
sys.modules["seaborn"] = MagicMock()
sys.modules["matplotlib"] = MagicMock()
sys.modules["matplotlib.pyplot"] = MagicMock()

from Raspbot_Lib import Raspbot
from McLumk_Wheel_Sports import McLumk_Sports

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

bot = Raspbot()
car = McLumk_Sports(bot)
camera = cv2.VideoCapture(0)
camera.set(3, 320)
camera.set(4, 320)

# 2. YOLO 모델 로드 (패치 적용)
model = None
try:
    yolo_path = '/home/pi/YOLOv5-Lite-master'
    sys.path.append(yolo_path)
    weights_path = os.path.join(yolo_path, 'weights/v5lite-e.pt')
    
    ckpt = torch.load(weights_path, map_location='cpu')
    model = ckpt['model'].float().eval()
    
    # Upsample 호환성 패치
    for m in model.modules():
        if isinstance(m, nn.Upsample):
            m.recompute_scale_factor = None
    print("✅ YOLO 모델 로드 완료")
except Exception as e:
    print(f"⚠️ 모델 로드 실패: {e}")

# 거리 측정 함수
def get_stable_distance():
    readings = []
    for _ in range(2):
        h = bot.read_data_array(0x1b, 1)[0]
        l = bot.read_data_array(0x1a, 1)[0]
        d = (h << 8 | l) / 10.0
        if 2 < d < 450: readings.append(d)
        time.sleep(0.005)
    return sum(readings) / len(readings) if readings else 0

# [핵심] 3D 스캔 핸들러
@socketio.on('start_3d_scan')
def handle_scan():
    bot.Ctrl_Ulatist_Switch(1)
    print("스캔 시작 (고정밀 모드)...")
    
    # 1. 틸트(위아래) 간격을 2도로 매우 촘촘하게 설정
    for tilt in range(30, 85, 2):
        bot.Ctrl_Servo(2, tilt) 
        time.sleep(0.1) # 모터 안정화
        
        # 2. 팬(좌우) 간격도 2도로 설정
        for pan in range(0, 181, 2):
            bot.Ctrl_Servo(1, pan)
            # 딜레이를 최소화하여 전체 속도 보정
            time.sleep(0.02)
            
            # A. 초음파 거리 측정
            sonic_dist = get_stable_distance()
            
            # B. YOLO 시각적 거리 추정
            visual_dist = 0
            if model:
                success, frame = camera.read()
                if success:
                    try:
                        # 이미지 전처리
                        img = cv2.resize(frame, (320, 320))
                        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0
                        
                        with torch.no_grad():
                            pred = model(img_tensor)[0]
                            # 신뢰도 0.5 이상인 물체 필터링
                            if pred.shape[0] > 0:
                                best_obj = pred[torch.argmax(pred[:, 4])]
                                conf = best_obj[4]
                                
                                if conf > 0.5:
                                    # 박스 높이 계산 (y2 - y1)
                                    h_pixels = best_obj[3] - best_obj[1]
                                    
                                    # [초점 거리 공식] 거리는 픽셀 크기에 반비례
                                    # 6000은 보정 계수 (K_FACTOR). 카메라에 맞춰 조절 가능
                                    # 물체가 가까우면 h가 커지고 dist는 작아짐
                                    visual_dist = 6000 / h_pixels.item()
                    except:
                        pass

            # C. 센서 융합 (Sensor Fusion)
            # 초음파가 유효하면 초음파 우선, 아니면 시각 거리 사용
            # 혹은 둘 다 있으면 평균값 사용 등 전략 선택
            
            final_dist = 0
            # 1순위: 초음파 (가장 정확함)
            if 2 < sonic_dist < 400:
                final_dist = sonic_dist
            # 2순위: YOLO (초음파가 놓쳤을 때 보완)
            elif visual_dist > 0:
                final_dist = visual_dist
            
            # 유효한 거리 데이터가 없으면 스킵
            if final_dist == 0: continue

            # 좌표 변환
            p_rad = math.radians(180 - pan)
            t_rad = math.radians(tilt)
            
            x = final_dist * math.cos(t_rad) * math.cos(p_rad)
            y = final_dist * math.sin(t_rad)
            z = final_dist * math.cos(t_rad) * math.sin(p_rad)
            
            emit('3d_data', {
                'x': x, 'y': y, 'z': z, 
                'dist': final_dist
            })

    bot.Ctrl_Servo(1, 90)
    bot.Ctrl_Servo(2, 50)
    print("스캔 완료")

# (주행 핸들러 및 라우팅은 기존과 동일)
@socketio.on('move')
def handle_move(data):
    cmd = data['cmd']
    if cmd == 'stop': car.stop_robot()
    elif cmd == 'forward': car.move_forward(100)
    elif cmd == 'backward': car.move_backward(100)
    elif cmd == 'left': car.move_left(100)
    elif cmd == 'right': car.move_right(100)
    elif cmd == 'q_front': car.move_diagonal_left_front(100)
    elif cmd == 'e_front': car.move_diagonal_right_front(100)
    elif cmd == 'z_back': car.move_diagonal_left_back(100)
    elif cmd == 'c_back': car.move_diagonal_right_back(100)

@app.route('/')
def index(): return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    def gen():
        while True:
            _, frame = camera.read()
            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)