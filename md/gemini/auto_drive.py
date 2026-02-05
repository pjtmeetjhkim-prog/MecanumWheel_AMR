import cv2
import torch
import numpy as np
import time
import sys
import os

# 1. 경로 설정 (반드시 확인!)
yolo_dir = '/home/pi/YOLOv5-Lite-master'
model_path = os.path.join(yolo_dir, 'weights/v5lite-e.pt')

# YOLOv5-Lite 내부 모듈을 인식시키기 위해 경로 추가
sys.path.append(yolo_dir)

from Raspbot_Lib import Raspbot # 파일명에 맞춰 수정하세요
bot = Raspbot()

# 2. 모델 로드 (에러 방지용 직접 로드 방식)
try:
    # weights 파일만 직접 불러오기
    model = torch.load(model_path, map_location='cpu')
    if isinstance(model, dict):
        model = model['model']
    model.float().eval()
    print("모델 직접 로드 성공!")
except Exception as e:
    print(f"모델 로드 에러: {e}")
    print("팁: YOLOv5-Lite-master 폴더 안에 models 폴더가 있는지 확인해주세요.")
    exit()

# 3. 주행 제어 함수
def car_control(speed):
    for i in range(4):
        bot.Ctrl_Car(i, 0, speed)

# 4. 카메라 및 감지 루프
cap = cv2.VideoCapture(0)
cap.set(3, 320)
cap.set(4, 240)

# ... (상단 모델 로드 부분은 동일) ...

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # 1. 객체 탐지 수행
        results = model(frame)
        
        # 2. pandas 대신 원본 데이터(xyxy) 직접 사용
        # results.xyxy[0]은 [xmin, ymin, xmax, ymax, conf, class] 형태의 행렬입니다.
        detections = results.xyxy[0].cpu().numpy()
        
        person_detected = False
        
        for det in detections:
            conf = det[4]
            cls = int(det[5])
            # YOLOv5 기본 모델에서 사람(person)의 클래스 번호는 0번입니다.
            if cls == 0 and conf > 0.4:
                person_detected = True
                break

        # 3. 주행 로직
        if person_detected:
            print("사람 감지! 정지")
            car_stop()
            bot.Ctrl_WQ2812_ALL(1, 0)
        else:
            car_forward(100)
            bot.Ctrl_WQ2812_ALL(1, 1)

        cv2.imshow('YOLOv5-Lite (No Pandas)', np.squeeze(results.render()))
# ... (하단 종료 부분 동일) ...

finally:
    car_control(0)
    cap.release()
    cv2.destroyAllWindows()