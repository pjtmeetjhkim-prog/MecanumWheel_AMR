# page.py (I2C 폭주 방지 최종판)

import sys
import os
import time
import threading
import math
import csv
import signal
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

import cv2
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit

hardware_lock = threading.Lock()

# 설정
VISUAL_SCALE_FORWARD = 10.0 
VISUAL_SCALE_ROTATE = 5.0
SPEED_CM_PER_SEC = 15.0 
LEFT_TRIM = 1.0
RIGHT_TRIM = 0.95

try:
    from Raspbot_Lib import Raspbot
    bot = Raspbot()
except ImportError:
    class MockBot:
        def Ctrl_Servo(self, i, a): pass
        def Ctrl_Muto(self, i, s): pass
        def Ctrl_Ulatist_Switch(self, s): pass
        def read_data_array(self, r, l): return [0]
    bot = MockBot()

# 🛡️ I2C 안전 제어 (딜레이 증가)
def safe_motor(m1, m2, m3, m4):
    with hardware_lock:
        try:
            bot.Ctrl_Muto(0, int(m1)); bot.Ctrl_Muto(1, int(m2))
            bot.Ctrl_Muto(2, int(m3)); bot.Ctrl_Muto(3, int(m4))
        except: pass
    time.sleep(0.01) # 0.01초 대기 (안정성 확보)

def safe_servo(id, angle):
    with hardware_lock:
        try: bot.Ctrl_Servo(id, angle)
        except: pass
    time.sleep(0.01)

def safe_read_ultrasonic():
    with hardware_lock:
        try:
            h = bot.read_data_array(0x1b, 1)[0]
            l = bot.read_data_array(0x1a, 1)[0]
            return (h << 8) | l
        except: return -1

# YOLO
YOLO_PATH = '/home/pi/YOLOv5-Lite-master'
if not os.path.exists(YOLO_PATH): YOLO_PATH = '/home/pi/yolov5-lite-master'
try:
    sys.path.insert(0, YOLO_PATH)
    from models.experimental import attempt_load
    from utils.general import non_max_suppression
    model = attempt_load(os.path.join(YOLO_PATH, 'weights/v5lite-e.pt'), map_location='cpu')
    for m in model.modules():
        if isinstance(m, nn.Upsample): m.recompute_scale_factor = None
    print("✅ YOLO 로드 완료")
except: model = None

app = Flask(__name__)
socketio = SocketIO(app, async_mode='threading', cors_allowed_origins='*')

is_running = True
is_auto_driving = False
use_ai = False  
robot_pose = {'x': 0, 'y': 0, 'theta': 90}
robot_path = []
real_distance = 0
cam_pan = 90
cam_tilt = 30

last_servo_time = 0
last_motor_time = 0 # 모터 쿨타임

LOG_FILE = 'scan_log.csv'
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, 'w', newline='') as f:
        csv.writer(f).writerow(['Time', 'Object', 'Distance(cm)', 'Angle'])

def save_log(name, dist, angle):
    try:
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(LOG_FILE, 'a', newline='') as f:
            csv.writer(f).writerow([now, name, f"{dist:.1f}", angle])
    except: pass

class VisualOdometry:
    def __init__(self):
        self.prev_gray = None
        self.p0 = None
        self.lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

    def process(self, frame):
        if frame is None: return 0, 0
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (160, 120)) 
        if self.prev_gray is None:
            self.prev_gray = gray.copy()
            self.p0 = cv2.goodFeaturesToTrack(self.prev_gray, mask=None, **self.feature_params)
            return 0, 0
        if self.p0 is None or len(self.p0) < 5:
            self.p0 = cv2.goodFeaturesToTrack(self.prev_gray, mask=None, **self.feature_params)
            self.prev_gray = gray.copy()
            return 0, 0
        p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.p0, None, **self.lk_params)
        dx, dy = 0, 0
        if p1 is not None:
            good_new = p1[st == 1]; good_old = self.p0[st == 1]
            vec = good_new - good_old
            if len(vec) > 0: dx = np.mean(vec[:, 0]); dy = np.mean(vec[:, 1])
            self.prev_gray = gray.copy(); self.p0 = good_new.reshape(-1, 1, 2)
        return dx, dy

vo = VisualOdometry()

class CameraThread:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src, cv2.CAP_V4L2)
        self.stream.set(3, 320); self.stream.set(4, 240); self.stream.set(5, 30)
        self.grabbed = False
        self.frame = None
        self.stopped = False
    def start(self):
        t = threading.Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self
    def update(self):
        while not self.stopped and is_running:
            if self.stream.isOpened():
                (grabbed, frame) = self.stream.read()
                if grabbed: self.grabbed = True; self.frame = frame
                else: self.grabbed = False
            else: time.sleep(1.0)
            time.sleep(0.01)
    def read(self): return self.frame
    def stop(self): self.stopped = True; self.stream.release()

webcam = None
for i in [0, 1]:
    cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
    if cap.isOpened(): cap.release(); webcam = CameraThread(src=i).start(); print(f"📸 카메라 {i}번 시작"); break
if not webcam: print("❌ 카메라 없음")

# 🦇 초음파 스레드 (주기 0.3초로 변경 - 중요!)
def ultrasonic_task():
    global real_distance
    try:
        with hardware_lock: bot.Ctrl_Ulatist_Switch(1)
    except: pass
    while is_running:
        mm = safe_read_ultrasonic()
        if mm != -1 and 0 < mm < 5000: real_distance = mm / 10.0
        
        # 🚨 [수정] 0.3초 대기 (모터가 통신할 시간을 줌)
        time.sleep(0.3) 

t_ultra = threading.Thread(target=ultrasonic_task); t_ultra.daemon = True; t_ultra.start()

def update_pose_visual(dx, dy):
    global robot_pose, robot_path
    move = dy / VISUAL_SCALE_FORWARD
    rot = dx / VISUAL_SCALE_ROTATE
    if abs(move) < 0.2: move = 0
    if abs(rot) < 0.5: rot = 0
    robot_pose['theta'] += rot
    rad = math.radians(robot_pose['theta'])
    robot_pose['x'] += move * math.cos(rad)
    robot_pose['y'] += move * math.sin(rad)
    if abs(move) > 0 or abs(rot) > 0:
        robot_path.append({'x': robot_pose['x'], 'y': robot_pose['y']})
        socketio.emit('map_update', {'robot': robot_pose})

def update_map(angle, dist):
    global robot_pose
    rad = math.radians(robot_pose['theta'] + (angle - 90))
    ox = robot_pose['x'] + (dist * math.cos(rad))
    oy = robot_pose['y'] + (dist * math.sin(rad))
    socketio.emit('map_update', {'obstacle': {'x': ox, 'y': oy}})

def auto_drive_task():
    global is_auto_driving, robot_pose, cam_pan
    print("🤖 자율 주행 시작")
    while is_auto_driving and is_running:
        scan_data = []
        for ang in range(0, 181, 45):
            if not is_auto_driving: break
            cam_pan = ang; safe_servo(1, ang); time.sleep(0.3)
            d = real_distance; scan_data.append((ang, d))
            if 0 < d < 200: update_map(ang, d)
        if not is_auto_driving: break

        best_angle = 90; max_dist = 0; blocked = False
        for ang, d in scan_data:
            if d > max_dist: max_dist = d; best_angle = ang
            if 60 <= ang <= 120 and 0 < d < 30: blocked = True

        if blocked:
            safe_motor(-40, -40, -40, -40); time.sleep(0.5)
            best_angle = 0 if best_angle > 90 else 180
        
        safe_servo(1, 90)
        turn = 90 - best_angle
        if turn != 0:
            L, R = (50, -50) if turn > 0 else (-50, 50)
            safe_motor(L, L, R, R); time.sleep(0.5); safe_motor(0,0,0,0); time.sleep(0.2)
        
        for _ in range(10):
            if not is_auto_driving or real_distance < 25: break
            safe_motor(40, 40, 40, 40); time.sleep(0.1)
        safe_motor(0,0,0,0); time.sleep(0.5)
    is_auto_driving = False

@app.route('/')
def index(): return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    def gen():
        frame_cnt=0; last_preds=[]; last_save=0
        while is_running:
            if webcam and webcam.grabbed:
                frame = webcam.read().copy()
                h, w, _ = frame.shape
                
                dx, dy = vo.process(frame)
                if abs(dx) < 50 and abs(dy) < 50: update_pose_visual(dx, dy)

                color=(0,255,0)
                if real_distance<30: color=(0,0,255)
                cv2.rectangle(frame,(0,0),(w,h),color,3)
                cv2.putText(frame,f"D:{real_distance:.0f}cm",(10,h-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,color,2)
                
                if use_ai and model:
                    frame_cnt+=1
                    if frame_cnt%5==0:
                        img=cv2.resize(frame,(320,320)); img=img[:,:,::-1].transpose(2,0,1)
                        img=np.ascontiguousarray(img); img=torch.from_numpy(img).float()/255.0
                        if img.ndimension()==3: img=img.unsqueeze(0)
                        pred=model(img)[0]; pred=non_max_suppression(pred,0.4,0.45); last_preds=pred
                    if len(last_preds)>0:
                        for det in last_preds[0]:
                            x1=int(det[0]*w/320); y1=int(det[1]*h/320); x2=int(det[2]*w/320); y2=int(det[3]*h/320)
                            cls=int(det[5]); name=model.names[cls] if hasattr(model,'names') else "Obj"
                            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
                            cv2.putText(frame,name,(x1,y1-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)
                            if time.time()-last_save>1.0: save_log(name,real_distance,cam_pan); last_save=time.time()
                
                ret, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
                if ret: yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'+buf.tobytes()+b'\r\n')
                time.sleep(0.01)
            else: time.sleep(0.1)
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@socketio.on('toggle_ai')
def toggle_ai(): 
    global use_ai; use_ai = not use_ai
    print(f"🤖 AI 모드: {use_ai}")
    emit('ai_state', {'enabled': use_ai}, broadcast=True)

@socketio.on('toggle_auto_drive')
def toggle_auto(): 
    global is_auto_driving; is_auto_driving = not is_auto_driving
    if is_auto_driving: socketio.start_background_task(auto_drive_task)

@socketio.on('car_control')
def car(d): 
    global last_motor_time
    if time.time() - last_motor_time < 0.1: return # 0.1초 제한
    
    safe_motor(int((d['y']*150)+(d['x']*100)), int((d['y']*150)+(d['x']*100)), int((d['y']*150)-(d['x']*100)), int((d['y']*150)-(d['x']*100)))
    last_motor_time = time.time()

@socketio.on('car_stop')
def stop(): safe_motor(0,0,0,0)

@socketio.on('cam_control')
def cam_control(data):
    global cam_pan, cam_tilt, last_servo_time
    if is_auto_driving: return 
    if time.time() - last_servo_time < 0.1: return # 0.1초 제한 (서보 보호)
    
    try:
        x = float(data.get('x', 0)); y = float(data.get('y', 0))
        cam_pan -= x * 1.5; cam_tilt += y * 1.5 
        cam_pan = max(0, min(180, cam_pan)); cam_tilt = max(0, min(100, cam_tilt))
        safe_servo(1, cam_pan); safe_servo(2, cam_tilt)
        last_servo_time = time.time()
    except: pass

@socketio.on('reset_map')
def reset(): global robot_pose, robot_path; robot_pose={'x':0,'y':0,'theta':90}; robot_path=[]

def cleanup(sig, frame):
    global is_running
    print("\n🛑 종료...")
    is_running = False
    safe_motor(0,0,0,0)
    if webcam: webcam.stop()
    os._exit(0)
signal.signal(signal.SIGINT, cleanup)

if __name__ == '__main__':
    try:
        print("🚀 서버 시작...")
        socketio.run(app, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)
    except: cleanup(None, None)