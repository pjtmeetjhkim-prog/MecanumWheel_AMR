import cv2
import time
from Raspbot_Lib import Raspbot

bot = Raspbot()

SAFE_DIST = 250  # mm

#################################
# 초음파
#################################
def get_distance():
    bot.Ctrl_Ulatist_Switch(1)
    time.sleep(0.05)
    H = bot.read_data_array(0x1b,1)[0]
    L = bot.read_data_array(0x1a,1)[0]
    bot.Ctrl_Ulatist_Switch(0)
    return (H<<8) | L


#################################
# 메카넘 제어 (pdf 표 기준 매핑) :contentReference[oaicite:1]{index=1}
#################################
def drive(vx, vy, w):
    L1 = vx - vy - w
    L2 = vx + vy - w
    R1 = vx + vy + w
    R2 = vx - vy + w

    motors = [L1, L2, R1, R2]

    for i, m in enumerate(motors):
        bot.Ctrl_Muto(i, int(m))


def stop():
    for i in range(4):
        bot.Ctrl_Muto(i, 0)


#################################
# AI (임시 규칙 기반 → 나중에 학습 모델 교체)
#################################
def ai_policy(frame):
    h, w, _ = frame.shape
    center = frame[:, w//3:2*w//3]

    gray = cv2.cvtColor(center, cv2.COLOR_BGR2GRAY)
    mean = gray.mean()

    if mean < 60:
        return 0, 0, 0   # 정지
    else:
        return 80, 0, 0  # 전진


#################################
# 디버그 UI
#################################
def draw_debug(frame, dist, action):
    text = f"dist:{dist}  action:{action}"
    cv2.putText(frame, text, (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)


#################################
# 메인 루프
#################################
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    vx, vy, w = ai_policy(frame)

    dist = get_distance()

    if dist < SAFE_DIST:
        stop()
        action = "STOP (ultra)"
    else:
        drive(vx, vy, w)
        action = f"MOVE {vx},{vy},{w}"

    draw_debug(frame, dist, action)

    cv2.imshow("Raspbot AI Monitor", frame)

    if cv2.waitKey(1) == 27:
        break

stop()
cap.release()
cv2.destroyAllWindows()
