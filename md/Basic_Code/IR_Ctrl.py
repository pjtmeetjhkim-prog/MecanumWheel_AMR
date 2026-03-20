from Raspbot_Lib import Raspbot, LightShow
from observer_pattern import observer
from Moto_Ctrl import Moto_Ctrl, Moto_Num
from Buzzer import Buzzer
from Cam_PTZ import Cam_Ctl
from RGBBar import RGBBar, RGBColor
from line_tracking_mng import line_moving_mng
import asyncio
import os, time
import threading

class IR_Ctrl(observer):
    task = None
    bot:Raspbot = None
    moto:Moto_Ctrl = None
    cam:Cam_Ctl = None
    buz:Buzzer = None
    line:line_moving_mng = None
    power:int = 255
    rgb:RGBBar = None

    def __init__(self, bot:Raspbot, moto:Moto_Ctrl, cam:Cam_Ctl = None, buz:Buzzer = None, line:line_moving_mng = None, rgb:RGBBar = None):
        super().__init__()
        self.bot = bot
        self.moto = moto
        if cam:
            self.cam = cam
        if buz:
            self.buz = buz
        if line:
            self.line = line
        if rgb:
            self.rgb = rgb

    def __del__(self):
        if self.task.is_alive():
            # self.task.cancel()
            self.task = None
        self.bot.Ctrl_IR_Switch(0)

    def run(self):
        if self.task:
            return
        self.task = threading.Thread(target=self._task)
        self.task.daemon = True
        self.task.run()

    def receive_data(self, data:str):
        # 초기화
        num = int(data)
        
        if num == 0:
            self.moto.stop()
            if self.buz:
                self.buz.off()
            if self.cam:
                self.cam.move_pan(self.cam.init_pan_angle)
                self.cam.move_tilt(self.cam.init_tilt_angle)
            if self.rgb:
                self.rgb.set_color(RGBColor.OFF)
        # 전진
        elif num == 1:
            self.bot.Ctrl_Car(Moto_Num.LFront, -1, self.power)
            self.bot.Ctrl_Car(Moto_Num.LBack, -1, self.power)
            self.bot.Ctrl_Car(Moto_Num.RFront, -1, self.power)
            self.bot.Ctrl_Car(Moto_Num.RBack, -1, self.power)
        # 후진
        elif num == 9:
            self.bot.Ctrl_Car(Moto_Num.LFront, 1, self.power)
            self.bot.Ctrl_Car(Moto_Num.LBack, 1, self.power)
            self.bot.Ctrl_Car(Moto_Num.RFront, 1, self.power)
            self.bot.Ctrl_Car(Moto_Num.RBack, 1, self.power)
        # 좌측이동
        elif num == 4:
            self.bot.Ctrl_Car(Moto_Num.LFront, 1, self.power)
            self.bot.Ctrl_Car(Moto_Num.LBack, -1, self.power)
            self.bot.Ctrl_Car(Moto_Num.RFront, -1, self.power)
            self.bot.Ctrl_Car(Moto_Num.RBack, 1, self.power)
        # 우측이동
        elif num == 6:
            self.bot.Ctrl_Car(Moto_Num.LFront, -1, self.power)
            self.bot.Ctrl_Car(Moto_Num.LBack, 1, self.power)
            self.bot.Ctrl_Car(Moto_Num.RFront, 1, self.power)
            self.bot.Ctrl_Car(Moto_Num.RBack, -1, self.power)
        # 반시계방향 선회
        elif num == 8:
            self.bot.Ctrl_Car(Moto_Num.LFront, -1, self.power)
            self.bot.Ctrl_Car(Moto_Num.LBack, -1, self.power)
            self.bot.Ctrl_Car(Moto_Num.RFront, 1, self.power)
            self.bot.Ctrl_Car(Moto_Num.RBack, 1, self.power)
        # 시계방향 선회
        elif num == 10:
            self.bot.Ctrl_Car(Moto_Num.LFront, 1, self.power)
            self.bot.Ctrl_Car(Moto_Num.LBack, 1, self.power)
            self.bot.Ctrl_Car(Moto_Num.RFront, -1, self.power)
            self.bot.Ctrl_Car(Moto_Num.RBack, -1, self.power)
        # 모터 파워 상승
        if num == 12:
            self.power += 1
            if self.line:
                self.line.speed = self.power
        # 모터 파워 하강
        if num == 14:
            self.power -= 1
            if self.line:
                self.line.speed = self.power


        # 부저
        if num == 5 and self.buz:
            if self.buz.state == 0:
                self.buz.on()
            else:
                self.buz.off()


        # 라인
        if num == 2 and self.line:
            if not self.line.task:
                self.line.run()
            else:
                self.line.stop()


        # 캠 상승
        if num == 13 and self.cam:
            angle = self.cam.tilt_angle
            angle += 1
            self.cam.move_tilt(angle)
        # 캠 하강
        elif num == 17 and self.cam:
            angle = self.cam.tilt_angle
            angle -= 1
            self.cam.move_tilt(angle)
        # 캠 좌측이동
        elif num == 16 and self.cam:
            angle = self.cam.pan_angle
            angle += 1
            self.cam.move_pan(angle)
        # 캠 우측이동
        elif num == 18 and self.cam:
            angle = self.cam.pan_angle
            angle -= 1
            self.cam.move_pan(angle)
        
        # 20 21 22
        elif num == 20 and self.rgb:
            self.rgb.set_color(RGBColor.RED)
        elif num == 21 and self.rgb:
            self.rgb.set_color(RGBColor.GREEN)
        elif num == 22 and self.rgb:
            self.rgb.set_color(RGBColor.BLUE)
        # 24 25 26
        elif num == 24 and self.rgb:
            self.rgb.light.execute_effect('gradient', 3, 0.01, RGBColor.YELLOW)
        elif num == 25 and self.rgb:
            self.rgb.light.execute_effect('random_running', 3, 0.01, RGBColor.PURPLE)
        elif num == 26 and self.rgb:
            self.rgb.light.execute_effect('starlight', 3, 0.01, RGBColor.CYAN)

    def _task(self):
        # 적외선 리모컨 수신을 켜기
        self.bot.Ctrl_IR_Switch(1)
        try:
            while True:
                # 적외선 리모컨의 가격을 읽기
                data = self.bot.read_data_array(0x0c, 1)
                data2h=hex(data[0])
                # 수신된 적외선 데이터 출력
                if(data[0]<30):
                    print("Received IR data:", data,data2h)
                    self.receive_data(data[0])
                else:
                    print("Received IR not data:", data,data2h)
                    if self.line is None or self.line.task is None or self.line.task.is_alive() is False:
                        self.moto.stop()
                # 출력이 너무 빨리 되는 것을 방지하기 위해 짧은 지연 시간을 추가
                time.sleep(0.2)  # 0.2초마다 데이터를 읽기

        except KeyboardInterrupt:
            # 프로그램이 끝나면 적외선 리모컨 수신기를 끄기기
            self.bot.Ctrl_IR_Switch(0)
            print("IR receiver turned off.")

