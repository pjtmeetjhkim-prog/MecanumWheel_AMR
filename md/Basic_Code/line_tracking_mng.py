from line_tracking import line_tracker
from Raspbot_Lib import Raspbot
from Moto_Ctrl import Moto_Ctrl
import time

class line_moving_mng(line_tracker):
    moto:Moto_Ctrl = None
    speed = 30

    def __init__(self, bot: Raspbot, moto:Moto_Ctrl):
        super().__init__(bot)
        self.moto = moto

        self.connect(self.move_ctrl)

    def __del__(self):
        self.stop()
        super().__del__()

    def move_ctrl(self, arg):
        x1, x2, x3, x4 = arg
        # if x1 == 0 and x2 == 1:
        #     self.moto.move(-0.75,0.5)
        # elif x1 == 0 and x2 == 0:
        #     self.moto.move(-0.5, 0.5)
        # elif x2 == 0 and x3 == 1:
        #     self.moto.move(-0.25, 0.5)
        # elif x2 == 0 and x3 == 0:
        #     self.moto.move(1, 1)
        # elif x3 == 0 and x4 == 1:
        #     self.moto.move(0.25, 0.5)
        # elif x3 == 0 and x4 == 0:
        #     self.moto.move(0.5, 0.5)
        # elif x4 == 0 and x3 == 1:
        #     self.moto.move(0.75, 0.5)

        """
        X2 X1 X3 X4
        |  |  |  |
        L1 L2 R1 R2
        """
        lineL1=x2
        lineL2=x1
        lineR1=x3
        lineR2=x4

        if lineL1 == 0 and lineL2 == 0 and lineR1 == 0 and lineR2 == 0:  # 모든것이 검은색, 속도 올리기
            print("1")
            print(lineL1,lineL2,lineR1,lineR2)
            self.moto.move_forward(int(self.speed))
        elif( (lineL2 == 0 or lineL1 == 0) and lineR2 == 0):  # 직각: 급격한 오른쪽 굽힘; 0은 검은색 선이 감지
            print("2")
            print(lineL1,lineL2,lineR1,lineR2)
            self.moto.rotate_right(self.speed)
            time.sleep(0.05)
        elif lineL1 == 0 and (lineR2 == 0 or lineR1 == 0):  # 왼쪽 급각 또는 왼쪽 급곡
            print("3")
            print(lineL1,lineL2,lineR1,lineR2) 
            self.moto.rotate_left(int(self.speed*1.5))  # 급격한 좌회전
            time.sleep(0.15)
        elif lineL1 == 0:  # 왼쪽 가장 바깥쪽 감지
            print("4")
            print(lineL1,lineL2,lineR1,lineR2)
            self.moto.rotate_left(self.speed)  # 급격한 좌회전
            time.sleep(0.02)
        elif lineR2 == 0:  # 오른쪽 가장 바깥쪽 감지
            print("5")
            print(lineL1,lineL2,lineR1,lineR2)
            self.moto.rotate_right(self.speed)
            time.sleep(0.01)
        elif lineL2 == 0 and lineR1 == 1:  # 가운데 검은색 선에 있는 센서가 차량의 좌회전을 미세 조정
            print("6")
            print(lineL1,lineL2,lineR1,lineR2)
            self.moto.rotate_left(int(self.speed))  # 좌회전
        elif lineL2 == 1 and lineR1 == 0:  # 가운데 검은색 선에 있는 센서가 차량의 우회전을 미세 조정
            print("7")
            print(lineL1,lineL2,lineR1,lineR2) 
            self.moto.rotate_right(int(self.speed)) # 우회전
        elif lineL2 == 0 and lineR1 == 0:  # 모든것이 검은색, 속도 올리기
            print("8")
            print(lineL1,lineL2,lineR1,lineR2)
            self.moto.move_forward(self.speed)

    def move(self):
        if self.task:
            return
        self.run()

    def stop(self):
        if self.moto:
            self.moto.stop()
        if self.task and self.task.is_alive():
            self.task = None