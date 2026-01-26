#!/usr/bin/python3
# -*- coding: UTF-8 -*-
from Raspbot_Lib import Raspbot
import time,math

class McLumk_Sports():
    bot:Raspbot = None
    # 로봇 객체를 초기화
    def __init__(self, bot:Raspbot, debug:int = 0):
        self.bot = bot
        """
        q w e
        a--丨--d
        z x c
        """
        # 디버그 변수 추가
        self.debug = debug

    def move_forward(self, speed):
        l1, l2, r1, r2 = self.set_deflection(speed, 90)
        if self.debug == 1:
            print(f"L1:{l1:>4}| w |R1:{r1:<4}")
            print(f"L2:{l2:>4}|   |R2:{r2:<4}\n")
        self.bot.Ctrl_Muto(0, l1 + 0)
        self.bot.Ctrl_Muto(1, l2 + 0)
        self.bot.Ctrl_Muto(2, r1 + 0)
        self.bot.Ctrl_Muto(3, r2 + 0)


    def move_param_forward(self, speed, param):
        l1, l2, r1, r2 = self.set_deflection(speed, 90)
        if self.debug == 1:
            print(f"L1:{l1:>4}| w |R1:{r1:<4}")
            print(f"L2:{l2:>4}|   |R2:{r2:<4}\n")
        if param>=0:
            self.bot.Ctrl_Muto(0, l1 + 0)
            self.bot.Ctrl_Muto(1, l2 + 0)
            self.bot.Ctrl_Muto(2, r1 + int((param)*1.2))
            self.bot.Ctrl_Muto(3, r2 + int((param)*1.2))
        elif param<0:
            self.bot.Ctrl_Muto(0, l1 + int(abs((param)*1.2)))
            self.bot.Ctrl_Muto(1, l2 + int(abs((param)*1.2)))
            self.bot.Ctrl_Muto(2, r1)
            self.bot.Ctrl_Muto(3, r2)  


    def move_backward(self, speed):
        l1, l2, r1, r2 = self.set_deflection(speed, 270)
        if self.debug == 1:
            print(f"L1:{l1:>4}| x |R1:{r1:<4}")
            print(f"L2:{l2:>4}|   |R2:{r2:<4}\n")
        self.bot.Ctrl_Muto(0, l1 + 0)
        self.bot.Ctrl_Muto(1, l2 + 0)
        self.bot.Ctrl_Muto(2, r1 + 0)
        self.bot.Ctrl_Muto(3, r2 + 0)

    def move_left(self, speed):
        l1, l2, r1, r2 = self.set_deflection(speed, 180)
        if self.debug == 1:
            print(f"L1:{l1:>4}| a |R1:{r1:<4}")
            print(f"L2:{l2:>4}|   |R2:{r2:<4}\n")
        self.bot.Ctrl_Muto(0, l1 + 0)
        self.bot.Ctrl_Muto(1, l2 + 0)
        self.bot.Ctrl_Muto(2, r1 + 0)
        self.bot.Ctrl_Muto(3, r2 + 0)

    def move_right(self, speed):
        l1, l2, r1, r2 = self.set_deflection(speed, 0)
        if self.debug == 1:
            print(f"L1:{l1:>4}| d |R1:{r1:<4}")
            print(f"L2:{l2:>4}|   |R2:{r2:<4}\n")
        self.bot.Ctrl_Muto(0, l1 + 0)
        self.bot.Ctrl_Muto(1, l2 + 0)
        self.bot.Ctrl_Muto(2, r1 + 0)
        self.bot.Ctrl_Muto(3, r2 + 0)

    def rotate_left(self, speed):
        l1, l2, r1, r2 = self.set_deflection(speed, 180)
        if self.debug == 1:
            print(f"L1:{l1:>4}| q |R1:{r1:<4}")
            print(f"L2:{-l2:>4}|   |R2:{abs(r2):<4}\n")
        self.bot.Ctrl_Muto(0, l1 + 0)
        self.bot.Ctrl_Muto(1, -l2 + 0)
        self.bot.Ctrl_Muto(2, r1 + 0)
        self.bot.Ctrl_Muto(3, abs(r2) + 0)

    def rotate_right(self, speed):
        l1, l2, r1, r2 = self.set_deflection(speed, 0)
        if self.debug == 1:
            print(f"L1:{l1:>4}| e |R1:{r1:<4}")
            print(f"L2:{abs(l2):>4}|   |R2:{-r2:<4}\n")
        self.bot.Ctrl_Muto(0, l1 + 0)
        self.bot.Ctrl_Muto(1, abs(l2) + 0)
        self.bot.Ctrl_Muto(2, r1 + 0)
        self.bot.Ctrl_Muto(3, -r2 + 0)

    def move_diagonal_left_front(self, speed):
        l1, l2, r1, r2 = self.set_deflection(speed, 135)
        if self.debug == 1:
            print(f"L1:{l1:>4}| q |R1:{r1:<4}")
            print(f"L2:{l2:>4}|   |R2:{r2:<4}\n")
        self.bot.Ctrl_Muto(0, l1 + 0)
        self.bot.Ctrl_Muto(1, l2 + 0)
        self.bot.Ctrl_Muto(2, r1 + 0)
        self.bot.Ctrl_Muto(3, r2 + 0)

    def move_diagonal_left_back(self, speed):
        l1, l2, r1, r2 = self.set_deflection(speed, 225)
        if self.debug == 1:
            print(f"L1:{l1:>4}| z |R1:{r1:<4}")
            print(f"L2:{l2:>4}|   |R2:{r2:<4}\n")
        self.bot.Ctrl_Muto(0, l1 + 0)
        self.bot.Ctrl_Muto(1, l2 + 0)
        self.bot.Ctrl_Muto(2, r1 + 0)
        self.bot.Ctrl_Muto(3, r2 + 0)

    def move_diagonal_right_front(self, speed):
        l1, l2, r1, r2 = self.set_deflection(speed, 45)
        if self.debug == 1:
            print(f"L1:{l1:>4}| e |R1:{r1:<4}")
            print(f"L2:{l2:>4}|   |R2:{r2:<4}\n")
        self.bot.Ctrl_Muto(0, l1 + 0)
        self.bot.Ctrl_Muto(1, l2 + 0)
        self.bot.Ctrl_Muto(2, r1 + 0)
        self.bot.Ctrl_Muto(3, r2 + 0)

    def move_diagonal_right_back(self, speed):
        l1, l2, r1, r2 = self.set_deflection(speed, 315)
        if self.debug == 1:
            print(f"L1:={l1:>4}| c |R1:={r1:<4}")
            print(f"L2:={l2:>4}|   |R2:={r2:<4}\n")
        self.bot.Ctrl_Muto(0, l1 + 0)
        self.bot.Ctrl_Muto(1, l2 + 0)
        self.bot.Ctrl_Muto(2, r1 + 0)
        self.bot.Ctrl_Muto(3, r2 + 0)

    def stop_robot(self):
        self.bot.Ctrl_Car(0, 0, 0)
        self.bot.Ctrl_Car(1, 0, 0)
        self.bot.Ctrl_Car(2, 0, 0)
        self.bot.Ctrl_Car(3, 0, 0)

    def stop(self):
        for i in range(4):
            #print(i)
            time.sleep(0.25)
            self.bot.Ctrl_Car(0, 0, 0)
            self.bot.Ctrl_Car(1, 0, 0)
            self.bot.Ctrl_Car(2, 0, 0)
            self.bot.Ctrl_Car(3, 0, 0)

    def set_deflection(speed, deflection):
        """
            90
        180--丨--0
            270
        """
        if(speed>255):speed=255
        if(speed<0):speed=0
        rad2deg = math.pi / 180
        vx = speed * math.cos(deflection * rad2deg)
        vy = speed * math.sin(deflection * rad2deg)
        l1 = int(vy + vx) 
        l2 = int(vy - vx)
        r1 = int(vy - vx)
        r2 = int(vy + vx)
        return l1,l2,r1,r2

    def set_deflection_rate(speed, deflection,rate):
        """
            90
        180--丨--0
            270
        """
        if(speed>255):speed=255
        if(speed<0):speed=0
        rad2deg = math.pi / 180
        vx = speed * math.cos(deflection * rad2deg)
        vy = speed * math.sin(deflection * rad2deg)
        vp = -rate * (117+ 132)/2
        l1 = int(vy + vx - vp) 
        l2 = int(vy - vx + vp)
        r1 = int(vy - vx - vp)
        r2 = int(vy + vx + vp)
        return l1,l2,r1,r2

    def drifting(self, speed,deflection,rate):
        '''
        180--丨--0
        '''
        l1,l2,r1,r2=self.set_deflection_rate(speed,deflection,rate)
        self.bot.Ctrl_Muto(0, l1+ 0)
        self.bot.Ctrl_Muto(1, l2+ 0)
        self.bot.Ctrl_Muto(2, r1+ 0)
        self.bot.Ctrl_Muto(3, r2+ 0)