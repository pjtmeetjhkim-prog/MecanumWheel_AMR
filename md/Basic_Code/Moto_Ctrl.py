from enum import IntEnum
from Raspbot_Lib import Raspbot
from typing import Dict
from McLumk_Wheel_Sports import McLumk_Sports

class Moto_Num(IntEnum):
    LFront = 0
    LBack = 1
    RFront = 2
    RBack = 3

class Moto_Ctrl(McLumk_Sports):
    x:float = 0
    y:float = 0
    
    def __init__(self, bot: Raspbot):
        super().__init__(bot)

    def correction(self, _x:float, _y:float):
        if self.x == _x and self.y == _y:
            return None
        
        if _x > 1:
            _x = 1
        elif _x < -1:
            _x = -1
        if _y > 1:
            _y = 1
        elif _y < -1:
            _y = -1
        return _x, _y

    # def move(self, _x:float, _y:float):
        x, y = self.correction(_x, _y)
        if x is None or y is None:
            return
        self.x = int(x*255)
        self.y = int(y*255)

        #전진이동
        if self.y > 0:
            #전진이동인데 북동쪽보다 작은 각도로 우측이동
            if self.x < 0.5 and self.x > 0:
                self.bot.Ctrl_Car(Moto_Num.LFront, 0, self.x)
                self.bot.Ctrl_Car(Moto_Num.RBack, 0, self.x)

                self.bot.Ctrl_Car(Moto_Num.LBack, 0, self.x)
                self.bot.Ctrl_Car(Moto_Num.RFront, 0, self.x)


            #우측이동인데 북동이동보다 큰 각도로 우측이동
            elif self.x > 0.5:
                self.bot.Ctrl_Car(Moto_Num.LFront, 0, self.x)
                self.bot.Ctrl_Car(Moto_Num.RBack, 0, self.x)

                self.bot.Ctrl_Car(Moto_Num.LBack, 0, -self.x)
                self.bot.Ctrl_Car(Moto_Num.RFront, 0, -self.x)


            #죄측이동인데 북서이동보다 큰각도로 좌측이동
            elif self.x < -0.5:
                self.bot.Ctrl_Car(Moto_Num.LFront, 0, -self.x)
                self.bot.Ctrl_Car(Moto_Num.RBack, 0, -self.x)

                self.bot.Ctrl_Car(Moto_Num.LBack, 0, self.x)
                self.bot.Ctrl_Car(Moto_Num.RFront, 0, self.x)


            #전진이동인데 북서쪽보다 작은 각도로 좌측이동
            elif self.x > -0.5 and self.x < 0:
                self.bot.Ctrl_Car(Moto_Num.LFront, 0, self.x)
                self.bot.Ctrl_Car(Moto_Num.RBack, 0, self.x)

                self.bot.Ctrl_Car(Moto_Num.LBack, 0, self.x)
                self.bot.Ctrl_Car(Moto_Num.RFront, 0, self.x)


            #북동이동
            elif self.x == 0.5:
                self.bot.Ctrl_Car(Moto_Num.RFront, 0, 0)
                self.bot.Ctrl_Car(Moto_Num.LBack, 0, 0)
            #북서이동
            elif self.x == -0.5:
                self.bot.Ctrl_Car(Moto_Num.LFront, 0, 0)
                self.bot.Ctrl_Car(Moto_Num.RBack, 0, 0)











        elif self.y < 0:
            if self.x > 0:
                self.bot.Ctrl_Car(Moto_Num.LBack, 0, self.y)
                self.bot.Ctrl_Car(Moto_Num.RFront, 0, self.y)
            elif self.x < 0:
                self.bot.Ctrl_Car(Moto_Num.LFront, 0, self.y)
                self.bot.Ctrl_Car(Moto_Num.RBack, 0, self.y)
                self.bot.Ctrl_Car(Moto_Num.RFront, 0, self.y)
                self.bot.Ctrl_Car(Moto_Num.RBack, 0, self.y)
        elif self.x > 0:
            self.bot.Ctrl_Car(Moto_Num.LFront, 0, self.x)
            self.bot.Ctrl_Car(Moto_Num.RBack, 0, self.x)
            self.bot.Ctrl_Car(Moto_Num.RFront, 0, self.x)
            self.bot.Ctrl_Car(Moto_Num.RBack, 0, self.x)
        elif self.x < 0:
            self.bot.Ctrl_Car(Moto_Num.LFront, 0, self.x)
            self.bot.Ctrl_Car(Moto_Num.RBack, 0, self.x)
            self.bot.Ctrl_Car(Moto_Num.RFront, 0, self.x)
            self.bot.Ctrl_Car(Moto_Num.RBack, 0, self.x)
        else:
            self.bot.Ctrl_Car(Moto_Num.LFront, 0, 0)
            self.bot.Ctrl_Car(Moto_Num.LBack, 0, 0)
            self.bot.Ctrl_Car(Moto_Num.RFront, 0, 0)
            self.bot.Ctrl_Car(Moto_Num.RBack, 0, 0)
        











        # self.bot.Ctrl_Car(Moto_Num.LFront, 0, x)
        # self.bot.Ctrl_Car(Moto_Num.LBack, 0, y)
        # self.bot.Ctrl_Car(Moto_Num.RFront, 0, x)
        # self.bot.Ctrl_Car(Moto_Num.RBack, 0, y)
    
    def _map_speed_to_255(self, speed_f: float) -> int:
        """[-1.0, 1.0] -> [0, 255] 매핑. 0.0은 정지(약 127)에 해당합니다."""
        # speed_255 = (speed_f + 1.0) * 127.5
        # 0.5를 더하고 정수화하여 반올림 효과를 줍니다.
        speed_255 = int((speed_f + 1.0) * 127.5 + 0.5)
        
        # 안전을 위해 0-255 범위로 클램프(제한)
        return max(0, min(255, speed_255))

    def _set_motor_speed(self, motor_id: int, speed: int):
        """실제 모터에 명령을 내리는 가상의 함수. 중복 명령을 무시합니다."""
        global LAST_SPEEDS
        motor_id -= 1
        if LAST_SPEEDS is not set:
            LAST_SPEEDS = [0,0,0,0]
        
        # *** 핵심 로직: 동일한 명령인 경우 무시 ***
        if LAST_SPEEDS[motor_id] != speed:
            print(f"✅ Motor {motor_id} 명령 전송: {speed}")
            LAST_SPEEDS[motor_id] = speed
        else:
            print(f"❌ Motor {motor_id} 명령 무시 (이전과 동일: {speed})")

    def move(self, x: float, y: float):
        """
        X, Y 입력을 받아 모터 속도를 계산하고 명령을 전송합니다.
        
        Args:
            x (float): 회전/좌우 움직임 제어. (-1.0: 좌측, 1.0: 우측)
            y (float): 전진/후진 제어. (-1.0: 후진, 1.0: 전진)
        """
        # 1. 입력 값 클램프 ([-1.0, 1.0] 범위 보장)
        x = max(-1.0, min(1.0, x))
        y = max(-1.0, min(1.0, y))

        # 2. 차동 구동 로직으로 좌/우 모터의 속도 (범위 [-1.0, 1.0]) 계산
        # 이 로직은 제공하신 8가지 예시 중 4가지 (전진, 후진, 좌이동(회전), 우이동(회전))에 해당합니다.
        
        # 왼쪽 모터 (M1, M2) 속도: 전진(y) - 회전(x)
        # x가 양수(우회전)면 왼쪽 모터는 느려지거나 역방향
        motor_L_speed_f = y - x
        
        # 오른쪽 모터 (M3, M4) 속도: 전진(y) + 회전(x)
        # x가 양수(우회전)면 오른쪽 모터는 빨라지거나 정방향
        motor_R_speed_f = y + x
        
        # 합산 결과가 [-1.0, 1.0] 범위를 초과할 수 있으므로 다시 클램프
        motor_L_speed_f = max(-1.0, min(1.0, motor_L_speed_f))
        motor_R_speed_f = max(-1.0, min(1.0, motor_R_speed_f))

        # 3. 목표 모터 속도 (범위 [0, 255]) 계산
        target_speeds: Dict[int, int] = {
            1: self._map_speed_to_255(motor_L_speed_f),  # M1 (왼쪽 앞)
            2: self._map_speed_to_255(motor_L_speed_f),  # M2 (왼쪽 뒤)
            3: self._map_speed_to_255(motor_R_speed_f),  # M3 (오른쪽 앞)
            4: self._map_speed_to_255(motor_R_speed_f)   # M4 (오른쪽 뒤)
        }
        
        # 4. 모터 명령 전송 (M1, M3), (M2, M4) 쌍으로 전송
        print(f"\n--- move(x={x:.2f}, y={y:.2f}) 명령 처리 시작 ---")
        
        # M1, M3 (앞 모터)
        print("--- [M1, M3] 명령 ---")
        self._set_motor_speed(1, target_speeds[1])
        self._set_motor_speed(3, target_speeds[3])
        
        # M2, M4 (뒤 모터)
        print("--- [M2, M4] 명령 ---")
        self._set_motor_speed(2, target_speeds[2])
        self._set_motor_speed(4, target_speeds[4])
        
        print("-------------------------------------------------")

    def clockwise_turning(self, power:int = 255):
        self.bot.Ctrl_Car(Moto_Num.LFront, 0, power)
        self.bot.Ctrl_Car(Moto_Num.LBack, 0, power)
        self.bot.Ctrl_Car(Moto_Num.RFront, 0, -power)
        self.bot.Ctrl_Car(Moto_Num.RBack, 0, -power)
    def counterclockwise_turning(self, power:int):
        self.bot.Ctrl_Car(Moto_Num.LFront, 0, -power)
        self.bot.Ctrl_Car(Moto_Num.LBack, 0, -power)
        self.bot.Ctrl_Car(Moto_Num.RFront, 0, power)
        self.bot.Ctrl_Car(Moto_Num.RBack, 0, power)
    def stop(self):
        self.bot.Ctrl_Car(Moto_Num.LFront, 0, 0)
        self.bot.Ctrl_Car(Moto_Num.LBack, 0, 0)
        self.bot.Ctrl_Car(Moto_Num.RFront, 0, 0)
        self.bot.Ctrl_Car(Moto_Num.RBack, 0, 0)
