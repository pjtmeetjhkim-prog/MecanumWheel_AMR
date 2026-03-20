from Raspbot_Lib import Raspbot

class Cam_Ctl():
    init_pan_angle: int = 90
    pan_angle: int = 90
    init_tilt_angle: int = 25
    tilt_angle: int = 25

    def __init__(self, bot: Raspbot):
        self.bot = bot
        self.move_pan(25)
        self.move_tilt(90)

    def move_pan(self, angle: int):
        """ 팬 이동 (0~180) """
        if angle < 0 or angle > 180:
            return
        elif self.pan_angle + angle < 0 or self.pan_angle + angle > 180:
            return
        
        self.bot.Ctrl_Servo(1, angle)
        self.pan_angle = angle

    def move_tilt(self, angle: int):
        """ 틸트 이동 (0~110) """
        if angle < 0 or angle > 110:
            return
        elif self.tilt_angle + angle < 0 or self.tilt_angle + angle > 110:
            return
            
        self.bot.Ctrl_Servo(2, angle)
        self.tilt_angle = angle

    def stop(self):
        """ 정지 """
        self.move_pan(25)
        self.move_tilt(90)