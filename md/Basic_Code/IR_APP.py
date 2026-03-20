from Buzzer import Buzzer as BZ
from Cam_PTZ import Cam_Ctl as Cam
from IR_Ctrl import IR_Ctrl as IR
from Key1_event import Key1 as K1
from line_tracking_mng import line_moving_mng as Line
from Moto_Ctrl import Moto_Ctrl as Moto
from RGBBar import RGBBar as RGB
from ultrasound_distance import ultrasound as US
from Raspbot_Lib import Raspbot

class Main():
    bot:Raspbot = None
    cam:Cam = None
    k1:K1 = None
    rgb:RGB = None
    us:US = None
    moto:Moto = None
    bg:BZ = None
    line:Line = None
    ir:IR = None
    
    def __init__(self):
        self.bot = Raspbot()
        
        self.cam = Cam(self.bot)
        self.k1 = K1(self.bot)
        self.rgb = RGB(self.bot)
        self.us = US(self.bot)
        self.moto = Moto(self.bot)
        
        self.bz = BZ(self.bot)
        
        self.line = Line(self.bot, self.moto)
        self.ir = IR(self.bot, self.moto, self.cam, self.bz, self.line, self.rgb)
        
        self.ir.run()
        

if __name__ == "__main__":
    Main()
        
        