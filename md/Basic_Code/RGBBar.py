from enum import IntEnum
from Raspbot_Lib import Raspbot, LightShow

class RGBColor(IntEnum):
    OFF = -1
    RED = 0
    GREEN = 1
    BLUE = 2
    YELLOW = 3
    PURPLE = 4
    CYAN = 5
    WHITE = 6

class RGBBar():
    def __init__(self, bot: Raspbot):
        self.bot = bot
        self.light = LightShow()
    def set_color(self, color: RGBColor):
        if color == RGBColor.OFF:
            self.light.turn_off_all_lights()
        else:
            self.bot.Ctrl_WQ2812_ALL(1, color)
    def set_rgb(self, R:int, G:int, B:int):
        if R > 255:
            R = 255
        if R < 0:
            R = 0
        if G > 255:
            G = 255
        if G < 0:
            G = 0
        if B > 255:
            B = 255
        if B < 0:
            B = 0
        self.bot.Ctrl_WQ2812_brightness_ALL(1, (R, G, B))

    def __del__(self):
        self.light.turn_off_all_lights()
        