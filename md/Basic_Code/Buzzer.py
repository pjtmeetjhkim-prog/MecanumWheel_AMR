from Raspbot_Lib import Raspbot

class Buzzer():
    state:int = 0
    def __init__(self, bot: Raspbot):
        self.bot = bot
        self.off()
    def on(self):
        self.bot.Ctrl_BEEP_Switch(1)
        self.state = 1
    def off(self):
        self.bot.Ctrl_BEEP_Switch(0)
        self.state = 0
    def __del__(self):
        self.off()