from Raspbot_Lib import Raspbot
from observer_pattern import observer
import time, os
#import asyncio
import threading

class ultrasound(observer):
    bot:Raspbot = None
    task:threading.Thread = None

    def __init__(self, bot: Raspbot):
        super().__init__()
        self.bot = bot
        # 초음파 거리 측정 기능을 활성화

    def __del__(self):
        """ 초음파 거리 측정 기능을 비활성화 """
        if self.task and self.task.is_alive():
            self.task = None

        self.bot.Ctrl_Ulatist_Switch(0)

    def run(self):
        if self.task and self.task.is_alive():
            return

        self.task = threading.Thread(target=self._task)
        self.task.daemon = True
        self.task.start()

    def _task(self):
        try:
            self.bot.Ctrl_Ulatist_Switch(1)
            time.sleep(0.1)  # 초음파 센서가 측정하는 데 시간이 좀 걸릴 수 있습니다
            while True:
                # 초음파 센서에서 측정한 거리
                diss_H =self.bot.read_data_array(0x1b,1)[0]
                diss_L =self.bot.read_data_array(0x1a,1)[0]
                dis = diss_H << 8 | diss_L 
                # 측정거리 출력
                print(f"Ultrasonic Distance: {dis} mm")
                self._notify_handlers(dis)

                time.sleep(0.05)  # 0.05초마다 거리를 측정하세요.

        except KeyboardInterrupt:
            # 프로그램이 종료되면 초음파 거리 측정 기능을 끄기
            self.bot.Ctrl_Ulatist_Switch(0)
            # 화면에 기본 데이터 표시를 복원
            os.system("python3 /home/pi/software/oled_yahboom/yahboom_oled.py &")
            print("Ultrasonic sensor turned off.")