from Raspbot_Lib.Raspbot_Lib import Raspbot
from observer_pattern import observer
# import asyncio
import os,time
import threading

class line_tracker(observer):
    bot:Raspbot = None
    task:threading.Thread = None

    def __init__(self, bot:Raspbot):
        super().__init__()
        self.bot = bot

    def __del__(self):
        if self.task and self.task.is_alive():
            self.task = None

    def run(self):
        if self.task and self.task.is_alive():
            return

        self.task = threading.Thread(target=self._task)
        self.task.daemon = True
        self.task.start()

    def _task(self):
        try:
            while True:
                # 라인 순찰 센서의 상태를 읽기
                track_data = self.bot.read_data_array(0x0a, 1)
                track = int(track_data[0])

                # 순찰선 센서의 상태를 분석
                x1 = (track >> 3) & 0x01
                x2 = (track >> 2) & 0x01
                x3 = (track >> 1) & 0x01
                x4 = track & 0x01

                # 차선 유지 센서의 상태를 출력
                print(f"Line Tracker Status: {x2}, {x1}, {x3}, {x4}")
                self._notify_handlers((x1, x2, x3, x4))

                # 잠시 멈춰서 데이터를 다시 읽기
                time.sleep(0.1)
        except KeyboardInterrupt:
            # 사용자가 Ctrl+C를 누르면 프로그램이 종료
            # 화면에 기본 데이터 표시를 복원
            os.system("python3 /home/pi/software/oled_yahboom/yahboom_oled.py &")
            print("Line tracking stopped by user.")