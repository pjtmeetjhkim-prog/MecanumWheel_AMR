from Raspbot_Lib import Raspbot
from observer_pattern import observer
import time,os
# import asyncio
import threading

class Key1(observer):
    bot:Raspbot = None
    task:threading.Thread = None
    
    def __init__(self, bot: Raspbot, is_run:bool = False):
        super().__init__()
        self.bot = bot
        self.task = None

        # 옵저버 패턴
        self._handlers = []

        if is_run:
            self.run()

    def __del__(self):
        """작업중일경우 중지하고 삭제"""
        if self.task and self.task.is_alive():
            # self.task.cancel()
            self.task = None

    def run(self):
        """ 작업시작 """
        if self.task:
            return

        self.task = threading.Thread(target=self._task)
        self.task.start()

    def _task(self):
        """ 비동기 데이터 읽기 """
        key_down= False
        try:
            while True:
                data = self.bot.read_data_array(0x0d, 1)
                state=data[0]
                # 버튼이 눌린 상태에서 눌린 상태로 바뀔 때만 인쇄가 발생합니다.
                # 키가 눌리지 않았을 때와 눌렸을 때만 인쇄합니다.
                if state == 1 and not key_down:
                    print("key pressed", state)
                    key_down = True  
                    # key_str=f'{state:>10}'
                    self._notify_handlers(True)
            
                if state == 0 and key_down:
                    print("key released", state)
                    key_down = False
                    # key_str=f'{state:>10}'
                    self._notify_handlers(False)
        
                # 너무 빠른 출력을 방지하기 위해 짧은 지연 시간을 추가할 수 있습니다.
                # 너무 빨리 출력되는 것을 방지하려면 짧은 지연 시간을 추가하세요.
                time.sleep(0.05)  # 데이터는 0.05초마다 읽힙니다.
        except KeyboardInterrupt:
                # 화면에 기본 데이터 표시를 복원합니다.
                os.system("python3 /home/pi/software/oled_yahboom/yahboom_oled.py &")