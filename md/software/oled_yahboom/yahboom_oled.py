#!/usr/bin/env python3
# coding=utf-8
import time
import os

import Adafruit_SSD1306 as SSD

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import smbus2

import subprocess

# V1.0.1
class Yahboom_OLED:
    def __init__(self, i2c_bus=1, debug=False):
        self.__debug = debug
        self.__i2c_bus = i2c_bus
        self.__top = -2
        self.__x = 0

        self.__total_last = 0
        self.__idle_last = 0
        self.__str_CPU = "CPU:0%"

        self.h = 0
        self.m = 0
        self.s = 0

    def __del__(self):
        if self.__debug:
            print("---OLED-DEL---")

    # OLED를 초기화합니다. 성공 시: True, 실패 시: False
    # Initialize OLED, return True on success, False on failure
    def begin(self):
        try:
            self.__oled = SSD.SSD1306_128_32(
                rst=None, i2c_bus=self.__i2c_bus, gpio=1)
            self.__oled.begin()
            self.__oled.clear()
            self.__oled.display()
            self.__width = self.__oled.width
            self.__height = self.__oled.height
            self.__image = Image.new('1', (self.__width, self.__height))
            self.__draw = ImageDraw.Draw(self.__image)
            self.__font = ImageFont.load_default()
            if self.__debug:
                print("---OLED begin ok!---")
            return True
        except:
            if self.__debug:
                print("---OLED no found!---")
            return False

    # 화면을 지웁니다. refresh=True 즉시 새로 고침, refresh=False 새로 고침 없음.
    # Clear the display.  Refresh =True Refresh immediately, refresh=False refresh not
    def clear(self, refresh=False):
        self.__draw.rectangle(
            (0, 0, self.__width, self.__height), outline=0, fill=0)
        if refresh:
            self.refresh()

    # 문자를 추가합니다. start_x start_y는 시작 지점을 나타냅니다. text는 추가할 문자입니다.
    # refresh=True立即刷新，refresh=False不刷新。
    # Add characters.  Start_x Start_y indicates the starting point.  Text is the character to be added
    # Refresh =True Refresh immediately, refresh=False refresh not
    def add_text(self, start_x, start_y, text, refresh=False):
        if start_x > 128 or start_x < 0 or start_y < 0 or start_y > 32:
            if self.__debug:
                print("oled text: x, y input error!")
            return
        x = int(start_x + self.__x)
        y = int(start_y + self.__top)
        self.__draw.text((x, y), str(text), font=self.__font, fill=255)
        if refresh:
            self.refresh()

    def add_cntext(self, start_x, start_y, text, refresh=False):
        if start_x > 128 or start_x < 0 or start_y < 0 or start_y > 32:
            if self.__debug:
                print("oled text: x, y input error!")
            return
        x = int(start_x + self.__x)
        y = int(start_y + self.__top)
        self.__draw.text((x, y), str(text), font=ImageFont.truetype("platech.ttf",12), fill=255)
        if refresh:
            self.refresh()

    # 한 줄의 문자 텍스트를 입력합니다. refresh=True 즉시 새로 고침, refresh=False 새로 고침 없음.
    # line=[1, 4]
    # Write a line of character text.  Refresh =True Refresh immediately, refresh=False refresh not.
    def add_line(self, text, line=1, refresh=False):
        if line < 1 or line > 4:
            if self.__debug:
                print("oled line input error!")
            return
        y = int(8 * (line - 1))
        self.add_text(0, y, text, refresh)

    def add_cnline(self, text, line=1, refresh=False):
        if line < 1 or line > 4:
            if self.__debug:
                print("oled line input error!")
            return
        y = int(8 * (line - 1))
        self.add_cntext(0, y, text, refresh)

    # 콘텐츠를 표시하려면 OLED 화면을 새로 고치세요
    # Refresh the OLED to display the content
    def refresh(self):
        self.__oled.image(self.__image)
        self.__oled.display()

    # CPU 사용량 읽기
    # Read the CPU usage rate
    def getCPULoadRate(self, index):
        count = 10
        if index == 0:
            f1 = os.popen("cat /proc/stat", 'r')
            stat1 = f1.readline()
            data_1 = []
            for i in range(count):
                data_1.append(int(stat1.split(' ')[i+2]))
            self.__total_last = data_1[0]+data_1[1]+data_1[2]+data_1[3] + \
                data_1[4]+data_1[5]+data_1[6]+data_1[7]+data_1[8]+data_1[9]
            self.__idle_last = data_1[3]
        elif index == 4:
            f2 = os.popen("cat /proc/stat", 'r')
            stat2 = f2.readline()
            data_2 = []
            for i in range(count):
                data_2.append(int(stat2.split(' ')[i+2]))
            total_now = data_2[0]+data_2[1]+data_2[2]+data_2[3] + \
                data_2[4]+data_2[5]+data_2[6]+data_2[7]+data_2[8]+data_2[9]
            idle_now = data_2[3]
            total = int(total_now - self.__total_last)
            idle = int(idle_now - self.__idle_last)
            usage = int(total - idle)
            usageRate = int(float(usage / total) * 100)
            self.__str_CPU = "CPU:" + str(usageRate) + "%"
            self.__total_last = 0
            self.__idle_last = 0
            # if self.__debug:
            #     print(self.__str_CPU)
        return self.__str_CPU

    # 시스템 시간 읽기
    # Read system time
    def getSystemTime(self):
        self.s += .1
        if self.s >= 60:
            self.s = 0
            self.m += 1
        if self.m >= 60:
            self.m = 0
            self.h += 1

        str_time = f"{self.h:02d}:{self.m:02d}:{int(self.s):02d}"

        # cmd = "date +%H:%M:%S"
        # date_time = subprocess.check_output(cmd, shell=True)
        # str_Time = str(date_time).lstrip('b\'')
        # str_Time = str_Time.rstrip('\\n\'')
        # print(date_time)
        return str_time

    # 메모리 사용량과 총 메모리 용량을 읽어보세요.
    # Read the memory usage and total memory
    def getUsagedRAM(self):
        cmd = "free | awk 'NR==2{printf \"RAM:%2d%% -> %.1fGB \", 100*($2-$7)/$2, ($2/1048576.0)}'"
        FreeRam = subprocess.check_output(cmd, shell=True)
        str_FreeRam = str(FreeRam).lstrip('b\'')
        str_FreeRam = str_FreeRam.rstrip('\'')
        return str_FreeRam

    # 읽기 전용 메모리 / 전체 메모리
    # Read free memory/total memory
    def getFreeRAM(self):
        cmd = "free -h | awk 'NR==2{printf \"RAM: %.1f/%.1fGB \", $7,$2}'"
        FreeRam = subprocess.check_output(cmd, shell=True)
        str_FreeRam = str(FreeRam).lstrip('b\'')
        str_FreeRam = str_FreeRam.rstrip('\'')
        return str_FreeRam

    # TF 카드 공간 사용량 / TF 카드 총 공간 읽기
    # Read the TF card space usage/TOTAL TF card space
    def getUsagedDisk(self):
        cmd = "df -h | awk '$NF==\"/\"{printf \"SDC:%s -> %.1fGB\", $5, $2}'"
        Disk = subprocess.check_output(cmd, shell=True)
        str_Disk = str(Disk).lstrip('b\'')
        str_Disk = str_Disk.rstrip('\'')
        return str_Disk

    # TF 카드 사용 가능 공간 / 총 TF 카드 공간
    # Read the free TF card space/total TF card space
    def getFreeDisk(self):
        cmd = "df -h | awk '$NF==\"/\"{printf \"Disk:%.1f/%.1fGB\", $4,$2}'"
        Disk = subprocess.check_output(cmd, shell=True)
        str_Disk = str(Disk).lstrip('b\'')
        str_Disk = str_Disk.rstrip('\'')
        return str_Disk

    # 로컬 IP 얻기
    # Read the local IP address
    def getLocalIP(self):
        ip = os.popen(
            "/sbin/ifconfig eth0 | grep 'inet' | awk '{print $2}'").read()
        ip = ip[0: ip.find('\n')]
        if(ip == ''):
            ip = os.popen(
                "/sbin/ifconfig wlan0 | grep 'inet' | awk '{print $2}'").read()
            ip = ip[0: ip.find('\n')]
            if(ip == ''):
                ip = 'x.x.x.x'
        # if len(ip) > 15:
        #     ip = 'x.x.x.x'
        return ip

    # OLED 메인 함수는 while 루프 내에서 호출되므로 핫플러그 기능을 사용할 수 있습니다.
    # Oled mainly runs functions that are called in a while loop and can be hot-pluggable
    def main_program(self):
        state = False
        try:
            cpu_index = 0
            state = self.begin()
            while state:
                self.clear()
                str_CPU = self.getCPULoadRate(cpu_index)
                str_Time = self.getSystemTime()
                if cpu_index == 0:
                    str_FreeRAM = self.getUsagedRAM()
                    # str_Disk = self.getUsagedDisk()
                    v = self.read_voltage()
                    str_btr = f"BTR: {v:.2f}V / {self.get_percentage(v):.1f}%"
                    str_IP = "IPA:" + self.getLocalIP()
                self.add_text(0, 0, str_CPU)
                self.add_text(50, 0, str_Time)
                self.add_line(str_FreeRAM, 2)
                # self.add_line(str_Disk, 3)
                self.add_line(str_btr, 3)
                self.add_line(str_IP, 4)
                # Display image.
                self.refresh()
                cpu_index = cpu_index + 1
                if cpu_index >= 5:
                    cpu_index = 0
                time.sleep(.1)
        except:
            if self.__debug:
                print("!!!---OLED refresh error---!!!")
            pass
    
    def init_oled_process(self):
        try:
            # OLED를 초기화합니다
            self.begin()
            # pgrep을 사용하여 스크립트를 찾으세요
            result = subprocess.run(['pgrep', '-f', '/home/pi/software/oled_yahboom/yahboom_oled.py'], capture_output=True, text=True, check=True)
            pids = result.stdout.strip().split('\n')
            # 발견된 모든 PID를 순회합니다.
            for pid in pids:
                try:
                    # 프로세스 종료
                    subprocess.run(['kill', pid], check=True)
                    print(f"Process {pid} has been terminated.")
                except subprocess.CalledProcessError:
                    print(f"Failed to terminate process {pid}.")
        except subprocess.CalledProcessError:
            print("No matching processes found.")

    def read_voltage(self):
        bus = smbus2.SMBus(1)
        address = 0x40
        REG_BUS_VOLTAGE = 0x02
        # 1. 레지스터에서 2바이트 데이터 읽기
        read_data = bus.read_word_data(address, REG_BUS_VOLTAGE)
        
        # 2. 바이트 순서 교정 (Big-endian 변환)
        # smbus2는 Little-endian으로 읽어오므로 상하위 바이트를 바꿔야 합니다.
        raw_v = ((read_data << 8) & 0xFF00) | ((read_data >> 8) & 0x00FF)
        
        # 3. 실제 전압 수치로 변환 (INA219의 경우 하위 3비트 제외 후 4mV 곱함)
        voltage_mv = (raw_v >> 3) * 4
        return voltage_mv / 1000.0

    def get_percentage(self, voltage):
        # 2셀 리튬이온 기준 (만충 8.4V, 최소 권장 6.6V)
        max_v = 8.4
        min_v = 6.6
        
        percent = ((voltage - min_v) / (max_v - min_v)) * 100
        # 0% ~ 100% 사이로 값 고정
        return max(0, min(100, percent))


if __name__ == "__main__":
    try:
        oled = Yahboom_OLED(debug=True)
        while True:
            oled.main_program()
            time.sleep(2)
    except KeyboardInterrupt:
        oled.clear(True)
        del oled
        print(" Program closed! ")
        pass
