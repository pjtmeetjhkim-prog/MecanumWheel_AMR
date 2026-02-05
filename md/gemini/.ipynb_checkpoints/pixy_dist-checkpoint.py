from __future__ import print_function
import pixy 
from ctypes import *
from pixy import *

# Pixy 초기화
print("Pixy2 연결 중...")
pixy.init()
pixy.change_prog("color_connected_components")

class Blocks (Structure):
    _fields_ = [ ("m_signature", c_uint),
                 ("m_x", c_uint),
                 ("m_y", c_uint),
                 ("m_width", c_uint),
                 ("m_height", c_uint),
                 ("m_angle", c_uint),
                 ("m_index", c_uint),
                 ("m_age", c_uint) ]

blocks = BlockArray(100)
frame = 0

print("🚀 측정 시작! (물체를 50cm 앞에 두세요)")

while 1:
    count = pixy.ccc_get_blocks(100, blocks)

    if count > 0:
        # 가장 큰 물체 하나만 봄
        target = blocks[0]
        width = target.m_width
        
        # 지금은 K값을 모르니, 일단 너비(Width)만 출력해서 K를 구합시다.
        print(f"[{frame}] 감지됨! 너비(Width): {width} px")
        
    frame += 1