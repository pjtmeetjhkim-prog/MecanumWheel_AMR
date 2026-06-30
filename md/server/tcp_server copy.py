#!/usr/bin/env python3
# coding: utf-8

"""
라즈베리파이(로봇)에서 실행하는 TCP JSON 제어 서버.

사용 예)
  python3 raspbot_tcp_server.py

PC/앱에서는 `mecanumwheel/git/MecanumWheel_AMR/md/protocal.py`의 `RaspbotClient`로 접속해
NDJSON(JSON 1줄 + \\n) 메시지를 보내면 됩니다.
"""

from __future__ import annotations

import argparse
import datetime
import importlib.util
import math
import os, sys, time
import threading
from typing import Any, Dict
from protocal import ProtocolError, RaspbotCommandHandler, serve_tcp
from Raspbot_Lib import Raspbot
from McLumk_Wheel_Sports import McLumk_Sports
try:
    import smbus  # hw-290.py 와 동일한 방식으로 MPU-6050 접근
except ImportError:
    smbus = None

# ── HW-579 / GY-85 (ADXL345, ITG3205, HMC5883L) 레지스터 주소 ─────────
_I2C_BUS = 4  # 라즈베리파이 I2C 버스 (기존 MPU6050 사용 시와 동일하게 설정)

# 1. ITG3205 (Gyro)
_ITG_ADDR = 0x68
_ITG_DLPF_FS = 0x16
_ITG_PWR_MGM = 0x3E
_ITG_GYRO_XOUT_H = 0x1D
_ITG_GYRO_YOUT_H = 0x1F
_ITG_GYRO_ZOUT_H = 0x21

# 2. ADXL345 (Accel)
_ADXL_ADDR = 0x53
_ADXL_POWER_CTL = 0x2D
_ADXL_DATA_FORMAT = 0x31
_ADXL_DATAX0 = 0x32
_ADXL_DATAY0 = 0x34
_ADXL_DATAZ0 = 0x36

# 3. HMC5883L (Mag) - QMC5883L은 0x0D이나 보편적인 HMC5883L 적용
_HMC_ADDR = 0x1E
_HMC_CONFIG_A = 0x00
_HMC_CONFIG_B = 0x01
_HMC_MODE = 0x02
_HMC_DATAX_H = 0x03
_HMC_DATAZ_H = 0x05
_HMC_DATAY_H = 0x07

def _read_word_2c(bus, addr: int, reg: int) -> int:
    """Big-Endian 16비트 읽기 (ITG3205, HMC5883L 용)"""
    high = bus.read_byte_data(addr, reg)
    low  = bus.read_byte_data(addr, reg + 1)
    value = (high << 8) | low
    if value >= 32768:          # off-by-one 수정: 32768 이상이면 음수 (hw-579.py: > 32767)
        value -= 65536
    return value

def _read_word_2c_le(bus, addr: int, reg: int) -> int:
    """Little-Endian 16비트 읽기 (ADXL345 용)"""
    low  = bus.read_byte_data(addr, reg)
    high = bus.read_byte_data(addr, reg + 1)
    value = (high << 8) | low
    if value >= 32768:          # off-by-one 수정: 32768 이상이면 음수
        value -= 65536
    return value

def _clamp_int(x: Any, lo: int, hi: int, default: int) -> int:
    try:
        v = int(x)
    except Exception:
        return default
    return max(lo, min(hi, v))


def _parse_bool(x: Any, default: bool = False) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    if isinstance(x, str):
        s = x.strip().lower()
        if s in ("1", "true", "t", "yes", "y", "on"):
            return True
        if s in ("0", "false", "f", "no", "n", "off"):
            return False
    return default


def _hex_to_rgb(s: Any) -> Dict[str, int]:
    if not isinstance(s, str):
        raise ValueError("led must be string like '#RRGGBB'")
    t = s.strip()
    if t.startswith("#"):
        t = t[1:]
    if len(t) != 6:
        raise ValueError("led must be '#RRGGBB'")
    r = int(t[0:2], 16)
    g = int(t[2:4], 16)
    b = int(t[4:6], 16)
    return {"r": r, "g": g, "b": b}


class DefaultRaspbotHandler(RaspbotCommandHandler):
    def __init__(self):
        self.bot = Raspbot()
        self.sports = McLumk_Sports(self.bot)
        self._ultra_enabled = False
        self.last_move_time = 0
        self.move_timeout = 1.0 # 0.5초에서 1.0초로 연장하여 지터 대응

        # 이전 명령 캐싱 (중복 요청 무시용)
        self._last_move_params = None
        self._last_motor_state = None

        # ── 명령 로그 파일 초기화 ──────────────────────────────────────────
        # 실행 시각을 파일명에 포함: logs/log_YYYYMMDD_HHMMSS.txt
        _start_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        _log_dir  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
        os.makedirs(_log_dir, exist_ok=True)   # logs/ 폴더 없으면 자동 생성
        self._log_path = os.path.join(_log_dir, f"log_{_start_ts}.txt")
        self._log_lock = threading.Lock()
        with open(self._log_path, "w", encoding="utf-8") as _f:
            _f.write(f"=== TCP Server Log (started {_start_ts}) ===\n")
        print(f"[LOG] 로그 파일 생성: {self._log_path}")

        # 하드웨어(I2C) 동시 접근 방지를 위한 Lock (스레드 시작 전에 초기화)
        self.hw_lock = threading.Lock()

        # run_for 타이머 취소용 이벤트
        self._run_for_cancel: threading.Event = threading.Event()
        # run_for 실행 중 플래그 (ping 등이 last_move_time을 갱신하지 못하도록)
        self._run_for_active: bool = False

        # 정지 보장 카운터: 0보다 크면 _hardware_loop이 계속 0을 전송
        # (I2C 일시 실패 시에도 MCU에 정지가 확실히 전달되도록)
        self._stop_pending: int = 0

        # ── 초음파 LED 경보 상태 추적 ──────────────────────────────────
        self._led_alert_state: int = 0  # 0=off, 1=red, 2=orange

        # ── MPU-6050 IMU (hw-290.py 방식) ──────────────────────────────
        # 별도 I2C bus 객체 사용 (Raspbot I2C와 분리)
        self._imu_bus = None
        self._imu_enabled = False
        # 최신 IMU 읽기값 (백그라운드 스레드가 갱신)
        self._imu_data: Dict[str, float] = {
            "Ax": 0.0, "Ay": 0.0, "Az": 0.0,
            "Gx": 0.0, "Gy": 0.0, "Gz": 0.0,
            "Mx": 0.0, "My": 0.0, "Mz": 0.0,
            "Roll": 0.0, "Pitch": 0.0, "Yaw": 0.0,
        }
        # 자이로 오프셋 (정지 상태 캘리브레이션으로 결정)
        self._gyro_offset: Dict[str, float] = {"x": 0.0, "y": 0.0, "z": 0.0}
        self._imu_lock = threading.Lock()  # _imu_data 보호용 별도 lock
        self._imu_init()  # IMU 초기화 시도
        self._imu_last_traveled_mm: float = 0.0  # 마지막 move_distance 실측 거리(mm)

        self.move_thread = threading.Thread(target=self._hardware_loop)
        self.move_thread.daemon = True
        self.move_thread.start()

        # 초음파 LED 경보 폴링 스레드 (0.1초 주기)
        self._ultra_led_thread = threading.Thread(
            target=self._ultrasound_led_loop, daemon=True
        )
        self._ultra_led_thread.start()

        # IMU 폴링 스레드 (hw-290.py 루프 동일 주기: 0.01초)
        if self._imu_enabled:
            self._imu_thread = threading.Thread(
                target=self._imu_loop, daemon=True
            )
            self._imu_thread.start()

    # ----------------------------------------------------------------
    # 속도(PWM 0~100) → cm/s 변환
    # 실측값 기반 보정 계수 (pwm=60 기준 전진 약 42cm/s)
    def _speed_to_cms(self, pwm: int, sideways: bool = False) -> float:
        pwm = max(0, min(100, pwm))
        if sideways:
            # 측면 이동은 메카넘 특성상 전진 대비 효율이 낮음
            # speed=50 실측 평균 -2.1cm/s 편차 반영
            return 0.38 * pwm + 5.0
        return 0.45 * pwm + 15.0

    def _cms_to_speed(self, velocity_cms: float, sideways: bool = False) -> int:
        """
        cm/s → PWM speed (1~100) 역산. move_distance_time 전용.
        """
        if sideways:
            pwm = (velocity_cms - 5.0) / 0.38
        else:
            if velocity_cms >= 47.0:
                pwm = (velocity_cms - 15.0) / 0.45
            else:
                pwm = (velocity_cms - 17.0) / 0.45
        return max(1, min(100, int(round(pwm))))

    def _active_brake(self, vx, vy, wz, speed):

        brake_speed = int(speed * 0.45)

        brake_speed = max(15, brake_speed)
        brake_speed = min(60, brake_speed)

        brake_time = 0.01 + (speed * 0.0002)

        self._request_move(
            -vx,
            -vy,
            -wz,
            brake_speed
        )

        time.sleep(brake_time)

        self._request_stop()

    def _distance_compensation(self, distance_mm: float, speed: int,
                               sideways: bool = False) -> float:
        """
        선형 보정 모델:  offset(cm) = a × distance_cm + b

        보정 후 목표거리 = 원래거리 + offset
          offset > 0 → 더 가게 명령  (로봇이 부족하게 이동할 때)
          offset < 0 → 덜 가게 명령  (로봇이 과이동할 때)

        파라미터 (a, b)는 실측 오차 데이터를 최소제곱법으로 피팅.

        [전진 PARAMS]
          speed 30 : a=-0.190, b=+5.93
          speed 50 : a=-0.144, b=+3.13
          speed 80 : a=-0.099, b=+3.27

        [우측이동 SIDEWAYS_PARAMS]  ← 실측 오차(오른쪽 이동 기준)
          speed 20 : 데이터 부족 (50cm 이상 모터 비정상) → speed 50 파라미터로 클램프
          speed 50 : a=-0.038, b=+1.87  (오차: +1,-4,+0,+2,-4,-5,-5,-3,-1,-2 cm)
          speed 80 : a=-0.025, b=+2.47  (오차: -3,-4,-2,-7,-6,-6,-4,-7,-5,-5 cm)

        새 speed 추가 방법:
          1. 20cm, 80cm 두 거리만 측정해 오차 e1, e2를 구한다.
          2. a = (e2 - e1) / (80 - 20)
             b = e1 - a * 20
          3. 해당 PARAMS 딕셔너리에 추가.
        """

        distance_cm = distance_mm / 10.0

        # ── 전진용 파라미터 테이블 {speed: (a, b)} ───────────────────
        PARAMS = {
            30: (-0.190, +5.93),
            50: (-0.144, +3.13),
            80: (-0.099, +3.27),
        }

        # ── 측면(좌/우) 이동용 파라미터 테이블 {speed: (a, b)} ───────
        # 실측 데이터 (우측 이동, 단위 cm):
        #   speed 50: distances=[10,20,30,40,50,60,70,80,90,100]
        #             errors   =[+1,-4,+0,+2,-4,-5,-5,-3,-1,-2]
        #   speed 80: distances=[10,20,30,40,50,60,70,80,90,100]
        #             errors   =[-3,-4,-2,-7,-6,-6,-4,-7,-5,-5]
        # 최소제곱법 피팅 결과:
        #   speed 50: a ≈ -0.038, b ≈ +1.87
        #   speed 80: a ≈ -0.025, b ≈ +2.47  (큰 거리에서도 지속적으로 부족)
        # speed 20은 50cm 이상에서 한쪽 모터가 달라지는 비선형 현상 →
        #   30cm 이하만 사용하거나 speed를 높여야 함 (speed 50으로 클램프)
        SIDEWAYS_PARAMS = {
            50: (-0.038, +1.87),
            80: (-0.025, +2.47),
        }
        # ────────────────────────────────────────────────────────────

        table = SIDEWAYS_PARAMS if sideways else PARAMS
        speeds = sorted(table.keys())

        # speed 클램프 + 보간
        if speed <= speeds[0]:
            a, b = table[speeds[0]]
        elif speed >= speeds[-1]:
            a, b = table[speeds[-1]]
        else:
            for i in range(len(speeds) - 1):
                s0, s1 = speeds[i], speeds[i + 1]
                if s0 <= speed <= s1:
                    t = (speed - s0) / (s1 - s0)
                    a0, b0 = table[s0]
                    a1, b1 = table[s1]
                    a = a0 + t * (a1 - a0)
                    b = b0 + t * (b1 - b0)
                    break

        offset = a * distance_cm + b
        corrected_cm = max(1.0, distance_cm + offset)
        return corrected_cm * 10.0

    @staticmethod
    def _vxvywz_to_motors(vx: float, vy: float, wz: float, speed: int) -> Dict[int, int]:
        s = _clamp_int(speed, 0, 255, 180)
        m = {0: 0, 1: 0, 2: 0, 3: 0}

        if wz != 0:                        # 회전 우선
            r = s if wz > 0 else -s        # wz > 0 → 우회전
            m = {0: r, 1: r, 2: -r, 3: -r}
        elif vy > 0 and vx == 0:           # 전진
            m = {0: s,  1: s,  2: s,  3: s}
        elif vy < 0 and vx == 0:           # 후진
            m = {0: -s, 1: -s, 2: -s, 3: -s}
        elif vx > 0 and vy == 0:           # 우측
            m = {0: s,  1: -s, 2: -s, 3: s}
        elif vx < 0 and vy == 0:           # 좌측
            m = {0: -s, 1: s,  2: s,  3: -s}
        elif vy > 0 and vx < 0:            # 대각 전좌
            m = {0: 0,  1: s,  2: s,  3: 0}
        elif vy > 0 and vx > 0:            # 대각 전우
            m = {0: s,  1: 0,  2: 0,  3: s}
        elif vy < 0 and vx < 0:            # 대각 후좌
            m = {0: -s, 1: 0,  2: 0,  3: -s}
        elif vy < 0 and vx > 0:            # 대각 후우
            m = {0: 0,  1: -s, 2: -s, 3: 0}
            
        # ── 하드웨어 편차 보정 (왼쪽 모터 파워 증가) ──
        # 우측 모터가 강해서 발생하는 쏠림을 막기 위해 좌측(0,1번) 모터 파워를 비율만큼 올립니다.
        LEFT_BOOST = 1.08  # 8% 출력 증가 (필요시 이 값을 조절하세요)
        for i in (0, 1):
            if m[i] != 0:
                boosted = int(m[i] * LEFT_BOOST)
                # 방향을 유지하면서 최대 255까지만 적용
                m[i] = max(-255, min(255, boosted))

        return m

    def _send_motors_sync(self, motors: Dict[int, int]) -> None:
        """
        4개 모터를 t0 기준 절대 오프셋 타이밍으로 전송합니다.

        motors: {motor_id: speed(-255~255)}

        MOTOR_OFFSETS 클래스 변수로 각 모터의 전송 시점을 독립적으로 제어합니다.
        한 모터의 오프셋을 바꿔도 다른 모터 타이밍에 영향이 없습니다.

        ※ 펌웨어가 단일 패킷(12바이트)으로 4모터 동시 수신을 지원한다면
          아래처럼 교체하면 완전 동시 제어가 가능합니다:
          payload = []
          for mid in range(4):
              val = motors.get(mid, 0)
              dir_ = 1 if val < 0 else 0
              payload += [mid, dir_, abs(val)]
          self.bot._device.write_i2c_block_data(0x2B, 0x01, payload)  # 12바이트
        """
        

        # ----------------------------------------------------------------
        # vx/vy/wz/speed → 4개 모터 값 변환 (mecanum wheel)
        # main_app.py의 _set_motor_direction과 동일한 매핑
        # wz != 0 이면 제자리 회전, 같은 업데이트 중 를 유지
        # ----------------------------------------------------------------
        # 각 모터의 전송 시점: _send_motors_sync 진입 시각(t0) 기준 절대 오프셋(초)
        # 값을 바꿔도 다른 모터 타이밍에 영향 없음
        MOTOR_OFFSETS: Dict[int, float] = {
            0: 0.000,   # 즉시 전송
            1: 0.000,   # 0ms 후
            2: 0.002,   # 0ms 후
            3: 0.002,   # 0ms 후
        }

        t0 = time.perf_counter()  # 기준 시각

        for mid in range(4):      # 순서 보장: 0, 1, 2, 3
            val = motors.get(mid, 0)
            motor_dir = 1 if val < 0 else 0
            motor_spd = min(abs(val), 255)

            # 이 모터의 목표 전송 시각까지 대기 (절대 오프셋 기준)
            wait = MOTOR_OFFSETS.get(mid, 0.0) - (time.perf_counter() - t0)
            if wait > 0:
                time.sleep(wait)

            data = [mid, motor_dir, motor_spd]
            try:
                self.bot._device.write_i2c_block_data(0x2B, 0x01, data)
            except Exception:
                print(f"write_array I2C error (motor {mid})")

    def _request_stop(self) -> None:
        """
        즉시 상태를 초기화하고 _stop_pending 카운터를 설정합니다.
        _hardware_loop 이 카운터가 소진될 때까지 매 틱마다 0 속도를 전송합니다.
        hw_lock 안에서 호출해야 합니다.

        이전 _stop_all_motors()는 hw_lock을 잡은 채 I2C를 최대 12번 블로킹해
        서버 응답 지연 및 일부 모터 미정지 문제를 유발했습니다.
        """
        self.last_move_time = 0
        self._last_move_params = None
        self._last_motor_state = None
        self._stop_pending = 10  # 0.03s × 10 = 0.3초간 반복 전송

    def _cancel_run_for(self) -> None:
        """
        진행 중인 run_for 타이머를 취소합니다.
        move/stop 등 새 명령이 오면 run_for 타이머가 나중에 새 명령을 방해하지 않도록.
        hw_lock 안에서 호출해야 합니다.
        """
        if self._run_for_active:
            self._run_for_cancel.set()    # _timed_stop 직접 종료
            self._run_for_active = False  # 타임아웃 감시 재개

    def _hardware_loop(self):
        """0.03초마다 상태를 확인하여 하드웨어에 명령 전달 (상태 기반 제어)"""
        while True:
            with self.hw_lock:
                # 0. 정지 보장 카운터 처리
                #    _request_stop() 호출 후 stop_pending 횟수만큼 0을 반복 전송
                if self._stop_pending > 0:
                    self._stop_pending -= 1
                    for mid in range(4):
                        try:
                            self.bot._device.write_i2c_block_data(0x2B, 0x01, [mid, 0, 0])
                        except Exception:
                            print(f"stop I2C error (motor {mid})")
                            self._stop_pending = max(self._stop_pending, 1)  # 실패 시 재시도 유지

                # 1. 타임아웃 체크 (run_for 실행 중엔 _timed_stop이 처리하므로 스킵)
                elif (self.last_move_time > 0
                        and not self._run_for_active
                        and time.time() - self.last_move_time > self.move_timeout):
                    # self._request_stop()  # 상태 초기화 + stop_pending 설정
                    self._active_brake(vx, vy, wz, speed)

                # 2. 하드웨어 갱신 (명령이 있을 때만)
                elif self.last_move_time > 0:
                    if self._last_move_params:
                        vx, vy, wz, speed = self._last_move_params
                        
                        # ── 초음파 센서 연동 제동 로직 ──
                        # 전진 성분(vy > 0)이 있을 때 전방 장애물 감지 시 개입
                        if vy > 0:
                            if self._led_alert_state == 1:
                                if getattr(self, '_red_start_time', 0) == 0:
                                    self._red_start_time = time.time()
                                speed = 0               # 100mm 이내: 강제 정지
                                
                                # TODO: 계속 빨강 상태라면(예: time.time() - self._red_start_time > 3.0 등)
                                # 현재 실행 중인 명령(run_for, move_distance 등)을 아예 중단(취소)하는 기능 추가 필요
                                
                            else:
                                # 빨강 상태가 해제되었더라도, 빨강 감지 시점으로부터 최소 0.5초간 정지 유지
                                if getattr(self, '_red_start_time', 0) > 0:
                                    if time.time() - self._red_start_time < 0.5:
                                        speed = 0
                                    else:
                                        self._red_start_time = 0  # 0.5초 경과 시 해제
                                
                                if self._led_alert_state == 2 and speed > 0:
                                    speed = min(speed, 30)  # 200mm 이내: 서행 (최대 속도 30으로 제한)
                                
                        motors = self._vxvywz_to_motors(vx, vy, wz, speed)
                        self._send_motors_sync(motors)
                    elif self._last_motor_state:
                        motors = {}
                        for k, v in self._last_motor_state.items():
                            try:
                                val = _clamp_int(v, -255, 255, 0)
                                # 개별 모터 제어 시에도 전진(양수) 방향이면 제동 개입
                                if val > 0:
                                    if self._led_alert_state == 1:
                                        if getattr(self, '_red_start_time', 0) == 0:
                                            self._red_start_time = time.time()
                                        val = 0
                                        # TODO: 장기 지속 시 취소 기능 추가
                                    else:
                                        if getattr(self, '_red_start_time', 0) > 0:
                                            if time.time() - self._red_start_time < 0.5:
                                                val = 0
                                            else:
                                                self._red_start_time = 0
                                        
                                        if self._led_alert_state == 2 and val > 0:
                                            val = min(val, 80)  # 서행 (PWM 80)
                                motors[int(k)] = val
                            except Exception:
                                continue
                        self._send_motors_sync(motors)
            time.sleep(0.03)

    def _read_ultrasound_mm(self) -> int:
        """
        초음파(mm) 읽기.
        - 0x1B: High byte, 0x1A: Low byte
        - 스위치는 한번 켠 뒤 계속 유지(주기 폴링에 유리)
        """
        if not self._ultra_enabled:
            self.bot.Ctrl_Ulatist_Switch(1)
            time.sleep(0.05)
            self._ultra_enabled = True
        diss_h = self.bot.read_data_array(0x1B, 1)[0]
        diss_l = self.bot.read_data_array(0x1A, 1)[0]
        return (int(diss_h) << 8) | int(diss_l)

    def _ultrasound_led_loop(self) -> None:
        """
        0.1초 동안 초음파를 여러 번 샘플링하여 평균값으로 LED 경보 색상을 제어합니다.

        ▶ 샘플링 방식 (노이즈 대응)
          - 0.1초 창 안에서 SAMPLE_N 회 읽기 (간격: 0.1/SAMPLE_N 초)
          - 평균값(avg_mm) 으로 임계값 판단
          → 센서 정밀도 부족으로 한 번씩 낮게 튀는 값에 오반응 방지

        임계값:
          avg ≤ 100 mm → 빨간불  (R=255, G=0,   B=0)
          avg ≤ 200 mm → 주황불  (R=255, G=128, B=0)
          avg >  200 mm → 소등    (R=0,   G=0,   B=0)
        """
        THRESH_RED    = 100   # mm
        THRESH_ORANGE = 200   # mm
        SAMPLE_N      = 5     # 0.1초 안에 읽을 횟수
        SAMPLE_SLEEP  = 0.1 / SAMPLE_N   # 각 샘플 간격(초)

        # (state_code) → (R, G, B)
        LED_MAP = {
            0: (0,   0,   0),   # 소등
            1: (255, 0,   0),   # 빨강
            2: (255, 128, 0),   # 주황
        }

        while True:
            try:
                # ── 0.1초 동안 SAMPLE_N 회 샘플링 ──────────────────────
                samples = []
                for _ in range(SAMPLE_N):
                    with self.hw_lock:
                        val = self._read_ultrasound_mm()
                    if val > 0:           # 0은 센서 오류값으로 제외
                        samples.append(val)
                    time.sleep(SAMPLE_SLEEP)

                if not samples:
                    continue  # 유효 샘플 없으면 다음 창으로

                avg_mm = sum(samples) / len(samples)

                # ── 평균값으로 임계값 판별 ─────────────────────────────
                if avg_mm <= THRESH_RED:
                    new_state = 1
                elif avg_mm <= THRESH_ORANGE:
                    new_state = 2
                else:
                    new_state = 0

                # ── 상태가 바뀔 때만 LED 업데이트 ─────────────────────
                if new_state != self._led_alert_state:
                    r, g, b = LED_MAP[new_state]
                    with self.hw_lock:
                        self.bot.Ctrl_WQ2812_brightness_ALL(r, g, b)
                    self._led_alert_state = new_state
                    label = ["소등", "빨강(위험 ≤100mm)", "주황(주의 ≤200mm)"][new_state]
                    print(f"[UltraLED] avg={avg_mm:.0f}mm (n={len(samples)}) → {label}")

            except Exception as e:
                print(f"[UltraLED] 오류: {e}")
                time.sleep(0.1)   # 오류 시에도 0.1초 대기 유지


    # ── MPU-6050 IMU (hw-290.py 방식 그대로) ────────────────────────────

    def _imu_init(self) -> None:
        """
        HW-579 (GY-85) 센서 통합 초기화.
        가속도(ADXL345), 자이로(ITG3205), 지자기(HMC5883L)를 각각 세팅합니다.
        hw-579.py 레퍼런스 기준으로 초기화 순서·레지스터값 정렬.
        """
        if smbus is None:
            print("[IMU] smbus 모듈 없음 — IMU 기능 비활성화")
            return
        try:
            self._imu_bus = smbus.SMBus(_I2C_BUS)

            # 1. ADXL345 초기화 (0x53)
            # POWER_CTL=0x08: Measure 모드
            # DATA_FORMAT=0x0B: Full-resolution, ±16g (4mg/LSB)
            self._imu_bus.write_byte_data(_ADXL_ADDR, _ADXL_POWER_CTL, 0x08)
            self._imu_bus.write_byte_data(_ADXL_ADDR, _ADXL_DATA_FORMAT, 0x0B)
            print("[IMU] ADXL345 (Accel) 초기화 완료")

            # 2. ITG3205 초기화 (0x68)
            # hw-579.py 기준: PWR_MGM=0x00(전원 초기화) → sleep(0.1) → DLPF_FS=0x18
            self._imu_bus.write_byte_data(_ITG_ADDR, _ITG_PWR_MGM, 0x00)         # 전원 초기화 (클럭 내부)
            time.sleep(0.1)                                                        # 안정화 대기
            self._imu_bus.write_byte_data(_ITG_ADDR, _ITG_DLPF_FS, 0x18)         # ±2000°/s, DLPF
            print("[IMU] ITG3205 (Gyro) 초기화 완료")

            # 3. HMC5883L 초기화 (0x1E)
            try:
                self._imu_bus.write_byte_data(_HMC_ADDR, _HMC_CONFIG_A, 0x70)    # 8-average, 15Hz
                self._imu_bus.write_byte_data(_HMC_ADDR, _HMC_CONFIG_B, 0x20)    # Gain=1.3 Ga
                self._imu_bus.write_byte_data(_HMC_ADDR, _HMC_MODE, 0x00)        # Continuous mode
                time.sleep(0.1)                                                    # 첫 변환 대기
                self._hmc_enabled = True
                print("[IMU] HMC5883L (Mag) 초기화 완료")
            except Exception as e:
                print(f"[IMU] HMC5883L 초기화 실패 (또는 QMC5883L 장착): {e}")
                self._hmc_enabled = False

            self._imu_enabled = True
            print(f"[IMU] HW-579(GY-85) 전체 초기화 완료 (I2C bus={_I2C_BUS})")

            # ── 자이로 오프셋 캘리브레이션 (hw-579.py 방식: 300샘플 평균) ──
            # 센서 정착 후 정지 상태에서 수행. 부팅 시 한 번만 실행.
            print("[IMU] Gyro calibration... (센서를 움직이지 마세요)")
            _CAL_SAMPLES = 300
            sx = sy = sz = 0
            for _ in range(_CAL_SAMPLES):
                gx_raw = _read_word_2c(self._imu_bus, _ITG_ADDR, _ITG_GYRO_XOUT_H)
                gy_raw = _read_word_2c(self._imu_bus, _ITG_ADDR, _ITG_GYRO_YOUT_H)
                gz_raw = _read_word_2c(self._imu_bus, _ITG_ADDR, _ITG_GYRO_ZOUT_H)
                sx += gx_raw; sy += gy_raw; sz += gz_raw
                time.sleep(0.01)
            self._gyro_offset["x"] = sx / _CAL_SAMPLES
            self._gyro_offset["y"] = sy / _CAL_SAMPLES
            self._gyro_offset["z"] = sz / _CAL_SAMPLES
            print(f"[IMU] Gyro offset: x={self._gyro_offset['x']:.2f}, "
                  f"y={self._gyro_offset['y']:.2f}, z={self._gyro_offset['z']:.2f}")

        except Exception as e:
            print(f"[IMU] HW-579 초기화 실패: {e}")
            self._imu_enabled = False

    def _imu_loop(self) -> None:
        """HW-579(GY-85) 데이터 폴링 루프 (hw-579.py 레퍼런스 기반)"""
        # 루프 내부 상수 (반복 생성 방지)
        _A_DEAD = 0.05   # g  — 가속도 데드밴드
        _G_DEAD = 0.5    # °/s — 자이로 데드밴드

        def _db(v: float, thr: float) -> float:
            return 0.0 if abs(v) < thr else v

        while True:
            try:
                # ── 1. Accel 읽기 (ADXL345: Little-Endian) ──────────────
                acc_x = _read_word_2c_le(self._imu_bus, _ADXL_ADDR, _ADXL_DATAX0)
                acc_y = _read_word_2c_le(self._imu_bus, _ADXL_ADDR, _ADXL_DATAY0)
                acc_z = _read_word_2c_le(self._imu_bus, _ADXL_ADDR, _ADXL_DATAZ0)

                # ADXL345 ±16g Full-resolution 모드: 스케일 0.0039 g/LSB
                # (hw-579.py: ax *= 0.0039)
                Ax = acc_x * 0.0039
                Ay = acc_y * 0.0039
                Az = acc_z * 0.0039

                # ── 2. Gyro 읽기 (ITG3205: Big-Endian) ──────────────────
                gyro_x_raw = _read_word_2c(self._imu_bus, _ITG_ADDR, _ITG_GYRO_XOUT_H)
                gyro_y_raw = _read_word_2c(self._imu_bus, _ITG_ADDR, _ITG_GYRO_YOUT_H)
                gyro_z_raw = _read_word_2c(self._imu_bus, _ITG_ADDR, _ITG_GYRO_ZOUT_H)

                # 오프셋 보정 후 단위 변환: 14.375 LSB/(°/s)
                # (hw-579.py: gx = (gx - offset) / 14.375)
                Gx = (gyro_x_raw - self._gyro_offset["x"]) / 14.375
                Gy = (gyro_y_raw - self._gyro_offset["y"]) / 14.375
                Gz = (gyro_z_raw - self._gyro_offset["z"]) / 14.375

                # ── 3. Roll / Pitch 계산 (hw-579.py: calc_roll_pitch) ──
                # Roll  = atan2(Ay, Az)
                # Pitch = atan2(-Ax, sqrt(Ay²+Az²))
                roll_deg  = math.degrees(math.atan2(Ay, Az))
                pitch_deg = math.degrees(math.atan2(-Ax, math.sqrt(Ay * Ay + Az * Az)))

                # ── 4. Mag 읽기 (HMC5883L: Big-Endian, X-Z-Y 순서) ─────
                # hw-579.py: read16_be(MAG, 0x03/0x05/0x07), raw int 사용
                mag_x = mag_y = mag_z = yaw_deg = 0.0
                if getattr(self, '_hmc_enabled', False):
                    raw_mx = _read_word_2c(self._imu_bus, _HMC_ADDR, _HMC_DATAX_H)  # reg 0x03
                    raw_mz = _read_word_2c(self._imu_bus, _HMC_ADDR, _HMC_DATAZ_H)  # reg 0x05
                    raw_my = _read_word_2c(self._imu_bus, _HMC_ADDR, _HMC_DATAY_H)  # reg 0x07

                    # Gain=1.3Ga 기준 스케일: 1090 LSB/Gauss
                    mag_x = raw_mx / 1090.0
                    mag_y = raw_my / 1090.0
                    mag_z = raw_mz / 1090.0

                    # Heading(Yaw) 계산 (hw-579.py: calc_heading)
                    yaw_deg = math.degrees(math.atan2(mag_y, mag_x))
                    if yaw_deg < 0:
                        yaw_deg += 360.0

                # ── 5. 데드밴드 필터 ─────────────────────────────────────
                Ax = _db(Ax, _A_DEAD)
                Ay = _db(Ay, _A_DEAD)
                Az = _db(Az, _A_DEAD)
                Gx = _db(Gx, _G_DEAD)
                Gy = _db(Gy, _G_DEAD)
                Gz = _db(Gz, _G_DEAD)

                # ── 6. 데이터 저장 (lock 보호) ───────────────────────────
                with self._imu_lock:
                    self._imu_data = {
                        "Ax":    round(Ax, 4),
                        "Ay":    round(Ay, 4),
                        "Az":    round(Az, 4),
                        "Gx":    round(Gx, 3),
                        "Gy":    round(Gy, 3),
                        "Gz":    round(Gz, 3),
                        "Mx":    round(mag_x, 4),
                        "My":    round(mag_y, 4),
                        "Mz":    round(mag_z, 4),
                        "Roll":  round(roll_deg,  2),
                        "Pitch": round(pitch_deg, 2),
                        "Yaw":   round(yaw_deg,   2),
                    }

            except Exception as e:
                print(f"[IMU] 읽기 오류: {e}")

            time.sleep(0.01)

    def handle(self, op: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        타임아웃 감시를 위한 last_move_time 갱신 (move 계열만)
        run_for 실행 중이 아닌 경우에만 last_move_time 갱신
        (run_for 중 ping 등이 타임아웃 감시를 방해하지 않도록)
        """
        with self.hw_lock:
            if self.last_move_time > 0 and not self._run_for_active:
                self.last_move_time = time.time()
            # 실제 명령 처리 시작
            return self._handle_locked(op, data)

    def _log_command(self, op: str, data: Dict[str, Any]) -> None:
        """수신된 명령을 타임스탬프와 함께 로그 파일에 기록합니다."""
        ts  = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  # ms 단위
        line = f"[{ts}] op={op}  data={data}\n"
        with self._log_lock:
            try:
                with open(self._log_path, "a", encoding="utf-8") as f:
                    f.write(line)
            except Exception as e:
                print(f"[LOG] 기록 실패: {e}")

    def _handle_locked(self, op: str, data: Dict[str, Any]) -> Dict[str, Any]:

        # ── 수신 명령 로그 기록 ──────────────────────────────────────────
        self._log_command(op, data)

        #통신 확인
        if op == "ping":
            return {"pong": True}
        #정지 명령 수신
        if op == "stop":
            self._cancel_run_for()        # run_for 진행 중이라면 취소
            self._request_stop()          # 상태 초기화 + stop_pending 설정
            return {"stopped": True}
        #모터 구동 명령 수신
        if op == "move":
            # vx, vy, wz: -1.0 ~ 1.0
            vx = float(data.get("vx", 0.0))
            vy = float(data.get("vy", 0.0))
            wz = float(data.get("wz", 0.0))
            speed = int(data.get("speed", 180))

            # 정지 명령인지 확인
            if abs(vx) < 1e-6 and abs(vy) < 1e-6 and abs(wz) < 1e-6:
                self._cancel_run_for()    # run_for 진행 중이라면 취소
                self._request_stop()      # 상태 초기화 + stop_pending 설정
                return {"stopped": True}

            # 새 이동 명령 수신 → run_for 취소 후 타겟 업데이트
            self._cancel_run_for()        # run_for 진행 중이라면 취소
            self._last_move_params = (vx, vy, wz, speed)
            self._last_motor_state = None
            self.last_move_time = time.time()
            return {"target_set": "move", "params": self._last_move_params}

        #서보 모터 제어 명령 수신
        if op == "servo":
            # data: {id: 1..?, angle: 0..180}
            sid = int(data.get("id", 1))
            angle = int(data.get("angle", 90))
            self.bot.Ctrl_Servo(sid, angle)
            return {"servo": {"id": sid, "angle": angle}}
        #RGB LED 제어 명령 수신
        if op == "rgb":
            # data: {state:0/1, color:0..6} (Raspbot_Lib Ctrl_WQ2812_ALL)
            state = int(data.get("state", 1))
            color = int(data.get("color", 0))
            self.bot.Ctrl_WQ2812_ALL(state, color)
            return {"rgb": {"state": state, "color": color}}
        #부저 제어 명령 수신
        if op == "beep":
            state = int(data.get("state", 1))
            self.bot.Ctrl_BEEP_Switch(state)
            return {"beep": {"state": state}}
        #초음파 센서 데이터 요청 메세지 수신
        if op == "get_ultrasound":
            mm = self._read_ultrasound_mm()
            return {"mm": mm}
        #IMU (MPU-6050/hw-290.py) 데이터 요청
        if op == "get_imu":
            if not self._imu_enabled:
                return {"error": "IMU not available"}
            with self._imu_lock:
                data_copy = dict(self._imu_data)
            return {"imu": data_copy}
        #IMU 초기화/보정 요청
        if op == "init_imu":
            if not self._imu_enabled:
                return {"error": "IMU not available"}
            self._imu_init()
            return {"init_imu": True}
        #move_distance 완료 후 실측 이동 거리 조회
        if op == "get_imu_result":
            return {
                "traveled_mm": self._imu_last_traveled_mm,
                "imu_tracking": self._imu_enabled,
            }
        #로봇의 모든 상태 요청 메세지 수신
        if op == "set_state":
            applied: Dict[str, Any] = {}

            # motor: 각 모터 속도(-255~255 권장)
            motor = data.get("motor")
            if isinstance(motor, dict):
                # 정지 확인
                is_stop = all(v == 0 for v in motor.values())
                if is_stop:
                    self._cancel_run_for()    # run_for 진행 중이라면 취소
                    self._request_stop()      # 상태 초기화 + stop_pending 설정
                    applied["motor"] = "stopped"
                else:
                    self._cancel_run_for()    # run_for 진행 중이라면 취소
                    self._last_motor_state = motor
                    self._last_move_params = None
                    self.last_move_time = time.time()
                    applied["motor"] = "target_set"

            # ptz: pan/tilt -> 서보(기본: 1=pan, 2=tilt)
            ptz = data.get("ptz")
            if isinstance(ptz, dict):
                pan = _clamp_int(ptz.get("pan", 90), 0, 180, 90)
                tilt = _clamp_int(ptz.get("tilt", 90), 0, 180, 90)
                self.bot.Ctrl_Servo(1, pan)
                self.bot.Ctrl_Servo(2, tilt)
                applied["ptz"] = {"pan": pan, "tilt": tilt}

            # led: "#RRGGBB" -> 밝기 RGB
            if "led" in data:
                rgb = _hex_to_rgb(data.get("led"))
                self.bot.Ctrl_WQ2812_brightness_ALL(rgb["r"], rgb["g"], rgb["b"])
                applied["led"] = {"hex": str(data.get("led")), **rgb}

            # buzzer: on/off
            if "buzzer" in data:
                on = _parse_bool(data.get("buzzer"), default=False)
                self.bot.Ctrl_BEEP_Switch(1 if on else 0)
                applied["buzzer"] = {"on": on}

            return {"applied": applied}

        #모터 일정시간 구동 명령 수신
        if op == "run_for":
            # 지정한 시간(초)만큼 주행 후 자동 정지
            # 필수: seconds (float, 0 초과)
            # 선택: speed (int, 기본 180), vx/vy/wz (float, 기본 0.0)
            #   vx=1.0  → 전진, vx=-1.0 → 후진
            #   vy=1.0  → 오른쪽, vy=-1.0 → 왼쪽
            #   wz=1.0  → 시계방향 회전, wz=-1.0 → 반시계
            try:
                seconds = float(data.get("seconds", 0))
            except (TypeError, ValueError):
                raise ProtocolError("run_for: 'seconds' must be a number")
            if seconds <= 0:
                raise ProtocolError("run_for: 'seconds' must be positive")

            speed  = _clamp_int(data.get("speed", 180), 0, 255, 180)
            vx     = float(data.get("vx", 0.0))
            vy     = float(data.get("vy", 0.0))
            wz     = float(data.get("wz", 0.0))

            # 진행 중인 run_for 가 있으면 취소
            self._run_for_cancel.set()
            self._run_for_cancel.clear()
            self._run_for_active = True  # run_for 실행 중 마킹

            # 움직임 상태 업데이트
            self._last_move_params = (vx, vy, wz, speed)
            self._last_motor_state = None
            self.last_move_time = time.time()

            cancel_event = self._run_for_cancel

            def _timed_stop():
                """seconds 후 자동 정지. cancel_event 가 set 되면 즉시 종료."""
                cancelled = cancel_event.wait(timeout=seconds)
                with self.hw_lock:
                    self._run_for_active = False  # run_for 종료 (취소/완료 공통)
                    if not cancelled:
                        # 시간이 다 됐고 취소되지 않았으면 정지
                        if self._last_move_params is not None:
                            self._request_stop()

            t = threading.Thread(target=_timed_stop, daemon=True)
            t.start()

            return {
                "run_for": "started",
                "seconds": seconds,
                "speed": speed,
                "vx": vx, "vy": vy, "wz": wz,
            }

        if op == "move_distance":

            distance = float(data.get("distance", 0))
            speed = _clamp_int(data.get("speed", 60), 0, 100, 60)

            vx = float(data.get("vx", 0.0))
            vy = float(data.get("vy", 0.0))
            wz = float(data.get("wz", 0.0))

            if distance <= 0:
                raise ProtocolError("distance must be positive")

            # =========================
            # 방향 판별
            # =========================
            sideways = abs(vx) > abs(vy)

            # =========================
            # 거리 보정
            # =========================
            distance = self._distance_compensation(distance, speed, sideways=sideways)
            distance_cm = distance / 10.0          # mm → cm

            # =========================
            # 속도 계산 · 이동시간
            # =========================
            velocity = self._speed_to_cms(speed, sideways)
            seconds  = max(0.05, distance_cm / velocity)

            # =========================
            # run_for 취소 이벤트 초기화
            # =========================
            self._run_for_cancel.set()
            self._run_for_cancel.clear()
            self._run_for_active = True

            self._last_move_params = (vx, vy, wz, speed)
            self._last_motor_state = None
            self.last_move_time = time.time()

            cancel_event  = self._run_for_cancel
            target_mm     = distance          # 보정 후 목표 거리(mm)
            imu_available = self._imu_enabled

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # IMU 기반 실거리 추적 + 파워 보상 스레드
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # 동작 원리:
            #   1) IMU 가속도(Ax/Ay)를 0.02초마다 이중적분 → 추정 이동거리
            #   2) 목표 거리 도달 → 즉시 정지
            #   3) 시간초과(seconds)됐는데 부족하면 speed +10 (최대 +30) 후 재시도
            #   4) 이동 완료 후 실측 거리를 self._imu_last_traveled_mm 에 저장
            #      → 클라이언트 응답(result_queue)에 포함
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            GRAVITY_MS2   = 9.80665
            ACCEL_THRESH  = 0.03     # g: 이 이하면 정지로 간주 (ZVU)
            POLL_DT       = 0.02     # 초: 적분 주기
            MAX_BOOST     = 30       # 최대 speed 보상량
            BOOST_STEP    = 10       # 1회 보상 단계

            def _imu_distance_runner():
                cur_speed = speed
                total_traveled_mm = 0.0   # float 누적 (내부 계산용)
                target_mm_1 = round(target_mm, 1)  # 목표 거리: 0.1 단위 (소수점 첫째자리)

                def _integrate_until(target_mm_local, dur_seconds, cur_spd):
                    """IMU 적분으로 실거리 추적. 목표 도달 or 취소 시 즉시 정지."""
                    nonlocal total_traveled_mm
                    vx_vel = vy_vel = 0.0   # 속도 적분값 (m/s)
                    traveled_m = 0.0        # 이번 구간 이동거리 (m)
                    t_end = time.perf_counter() + dur_seconds

                    while time.perf_counter() < t_end:
                        # 취소 요청 확인
                        if cancel_event.is_set():
                            return traveled_m * 1000.0, True   # (mm, cancelled)

                        if imu_available:
                            with self._imu_lock:
                                imu = dict(self._imu_data)

                            ax_g = imu.get("Ax", 0.0)
                            ay_g = imu.get("Ay", 0.0)

                            # Zero-Velocity Update: 정지 구간은 속도 감쇠
                            if abs(ax_g) < ACCEL_THRESH and abs(ay_g) < ACCEL_THRESH:
                                vx_vel *= 0.5
                                vy_vel *= 0.5
                            else:
                                ax_ms2 = ax_g * GRAVITY_MS2
                                ay_ms2 = ay_g * GRAVITY_MS2
                                vx_vel += ax_ms2 * POLL_DT
                                vy_vel += ay_ms2 * POLL_DT

                            # 수평 이동거리 적분 (방향 무관 절댓값)
                            step_m             = math.hypot(vx_vel, vy_vel) * POLL_DT
                            traveled_m         += step_m
                            total_traveled_mm  += step_m * 1000.0

                            # ── 실시간 진행 거리 갱신 (클라이언트 get_imu_result 폴링용, 소수점 3자리) ──
                            self._imu_last_traveled_mm = round(total_traveled_mm, 3)

                            # 목표 도달 판정: 소수점 첫째자리(0.1mm) 단위 비교
                            if round(total_traveled_mm, 1) >= round(target_mm_local, 1):
                                return traveled_m * 1000.0, False

                        time.sleep(POLL_DT)

                    return traveled_m * 1000.0, False

                boost_used = 0
                while True:
                    seg_mm, cancelled = _integrate_until(target_mm_1, seconds, cur_speed)

                    if cancelled:
                        break

                    # 남은 거리: 0.1mm 단위 비교
                    remaining_mm = round(target_mm_1 - round(total_traveled_mm, 1), 1)

                    # 목표 달성 or IMU 없음 → 종료
                    if remaining_mm <= 0 or not imu_available:
                        break

                    # 부족 거리 남아 있고 보상 여력 있으면 speed 올려 재시도
                    if boost_used < MAX_BOOST:
                        boost_used  = min(boost_used + BOOST_STEP, MAX_BOOST)
                        cur_speed   = min(100, speed + boost_used)
                        extra_sec   = max(0.1, (remaining_mm / 10.0) /
                                         self._speed_to_cms(cur_speed, sideways))
                        print(f"[IMU] 부족 {remaining_mm}mm → speed {cur_speed}, {extra_sec:.2f}s 추가")
                        with self.hw_lock:
                            self._last_move_params = (vx, vy, wz, cur_speed)
                            self.last_move_time = time.time()
                        seg_mm2, cancelled2 = _integrate_until(
                            target_mm_1, extra_sec, cur_speed)
                        if cancelled2:
                            break
                        if round(target_mm_1 - round(total_traveled_mm, 1), 1) <= 0:
                            break
                    else:
                        print(f"[IMU] 최대 보상 도달, 실측={round(total_traveled_mm, 1)}mm / 목표={target_mm_1}mm")
                        break

                # 정지
                with self.hw_lock:
                    self._run_for_active = False
                    self._request_stop()

                # 실측 거리 최종 저장 (클라이언트 전송용: 소수점 3자리)
                self._imu_last_traveled_mm = round(total_traveled_mm, 3)
                print(f"[IMU] 완료 → 실측 {self._imu_last_traveled_mm}mm / 목표 {target_mm_1}mm")


            # 스레드 시작 전 실측 거리 초기화
            self._imu_last_traveled_mm = 0.0
            t = threading.Thread(target=_imu_distance_runner, daemon=True)
            t.start()

            return {
                "move_distance": "started",
                "distance_mm": distance,
                "seconds": round(seconds, 3),
                "speed": speed,
                "vx": vx, "vy": vy, "wz": wz,
                "imu_tracking": imu_available,
            }

        # ──────────────────────────────────────────────────────────────────
        # op: move_distance_time
        # 클라이언트가 거리(mm)와 목표시간(seconds)을 보내면
        # 서버가 필요 속도를 역산하여 지정 시간 동안 구동.
        # ──────────────────────────────────────────────────────────────────
        if op == "move_distance_time":

            try:
                distance = float(data.get("distance", 0))
            except (TypeError, ValueError):
                raise ProtocolError("move_distance_time: 'distance' must be a number (mm)")

            try:
                seconds = float(data.get("seconds", 0))
            except (TypeError, ValueError):
                raise ProtocolError("move_distance_time: 'seconds' must be a number")

            if distance <= 0:
                raise ProtocolError("move_distance_time: 'distance' must be positive")
            if seconds <= 0:
                raise ProtocolError("move_distance_time: 'seconds' must be positive")

            vx = float(data.get("vx", 0.0))
            vy = float(data.get("vy", 0.0))
            wz = float(data.get("wz", 0.0))

            sideways       = abs(vx) > abs(vy)
            distance_cm    = distance / 10.0          # mm → cm
            velocity_needed = distance_cm / seconds   # cm/s
            speed          = self._cms_to_speed(velocity_needed, sideways)

            self._run_for_cancel.set()
            self._run_for_cancel.clear()
            self._run_for_active = True

            self._last_move_params = (vx, vy, wz, speed)
            self._last_motor_state = None
            self.last_move_time    = time.time()

            cancel_event = self._run_for_cancel

            def _timed_stop_dt():
                cancelled = cancel_event.wait(timeout=seconds)
                with self.hw_lock:
                    self._run_for_active = False
                    if not cancelled:
                        self._request_stop()

            threading.Thread(target=_timed_stop_dt, daemon=True).start()

            return {
                "move_distance_time": "started",
                "distance_mm":      distance,
                "seconds":          seconds,
                "velocity_cms":     round(velocity_needed, 3),
                "speed":            speed,
                "run_seconds":      seconds,
                "vx": vx, "vy": vy, "wz": wz,
            }

        raise ProtocolError(f"unsupported op: {op}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=9000)
    args = ap.parse_args()

    handler = DefaultRaspbotHandler()
    try:
        serve_tcp(args.host, args.port, handler)
    except KeyboardInterrupt as m:
        print("프로세스 종료")


if __name__ == "__main__":
    main()
