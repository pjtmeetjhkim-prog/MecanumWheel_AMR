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
import importlib.util
import os
import sys
import time
import threading
from typing import Any, Dict
from protocal import ProtocolError, RaspbotCommandHandler, serve_tcp
from Raspbot_Lib import Raspbot
from McLumk_Wheel_Sports import McLumk_Sports

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

        # 하드웨어(I2C) 동시 접근 방지를 위한 Lock (스레드 시작 전에 초기화)
        self.hw_lock = threading.Lock()

        # run_for 타이머 취소용 이벤트
        self._run_for_cancel: threading.Event = threading.Event()
        # run_for 실행 중 플래그 (ping 등이 last_move_time을 갱신하지 못하도록)
        self._run_for_active: bool = False

        # 정지 보장 카운터: 0보다 크면 _hardware_loop이 계속 0을 전송
        # (I2C 일시 실패 시에도 MCU에 정지가 확실히 전달되도록)
        self._stop_pending: int = 0

        self.move_thread = threading.Thread(target=self._hardware_loop)
        self.move_thread.daemon = True
        self.move_thread.start()

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

    # ----------------------------------------------------------------
    # vx/vy/wz/speed → 4개 모터 값 변환 (mecanum wheel)
    # main_app.py의 _set_motor_direction과 동일한 매핑
    # wz != 0 이면 제자리 회전, 같은 업데이트 중 를 유지
    # ----------------------------------------------------------------
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
        return m

    def _send_motors_sync(self, motors: Dict[int, int]) -> None:
        """
        4개 모터를 최소 지연으로 연속 전송하여 동시 구동에 근접시킵니다.

        motors: {motor_id: speed(-255~255)}

        ※ 펌웨어가 단일 패킷(12바이트)으로 4모터 동시 수신을 지원한다면
          아래처럼 교체하면 완전 동시 제어가 가능합니다:
          payload = []
          for mid in range(4):
              val = motors.get(mid, 0)
              dir_ = 1 if val < 0 else 0
              payload += [mid, dir_, abs(val)]
          self.bot._device.write_i2c_block_data(0x2B, 0x01, payload)  # 12바이트
        """
        for mid in range(4):          # 순서 보장: 0,1,2,3
            val = motors.get(mid, 0)
            motor_dir = 1 if val < 0 else 0
            motor_spd = min(abs(val), 255)
            if mid > 1:
                motor_spd -= 1
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
                        # sports.drifting() 은 라이브러리 내부 버그(set_deflection_rate 인자 불일치)로
                        # 사용 불가. vx/vy/wz → mecanum motor 매핑으로 대체
                        vx, vy, wz, speed = self._last_move_params
                        motors = self._vxvywz_to_motors(vx, vy, wz, speed)
                        self._send_motors_sync(motors)
                    elif self._last_motor_state:
                        motors = {}
                        for k, v in self._last_motor_state.items():
                            try:
                                motors[int(k)] = _clamp_int(v, -255, 255, 0)
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

    def _handle_locked(self, op: str, data: Dict[str, Any]) -> Dict[str, Any]:
        
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

            # mm -> cm
            distance_cm = distance / 10.0

            # =========================
            # 속도 계산
            # =========================

            velocity = self._speed_to_cms(speed, sideways)

            # =========================
            # 이동시간 계산
            # =========================

            seconds = distance_cm / velocity

            # =========================
            # 가속/감속 보정
            # =========================

            # =========================
            # 최소 이동시간 보장
            # =========================

            seconds = max(0.05, seconds)

            # =========================
            # run_for 재사용
            # =========================

            self._run_for_cancel.set()
            self._run_for_cancel.clear()

            self._run_for_active = True

            self._last_move_params = (vx, vy, wz, speed)
            self._last_motor_state = None
            self.last_move_time = time.time()

            cancel_event = self._run_for_cancel

            def _timed_stop():
                cancelled = cancel_event.wait(timeout=seconds)
                with self.hw_lock:
                    self._run_for_active = False
                    if not cancelled:
                        self._request_stop()

            t = threading.Thread(target=_timed_stop, daemon=True)
            t.start()

            return {
                "move_distance": "started",
                "distance_mm": distance,
                "seconds": seconds,
                "speed": speed,
                "vx": vx, "vy": vy, "wz": wz,
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

