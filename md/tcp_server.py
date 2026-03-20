#!/usr/bin/env python3
# coding: utf-8

"""
라즈베리파이(로봇)에서 실행하는 TCP JSON 제어 서버.

사용 예)
  python3 raspbot_tcp_server.py --host 0.0.0.0 --port 9000

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


# def _load_protocol_module():
#     """
#     현재 파일 기준 상대경로로 `protocal.py`를 로드.
#     (raspbot/MD/raspbot_tcp_server.py → 같은 디렉터리의 protocal.py)
#     """
#     here = os.path.dirname(os.path.abspath(__file__))
#     proto_path = os.path.join(here, "protocal.py")
#     spec = importlib.util.spec_from_file_location("raspbot_protocol", proto_path)
#     if spec is None or spec.loader is None:
#         raise RuntimeError(f"failed to load protocol module: {proto_path}")
#     mod = importlib.util.module_from_spec(spec)
#     sys.modules["raspbot_protocol"] = mod
#     spec.loader.exec_module(mod)  # type: ignore[attr-defined]
#     return mod


# _proto = _load_protocol_module()
# ProtocolError = protocal.ProtocolError
# RaspbotCommandHandler = protocal.RaspbotCommandHandler
# serve_tcp = protocal.serve_tcp


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

        self.move_thread = threading.Thread(target=self._hardware_loop)
        self.move_thread.daemon = True
        self.move_thread.start()

    def _hardware_loop(self):
        """0.03초마다 상태를 확인하여 하드웨어에 명령 전달 (상태 기반 제어)"""
        while True:
            with self.hw_lock:
                # 1. 타임아웃 체크
                if self.last_move_time > 0 and (time.time() - self.last_move_time > self.move_timeout):
                    self.sports.stop_robot()
                    self.last_move_time = 0
                    self._last_move_params = None
                    self._last_motor_state = None
                
                # 2. 하드웨어 갱신 (명령이 있을 때만)
                elif self.last_move_time > 0:
                    if self._last_move_params:
                        vx, vy, wz, speed = self._last_move_params
                        import math
                        deflection = (math.degrees(math.atan2(vy, vx)) + 360.0) % 360.0
                        self.sports.drifting(speed, deflection, wz)
                    elif self._last_motor_state:
                        for k, v in self._last_motor_state.items():
                            try:
                                mid = int(k)
                                speed = _clamp_int(v, -255, 255, 0)
                                self.bot.Ctrl_Muto(mid, speed)
                            except Exception:
                                continue
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
        with self.hw_lock:
            # 모든 요청(ping, get_ultrasound 등)이 오면 연결이 살아있는 것으로 간주
            if self.last_move_time > 0:
                self.last_move_time = time.time()
            
            # 실제 명령 처리 시작
            return self._handle_locked(op, data)

    def _handle_locked(self, op: str, data: Dict[str, Any]) -> Dict[str, Any]:
        if op == "ping":
            return {"pong": True}

        if op == "stop":
            self.last_move_time = 0
            self._last_move_params = None
            self._last_motor_state = None
            self.sports.stop_robot()
            return {"stopped": True}

        if op == "move":
            # vx, vy, wz: -1.0 ~ 1.0
            vx = float(data.get("vx", 0.0))
            vy = float(data.get("vy", 0.0))
            wz = float(data.get("wz", 0.0))
            speed = int(data.get("speed", 180))

            # 정지 명령인지 확인
            if abs(vx) < 1e-6 and abs(vy) < 1e-6 and abs(wz) < 1e-6:
                self.last_move_time = 0
                self._last_move_params = None
                self._last_motor_state = None
                self.sports.stop_robot()
                return {"stopped": True}

            # 타겟 상태 업데이트 (실제 제어는 _hardware_loop에서 수행)
            self._last_move_params = (vx, vy, wz, speed)
            self._last_motor_state = None
            self.last_move_time = time.time()
            return {"target_set": "move", "params": self._last_move_params}

        if op == "servo":
            # data: {id: 1..?, angle: 0..180}
            sid = int(data.get("id", 1))
            angle = int(data.get("angle", 90))
            self.bot.Ctrl_Servo(sid, angle)
            return {"servo": {"id": sid, "angle": angle}}

        if op == "rgb":
            # data: {state:0/1, color:0..6} (Raspbot_Lib Ctrl_WQ2812_ALL)
            state = int(data.get("state", 1))
            color = int(data.get("color", 0))
            self.bot.Ctrl_WQ2812_ALL(state, color)
            return {"rgb": {"state": state, "color": color}}

        if op == "beep":
            state = int(data.get("state", 1))
            self.bot.Ctrl_BEEP_Switch(state)
            return {"beep": {"state": state}}

        if op == "get_ultrasound":
            mm = self._read_ultrasound_mm()
            return {"mm": mm}

        if op == "set_state":
            applied: Dict[str, Any] = {}

            # motor: 각 모터 속도(-255~255 권장)
            motor = data.get("motor")
            if isinstance(motor, dict):
                # 정지 확인
                is_stop = all(v == 0 for v in motor.values())
                if is_stop:
                    self.last_move_time = 0
                    self._last_motor_state = None
                    self._last_move_params = None
                    self.sports.stop_robot()
                    applied["motor"] = "stopped"
                else:
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

        raise ProtocolError(f"unsupported op: {op}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=9000)
    args = ap.parse_args()

    handler = DefaultRaspbotHandler()
    serve_tcp(args.host, args.port, handler)


if __name__ == "__main__":
    main()

