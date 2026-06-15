from UI.app_ui import Ui_Form
from PyQt6.QtWidgets import QApplication, QWidget, QMessageBox
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QImage, QPixmap
from protocal import RaspbotClient
import threading
import sys
import importlib.util
import os
import json

try:
    import cv2
except ImportError:
    cv2 = None


class RtspWorker(QThread):
    """RTSP 스트림을 읽어서 프레임을 시그널로 전달. 타임아웃/끊김 시 자동 재연결."""
    frame_ready = pyqtSignal(QImage)

    def __init__(self, rtsp_url: str, parent=None):
        super().__init__(parent)
        self.rtsp_url = rtsp_url
        self._running = True
        self._reconnect_delay_sec = 2

    def stop(self):
        self._running = False

    def run(self):
        if cv2 is None:
            return
        import time
        while self._running:
            # FFMPEG 소켓 타임아웃을 60초로 늘려 30초 경고 완화 (환경변수는 OpenCV 빌드에 따라 동작할 수 있음)
            env_prev = os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS")
            try:
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "stimeout=60000000"
            except Exception:
                pass
            cap = cv2.VideoCapture(self.rtsp_url)
            try:
                if env_prev is not None:
                    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = env_prev
                else:
                    os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)
            except Exception:
                pass
            if not cap.isOpened():
                time.sleep(self._reconnect_delay_sec)
                continue
            try:
                while self._running:
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        break
                    h, w, ch = frame.shape
                    bytes_per_line = ch * w
                    qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_BGR888)
                    self.frame_ready.emit(qimg.copy())
            finally:
                cap.release()
            if self._running:
                time.sleep(self._reconnect_delay_sec)


class UltrasoundWorker(QThread):
    """TCP로 초음파(mm) 주기 조회 후 시그널로 전달."""

    distance_ready = pyqtSignal(int)

    def __init__(self, client_getter, interval_ms: int = 200, parent=None):
        super().__init__(parent)
        self._client_getter = client_getter
        self._interval_ms = max(50, int(interval_ms))
        self._running = True

    def stop(self):
        self._running = False

    def run(self):
        import time

        while self._running:
            c = self._client_getter()
            if c is None:
                time.sleep(self._interval_ms / 1000.0)
                continue
            try:
                with self.parent()._comm_lock:
                    data = c.request("get_ultrasound", {})
                mm = int(data.get("mm", 0))
                self.distance_ready.emit(mm)
            except Exception:
                pass
            time.sleep(self._interval_ms / 1000.0)


class ImuResultWorker(QThread):
    """
    move_distance 완료 대기 후 get_imu_result 를 한 번 조회하여
    실측 이동 거리(mm)를 시그널로 전달합니다.
    """
    arrived = pyqtSignal(float, float)  # (traveled_mm, target_mm)

    def __init__(self, client_getter, comm_lock, wait_sec: float,
                 target_mm: float, parent=None):
        super().__init__(parent)
        self._client_getter = client_getter
        self._comm_lock = comm_lock
        self._wait_sec = wait_sec
        self._target_mm = target_mm

    def run(self):
        import time
        # 주행 완료를 기다림
        time.sleep(max(0.1, self._wait_sec))
        c = self._client_getter()
        if c is None:
            self.arrived.emit(0.0, self._target_mm)
            return
        try:
            with self._comm_lock:
                resp = c.request("get_imu_result", {})
            traveled = float(resp.get("traveled_mm", 0.0))
        except Exception:
            traveled = 0.0
        self.arrived.emit(traveled, self._target_mm)


class ImuPollWorker(QThread):
    """
    TCP로 get_imu 를 주기 조회하여
    Ax/Ay/Az/Gx/Gy/Gz 데이터를 시그널로 전달합니다.
    """
    imu_ready = pyqtSignal(dict)   # {"Ax":..., "Ay":..., "Az":..., "Gx":..., "Gy":..., "Gz":...}

    def __init__(self, client_getter, comm_lock, interval_ms: int = 200, parent=None):
        super().__init__(parent)
        self._client_getter = client_getter
        self._comm_lock = comm_lock
        self._interval_ms = max(50, int(interval_ms))
        self._running = True

    def stop(self):
        self._running = False

    def run(self):
        import time
        while self._running:
            c = self._client_getter()
            if c is None:
                time.sleep(self._interval_ms / 1000.0)
                continue
            try:
                with self._comm_lock:
                    resp = c.request("get_imu", {})
                if "imu" in resp:
                    self.imu_ready.emit(resp["imu"])
            except Exception:
                pass
            time.sleep(self._interval_ms / 1000.0)

class ImuInitWorker(QThread):
    """IMU 초기화 명령을 서버에 백그라운드에서 전송하고 응답을 대기합니다."""
    finished_signal = pyqtSignal(bool, str)  # (성공여부, 메시지)

    def __init__(self, client_getter, comm_lock, parent=None):
        super().__init__(parent)
        self._client_getter = client_getter
        self._comm_lock = comm_lock

    def run(self):
        c = self._client_getter()
        if c is None:
            self.finished_signal.emit(False, "서버와 연결되어 있지 않습니다.")
            return
        try:
            with self._comm_lock:
                resp = c.request("init_imu", {})
            if resp.get("init_imu") is True:
                self.finished_signal.emit(True, "IMU 초기화 완료.")
            else:
                self.finished_signal.emit(False, str(resp.get("error", "알 수 없는 에러")))
        except Exception as e:
            self.finished_signal.emit(False, str(e))



class MoveProgressWorker(QThread):
    """
    move_distance 실행 중 300ms마다 get_imu_result 를 폴링하여
    현재 이동 거리(mm)를 실시간으로 수신합니다.
    """
    progress = pyqtSignal(int, int)   # (traveled_mm, target_mm)

    def __init__(self, client_getter, comm_lock,
                 total_sec: float, target_mm: int, parent=None):
        super().__init__(parent)
        self._client_getter = client_getter
        self._comm_lock = comm_lock
        self._total_sec = total_sec     # 이동 예상 시간
        self._target_mm = int(target_mm)
        self._running = True

    def stop(self):
        self._running = False

    def run(self):
        import time
        t_end = time.time() + self._total_sec + 1.0   # +1초 여유
        POLL = 0.3
        while self._running and time.time() < t_end:
            c = self._client_getter()
            if c is not None:
                try:
                    with self._comm_lock:
                        resp = c.request("get_imu_result", {})
                    traveled = int(resp.get("traveled_mm", 0))
                    self.progress.emit(traveled, self._target_mm)
                    # 목표 도달 시 조기 종료
                    if traveled >= self._target_mm:
                        break
                except Exception:
                    pass
            time.sleep(POLL)



class MainApp(QWidget, Ui_Form):
    """
    PyQt UI + Raspbot JSON 제어 통합.
    - IP 입력 + Connect 버튼으로 TCP 연결
    - 모터/회전 버튼, PTZ 슬라이더, RGB 버튼, Close 버튼과 연동
    """

    # pyqtSignal must be defined on the QObject subclass (MainApp), not on the mixin (Ui_Form)
    motor_speed_signal = pyqtSignal(int, bool)

    def __init__(self):
        super().__init__()
        self.speed = 150          # 초기 모터 속도
        self.setupUi(self)

        self.client = None  # type: ignore[assignment]
        self._rtsp_worker = None  # RTSP 캠 화면 스레드
        self._ultra_worker = None  # 초음파 폴링 스레드
        self._imu_result_worker = None  # move_distance 완료 후 실측 거리 조회 스레드
        self._move_progress_worker = None  # 이동 중 실시간 진행 거리 폴링 스레드
        self._rtsp_port = 8554
        self._rtsp_path = "/raspbot"
        # 현재 상태를 항상 보관해서 set_state로 한 번에 보냄
        self.state = {
            "motor": {"0": 0, "1": 0, "2": 0, "3": 0},
            "ptz": {"pan": 90, "tilt": 90},
            "led": "#000000",
            "buzzer": False,
        }
        self.motor_speed_update(150, True)

        # 모터 제어용 주기적 타이머 (서버 타임아웃 0.5초 대응)
        self._move_timer = QTimer()
        self._move_timer.setInterval(200)  # 0.2초마다 전송 (서버 0.5초 타임아웃 대응)
        self._move_timer.timeout.connect(self._send_state)

        # socat 통신 동시 접근 방지를 위한 Lock
        self._comm_lock = threading.Lock()

        # run_for 프로그레스바용 타이머
        self._run_for_total_ms: int = 0
        self._run_for_elapsed_ms: int = 0
        self._run_for_tick_timer = QTimer()
        self._run_for_tick_timer.setInterval(50)   # 50ms 단위로 업데이트
        self._run_for_tick_timer.timeout.connect(self._run_for_tick)

        # 위젯 시그널 연결
        self._connect_signals()
        self.distance_label.setText("거리: --mm")

        # 숫자패드 키 입력 추적 (키 반복 방지용)
        self._numpad_pressed: set = set()

        self._csv_running = False

        # ── IMU 레이블 매핑 (app_ui.py에서 생성된 위젯 이용) ──────────────────
        # app_ui.py의 imu_lbl_ax ... imu_lbl_gz 속성을
        # {"Ax": lbl, "Ay": lbl, ...} 딕셔너리로 매핑여 _on_imu_data에서 사용
        self._imu_labels: dict = {
            "Ax": self.imu_lbl_ax,
            "Ay": self.imu_lbl_ay,
            "Az": self.imu_lbl_az,
            "Gx": self.imu_lbl_gx,
            "Gy": self.imu_lbl_gy,
            "Gz": self.imu_lbl_gz,
            "Mx": self.imu_lbl_mx,
            "My": self.imu_lbl_my,
            "Mz": self.imu_lbl_mz,
            "Roll":  self.imu_lbl_roll,
            "Pitch": self.imu_lbl_pitch,
            "Yaw":   self.imu_lbl_yaw,
        }

        self._imu_poll_worker = None  # IMU 폴링 스레드
        
        # ── IMU 로깅 관련 ──
        self._is_imu_logging = False
        self._imu_log_start_time = 0.0
        self._imu_csv_file = None
        self._imu_csv_writer = None

    # ----------------------------
    # UI <-> 이벤트 연결
    # ----------------------------
    def _connect_signals(self):
        # Connect / Close
        self.pushButton.clicked.connect(self.press_connect)
        self.pushButton_11.clicked.connect(self.press_close)

        if hasattr(self, 'imu_init_btn'):
            self.imu_init_btn.clicked.connect(self._request_imu_init)

        if hasattr(self, 'imu_log_btn'):
            self.imu_log_btn.clicked.connect(self._toggle_imu_log)

        # 모터 이동/회전 버튼
        self.moto_btn_front.pressed.connect(lambda: self._set_motor_direction(vx=0, vy=1))
        self.moto_btn_back.pressed.connect(lambda: self._set_motor_direction(vx=0, vy=-1))
        self.moto_btn_left.pressed.connect(lambda: self._set_motor_direction(vx=-1, vy=0))
        self.moto_btn_right.pressed.connect(lambda: self._set_motor_direction(vx=1, vy=0))
        self.moto_btn_frontleft.pressed.connect(lambda: self._set_motor_direction(vx=-1, vy=1))
        self.moto_btn_frontright.pressed.connect(lambda: self._set_motor_direction(vx=1, vy=1))
        self.moto_btn_backleft.pressed.connect(lambda: self._set_motor_direction(vx=-1, vy=-1))
        self.moto_btn_backright.pressed.connect(lambda: self._set_motor_direction(vx=1, vy=-1))
        self.moto_btn_rotate_left.pressed.connect(lambda: self._set_motor_direction(vx=-1, vy=0, rotate=True))
        self.moto_btn_rotate_right.pressed.connect(lambda: self._set_motor_direction(vx=1, vy=0, rotate=True))
        self.buzzer_btn.pressed.connect(lambda: self._set_buzzer(True))
        self.buzzer_btn.released.connect(lambda: self._set_buzzer(False))

        # 모터 개별 컨트롤
        self.moto_ctnl1_up_btn.pressed.connect(lambda: self._set_motor_value(key=0, isReverse=False))
        self.moto_ctnl1_up_btn.released.connect(lambda: self._stop_motor(key=0))
        self.moto_ctnl1_down_btn.pressed.connect(lambda: self._set_motor_value(key=0, isReverse=True))
        self.moto_ctnl1_down_btn.released.connect(lambda: self._stop_motor(key=0))
        self.moto_ctnl2_up_btn.pressed.connect(lambda: self._set_motor_value(key=1, isReverse=False))
        self.moto_ctnl2_up_btn.released.connect(lambda: self._stop_motor(key=1))
        self.moto_ctnl2_down_btn.pressed.connect(lambda: self._set_motor_value(key=1, isReverse=True))
        self.moto_ctnl2_down_btn.released.connect(lambda: self._stop_motor(key=1))
        self.moto_ctnl3_up_btn.pressed.connect(lambda: self._set_motor_value(key=2, isReverse=False))
        self.moto_ctnl3_up_btn.released.connect(lambda: self._stop_motor(key=2))
        self.moto_ctnl3_down_btn.pressed.connect(lambda: self._set_motor_value(key=2, isReverse=True))
        self.moto_ctnl3_down_btn.released.connect(lambda: self._stop_motor(key=2))
        self.moto_ctnl4_up_btn.pressed.connect(lambda: self._set_motor_value(key=3, isReverse=False))
        self.moto_ctnl4_up_btn.released.connect(lambda: self._stop_motor(key=3))
        self.moto_ctnl4_down_btn.pressed.connect(lambda: self._set_motor_value(key=3, isReverse=True))
        self.moto_ctnl4_down_btn.released.connect(lambda: self._stop_motor(key=3))

        # 버튼에서 손 떼면 정지
        for btn in [
            self.moto_btn_front,
            self.moto_btn_back,
            self.moto_btn_left,
            self.moto_btn_right,
            self.moto_btn_rotate_left,
            self.moto_btn_rotate_right,
            self.moto_btn_frontleft,
            self.moto_btn_frontright,
            self.moto_btn_backleft,
            self.moto_btn_backright,
        ]:
            btn.released.connect(self._stop_motor)

        # PTZ 슬라이더
        self.cam_pan_slider.setMinimum(0)
        self.cam_pan_slider.setMaximum(180)
        self.cam_pan_slider.setValue(90)
        self.cam_tilt_slider.setMinimum(0)
        self.cam_tilt_slider.setMaximum(180)
        self.cam_tilt_slider.setValue(90)
        self.cam_pan_slider.valueChanged.connect(self._update_ptz_from_slider)
        self.cam_tilt_slider.valueChanged.connect(self._update_ptz_from_slider)

        # RGB 버튼 (단순 프리셋)
        self.pushButton_2.clicked.connect(lambda: self._set_led("#000000"))  # OFF
        self.pushButton_3.clicked.connect(lambda: self._set_led("#ff0000"))  # RED
        self.pushButton_4.clicked.connect(lambda: self._set_led("#00ff00"))  # GREEN
        self.pushButton_5.clicked.connect(lambda: self._set_led("#0000ff"))  # BLUE
        self.pushButton_6.clicked.connect(lambda: self._set_led("#ffff00"))  # YELLOW
        self.pushButton_7.clicked.connect(lambda: self._set_led("#ff00ff"))  # PURPLE
        self.pushButton_8.clicked.connect(lambda: self._set_led("#00ffff"))  # CYAN
        self.pushButton_9.clicked.connect(lambda: self._set_led("#ffffff"))  # WHITE

        # RGB 직접 입력 + Change
        self.pushButton_10.clicked.connect(self._change_led_from_inputs)

        # run_for 실행 버튼
        self.run_for_btn.clicked.connect(self._run_for_send)

        # 거리 + 방향 실행 버튼
        self.dist_dir_run_btn.clicked.connect(self._dist_dir_run)

        # 거리 + 방향 + 시간 실행 버튼
        self.dist_dir_time_run_btn.clicked.connect(self._dist_dir_time_run)

        # CSV 제어 버튼
        self.csv_load_btn.clicked.connect(self._csv_load_file)
        self.csv_new_btn.clicked.connect(self._csv_new_file)
        self.csv_run_btn.clicked.connect(self._csv_run)

        # 비상정지 버튼 (빨간색) — 모든 모터 즉시 정지
        self.emg_stop_btn.clicked.connect(lambda: self._stop_motor(key=-1))

    # ----------------------------
    # 네트워크 / 통신
    # ----------------------------
    def _ensure_connected(self, show_ui: bool = True) -> bool:
        if self.client is not None:
            return True
        ip = self.lineEdit.text().strip()
        if not ip:
            if show_ui:
                QMessageBox.warning(self, "Raspbot", "IP를 먼저 입력하세요.")
            return False
        try:
            self.client = RaspbotClient(ip, 9000, timeout_s=1.0)
            self.client.connect()
        except Exception as e:
            self.client = None
            if show_ui:
                QMessageBox.critical(self, "Raspbot", f"연결 실패: {e}")
            return False
        return True

    def _send_state(self):
        if not self._ensure_connected(show_ui=False):
            self._move_timer.stop()
            return
        try:
            with self._comm_lock:
                _str:str = json.dumps(self.state)
                _str = _str.replace(",", ",\n")
                _str = _str.replace(" ", "\n")
                self.sendlabel.setText("Send :\n"+_str)
                self.client.request("set_state", self.state)  # type: ignore[union-attr]
        except Exception as e:
            # QMessageBox.critical(self, "Raspbot", f"명령 전송 실패: {e}")
            self.sendlabel.setText("Send :\n"+str(e))
            pass

    # ----------------------------
    # 모터 제어
    # ----------------------------
    def _set_motor_direction(self, vx: int, vy: int, rotate: bool = False):
        """
        간단히 4개 모터 속도를 방향/회전 기반으로 셋업.
        - vx: 좌(-1)/우(+1)
        - vy: 후진(-1)/전진(+1)
        - rotate=True 이면 제자리 회전
        """
        if not self._ensure_connected(show_ui=True):
            return

        speed = self.speed
        m = {"0": 0, "1": 0, "2": 0, "3": 0}

        if rotate:
            # 좌/우 회전
            if vx > 0:  # 오른쪽 회전
                m["0"] = speed
                m["1"] = speed
                m["2"] = -speed
                m["3"] = -speed
            else:  # 왼쪽 회전
                m["0"] = -speed
                m["1"] = -speed
                m["2"] = speed
                m["3"] = speed
        else:
            # if vy > 0:  # 전진
            #     m = {"0": speed, "1": speed, "2": speed, "3": speed}
            # elif vy < 0:  # 후진
            #     m = {"0": -speed, "1": -speed, "2": -speed, "3": -speed}
            # elif vx > 0:  # 우측 이동
            #     m = {"0": speed, "1": -speed, "2": -speed, "3": speed}
            # elif vx < 0:  # 좌측 이동
            #     m = {"0": -speed, "1": speed, "2": speed, "3": -speed}

            # Logic for _set_motor_direction
            if vy > 0 and vx == 0:  # 전진
                m = {"0": speed,    "1": speed,    "2": speed,    "3": speed}

            elif vy < 0 and vx == 0: # 후진
                m = {"0": -speed,   "1": -speed,   "2": -speed,   "3": -speed}

            elif vx > 0 and vy == 0: # 우측 이동
                m = {"0": speed,    "1": -speed,   "2": -speed,   "3": speed}

            elif vx < 0 and vy == 0: # 좌측 이동
                m = {"0": -speed,   "1": speed,    "2": speed,    "3": -speed}

            elif vy > 0 and vx < 0:   # 대각선 전좌
                m = {"0": 0,        "1": speed,    "2": speed,    "3": 0}

            elif vy > 0 and vx > 0: # 대각선 전우
                m = {"0": speed,    "1": 0,        "2": 0,        "3": speed}

            elif vy < 0 and vx < 0: # 대각선 후좌
                m = {"0": -speed,   "1": 0,        "2": 0,        "3": -speed}

            elif vy < 0 and vx > 0: # 대각선 후우
                m = {"0": 0,        "1": -speed,   "2": -speed,   "3": 0}


        self.state["motor"] = m
        self._send_state()
        # 타이머 시작 (이미 돌고 있으면 무시됨)
        if not self._move_timer.isActive():
            self._move_timer.start()

    def _set_motor_value(self, key: int, isReverse:bool):
        speed = self.speed if not isReverse else -self.speed
        m = self.state["motor"]
        m[f"{key}"] = speed
        self.state["motor"] = m
        self._send_state()
        if not self._move_timer.isActive():
            self._move_timer.start()

    def _stop_motor(self, key: int = -1):
        self._move_timer.stop()  # 주기적 전송 중단
        self._csv_running = False
        if key == -1:
            self.state["motor"] = {"0": 0, "1": 0, "2": 0, "3": 0}
        else:
            m = self.state["motor"]
            m[f"{key}"] = 0
            self.state["motor"] = m

        # set_state로 모터 0값 전송 + 명시적 stop op로 이중 보장
        # (서버 큐에 쌓인 move 명령이 처리되더라도 stop이 뒤따라옴)
        self._send_state()
        try:
            _stop_payload = {}
            self.sendlabel.setText("Send :\nop: stop\n" + json.dumps(_stop_payload))
            with self._comm_lock:
                self.client.request("stop", _stop_payload)
        except Exception:
            pass  # 연결 없거나 오류 → 무시 (타임아웃으로 자동 정지)

    # ----------------------------
    # run_for 시간 주행
    # ----------------------------
    # 방향 인덱스 → (vx, vy, wz, rotate)
    _DIR_TABLE = [
        ( 0,  1,  0, False),   # 0: 전진
        ( 0, -1,  0, False),   # 1: 후진
        (-1,  0,  0, False),   # 2: 좌측
        ( 1,  0,  0, False),   # 3: 우측
        (-1,  1,  0, False),   # 4: 대각 전좌
        ( 1,  1,  0, False),   # 5: 대각 전우
        (-1, -1,  0, False),   # 6: 대각 후좌
        ( 1, -1,  0, False),   # 7: 대각 후우
        (-1,  0, -1, True),    # 8: 좌회전
        ( 1,  0,  1, True),    # 9: 우회전
    ]

    def _run_for_send(self):
        """run_for_btn 클릭 시 호출: 서버에 run_for 명령 전송 후 프로그레스바 시작"""
        if not self._ensure_connected(show_ui=True):
            return

        idx     = self.run_for_dir_combo.currentIndex()
        speed   = int(self.run_for_speed_spin.value())
        seconds = float(self.run_for_sec_spin.value())
        vx, vy, wz, _rotate = self._DIR_TABLE[idx]

        try:
            _obj = {   # type: ignore[union-attr]
                    "seconds": seconds,
                    "speed":   speed,
                    "vx":      float(vx),
                    "vy":      float(vy),
                    "wz":      float(wz),
                }
            _s = json.dumps(_obj)
            self.sendlabel.setText("Send :\n"+_s)
            with self._comm_lock:
                self.client.request("run_for", _obj)
        except Exception as e:
            QMessageBox.critical(self, "run_for", f"전송 실패: {e}")
            return

        # UI 피드백
        dir_name = self.run_for_dir_combo.currentText()
        self.run_for_status_label.setText(f"실행 중: {dir_name}")
        self.run_for_btn.setEnabled(False)
        self.run_for_progress.setValue(0)

        # 프로그레스바 타이머 시작
        self._run_for_total_ms   = int(seconds * 1000)
        self._run_for_elapsed_ms = 0
        self._run_for_tick_timer.start()

    def _run_for_tick(self):
        """50ms마다 프로그레스바를 업데이트, 완료되면 정지"""
        self._run_for_elapsed_ms += 50
        pct = min(100, int(self._run_for_elapsed_ms * 100 / self._run_for_total_ms))
        self.run_for_progress.setValue(pct)
        if self._run_for_elapsed_ms >= self._run_for_total_ms:
            self._run_for_tick_timer.stop()
            self.run_for_progress.setValue(100)
            self.run_for_status_label.setText("완료 ✔")
            self.run_for_btn.setEnabled(True)
            # 서버가 IMU 기반으로 목표 도달 시 스스로 정지하므로
            # 클라이언트에서 명시적 정지(stop) 명령을 보내지 않습니다.


    # ----------------------------
    # 거리 + 방향 실행
    # ----------------------------
    # 방향 인덱스 맵 (dist_dir_combo / dist_dir_time_dir_combo 공용)
    _DIST_DIR_TABLE = [
        ( 0,  1,  0),   # 0: 전진
        ( 0, -1,  0),   # 1: 후진
        (-1,  0,  0),   # 2: 좌측
        ( 1,  0,  0),   # 3: 우측
        (-1,  1,  0),   # 4: 대각 전좌
        ( 1,  1,  0),   # 5: 대각 전우
        (-1, -1,  0),   # 6: 대각 후좌
        ( 1, -1,  0),   # 7: 대각 후우
        (-1,  0, -1),   # 8: 좌회전
        ( 1,  0,  1),   # 9: 우회전
    ]

    _CM_PER_SEC_AT_150 = 20.0

    # direction 콤보 인덱스 → 프로토콜 문자열 매핑
    _DIST_DIR_NAMES = [
        "forward", "backward", "left", "right",
        "frontleft", "frontright", "backleft", "backright",
        "rotatel", "rotater",
    ]

    def _dist_dir_run(self):
        """move_distance op 전송.
        포맷: {op, direction(str), distance(mm), speed(0~100)}
        서버가 주행 시간을 계산해 반환하면 프로그레스바를 설정합니다."""
        if not self._ensure_connected(show_ui=True):
            return
        idx       = self.dist_dir_combo.currentIndex()
        cm        = float(self.dist_dir_cm_spin.value())
        speed     = int(self.dist_dir_speed_spin.value())   # 0~100
        direction = self._DIST_DIR_NAMES[idx]
        vx, vy, wz = self._DIST_DIR_TABLE[idx]
        mm        = cm * 10.0   # cm → mm 변환

        obj = {
            "direction": direction,
            "distance":  mm,
            "speed":     speed,
            "vx":        float(vx),
            "vy":        float(vy),
            "wz":        float(wz),
        }
        try:
            self.sendlabel.setText("Send :\nop: move_distance\n" + json.dumps(obj, indent=2))
            with self._comm_lock:
                resp = self.client.request("move_distance", obj)
        except Exception as e:
            QMessageBox.critical(self, "move_distance", f"전송 실패: {e}")
            return

        # 서버가 계산한 주행 시간으로 프로그레스바 설정
        seconds = float(resp.get("seconds", mm / 200.0))   # fallback: 200mm/s
        imu_tracking = bool(resp.get("imu_tracking", False))
        dir_name = self.dist_dir_combo.currentText()
        self.run_for_status_label.setText(
            f"실행 중 ({dir_name}, {cm}cm, spd={speed})"
        )
        self._run_for_total_ms   = int(seconds * 1000)
        self._run_for_elapsed_ms = 0
        self.run_for_progress.setValue(0)
        self._run_for_tick_timer.start()

        # 주행 완료 후 실측 거리 + 이동 중 실시간 진행 폴링 (IMU 활성 시)
        if imu_tracking:
            self._start_move_progress(total_sec=seconds, target_mm=int(mm))
            self._start_imu_result_poll(wait_sec=seconds + 0.2, target_mm=mm)

    def _start_imu_result_poll(self, wait_sec: float, target_mm: float):
        """주행 시간 + 여유 후 get_imu_result 를 한 번 요청하는 워커를 시작합니다."""
        if self._imu_result_worker is not None:
            self._imu_result_worker.quit()
            self._imu_result_worker.wait(500)
        self._imu_result_worker = ImuResultWorker(
            client_getter=lambda: self.client,
            comm_lock=self._comm_lock,
            wait_sec=wait_sec,
            target_mm=target_mm,
            parent=self,
        )
        self._imu_result_worker.arrived.connect(self._on_move_distance_done)
        self._imu_result_worker.start()

    def _start_move_progress(self, total_sec: float, target_mm: int):
        """move_distance 실행 중 실시간 진행 거리를 폴링하는 워커 시작."""
        if self._move_progress_worker is not None:
            self._move_progress_worker.stop()
            self._move_progress_worker.wait(500)
        self._move_progress_worker = MoveProgressWorker(
            client_getter=lambda: self.client,
            comm_lock=self._comm_lock,
            total_sec=total_sec,
            target_mm=target_mm,
            parent=self,
        )
        self._move_progress_worker.progress.connect(self._on_move_progress)
        self._move_progress_worker.start()

    def _on_move_progress(self, traveled_mm: int, target_mm: int):
        """move_distance 실행 중 진행 거리를 실시간으로 표시."""
        remaining = max(0, target_mm - traveled_mm)
        self.run_for_status_label.setText(
            f"현재 {traveled_mm}mm / 목표 {target_mm}mm  (남은 {remaining}mm)"
        )

    def _on_move_distance_done(self, traveled_mm: float, target_mm: float):
        """이동 완료 후 실측 거리를 UI에 표시."""
        diff_mm = traveled_mm - target_mm
        if abs(diff_mm) < 10:
            # 목표 거리 ±10mm 이내 → 도착
            self.run_for_status_label.setText(
                f"✅ 도착!  실측 {traveled_mm:.1f}mm / 목표 {target_mm:.0f}mm"
            )
        else:
            sign = "+" if diff_mm > 0 else ""
            self.run_for_status_label.setText(
                f"편차 {sign}{diff_mm:.1f}mm  (실측 {traveled_mm:.1f}mm / 목표 {target_mm:.0f}mm)"
            )

    def _dist_dir_time_run(self):
        """거리(cm) + 방향 + 시간(s) 모두 지정 후 실행"""
        if not self._ensure_connected(show_ui=True):
            return
        idx     = self.dist_dir_time_dir_combo.currentIndex()
        cm      = float(self.dist_dir_time_cm_spin.value())
        seconds = float(self.dist_dir_time_sec_spin.value())
        vx, vy, wz = self._DIST_DIR_TABLE[idx]
        obj = {"distance": cm * 10,           # mm 단위로 전송
               "seconds":  seconds,           # 서버가 이 시간 안에 도달하도록 속도 역산
               "vx": float(vx), "vy": float(vy), "wz": float(wz)}
        try:
            self.sendlabel.setText("Send :\nop: move_distance_time\n" + json.dumps(obj, indent=2))
            with self._comm_lock:
                resp = self.client.request("move_distance_time", obj)
        except Exception as e:
            QMessageBox.critical(self, "dist_dir_time", f"전송 실패: {e}")
            return
        # 서버가 계산한 실제 주행 시간으로 프로그레스바 설정
        run_sec = float(resp.get("run_seconds", seconds))
        self.run_for_status_label.setText(
            f"실행 중 ({self.dist_dir_time_dir_combo.currentText()}, {cm}cm/{seconds}s)")
        self._run_for_total_ms   = int(run_sec * 1000)
        self._run_for_elapsed_ms = 0
        self.run_for_progress.setValue(0)
        self._run_for_tick_timer.start()

    # ----------------------------
    # CSV 경로 제어
    # ----------------------------
    # 로드된 CSV 파일 경로
    _csv_file_path: str = ""

    _DIR_LABEL = [
        "전진", "후진", "좌측", "우측",
        "대각전좌", "대각전우", "대각후좌", "대각후우",
        "좌회전", "우회전",
    ]

    def _csv_load_file(self):
        """파일 다이얼로그로 CSV 1개를 선택하여 경로와 스탭 수를 표시"""
        import csv
        from PyQt6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(
            self, "CSV 파일 선택", "", "CSV Files (*.csv);;All Files (*)"
        )
        if not path:
            return
        self._csv_file_path = path
        self.csv_path_edit.setText(path)

        # 스탭 수 미리구성
        try:
            with open(path, newline="", encoding="utf-8-sig") as f:
                rows = list(csv.reader(f))
            start = 1 if rows and rows[0][0].strip().lower() in ("direction", "방향", "dir") else 0
            valid = sum(1 for r in rows[start:] if len(r) >= 2)
            self.csv_total_label.setText(f"옵 스탭: {valid}개")
            self.csv_step_label.setText("파일 로드 완료 — 실행 버튼을 눌러 시작")
        except Exception as e:
            self.csv_total_label.setText("(읽기 실패)")
            self.csv_step_label.setText(str(e))

    def _csv_new_file(self):
        """새 파일명으로 빈 CSV 생성 (no, action, time, distance, pwm 헤더 포함)"""
        import csv, os
        name = self.csv_new_name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "CSV", "새 파일명을 입력하세요.")
            return
        if not name.endswith(".csv"):
            name += ".csv"
        # 스크립트와 같은 디렉터리에 저장
        base_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(base_dir, name)
        if os.path.exists(path):
            ret = QMessageBox.question(
                self, "CSV", f"{name} 이미 존재합니다.\n덮어쓰시겠습니까?"
            )
            if ret != QMessageBox.Yes:
                return
        try:
            with open(path, "w", newline="", encoding="utf-8-sig") as f:
                csv.writer(f).writerow(["no", "action", "time", "distance", "pwm"])
            self._csv_file_path = path
            self.csv_path_edit.setText(path)
            self.csv_total_label.setText("스탭: 0개")
            self.csv_step_label.setText(f"{name} 생성 완료")
        except Exception as e:
            QMessageBox.critical(self, "CSV", f"파일 생성 실패:\n{e}")

    def _csv_sleep(self, seconds: float) -> bool:
        """지정된 초만큼 대기하되, _csv_running이 False가 되면 즉시 중단(False 반환)합니다."""
        import time
        from PyQt6.QtWidgets import QApplication
        start = time.time()
        while time.time() - start < seconds:
            if getattr(self, "_csv_running", False) is False:
                return False
            QApplication.processEvents()
            time.sleep(0.05)
        return True

    def _csv_run(self):
        """로드된 CSV를 행순서로 실행. 스탭마다 csv_step_label 갱신.

        CSV 포맷:
          no,action,time,distance,pwm
          1,sleep,0.5,,
          2,front,,1000,80
          13,end,,,
        """
        import csv, time as _time
        from PyQt6.QtWidgets import QApplication

        if not self._ensure_connected(show_ui=True):
            return
        if not self._csv_file_path:
            QMessageBox.warning(self, "CSV", "먼저 CSV 파일을 불러오세요.")
            return

        _action_map = {
            "front": (0, "forward"),
            "back": (1, "backward"),
            "left": (2, "left"),
            "right": (3, "right"),
            "frontleft": (4, "frontleft"),
            "frontright": (5, "frontright"),
            "backleft": (6, "backleft"),
            "backright": (7, "backright"),
            "rotatel": (8, "rotatel"),
            "rotater": (9, "rotater"),
            "전진": (0, "forward"),
            "후진": (1, "backward"),
            "좌측": (2, "left"),
            "우측": (3, "right"),
            "좌회전": (8, "rotatel"),
            "우회전": (9, "rotater"),
        }

        try:
            with open(self._csv_file_path, newline="", encoding="utf-8-sig") as f:
                rows = list(csv.reader(f))
        except Exception as e:
            QMessageBox.critical(self, "CSV", f"파일 읽기 실패:\n{e}")
            return

        start = 1 if rows and rows[0][0].strip().lower() in ("no", "번호", "direction", "방향", "dir") else 0
        data_rows = [r for r in rows[start:] if len(r) >= 2]
        total = len(data_rows)
        if total == 0:
            QMessageBox.warning(self, "CSV", "실행할 스탭이 없습니다.")
            return

        errors = []
        self.csv_run_btn.setEnabled(False)
        self._csv_running = True
        try:
            for step_i, row in enumerate(data_rows, start=1):
                if not getattr(self, "_csv_running", False):
                    errors.append("강제 정지됨")
                    break

                # row 포맷: no, action, time, distance, pwm
                col0 = row[0].strip().lower() if len(row) > 0 else ""
                col1 = row[1].strip().lower() if len(row) > 1 else ""
                
                no       = col0
                action   = col1
                time_val = row[2].strip() if len(row) > 2 else ""
                dist_val = row[3].strip() if len(row) > 3 else ""
                pwm_val  = row[4].strip() if len(row) > 4 else ""

                if action == "end" or action == "종료":
                    self.csv_step_label.setText(f"{step_i}/{total}\t종료")
                    QApplication.processEvents()
                    break

                if action == "sleep" or action == "대기":
                    try:
                        seconds = float(time_val) if time_val else 1.0
                    except ValueError:
                        seconds = 1.0
                    
                    self.csv_step_label.setText(f"{step_i}/{total}\t대기\t{seconds}s")
                    QApplication.processEvents()
                    if not self._csv_sleep(seconds):
                        break
                    continue

                if action == "buz" or action == "부저":
                    try:
                        on = (int(pwm_val) == 1) if pwm_val else False
                    except ValueError:
                        on = False
                    
                    self.csv_step_label.setText(f"{step_i}/{total}\t부저\t{'ON' if on else 'OFF'}")
                    QApplication.processEvents()
                    self.state["buzzer"] = on
                    try:
                        self.sendlabel.setText(f"Send :\n부저 {on}")
                        self._send_state()
                    except Exception as e:
                        errors.append(f"Row {step_i}: {e}")
                    
                    try:
                        seconds = float(time_val) if time_val else 0.0
                    except ValueError:
                        seconds = 0.0
                    if seconds > 0:
                        if not self._csv_sleep(seconds):
                            break
                    continue

                if action == "cam" or action == "카메라":
                    try:
                        target_pan = int(dist_val) if dist_val else self.state["ptz"]["pan"]
                    except ValueError:
                        target_pan = self.state["ptz"]["pan"]
                        
                    try:
                        target_tilt = int(pwm_val) if pwm_val else self.state["ptz"]["tilt"]
                    except ValueError:
                        target_tilt = self.state["ptz"]["tilt"]
                        
                    try:
                        seconds = float(time_val) if time_val else 0.0
                    except ValueError:
                        seconds = 0.0

                    self.csv_step_label.setText(f"{step_i}/{total}\t카메라\tPan:{target_pan} Tilt:{target_tilt} / {seconds}s")
                    QApplication.processEvents()

                    start_pan = self.state["ptz"]["pan"]
                    start_tilt = self.state["ptz"]["tilt"]

                    if seconds <= 0:
                        self.state["ptz"] = {"pan": target_pan, "tilt": target_tilt}
                        try:
                            self._send_state()
                        except Exception as e:
                            errors.append(f"Row {step_i}: {e}")
                    else:
                        steps = int(seconds / 0.05)
                        if steps <= 0:
                            steps = 1
                        
                        pan_step = (target_pan - start_pan) / steps
                        tilt_step = (target_tilt - start_tilt) / steps

                        try:
                            for i in range(1, steps + 1):
                                if not getattr(self, "_csv_running", False):
                                    break
                                current_pan = int(start_pan + pan_step * i)
                                current_tilt = int(start_tilt + tilt_step * i)
                                self.state["ptz"] = {"pan": current_pan, "tilt": current_tilt}
                                self._send_state()
                                if not self._csv_sleep(0.05):
                                    break
                        except Exception as e:
                            errors.append(f"Row {step_i}: {e}")
                            
                        self.state["ptz"] = {"pan": target_pan, "tilt": target_tilt}
                        try:
                            self._send_state()
                        except Exception as e:
                            pass
                    continue

                if action == "led" or action == "조명":
                    led_map = {
                        "0": "#000000",
                        "1": "#ff0000",
                        "2": "#00ff00",
                        "3": "#0000ff",
                        "4": "#ffff00",
                        "5": "#ff00ff",
                        "6": "#00ffff",
                        "7": "#ffffff"
                    }
                    hex_color = "#000000"
                    
                    if pwm_val in led_map:
                        hex_color = led_map[pwm_val]
                    elif pwm_val.startswith("#") and len(pwm_val) == 7:
                        hex_color = pwm_val
                    elif len(pwm_val) == 6:
                        # 헥스코드에 #이 빠진 경우
                        hex_color = f"#{pwm_val}"
                    
                    self.csv_step_label.setText(f"{step_i}/{total}\tLED\t{hex_color}")
                    QApplication.processEvents()
                    
                    self.state["led"] = hex_color
                    try:
                        self.sendlabel.setText(f"Send :\nLED {hex_color}")
                        self._send_state()
                    except Exception as e:
                        errors.append(f"Row {step_i}: {e}")
                        
                    try:
                        seconds = float(time_val) if time_val else 0.0
                    except ValueError:
                        seconds = 0.0
                    if seconds > 0:
                        if not self._csv_sleep(seconds):
                            break
                    continue

                if action not in _action_map:
                    errors.append(f"Row {step_i}: 알 수 없는 액션 '{action}'")
                    continue

                idx, direction = _action_map[action]
                vx, vy, wz = self._DIST_DIR_TABLE[idx]
                dir_name = self._DIR_LABEL[idx]

                try:
                    distance = float(dist_val) if dist_val else 0.0
                except ValueError:
                    distance = 0.0

                try:
                    speed = int(pwm_val) if pwm_val else self.speed
                except ValueError:
                    speed = self.speed

                if time_val and distance <= 0:
                    # time만 있고 distance가 없는 경우 -> run_for
                    try:
                        seconds = float(time_val)
                    except ValueError:
                        seconds = 1.0
                    
                    self.csv_step_label.setText(f"{step_i}/{total}\t{dir_name}\t{seconds}s (spd:{speed})")
                    QApplication.processEvents()
                    
                    obj = {
                        "seconds": seconds,
                        "speed": speed,
                        "vx": float(vx), "vy": float(vy), "wz": float(wz)
                    }
                    try:
                        self.sendlabel.setText("Send :\nop: run_for\n" + json.dumps(obj, indent=2))
                        with self._comm_lock:
                            self.client.request("run_for", obj)
                        if not self._csv_sleep(seconds + 0.15):
                            break
                    except Exception as e:
                        errors.append(f"Row {step_i}: {e}")
                        
                elif time_val and distance > 0:
                    # time과 distance가 모두 있는 경우 -> move_distance_time
                    try:
                        seconds = float(time_val)
                    except ValueError:
                        seconds = 1.0
                    
                    self.csv_step_label.setText(f"{step_i}/{total}\t{dir_name}\t{distance}mm/{seconds}s")
                    QApplication.processEvents()
                    
                    obj = {
                        "distance": distance,
                        "seconds": seconds,
                        "vx": float(vx), "vy": float(vy), "wz": float(wz)
                    }
                    try:
                        self.sendlabel.setText("Send :\nop: move_distance_time\n" + json.dumps(obj, indent=2))
                        with self._comm_lock:
                            resp = self.client.request("move_distance_time", obj)
                        run_sec = float(resp.get("run_seconds", seconds))
                        if not self._csv_sleep(run_sec + 0.15):
                            break
                    except Exception as e:
                        errors.append(f"Row {step_i}: {e}")
                else:
                    # distance와 speed가 있는 경우 -> move_distance (기본)
                    self.csv_step_label.setText(f"{step_i}/{total}\t{dir_name}\t{distance}mm (spd:{speed})")
                    QApplication.processEvents()

                    obj = {
                        "direction": direction,
                        "distance": distance,
                        "speed": speed,
                        "vx": float(vx), "vy": float(vy), "wz": float(wz)
                    }
                    try:
                        self.sendlabel.setText("Send :\nop: move_distance\n" + json.dumps(obj, indent=2))
                        with self._comm_lock:
                            resp = self.client.request("move_distance", obj)
                        run_sec = float(resp.get("seconds", distance / 200.0))
                        imu_ok  = bool(resp.get("imu_tracking", False))
                        if not self._csv_sleep(run_sec + 0.15):
                            break
                        # 주행 완료 후 실측 거리 확인
                        if imu_ok:
                            try:
                                with self._comm_lock:
                                    r2 = self.client.request("get_imu_result", {})
                                traveled = float(r2.get("traveled_mm", 0.0))
                                diff = traveled - distance
                                sign = "+" if diff > 0 else ""
                                arrived_txt = (
                                    f"✅ 도착  실측 {traveled:.0f}mm" if abs(diff) < 10
                                    else f"편차 {sign}{diff:.0f}mm  (실측 {traveled:.0f}mm)"
                                )
                                self.csv_step_label.setText(
                                    f"{step_i}/{total}\t{dir_name}\t{arrived_txt}"
                                )
                                QApplication.processEvents()
                            except Exception:
                                pass
                    except Exception as e:
                        errors.append(f"Row {step_i}: {e}")
        finally:
            self.csv_run_btn.setEnabled(True)

        if errors:
            self.csv_step_label.setText(f"완료 (오류 {len(errors)}개)")
            QMessageBox.warning(self, "CSV 실행", "\n".join(errors))
        else:
            self.csv_step_label.setText(f"완료 ✔  ({total}스탭)")

    def _update_ptz_from_slider(self):
        pan = int(self.cam_pan_slider.value())
        tilt = int(self.cam_tilt_slider.value())
        self.state["ptz"] = {"pan": pan, "tilt": tilt}
        self._send_state()

    # ----------------------------
    # LED 제어
    # ----------------------------
    def _set_led(self, hex_color: str):
        self.state["led"] = hex_color
        self._send_state()

    def _change_led_from_inputs(self):
        try:
            r = int(self.lineEdit_2.text() or "0")
            g = int(self.lineEdit_3.text() or "0")
            b = int(self.lineEdit_4.text() or "0")
        except ValueError:
            QMessageBox.warning(self, "RGB", "R/G/B 값은 0~255 정수여야 합니다.")
            return
        r = max(0, min(255, r))
        g = max(0, min(255, g))
        b = max(0, min(255, b))
        self.state["led"] = "#{:02x}{:02x}{:02x}".format(r, g, b)
        self._send_state()

    # ----------------------------
    # 부저 제어
    # ----------------------------
    def _set_buzzer(self, on: bool):
        self.state["buzzer"] = on
        self._send_state()

    # ----------------------------
    # 숫자패드 키보드 모터 제어
    # ----------------------------
    # 숫자패드 키 → (motor_key, isReverse, button_widget_attr)
    # 7:moto_ctnl1_up, 4:moto_ctnl1_down
    # 9:moto_ctnl3_up, 6:moto_ctnl3_down
    # 1:moto_ctnl2_up, 0:moto_ctnl2_down
    # 3:moto_ctnl4_up, .:moto_ctnl4_down
    _NUMPAD_MAP = {
        Qt.Key.Key_7:      (0, False, "moto_ctnl1_up_btn"),
        Qt.Key.Key_4:      (0, True,  "moto_ctnl1_down_btn"),
        Qt.Key.Key_9:      (2, False, "moto_ctnl3_up_btn"),
        Qt.Key.Key_6:      (2, True,  "moto_ctnl3_down_btn"),
        Qt.Key.Key_1:      (1, False, "moto_ctnl2_up_btn"),
        Qt.Key.Key_0:      (1, True,  "moto_ctnl2_down_btn"),
        Qt.Key.Key_3:      (3, False, "moto_ctnl4_up_btn"),
        Qt.Key.Key_Period: (3, True,  "moto_ctnl4_down_btn"),
    }

    def keyPressEvent(self, event):
        """숫자패드 키 누름 → 해당 모터 버튼 pressed 동작"""
        key = event.key()
        # auto-repeat(키 누름 유지) 및 이미 처리 중인 키 무시
        if event.isAutoRepeat() or key in self._numpad_pressed:
            event.accept()
            return
        info = self._NUMPAD_MAP.get(key)
        if info is not None:
            motor_key, is_reverse, btn_attr = info
            self._numpad_pressed.add(key)
            # 버튼 시각적 눌림 상태
            btn = getattr(self, btn_attr, None)
            if btn is not None:
                btn.setDown(True)
            self._set_motor_value(key=motor_key, isReverse=is_reverse)
            event.accept()
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        """숫자패드 키 뗌 → 해당 모터 버튼 released 동작"""
        key = event.key()
        if event.isAutoRepeat():
            event.accept()
            return
        info = self._NUMPAD_MAP.get(key)
        if info is not None and key in self._numpad_pressed:
            motor_key, is_reverse, btn_attr = info
            self._numpad_pressed.discard(key)
            # 버튼 시각적 해제
            btn = getattr(self, btn_attr, None)
            if btn is not None:
                btn.setDown(False)
            # 해당 모터만 0으로 설정 (다른 키가 눌려 있으면 global stop 금지)
            m = self.state["motor"]
            m[f"{motor_key}"] = 0
            self.state["motor"] = m
            # 모든 numpad 키가 해제된 경우에만 타이머 정지 + global stop 전송
            if not self._numpad_pressed:
                self._move_timer.stop()
                self._send_state()
                try:
                    self.sendlabel.setText("Send :\nop: stop\n{}")
                    with self._comm_lock:
                        self.client.request("stop", {})
                except Exception:
                    pass
            else:
                # 아직 다른 키가 눌려 있음 → set_state만 전송
                self._send_state()
            event.accept()
        else:
            super().keyReleaseEvent(event)

    # ----------------------------
    # 버튼 핸들러
    # ----------------------------
    def closeEvent(self, event):
        """창이 닫힐 때 하위 스레드와 타이머를 안전하게 종료"""
        self._move_timer.stop()
        self._stop_rtsp_display()
        self._stop_ultrasound_poll()
        self._stop_imu_poll()
        if self._move_progress_worker is not None:
            self._move_progress_worker.stop()
            self._move_progress_worker.wait(500)
        if self.client is not None:
            try:
                self.client.close()
            except Exception:
                pass
        event.accept()

    def press_close(self):
        self.close()

    def _toggle_fullscreen(self):
        """전체화면 ↔ 일반 창 토글"""
        if self.isFullScreen():
            self.showNormal()
            if hasattr(self, 'fullscreen_btn'):
                self.fullscreen_btn.setText("⛶ 전체화면")
        else:
            self.showFullScreen()
            if hasattr(self, 'fullscreen_btn'):
                self.fullscreen_btn.setText("⊡ 창모드")

    def press_connect(self):
        # TCP 연결 시도
        if not self._ensure_connected():
            return
        # RTSP 캠 화면 시작 (같은 IP, 포트 8554, 경로 /raspbot)
        self._start_rtsp_display()
        # 초음파 폴링 시작
        self._start_ultrasound_poll()
        # IMU 폴링 시작
        self._start_imu_poll()

    # ----------------------------
    # RTSP 캠 화면 (cam_label)
    # ----------------------------
    def _start_rtsp_display(self):
        if cv2 is None:
            self.cam_label.setText("OpenCV 필요")
            return
        self._stop_rtsp_display()
        ip = self.lineEdit.text().strip()
        if not ip:
            return
        url = f"rtsp://{ip}:{self._rtsp_port}{self._rtsp_path}"
        # url = f"http://{ip}:8080/stream?topic=/apriltag_overlay&type=mjpeg"
        self._rtsp_worker = RtspWorker(url, self)
        self._rtsp_worker.frame_ready.connect(self._on_rtsp_frame)
        self._rtsp_worker.start()
        self.cam_label.setText("RTSP 연결 중…")

    def _stop_rtsp_display(self):
        if self._rtsp_worker is not None:
            self._rtsp_worker.stop()
            self._rtsp_worker.wait(1500)
            self._rtsp_worker = None
        self.cam_label.setText("Video")

    # ----------------------------
    # 초음파 거리 폴링 (distance_label)
    # ----------------------------
    def _start_ultrasound_poll(self):
        self._stop_ultrasound_poll()
        if self.client is None:
            return
        self._ultra_worker = UltrasoundWorker(lambda: self.client, interval_ms=200, parent=self)
        self._ultra_worker.distance_ready.connect(self._on_ultrasound)
        self._ultra_worker.start()

    def _stop_ultrasound_poll(self):
        if self._ultra_worker is not None:
            self._ultra_worker.stop()
            self._ultra_worker.wait(1500)
            self._ultra_worker = None

    # ----------------------------
    # IMU 폴링 (imu_groupbox)
    # ----------------------------
    def _start_imu_poll(self):
        self._stop_imu_poll()
        if self.client is None:
            return
        self._imu_poll_worker = ImuPollWorker(
            client_getter=lambda: self.client,
            comm_lock=self._comm_lock,
            interval_ms=200,
            parent=self,
        )
        self._imu_poll_worker.imu_ready.connect(self._on_imu_data)
        self._imu_poll_worker.start()

    def _stop_imu_poll(self):
        if self._imu_poll_worker is not None:
            self._imu_poll_worker.stop()
            self._imu_poll_worker.wait(1500)
            self._imu_poll_worker = None

    def _on_imu_data(self, imu: dict):
        """수신된 IMU 데이터를 그룹박스 레이블에 표시 및 로깅"""
        for key, lbl in self._imu_labels.items():
            val = imu.get(key, 0.0)
            lbl.setText(f"{val:+.4f}")
            
        if self._is_imu_logging and self._imu_csv_writer:
            try:
                elapsed_s = time.time() - self._imu_log_start_time
                self._imu_csv_writer.writerow([
                    f"{elapsed_s:.3f}",
                    imu.get("Ax", 0), imu.get("Ay", 0), imu.get("Az", 0),
                    imu.get("Gx", 0), imu.get("Gy", 0), imu.get("Gz", 0),
                    imu.get("Mx", 0), imu.get("My", 0), imu.get("Mz", 0),
                    imu.get("Roll", 0), imu.get("Pitch", 0), imu.get("Yaw", 0)
                ])
            except Exception as e:
                print(f"[IMU Log Error] {e}")

    def _request_imu_init(self):
        """서버에 IMU 캘리브레이션/초기화 명령을 전송"""
        if not self._ensure_connected(show_ui=True):
            return
            
        # 기존 폴링 잠시 중단
        self._stop_imu_poll()
        self.sendlabel.setText("Send :\nop: init_imu\n{}")
        
        self.imu_init_btn.setEnabled(False)
        self.imu_init_btn.setText("보정중..")
        
        # 워커 생성 및 시작
        self._imu_init_worker = ImuInitWorker(
            client_getter=lambda: self.client,
            comm_lock=self._comm_lock,
            parent=self
        )
        self._imu_init_worker.finished_signal.connect(self._on_imu_init_done)
        self._imu_init_worker.start()

    def _on_imu_init_done(self, success: bool, msg: str):
        self.imu_init_btn.setEnabled(True)
        self.imu_init_btn.setText("초기화")
        
        if success:
            QMessageBox.information(self, "IMU", msg)
        else:
            QMessageBox.critical(self, "IMU", f"IMU 초기화 요청 실패: {msg}")
            
        # 폴링 재개
        self._start_imu_poll()

    def _toggle_imu_log(self):
        import csv
        from datetime import datetime
        
        if not self._is_imu_logging:
            # 로깅 시작
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"imu_log_{timestamp}.csv"
            try:
                self._imu_csv_file = open(filename, 'w', newline='', encoding='utf-8')
                self._imu_csv_writer = csv.writer(self._imu_csv_file)
                self._imu_csv_writer.writerow([
                    "Elapsed_s", "Ax(g)", "Ay(g)", "Az(g)",
                    "Gx(deg/s)", "Gy(deg/s)", "Gz(deg/s)",
                    "Mx(Gs)", "My(Gs)", "Mz(Gs)",
                    "Roll(deg)", "Pitch(deg)", "Yaw(deg)"
                ])
                
                self._imu_log_start_time = time.time()
                self._is_imu_logging = True
                
                self.imu_log_btn.setText("IMU CSV 로깅 중단 (저장중...)")
                self.imu_log_btn.setStyleSheet("background-color: #ffcccc; color: #ff0000; font-weight: bold;")
                print(f"[IMU] 로깅 시작: {filename}")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"IMU 로그 파일 생성 실패: {e}")
        else:
            # 로깅 종료
            self._is_imu_logging = False
            if self._imu_csv_file:
                try:
                    self._imu_csv_file.close()
                except:
                    pass
                self._imu_csv_file = None
                self._imu_csv_writer = None
            
            self.imu_log_btn.setText("IMU CSV 로깅 시작")
            self.imu_log_btn.setStyleSheet("")
            print("[IMU] 로깅 종료")

    def _on_ultrasound(self, mm: int):
        self.distance_label.setText(f"{mm}mm")

    def _on_rtsp_frame(self, qimg: QImage):
        if qimg.isNull():
            return
        pix = QPixmap.fromImage(qimg)
        self.cam_label.setPixmap(
            pix.scaled(
                self.cam_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_app = MainApp()
    main_app.show()
    sys.exit(app.exec())
