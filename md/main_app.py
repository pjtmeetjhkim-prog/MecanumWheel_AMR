from UI.app_ui import Ui_Form
from PyQt5.QtWidgets import QApplication, QWidget, QMessageBox
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QImage, QPixmap
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
                    qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format_BGR888)
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


class MainApp(QWidget, Ui_Form):
    """
    PyQt UI + Raspbot JSON 제어 통합.
    - IP 입력 + Connect 버튼으로 TCP 연결
    - 모터/회전 버튼, PTZ 슬라이더, RGB 버튼, Close 버튼과 연동
    """

    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.client = None  # type: ignore[assignment]
        self._rtsp_worker = None  # RTSP 캠 화면 스레드
        self._ultra_worker = None  # 초음파 폴링 스레드
        self._rtsp_port = 8554
        self._rtsp_path = "/raspbot"
        # 현재 상태를 항상 보관해서 set_state로 한 번에 보냄
        self.state = {
            "motor": {"0": 0, "1": 0, "2": 0, "3": 0},
            "ptz": {"pan": 90, "tilt": 90},
            "led": "#000000",
            "buzzer": False,
        }

        # 모터 제어용 주기적 타이머 (서버 타임아웃 0.5초 대응)
        self._move_timer = QTimer()
        self._move_timer.setInterval(200)  # 0.2초마다 전송 (서버 0.5초 타임아웃 대응)
        self._move_timer.timeout.connect(self._send_state)

        # 소켓 통신 동시 접근 방지를 위한 Lock
        self._comm_lock = threading.Lock()

        # 위젯 시그널 연결
        self._connect_signals()
        self.distance_label.setText("거리: --mm")

    # ----------------------------
    # UI <-> 이벤트 연결
    # ----------------------------
    def _connect_signals(self):
        # Connect / Close
        self.pushButton.clicked.connect(self.press_connect)
        self.pushButton_11.clicked.connect(self.press_close)

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

        speed = 150
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

    def _stop_motor(self):
        self._move_timer.stop()  # 주기적 전송 중단
        self.state["motor"] = {"0": 0, "1": 0, "2": 0, "3": 0}
        self._send_state()

    # ----------------------------
    # PTZ 제어
    # ----------------------------
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
    # 버튼 핸들러
    # ----------------------------
    def closeEvent(self, event):
        """창이 닫힐 때 하위 스레드와 타이머를 안전하게 종료"""
        self._move_timer.stop()
        self._stop_rtsp_display()
        self._stop_ultrasound_poll()
        if self.client is not None:
            try:
                self.client.close()
            except Exception:
                pass
        event.accept()

    def press_close(self):
        self.close()

    def press_connect(self):
        # TCP 연결 시도
        if not self._ensure_connected():
            return
        # RTSP 캠 화면 시작 (같은 IP, 포트 8554, 경로 /raspbot)
        self._start_rtsp_display()
        # 초음파 폴링 시작
        self._start_ultrasound_poll()

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
        # url = f"rtsp://{ip}:{self._rtsp_port}{self._rtsp_path}"
        url = f"http://{ip}:8080/stream?topic=/apriltag_overlay&type=mjpeg"
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

    def _on_ultrasound(self, mm: int):
        self.distance_label.setText(f"{mm}mm")

    def _on_rtsp_frame(self, qimg: QImage):
        if qimg.isNull():
            return
        pix = QPixmap.fromImage(qimg)
        self.cam_label.setPixmap(
            pix.scaled(
                self.cam_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_app = MainApp()
    main_app.show()
    sys.exit(app.exec_())
