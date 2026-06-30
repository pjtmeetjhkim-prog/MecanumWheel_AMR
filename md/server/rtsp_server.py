#!/usr/bin/env python3
# coding: utf-8

"""
라즈베리파이 카메라(또는 /dev/video0)를 RTSP로 송출하는 서버.

- 프로토콜: RTSP
- 기본 포트: 8554
- 기본 경로: /raspbot

접속 예)
  rtsp://<라즈베리파이 IP>:8554/raspbot

의존성:
  - GStreamer, gst-rtsp-server, PyGObject
    sudo apt install -y \
      gstreamer1.0-tools gstreamer1.0-plugins-base \
      gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
      gstreamer1.0-plugins-ugly \
      gir1.2-gst-rtsp-server-1.0 python3-gi python3-gst-1.0
"""

from __future__ import annotations

import argparse
import sys

import gi
gi.require_version("Gst", "1.0")
gi.require_version("GstRtspServer", "1.0")
gi.require_version("GLib", "2.0")
from gi.repository import Gst, GstRtspServer, GLib


class RaspbotRTSPFactory(GstRtspServer.RTSPMediaFactory):
    def __init__(self, device: str, width: int, height: int, fps: int, mode: str):
        super().__init__()
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self.mode = mode  # auto|mjpeg|h264|test
        self.set_shared(True)

    def do_create_element(self, url):
        """
        가능한 경우 H.264(RTP/H264)로 송출하고,
        인코더가 없거나 실패하면 MJPEG(RTP/JPEG)로 폴백합니다.
        그래도 실패하면 videotestsrc로 폴백(503 방지/디버그용).
        """
        def _try_parse(pipeline_str: str):
            try:
                el = Gst.parse_launch(pipeline_str)
                return el
            except Exception as e:
                print(f"[RTSP] pipeline failed: {e}\n  {pipeline_str}", file=sys.stderr)
                return None
                                              
        has_x264 = Gst.ElementFactory.find("x264enc") is not None
        has_rtph264pay = Gst.ElementFactory.find("rtph264pay") is not None
        has_jpegenc = Gst.ElementFactory.find("jpegenc") is not None
        has_rtpjpegpay = Gst.ElementFactory.find("rtpjpegpay") is not None

        # 0) 강제 test 모드
        if self.mode == "test":
            print("[RTSP] mode=test (videotestsrc)", file=sys.stderr)
            return Gst.parse_launch(
                "videotestsrc is-live=true pattern=smpte ! "
                f"video/x-raw,width={self.width},height={self.height},framerate={self.fps}/1 ! "
                "jpegenc ! rtpjpegpay name=pay0 pt=26"
            )

        # MJPG 패스스루가 가장 안정적(USB캠 지원 목록 기반)
        if self.mode in ("auto", "mjpeg") and has_rtpjpegpay:
            # 1) MJPEG passthrough: v4l2src가 MJPG를 직접 뱉는 경우
            mjpeg_passthrough = (
                f"v4l2src device={self.device} ! "
                f"image/jpeg,width={self.width},height={self.height},framerate={self.fps}/1 ! "
                "jpegparse ! "
                "rtpjpegpay name=pay0 pt=26"
            )
            el = _try_parse(mjpeg_passthrough)
            if el is not None:
                print("[RTSP] selected=mjpeg_passthrough", file=sys.stderr)
                return el

        # 2) H.264 (소프트웨어 x264enc). 플러그인이 없으면 바로 스킵.
        if self.mode in ("auto", "h264") and has_x264 and has_rtph264pay:
            h264 = (
                f"v4l2src device={self.device} ! "
                f"video/x-raw,width={self.width},height={self.height},framerate={self.fps}/1 ! "
                "videoconvert ! "
                "x264enc tune=zerolatency bitrate=2048 speed-preset=ultrafast key-int-max=30 ! "
                "video/x-h264,profile=baseline ! "
                "rtph264pay name=pay0 pt=96 config-interval=1"
            )
            el = _try_parse(h264)
            if el is not None:
                print("[RTSP] selected=h264_x264enc", file=sys.stderr)
                return el

        # 3) RAW -> JPEG 인코딩 (mjpeg passthrough가 안 될 때)
        if self.mode in ("auto", "mjpeg") and has_jpegenc and has_rtpjpegpay:
            mjpeg = (
                f"v4l2src device={self.device} ! "
                f"video/x-raw,width={self.width},height={self.height},framerate={self.fps}/1 ! "
                "videoconvert ! "
                "jpegenc ! "
                "rtpjpegpay name=pay0 pt=26"
            )
            el = _try_parse(mjpeg)
            if el is not None:
                print("[RTSP] selected=raw_to_jpegenc", file=sys.stderr)
                return el

        # 4) 최후 폴백: 테스트 패턴(JPEG)
        fallback_jpeg = (
            "videotestsrc is-live=true pattern=smpte ! "
            f"video/x-raw,width={self.width},height={self.height},framerate={self.fps}/1 ! "
            "jpegenc ! "
            "rtpjpegpay name=pay0 pt=26"
        )
        el = _try_parse(fallback_jpeg)
        if el is not None:
            print("[RTSP] selected=fallback_videotestsrc", file=sys.stderr)
            return el

        # 그래도 안되면 최소한의 element 반환(이 경우 여전히 에러 가능)
        return Gst.parse_launch("videotestsrc is-live=true ! fakesink")


class RaspbotRTSPServer(GstRtspServer.RTSPServer):
    def __init__(self, port: int, path: str, device: str, width: int, height: int, fps: int, mode: str):
        super().__init__()
        self.factory = RaspbotRTSPFactory(device, width, height, fps, mode)
        self.get_mount_points().add_factory(path, self.factory)
        self.props.service = str(port)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8554, help="RTSP 포트 (기본: 8554)")
    parser.add_argument("--path", default="/raspbot", help="RTSP 경로 (기본: /raspbot)")
    parser.add_argument("--device", default="/dev/video0", help="카메라 디바이스 경로")
    parser.add_argument("--width", type=int, default=640, help="너비 (많은 USB캠은 640 지원)")
    parser.add_argument("--height", type=int, default=480, help="높이")
    parser.add_argument("--fps", type=int, default=30, help="초당 프레임")
    parser.add_argument("--mode", choices=["auto", "mjpeg", "h264", "test"], default="auto", help="파이프라인 선택 모드")
    args = parser.parse_args()

    Gst.init(None)

    server = RaspbotRTSPServer(
        port=args.port,
        path=args.path,
        device=args.device,
        width=args.width,
        height=args.height,
        fps=args.fps,
        mode=args.mode,
    )
    server.attach(None)

    print(f"RTSP 서버 시작: rtsp://<IP>:{args.port}{args.path}")
    print(f"디바이스: {args.device}, 해상도: {args.width}x{args.height}@{args.fps}fps")

    loop = GLib.MainLoop()
    loop.run()


if __name__ == "__main__":
    main()

