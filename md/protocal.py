"""
Raspbot 원격 조작용 JSON 프로토콜(간단/견고) 구현.

- 전송 포맷: NDJSON (Newline Delimited JSON)
  한 메시지 = JSON 1개 + '\\n'
- 목적: PC/앱(UI) -> 라즈베리파이(로봇) 로 명령 전달, 로봇 -> 상태/센서/ACK 회신

이 파일은 "규격 + 인코딩/디코딩 + TCP 송수신 헬퍼"까지 포함합니다.
실제 하드웨어 제어(Raspbot_Lib 호출)는 서버 쪽 핸들러에서 연결하세요.
"""

from __future__ import annotations

import json
import secrets
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

PROTOCOL_VERSION = 1


class ProtocolError(ValueError):
    pass


def _now_ms() -> int:
    return int(time.time() * 1000)


def new_id(prefix: str = "m") -> str:
    # 짧고 충돌 가능성 낮은 메시지 id
    return f"{prefix}_{secrets.token_hex(8)}"


def dumps_ndjson(obj: Dict[str, Any]) -> bytes:
    """
    JSON을 1줄로 직렬화하고 \\n을 붙여 bytes로 반환.
    - ensure_ascii=False: 한글/유니코드 그대로
    - separators: 공백 제거로 전송량 절감
    """
    try:
        line = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError) as e:
        raise ProtocolError(f"json encode failed: {e}") from e
    return (line + "\n").encode("utf-8")


def loads_ndjson_line(line: bytes) -> Dict[str, Any]:
    """
    NDJSON 한 줄(bytes)을 dict로 파싱.
    line은 반드시 '\\n'을 포함하지 않아도 되지만, 포함되어도 안전하게 처리.
    """
    try:
        text = line.decode("utf-8").strip()
        if not text:
            raise ProtocolError("empty line")
        obj = json.loads(text)
    except UnicodeDecodeError as e:
        raise ProtocolError(f"utf-8 decode failed: {e}") from e
    except json.JSONDecodeError as e:
        raise ProtocolError(f"json decode failed: {e}") from e
    if not isinstance(obj, dict):
        raise ProtocolError("message must be a JSON object")
    return obj


@dataclass(frozen=True)
class Request:
    """
    PC/앱 -> 로봇 요청 메시지

    필드:
    - v: 프로토콜 버전
    - id: 메시지 id (응답과 매칭)
    - type: "req"
    - op: 작업 종류 (예: "move", "stop", "servo", "rgb", "ping", "get_state")
    - data: op별 파라미터 object
    - ts: 송신 시각(ms)
    """

    op: str
    data: Dict[str, Any]
    id: str = ""
    ts: int = 0
    v: int = PROTOCOL_VERSION

    def to_dict(self) -> Dict[str, Any]:
        mid = self.id or new_id("req")
        ts = self.ts or _now_ms()
        return {"v": self.v, "id": mid, "type": "req", "op": self.op, "data": self.data, "ts": ts}


@dataclass(frozen=True)
class Response:
    """
    로봇 -> PC/앱 응답 메시지

    - type: "res"
    - ok: 성공 여부
    - id: 요청 id 그대로
    - data: 성공 payload
    - err: 실패 payload {code, message, details?}
    """

    id: str
    ok: bool
    data: Optional[Dict[str, Any]] = None
    err: Optional[Dict[str, Any]] = None
    ts: int = 0
    v: int = PROTOCOL_VERSION

    def to_dict(self) -> Dict[str, Any]:
        ts = self.ts or _now_ms()
        msg: Dict[str, Any] = {"v": self.v, "id": self.id, "type": "res", "ok": self.ok, "ts": ts}
        if self.ok:
            msg["data"] = self.data or {}
        else:
            msg["err"] = self.err or {"code": "ERR", "message": "unknown error"}
        return msg


def parse_message(obj: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """
    메시지 최소 유효성 검사.
    반환: (type, obj)
    """
    v = obj.get("v", None)
    if v != PROTOCOL_VERSION:
        raise ProtocolError(f"unsupported protocol version: {v}")
    mtype = obj.get("type")
    if mtype not in ("req", "res", "evt"):
        raise ProtocolError(f"invalid type: {mtype}")
    mid = obj.get("id")
    if not isinstance(mid, str) or not mid:
        raise ProtocolError("missing/invalid id")
    return mtype, obj


# -------------------------
# TCP 유틸 (동기)
# -------------------------

import socket


class NDJsonSocket:
    """
    socket 위에서 NDJSON 라인 단위로 송수신하는 얇은 래퍼.
    """

    def __init__(self, sock: socket.socket):
        self.sock = sock
        self._buf = bytearray()

    def send_obj(self, obj: Dict[str, Any]) -> None:
        self.sock.sendall(dumps_ndjson(obj))

    def recv_obj(self, timeout_s: Optional[float] = None, max_line_bytes: int = 64 * 1024) -> Dict[str, Any]:
        """
        '\\n'까지 읽어서 1개 메시지 반환.
        """
        if timeout_s is not None:
            self.sock.settimeout(timeout_s)
        else:
            self.sock.settimeout(None)

        while True:
            nl = self._buf.find(b"\n")
            if nl != -1:
                line = bytes(self._buf[:nl])
                del self._buf[: nl + 1]
                if not line.strip():
                    continue
                return loads_ndjson_line(line)

            if len(self._buf) > max_line_bytes:
                raise ProtocolError("line too large")

            chunk = self.sock.recv(4096)
            if not chunk:
                raise ConnectionError("socket closed")
            self._buf.extend(chunk)


class RaspbotClient:
    """
    PC/앱 측 TCP 클라이언트.
    - 요청을 보내고 같은 id의 응답을 기다림(단일 in-flight 가정)
    """

    def __init__(self, host: str, port: int, timeout_s: float = 2.0):
        self.host = host
        self.port = port
        self.timeout_s = timeout_s
        self._sock: Optional[socket.socket] = None
        self._nd: Optional[NDJsonSocket] = None

    def connect(self) -> None:
        sock = socket.create_connection((self.host, self.port), timeout=self.timeout_s)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._sock = sock
        self._nd = NDJsonSocket(sock)

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None
                self._nd = None

    def request(self, op: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if self._nd is None:
            raise RuntimeError("not connected")
        req = Request(op=op, data=data or {}).to_dict()
        self._nd.send_obj(req)

        res = self._nd.recv_obj(timeout_s=self.timeout_s)
        mtype, msg = parse_message(res)
        if mtype != "res":
            raise ProtocolError(f"expected res, got {mtype}")
        if msg.get("id") != req["id"]:
            raise ProtocolError("response id mismatch")
        if not msg.get("ok", False):
            err = msg.get("err") or {}
            raise RuntimeError(f"request failed: {err.get('code')} {err.get('message')}")
        return msg.get("data") or {}

    # 자주 쓰는 op들(편의)
    def ping(self) -> Dict[str, Any]:
        return self.request("ping", {})

    def stop(self) -> Dict[str, Any]:
        return self.request("stop", {})

    def move(self, vx: float, vy: float, wz: float, speed: int = 180) -> Dict[str, Any]:
        """
        - vx, vy: -1.0~1.0 (로봇 좌표계: x 전/후, y 좌/우)
        - wz: -1.0~1.0 (좌/우 회전)
        - speed: 0~255 (기본 속도 스케일)
        """
        return self.request("move", {"vx": vx, "vy": vy, "wz": wz, "speed": int(speed)})


# -------------------------
# 서버(로봇 측) 스켈레톤
# -------------------------


class RaspbotCommandHandler:
    """
    로봇 측에서 '요청(op,data)'를 실제 동작으로 변환하는 인터페이스.
    프로젝트 상황에 맞게 이 클래스를 상속해서 구현하세요.
    """

    def handle(self, op: str, data: Dict[str, Any]) -> Dict[str, Any]:
        if op == "ping":
            return {"pong": True}
        raise ProtocolError(f"unsupported op: {op}")


def serve_tcp(host: str, port: int, handler: RaspbotCommandHandler) -> None:
    """
    단일 스레드/동기 방식 간단 서버.
    여러 클라이언트 동시 처리까지 필요하면 threading/asyncio로 확장하세요.
    """
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(1)
    while True:
        try:
            conn, _addr = srv.accept()
            nd = NDJsonSocket(conn)
            while True:
                try:
                    msg = nd.recv_obj(timeout_s=None)
                    mtype, obj = parse_message(msg)
                    if mtype != "req":
                        continue
                    op = obj.get("op")
                    data = obj.get("data") or {}
                    if not isinstance(op, str) or not op:
                        raise ProtocolError("missing/invalid op")
                    if not isinstance(data, dict):
                        raise ProtocolError("data must be object")
                    out = handler.handle(op, data)
                    nd.send_obj(Response(id=obj["id"], ok=True, data=out).to_dict())
                except ProtocolError as e:
                    mid = msg.get("id") if isinstance(msg, dict) else new_id("bad")
                    nd.send_obj(
                        Response(
                            id=mid if isinstance(mid, str) and mid else new_id("bad"),
                            ok=False,
                            err={"code": "BAD_REQUEST", "message": str(e)},
                        ).to_dict()
                    )
                except Exception as e:
                    mid = msg.get("id") if isinstance(msg, dict) else new_id("err")
                    nd.send_obj(
                        Response(
                            id=mid if isinstance(mid, str) and mid else new_id("err"),
                            ok=False,
                            err={"code": "INTERNAL", "message": str(e)},
                        ).to_dict()
                    )
        except (ConnectionError, OSError):
            pass
        finally:
            try:
                conn.close()
            except Exception:
                pass
