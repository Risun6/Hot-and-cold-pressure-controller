"""Lightweight TCP JSON line utilities for controller coordination."""
from __future__ import annotations

import json
import socket
import threading
import time
from typing import Any, Callable, Dict, Optional

Handler = Callable[[Dict[str, Any]], Dict[str, Any]]


class JSONLineServer:
    """Simple TCP server that exchanges JSON objects delimited by newlines."""

    def __init__(self, host: str, port: int, handler: Handler, name: str = "json-server"):
        self.host = host
        self.port = port
        self._handler = handler
        self._name = name
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._sock: Optional[socket.socket] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._serve, name=f"{self._name}-thread", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop_event.set()
        sock = self._sock
        if sock:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
            try:
                sock.close()
            except Exception:
                pass
        if self._thread:
            self._thread.join(timeout=timeout)
        self._sock = None

    # ------------------------------------------------------------------
    def _serve(self) -> None:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind((self.host, self.port))
                sock.listen(5)
                sock.settimeout(1.0)
                self._sock = sock
                while not self._stop_event.is_set():
                    try:
                        conn, addr = sock.accept()
                    except socket.timeout:
                        continue
                    except OSError:
                        break
                    threading.Thread(
                        target=self._handle_connection,
                        name=f"{self._name}-client",
                        args=(conn,),
                        daemon=True,
                    ).start()
        finally:
            self._sock = None

    def _handle_connection(self, conn: socket.socket) -> None:
        with conn:
            conn.settimeout(5.0)
            buffer = b""
            while not self._stop_event.is_set():
                try:
                    chunk = conn.recv(4096)
                except socket.timeout:
                    continue
                except OSError:
                    break
                if not chunk:
                    break
                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    response = self._dispatch(line)
                    try:
                        conn.sendall(response)
                    except OSError:
                        return

    def _dispatch(self, raw: bytes) -> bytes:
        try:
            request = json.loads(raw.decode("utf-8"))
        except Exception as exc:
            payload = {"ok": False, "error": f"invalid json: {exc}"}
            return (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")

        try:
            result = self._handler(request) or {}
            if "ok" not in result:
                result["ok"] = True
        except Exception as exc:  # noqa: BLE001 - surface error to client
            result = {"ok": False, "error": str(exc)}
        return (json.dumps(result, ensure_ascii=False) + "\n").encode("utf-8")


class JSONLineClient:
    """Small helper to send a single JSON command and read the response."""

    def __init__(self, host: str, port: int, timeout: float = 3.0):
        self.host = host
        self.port = port
        self.timeout = timeout

    def request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        data = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
        with socket.create_connection((self.host, self.port), timeout=self.timeout) as sock:
            sock.sendall(data)
            sock.shutdown(socket.SHUT_WR)
            sock.settimeout(self.timeout)
            chunks = []
            while True:
                try:
                    chunk = sock.recv(4096)
                except socket.timeout:
                    raise TimeoutError("no response from server")
                if not chunk:
                    break
                chunks.append(chunk)
        text = b"".join(chunks).decode("utf-8").strip()
        if not text:
            raise ConnectionError("empty response")
        # Pick the last line in case TCP merged multiple responses
        line = text.splitlines()[-1]
        return json.loads(line)


def wait_for_server(host: str, port: int, timeout: float = 5.0) -> bool:
    """Wait until a TCP server becomes reachable."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.1)
    return False
