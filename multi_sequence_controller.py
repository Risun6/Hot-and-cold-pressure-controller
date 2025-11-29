import csv
import json
import math
import socket
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import tkinter as tk
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinter import filedialog, messagebox, scrolledtext

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from PIL import Image, ImageChops

import ctypes


# =========================
# TCP JSON 行协议（内置客户端）
# =========================


class JSONLineClient:
    """发送单次 JSON 行请求的轻量客户端。"""

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
        line = text.splitlines()[-1]
        return json.loads(line)


# =========================
# 模拟点击（内置）
# =========================
Point = Tuple[int, int]
Reporter = Optional[Callable[[str], None]]


def _get_cursor_pos() -> Point:
    if pyautogui is not None:
        pos = pyautogui.position()
        return int(pos[0]), int(pos[1])
    if sys.platform.startswith("win") and ctypes is not None:
        class _POINT(ctypes.Structure):  # type: ignore[misc, valid-type]
            _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

        pt = _POINT()
        if ctypes.windll.user32.GetCursorPos(ctypes.byref(pt)) == 0:
            raise RuntimeError("获取鼠标坐标失败")
        return int(pt.x), int(pt.y)
    raise RuntimeError("缺少 pyautogui 且非 Windows 平台，无法获取坐标")


def capture_click_point(
    master: tk.Misc,
    *,
    title: str = "设置点击点",
    hint: str = "移动鼠标到目标处，按 Enter 键记录",
    reporter: Reporter = None,
) -> Optional[Point]:
    """提示用户移动鼠标并按回车获取坐标。"""

    top = tk.Toplevel(master)
    top.title(title)
    top.geometry("320x120")
    top.transient(master.winfo_toplevel())
    top.grab_set()
    top.focus_force()

    label = tk.Label(top, text=hint)
    label.pack(expand=True)

    result: Optional[Point] = None

    def _finish(_: Optional[tk.Event] = None) -> None:
        nonlocal result
        try:
            result = _get_cursor_pos()
            if reporter:
                reporter(f"已记录模拟点击点: {result}")
        except Exception as exc:
            messagebox.showerror("错误", f"获取坐标失败: {exc}")
            result = None
        finally:
            try:
                top.grab_release()
            except Exception:
                pass
            top.destroy()

    top.bind("<Return>", _finish)
    top.protocol("WM_DELETE_WINDOW", _finish)
    top.wait_window()
    return result


def _click_once(x: int, y: int, button: str) -> None:
    btn = (button or "left").lower()
    if pyautogui is not None:
        if btn not in ("left", "right", "middle"):
            btn = "left"
        pyautogui.click(x=int(x), y=int(y), button=btn)
        return
    if not sys.platform.startswith("win") or ctypes is None:
        raise RuntimeError("缺少 pyautogui 且非 Windows 平台，无法模拟点击")
    user32 = ctypes.windll.user32  # type: ignore[attr-defined]
    user32.SetCursorPos(int(x), int(y))
    mapping = {
        "left": (0x0002, 0x0004),
        "right": (0x0008, 0x0010),
        "middle": (0x0020, 0x0040),
    }
    down, up = mapping.get(btn, mapping["left"])
    user32.mouse_event(down, 0, 0, 0, 0)
    user32.mouse_event(up, 0, 0, 0, 0)


def perform_click_async(
    x: int,
    y: int,
    *,
    button: str = "left",
    double: bool = False,
    delay_ms: int = 0,
    reporter: Reporter = None,
) -> threading.Thread:
    """后台线程执行模拟点击，避免阻塞 UI。"""

    def _worker() -> None:
        try:
            if delay_ms > 0:
                time.sleep(delay_ms / 1000.0)
            _click_once(int(x), int(y), button)
            if double:
                time.sleep(0.03)
                _click_once(int(x), int(y), button)
            if reporter:
                reporter(
                    f"已模拟点击：({int(x)}, {int(y)}) {button}{' 双击' if double else ''}".strip()
                )
        except Exception as exc:
            if reporter:
                reporter(f"执行模拟点击失败: {exc}")

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    return thread

try:
    import pyautogui
    pyautogui.FAILSAFE = False
except Exception:  # pragma: no cover - 依赖缺失时降级
    pyautogui = None


TEMP_DEFAULT_HOST = "127.0.0.1"
TEMP_DEFAULT_PORT = 50010
PRESS_DEFAULT_HOST = "127.0.0.1"
PRESS_DEFAULT_PORT = 50020

DEFAULT_TEMP_MAE = 0.3
DEFAULT_TEMP_HOLD = 60.0
DEFAULT_TEMP_RECHECK_HOLD = 15.0
DEFAULT_PRESS_MAE = 50.0
DEFAULT_PRESS_HOLD = 30.0
DEFAULT_PRESS_STABLE_WINDOW = 15.0
DEFAULT_PRESS_FLUCT_RANGE = 50.0
DEFAULT_CELL_TIMEOUT = 300.0
DEFAULT_CONFIRM_COUNT = 2
REALTIME_REFRESH_S = 1.0
CHART_HISTORY_SECONDS = 300.0

CONFIG_DIRNAME = "PID_冷热台"


def resolve_documents_dir() -> Path:
    home = Path.home()
    candidates = [
        home / "Documents",
        home / "文档",
        home / "OneDrive" / "Documents",
        home / "OneDrive" / "文档",
    ]
    for path in candidates:
        if path.exists():
            return path
    return home


def get_logs_dir() -> Path:
    root = resolve_documents_dir() / CONFIG_DIRNAME / "logs"
    root.mkdir(parents=True, exist_ok=True)
    return root


class ControllerClient:
    def __init__(
        self,
        host: str,
        port: int,
        name: str,
        handler: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        lock: Optional[threading.RLock] = None,
    ):
        self.host = host
        self.port = port
        self.name = name
        self._handler = handler
        self._lock = lock or (threading.RLock() if handler else None)

    def _request(self, payload: Dict) -> Dict:
        if self._handler:
            lock = self._lock
            if lock:
                lock.acquire()
            try:
                result = self._handler(dict(payload))  # type: ignore[arg-type]
            except Exception as exc:  # noqa: BLE001 - surface to caller
                result = {"ok": False, "error": str(exc)}
            finally:
                if lock:
                    lock.release()
            if result is None:
                result = {}
            if "ok" not in result:
                result["ok"] = True
        else:
            client = JSONLineClient(self.host, self.port, timeout=5.0)
            result = client.request(payload)
        if not result.get("ok", True):
            raise RuntimeError(f"{self.name} 控制失败: {result.get('error', '未知错误')}")
        return result

    def ping(self) -> Dict:
        return self._request({"cmd": "ping"})

    def status(self) -> Dict:
        return self._request({"cmd": "status"})

    def set_temperature(self, value: float, start: bool = False) -> Dict:
        return self._request({"cmd": "set_target", "value": value, "start": start})

    def start_pid(self) -> Dict:
        return self._request({"cmd": "start_pid"})

    def stop_pid(self) -> Dict:
        return self._request({"cmd": "stop_pid"})

    def start_ramp(self, start: float, end: float, rate: float, hold: float = 0.0, loop: bool = False) -> Dict:
        payload = {
            "cmd": "start_ramp",
            "params": {"start": start, "end": end, "rate": rate, "hold": hold, "loop": loop},
        }
        return self._request(payload)

    def stop_ramp(self) -> Dict:
        return self._request({"cmd": "stop_ramp"})

    def set_pressure(self, value: float, start: bool = False) -> Dict:
        payload = {"cmd": "set_target", "value": value, "start": start}
        return self._request(payload)

    def start_pressure(self) -> Dict:
        return self._request({"cmd": "start_control"})

    def stop_pressure(self) -> Dict:
        return self._request({"cmd": "stop_control"})

    def stop_all(self) -> None:
        try:
            self._request({"cmd": "stop_all"})
        except Exception:
            pass


class SequenceRunner(threading.Thread):
    def __init__(
        self,
        app: "MultiSequenceApp",
        plan: Dict,
        temp_handler: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        press_handler: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        temp_lock: Optional[threading.RLock] = None,
        press_lock: Optional[threading.RLock] = None,
    ):
        super().__init__(daemon=True)
        self.app = app
        self.plan = plan
        self.pixel_region: Optional[Tuple[int, int, int, int]] = None
        self.pixel_sensitivity = 5.0
        self.pixel_timeout = 180.0
        self.pixel_interval = 1.0
        self.pixel_base = None
        self.pixel_prev = None
        self.pixel_log_enabled = False
        self._last_click_ts = 0.0
        self.stop_event = threading.Event()
        self.temp_handler = temp_handler
        self.press_handler = press_handler
        self.temp_lock = temp_lock
        self.press_lock = press_lock

        pixel_cfg = (self.plan.get("auto_test") or {}).get("pixel") or {}
        region = pixel_cfg.get("region")
        if isinstance(region, (list, tuple)) and len(region) == 4:
            try:
                self.pixel_region = tuple(int(v) for v in region)
            except Exception:
                self.pixel_region = None
        try:
            self.pixel_sensitivity = float(pixel_cfg.get("sensitivity", self.pixel_sensitivity))
        except Exception:
            pass
        try:
            self.pixel_timeout = float(pixel_cfg.get("timeout", self.pixel_timeout))
        except Exception:
            pass
        try:
            self.pixel_interval = max(0.05, float(pixel_cfg.get("interval_ms", 1000)) / 1000.0)
        except Exception:
            pass
        self.pixel_log_enabled = bool(pixel_cfg.get("log_enabled", False))

    def stop(self) -> None:
        self.stop_event.set()

    def log(self, message: str) -> None:
        self.app.log(message)

    @staticmethod
    def _safe_float(value) -> Optional[float]:
        try:
            if value is None:
                return None
            f = float(value)
            if f != f:
                return None
            return f
        except Exception:
            return None

    def wait_temperature(
        self,
        client: ControllerClient,
        mae_thr: float,
        hold_s: float,
        *,
        timeout: Optional[float],
        confirm_need: int,
        context: str,
    ) -> Tuple[Optional[Dict], bool]:
        start_time = time.time()
        stable_since = None
        confirm_count = 0
        notified_start = False
        while not self.stop_event.is_set():
            try:
                resp = client.status()
            except Exception as exc:
                self.log(f"{context}状态读取失败: {exc}")
                time.sleep(1.0)
                continue
            status = resp.get("status", {})
            temperature = self._safe_float(status.get("temperature"))
            target = self._safe_float(status.get("target"))
            mae = self._safe_float(status.get("mae"))
            error = self._safe_float(status.get("error"))
            stable_flag = bool(status.get("stable"))
            if not stable_flag:
                if mae is not None:
                    stable_flag = mae <= mae_thr
                elif error is not None:
                    stable_flag = abs(error) <= mae_thr
            if stable_flag:
                if not notified_start:
                    self.log(f"{context}进入稳定范围，开始计时")
                    notified_start = True
                if stable_since is None:
                    stable_since = time.time()
                    confirm_count = 0
                elif time.time() - stable_since >= hold_s:
                    confirm_count += 1
                    if confirm_count >= max(1, confirm_need):
                        current_info = []
                        if temperature is not None:
                            current_info.append(f"当前 {temperature:.3f}°C")
                        if target is not None:
                            current_info.append(f"目标 {target:.3f}°C")
                        if current_info:
                            self.log(f"{context}判稳完成（" + ", ".join(current_info) + ")")
                        else:
                            self.log(f"{context}判稳完成")
                        return status, False
            else:
                if stable_since is not None:
                    self.log(f"{context}偏离阈值，重新计时")
                stable_since = None
                confirm_count = 0
                notified_start = False
            if timeout is not None and time.time() - start_time >= timeout:
                self.log(f"{context}判稳超时")
                return None, True
            time.sleep(1.0)
        return None, False

    def wait_pressure(
        self,
        client: ControllerClient,
        mae_thr: float,
        hold_s: float,
        *,
        timeout: Optional[float],
        confirm_need: int,
        context: str,
    ) -> Tuple[Optional[Dict], bool]:
        start_time = time.time()
        stable_since = None
        confirm_count = 0
        notified_start = False
        while not self.stop_event.is_set():
            try:
                resp = client.status()
            except Exception as exc:
                self.log(f"{context}状态读取失败: {exc}")
                time.sleep(1.0)
                continue
            status = resp.get("status", {})
            mae = self._safe_float(status.get("mae"))
            last = status.get("last", {}) if isinstance(status.get("last"), dict) else {}
            error = self._safe_float(last.get("error"))
            stable_flag = bool(status.get("stable"))
            if not stable_flag:
                if mae is not None:
                    stable_flag = mae <= mae_thr
                elif error is not None:
                    stable_flag = abs(error) <= mae_thr
            if stable_flag:
                if not notified_start:
                    self.log(f"{context}进入稳定范围，开始计时")
                    notified_start = True
                if stable_since is None:
                    stable_since = time.time()
                    confirm_count = 0
                elif time.time() - stable_since >= hold_s:
                    confirm_count += 1
                    if confirm_count >= max(1, confirm_need):
                        self.log(f"{context}判稳完成")
                        return status, False
            else:
                if stable_since is not None:
                    self.log(f"{context}偏离阈值，重新计时")
                stable_since = None
                confirm_count = 0
                notified_start = False
            if timeout is not None and time.time() - start_time >= timeout:
                self.log(f"{context}判稳超时")
                return None, True
            time.sleep(1.0)
        return None, False

    def ensure_temperature_running(self, client: ControllerClient, target: float) -> bool:
        try:
            resp = client.set_temperature(target, start=True)
        except Exception as exc:
            self.log(f"温控目标设置失败: {exc}")
            return False
        if not resp.get("result", True):
            self.log("温控拒绝更新目标温度")
            return False
        try:
            resp = client.start_pid()
        except Exception as exc:
            self.log(f"温控 PID 启动失败: {exc}")
            return False
        if not resp.get("result", True):
            self.log("温控 PID 未能启动")
            return False
        return True

    def ensure_pressure_running(self, client: ControllerClient, target: float) -> bool:
        try:
            resp = client.set_pressure(target, start=True)
        except Exception as exc:
            self.log(f"压力目标设置失败: {exc}")
            return False
        if not resp.get("result", True):
            self.log("压力控制拒绝更新目标")
            return False
        try:
            resp = client.start_pressure()
        except Exception as exc:
            self.log(f"压力控制启动失败: {exc}")
            return False
        if not resp.get("result", True):
            self.log("压力控制未能启动")
            return False
        return True

    def wait_pressure_stable_after_temp(
        self,
        client: ControllerClient,
        window_s: float,
        fluct_threshold: float,
        *,
        timeout: Optional[float],
    ) -> Tuple[Optional[Dict], bool]:
        start_time = time.time()
        window_s = max(1.0, float(window_s))
        fluct_threshold = max(0.0, float(fluct_threshold))
        samples: Deque[Tuple[float, float]] = deque()
        last_report = start_time
        self.log(
            f"温度切换后，监测压力在 {window_s:.1f}s 内波动不超过 {fluct_threshold:.3f} 以开始测试"
        )
        while not self.stop_event.is_set():
            try:
                resp = client.status()
            except Exception as exc:
                self.log(f"温度切换后压力监测失败: {exc}")
                time.sleep(1.0)
                continue
            status = resp.get("status", {})
            pressure_val = self._safe_float(self.app._extract_pressure_value(status))
            now = time.time()
            if pressure_val is not None:
                samples.append((now, pressure_val))
                cutoff = now - window_s
                while samples and samples[0][0] < cutoff:
                    samples.popleft()
                if samples and samples[-1][0] - samples[0][0] >= window_s * 0.95:
                    values = [val for _, val in samples]
                    fluct = max(values) - min(values)
                    if fluct <= fluct_threshold:
                        self.log(
                            f"压力波动 {fluct:.3f} 满足阈值，开始当前温度的压力列表"
                        )
                        return status, False
                    if now - last_report >= 5.0:
                        self.log(
                            f"当前压力波动 {fluct:.3f} 超过阈值 {fluct_threshold:.3f}，继续等待"
                        )
                        last_report = now
            if timeout is not None and now - start_time >= timeout:
                self.log("温度切换后的压力判稳超时")
                return None, True
            time.sleep(1.0)
        return None, False

    def _record_result(
        self,
        row: int,
        col: int,
        temperature: float,
        current: float,
        temp_status: Dict,
        pressure_status: Dict,
    ) -> None:
        result = {
            "timestamp": time.time(),
            "row": row,
            "col": col,
            "temperature": temperature,
            "current": current,
            "pressure": current,  # 按原始字段保留
            "temp_status": temp_status,
            "pressure_status": pressure_status,
            "status": "completed",
        }
        self.app.add_result(result)
        self.app.append_csv_record(result)

    def _capture_pixel_baseline(self) -> bool:
        if pyautogui is None or not self.pixel_region:
            return False
        try:
            snap = pyautogui.screenshot(region=self.pixel_region)
            if snap.mode != "RGB":
                snap = snap.convert("RGB")
            self.pixel_base = snap
            self.pixel_prev = None
            return True
        except Exception as exc:
            self.log(f"基线截屏失败：{exc}")
            return False

    def _check_pixel_change(self) -> Tuple[bool, str]:
        if not self.pixel_region:
            return False, "未设置检测区域"
        if pyautogui is None:
            return False, "未安装 pyautogui，无法截屏"
        if self.pixel_base is None:
            if not self._capture_pixel_baseline():
                return False, "尚未截取基线截图，无法检测"
        try:
            curr = pyautogui.screenshot(region=self.pixel_region)
            if curr.mode != "RGB":
                curr = curr.convert("RGB")
            base = self.pixel_prev or self.pixel_base
            if base is None:
                base = curr
            diff = ImageChops.difference(base, curr)
            self.pixel_prev = curr
            diff_data = diff.getdata()
            diff_pixels = sum(1 for px in diff_data if px != (0, 0, 0))
            total = curr.width * curr.height
            pct = diff_pixels / total * 100 if total else 0
            if pct >= self.pixel_sensitivity:
                return True, f"检测到像素变化: {pct:.2f}%"
            return False, f"无显著变化: {pct:.2f}%"
        except Exception as exc:
            return False, f"检测失败：{exc}"

    def _wait_for_pixel_change(self) -> Tuple[bool, str]:
        if not self.pixel_region:
            return False, "未设置检测区域"
        if not self._capture_pixel_baseline():
            return False, "无法截取基线，跳过像素检测"
        end_time = time.time() + max(0.0, self.pixel_timeout)
        while not self.stop_event.is_set() and time.time() <= end_time:
            changed, msg = self._check_pixel_change()
            if self.pixel_log_enabled:
                self.log(msg)
            if changed:
                self.app.after(0, lambda: self.app.set_indicator_color("green"))
                return True, msg
            self.stop_event.wait(self.pixel_interval)
        return False, "像素检测超时"

    def _maybe_sim_click(self) -> None:
        auto_cfg = self.plan.get("auto_test") or {}
        cfg = auto_cfg.get("sim_click") or {}
        pixel_cfg = auto_cfg.get("pixel") or {}
        if not isinstance(cfg, dict) or not cfg.get("enabled"):
            return
        point = cfg.get("point")
        if not point or len(point) != 2:
            self.log("模拟点击点未设置，跳过点击")
            return
        try:
            repeat = max(1, int(cfg.get("repeat", 1)))
        except Exception:
            repeat = 1
        try:
            delay_ms = max(0, int(cfg.get("delay_ms", 0)))
        except Exception:
            delay_ms = 0
        try:
            cooldown_ms = max(0, int(cfg.get("cooldown_ms", 0)))
        except Exception:
            cooldown_ms = 0
        now = time.time()
        if cooldown_ms and now - self._last_click_ts < cooldown_ms / 1000.0:
            self.log("模拟点击冷却中，跳过本轮触发")
            return
        for idx in range(repeat):
            perform_click_async(int(point[0]), int(point[1]), delay_ms=delay_ms, reporter=self.log)
            if cooldown_ms and idx < repeat - 1:
                if self.stop_event.wait(cooldown_ms / 1000.0):
                    break
        self._last_click_ts = time.time()

        if pixel_cfg.get("enabled"):
            if not self.pixel_region:
                self.log("已启用像素检测但未设置区域，跳过")
                return
            ok, msg = self._wait_for_pixel_change()
            if ok:
                self.log(f"像素检测完成：{msg}")
            else:
                self.log(msg)

    def run(self) -> None:
        temp_client = ControllerClient(
            self.plan["temp_host"],
            self.plan["temp_port"],
            "温控",
            handler=self.temp_handler,
            lock=self.temp_lock,
        )
        pressure_client = ControllerClient(
            self.plan["press_host"],
            self.plan["press_port"],
            "压力",
            handler=self.press_handler,
            lock=self.press_lock,
        )
        try:
            temp_client.ping()
            pressure_client.ping()
        except Exception as exc:
            self.log(f"连接设备失败: {exc}")
            self.app.after(0, self.app.on_runner_finished)
            return

        temps = self.plan["temperatures"]
        currents = self.plan["pressures"]
        timeout = self.plan.get("cell_timeout", DEFAULT_CELL_TIMEOUT)
        t_thr = self.plan.get("temp_mae", DEFAULT_TEMP_MAE)
        t_hold = self.plan.get("temp_hold", DEFAULT_TEMP_HOLD)
        p_thr = self.plan.get("press_mae", DEFAULT_PRESS_MAE)
        p_hold = self.plan.get("press_hold", DEFAULT_PRESS_HOLD)
        confirm_need = int(self.plan.get("repeat_confirm", DEFAULT_CONFIRM_COUNT))
        recheck_hold = float(self.plan.get("temp_recheck_hold", DEFAULT_TEMP_RECHECK_HOLD))
        press_window = float(self.plan.get("press_settle_window", DEFAULT_PRESS_STABLE_WINDOW))
        press_fluct = float(self.plan.get("press_settle_range", DEFAULT_PRESS_FLUCT_RANGE))

        try:
            for col, temp_target in enumerate(temps):
                if self.stop_event.is_set():
                    break
                self.log(f"设定温度 {temp_target}°C，等待稳定")
                if not self.ensure_temperature_running(temp_client, temp_target):
                    self.app.mark_column_error(col, "温控失败")
                    return
                temp_status, temp_timeout = self.wait_temperature(
                    temp_client,
                    t_thr,
                    t_hold,
                    timeout=timeout,
                    confirm_need=confirm_need,
                    context="温度",
                )
                if temp_status is None:
                    if temp_timeout:
                        self.app.mark_column_timeout(col, "温度超时")
                        continue
                    break
                settle_status, settle_timeout = self.wait_pressure_stable_after_temp(
                    pressure_client,
                    press_window,
                    press_fluct,
                    timeout=timeout,
                )
                if settle_status is None:
                    if settle_timeout:
                        self.app.mark_column_timeout(col, "温度后压力不稳")
                        continue
                    break
                for row, current_target in enumerate(currents):
                    if self.stop_event.is_set():
                        break
                    self.app.mark_cell_in_progress(row, col)
                    self.log(f"设定电流 {current_target}，等待稳定")
                    if not self.ensure_pressure_running(pressure_client, current_target):
                        self.app.mark_cell_error(row, col, "压力失败")
                        continue
                    press_status, press_timeout = self.wait_pressure(
                        pressure_client,
                        p_thr,
                        p_hold,
                        timeout=timeout,
                        confirm_need=confirm_need,
                        context="压力",
                    )
                    if press_status is None:
                        if press_timeout:
                            self.app.mark_cell_timeout(row, col, "压力超时")
                            continue
                        break
                    self._maybe_sim_click()
                    temp_recheck, temp_recheck_timeout = self.wait_temperature(
                        temp_client,
                        t_thr,
                        recheck_hold,
                        timeout=timeout,
                        confirm_need=confirm_need,
                        context="温度复检",
                    )
                    if temp_recheck is None:
                        if temp_recheck_timeout:
                            self.app.mark_cell_timeout(row, col, "温度复检超时")
                            continue
                        break
                    self._record_result(row, col, temp_target, current_target, temp_recheck, press_status)
                    self.app.mark_cell_done(row, col)
        finally:
            try:
                pressure_client.stop_all()
            except Exception:
                pass
            try:
                temp_client.stop_ramp()
            except Exception:
                pass
            try:
                temp_client.stop_pid()
            except Exception:
                pass
            self.app.after(0, self.app.on_runner_finished)


class MultiSequenceApp(ttk.Frame):
    def __init__(
        self,
        master: Optional[tk.Misc] = None,
        temp_controller: Optional[Any] = None,
        pressure_controller: Optional[Any] = None,
    ) -> None:
        if master is None:
            window = ttk.Window(themename="cosmo")
            master_widget: tk.Misc = window
            self._owns_window = True
        else:
            master_widget = master
            if isinstance(master, (tk.Tk, tk.Toplevel, ttk.Window)):
                window = master  # type: ignore[assignment]
            else:
                window = master.winfo_toplevel()
            self._owns_window = False

        super().__init__(master_widget)
        self.pack(fill=tk.BOTH, expand=True)

        self._window = window
        if hasattr(self._window, "title"):
            self._window.title("冷热平台多点自动测试")
        if hasattr(self._window, "geometry"):
            self._window.geometry("960x720")
        if hasattr(self._window, "protocol"):
            self._window.protocol("WM_DELETE_WINDOW", self.on_close)

        self._pressure_controller = pressure_controller
        self._temp_handler = getattr(temp_controller, "_handle_tcp_command", None)
        self._press_handler = getattr(pressure_controller, "_handle_tcp_command", None)
        self._temp_handler_lock = threading.RLock() if self._temp_handler else None
        self._press_handler_lock = threading.RLock() if self._press_handler else None

        self.temperature_vars: List[tk.StringVar] = []
        self.current_vars: List[tk.StringVar] = []
        self.cell_labels: Dict[Tuple[int, int], ttk.Label] = {}
        self.cell_states: Dict[Tuple[int, int], Tuple[str, str]] = {}

        self.cell_timeout_var = tk.DoubleVar(value=DEFAULT_CELL_TIMEOUT)
        self.confirm_count_var = tk.IntVar(value=DEFAULT_CONFIRM_COUNT)
        self.press_settle_window_var = tk.DoubleVar(value=DEFAULT_PRESS_STABLE_WINDOW)
        self.press_settle_range_var = tk.DoubleVar(
            value=self._get_pressure_tolerance(DEFAULT_PRESS_FLUCT_RANGE)
        )

        self.temp_live_var = tk.StringVar(value="--")
        self.temp_target_var = tk.StringVar(value="--")
        self.temp_error_var = tk.StringVar(value="--")
        self.press_live_var = tk.StringVar(value="--")
        self.press_target_var = tk.StringVar(value="--")
        self.press_error_var = tk.StringVar(value="--")
        self.realtime_timestamp_var = tk.StringVar(value="--")
        self.realtime_status_var = tk.StringVar(value="")
        self.temp_conn_status = tk.StringVar(value="未检查")
        self.press_conn_status = tk.StringVar(value="未检查")

        self.csv_dir = tk.StringVar(value=str(get_logs_dir()))
        self._csv_lock = threading.Lock()
        self._csv_file: Optional[Any] = None
        self._csv_writer: Optional[csv.writer] = None
        self._csv_path: Optional[Path] = None
        self._csv_has_data = False

        self.sim_click_enabled = tk.BooleanVar(value=False)
        self.sim_click_delay_var = tk.IntVar(value=0)
        self.sim_click_repeat_var = tk.IntVar(value=2)
        self.sim_click_cooldown_var = tk.IntVar(value=500)
        self.sim_click_pos: Optional[Tuple[int, int]] = None
        self.sim_click_display_var = tk.StringVar(value="未设置")

        self.pixel_detection_enabled = tk.BooleanVar(value=False)
        self.pixel_log_enabled = tk.BooleanVar(value=False)
        self.pixel_sensitivity_var = tk.DoubleVar(value=5.0)
        self.pixel_timeout_var = tk.DoubleVar(value=180.0)
        self.detection_interval_var = tk.IntVar(value=1000)
        self.pixel_detection_region: Optional[Tuple[int, int, int, int]] = None
        self.initial_pixel_snapshot = None
        self.prev_pixel_snapshot = None

        self._external_log: Optional[Callable[[str], None]] = None
        self._use_internal_log = self._owns_window
        self.log_text: Optional[scrolledtext.ScrolledText] = None

        self.runner: Optional[SequenceRunner] = None
        self.results: List[Dict] = []

        self._chart_history_s = CHART_HISTORY_SECONDS
        self._temp_history: Deque[Tuple[float, float, float]] = deque()
        self._press_history: Deque[Tuple[float, float, float]] = deque()
        self.chart_canvas: Optional[FigureCanvasTkAgg] = None
        self._chart_fig: Optional[Figure] = None
        self._temp_line_actual = None
        self._temp_line_target = None
        self._press_line_actual = None
        self._press_line_target = None

        self._rt_stop = threading.Event()
        self._rt_lock = threading.Lock()
        self._rt_latest: Optional[Tuple[Optional[Dict], Optional[Dict], float]] = None
        self._rt_thread: Optional[threading.Thread] = None

        self._build_ui()
        self._refresh_charts()
        self._init_default_matrix()
        self._start_realtime_monitor()
        self.refresh_controller_status()

    # UI -----------------------------------------------------------------
    def _build_ui(self) -> None:
        outer = ttk.Frame(self)
        outer.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(outer, highlightthickness=0)
        scrollbar = ttk.Scrollbar(outer, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        main = ttk.Frame(canvas, padding=12)
        window_id = canvas.create_window((0, 0), window=main, anchor="nw")

        def _sync_scroll_region(event: tk.Event) -> None:  # type: ignore[type-arg]
            canvas.configure(scrollregion=canvas.bbox("all"))
            canvas.itemconfigure(window_id, width=canvas.winfo_width())

        main.bind("<Configure>", _sync_scroll_region)
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(window_id, width=e.width))
        self._bind_mousewheel(canvas)

        matrix_frame = ttk.Labelframe(main, text="测试矩阵（列：温度 °C / 行：电流）")
        matrix_frame.pack(fill=tk.X, pady=8)

        toolbar = ttk.Frame(matrix_frame)
        toolbar.pack(fill=tk.X, pady=4)
        ttk.Button(toolbar, text="添加温度列", command=self.add_temperature_column, bootstyle="outline-info").pack(
            side=tk.LEFT, padx=4
        )
        ttk.Button(toolbar, text="删除温度列", command=self.remove_temperature_column, bootstyle="outline-secondary").pack(
            side=tk.LEFT, padx=4
        )
        ttk.Button(toolbar, text="添加电流行", command=self.add_current_row, bootstyle="outline-info").pack(
            side=tk.LEFT, padx=12
        )
        ttk.Button(toolbar, text="删除电流行", command=self.remove_current_row, bootstyle="outline-secondary").pack(
            side=tk.LEFT, padx=4
        )
        ttk.Button(toolbar, text="重置状态", command=self.reset_matrix_statuses, bootstyle="outline-warning").pack(
            side=tk.RIGHT, padx=4
        )

        self.matrix_grid = ttk.Frame(matrix_frame)
        self.matrix_grid.pack(fill=tk.X, pady=4)

        matrix_actions = ttk.Frame(matrix_frame)
        matrix_actions.pack(fill=tk.X, pady=4)
        ttk.Button(matrix_actions, text="开始执行", bootstyle=SUCCESS, command=self.start_plan).pack(side=tk.LEFT, padx=4)
        ttk.Button(matrix_actions, text="停止", bootstyle=DANGER, command=self.stop_plan).pack(side=tk.LEFT, padx=4)
        ttk.Button(matrix_actions, text="导出结果", command=self.export_results, bootstyle="outline-primary").pack(
            side=tk.RIGHT, padx=4
        )

        options_frame = ttk.Labelframe(main, text="执行设置")
        options_frame.pack(fill=tk.X, pady=8)
        opt_row = ttk.Frame(options_frame)
        opt_row.pack(fill=tk.X, pady=4)
        ttk.Label(opt_row, text="单点超时时间 (s)").pack(side=tk.LEFT, padx=4)
        ttk.Entry(opt_row, textvariable=self.cell_timeout_var, width=8).pack(side=tk.LEFT, padx=4)
        ttk.Label(opt_row, text="稳定重复确认次数").pack(side=tk.LEFT, padx=(12, 4))
        ttk.Entry(opt_row, textvariable=self.confirm_count_var, width=6).pack(side=tk.LEFT, padx=4)
        ttk.Label(opt_row, text="说明：采用固定判稳阈值自动复检，超时将标记为红色并跳过。", bootstyle=INFO).pack(
            side=tk.LEFT, padx=8
        )

        press_precheck_row = ttk.Frame(options_frame)
        press_precheck_row.pack(fill=tk.X, pady=4)
        ttk.Label(press_precheck_row, text="温度切换后压力稳定时间 (s)").pack(side=tk.LEFT, padx=4)
        ttk.Entry(press_precheck_row, textvariable=self.press_settle_window_var, width=8).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Label(press_precheck_row, text="压力波动阈值").pack(side=tk.LEFT, padx=(12, 4))
        ttk.Entry(press_precheck_row, textvariable=self.press_settle_range_var, width=10).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Label(
            press_precheck_row,
            text="温度变化后，只有在设定时间内压力波动低于阈值才开始压力列表。",
            bootstyle=INFO,
        ).pack(side=tk.LEFT, padx=8)

        record_frame = ttk.Labelframe(main, text="数据记录（CSV 导出）")
        record_frame.pack(fill=tk.X, pady=8)
        record_row = ttk.Frame(record_frame)
        record_row.pack(fill=tk.X, pady=4)
        ttk.Label(record_row, text="保存目录").pack(side=tk.LEFT, padx=4)
        ttk.Entry(record_row, textvariable=self.csv_dir, width=48).pack(side=tk.LEFT, padx=6)
        ttk.Button(
            record_row,
            text="选择目录",
            command=self._choose_csv_dir,
            bootstyle="outline-secondary",
        ).pack(side=tk.LEFT, padx=4)
        ttk.Label(
            record_frame,
            text="规则：点击“开始执行”新建文件，任务结束后自动保存。",
            bootstyle=INFO,
            wraplength=680,
            justify=tk.LEFT,
        ).pack(anchor=tk.W, padx=4, pady=(0, 2))

        auto_frame = ttk.Labelframe(options_frame, text="自动测试")
        auto_frame.pack(fill=tk.X, pady=6)

        pixel_row = ttk.Frame(auto_frame)
        pixel_row.pack(fill=tk.X, pady=2)
        ttk.Checkbutton(pixel_row, text="启用像素检测", variable=self.pixel_detection_enabled).pack(side=tk.LEFT, padx=4)
        ttk.Button(pixel_row, text="设置检测区域", command=self.set_pixel_detection_region, bootstyle="outline-info").pack(
            side=tk.LEFT, padx=4
        )
        ttk.Checkbutton(pixel_row, text="记录检测日志", variable=self.pixel_log_enabled).pack(side=tk.LEFT, padx=6)
        self.indicator_canvas = tk.Canvas(pixel_row, width=18, height=18, highlightthickness=0)
        self.indicator_circle = self.indicator_canvas.create_oval(2, 2, 16, 16, fill="red")
        self.indicator_canvas.pack(side=tk.LEFT, padx=6)
        ttk.Label(pixel_row, text="灵敏度(%)").pack(side=tk.LEFT, padx=(8, 4))
        ttk.Entry(pixel_row, textvariable=self.pixel_sensitivity_var, width=8).pack(side=tk.LEFT)
        ttk.Label(pixel_row, text="超时(s)").pack(side=tk.LEFT, padx=(8, 4))
        ttk.Entry(pixel_row, textvariable=self.pixel_timeout_var, width=8).pack(side=tk.LEFT)

        interval_row = ttk.Frame(auto_frame)
        interval_row.pack(fill=tk.X, pady=2)
        ttk.Label(interval_row, text="检测间隔(ms)").pack(side=tk.LEFT, padx=4)
        ttk.Entry(interval_row, textvariable=self.detection_interval_var, width=10).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Label(interval_row, text="点击冷却(ms)").pack(side=tk.LEFT, padx=4)
        ttk.Entry(interval_row, textvariable=self.sim_click_cooldown_var, width=10).pack(side=tk.LEFT)

        click_row = ttk.Frame(auto_frame)
        click_row.pack(fill=tk.X, pady=2)
        ttk.Checkbutton(click_row, text="稳定后执行模拟点击", variable=self.sim_click_enabled).pack(side=tk.LEFT, padx=4)
        ttk.Button(click_row, text="设置点击点", command=self.set_sim_click_point, bootstyle="outline-info").pack(
            side=tk.LEFT, padx=4
        )
        ttk.Label(click_row, textvariable=self.sim_click_display_var, bootstyle=SECONDARY).pack(side=tk.LEFT, padx=6)
        ttk.Label(click_row, text="延时(ms)").pack(side=tk.LEFT, padx=(12, 4))
        ttk.Entry(click_row, textvariable=self.sim_click_delay_var, width=7).pack(side=tk.LEFT)
        ttk.Label(click_row, text="重复次数").pack(side=tk.LEFT, padx=(12, 4))
        ttk.Entry(click_row, textvariable=self.sim_click_repeat_var, width=5).pack(side=tk.LEFT)

        realtime_frame = ttk.Labelframe(main, text="实时状态（来自温度/压力控制程序）")
        realtime_frame.pack(fill=tk.X, pady=8)

        rt_row1 = ttk.Frame(realtime_frame)
        rt_row1.pack(fill=tk.X, pady=2)
        ttk.Label(rt_row1, text="温度 当前/目标 (°C)：").pack(side=tk.LEFT, padx=4)
        ttk.Label(rt_row1, textvariable=self.temp_live_var, width=10, anchor=tk.E).pack(side=tk.LEFT)
        ttk.Label(rt_row1, text="→").pack(side=tk.LEFT, padx=2)
        ttk.Label(rt_row1, textvariable=self.temp_target_var, width=10, anchor=tk.E).pack(side=tk.LEFT)
        ttk.Label(rt_row1, text="误差：").pack(side=tk.LEFT, padx=(12, 2))
        ttk.Label(rt_row1, textvariable=self.temp_error_var, width=10, anchor=tk.E).pack(side=tk.LEFT)

        rt_row2 = ttk.Frame(realtime_frame)
        rt_row2.pack(fill=tk.X, pady=2)
        ttk.Label(rt_row2, text="压力/电流 当前/目标：").pack(side=tk.LEFT, padx=4)
        ttk.Label(rt_row2, textvariable=self.press_live_var, width=10, anchor=tk.E).pack(side=tk.LEFT)
        ttk.Label(rt_row2, text="→").pack(side=tk.LEFT, padx=2)
        ttk.Label(rt_row2, textvariable=self.press_target_var, width=10, anchor=tk.E).pack(side=tk.LEFT)
        ttk.Label(rt_row2, text="误差：").pack(side=tk.LEFT, padx=(12, 2))
        ttk.Label(rt_row2, textvariable=self.press_error_var, width=10, anchor=tk.E).pack(side=tk.LEFT)

        rt_row3 = ttk.Frame(realtime_frame)
        rt_row3.pack(fill=tk.X, pady=2)
        ttk.Label(rt_row3, text="最近更新时间：").pack(side=tk.LEFT, padx=4)
        ttk.Label(rt_row3, textvariable=self.realtime_timestamp_var, width=18).pack(side=tk.LEFT)
        ttk.Label(rt_row3, textvariable=self.realtime_status_var, bootstyle=WARNING).pack(side=tk.LEFT, padx=8)

        rt_row4 = ttk.Frame(realtime_frame)
        rt_row4.pack(fill=tk.X, pady=2)
        ttk.Label(rt_row4, text="连接状态：").pack(side=tk.LEFT, padx=4)
        self.temp_conn_label = ttk.Label(rt_row4, textvariable=self.temp_conn_status, bootstyle=SECONDARY)
        self.temp_conn_label.pack(side=tk.LEFT, padx=6)
        self.press_conn_label = ttk.Label(rt_row4, textvariable=self.press_conn_status, bootstyle=SECONDARY)
        self.press_conn_label.pack(side=tk.LEFT, padx=6)
        ttk.Button(rt_row4, text="刷新连接", command=self.refresh_controller_status, bootstyle="outline-secondary").pack(
            side=tk.LEFT, padx=(12, 0)
        )

        charts_frame = ttk.Labelframe(main, text="Live Charts (Last 5 Minutes)")
        charts_frame.pack(fill=tk.BOTH, expand=True, pady=8)


        self._chart_fig = Figure(figsize=(10, 4.5), dpi=100)
        grid = self._chart_fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.28)
        ax_temp = self._chart_fig.add_subplot(grid[0, 0])
        ax_press = self._chart_fig.add_subplot(grid[0, 1])

        ax_temp.set_title("Temperature Over Time")
        ax_temp.set_ylabel("Temperature (°C)")
        ax_temp.set_xlabel("Time (s)")
        self._temp_line_actual, = ax_temp.plot([], [], color="#d62728", linewidth=1.6, label="Actual Temperature")
        self._temp_line_target, = ax_temp.plot(
            [], [], color="#1f77b4", linestyle="--", linewidth=1.2, label="Target Temperature"
        )
        ax_temp.grid(True, alpha=0.3)
        ax_temp.legend(loc="upper right", fontsize=8)

        ax_press.set_title("Pressure Over Time")
        ax_press.set_ylabel("Pressure")
        ax_press.set_xlabel("Time (s)")
        self._press_line_actual, = ax_press.plot([], [], color="#2ca02c", linewidth=1.6, label="Actual Pressure")
        self._press_line_target, = ax_press.plot(
            [], [], color="#ff7f0e", linestyle="--", linewidth=1.2, label="Target Pressure"
        )
        ax_press.grid(True, alpha=0.3)
        ax_press.legend(loc="upper right", fontsize=8)

        self.chart_canvas = FigureCanvasTkAgg(self._chart_fig, master=charts_frame)
        self.chart_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        if self._use_internal_log:
            log_frame = ttk.Labelframe(main, text="日志")
            log_frame.pack(fill=tk.BOTH, expand=True, pady=8)
            self.log_text = scrolledtext.ScrolledText(log_frame, height=15)
            self.log_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

    # Matrix helpers ------------------------------------------------------
    def _init_default_matrix(self) -> None:
        for value in (25, 35, 45):
            self.add_temperature_column(str(value))
        for value in (500, 1000, 1500):
            self.add_current_row(str(value))
        self.reset_matrix_statuses()

    def add_temperature_column(self, value: Optional[str] = None) -> None:
        var = tk.StringVar(value=value or "")
        self.temperature_vars.append(var)
        self._render_matrix_table()

    def remove_temperature_column(self) -> None:
        if not self.temperature_vars:
            return
        self.temperature_vars.pop()
        self._render_matrix_table()

    def add_current_row(self, value: Optional[str] = None) -> None:
        var = tk.StringVar(value=value or "")
        self.current_vars.append(var)
        self._render_matrix_table()

    def remove_current_row(self) -> None:
        if not self.current_vars:
            return
        self.current_vars.pop()
        self._render_matrix_table()

    def _render_matrix_table(self) -> None:
        for widget in self.matrix_grid.winfo_children():
            widget.destroy()
        self.cell_labels.clear()
        current_states = dict(self.cell_states)

        header = ttk.Label(self.matrix_grid, text="电流→温度", anchor=tk.CENTER, width=12, bootstyle=SECONDARY)
        header.grid(row=0, column=0, padx=2, pady=2, sticky=tk.NSEW)

        for col, var in enumerate(self.temperature_vars, start=1):
            entry = ttk.Entry(self.matrix_grid, textvariable=var, width=10)
            entry.grid(row=0, column=col, padx=2, pady=2, sticky=tk.NSEW)

        for row, var in enumerate(self.current_vars, start=1):
            entry = ttk.Entry(self.matrix_grid, textvariable=var, width=10)
            entry.grid(row=row, column=0, padx=2, pady=2, sticky=tk.NSEW)
            for col in range(len(self.temperature_vars)):
                label = ttk.Label(self.matrix_grid, text="待测", width=10, anchor=tk.CENTER, bootstyle=SECONDARY)
                label.grid(row=row, column=col + 1, padx=2, pady=2, sticky=tk.NSEW)
                key = (row - 1, col)
                self.cell_labels[key] = label
                state = current_states.get(key, ("pending", "待测"))
                self._apply_cell_state(key, state[0], state[1])
        self.cell_states = {k: v for k, v in current_states.items() if k in self.cell_labels}

    def _apply_cell_state(self, key: Tuple[int, int], state: str, text: Optional[str] = None) -> None:
        label = self.cell_labels.get(key)
        if not label:
            return
        styles = {
            "pending": ("待测" if text is None else text, "secondary"),
            "running": ("进行中" if text is None else text, "info"),
            "success": ("完成" if text is None else text, "success"),
            "timeout": ("超时" if text is None else text, "danger"),
            "error": ("失败" if text is None else text, "danger"),
        }
        display_text, style = styles.get(state, (text or "?", "secondary"))
        label.configure(text=display_text, bootstyle=style)
        self.cell_states[key] = (state, display_text)

    def reset_matrix_statuses(self) -> None:
        for key in list(self.cell_labels.keys()):
            self._apply_cell_state(key, "pending")

    # Runner integration --------------------------------------------------
    def get_row_count(self) -> int:
        return len(self.current_vars)

    def get_col_count(self) -> int:
        return len(self.temperature_vars)

    def mark_cell_in_progress(self, row: int, col: int) -> None:
        self.after(0, lambda: self._apply_cell_state((row, col), "running"))

    def mark_cell_done(self, row: int, col: int) -> None:
        self.after(0, lambda: self._apply_cell_state((row, col), "success"))

    def mark_cell_timeout(self, row: int, col: int, reason: str) -> None:
        self.log(f"[{row+1}, {col+1}] 超时：{reason}")
        self.after(0, lambda: self._apply_cell_state((row, col), "timeout"))

    def mark_cell_error(self, row: int, col: int, reason: str) -> None:
        self.log(f"[{row+1}, {col+1}] 失败：{reason}")
        self.after(0, lambda: self._apply_cell_state((row, col), "error"))

    def mark_column_error(self, col: int, reason: str) -> None:
        for row in range(self.get_row_count()):
            self.mark_cell_error(row, col, reason)

    def mark_column_timeout(self, col: int, reason: str) -> None:
        for row in range(self.get_row_count()):
            self.mark_cell_timeout(row, col, reason)

    def build_plan(self) -> Dict:
        temps = self._collect_numbers(self.temperature_vars, "温度")
        currents = self._collect_numbers(self.current_vars, "电流")
        if not temps:
            raise ValueError("请至少输入一个温度目标")
        if not currents:
            raise ValueError("请至少输入一个电流目标")
        timeout = max(5.0, float(self.cell_timeout_var.get()))
        confirm_need = max(1, int(self.confirm_count_var.get()))
        try:
            press_window = max(1.0, float(self.press_settle_window_var.get()))
        except Exception:
            press_window = DEFAULT_PRESS_STABLE_WINDOW
        try:
            press_fluct_range = max(0.0, float(self.press_settle_range_var.get()))
        except Exception:
            press_fluct_range = self._get_pressure_tolerance(DEFAULT_PRESS_FLUCT_RANGE)
        press_mae = self._get_pressure_tolerance(DEFAULT_PRESS_MAE)
        plan = {
            "temp_host": TEMP_DEFAULT_HOST,
            "temp_port": TEMP_DEFAULT_PORT,
            "press_host": PRESS_DEFAULT_HOST,
            "press_port": PRESS_DEFAULT_PORT,
            "temperatures": temps,
            "pressures": currents,
            "cell_timeout": timeout,
            "temp_mae": DEFAULT_TEMP_MAE,
            "temp_hold": DEFAULT_TEMP_HOLD,
            "temp_recheck_hold": DEFAULT_TEMP_RECHECK_HOLD,
            "press_mae": press_mae,
            "press_hold": DEFAULT_PRESS_HOLD,
            "repeat_confirm": confirm_need,
            "press_settle_window": press_window,
            "press_settle_range": press_fluct_range,
        }
        try:
            delay_ms = max(0, int(self.sim_click_delay_var.get()))
        except Exception:
            delay_ms = 0
        try:
            repeat = max(1, int(self.sim_click_repeat_var.get()))
        except Exception:
            repeat = 1
        try:
            cooldown = max(0, int(self.sim_click_cooldown_var.get()))
        except Exception:
            cooldown = 0
        sim_enabled = bool(self.sim_click_enabled.get())
        if sim_enabled and not self.sim_click_pos:
            raise ValueError("请先设置模拟点击点")
        auto_test = {
            "sim_click": {
                "enabled": sim_enabled,
                "point": list(self.sim_click_pos) if self.sim_click_pos else None,
                "delay_ms": delay_ms,
                "repeat": repeat,
                "cooldown_ms": cooldown,
            },
            "pixel": {
                "enabled": bool(self.pixel_detection_enabled.get()),
                "region": list(self.pixel_detection_region) if self.pixel_detection_region else None,
                "sensitivity": float(self.pixel_sensitivity_var.get()),
                "timeout": float(self.pixel_timeout_var.get()),
                "interval_ms": int(self.detection_interval_var.get()),
                "log_enabled": bool(self.pixel_log_enabled.get()),
            },
        }
        if auto_test["pixel"]["enabled"] and not auto_test["pixel"]["region"]:
            raise ValueError("请先设置像素检测区域")
        plan["auto_test"] = auto_test
        plan["sim_click"] = auto_test["sim_click"]
        return plan

    def _preflight_check_controllers(self) -> Tuple[bool, str]:
        temp_ok, temp_msg = self._ping_controller(
            "温控", TEMP_DEFAULT_HOST, TEMP_DEFAULT_PORT, handler=self._temp_handler, lock=self._temp_handler_lock
        )
        press_ok, press_msg = self._ping_controller(
            "压力", PRESS_DEFAULT_HOST, PRESS_DEFAULT_PORT, handler=self._press_handler, lock=self._press_handler_lock
        )
        if temp_ok and press_ok:
            return True, ""
        reasons = []
        if not temp_ok:
            reasons.append(f"温控连接失败：{temp_msg}")
        if not press_ok:
            reasons.append(f"压力连接失败：{press_msg}")
        return False, "；".join(reasons)

    def _collect_numbers(self, vars_list: List[tk.StringVar], name: str) -> List[float]:
        values: List[float] = []
        for idx, var in enumerate(vars_list, start=1):
            text = var.get().strip()
            if not text:
                raise ValueError(f"{name}第 {idx} 项为空")
            try:
                values.append(float(text))
            except ValueError:
                raise ValueError(f"{name}第 {idx} 项格式错误：{text}") from None
        return values

    def _get_pressure_tolerance(self, fallback: float) -> float:
        controller = getattr(self, "_pressure_controller", None)
        if controller is None:
            return float(fallback)
        candidates = []
        try:
            tol_attr = getattr(controller, "pressure_tolerance", None)
            if tol_attr is not None:
                candidates.append(tol_attr)
        except Exception:
            pass
        try:
            tol_var = getattr(controller, "tolerance_var", None)
            if tol_var is not None:
                candidates.append(tol_var)
        except Exception:
            pass
        for candidate in candidates:
            try:
                if isinstance(candidate, tk.Variable):
                    value = candidate.get()
                else:
                    value = candidate
                return float(value)
            except Exception:
                continue
        return float(fallback)

    def set_sim_click_point(self) -> None:
        pos = capture_click_point(
            self,
            title="设置模拟点击点",
            hint="移动鼠标到目标软件按钮处，按 Enter 键记录",
            reporter=self.log,
        )
        if pos:
            self.sim_click_pos = (int(pos[0]), int(pos[1]))
            self.sim_click_display_var.set(f"{int(pos[0])}, {int(pos[1])}")
        else:
            self.sim_click_display_var.set("未设置")

    def set_indicator_color(self, color: str) -> None:
        try:
            self.indicator_canvas.itemconfig(self.indicator_circle, fill=color)
            if color == "green":
                self.after(600, lambda: self.set_indicator_color("red"))
        except Exception:
            pass

    def _capture_pixel_baseline(self) -> bool:
        if pyautogui is None or not self.pixel_detection_region:
            return False
        try:
            snap = pyautogui.screenshot(region=self.pixel_detection_region)
            if snap.mode != "RGB":
                snap = snap.convert("RGB")
            self.initial_pixel_snapshot = snap
            self.prev_pixel_snapshot = None
            return True
        except Exception as exc:
            self.log(f"基线截图失败：{exc}")
            return False

    def set_pixel_detection_region(self) -> None:
        messagebox.showinfo("设置检测区域", "请用鼠标框选需要检测的屏幕区域，按 Enter 键确认。")
        overlay = tk.Toplevel(self)
        overlay.attributes("-fullscreen", True)
        overlay.attributes("-alpha", 0.3)
        overlay.attributes("-topmost", True)
        overlay.configure(background="gray")
        overlay.grab_set()
        overlay.focus_force()

        canvas = tk.Canvas(overlay, cursor="cross", highlightthickness=0)
        canvas.pack(fill=tk.BOTH, expand=True)
        canvas.focus_set()

        coords = {'x1': 0, 'y1': 0, 'x2': 0, 'y2': 0, 'rect': None}

        def on_press(event: tk.Event) -> None:  # type: ignore[type-arg]
            coords['x1'], coords['y1'] = event.x, event.y
            if coords['rect']:
                canvas.delete(coords['rect'])
            coords['rect'] = canvas.create_rectangle(coords['x1'], coords['y1'], event.x, event.y, outline="red", width=2)

        def on_drag(event: tk.Event) -> None:  # type: ignore[type-arg]
            canvas.coords(coords['rect'], coords['x1'], coords['y1'], event.x, event.y)

        def on_release(event: tk.Event) -> None:  # type: ignore[type-arg]
            coords['x2'], coords['y2'] = event.x, event.y

        def on_enter(event=None):
            overlay.grab_release()
            x1, y1 = coords['x1'], coords['y1']
            x2, y2 = coords['x2'], coords['y2']
            left, right = sorted((x1, x2))
            top, bottom = sorted((y1, y2))
            width = max(0, right - left)
            height = max(0, bottom - top)
            abs_x = overlay.winfo_rootx() + left
            abs_y = overlay.winfo_rooty() + top

            try:
                if width <= 0 or height <= 0:
                    raise ValueError('selected area too small')

                try:
                    screen_w, screen_h = pyautogui.size() if pyautogui else (None, None)
                except Exception:
                    screen_w = screen_h = None

                if screen_w is not None and screen_h is not None:
                    abs_x = max(0, min(abs_x, screen_w - 1))
                    abs_y = max(0, min(abs_y, screen_h - 1))
                    width = min(width, max(0, screen_w - abs_x))
                    height = min(height, max(0, screen_h - abs_y))
                    if width <= 0 or height <= 0:
                        raise ValueError('selected area too small')

                region = (int(abs_x), int(abs_y), int(width), int(height))
                self.pixel_detection_region = region
                if self._capture_pixel_baseline():
                    self.log(f"像素检测区域已设置：{region}")
                else:
                    self.log(f"像素区域已设置：{region}（未能截取基线）")
            finally:
                overlay.destroy()

        canvas.bind("<ButtonPress-1>", on_press)
        canvas.bind("<B1-Motion>", on_drag)
        canvas.bind("<ButtonRelease-1>", on_release)
        canvas.bind("<Return>", on_enter)
        canvas.bind("<Escape>", lambda e=None: overlay.destroy())
        overlay.bind("<Return>", on_enter)
        overlay.bind("<Escape>", lambda e=None: overlay.destroy())

    def _choose_csv_dir(self) -> None:
        current = self.csv_dir.get().strip() or str(get_logs_dir())
        selected = filedialog.askdirectory(initialdir=current, title="选择 CSV 保存目录")
        if selected:
            path = Path(selected)
            try:
                path.mkdir(parents=True, exist_ok=True)
            except Exception as exc:
                messagebox.showerror("错误", f"创建目录失败：{exc}")
                return
            self.csv_dir.set(str(path))
            self.log(f"CSV 保存目录已设置为 {path}")

    def _bind_mousewheel(self, widget: tk.Widget) -> None:
        def _on_mousewheel(event: tk.Event) -> None:  # type: ignore[type-arg]
            delta = 0
            if hasattr(event, "delta") and event.delta:
                delta = -1 * int(event.delta / 120)
            elif getattr(event, "num", None) == 4:
                delta = -1
            elif getattr(event, "num", None) == 5:
                delta = 1
            if delta:
                widget.yview_scroll(delta, "units")

        widget.bind_all("<MouseWheel>", _on_mousewheel, add=True)
        widget.bind_all("<Button-4>", _on_mousewheel, add=True)
        widget.bind_all("<Button-5>", _on_mousewheel, add=True)

    @staticmethod
    def _format_csv_number(value: Optional[float]) -> str:
        if value is None:
            return ""
        try:
            return f"{float(value):.6f}"
        except Exception:
            return ""

    def _open_csv_session(self) -> bool:
        self._close_csv_session(log=False)
        base = self.csv_dir.get().strip()
        if not base:
            base_path = get_logs_dir()
        else:
            base_path = Path(base).expanduser()
        try:
            base_path.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            messagebox.showerror("错误", f"创建目录失败：{exc}")
            return False
        filename = time.strftime("multi_sequence_%Y%m%d_%H%M%S.csv")
        file_path = base_path / filename
        try:
            handle = open(file_path, "w", newline="", encoding="utf-8-sig")
        except Exception as exc:
            messagebox.showerror("错误", f"创建 CSV 文件失败：{exc}")
            return False
        writer = csv.writer(handle)
        header = [
            "timestamp",
            "row",
            "column",
            "temperature_plan",
            "temperature_target",
            "temperature_actual",
            "temperature_error",
            "temperature_mae",
            "pressure_plan",
            "pressure_target",
            "pressure_actual",
            "pressure_error",
            "pressure_mae",
        ]
        try:
            writer.writerow(header)
            handle.flush()
        except Exception as exc:
            handle.close()
            messagebox.showerror("错误", f"写入 CSV 失败：{exc}")
            return False
        with self._csv_lock:
            self._csv_file = handle
            self._csv_writer = writer
            self._csv_path = file_path
            self._csv_has_data = False
        self.log(f"数据记录开始：{file_path}")
        return True

    def _close_csv_session(self, *, log: bool = True) -> None:
        with self._csv_lock:
            handle = self._csv_file
            path = self._csv_path
            has_data = self._csv_has_data
            self._csv_file = None
            self._csv_writer = None
            self._csv_path = None
            self._csv_has_data = False
        if not handle:
            return
        try:
            handle.flush()
        except Exception:
            pass
        try:
            handle.close()
        except Exception:
            pass
        if not path:
            return
        if not has_data:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            except Exception as exc:
                if log:
                    self.log(f"CSV 文件删除失败：{exc}")
            else:
                if log:
                    self.log("测试未产生有效数据，已删除空的 CSV 文件")
        elif log:
            self.log(f"数据记录已保存到 {path}")

    def append_csv_record(self, result: Dict[str, Any]) -> None:
        with self._csv_lock:
            writer = self._csv_writer
            handle = self._csv_file
            path = self._csv_path
        if not writer or not handle or not path:
            return
        timestamp = result.get("timestamp")
        if not isinstance(timestamp, (int, float)):
            timestamp = time.time()
        timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
        row_idx = result.get("row")
        col_idx = result.get("col")
        if isinstance(row_idx, int):
            row_display: Any = row_idx + 1
        else:
            row_display = ""
        if isinstance(col_idx, int):
            col_display: Any = col_idx + 1
        else:
            col_display = ""

        temp_plan = self._coerce_float(result.get("temperature"))
        temp_status = result.get("temp_status") or {}
        temp_target = self._coerce_float(temp_status.get("target"))
        temp_actual = self._coerce_float(temp_status.get("temperature"))
        temp_error = self._coerce_float(temp_status.get("error"))
        temp_mae = self._coerce_float(temp_status.get("mae"))

        press_plan = self._coerce_float(result.get("pressure", result.get("current")))
        press_status = result.get("pressure_status") or {}
        press_target = self._coerce_float(press_status.get("target"))
        press_actual = None
        if isinstance(press_status, dict):
            feedback_value = self._extract_pressure_value(press_status)
            if feedback_value is None:
                feedback_value = press_status.get("pressure")
            press_actual = self._coerce_float(feedback_value)
        press_error = None
        if isinstance(press_status, dict):
            last = press_status.get("last") if isinstance(press_status.get("last"), dict) else None
            if last and last.get("error") is not None:
                press_error = self._coerce_float(last.get("error"))
            elif press_status.get("error") is not None:
                press_error = self._coerce_float(press_status.get("error"))
        press_mae = self._coerce_float(press_status.get("mae"))

        row_data = [
            timestamp_str,
            row_display,
            col_display,
            self._format_csv_number(temp_plan),
            self._format_csv_number(temp_target),
            self._format_csv_number(temp_actual),
            self._format_csv_number(temp_error),
            self._format_csv_number(temp_mae),
            self._format_csv_number(press_plan),
            self._format_csv_number(press_target),
            self._format_csv_number(press_actual),
            self._format_csv_number(press_error),
            self._format_csv_number(press_mae),
        ]

        with self._csv_lock:
            writer = self._csv_writer
            handle = self._csv_file
            if not writer or not handle:
                return
            try:
                writer.writerow(row_data)
                handle.flush()
                self._csv_has_data = True
            except Exception as exc:
                # 记录错误但不中断执行
                err_msg = f"写入 CSV 失败：{exc}"
                self.after(0, lambda m=err_msg: self.log(m))

    def start_plan(self) -> None:
        if self.runner and self.runner.is_alive():
            messagebox.showwarning("提示", "任务正在运行")
            return
        try:
            plan = self.build_plan()
        except Exception as exc:
            messagebox.showerror("参数错误", str(exc))
            return
        ok, reason = self._preflight_check_controllers()
        if not ok:
            messagebox.showerror("连接失败", reason)
            return
        if not self._open_csv_session():
            return
        self.results.clear()
        if self.log_text:
            self.log_text.delete("1.0", tk.END)
        self.reset_matrix_statuses()
        self.log("开始执行测试计划…")
        self.runner = SequenceRunner(
            self,
            plan,
            temp_handler=self._temp_handler,
            press_handler=self._press_handler,
            temp_lock=self._temp_handler_lock,
            press_lock=self._press_handler_lock,
        )
        self.runner.start()

    def stop_plan(self) -> None:
        if self.runner:
            self.runner.stop()
            self.log("已发送停止指令，等待当前步骤完成…")

    def on_runner_finished(self) -> None:
        self.runner = None
        self.log("测试计划已结束")
        self._close_csv_session()

    def export_results(self) -> None:
        if not self.results:
            messagebox.showinfo("提示", "暂无数据可导出")
            return
        timestamp = time.strftime("matrix_result_%Y%m%d_%H%M%S.json")
        with open(timestamp, "w", encoding="utf-8") as fh:
            json.dump(self.results, fh, ensure_ascii=False, indent=2)
        messagebox.showinfo("导出完成", f"结果已保存到 {timestamp}")

    # Logging -------------------------------------------------------------
    def log(self, message: str) -> None:
        if callable(self._external_log):
            try:
                self._external_log(message)
            except Exception:
                pass
        if not self._use_internal_log or not self.log_text:
            return

        timestamp = time.strftime("%H:%M:%S")

        def append() -> None:
            self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
            self.log_text.see(tk.END)

        self.after(0, append)

    def set_external_logger(self, callback: Callable[[str], None]) -> None:
        self._external_log = callback

    def add_result(self, result: Dict) -> None:
        self.results.append(result)
        pressure_error = result.get("pressure_status", {}).get("last", {}).get("error")
        temp_error = result.get("temp_status", {}).get("error")
        self.log(
            f"记录结果 行{result['row']+1}/列{result['col']+1}：温度={result['temperature']}, 电流={result['current']}, "
            f"温度误差={temp_error}, 压力误差={pressure_error}"
        )

    # Real-time monitor ---------------------------------------------------
    def _start_realtime_monitor(self) -> None:
        if self._rt_thread and self._rt_thread.is_alive():
            return

        def worker() -> None:
            while not self._rt_stop.is_set():
                temp_status = self._fetch_status(
                    TEMP_DEFAULT_HOST,
                    TEMP_DEFAULT_PORT,
                    "温控",
                    handler=self._temp_handler,
                    lock=self._temp_handler_lock,
                )
                press_status = self._fetch_status(
                    PRESS_DEFAULT_HOST,
                    PRESS_DEFAULT_PORT,
                    "压力",
                    handler=self._press_handler,
                    lock=self._press_handler_lock,
                )
                with self._rt_lock:
                    self._rt_latest = (temp_status, press_status, time.time())
                if self._rt_stop.wait(REALTIME_REFRESH_S):
                    break

        self._rt_thread = threading.Thread(target=worker, daemon=True)
        self._rt_thread.start()
        self.after(500, self._update_realtime_ui)

    def refresh_controller_status(self) -> None:
        threading.Thread(target=self._refresh_controller_status_worker, daemon=True).start()

    def _refresh_controller_status_worker(self) -> None:
        temp_ok, temp_msg = self._ping_controller(
            "温控", TEMP_DEFAULT_HOST, TEMP_DEFAULT_PORT, handler=self._temp_handler, lock=self._temp_handler_lock
        )
        press_ok, press_msg = self._ping_controller(
            "压力", PRESS_DEFAULT_HOST, PRESS_DEFAULT_PORT, handler=self._press_handler, lock=self._press_handler_lock
        )

        def update_labels() -> None:
            self.temp_conn_status.set("温控正常" if temp_ok else f"温控异常：{temp_msg}")
            self.press_conn_status.set("压力正常" if press_ok else f"压力异常：{press_msg}")
            self.temp_conn_label.configure(bootstyle=SUCCESS if temp_ok else DANGER)
            self.press_conn_label.configure(bootstyle=SUCCESS if press_ok else DANGER)

        self.after(0, update_labels)

    def _fetch_status(
        self,
        host: str,
        port: int,
        name: str,
        *,
        handler: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        lock: Optional[threading.RLock] = None,
    ) -> Optional[Dict]:
        try:
            client = ControllerClient(host, port, name, handler=handler, lock=lock)
            resp = client.status()
            return resp.get("status")
        except Exception as exc:
            return {"__error__": str(exc)}

    def _ping_controller(
        self,
        name: str,
        host: str,
        port: int,
        *,
        handler: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        lock: Optional[threading.RLock] = None,
    ) -> Tuple[bool, str]:
        try:
            client = ControllerClient(host, port, name, handler=handler, lock=lock)
            client.ping()
            return True, ""
        except Exception as exc:
            return False, str(exc)

    def _update_realtime_ui(self) -> None:
        if self._rt_stop.is_set():
            return
        with self._rt_lock:
            latest = self._rt_latest
        if latest:
            temp_status, press_status, ts = latest
            self._update_temp_display(temp_status)
            self._update_press_display(press_status)
            self._collect_chart_points(temp_status, press_status, ts)
            self._refresh_charts()
            self.realtime_timestamp_var.set(time.strftime("%H:%M:%S", time.localtime(ts)))
            errors = []
            if temp_status and temp_status.get("__error__"):
                errors.append(f"温控：{temp_status['__error__']}")
            if press_status and press_status.get("__error__"):
                errors.append(f"压力：{press_status['__error__']}")
            self.realtime_status_var.set("；".join(errors))
        self.after(int(REALTIME_REFRESH_S * 1000), self._update_realtime_ui)

    def _fmt_value(self, value: Optional[float], unit: str = "") -> str:
        if value is None:
            return "--"
        try:
            return f"{float(value):.3f}{unit}"
        except Exception:
            return "--"

    @staticmethod
    def _coerce_float(value: Any) -> Optional[float]:
        try:
            f = float(value)
        except Exception:
            return None
        if math.isnan(f):
            return None
        return f

    def _append_history(
        self,
        history: Deque[Tuple[float, float, float]],
        timestamp: float,
        actual: float,
        target: float,
    ) -> None:
        if math.isnan(actual) and math.isnan(target):
            return
        history.append((timestamp, actual, target))
        cutoff = timestamp - float(self._chart_history_s)
        while history and history[0][0] < cutoff:
            history.popleft()

    @staticmethod
    def _extract_pressure_value(status: Dict[str, Any]) -> Optional[float]:
        last = status.get("last")
        if isinstance(last, dict):
            for key in ("feedback", "pressure", "value"):
                if last.get(key) is not None:
                    return last.get(key)
        return status.get("pressure")

    def _collect_chart_points(
        self,
        temp_status: Optional[Dict[str, Any]],
        press_status: Optional[Dict[str, Any]],
        timestamp: float,
    ) -> None:
        if temp_status and not temp_status.get("__error__"):
            actual = self._coerce_float(temp_status.get("temperature"))
            target = self._coerce_float(temp_status.get("target"))
            actual = actual if actual is not None else math.nan
            target = target if target is not None else math.nan
            self._append_history(self._temp_history, timestamp, actual, target)
        if press_status and not press_status.get("__error__"):
            actual_val = self._coerce_float(self._extract_pressure_value(press_status))
            target_val = self._coerce_float(press_status.get("target"))
            actual_val = actual_val if actual_val is not None else math.nan
            target_val = target_val if target_val is not None else math.nan
            self._append_history(self._press_history, timestamp, actual_val, target_val)

    def _refresh_charts(self) -> None:
        if not self.chart_canvas or not self._chart_fig:
            return
        self._update_chart_lines(self._temp_history, self._temp_line_actual, self._temp_line_target)
        self._update_chart_lines(self._press_history, self._press_line_actual, self._press_line_target)
        self._chart_fig.canvas.draw_idle()

    def _update_chart_lines(
        self,
        history: Deque[Tuple[float, float, float]],
        line_actual,
        line_target,
    ) -> None:
        if line_actual is None or line_target is None:
            return
        if not history:
            line_actual.set_data([], [])
            line_target.set_data([], [])
            return
        base = history[0][0]
        times = [item[0] - base for item in history]
        actual = [item[1] for item in history]
        target = [item[2] for item in history]
        line_actual.set_data(times, actual)
        line_target.set_data(times, target)
        axis = line_actual.axes
        axis.relim()
        axis.autoscale_view()
        if times[-1] <= 0:
            axis.set_xlim(0, self._chart_history_s)
        else:
            axis.set_xlim(max(0.0, times[-1] - self._chart_history_s), times[-1])

    def _update_temp_display(self, status: Optional[Dict]) -> None:
        if not status:
            self.temp_live_var.set("--")
            self.temp_target_var.set("--")
            self.temp_error_var.set("--")
            return
        if status.get("__error__"):
            self.temp_live_var.set("--")
            self.temp_target_var.set("--")
            self.temp_error_var.set(status.get("__error__"))
            return
        self.temp_live_var.set(self._fmt_value(status.get("temperature")))
        self.temp_target_var.set(self._fmt_value(status.get("target")))
        error = status.get("error")
        if error is None and status.get("mae") is not None:
            error = status.get("mae")
        self.temp_error_var.set(self._fmt_value(error))

    def _update_press_display(self, status: Optional[Dict]) -> None:
        if not status:
            self.press_live_var.set("--")
            self.press_target_var.set("--")
            self.press_error_var.set("--")
            return
        if status.get("__error__"):
            self.press_live_var.set("--")
            self.press_target_var.set("--")
            self.press_error_var.set(status.get("__error__"))
            return
        last = status.get("last", {}) if isinstance(status.get("last"), dict) else {}
        feedback = last.get("feedback")
        if feedback is None:
            feedback = last.get("pressure")
        if feedback is None:
            feedback = status.get("pressure")
        self.press_live_var.set(self._fmt_value(feedback))
        self.press_target_var.set(self._fmt_value(status.get("target")))
        error = last.get("error")
        if error is None and status.get("mae") is not None:
            error = status.get("mae")
        self.press_error_var.set(self._fmt_value(error))

    # Shutdown ------------------------------------------------------------
    def shutdown(self, destroy_window: Optional[bool] = None) -> None:
        if destroy_window is None:
            destroy_window = self._owns_window
        self.stop_plan()
        self._rt_stop.set()
        if self._rt_thread and self._rt_thread.is_alive():
            try:
                self._rt_thread.join(timeout=1.0)
            except Exception:
                pass
        self._close_csv_session()
        target = self._window if destroy_window else self
        try:
            target.destroy()
        except Exception:
            pass

    def on_close(self) -> None:
        self.shutdown(destroy_window=True)


if __name__ == "__main__":
    app = MultiSequenceApp()
    app._window.mainloop()
