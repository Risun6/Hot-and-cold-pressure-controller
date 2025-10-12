import json
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import tkinter as tk
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinter import messagebox, scrolledtext

from tcp_utils import JSONLineClient


TEMP_DEFAULT_HOST = "127.0.0.1"
TEMP_DEFAULT_PORT = 50010
PRESS_DEFAULT_HOST = "127.0.0.1"
PRESS_DEFAULT_PORT = 50020

DEFAULT_TEMP_MAE = 0.3
DEFAULT_TEMP_HOLD = 60.0
DEFAULT_TEMP_RECHECK_HOLD = 15.0
DEFAULT_PRESS_MAE = 50.0
DEFAULT_PRESS_HOLD = 30.0
DEFAULT_CELL_TIMEOUT = 300.0
DEFAULT_CONFIRM_COUNT = 2
REALTIME_REFRESH_S = 1.0


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
        self.stop_event = threading.Event()
        self.temp_handler = temp_handler
        self.press_handler = press_handler
        self.temp_lock = temp_lock
        self.press_lock = press_lock

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

        self.temp_live_var = tk.StringVar(value="--")
        self.temp_target_var = tk.StringVar(value="--")
        self.temp_error_var = tk.StringVar(value="--")
        self.press_live_var = tk.StringVar(value="--")
        self.press_target_var = tk.StringVar(value="--")
        self.press_error_var = tk.StringVar(value="--")
        self.realtime_timestamp_var = tk.StringVar(value="--")
        self.realtime_status_var = tk.StringVar(value="")

        self.runner: Optional[SequenceRunner] = None
        self.results: List[Dict] = []

        self._rt_stop = threading.Event()
        self._rt_lock = threading.Lock()
        self._rt_latest: Optional[Tuple[Optional[Dict], Optional[Dict], float]] = None
        self._rt_thread: Optional[threading.Thread] = None

        self._build_ui()
        self._init_default_matrix()
        self._start_realtime_monitor()

    # UI -----------------------------------------------------------------
    def _build_ui(self) -> None:
        main = ttk.Frame(self, padding=12)
        main.pack(fill=tk.BOTH, expand=True)

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

        buttons = ttk.Frame(main)
        buttons.pack(fill=tk.X, pady=8)
        ttk.Button(buttons, text="开始执行", bootstyle=SUCCESS, command=self.start_plan).pack(side=tk.LEFT, padx=4)
        ttk.Button(buttons, text="停止", bootstyle=DANGER, command=self.stop_plan).pack(side=tk.LEFT, padx=4)
        ttk.Button(buttons, text="导出结果", command=self.export_results, bootstyle="outline-primary").pack(
            side=tk.RIGHT, padx=4
        )

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
            "press_mae": DEFAULT_PRESS_MAE,
            "press_hold": DEFAULT_PRESS_HOLD,
            "repeat_confirm": confirm_need,
        }
        return plan

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

    def start_plan(self) -> None:
        if self.runner and self.runner.is_alive():
            messagebox.showwarning("提示", "任务正在运行")
            return
        try:
            plan = self.build_plan()
        except Exception as exc:
            messagebox.showerror("参数错误", str(exc))
            return
        self.results.clear()
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
        timestamp = time.strftime("%H:%M:%S")
        def append() -> None:
            self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
            self.log_text.see(tk.END)

        self.after(0, append)

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

    def _update_realtime_ui(self) -> None:
        if self._rt_stop.is_set():
            return
        with self._rt_lock:
            latest = self._rt_latest
        if latest:
            temp_status, press_status, ts = latest
            self._update_temp_display(temp_status)
            self._update_press_display(press_status)
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
        self.press_live_var.set(self._fmt_value(last.get("feedback")))
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
