import json
import threading
import time
from dataclasses import dataclass
from typing import List, Optional

import tkinter as tk
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinter import messagebox, scrolledtext

import pyautogui

from tcp_utils import JSONLineClient

TEMP_DEFAULT_HOST = "127.0.0.1"
TEMP_DEFAULT_PORT = 50010
PRESS_DEFAULT_HOST = "127.0.0.1"
PRESS_DEFAULT_PORT = 50020


class ControllerClient:
    def __init__(self, host: str, port: int, name: str):
        self.host = host
        self.port = port
        self.name = name

    def _request(self, payload):
        client = JSONLineClient(self.host, self.port, timeout=5.0)
        response = client.request(payload)
        if not response.get("ok", True):
            raise RuntimeError(f"{self.name} 控制失败: {response.get('error', '未知错误')}")
        return response

    def ping(self):
        return self._request({"cmd": "ping"})

    def status(self):
        return self._request({"cmd": "status"})

    def set_temperature(self, value: float, start: bool = False):
        return self._request({"cmd": "set_target", "value": value, "start": start})

    def start_pid(self):
        return self._request({"cmd": "start_pid"})

    def stop_pid(self):
        return self._request({"cmd": "stop_pid"})

    def start_ramp(self, start: float, end: float, rate: float, hold: float = 0.0, loop: bool = False):
        payload = {
            "cmd": "start_ramp",
            "params": {"start": start, "end": end, "rate": rate, "hold": hold, "loop": loop},
        }
        return self._request(payload)

    def stop_ramp(self):
        return self._request({"cmd": "stop_ramp"})

    def set_pressure(self, value: float, start: bool = False):
        payload = {"cmd": "set_target", "value": value, "start": start}
        return self._request(payload)

    def start_pressure(self):
        return self._request({"cmd": "start_control"})

    def stop_pressure(self):
        return self._request({"cmd": "stop_control"})

    def stop_all(self):
        try:
            self._request({"cmd": "stop_all"})
        except Exception:
            pass


@dataclass
class RampSegment:
    start: float
    end: float
    rate: float
    hold: float = 0.0


class SequenceRunner(threading.Thread):
    def __init__(self, app, plan):
        super().__init__(daemon=True)
        self.app = app
        self.plan = plan
        self.stop_event = threading.Event()

    def stop(self):
        self.stop_event.set()

    def log(self, msg: str):
        self.app.log(msg)

    # Utility waiters -------------------------------------------------
    def wait_temperature(self, client: ControllerClient, mae_thr: float, hold_s: float) -> Optional[dict]:
        stable_since = None
        while not self.stop_event.is_set():
            try:
                resp = client.status()
            except Exception as exc:
                self.log(f"温控状态读取失败: {exc}")
                time.sleep(1.0)
                continue
            status = resp.get("status", {})
            mae_raw = status.get("mae")
            err_raw = status.get("error")
            mae = None
            error = None
            try:
                if mae_raw is not None:
                    mae = float(mae_raw)
            except (TypeError, ValueError):
                mae = None
            try:
                if err_raw is not None:
                    error = float(err_raw)
            except (TypeError, ValueError):
                error = None
            stable = False
            if mae is not None:
                stable = mae <= mae_thr
            elif error is not None:
                stable = abs(error) <= mae_thr
            if stable:
                if stable_since is None:
                    stable_since = time.time()
                    if mae is not None:
                        self.log(f"温度 MAE={mae:.3f}≤{mae_thr:.3f}，开始计时")
                    elif error is not None:
                        self.log(f"温度偏差={abs(error):.3f}≤{mae_thr:.3f}，开始计时")
                elif time.time() - stable_since >= hold_s:
                    self.log("温度判稳完成")
                    return status
            else:
                if stable_since is not None:
                    self.log("温度偏离阈值，重新计时")
                stable_since = None
            time.sleep(1.0)
        return None

    def wait_pressure(self, client: ControllerClient, mae_thr: float, hold_s: float) -> Optional[dict]:
        stable_since = None
        while not self.stop_event.is_set():
            try:
                resp = client.status()
            except Exception as exc:
                self.log(f"压力状态读取失败: {exc}")
                time.sleep(1.0)
                continue
            status = resp.get("status", {})
            mae_raw = status.get("mae")
            last = status.get("last", {}) if isinstance(status.get("last"), dict) else {}
            err_raw = last.get("error")
            mae = None
            error = None
            try:
                if mae_raw is not None:
                    mae = float(mae_raw)
            except (TypeError, ValueError):
                mae = None
            try:
                if err_raw is not None:
                    error = float(err_raw)
            except (TypeError, ValueError):
                error = None
            stable = False
            if mae is not None:
                stable = mae <= mae_thr
            elif error is not None:
                stable = abs(error) <= mae_thr
            if stable:
                if stable_since is None:
                    stable_since = time.time()
                    if mae is not None:
                        self.log(f"压力 MAE={mae:.3f}≤{mae_thr:.3f}，开始计时")
                    elif error is not None:
                        self.log(f"压力偏差={abs(error):.3f}≤{mae_thr:.3f}，开始计时")
                elif time.time() - stable_since >= hold_s:
                    self.log("压力判稳完成")
                    return status
            else:
                stable_since = None
            time.sleep(1.0)
        return None

    def ensure_temperature_running(self, client: ControllerClient, target: float):
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

    def ensure_pressure_running(self, client: ControllerClient, target: float):
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

    # Core execution --------------------------------------------------
    def run(self):
        temp_client = ControllerClient(self.plan["temp_host"], self.plan["temp_port"], "温控")
        pressure_client = ControllerClient(self.plan["press_host"], self.plan["press_port"], "压力")

        try:
            temp_client.ping()
            pressure_client.ping()
        except Exception as exc:
            self.log(f"连接设备失败: {exc}")
            return

        try:
            if self.plan["test_type"] == "matrix":
                self._run_matrix(temp_client, pressure_client)
            else:
                self._run_ramp(temp_client, pressure_client)
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

    # Matrix mode -----------------------------------------------------
    def _run_matrix(self, temp_client: ControllerClient, pressure_client: ControllerClient):
        temps = self.plan["temperatures"]
        pressures = self.plan["pressures"]
        mode = self.plan["mode"]
        t_thr = self.plan["temp_mae"]
        t_hold = self.plan["temp_hold"]
        p_thr = self.plan["press_mae"]
        p_hold = self.plan["press_hold"]

        if mode == "temp_first":
            outer = temps
            inner = pressures
            for temp in outer:
                if self.stop_event.is_set():
                    break
                self.log(f"设定温度 {temp}°C，等待稳定")
                if not self.ensure_temperature_running(temp_client, temp):
                    return
                t_status = self.wait_temperature(temp_client, t_thr, t_hold)
                if t_status is None:
                    break
                for pressure in inner:
                    if self.stop_event.is_set():
                        break
                    self.log(f"设定压力 {pressure}g，等待稳定")
                    if not self.ensure_pressure_running(pressure_client, pressure):
                        return
                    p_status = self.wait_pressure(pressure_client, p_thr, p_hold)
                    if p_status is None:
                        break
                    self._record_result(temp, pressure, t_status, p_status)
        else:  # pressure first
            outer = pressures
            inner = temps
            for pressure in outer:
                if self.stop_event.is_set():
                    break
                self.log(f"设定压力 {pressure}g，等待稳定")
                if not self.ensure_pressure_running(pressure_client, pressure):
                    return
                p_status = self.wait_pressure(pressure_client, p_thr, p_hold)
                if p_status is None:
                    break
                for temp in inner:
                    if self.stop_event.is_set():
                        break
                    self.log(f"设定温度 {temp}°C，等待稳定")
                    if not self.ensure_temperature_running(temp_client, temp):
                        return
                    t_status = self.wait_temperature(temp_client, t_thr, t_hold)
                    if t_status is None:
                        break
                    p_status = self.wait_pressure(pressure_client, p_thr, p_hold)
                    if p_status is None:
                        break
                    self._record_result(temp, pressure, t_status, p_status)

    # Ramp mode -------------------------------------------------------
    def _run_ramp(self, temp_client: ControllerClient, pressure_client: ControllerClient):
        pressures = self.plan["pressures"]
        ramps: List[RampSegment] = self.plan["ramps"]
        mode = self.plan["mode"]
        if not ramps:
            self.log("未提供有效的变温程序，已取消")
            return
        p_thr = self.plan["press_mae"]
        p_hold = self.plan["press_hold"]
        t_thr = self.plan["temp_mae"]
        t_hold = self.plan["temp_hold"]

        if mode == "pressure_first":
            self._run_ramp_pressure_first(
                temp_client,
                pressure_client,
                pressures,
                ramps,
                t_thr,
                t_hold,
                p_thr,
                p_hold,
            )
        else:
            self._run_ramp_temp_first(
                temp_client,
                pressure_client,
                pressures,
                ramps,
                t_thr,
                t_hold,
                p_thr,
                p_hold,
            )

    def _wait_ramp_finish(self, temp_client: ControllerClient) -> bool:
        while not self.stop_event.is_set():
            try:
                resp = temp_client.status()
            except Exception as exc:
                self.log(f"温控状态读取失败: {exc}")
                time.sleep(1.0)
                continue
            status = resp.get("status", {})
            if not status.get("ramp_active", False):
                return True
            time.sleep(1.0)
        return False

    def _start_ramp_segment(self, temp_client: ControllerClient, seg: RampSegment) -> bool:
        try:
            resp = temp_client.start_ramp(seg.start, seg.end, seg.rate, hold=seg.hold, loop=False)
        except Exception as exc:
            self.log(f"温控启动变温失败: {exc}")
            return False
        if not resp.get("result", True):
            self.log("温控拒绝执行变温程序")
            return False
        return True

    def _run_ramp_pressure_first(
        self,
        temp_client: ControllerClient,
        pressure_client: ControllerClient,
        pressures: List[float],
        ramps: List[RampSegment],
        t_thr: float,
        t_hold: float,
        p_thr: float,
        p_hold: float,
    ):
        for pressure in pressures:
            if self.stop_event.is_set():
                break
            self.log(f"设定压力 {pressure}g，等待稳定")
            if not self.ensure_pressure_running(pressure_client, pressure):
                return
            p_status = self.wait_pressure(pressure_client, p_thr, p_hold)
            if p_status is None:
                break
            for seg in ramps:
                if self.stop_event.is_set():
                    break
                self.log(
                    f"执行变温程序: {seg.start}→{seg.end} °C @ {seg.rate} °C/min, 保温 {seg.hold} min"
                )
                if not self.ensure_temperature_running(temp_client, seg.start):
                    return
                t_status = self.wait_temperature(temp_client, t_thr, t_hold)
                if t_status is None:
                    break
                if not self._start_ramp_segment(temp_client, seg):
                    return
                if not self._wait_ramp_finish(temp_client):
                    return
                final_status = self.wait_temperature(temp_client, t_thr, t_hold)
                if final_status is None:
                    break
                p_status = self.wait_pressure(pressure_client, p_thr, p_hold)
                if p_status is None:
                    break
                self._record_result(final_status.get("target"), pressure, final_status, p_status)

    def _run_ramp_temp_first(
        self,
        temp_client: ControllerClient,
        pressure_client: ControllerClient,
        pressures: List[float],
        ramps: List[RampSegment],
        t_thr: float,
        t_hold: float,
        p_thr: float,
        p_hold: float,
    ):
        for seg in ramps:
            if self.stop_event.is_set():
                break
            self.log(
                f"准备变温段: {seg.start}→{seg.end} °C @ {seg.rate} °C/min, 保温 {seg.hold} min"
            )
            for pressure in pressures:
                if self.stop_event.is_set():
                    break
                self.log(f"设定压力 {pressure}g，等待稳定")
                if not self.ensure_pressure_running(pressure_client, pressure):
                    return
                p_status = self.wait_pressure(pressure_client, p_thr, p_hold)
                if p_status is None:
                    break
                if not self.ensure_temperature_running(temp_client, seg.start):
                    return
                t_status = self.wait_temperature(temp_client, t_thr, t_hold)
                if t_status is None:
                    break
                if not self._start_ramp_segment(temp_client, seg):
                    return
                if not self._wait_ramp_finish(temp_client):
                    return
                final_status = self.wait_temperature(temp_client, t_thr, t_hold)
                if final_status is None:
                    break
                p_status = self.wait_pressure(pressure_client, p_thr, p_hold)
                if p_status is None:
                    break
                self._record_result(final_status.get("target"), pressure, final_status, p_status)

    def _record_result(self, temp, pressure, temp_status, press_status):
        temp_value = temp if temp is not None else temp_status.get("temperature")
        pressure_value = pressure if pressure is not None else press_status.get("target")
        result = {
            "temperature": temp_value,
            "pressure": pressure_value,
            "temp_status": temp_status,
            "pressure_status": press_status,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.app.maybe_sim_click(self.log)
        self.app.add_result(result)


class MultiSequenceApp(ttk.Window):
    def __init__(self):
        super().__init__(themename="cosmo")
        self.title("多温度+压力自动测试调度")
        self.geometry("900x720")

        self.temp_host_var = tk.StringVar(value=TEMP_DEFAULT_HOST)
        self.temp_port_var = tk.IntVar(value=TEMP_DEFAULT_PORT)
        self.press_host_var = tk.StringVar(value=PRESS_DEFAULT_HOST)
        self.press_port_var = tk.IntVar(value=PRESS_DEFAULT_PORT)

        self.temps_var = tk.StringVar(value="25,30,40")
        self.pressures_var = tk.StringVar(value="1000,2000,3000")
        self.mode_var = tk.StringVar(value="temp_first")
        self.test_type_var = tk.StringVar(value="matrix")
        self.temp_mae_var = tk.DoubleVar(value=0.3)
        self.temp_hold_var = tk.DoubleVar(value=60.0)
        self.press_mae_var = tk.DoubleVar(value=50.0)
        self.press_hold_var = tk.DoubleVar(value=30.0)
        self.ramp_text = tk.StringVar(value="25,60,1.5,5\n60,20,1.0,3")

        self.runner: Optional[SequenceRunner] = None
        self.results: List[dict] = []

        self.sim_click_enabled = tk.BooleanVar(value=False)
        self.sim_click_pos: Optional[tuple[int, int]] = None
        self.sim_click_label = tk.StringVar(value="未设置")

        self._build_ui()

    # UI --------------------------------------------------------------
    def _build_ui(self):
        main = ttk.Frame(self, padding=12)
        main.pack(fill=tk.BOTH, expand=True)

        conn = ttk.Labelframe(main, text="连接设置")
        conn.pack(fill=tk.X, pady=6)
        row = ttk.Frame(conn)
        row.pack(fill=tk.X, pady=4)
        ttk.Label(row, text="温控主机").grid(row=0, column=0, sticky=tk.W, padx=4)
        ttk.Entry(row, textvariable=self.temp_host_var, width=15).grid(row=0, column=1, padx=4)
        ttk.Label(row, text="端口").grid(row=0, column=2)
        ttk.Entry(row, textvariable=self.temp_port_var, width=8).grid(row=0, column=3, padx=4)
        ttk.Label(row, text="压力主机").grid(row=0, column=4, padx=(16, 4))
        ttk.Entry(row, textvariable=self.press_host_var, width=15).grid(row=0, column=5, padx=4)
        ttk.Label(row, text="端口").grid(row=0, column=6)
        ttk.Entry(row, textvariable=self.press_port_var, width=8).grid(row=0, column=7, padx=4)

        plan = ttk.Labelframe(main, text="测试计划")
        plan.pack(fill=tk.X, pady=6)

        row1 = ttk.Frame(plan)
        row1.pack(fill=tk.X, pady=2)
        ttk.Label(row1, text="温度列表 (°C)").pack(side=tk.LEFT)
        ttk.Entry(row1, textvariable=self.temps_var, width=40).pack(side=tk.LEFT, padx=6)

        row2 = ttk.Frame(plan)
        row2.pack(fill=tk.X, pady=2)
        ttk.Label(row2, text="压力列表 (g)").pack(side=tk.LEFT)
        ttk.Entry(row2, textvariable=self.pressures_var, width=40).pack(side=tk.LEFT, padx=6)

        mode_frame = ttk.Frame(plan)
        mode_frame.pack(fill=tk.X, pady=4)
        ttk.Label(mode_frame, text="执行模式").pack(side=tk.LEFT)
        ttk.Radiobutton(mode_frame, text="定温→多压力", variable=self.mode_var, value="temp_first").pack(side=tk.LEFT, padx=6)
        ttk.Radiobutton(mode_frame, text="定压→多温度", variable=self.mode_var, value="pressure_first").pack(side=tk.LEFT, padx=6)

        test_type = ttk.Frame(plan)
        test_type.pack(fill=tk.X, pady=4)
        ttk.Label(test_type, text="测试类型").pack(side=tk.LEFT)
        ttk.Radiobutton(test_type, text="离散矩阵", variable=self.test_type_var, value="matrix").pack(side=tk.LEFT, padx=6)
        ttk.Radiobutton(test_type, text="变温程序", variable=self.test_type_var, value="ramp").pack(side=tk.LEFT, padx=6)

        click_box = ttk.Labelframe(main, text="模拟点击")
        click_box.pack(fill=tk.X, pady=6)
        click_row = ttk.Frame(click_box)
        click_row.pack(fill=tk.X, pady=2)
        ttk.Checkbutton(click_row, text="开启模拟点击", variable=self.sim_click_enabled).pack(side=tk.LEFT, padx=4)
        ttk.Button(click_row, text="设置点击点", command=self.set_sim_click_point,
                   bootstyle="outline-info").pack(side=tk.LEFT, padx=4)
        ttk.Label(click_row, textvariable=self.sim_click_label, width=18).pack(side=tk.LEFT, padx=4)
        ttk.Label(click_row, text="(窗口提示后按 Enter)").pack(side=tk.LEFT, padx=4)

        stability = ttk.Labelframe(main, text="判稳参数")
        stability.pack(fill=tk.X, pady=6)
        srow1 = ttk.Frame(stability)
        srow1.pack(fill=tk.X, pady=2)
        ttk.Label(srow1, text="温度 MAE 阈值 (°C)").pack(side=tk.LEFT)
        ttk.Entry(srow1, textvariable=self.temp_mae_var, width=8).pack(side=tk.LEFT, padx=4)
        ttk.Label(srow1, text="持续时间 (s)").pack(side=tk.LEFT)
        ttk.Entry(srow1, textvariable=self.temp_hold_var, width=8).pack(side=tk.LEFT, padx=4)

        srow2 = ttk.Frame(stability)
        srow2.pack(fill=tk.X, pady=2)
        ttk.Label(srow2, text="压力 MAE 阈值 (g)").pack(side=tk.LEFT)
        ttk.Entry(srow2, textvariable=self.press_mae_var, width=8).pack(side=tk.LEFT, padx=4)
        ttk.Label(srow2, text="持续时间 (s)").pack(side=tk.LEFT)
        ttk.Entry(srow2, textvariable=self.press_hold_var, width=8).pack(side=tk.LEFT, padx=4)

        ramp_frame = ttk.Labelframe(main, text="变温段 (start,end,rate,hold[min])")
        ramp_frame.pack(fill=tk.BOTH, expand=False, pady=6)
        self.ramp_box = tk.Text(ramp_frame, height=4)
        self.ramp_box.pack(fill=tk.X, padx=4, pady=4)
        self.ramp_box.insert("1.0", self.ramp_text.get())

        btns = ttk.Frame(main)
        btns.pack(fill=tk.X, pady=6)
        ttk.Button(btns, text="开始", bootstyle=SUCCESS, command=self.start_plan).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="停止", bootstyle=DANGER, command=self.stop_plan).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="导出结果", command=self.export_results).pack(side=tk.RIGHT, padx=4)

        log_frame = ttk.Labelframe(main, text="日志")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=6)
        self.log_text = scrolledtext.ScrolledText(log_frame, height=15)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

    # Helpers ---------------------------------------------------------
    def parse_float_list(self, text: str) -> List[float]:
        values = []
        for chunk in text.split(','):
            chunk = chunk.strip()
            if not chunk:
                continue
            try:
                values.append(float(chunk))
            except ValueError:
                raise ValueError(f"无法解析数值: {chunk}") from None
        return values

    def parse_ramps(self) -> List[RampSegment]:
        content = self.ramp_box.get("1.0", tk.END).strip()
        ramps: List[RampSegment] = []
        if not content:
            return ramps
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(',') if p.strip()]
            if len(parts) < 3:
                raise ValueError(f"变温段格式错误: {line}")
            start, end, rate = map(float, parts[:3])
            hold = float(parts[3]) if len(parts) >= 4 else 0.0
            ramps.append(RampSegment(start, end, rate, hold))
        return ramps

    def build_plan(self):
        mode = self.mode_var.get()
        temps = self.parse_float_list(self.temps_var.get())
        pressures = self.parse_float_list(self.pressures_var.get())
        ramps = self.parse_ramps()
        plan = {
            "temp_host": self.temp_host_var.get().strip() or TEMP_DEFAULT_HOST,
            "temp_port": int(self.temp_port_var.get() or TEMP_DEFAULT_PORT),
            "press_host": self.press_host_var.get().strip() or PRESS_DEFAULT_HOST,
            "press_port": int(self.press_port_var.get() or PRESS_DEFAULT_PORT),
            "temperatures": temps,
            "pressures": pressures,
            "mode": mode,
            "temp_mae": float(self.temp_mae_var.get()),
            "temp_hold": float(self.temp_hold_var.get()),
            "press_mae": float(self.press_mae_var.get()),
            "press_hold": float(self.press_hold_var.get()),
            "test_type": self.test_type_var.get(),
            "ramps": ramps,
        }
        if plan["test_type"] == "matrix":
            if not temps:
                raise ValueError("请提供至少一个温度目标")
            if not pressures:
                raise ValueError("请提供至少一个压力目标")
        else:
            if not pressures:
                raise ValueError("变温模式至少需要一个压力点")
        return plan

    def start_plan(self):
        if self.runner and self.runner.is_alive():
            messagebox.showwarning("提示", "任务正在运行")
            return
        try:
            plan = self.build_plan()
        except Exception as exc:
            messagebox.showerror("参数错误", str(exc))
            return
        if self.sim_click_enabled.get() and not self.sim_click_pos:
            messagebox.showerror("错误", "请先设置模拟点击点")
            return
        self.results.clear()
        self.log_text.delete("1.0", tk.END)
        self.log("开始执行测试计划…")
        self.runner = SequenceRunner(self, plan)
        self.runner.start()

    def stop_plan(self):
        if self.runner:
            self.runner.stop()
            self.log("停止指令已发送，等待当前步骤结束…")

    def export_results(self):
        if not self.results:
            messagebox.showinfo("提示", "暂无数据可导出")
            return
        path = time.strftime("multi_test_%Y%m%d_%H%M%S.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.results, fh, ensure_ascii=False, indent=2)
        messagebox.showinfo("导出完成", f"结果已保存到 {path}")

    # Simulated click ------------------------------------------------
    def set_sim_click_point(self):
        if pyautogui is None:
            messagebox.showerror("错误", "缺少 pyautogui，无法设置模拟点击点")
            return
        messagebox.showinfo("设置点击点", "移动鼠标到目标软件按钮处，按Enter键记录。")

        top = tk.Toplevel(self)
        top.title("请切换到目标点，按Enter记录")
        top.geometry("300x100")
        top.attributes("-topmost", True)

        tk.Label(top, text="移动鼠标到目标点，然后按Enter").pack(pady=20)

        def on_enter(event=None):
            try:
                pos = pyautogui.position()
                x, y = int(pos[0]), int(pos[1])
            except Exception as exc:
                self.log(f"记录模拟点击点失败: {exc}")
                top.destroy()
                return
            self.sim_click_pos = (x, y)
            self.sim_click_label.set(f"{x}, {y}")
            self.sim_click_enabled.set(True)
            self.log(f"已记录模拟点击点: ({x}, {y})")
            top.destroy()

        top.bind('<Return>', on_enter)
        top.grab_set()
        top.focus_set()
        top.wait_window()

    def perform_sim_click(self):
        if self.sim_click_pos is None:
            self.log("模拟点击点未设置，跳过点击")
            return
        if pyautogui is None:
            self.log("模拟点击失败：缺少 pyautogui")
            return
        x, y = self.sim_click_pos
        try:
            pyautogui.click(x, y)
            self.log(f"已模拟点击: ({x}, {y})")
        except Exception as exc:
            self.log(f"模拟点击失败: {exc}")

    def maybe_sim_click(self, logger):
        try:
            enabled = bool(self.sim_click_enabled.get())
        except Exception:
            enabled = False
        if not enabled:
            return
        if not self.sim_click_pos:
            logger("模拟点击点未设置，已关闭模拟点击")
            try:
                self.sim_click_enabled.set(False)
            except Exception:
                pass
            return
        logger("执行模拟点击…")
        self.perform_sim_click()
        self.perform_sim_click()

    # Logging ---------------------------------------------------------
    def log(self, message: str):
        timestamp = time.strftime("%H:%M:%S")
        def append():
            self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
            self.log_text.see(tk.END)
        self.after(0, append)

    def add_result(self, result: dict):
        self.results.append(result)
        self.log(
            f"记录结果: T={result['temperature']}, P={result['pressure']}, "
            f"温度误差={result['temp_status'].get('error')}, 压力误差={result['pressure_status'].get('last', {}).get('error') if result['pressure_status'] else None}"
        )


if __name__ == "__main__":
    app = MultiSequenceApp()
    app.mainloop()
