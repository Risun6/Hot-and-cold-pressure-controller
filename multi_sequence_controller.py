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
from PIL import Image, ImageChops

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

        if self.plan.get("pixel_detection_enabled") and not self.plan.get("pixel_region"):
            self.log("像素检测已启用但未设置区域，已自动关闭该功能")
            self.plan["pixel_detection_enabled"] = False

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
                    if not self._post_stable_actions(temp, pressure, t_status, p_status):
                        return
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
                    if not self._post_stable_actions(temp, pressure, t_status, p_status):
                        return
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
                if not self._post_stable_actions(
                    final_status.get("target"), pressure, final_status, p_status
                ):
                    return
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
                if not self._post_stable_actions(
                    final_status.get("target"), pressure, final_status, p_status
                ):
                    return
                self._record_result(final_status.get("target"), pressure, final_status, p_status)

    def _maybe_sim_click(self):
        if not self.plan.get("sim_click_enabled"):
            return False
        pos = self.plan.get("sim_click_pos")
        if not pos or len(pos) != 2:
            self.log("模拟点击已启用但未设置坐标，已自动关闭该功能")
            self.plan["sim_click_enabled"] = False
            return False
        x, y = int(pos[0]), int(pos[1])
        self.log("执行模拟点击…")
        success = self.app.perform_sim_click((x, y))
        if success:
            time.sleep(0.05)
            self.app.perform_sim_click((x, y))
        return success

    def _post_stable_actions(self, temp, pressure, temp_status, press_status):
        if self.stop_event.is_set():
            return False
        clicked = self._maybe_sim_click()
        if clicked and not self.stop_event.is_set():
            time.sleep(0.2)

        if not self.plan.get("pixel_detection_enabled") or self.stop_event.is_set():
            return not self.stop_event.is_set()

        region = self.plan.get("pixel_region")
        if not region:
            self.log("像素检测已启用但未设置区域，已自动关闭该功能")
            self.plan["pixel_detection_enabled"] = False
            return not self.stop_event.is_set()

        self.app.pixel_detection_region = tuple(region)
        try:
            self.app.capture_pixel_baseline()
        except Exception as exc:
            self.log(f"刷新像素基准失败: {exc}")
            return not self.stop_event.is_set()

        interval = max(0.05, float(self.plan.get("pixel_interval", 0.5)))
        timeout = max(0.0, float(self.plan.get("pixel_timeout", 30.0)))
        sensitivity = max(0.0, float(self.plan.get("pixel_sensitivity", 5.0)))
        log_enabled = bool(self.plan.get("pixel_log_enabled", False))

        temp_value = temp
        if temp_value is None:
            temp_value = temp_status.get("target")
        if temp_value is None:
            temp_value = temp_status.get("temperature")
        pressure_value = pressure
        if pressure_value is None:
            pressure_value = press_status.get("target") if press_status else None
        if pressure_value is None and press_status:
            pressure_value = press_status.get("last", {}).get("pressure")
        self.log(
            f"开始像素检测 (超时 {timeout:.1f}s，间隔 {interval:.2f}s，阈值 {sensitivity:.2f}%)…"
        )
        detected = self.app.wait_for_pixel_change(
            self.stop_event,
            timeout,
            interval,
            sensitivity,
            log_enabled,
        )
        if detected:
            self.log(
                f"✅ 像素检测通过：T={temp_value}°C, P={pressure_value}g"
            )
        else:
            self.log(
                f"⚠️ 像素检测超时或无变化：T={temp_value}°C, P={pressure_value}g"
            )
        return not self.stop_event.is_set()

    def _record_result(self, temp, pressure, temp_status, press_status):
        temp_value = temp
        if temp_value is None:
            temp_value = temp_status.get("target")
        if temp_value is None:
            temp_value = temp_status.get("temperature")
        pressure_value = pressure
        if pressure_value is None:
            pressure_value = press_status.get("target") if press_status else None
        if pressure_value is None and press_status:
            pressure_value = press_status.get("last", {}).get("pressure")
        result = {
            "temperature": temp_value,
            "pressure": pressure_value,
            "temp_status": temp_status,
            "pressure_status": press_status,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
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

        self.sim_click_enabled = tk.BooleanVar(value=False)
        self.sim_click_pos = None
        self.sim_click_label_var = tk.StringVar(value="坐标：未设置")

        self.pixel_detection_enabled = tk.BooleanVar(value=False)
        self.pixel_log_enabled = tk.BooleanVar(value=False)
        self.pixel_interval_var = tk.IntVar(value=500)  # ms
        self.pixel_timeout_var = tk.DoubleVar(value=30.0)
        self.pixel_sensitivity_var = tk.DoubleVar(value=5.0)
        self.pixel_detection_region: Optional[tuple[int, int, int, int]] = None
        self.initial_pixel_snapshot: Optional[Image.Image] = None
        self.prev_pixel_snapshot: Optional[Image.Image] = None
        self.pixel_indicator_canvas: Optional[tk.Canvas] = None
        self.pixel_indicator_circle = None

        self.runner: Optional[SequenceRunner] = None
        self.results: List[dict] = []

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

        click_frame = ttk.Labelframe(main, text="模拟点击")
        click_frame.pack(fill=tk.X, pady=6)
        click_row = ttk.Frame(click_frame)
        click_row.pack(fill=tk.X, pady=4)
        ttk.Checkbutton(click_row, text="启用模拟点击", variable=self.sim_click_enabled).pack(side=tk.LEFT, padx=4)
        ttk.Button(
            click_row,
            text="设置点击点",
            command=self.set_sim_click_point,
            bootstyle="outline-info",
        ).pack(side=tk.LEFT, padx=6)
        ttk.Label(click_row, textvariable=self.sim_click_label_var).pack(side=tk.LEFT, padx=6)

        pixel_frame = ttk.Labelframe(main, text="像素变化检测")
        pixel_frame.pack(fill=tk.X, pady=6)

        pixel_row1 = ttk.Frame(pixel_frame)
        pixel_row1.pack(fill=tk.X, pady=4)
        ttk.Checkbutton(pixel_row1, text="启用像素检测", variable=self.pixel_detection_enabled).pack(side=tk.LEFT, padx=4)
        ttk.Button(
            pixel_row1,
            text="设置检测区域",
            command=self.set_pixel_detection_region,
            bootstyle="outline-info",
        ).pack(side=tk.LEFT, padx=6)
        self.pixel_indicator_canvas = tk.Canvas(pixel_row1, width=18, height=18, highlightthickness=0)
        self.pixel_indicator_circle = self.pixel_indicator_canvas.create_oval(2, 2, 16, 16, fill="red")
        self.pixel_indicator_canvas.pack(side=tk.LEFT, padx=6)

        pixel_row2 = ttk.Frame(pixel_frame)
        pixel_row2.pack(fill=tk.X, pady=4)
        ttk.Label(pixel_row2, text="检测间隔(ms)").pack(side=tk.LEFT)
        ttk.Entry(pixel_row2, textvariable=self.pixel_interval_var, width=8).pack(side=tk.LEFT, padx=4)
        ttk.Label(pixel_row2, text="超时(s)").pack(side=tk.LEFT, padx=(12, 0))
        ttk.Entry(pixel_row2, textvariable=self.pixel_timeout_var, width=8).pack(side=tk.LEFT, padx=4)
        ttk.Label(pixel_row2, text="灵敏度(%)").pack(side=tk.LEFT, padx=(12, 0))
        ttk.Entry(pixel_row2, textvariable=self.pixel_sensitivity_var, width=6).pack(side=tk.LEFT, padx=4)
        ttk.Checkbutton(pixel_row2, text="记录检测日志", variable=self.pixel_log_enabled).pack(side=tk.LEFT, padx=8)

        pixel_row3 = ttk.Frame(pixel_frame)
        pixel_row3.pack(fill=tk.X, pady=2)
        ttk.Button(
            pixel_row3,
            text="测试像素检测",
            command=self.test_pixel_detection,
            bootstyle="outline-secondary",
        ).pack(side=tk.LEFT, padx=4)

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

    def set_sim_click_point(self):
        messagebox.showinfo("设置点击点", "移动鼠标到目标软件按钮处，按Enter键记录。")

        top = tk.Toplevel(self)
        top.title("请切换到目标点，按Enter记录")
        top.geometry("320x110")

        label = tk.Label(top, text="移动鼠标到目标点，然后按Enter")
        label.pack(pady=20)

        def on_enter(event=None):
            try:
                pos = pyautogui.position()
            except Exception as exc:
                messagebox.showerror("获取失败", f"无法获取鼠标位置: {exc}")
                top.destroy()
                return
            self.sim_click_pos = (int(pos[0]), int(pos[1]))
            self.sim_click_label_var.set(f"坐标：{self.sim_click_pos[0]}, {self.sim_click_pos[1]}")
            self.log(f"已记录模拟点击点: {self.sim_click_pos}")
            top.destroy()

        top.bind('<Return>', on_enter)
        top.grab_set()
        top.focus_set()
        top.wait_window()

    def set_indicator_color(self, color: str):
        if not self.pixel_indicator_canvas or not self.pixel_indicator_circle:
            return

        def update():
            self.pixel_indicator_canvas.itemconfig(self.pixel_indicator_circle, fill=color)
            if color == "green":
                self.after(600, lambda: self.set_indicator_color("red"))

        self.after(0, update)

    def set_pixel_detection_region(self):
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

        coords = {"x1": 0, "y1": 0, "x2": 0, "y2": 0, "rect": None}

        def on_press(event):
            coords["x1"], coords["y1"] = event.x, event.y
            if coords["rect"]:
                canvas.delete(coords["rect"])
            coords["rect"] = canvas.create_rectangle(
                coords["x1"], coords["y1"], event.x, event.y, outline="red", width=2
            )

        def on_drag(event):
            if coords["rect"]:
                canvas.coords(coords["rect"], coords["x1"], coords["y1"], event.x, event.y)

        def on_release(event):
            coords["x2"], coords["y2"] = event.x, event.y

        def on_enter(event=None):
            overlay.grab_release()
            x1, y1 = coords["x1"], coords["y1"]
            x2, y2 = coords["x2"], coords["y2"]
            left, right = sorted((x1, x2))
            top, bottom = sorted((y1, y2))
            width = max(0, right - left)
            height = max(0, bottom - top)
            abs_x = overlay.winfo_rootx() + left
            abs_y = overlay.winfo_rooty() + top

            try:
                if width <= 0 or height <= 0:
                    raise ValueError("selected area too small")

                try:
                    screen_w, screen_h = pyautogui.size()
                except Exception:
                    screen_w = screen_h = None

                if screen_w is not None and screen_h is not None:
                    abs_x = max(0, min(abs_x, screen_w - 1))
                    abs_y = max(0, min(abs_y, screen_h - 1))
                    width = min(width, max(0, screen_w - abs_x))
                    height = min(height, max(0, screen_h - abs_y))
                    if width <= 0 or height <= 0:
                        raise ValueError("selected area too small")

                region = (int(abs_x), int(abs_y), int(width), int(height))
                snapshot = pyautogui.screenshot(region=region)
                if snapshot.mode != "RGB":
                    snapshot = snapshot.convert("RGB")

            except ValueError:
                self.log("检测区域无效：选取的区域太小")
                messagebox.showwarning("无效区域", "选取的检测区域太小，请拖拽一个更大的区域。")
            except Exception as exc:
                self.log(f"设置检测区域失败: {exc}")
                messagebox.showerror("错误", f"无法截取屏幕区域: {exc}")
            else:
                self.pixel_detection_region = region
                self.initial_pixel_snapshot = snapshot
                self.prev_pixel_snapshot = None
                self.set_indicator_color("red")
                self.log(f"已设置检测区域: {self.pixel_detection_region}")
            finally:
                overlay.destroy()
            return "break"

        canvas.bind("<ButtonPress-1>", on_press)
        canvas.bind("<B1-Motion>", on_drag)
        canvas.bind("<ButtonRelease-1>", on_release)
        overlay.bind("<Return>", on_enter)

    def capture_pixel_baseline(self):
        if not self.pixel_detection_region:
            raise RuntimeError("像素检测区域未设置")
        snapshot = pyautogui.screenshot(region=self.pixel_detection_region)
        if snapshot.mode != "RGB":
            snapshot = snapshot.convert("RGB")
        self.initial_pixel_snapshot = snapshot
        self.prev_pixel_snapshot = None
        self.set_indicator_color("red")
        self.log("已刷新像素基准快照")

    def wait_for_pixel_change(
        self,
        stop_event: Optional[threading.Event],
        timeout: float,
        interval: float,
        sensitivity: float,
        log_enabled: bool,
    ) -> bool:
        if not self.pixel_detection_region or self.initial_pixel_snapshot is None:
            return False

        event = stop_event or threading.Event()
        baseline = self.initial_pixel_snapshot
        if baseline.mode != "RGB":
            baseline = baseline.convert("RGB")
        total_pixels = max(1, baseline.width * baseline.height)
        start_time = time.time()
        attempt = 0
        self.set_indicator_color("red")

        while time.time() - start_time < timeout and not event.is_set():
            try:
                current = pyautogui.screenshot(region=self.pixel_detection_region)
                if current.mode != "RGB":
                    current = current.convert("RGB")
                diff = ImageChops.difference(baseline, current)
                if sensitivity <= 0:
                    if diff.getbbox():
                        self.prev_pixel_snapshot = current
                        self.set_indicator_color("green")
                        return True
                    pct = 0.0
                else:
                    diff_data = diff.getdata()
                    diff_pixels = sum(1 for px in diff_data if px != (0, 0, 0))
                    pct = diff_pixels / total_pixels * 100
                    if pct >= sensitivity:
                        self.prev_pixel_snapshot = current
                        self.set_indicator_color("green")
                        return True
                if log_enabled:
                    self.log(f"像素检测第{attempt + 1}次：变化 {pct:.2f}%")
                self.prev_pixel_snapshot = current
            except Exception as exc:
                self.log(f"像素检测失败: {exc}")
                break
            attempt += 1
            time.sleep(max(0.05, interval))
        return False

    def test_pixel_detection(self):
        if not self.pixel_detection_region:
            messagebox.showerror("错误", "请先设置像素检测区域")
            return

        def worker():
            try:
                self.capture_pixel_baseline()
            except Exception as exc:
                self.log(f"测试像素检测失败: {exc}")
                self.after(0, lambda: messagebox.showerror("错误", str(exc)))
                return
            try:
                timeout = max(0.5, float(self.pixel_timeout_var.get()))
                interval = max(0.05, float(self.pixel_interval_var.get()) / 1000.0)
                sensitivity = float(self.pixel_sensitivity_var.get())
            except (tk.TclError, ValueError) as exc:
                self.after(0, lambda: messagebox.showerror("错误", f"像素检测参数格式错误: {exc}"))
                return
            self.log("测试像素检测中…")
            detected = self.wait_for_pixel_change(None, timeout, interval, sensitivity, True)

            def report():
                if detected:
                    messagebox.showinfo("像素检测", "检测到像素变化")
                else:
                    messagebox.showwarning("像素检测", "未检测到像素变化，可调整阈值后重试")

            self.after(0, report)

        threading.Thread(target=worker, daemon=True).start()

    def perform_sim_click(self, pos: Optional[tuple] = None) -> bool:
        target = pos if pos is not None else self.sim_click_pos
        if not target:
            self.log("模拟点击点未设置，跳过点击")
            return False
        x, y = int(target[0]), int(target[1])
        try:
            pyautogui.click(x, y)
            self.log(f"已模拟点击: ({x}, {y})")
            return True
        except Exception as exc:
            self.log(f"执行模拟点击失败: {exc}")
            return False

    def build_plan(self):
        mode = self.mode_var.get()
        temps = self.parse_float_list(self.temps_var.get())
        pressures = self.parse_float_list(self.pressures_var.get())
        ramps = self.parse_ramps()
        def safe_int(var, default=None):
            try:
                return int(var.get())
            except (tk.TclError, ValueError):
                return default

        def safe_float(var, default=None):
            try:
                return float(var.get())
            except (tk.TclError, ValueError):
                return default

        pixel_enabled = bool(self.pixel_detection_enabled.get())
        interval_ms = safe_int(self.pixel_interval_var, None)
        timeout_s = safe_float(self.pixel_timeout_var, None)
        sensitivity = safe_float(self.pixel_sensitivity_var, None)
        if pixel_enabled:
            if (
                interval_ms is None
                or interval_ms <= 0
                or timeout_s is None
                or timeout_s <= 0
                or sensitivity is None
            ):
                raise ValueError("像素检测参数格式错误")
        else:
            if interval_ms is None:
                interval_ms = 500
            if timeout_s is None:
                timeout_s = 30.0
            if sensitivity is None:
                sensitivity = 5.0
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
            "sim_click_enabled": bool(self.sim_click_enabled.get()),
            "sim_click_pos": tuple(self.sim_click_pos) if self.sim_click_pos else None,
            "pixel_detection_enabled": pixel_enabled,
            "pixel_region": tuple(self.pixel_detection_region) if self.pixel_detection_region else None,
            "pixel_interval": max(0.05, interval_ms / 1000.0),
            "pixel_timeout": max(0.0, timeout_s),
            "pixel_sensitivity": max(0.0, sensitivity),
            "pixel_log_enabled": bool(self.pixel_log_enabled.get()),
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
        if self.sim_click_enabled.get() and not self.sim_click_pos:
            messagebox.showerror("错误", "请先设置模拟点击点")
            return
        if self.pixel_detection_enabled.get() and not self.pixel_detection_region:
            messagebox.showerror("错误", "请先设置像素检测区域")
            return
        try:
            plan = self.build_plan()
        except Exception as exc:
            messagebox.showerror("参数错误", str(exc))
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
