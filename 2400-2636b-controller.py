"""2400 / 2636B 仪器控制与数据采集入口。

整体逻辑：
- 使用 `KeithleyInstrument` 封装 Keithley 2400 / 2636B 的连接与仿真，实现 VISA/串口/GPIB 的初始化、读写与简易错误兜底。
- 以 Tkinter UI 触发测试流程和文件操作，读写 CSV/JSON 配置并驱动仪器扫描/输出曲线。
- 提供仿真模式生成对称肖特基 I-V 曲线，确保无硬件也可演示数据流与绘图。

主要函数/类说明：
- `KeithleyInstrument`：管理仪器连接、读写命令以及仿真数据生成（内置 2400 / 2636B 双机型支持）。
- 与 GUI 交互的各类回调函数：处理连接、远端感测开关、扫压/扫流、文件保存/加载等用户动作。
- 绘图/数据处理函数：更新曲线、导出 OFR/OHT 数据，支持点/线/点线样式及中英文混合界面。
"""

import csv
import socket
import threading
import time
import math
import random
import os
import json
import copy
import queue
import datetime
import statistics
from collections import defaultdict
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np

try:
    import pyvisa
except Exception:
    pyvisa = None


class KeithleyInstrument:
    """封装 Keithley 2400 / 2636B 仪器，支持仿真模式（对称肖特基 I-V）"""

    def __init__(self):
        self.rm = None
        if pyvisa is not None:
            try:
                self.rm = pyvisa.ResourceManager()
            except Exception:
                self.rm = None
        self.session = None
        self.simulated = True
        self.lock = threading.RLock()
        self.last_setpoint = 0.0  # 用于仿真模型中的电压
        self.conn_type = "仿真"    # 连接类型描述（仿真 / RS-232 / GPIB / USB / VISA）
        self.remote_sense = False  # 是否开启四线制（远端感测）
        self.model = None          # 根据 *IDN? 粗略判断机型（"2400" / "2636B"/ 其他）
        self.forced_model = None   # 用户指定的型号覆盖（None / "2400" / "2636B"）
        self.channel = "A"        # 兼容旧字段，等同于 source_channel
        self.source_channel = "A"
        self.measure_channel = "A"
        self.low_current_speed_mode = False
        self.current_range_override = None
        self._low_current_applied = False
        self._low_current_snapshot = {}
        self.log_callback = None

    def list_resources(self):
        if self.rm is None:
            return []
        try:
            return list(self.rm.list_resources())
        except Exception:
            return []

    def set_forced_model(self, model_str: str | None):
        if model_str in ("2400", "2636B"):
            self.forced_model = model_str
        else:
            self.forced_model = None

    def set_channel(self, ch: str):
        self.set_source_channel(ch)
        self.set_measure_channel(ch)

    def set_source_channel(self, ch: str):
        if str(ch).upper() == "B":
            self.source_channel = "B"
        else:
            self.source_channel = "A"
        self.channel = self.source_channel

    def set_measure_channel(self, ch: str):
        if str(ch).upper() == "B":
            self.measure_channel = "B"
        else:
            self.measure_channel = "A"

    def _source_ch(self) -> str:
        return "smub" if str(self.source_channel).upper() == "B" else "smua"

    def _measure_ch(self) -> str:
        return "smub" if str(self.measure_channel).upper() == "B" else "smua"

    def _ch(self) -> str:
        # 默认仍返回源通道，保持旧接口兼容
        return self._source_ch()

    def connect(self, address, simulate=False, baud_rate: int | None = None):
        """
        address: VISA 资源字符串，例如 'GPIB0::24::INSTR' 或 'ASRL3::INSTR'
        simulate: True 则不连真机，进入仿真模式
        """
        with self.lock:
            # 先关掉旧连接
            if self.session is not None:
                try:
                    self.session.close()
                except Exception:
                    pass
                self.session = None

            # 仿真模式：不连任何设备
            if simulate or self.rm is None or not address:
                self.simulated = True
                self.conn_type = "仿真"
                self.model = self.forced_model or self.model or "unknown"
                return "仿真模式（未连接仪器）"

            self.simulated = False
            try:
                self.session = self.rm.open_resource(address, timeout=5000)

                addr_upper = address.upper()
                # 串口 RS-232
                if "ASRL" in addr_upper:
                    self.conn_type = "RS-232"
                    try:
                        if baud_rate:
                            self.session.baud_rate = int(baud_rate)
                        else:
                            self.session.baud_rate = 9600
                        self.session.data_bits = 8
                        self.session.stop_bits = 1
                        # 对于 pyvisa，parity 一般用枚举；兜底用 0
                        try:
                            self.session.parity = pyvisa.constants.Parity.none
                        except Exception:
                            self.session.parity = 0
                    except Exception:
                        # 某些后端不支持这些属性，忽略即可
                        pass
                    self.session.write_termination = "\n"
                    self.session.read_termination = "\n"
                # GPIB
                elif addr_upper.startswith("GPIB"):
                    self.conn_type = "GPIB"
                    self.session.write_termination = "\n"
                    self.session.read_termination = "\n"
                # USB（USB-TMC / USB-GPIB）
                elif addr_upper.startswith("USB"):
                    self.conn_type = "USB"
                    self.session.write_termination = "\n"
                    self.session.read_termination = "\n"
                else:
                    self.conn_type = "VISA"
                    self.session.write_termination = "\n"
                    self.session.read_termination = "\n"

                # 初始化 SMU：优先按 2636B 的 TSP 方式处理，同时兼容老 2400
                # 先做通用清状态
                try:
                    self.session.write("*CLS")
                except Exception:
                    pass

                ident = ""
                try:
                    ident = self.session.query("*IDN?").strip()
                except Exception:
                    ident = "Unknown SMU"

                # 根据 IDN 判断机型，简单区分 2400 / 2636B
                model = "unknown"
                if "2636" in ident:
                    model = "2636B"
                elif "2400" in ident:
                    model = "2400"
                if self.forced_model in ("2400", "2636B"):
                    model = self.forced_model
                self.model = model

                # 对 2636B：使用 TSP reset()/smua.reset()
                if model == "2636B":
                    try:
                        self.session.write("reset()")
                    except Exception:
                        try:
                            self.session.write("smua.reset()")
                        except Exception:
                            pass
                    if "B" in {self.source_channel.upper(), self.measure_channel.upper()}:
                        try:
                            self.session.write("smub.reset()")
                        except Exception:
                            pass
                else:
                    # 其他（包括老 2400），保持原来的 2400 初始化逻辑
                    try:
                        self.session.write("*RST")
                        self.session.write("FORM:ELEM VOLT,CURR")
                        self.session.write("SENS:FUNC 'CURR'")
                    except Exception:
                        # 某些机型在非 2400 下可能不接受这些命令，可以忽略
                        pass

                return f"已连接: {ident} [{self.conn_type}]"
            except Exception as exc:
                # 回退仿真
                self.simulated = True
                self.conn_type = "仿真"
                if self.session is not None:
                    try:
                        self.session.close()
                    except Exception:
                        pass
                    self.session = None
                return f"连接失败，已切换到仿真模式: {exc}"

    def close(self):
        with self.lock:
            try:
                if self.session is not None:
                    try:
                        self.session.close()
                    except Exception:
                        pass
                    finally:
                        self.session = None
            finally:
                try:
                    self.set_low_current_mode(False)
                except Exception:
                    pass
                self._low_current_snapshot.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def configure_source(self, mode, level, compliance):
        """设置源模式 + 电平，并记录 last_setpoint 用于仿真"""
        with self.lock:
            self.last_setpoint = float(level)
            if self.simulated or self.session is None:
                return

            # 尝试把参数转成 float，避免字符串拼接导致异常
            try:
                level_val = float(level)
                comp_val = float(compliance)
            except Exception:
                return

            model = getattr(self, "model", None)

            try:
                if model == "2636B":
                    # 2636B：使用 TSP / smua 通道
                    self._apply_low_current_speed_settings_2636()
                    ch = self._source_ch()
                    if mode == "Voltage":
                        self.session.write(f"{ch}.source.func = {ch}.OUTPUT_DCVOLTS")
                        self.session.write(f"{ch}.source.levelv = {level_val}")
                        self.session.write(f"{ch}.source.limiti = {comp_val}")
                    else:
                        self.session.write(f"{ch}.source.func = {ch}.OUTPUT_DCAMPS")
                        self.session.write(f"{ch}.source.leveli = {level_val}")
                        self.session.write(f"{ch}.source.limitv = {comp_val}")
                    self.session.write(f"{ch}.source.output = {ch}.OUTPUT_ON")
                else:
                    # 默认路径：保留原 2400 SCPI 行为
                    self._apply_low_current_speed_settings_2400()
                    src = "VOLT" if mode == "Voltage" else "CURR"
                    self.session.write(f"SOUR:FUNC {src}")
                    self.session.write(f"SOUR:{src} {level_val}")
                    if src == "VOLT":
                        self.session.write(f"SENS:CURR:PROT {comp_val}")
                    else:
                        self.session.write(f"SENS:VOLT:PROT {comp_val}")
                    self.session.write("OUTP ON")
            except Exception:
                # 避免底层异常炸掉上层流程
                pass

    def prepare_source_2636(self, mode, compliance):
        """为 2636B 进行一次性源配置，减少循环内重复命令。"""
        with self.lock:
            if self.simulated or self.session is None:
                return

            try:
                comp_val = float(compliance)
            except Exception:
                return

            try:
                ch = self._source_ch()
                if mode == "Voltage":
                    self.session.write(f"{ch}.source.func = {ch}.OUTPUT_DCVOLTS")
                    self.session.write(f"{ch}.source.limiti = {comp_val}")
                else:
                    self.session.write(f"{ch}.source.func = {ch}.OUTPUT_DCAMPS")
                    self.session.write(f"{ch}.source.limitv = {comp_val}")
                self.session.write(f"{ch}.source.output = {ch}.OUTPUT_ON")
                self._apply_low_current_speed_settings_2636()
            except Exception:
                pass

    def set_level_2636(self, mode, level):
        """仅设置 2636B 源电平，避免重复配置其他属性。"""
        with self.lock:
            try:
                level_val = float(level)
            except Exception:
                return

            self.last_setpoint = level_val

            if self.simulated or self.session is None:
                return

            try:
                if mode == "Voltage":
                    self.session.write(f"{self._source_ch()}.source.levelv = {level_val}")
                else:
                    self.session.write(f"{self._source_ch()}.source.leveli = {level_val}")
            except Exception:
                # 不让底层异常直接炸掉上层流程
                pass

    def set_remote_sense(self, enable: bool):
        """开启或关闭远端感测（四线制）"""
        with self.lock:
            self.remote_sense = bool(enable)
            if self.simulated or self.session is None:
                return

            model = getattr(self, "model", None)
            try:
                if model == "2636B":
                    # 2636B：使用 smua.sense
                    ch = self._measure_ch()
                    if enable:
                        self.session.write(f"{ch}.sense = {ch}.SENSE_REMOTE")
                    else:
                        self.session.write(f"{ch}.sense = {ch}.SENSE_LOCAL")
                else:
                    # 默认路径：保留原 2400 行为
                    cmd = "ON" if enable else "OFF"
                    self.session.write(f"SYST:RSEN {cmd}")
            except Exception:
                pass

    def set_nplc(self, nplc: float):
        """设置采样积分时间（NPLC）。"""
        with self.lock:
            if self.simulated or self.session is None:
                return
            try:
                nplc_val = float(nplc)
            except Exception:
                return
            if nplc_val <= 0:
                return

            nplc_val = max(0.01, min(nplc_val, 10.0))

            model = getattr(self, "model", None)
            try:
                if model == "2636B":
                    # 2636B：统一用 smua.measure.nplc
                    self.session.write(f"{self._measure_ch()}.measure.nplc = {nplc_val}")
                else:
                    # 默认路径：保留原 2400 行为
                    self.session.write(f"SENS:CURR:NPLC {nplc_val}")
                    self.session.write(f"SENS:VOLT:NPLC {nplc_val}")
            except Exception:
                pass

    def _warn(self, msg: str):
        try:
            if callable(self.log_callback):
                self.log_callback(f"警告: {msg}")
                return
        except Exception:
            pass
        print(f"[WARN] {msg}")

    def set_low_current_mode(self, enable: bool):
        """切换低电流模式，负责快照采集/恢复。"""
        if enable:
            self.low_current_speed_mode = True
            self._low_current_applied = False
            self._low_current_snapshot = {}
            return

        # 关闭低电流模式，尝试恢复快照
        if self.low_current_speed_mode and self._low_current_applied:
            try:
                self._restore_low_current_snapshot()
            except Exception as exc:  # noqa: BLE001
                self._warn(f"恢复低电流前快照失败: {exc}")
        self.low_current_speed_mode = False
        self._low_current_applied = False
        self._low_current_snapshot = {}

    def _capture_low_current_snapshot_2636(self):
        if self._low_current_snapshot or self.simulated or self.session is None:
            return
        ch = self._measure_ch()
        snapshot = {}
        queries = {
            "autorangei": f"print({ch}.measure.autorangei)",
            "rangei": f"print({ch}.measure.rangei)",
            "autozero": f"print({ch}.measure.autozero)",
            "filter": f"print({ch}.measure.filter.enable)",
            "nplc": f"print({ch}.measure.nplc)",
            "sense": f"print({ch}.sense)",
        }
        for key, cmd in queries.items():
            try:
                snapshot[key] = self.session.query(cmd).strip()
            except Exception as exc:  # noqa: BLE001
                self._warn(f"读取 {key} 快照失败: {exc}")
        self._low_current_snapshot = snapshot

    def _capture_low_current_snapshot_2400(self):
        if self._low_current_snapshot or self.simulated or self.session is None:
            return
        snapshot = {}
        queries = {
            "autorange": "SENS:CURR:RANG:AUTO?",
            "range": "SENS:CURR:RANG?",
            "autozero": "SYST:AZER?",
            "filter": "SENS:AVER:STAT?",
            "sense_func": "SENS:FUNC?",
            "nplc_curr": "SENS:CURR:NPLC?",
            "nplc_volt": "SENS:VOLT:NPLC?",
        }
        for key, cmd in queries.items():
            try:
                snapshot[key] = self.session.query(cmd).strip()
            except Exception as exc:  # noqa: BLE001
                self._warn(f"读取 {key} 快照失败: {exc}")
        self._low_current_snapshot = snapshot

    def _restore_low_current_snapshot(self):
        if not self._low_current_snapshot or self.simulated or self.session is None:
            return
        model = getattr(self, "model", None)
        if model == "2636B":
            self._restore_low_current_snapshot_2636()
        else:
            self._restore_low_current_snapshot_2400()

    def _restore_low_current_snapshot_2636(self):
        ch = self._measure_ch()
        snap = self._low_current_snapshot
        if not snap:
            return
        restorers = {
            "autorangei": lambda v: self.session.write(f"{ch}.measure.autorangei = {v}"),
            "rangei": lambda v: self.session.write(f"{ch}.measure.rangei = {float(v)}"),
            "autozero": lambda v: self.session.write(f"{ch}.measure.autozero = {v}"),
            "filter": lambda v: self.session.write(f"{ch}.measure.filter.enable = {v}"),
            "nplc": lambda v: self.session.write(f"{ch}.measure.nplc = {float(v)}"),
            "sense": lambda v: self.session.write(f"{ch}.sense = {v}"),
        }
        for key, action in restorers.items():
            if key not in snap:
                continue
            try:
                action(snap[key])
            except Exception as exc:  # noqa: BLE001
                self._warn(f"恢复 {key} 失败: {exc}")

    def _restore_low_current_snapshot_2400(self):
        snap = self._low_current_snapshot
        if not snap:
            return
        restorers = {
            "autorange": lambda v: self.session.write(f"SENS:CURR:RANG:AUTO {v}"),
            "range": lambda v: self.session.write(f"SENS:CURR:RANG {float(v)}"),
            "autozero": lambda v: self.session.write(f"SYST:AZER {v}"),
            "filter": lambda v: self.session.write(f"SENS:AVER:STAT {v}"),
            "sense_func": lambda v: self.session.write(f"SENS:FUNC {v}"),
            "nplc_curr": lambda v: self.session.write(f"SENS:CURR:NPLC {float(v)}"),
            "nplc_volt": lambda v: self.session.write(f"SENS:VOLT:NPLC {float(v)}"),
        }
        for key, action in restorers.items():
            if key not in snap:
                continue
            try:
                action(snap[key])
            except Exception as exc:  # noqa: BLE001
                self._warn(f"恢复 {key} 失败: {exc}")

    def _apply_low_current_speed_settings_2636(self):
        if self._low_current_applied or not self.low_current_speed_mode:
            return
        if self.simulated or self.session is None:
            return
        self._capture_low_current_snapshot_2636()
        range_val = self.current_range_override
        try:
            ch = self._measure_ch()
            self.session.write(f"{ch}.measure.autorangei = {ch}.AUTORANGE_OFF")
        except Exception as exc:
            self._warn(f"关闭电流自动量程失败: {exc}")
        if range_val:
            try:
                self.session.write(f"{self._measure_ch()}.measure.rangei = {float(range_val)}")
            except Exception as exc:
                self._warn(f"设置固定电流量程失败: {exc}")
        try:
            ch = self._measure_ch()
            self.session.write(f"{ch}.measure.autozero = {ch}.AUTOZERO_OFF")
        except Exception as exc:
            self._warn(f"关闭 AutoZero 失败: {exc}")
        try:
            self.session.write(f"{self._measure_ch()}.measure.filter.enable = 0")
        except Exception as exc:
            self._warn(f"关闭数字滤波失败: {exc}")
        self._low_current_applied = True

    def _apply_low_current_speed_settings_2400(self):
        if self._low_current_applied or not self.low_current_speed_mode:
            return
        if self.simulated or self.session is None:
            return
        self._capture_low_current_snapshot_2400()
        range_val = self.current_range_override
        try:
            self.session.write("SENS:CURR:RANG:AUTO OFF")
        except Exception as exc:
            self._warn(f"关闭电流自动量程失败: {exc}")
        if range_val:
            try:
                self.session.write(f"SENS:CURR:RANG {float(range_val)}")
            except Exception as exc:
                self._warn(f"设置固定电流量程失败: {exc}")
        try:
            self.session.write("SYST:AZER OFF")
        except Exception as exc:
            self._warn(f"关闭 AutoZero 失败: {exc}")
        try:
            self.session.write("SENS:AVER:STAT OFF")
        except Exception as exc:
            self._warn(f"关闭平均滤波失败: {exc}")
        self._low_current_applied = True

    def output_off(self):
        with self.lock:
            if self.simulated or self.session is None:
                return

            model = getattr(self, "model", None)
            try:
                if model == "2636B":
                    src = self._source_ch()
                    self.session.write(f"{src}.source.output = {src}.OUTPUT_OFF")
                    meas = self._measure_ch()
                    if meas != src:
                        try:
                            self.session.write(f"{meas}.source.output = {meas}.OUTPUT_OFF")
                        except Exception:
                            pass
                else:
                    self.session.write("OUTP OFF")
            except Exception:
                pass

    def _simulate_symmetric_schottky(self):
        """
        对称肖特基 I-V 模型（简单版）
        """
        V = float(self.last_setpoint or 0.0)
        Vt = 0.02585  # ~ kT/q at 300K
        n = 1.5
        Is = 1e-6  # 1 µA

        if V >= 0:
            I = Is * (math.exp(V / (n * Vt)) - 1.0)
        else:
            I = -Is * (math.exp(-V / (n * Vt)) - 1.0)

        # 防止指数暴飞，做个钳位（10 mA 级别）
        I = max(min(I, 1e-2), -1e-2)

        # 加一点噪声
        noise_scale = 0.05 * abs(I) + 1e-8
        I += random.uniform(-noise_scale, noise_scale)
        V_meas = V + random.uniform(-0.002, 0.002)

        return V_meas, I

    def measure_once(self):
        with self.lock:
            if self.simulated or self.session is None:
                now = time.time()
                v, i = self._simulate_symmetric_schottky()
                return {
                    "timestamp": now,
                    "voltage": v,
                    "current": i,
                }

            model = getattr(self, "model", None)

            try:
                if model == "2636B":
                    # 2636B：单条 TSP 命令，直接把 smua.measure.iv() 的两个返回值打印出来
                    # 官方文档：smua.measure.iv() -> [current, voltage]
                    raw = self.session.query(f"print({self._measure_ch()}.measure.iv())").strip()
                else:
                    # 默认路径：沿用 2400 的 READ? + FORM:ELEM VOLT,CURR
                    raw = self.session.query("READ?").strip()

                # 统一解析：允许逗号或空格分隔
                raw_norm = raw.replace(",", " ")
                parts = [p for p in raw_norm.split() if p]

                # 如果里面有 nil，说明 2636B 那边测量没配好，给出更明确提示
                if any(p.lower() == "nil" for p in parts):
                    raise RuntimeError(
                        f"2636B 返回 nil，请检查是否已正确配置源输出、量程和接线: {raw!r}"
                    )

                if len(parts) < 2:
                    raise ValueError(f"仪器返回格式异常: {raw!r}")

                # 2400: FORM:ELEM VOLT,CURR -> [V, I]
                # 2636B: smua.measure.iv() -> [I, V]
                if model == "2636B":
                    current = float(parts[0])
                    voltage = float(parts[1])
                else:
                    voltage = float(parts[0])
                    current = float(parts[1])

            except Exception as exc:
                raise RuntimeError(f"采样失败: {exc}") from exc

            return {
                "timestamp": time.time(),
                "voltage": voltage,
                "current": current,
            }

    def buffer_sweep_2636(self, source_mode, compliance, levels, delay):
        """使用 2636B 内部缓冲区一次性采集多个点。"""
        with self.lock:
            if self.simulated or self.session is None:
                raise RuntimeError("仿真或未连接状态下不支持缓存模式")
            if not levels:
                return []
            try:
                comp_val = float(compliance)
            except Exception as exc:
                raise RuntimeError(f"保护值无效: {exc}") from exc

            src_ch = self._source_ch()
            meas_ch = self._measure_ch()
            func = f"{src_ch}.OUTPUT_DCVOLTS" if source_mode == "Voltage" else f"{src_ch}.OUTPUT_DCAMPS"
            level_field = f"{src_ch}.source.levelv" if source_mode == "Voltage" else f"{src_ch}.source.leveli"
            limit_field = f"{src_ch}.source.limiti" if source_mode == "Voltage" else f"{src_ch}.source.limitv"
            try:
                # 基础配置 + 低电流加速
                self.session.write(f"{src_ch}.source.func = {func}")
                self.session.write(f"{limit_field} = {comp_val}")
                self.session.write(f"{src_ch}.source.output = {src_ch}.OUTPUT_ON")
                self._apply_low_current_speed_settings_2636()
            except Exception as exc:
                raise RuntimeError(f"预配置 2636B 失败: {exc}") from exc

            levels_str = ",".join(str(float(v)) for v in levels)
            delay_val = max(0.0, float(delay or 0.0))
            script = """
local src = %s
local meas = %s
local lvls = {%s}
local out = {}
for i, v in ipairs(lvls) do
    %s = v
    local m = meas.measure.iv()
    out[#out + 1] = string.format("%%g,%%g", m[2], m[1])
    if %f > 0 then delay(%f) end
end
print(table.concat(out, ";"))
""" % (src_ch, meas_ch, levels_str, level_field, delay_val, delay_val)
            try:
                raw = self.session.query(script).strip()
            except Exception as exc:
                raise RuntimeError(f"执行缓存采集失败: {exc}") from exc

        parts = [p for p in raw.replace(";", ",").split(",") if p]
        if len(parts) % 2 != 0:
            raise RuntimeError(f"缓冲返回格式异常: {raw}")
        base_ts = time.time()
        readings = []
        for idx in range(0, len(parts), 2):
            try:
                voltage = float(parts[idx])
                current = float(parts[idx + 1])
            except Exception as exc:
                raise RuntimeError(f"解析缓存数据失败: {exc}") from exc
            readings.append(
                {
                    "timestamp": base_ts + delay_val * (idx // 2),
                    "voltage": voltage,
                    "current": current,
                }
            )
        return readings

    def buffer_sweep_2400(self, source_mode, compliance, levels, delay):
        """使用 2400 的内部缓冲区一次性采集多个点。"""
        with self.lock:
            if self.simulated or self.session is None:
                raise RuntimeError("仿真或未连接状态下不支持缓存模式")
            if not levels:
                return []
            try:
                comp_val = float(compliance)
            except Exception as exc:
                raise RuntimeError(f"保护值无效: {exc}") from exc

            delay_val = max(0.0, float(delay or 0.0))
            unique_levels = set(float(v) for v in levels)
            try:
                src = "VOLT" if source_mode == "Voltage" else "CURR"
                self.session.write(f"SOUR:FUNC {src}")
                if src == "VOLT":
                    self.session.write(f"SENS:CURR:PROT {comp_val}")
                else:
                    self.session.write(f"SENS:VOLT:PROT {comp_val}")
                self.session.write("FORM:ELEM VOLT,CURR")
                self._apply_low_current_speed_settings_2400()
                self.session.write("TRAC:CLE")
                self.session.write(f"TRAC:POIN {len(levels)}")
                self.session.write("TRAC:FEED SENS")
                self.session.write("TRAC:FEED:CONT NEXT")
                self.session.write(f"TRIG:COUN {len(levels)}")
                self.session.write(f"TRIG:DEL {delay_val}")
            except Exception as exc:
                raise RuntimeError(f"配置 2400 缓冲失败: {exc}") from exc

            try:
                if len(unique_levels) == 1:
                    # 固定电平重复采样
                    level_val = float(levels[0])
                    if src == "VOLT":
                        self.session.write(f"SOUR:VOLT {level_val}")
                    else:
                        self.session.write(f"SOUR:CURR {level_val}")
                    self.session.write("OUTP ON")
                    self.session.write("INIT")
                else:
                    # 线性扫
                    step = self._infer_linear_step(levels)
                    self.session.write(f"SOUR:{src}:START {float(levels[0])}")
                    self.session.write(f"SOUR:{src}:STOP {float(levels[-1])}")
                    self.session.write(f"SOUR:{src}:STEP {step}")
                    self.session.write(f"SOUR:{src}:MODE SWE")
                    self.session.write("OUTP ON")
                    self.session.write("INIT")
                raw = self.session.query(f"TRAC:DATA? 1, {len(levels)}, \"defbuffer1\"")
            except Exception as exc:
                raise RuntimeError(f"执行 2400 缓存采集失败: {exc}") from exc

        raw_norm = raw.replace(",", " ")
        parts = [p for p in raw_norm.split() if p]
        if len(parts) < 2:
            raise RuntimeError(f"2400 缓冲返回格式异常: {raw}")
        if len(parts) % 2 != 0:
            raise RuntimeError(f"2400 缓冲数据不成对: {raw}")
        base_ts = time.time()
        readings = []
        for idx in range(0, len(parts), 2):
            try:
                voltage = float(parts[idx])
                current = float(parts[idx + 1])
            except Exception as exc:
                raise RuntimeError(f"解析 2400 缓冲数据失败: {exc}") from exc
            readings.append(
                {
                    "timestamp": base_ts + delay_val * (idx // 2),
                    "voltage": voltage,
                    "current": current,
                }
            )
        return readings

    def _infer_linear_step(self, levels):
        if len(levels) < 2:
            raise RuntimeError("点数不足，无法推断步长")
        start = float(levels[0])
        stop = float(levels[-1])
        step = (stop - start) / (len(levels) - 1)
        if step == 0:
            raise RuntimeError("步长为 0，无法执行线性扫描")
        for idx, val in enumerate(levels[1:], start=1):
            expect = start + step * idx
            if abs(float(val) - expect) > max(1e-9, abs(expect) * 1e-6):
                raise RuntimeError("点序列不是等步长，无法使用内置扫")
        return step

    def sweep_points(self, start, stop, count):
        return np.linspace(start, stop, max(2, int(count)))


from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("2400 / 2636B 扫描工具")

        # 启动尽量最大化
        try:
            self.root.state("zoomed")
        except Exception:
            try:
                self.root.attributes("-zoomed", True)
            except Exception:
                pass

        self.root.rowconfigure(1, weight=1)
        self.root.columnconfigure(0, weight=1)

        self._setup_modern_style()

        self.instrument = KeithleyInstrument()
        self.instrument.log_callback = self._log
        self.queue = queue.Queue()
        self.measurement_thread = None
        self.thread_semaphore = threading.Semaphore(1)
        self._measurement_lock_acquired = False
        self.is_measuring = False
        self.stop_event = threading.Event()
        self.tcp_stop_event = threading.Event()
        self.tcp_server_thread = None
        self.integration_time_var = tk.DoubleVar(value=0.0)  # 硬件积分时间（NPLC）
        self.low_current_speed_mode_var = tk.BooleanVar(value=False)
        self.current_range_override_var = tk.StringVar(value="1e-6")
        self.model_select_var = tk.StringVar(value="自动识别")
        self.channel_select_var = tk.StringVar(value="A")
        self.source_channel_var = tk.StringVar(value="A")
        self.measure_channel_var = tk.StringVar(value="A")
        self.buffer_mode_var = tk.BooleanVar(value=False)
        self.baud_rate_var = tk.StringVar(value="9600")
        self._filtered_pressure = None                       # 压力最新值（保留原接口）
        self._filtered_pressure_ts = None                    # 压力更新时间戳
        self.current_mode = None  # "IV", "It", "Vt", "Rt", "Pt"
        self.current_data = []
        self.total_points = 0
        self.completed_points = 0
        self.start_time = None
        self._low_current_range_widgets = []
        self.tcp_waiters = []
        self.tcp_waiters_lock = threading.Lock()
        self.multi_tcp_active = False
        self.multi_tcp_pressure = None
        self.multi_tcp_pending_pressure = None
        self.multi_tcp_records = []  # [(pressure_g, path, is_bad)]
        self.multi_tcp_session_start = None
        self.multi_tcp_retry_used = 0
        self.multi_tcp_last_iv_config = None

        # OFR 测试状态
        self.ofr_active = False
        self.ofr_test_id = ""
        self.ofr_raw_points = []
        self.ofr_off_points = []
        self.ofr_I_off = None
        self.ofr_stats = defaultdict(lambda: [0, 0.0])
        self.ofr_I_mean_by_pressure = {}
        self.ofr_pressures = []
        self.ofr_onoff_values = []
        self.ofr_line = None
        self.ofr_noise_k = 3.0
        self.ofr_instr_floor = 1e-12
        self.ofr_samples = []
        self.ofr_t0 = None

        # OFR 仿真线程状态
        self.ofr_sim_thread = None
        self.ofr_sim_stop = threading.Event()

        # 压力相关属性（主要用于兼容压力积分入口，默认不启用）
        self.modbus1 = None
        self.pressure_scale = 1.0
        self.tare_value = 0.0
        self.current_pressure = 0.0

        self.config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json")

        self._build_ui()
        self._load_settings()
        self._start_tcp_server()
        self._poll_queue()

    def _setup_modern_style(self):
        """设置现代化界面样式"""
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except Exception:
            pass

        colors = {
            "primary": "#007bff",
            "secondary": "#6c757d",
            "success": "#28a745",
            "danger": "#dc3545",
            "warning": "#ffc107",
            "info": "#17a2b8",
            "light": "#f8f9fa",
            "dark": "#343a40",
        }

        style.configure("TLabel", font=("Segoe UI", 9))
        style.configure("TButton", font=("Segoe UI", 9), padding=6)
        style.configure("TEntry", padding=5)
        style.configure("TCombobox", padding=5)
        style.configure("TCheckbutton", font=("Segoe UI", 9))
        style.configure("TRadiobutton", font=("Segoe UI", 9))
        style.configure("TNotebook.Tab", font=("Segoe UI", 9, "bold"), padding=[10, 5])
        style.configure("TLabelframe", background=colors["light"], relief="solid")
        style.configure("TLabelframe.Label", font=("Segoe UI", 10, "bold"))

        style.configure(
            "Horizontal.TProgressbar",
            background=colors["primary"],
            troughcolor=colors["light"],
            bordercolor=colors["light"],
            lightcolor=colors["primary"],
            darkcolor=colors["primary"],
            thickness=12,
        )
        style.configure(
            "Treeview",
            font=("Segoe UI", 9),
            rowheight=22,
            background=colors["light"],
            fieldbackground=colors["light"],
        )
        style.map(
            "Treeview",
            background=[("selected", colors["primary"])],
            foreground=[("selected", "white")],
        )
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"), padding=8)
        style.configure("Success.TButton", background=colors["success"], foreground="white")
        style.configure("Danger.TButton", background=colors["danger"], foreground="white")

    def _build_ui(self):
        # 背景与根布局
        self.root.configure(bg="#f0f0f0")
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)

        # 主分栏
        main_pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_pane.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

        left_frame = ttk.Frame(main_pane)
        right_frame = ttk.Frame(main_pane)
        main_pane.add(left_frame, weight=1)
        main_pane.add(right_frame, weight=3)

        self._build_left_panel(left_frame)
        self._build_right_panel(right_frame)

        # TCP 区域保持在底部
        tcp_lf = ttk.Labelframe(self.root, text="TCP 从机", padding=8)
        tcp_lf.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 8))
        tcp_lf.columnconfigure(3, weight=1)

        ttk.Label(tcp_lf, text="监听 IP:").grid(row=0, column=0, sticky="w")
        self.tcp_host_var = tk.StringVar(value="127.0.0.1")
        ttk.Entry(tcp_lf, textvariable=self.tcp_host_var, width=16).grid(
            row=0, column=1, sticky="w", padx=(4, 12)
        )

        ttk.Label(tcp_lf, text="端口:").grid(row=0, column=2, sticky="w")
        self.tcp_port_var = tk.IntVar(value=50000)
        ttk.Entry(tcp_lf, textvariable=self.tcp_port_var, width=10).grid(
            row=0, column=3, sticky="w", padx=(4, 12)
        )

        ttk.Button(tcp_lf, text="应用", command=self.apply_tcp_settings).grid(
            row=0, column=4, sticky="w", padx=(4, 0)
        )

        self._sync_model_channel_controls()
        self._sync_baud_control()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_left_panel(self, parent):
        left_notebook = ttk.Notebook(parent)
        left_notebook.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        conn_frame = self._build_connection_frame(left_notebook)
        left_notebook.add(conn_frame, text="连接设置")

        params_frame = self._build_parameters_frame(left_notebook)
        left_notebook.add(params_frame, text="测量参数")

        adv_frame = self._build_advanced_frame(left_notebook)
        left_notebook.add(adv_frame, text="高级设置")

        log_frame = ttk.Labelframe(parent, text="日志", padding=6)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=(0, 2))
        log_frame.rowconfigure(1, weight=1)
        log_frame.columnconfigure(0, weight=1)

        ttk.Label(log_frame, text="输出:").grid(row=0, column=0, sticky="w")
        self.log_text = tk.Text(log_frame, height=12, wrap="word")
        self.log_text.grid(row=1, column=0, sticky="nsew")

    def _build_right_panel(self, parent):
        self._build_control_buttons(parent)

        chart_frame = ttk.Frame(parent)
        chart_frame.pack(fill=tk.BOTH, expand=True)

        toolbar_frame = ttk.Frame(chart_frame)
        toolbar_frame.pack(fill=tk.X, pady=(0, 5))

        ttk.Label(toolbar_frame, text="曲线样式:").pack(side=tk.LEFT, padx=(0, 5))
        self.plot_style_var = tk.StringVar(value="线")
        style_combo = ttk.Combobox(
            toolbar_frame,
            textvariable=self.plot_style_var,
            values=["线", "点", "线+点"],
            state="readonly",
            width=8,
        )
        style_combo.pack(side=tk.LEFT, padx=(0, 10))
        style_combo.bind("<<ComboboxSelected>>", lambda e: self._apply_plot_style())

        self.fig = Figure(figsize=(6, 5))
        self.ax = self.fig.add_subplot(111)
        self.fig.set_tight_layout(True)
        self.ax.set_title("Live measurement")
        self.ax.set_xlabel("Point index")
        self.ax.set_ylabel("Value")
        self.ax.grid(True, alpha=0.3, linestyle="--")
        self.voltage_line, = self.ax.plot([], [], label="Voltage (V)")
        self.current_line, = self.ax.plot([], [], label="Current (A)")
        self.ofr_line, = self.ax.plot([], [], "o-", label="ON/OFF", color="#e67e22")

        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

        stats_frame = ttk.Frame(chart_frame)
        stats_frame.pack(fill=tk.X, pady=(4, 0))
        self.points_label = ttk.Label(stats_frame, text="点数: 0/0")
        self.points_label.pack(side=tk.LEFT)
        self.eta_label = ttk.Label(stats_frame, text="剩余时间: --")
        self.eta_label.pack(side=tk.LEFT, padx=(10, 0))

    def _build_control_buttons(self, parent):
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill=tk.X, pady=(0, 8))

        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(fill=tk.X)

        self.start_button = ttk.Button(
            btn_frame, text="▶ 开始测量", command=self.start_measurement, style="Success.TButton", width=12
        )
        self.start_button.pack(side=tk.LEFT, padx=(0, 8))

        self.stop_button = ttk.Button(
            btn_frame, text="⏹ 停止", command=self.stop_measurement, style="Danger.TButton", width=10, state="disabled"
        )
        self.stop_button.pack(side=tk.LEFT, padx=(0, 8))

        ttk.Button(btn_frame, text="📊 导出数据", command=self.export_data, width=10).pack(
            side=tk.LEFT, padx=(0, 6)
        )
        ttk.Button(btn_frame, text="📝 导出日志", command=self.export_log, width=10).pack(side=tk.LEFT)

        progress_frame = ttk.Frame(control_frame)
        progress_frame.pack(fill=tk.X, pady=(10, 0))

        ttk.Label(progress_frame, text="进度:").pack(side=tk.LEFT)
        self.progress = ttk.Progressbar(progress_frame, mode="determinate", maximum=100, length=260)
        self.progress.pack(side=tk.LEFT, padx=(5, 5), fill=tk.X, expand=True)
        self.progress_label = ttk.Label(progress_frame, text="0%", width=5)
        self.progress_label.pack(side=tk.LEFT)
        self.time_label = ttk.Label(progress_frame, text="剩余: --:--", width=10)
        self.time_label.pack(side=tk.LEFT, padx=(10, 0))

    def _build_connection_frame(self, parent):
        frame = ttk.Frame(parent, padding=10)
        frame.columnconfigure(0, weight=1)

        sections = [
            ("仪器连接", self._build_instrument_connection),
            ("通道设置", self._build_channel_settings),
            ("保存设置", self._build_save_settings),
        ]

        for i, (title, builder) in enumerate(sections):
            section_frame = ttk.LabelFrame(frame, text=title, padding=8)
            section_frame.grid(row=i, column=0, sticky="ew", pady=(0, 10))
            section_frame.columnconfigure(0, weight=1)
            builder(section_frame)

        return frame

    def _build_instrument_connection(self, parent):
        ttk.Label(parent, text="资源地址:").grid(row=0, column=0, sticky="w", padx=(0, 5))
        self.resource_combo = ttk.Combobox(parent, width=30, state="readonly")
        self.resource_combo.grid(row=0, column=1, sticky="ew", padx=(0, 5))
        self.resource_combo.bind("<<ComboboxSelected>>", lambda e: self._sync_baud_control())
        ttk.Button(parent, text="刷新", command=self.refresh_resources, width=8).grid(row=0, column=2, sticky="w")

        self.sim_var = tk.BooleanVar(value=True)
        sim_chk = ttk.Checkbutton(parent, text="仿真模式", variable=self.sim_var, command=self.on_sim_toggle)
        sim_chk.grid(row=1, column=0, columnspan=2, sticky="w", pady=(8, 0))

        self.baud_frame = ttk.Frame(parent)
        self.baud_frame.grid(row=2, column=0, columnspan=3, sticky="w", pady=(8, 0))
        ttk.Label(self.baud_frame, text="波特率:").pack(side=tk.LEFT, padx=(0, 5))
        self.baud_combo = ttk.Combobox(
            self.baud_frame,
            width=10,
            state="readonly",
            values=["9600", "19200", "57600", "115200"],
            textvariable=self.baud_rate_var,
        )
        self.baud_combo.pack(side=tk.LEFT)

        ttk.Button(parent, text="连接仪器", command=self.connect_instrument, style="Accent.TButton").grid(
            row=3, column=0, columnspan=3, pady=(12, 0), sticky="ew"
        )

        status_frame = ttk.Frame(parent)
        status_frame.grid(row=4, column=0, columnspan=3, pady=(10, 0), sticky="ew")
        ttk.Label(status_frame, text="状态:", font=("", 9, "bold")).pack(side=tk.LEFT)
        self.status_label = ttk.Label(status_frame, text="未连接（仿真）", foreground="gray")
        self.status_label.pack(side=tk.LEFT, padx=(5, 0))

    def _build_channel_settings(self, parent):
        row = 0
        ttk.Label(parent, text="仪器型号:").grid(row=row, column=0, sticky="w")
        self.model_combo = ttk.Combobox(
            parent,
            width=12,
            state="readonly",
            textvariable=self.model_select_var,
            values=["自动识别", "2400", "2636B"],
        )
        self.model_combo.grid(row=row, column=1, sticky="w", padx=(6, 0))
        self.model_combo.bind("<<ComboboxSelected>>", lambda e: self._sync_model_channel_controls())
        row += 1

        self.channel_label = ttk.Label(parent, text="2636B 通道:")
        self.channel_label.grid(row=row, column=0, sticky="w", pady=(6, 0))
        self.channel_combo = ttk.Combobox(
            parent,
            width=6,
            state="readonly",
            textvariable=self.channel_select_var,
            values=["A", "B"],
        )
        self.channel_combo.grid(row=row, column=1, sticky="w", padx=(6, 0), pady=(6, 0))
        self.channel_combo.bind("<<ComboboxSelected>>", lambda e: self._apply_channel_selection_to_instrument())
        row += 1

        four_wire_frame = ttk.Frame(parent)
        four_wire_frame.grid(row=row, column=0, columnspan=2, sticky="w", pady=(6, 0))
        self.four_wire_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            four_wire_frame,
            text="四线制",
            variable=self.four_wire_var,
            command=self.on_four_wire_toggle,
        ).pack(side=tk.LEFT)

        self.four_wire_channel_frame = ttk.Frame(parent)
        self.four_wire_channel_frame.grid(row=row + 1, column=0, columnspan=2, sticky="w", pady=(4, 0))
        self.source_channel_label = ttk.Label(self.four_wire_channel_frame, text="源通道:")
        self.source_channel_label.pack(side=tk.LEFT, padx=(0, 4))
        self.source_channel_combo = ttk.Combobox(
            self.four_wire_channel_frame,
            width=5,
            state="readonly",
            textvariable=self.source_channel_var,
            values=["A", "B"],
        )
        self.source_channel_combo.pack(side=tk.LEFT, padx=(0, 8))
        self.source_channel_combo.bind("<<ComboboxSelected>>", lambda e: self._apply_channel_selection_to_instrument())

        self.measure_channel_label = ttk.Label(self.four_wire_channel_frame, text="测量通道:")
        self.measure_channel_label.pack(side=tk.LEFT, padx=(0, 4))
        self.measure_channel_combo = ttk.Combobox(
            self.four_wire_channel_frame,
            width=5,
            state="readonly",
            textvariable=self.measure_channel_var,
            values=["A", "B"],
        )
        self.measure_channel_combo.pack(side=tk.LEFT, padx=(0, 8))
        self.measure_channel_combo.bind("<<ComboboxSelected>>", lambda e: self._apply_channel_selection_to_instrument())

    def _build_save_settings(self, parent):
        ttk.Label(parent, text="保存根文件夹:").grid(row=0, column=0, sticky="w")
        self.save_root_var = tk.StringVar()
        self.save_root_entry = ttk.Entry(parent, textvariable=self.save_root_var, width=34)
        self.save_root_entry.grid(row=0, column=1, sticky="ew", pady=(0, 4), padx=(6, 0))
        ttk.Button(parent, text="浏览...", command=self.choose_save_root, width=8).grid(
            row=0, column=2, sticky="w", padx=(6, 0)
        )

        self.auto_save_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(parent, text="自动保存", variable=self.auto_save_var).grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(6, 0)
        )

    def _build_parameters_frame(self, parent):
        frame = ttk.Frame(parent, padding=6)
        frame.columnconfigure(0, weight=1)

        self.notebook = ttk.Notebook(frame)
        self.notebook.grid(row=0, column=0, sticky="nsew")
        self._build_iv_tab()
        self._build_it_tab()
        self._build_vt_tab()
        self._build_rt_tab()
        self._build_pt_tab()
        self._build_ofr_tab()

        return frame

    def _build_advanced_frame(self, parent):
        frame = ttk.Frame(parent, padding=10)
        frame.columnconfigure(0, weight=1)

        interval_frame = ttk.LabelFrame(frame, text="测量节奏", padding=8)
        interval_frame.grid(row=0, column=0, sticky="ew")
        ttk.Label(interval_frame, text="积分时间(NPLC):").grid(row=0, column=0, sticky="w")
        self.integration_time_entry = ttk.Entry(interval_frame, width=10, textvariable=self.integration_time_var)
        self.integration_time_entry.grid(row=0, column=1, sticky="w", padx=(6, 0))

        return frame

        self._sync_model_channel_controls()
        self._sync_baud_control()

    # ---- 各模式参数区 ----

    def _add_buffer_mode_control(self, parent, row):
        chk = ttk.Checkbutton(
            parent,
            text="缓存模式（仪器内部批量采集，更快）",
            variable=self.buffer_mode_var,
        )
        chk.grid(row=row, column=0, columnspan=4, sticky="w", pady=(0, 4))
        return row + 1

    def _add_low_current_controls(self, parent, row):
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=0, columnspan=4, sticky="w", pady=(6, 0))
        ttk.Checkbutton(
            frame,
            text="低电流加速模式（更快/更噪）",
            variable=self.low_current_speed_mode_var,
            command=self._on_low_current_toggle,
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            frame,
            text="启用后会关闭/限制自动量程、AutoZero/平均滤波，并偏向更小 NPLC，\n可能降低稳定性/精度。",
            foreground="#666",
            wraplength=320,
            justify="left",
        ).grid(row=1, column=0, sticky="w", pady=(2, 0))

        range_frame = ttk.Frame(parent)
        range_frame.grid(row=row + 1, column=0, columnspan=4, sticky="w")
        lbl = ttk.Label(range_frame, text="电流量程(A):")
        lbl.grid(row=0, column=0, sticky="e", pady=4, padx=(0, 4))
        entry = ttk.Entry(range_frame, textvariable=self.current_range_override_var, width=14)
        entry.grid(row=0, column=1, sticky="w", pady=4)
        self._low_current_range_widgets.append((range_frame, lbl, entry))
        self._sync_low_current_controls()
        return row + 2

    def _sync_low_current_controls(self):
        visible = bool(self.low_current_speed_mode_var.get())
        for frame, lbl, entry in self._low_current_range_widgets:
            if visible:
                frame.grid()
                lbl.grid()
                entry.grid()
            else:
                frame.grid_remove()
                lbl.grid_remove()
                entry.grid_remove()

    def _on_low_current_toggle(self):
        self._sync_low_current_controls()
        try:
            self.instrument.set_low_current_mode(bool(self.low_current_speed_mode_var.get()))
        except Exception:
            pass

    def _build_iv_tab(self):
        frame = ttk.Frame(self.notebook, padding=6)
        self.notebook.add(frame, text="IV 扫描")

        frame.columnconfigure(0, weight=1)
        inner = ttk.Frame(frame)
        inner.grid(row=0, column=0, pady=4)
        for col in range(4):
            weight = 0 if col in (1, 3) else 1
            inner.columnconfigure(col, weight=weight)

        self.iv_source_mode_var = tk.StringVar(value="Voltage")
        self.iv_start_var = tk.DoubleVar(value=-1.0)
        self.iv_stop_var = tk.DoubleVar(value=1.0)
        self.iv_step_var = tk.DoubleVar(value=0.02)
        self.iv_points_var = tk.IntVar(value=101)
        self.iv_cycles_var = tk.IntVar(value=1)
        self.iv_backforth_var = tk.BooleanVar(value=False)
        self.iv_triangle_from_zero_var = tk.BooleanVar(value=False)
        self.iv_delay_var = tk.DoubleVar(value=0.0)
        self.iv_compliance_var = tk.DoubleVar(value=0.1)
        self.iv_quality_k_var = tk.DoubleVar(value=8.0)
        self.iv_quality_jump_ratio_var = tk.DoubleVar(value=0.02)
        self.iv_quality_flip_count_var = tk.IntVar(value=20)
        self.iv_quality_max_retry_var = tk.IntVar(value=2)
        self.iv_quality_enabled_var = tk.BooleanVar(value=False)
        self._iv_updating = False

        row = 0
        row = self._add_buffer_mode_control(inner, row)
        ttk.Label(inner, text="源模式:").grid(row=row, column=0, sticky="e", pady=4, padx=(0, 4))
        mode_combo = ttk.Combobox(
            inner,
            textvariable=self.iv_source_mode_var,
            values=["Voltage", "Current"],
            state="readonly",
            width=10,
        )
        mode_combo.grid(row=row, column=1, sticky="w", pady=4, padx=(0, 10))

        ttk.Label(inner, text="循环次数:").grid(row=row, column=2, sticky="e", pady=4, padx=(0, 4))
        ttk.Entry(inner, textvariable=self.iv_cycles_var, width=10).grid(row=row, column=3, sticky="w", pady=4)
        row += 1

        ttk.Label(inner, text="起点:").grid(row=row, column=0, sticky="e", pady=4, padx=(0, 4))
        ttk.Entry(inner, textvariable=self.iv_start_var, width=10).grid(row=row, column=1, sticky="w", pady=4, padx=(0, 10))
        ttk.Label(inner, text="终点:").grid(row=row, column=2, sticky="e", pady=4, padx=(0, 4))
        ttk.Entry(inner, textvariable=self.iv_stop_var, width=10).grid(row=row, column=3, sticky="w", pady=4)
        row += 1

        ttk.Label(inner, text="步长:").grid(row=row, column=0, sticky="e", pady=4, padx=(0, 4))
        step_entry = ttk.Entry(inner, textvariable=self.iv_step_var, width=10)
        step_entry.grid(row=row, column=1, sticky="w", pady=4, padx=(0, 10))
        ttk.Label(inner, text="点数:").grid(row=row, column=2, sticky="e", pady=4, padx=(0, 4))
        points_entry = ttk.Entry(inner, textvariable=self.iv_points_var, width=10)
        points_entry.grid(row=row, column=3, sticky="w", pady=4)
        row += 1

        ttk.Label(inner, text="间隔 (s):").grid(row=row, column=0, sticky="e", pady=4, padx=(0, 4))
        ttk.Entry(inner, textvariable=self.iv_delay_var, width=10).grid(row=row, column=1, sticky="w", pady=4, padx=(0, 10))
        ttk.Label(inner, text="保护电流(A):").grid(row=row, column=2, sticky="e", pady=4, padx=(0, 4))
        ttk.Entry(inner, textvariable=self.iv_compliance_var, width=10).grid(row=row, column=3, sticky="w", pady=4)
        row += 1

        ttk.Checkbutton(
            inner,
            text="起点-终点-起点（三角扫描）",
            variable=self.iv_backforth_var,
        ).grid(row=row, column=0, columnspan=4, sticky="w", pady=(6, 2))
        row += 1

        self.iv_triangle_from_zero_chk = ttk.Checkbutton(
            inner,
            text="三角扫描从原点开始（0-终点-起点-0）",
            variable=self.iv_triangle_from_zero_var,
            command=self._on_triangle_from_zero_toggle,
        )
        self.iv_triangle_from_zero_chk.grid(row=row, column=0, columnspan=4, sticky="w", pady=(0, 2))
        row += 1

        self.iv_backforth_var.trace_add("write", lambda *args: self._sync_triangle_from_zero_state())
        self._sync_triangle_from_zero_state()

        row = self._add_low_current_controls(inner, row)

        ttk.Checkbutton(
            frame,
            text="启用 IV 质量检测",
            variable=self.iv_quality_enabled_var,
            command=self._toggle_iv_quality_frame,
        ).grid(row=1, column=0, sticky="w", pady=(10, 0))

        adv = ttk.Labelframe(frame, text="IV 质量检测（高级）", padding=6)
        self.iv_quality_frame = adv
        adv.grid(row=2, column=0, sticky="ew", pady=(6, 0))
        for col in range(2):
            adv.columnconfigure(col * 2 + 1, weight=1)

        adv_row = 0
        ttk.Label(adv, text="跳变阈值系数 k:").grid(row=adv_row, column=0, sticky="e", pady=4, padx=(0, 4))
        ttk.Entry(adv, textvariable=self.iv_quality_k_var, width=10).grid(row=adv_row, column=1, sticky="w", pady=4)
        ttk.Label(adv, text="异常比例上限:").grid(row=adv_row, column=2, sticky="e", pady=4, padx=(10, 4))
        ttk.Entry(adv, textvariable=self.iv_quality_jump_ratio_var, width=10).grid(row=adv_row, column=3, sticky="w", pady=4)
        adv_row += 1

        ttk.Label(adv, text="符号翻转上限:").grid(row=adv_row, column=0, sticky="e", pady=4, padx=(0, 4))
        ttk.Entry(adv, textvariable=self.iv_quality_flip_count_var, width=10).grid(row=adv_row, column=1, sticky="w", pady=4)
        ttk.Label(adv, text="自动重测次数:").grid(row=adv_row, column=2, sticky="e", pady=4, padx=(10, 4))
        ttk.Entry(adv, textvariable=self.iv_quality_max_retry_var, width=10).grid(row=adv_row, column=3, sticky="w", pady=4)

        self._toggle_iv_quality_frame()

        # 步长 / 点数 联动
        step_entry.bind("<FocusOut>", lambda e: self._update_points_from_step())
        points_entry.bind("<FocusOut>", lambda e: self._update_step_from_points())
        for var in (self.iv_start_var, self.iv_stop_var):
            var.trace_add("write", lambda *args: self._update_points_from_step())

    def _toggle_iv_quality_frame(self):
        if self.iv_quality_enabled_var.get():
            self.iv_quality_frame.grid()
        else:
            self.iv_quality_frame.grid_remove()

    def _on_triangle_from_zero_toggle(self):
        if self.iv_triangle_from_zero_var.get() and not self.iv_backforth_var.get():
            self.iv_backforth_var.set(True)
        self._sync_triangle_from_zero_state()

    def _sync_triangle_from_zero_state(self):
        enable = bool(self.iv_backforth_var.get())
        if not enable:
            self.iv_triangle_from_zero_var.set(False)
        try:
            if enable:
                self.iv_triangle_from_zero_chk.state(["!disabled"])
            else:
                self.iv_triangle_from_zero_chk.state(["disabled"])
        except Exception:
            pass

    def _update_points_from_step(self):
        if self._iv_updating:
            return
        try:
            start = self.iv_start_var.get()
            stop = self.iv_stop_var.get()
            step = self.iv_step_var.get()
        except tk.TclError:
            return
        if step <= 0:
            return
        n = int(round((stop - start) / step)) + 1
        if n < 2:
            n = 2
        self._iv_updating = True
        try:
            self.iv_points_var.set(n)
        finally:
            self._iv_updating = False

    def _update_step_from_points(self):
        if self._iv_updating:
            return
        try:
            start = self.iv_start_var.get()
            stop = self.iv_stop_var.get()
            n = self.iv_points_var.get()
        except tk.TclError:
            return
        if n < 2:
            n = 2
        step = (stop - start) / (n - 1)
        self._iv_updating = True
        try:
            self.iv_step_var.set(step)
        finally:
            self._iv_updating = False

    def _build_it_tab(self):
        frame = ttk.Frame(self.notebook, padding=6)
        self.notebook.add(frame, text="I-t")

        frame.columnconfigure(0, weight=1)
        inner = ttk.Frame(frame)
        inner.grid(row=0, column=0, pady=4)
        for col in range(4):
            weight = 0 if col in (1, 3) else 1
            inner.columnconfigure(col, weight=weight)

        self.it_bias_var = tk.DoubleVar(value=0.0)
        self.it_delay_var = tk.DoubleVar(value=0.0)
        self.it_points_var = tk.IntVar(value=50)
        self.it_infinite_var = tk.BooleanVar(value=False)
        self.it_compliance_var = tk.DoubleVar(value=0.1)

        row = 0
        row = self._add_buffer_mode_control(inner, row)
        ttk.Label(inner, text="电压偏置:").grid(row=row, column=0, sticky="e", pady=4, padx=(0, 4))
        ttk.Entry(inner, textvariable=self.it_bias_var, width=10).grid(row=row, column=1, sticky="w", pady=4, padx=(0, 10))
        ttk.Label(inner, text="点数:").grid(row=row, column=2, sticky="e", pady=4, padx=(0, 4))
        ttk.Entry(inner, textvariable=self.it_points_var, width=10).grid(row=row, column=3, sticky="w", pady=4)
        row += 1

        ttk.Label(inner, text="间隔 (s):").grid(row=row, column=0, sticky="e", pady=4, padx=(0, 4))
        ttk.Entry(inner, textvariable=self.it_delay_var, width=10).grid(row=row, column=1, sticky="w", pady=4, padx=(0, 10))
        ttk.Label(inner, text="保护电流(A):").grid(row=row, column=2, sticky="e", pady=4, padx=(0, 4))
        ttk.Entry(inner, textvariable=self.it_compliance_var, width=10).grid(row=row, column=3, sticky="w", pady=4)
        row += 1

        row = self._add_low_current_controls(inner, row)

        ttk.Checkbutton(
            inner,
            text="不限时（直到手动停止）",
            variable=self.it_infinite_var,
        ).grid(row=row, column=0, columnspan=4, sticky="w", pady=(6, 2))
        row += 1

    def _build_vt_tab(self):
        frame = ttk.Frame(self.notebook, padding=6)
        self.notebook.add(frame, text="V-t")

        frame.columnconfigure(0, weight=1)
        inner = ttk.Frame(frame)
        inner.grid(row=0, column=0, pady=4)
        for col in range(4):
            weight = 0 if col in (1, 3) else 1
            inner.columnconfigure(col, weight=weight)

        self.vt_bias_var = tk.DoubleVar(value=0.0)
        self.vt_delay_var = tk.DoubleVar(value=0.0)
        self.vt_points_var = tk.IntVar(value=50)
        self.vt_infinite_var = tk.BooleanVar(value=False)
        self.vt_compliance_var = tk.DoubleVar(value=10.0)

        row = 0
        row = self._add_buffer_mode_control(inner, row)
        ttk.Label(inner, text="电流偏置:").grid(row=row, column=0, sticky="e", pady=4, padx=(0, 4))
        ttk.Entry(inner, textvariable=self.vt_bias_var, width=10).grid(row=row, column=1, sticky="w", pady=4, padx=(0, 10))
        ttk.Label(inner, text="点数:").grid(row=row, column=2, sticky="e", pady=4, padx=(0, 4))
        ttk.Entry(inner, textvariable=self.vt_points_var, width=10).grid(row=row, column=3, sticky="w", pady=4)
        row += 1

        ttk.Label(inner, text="间隔 (s):").grid(row=row, column=0, sticky="e", pady=4, padx=(0, 4))
        ttk.Entry(inner, textvariable=self.vt_delay_var, width=10).grid(row=row, column=1, sticky="w", pady=4, padx=(0, 10))
        ttk.Label(inner, text="保护电压(V):").grid(row=row, column=2, sticky="e", pady=4, padx=(0, 4))
        ttk.Entry(inner, textvariable=self.vt_compliance_var, width=10).grid(row=row, column=3, sticky="w", pady=4)
        row += 1

        row = self._add_low_current_controls(inner, row)

        ttk.Checkbutton(
            inner,
            text="不限时（直到手动停止）",
            variable=self.vt_infinite_var,
        ).grid(row=row, column=0, columnspan=4, sticky="w", pady=(6, 2))
        row += 1

    def _build_rt_tab(self):
        frame = ttk.Frame(self.notebook, padding=6)
        self.notebook.add(frame, text="R-t")

        frame.columnconfigure(0, weight=1)
        inner = ttk.Frame(frame)
        inner.grid(row=0, column=0, pady=4)
        for col in range(4):
            weight = 0 if col in (1, 3) else 1
            inner.columnconfigure(col, weight=weight)

        self.rt_bias_var = tk.DoubleVar(value=0.0)
        self.rt_delay_var = tk.DoubleVar(value=0.0)
        self.rt_points_var = tk.IntVar(value=50)
        self.rt_infinite_var = tk.BooleanVar(value=False)
        self.rt_compliance_var = tk.DoubleVar(value=0.1)

        row = 0
        row = self._add_buffer_mode_control(inner, row)
        ttk.Label(inner, text="电压偏置:").grid(row=row, column=0, sticky="e", pady=4, padx=(0, 4))
        ttk.Entry(inner, textvariable=self.rt_bias_var, width=10).grid(row=row, column=1, sticky="w", pady=4, padx=(0, 10))
        ttk.Label(inner, text="点数:").grid(row=row, column=2, sticky="e", pady=4, padx=(0, 4))
        ttk.Entry(inner, textvariable=self.rt_points_var, width=10).grid(row=row, column=3, sticky="w", pady=4)
        row += 1

        ttk.Label(inner, text="间隔 (s):").grid(row=row, column=0, sticky="e", pady=4, padx=(0, 4))
        ttk.Entry(inner, textvariable=self.rt_delay_var, width=10).grid(row=row, column=1, sticky="w", pady=4, padx=(0, 10))
        ttk.Label(inner, text="保护电流(A):").grid(row=row, column=2, sticky="e", pady=4, padx=(0, 4))
        ttk.Entry(inner, textvariable=self.rt_compliance_var, width=10).grid(row=row, column=3, sticky="w", pady=4)
        row += 1

        row = self._add_low_current_controls(inner, row)

        ttk.Checkbutton(
            inner,
            text="不限时（直到手动停止）",
            variable=self.rt_infinite_var,
        ).grid(row=row, column=0, columnspan=4, sticky="w", pady=(6, 2))
        row += 1

    def _build_pt_tab(self):
        frame = ttk.Frame(self.notebook, padding=6)
        self.notebook.add(frame, text="P-t")

        frame.columnconfigure(0, weight=1)
        inner = ttk.Frame(frame)
        inner.grid(row=0, column=0, pady=4)
        for col in range(4):
            weight = 0 if col in (1, 3) else 1
            inner.columnconfigure(col, weight=weight)

        self.pt_bias_var = tk.DoubleVar(value=0.0)
        self.pt_delay_var = tk.DoubleVar(value=0.0)
        self.pt_points_var = tk.IntVar(value=50)
        self.pt_infinite_var = tk.BooleanVar(value=False)
        self.pt_compliance_var = tk.DoubleVar(value=0.1)

        row = 0
        row = self._add_buffer_mode_control(inner, row)
        ttk.Label(inner, text="电压偏置:").grid(row=row, column=0, sticky="e", pady=4, padx=(0, 4))
        ttk.Entry(inner, textvariable=self.pt_bias_var, width=10).grid(row=row, column=1, sticky="w", pady=4, padx=(0, 10))
        ttk.Label(inner, text="点数:").grid(row=row, column=2, sticky="e", pady=4, padx=(0, 4))
        ttk.Entry(inner, textvariable=self.pt_points_var, width=10).grid(row=row, column=3, sticky="w", pady=4)
        row += 1

        ttk.Label(inner, text="间隔 (s):").grid(row=row, column=0, sticky="e", pady=4, padx=(0, 4))
        ttk.Entry(inner, textvariable=self.pt_delay_var, width=10).grid(row=row, column=1, sticky="w", pady=4, padx=(0, 10))
        ttk.Label(inner, text="保护电流(A):").grid(row=row, column=2, sticky="e", pady=4, padx=(0, 4))
        ttk.Entry(inner, textvariable=self.pt_compliance_var, width=10).grid(row=row, column=3, sticky="w", pady=4)
        row += 1

        row = self._add_low_current_controls(inner, row)

        ttk.Checkbutton(
            inner,
            text="不限时（直到手动停止）",
            variable=self.pt_infinite_var,
        ).grid(row=row, column=0, columnspan=4, sticky="w", pady=(6, 2))
        row += 1

    def _build_ofr_tab(self):
        frame = ttk.Frame(self.notebook, padding=6)
        self.notebook.add(frame, text="开关比测试")

        frame.columnconfigure(0, weight=1)
        inner = ttk.Frame(frame)
        inner.grid(row=0, column=0, pady=4, sticky="ew")
        inner.columnconfigure(1, weight=1)

        self.ofr_voltage_var = tk.DoubleVar(value=0.1)
        self.ofr_zero_tol_var = tk.DoubleVar(value=5.0)
        self.ofr_bin_step_var = tk.DoubleVar(value=10.0)
        self.ofr_off_min_points_var = tk.IntVar(value=5)

        ttk.Label(inner, text="测试电压 (V):").grid(row=0, column=0, sticky="e", pady=4, padx=(0, 6))
        ttk.Entry(inner, textvariable=self.ofr_voltage_var, width=12).grid(row=0, column=1, sticky="w", pady=4)

        ttk.Label(inner, text="零压容差:").grid(row=1, column=0, sticky="e", pady=4, padx=(0, 6))
        ttk.Entry(inner, textvariable=self.ofr_zero_tol_var, width=12).grid(row=1, column=1, sticky="w", pady=4)

        ttk.Label(inner, text="分组步长 ΔP:").grid(row=2, column=0, sticky="e", pady=4, padx=(0, 6))
        ttk.Entry(inner, textvariable=self.ofr_bin_step_var, width=12).grid(row=2, column=1, sticky="w", pady=4)

        ttk.Label(inner, text="关态平均最少点数:").grid(row=3, column=0, sticky="e", pady=4, padx=(0, 6))
        ttk.Entry(inner, textvariable=self.ofr_off_min_points_var, width=12).grid(row=3, column=1, sticky="w", pady=4)

        ttk.Button(inner, text="?", width=3, command=self.show_ofr_help).grid(
            row=0, column=3, sticky="e", padx=(10, 0)
        )

        display = ttk.Frame(frame)
        display.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        for i in range(2):
            display.columnconfigure(i, weight=1)

        self.ofr_pressure_var = tk.StringVar(value="P: --")
        self.ofr_current_var = tk.StringVar(value="I: --")
        self.ofr_onoff_var = tk.StringVar(value="ON/OFF: --")
        self.ofr_ioff_var = tk.StringVar(value="I_off: --")

        ttk.Label(display, textvariable=self.ofr_pressure_var).grid(row=0, column=0, sticky="w", padx=4)
        ttk.Label(display, textvariable=self.ofr_current_var).grid(row=0, column=1, sticky="w", padx=4)
        ttk.Label(display, textvariable=self.ofr_ioff_var).grid(row=1, column=0, sticky="w", padx=4)
        ttk.Label(display, textvariable=self.ofr_onoff_var).grid(row=1, column=1, sticky="w", padx=4)

        # 底部按钮区域：左侧导出结果，右侧启动仿真
        btn_row = 2
        btns = ttk.Frame(frame)
        btns.grid(row=btn_row, column=0, sticky="ew", pady=(8, 0))
        btns.columnconfigure(0, weight=1)
        btns.columnconfigure(1, weight=0)
        btns.columnconfigure(2, weight=0)

        ttk.Button(
            btns,
            text="导出当前 OFR 结果",
            command=lambda: self.finalize_and_export_ofr_results(aborted=False),
        ).grid(row=0, column=0, sticky="w")

        self.ofr_sim_start_btn = ttk.Button(
            btns,
            text="仿真开关比测试 (0–10000 g)",
            command=self.start_ofr_simulation,
        )
        self.ofr_sim_start_btn.grid(row=0, column=1, sticky="e", padx=(10, 0))

        self.ofr_sim_stop_btn = ttk.Button(
            btns,
            text="停止仿真开关比测试",
            command=self.stop_ofr_simulation,
        )
        self.ofr_sim_stop_btn.grid(row=0, column=2, sticky="e", padx=(10, 0))

        self._update_ofr_sim_buttons()

    # ---- 连接 & 测量逻辑 ----

    def on_sim_toggle(self):
        """切换仿真模式时隐藏/显示仿真相关控件。"""
        self._update_ofr_sim_buttons()

    def _update_ofr_sim_buttons(self):
        if not hasattr(self, "ofr_sim_start_btn"):
            return
        if self.sim_var.get():
            self.ofr_sim_start_btn.grid()
            self.ofr_sim_stop_btn.grid()
        else:
            self.ofr_sim_start_btn.grid_remove()
            self.ofr_sim_stop_btn.grid_remove()

    def _sync_model_channel_controls(self):
        model = self.model_select_var.get()
        four_wire_var = getattr(self, "four_wire_var", None)
        four_wire = bool(four_wire_var.get()) if isinstance(four_wire_var, tk.Variable) else False
        is_2636b = model == "2636B"
        try:
            if is_2636b and four_wire:
                self.channel_label.pack_forget()
                self.channel_combo.pack_forget()
                self.four_wire_channel_frame.pack(side=tk.LEFT, padx=(0, 8))
            elif is_2636b:
                self.four_wire_channel_frame.pack_forget()
                self.channel_label.pack(side=tk.LEFT, padx=(0, 4))
                self.channel_combo.pack(side=tk.LEFT, padx=(0, 8))
            else:
                self.four_wire_channel_frame.pack_forget()
                self.channel_label.pack_forget()
                self.channel_combo.pack_forget()
        except Exception:
            pass
        self._apply_channel_selection_to_instrument()

    def _apply_channel_selection_to_instrument(self):
        model = (self.model_select_var.get() or "").upper()
        instrument_model = (getattr(self.instrument, "model", None) or "").upper()
        is_2636b = model == "2636B" or instrument_model == "2636B"
        four_wire_var = getattr(self, "four_wire_var", None)
        four_wire = bool(four_wire_var.get()) if isinstance(four_wire_var, tk.Variable) else False
        try:
            if is_2636b and four_wire:
                self.instrument.set_source_channel(self.source_channel_var.get())
                self.instrument.set_measure_channel(self.measure_channel_var.get())
            else:
                self.instrument.set_channel(self.channel_select_var.get())
        except Exception:
            pass

    def _sync_baud_control(self):
        addr = (self.resource_combo.get() or "").upper()
        if "ASRL" in addr:
            try:
                self.baud_combo.config(state="readonly")
            except Exception:
                pass
        else:
            try:
                self.baud_combo.config(state="disabled")
            except Exception:
                pass

    def refresh_resources(self):
        resources = self.instrument.list_resources()
        self.resource_combo["values"] = resources
        if resources:
            self.resource_combo.current(0)
            self._log(f"找到地址: {resources}")
            self._sync_baud_control()
        else:
            self._log("未找到任何 VISA 资源")

    def on_four_wire_toggle(self):
        """用户勾选/取消 四线制 时调用"""
        # 先保存配置
        try:
            self._save_settings()
        except Exception:
            pass

        enable = bool(self.four_wire_var.get())
        if enable and self.model_select_var.get() == "2636B":
            try:
                base_ch = self.channel_select_var.get() or "A"
                self.source_channel_var.set(self.source_channel_var.get() or base_ch)
                self.measure_channel_var.set(self.measure_channel_var.get() or base_ch)
            except Exception:
                pass
        self._sync_model_channel_controls()
        self._apply_channel_selection_to_instrument()
        # 同步到仪器
        try:
            self.instrument.set_remote_sense(enable)
        except Exception as exc:
            self._log(f"设置四线制失败: {exc}")

        # 更新状态文字：保留原有前半段，只在后面追加四线状态
        status = self.status_label.cget("text")
        # 去掉之前可能追加的 " | 四线: ..." 部分
        if " | 四线:" in status:
            status = status.split(" | 四线:")[0].strip()
        sense_str = "ON" if enable else "OFF"
        self.status_label.config(text=f"{status} | 四线: {sense_str}")

    def connect_instrument(self):
        selected_model = self.model_select_var.get()
        forced_model = selected_model if selected_model in ("2400", "2636B") else None
        self.instrument.set_forced_model(forced_model)
        self._apply_channel_selection_to_instrument()
        simulate = self.sim_var.get()
        if simulate:
            status = self.instrument.connect(address=None, simulate=True)
        else:
            addr = self.resource_combo.get().strip()
            if not addr:
                messagebox.showwarning("未选择地址", "请先在下拉框中选择一个仪器地址，或勾选仿真模式。")
                return
            baud = None
            if "ASRL" in addr.upper():
                try:
                    baud = int(self.baud_rate_var.get())
                except Exception:
                    baud = None
            status = self.instrument.connect(address=addr, simulate=False, baud_rate=baud)

        # 连接成功后，根据当前勾选状态设置四线
        enable = bool(getattr(self, "four_wire_var", tk.BooleanVar(value=False)).get())
        try:
            self.instrument.set_remote_sense(enable)
        except Exception as exc:
            self._log(f"设置四线制失败: {exc}")

        sense_str = "ON" if enable else "OFF"
        self.status_label.config(text=f"{status} | 四线: {sense_str}")
        self._log(f"{status} | 四线: {sense_str}")

    def choose_save_root(self):
        path = filedialog.askdirectory()
        if path:
            self.save_root_var.set(path)

    def start_measurement(self):
        tab_index = self.notebook.index(self.notebook.select())
        tab_text = self.notebook.tab(tab_index, "text")
        mode_map = {"IV 扫描": "IV", "I-t": "It", "V-t": "Vt", "R-t": "Rt", "P-t": "Pt"}
        mode = mode_map.get(tab_text, "IV")

        collectors = {
            "IV": self._collect_iv_config,
            "It": self._collect_it_config,
            "Vt": self._collect_vt_config,
            "Rt": self._collect_rt_config,
            "Pt": self._collect_pt_config,
        }

        collector = collectors.get(mode)
        config = collector() if collector else None

        if config is None:
            return

        self._initiate_measurement(mode, config, show_dialog=True)

    def stop_measurement(self):
        if self.measurement_thread is None:
            return
        self.stop_event.set()
        self._log("已请求停止")

    def _initiate_measurement(self, mode, config, show_dialog: bool):
        if not self.thread_semaphore.acquire(blocking=False):
            if show_dialog:
                messagebox.showwarning("忙碌", "测量正在进行中")
            else:
                self._log("TCP 请求被忽略：测量正在进行中")
            return False

        self._measurement_lock_acquired = True
        success = False
        try:
            if not self.instrument.simulated and self.instrument.session is None:
                if show_dialog:
                    messagebox.showwarning("未连接", "请先连接仪器或勾选仿真模式")
                else:
                    self._log("TCP 请求被忽略：未连接仪器")
                return False

            model = getattr(self.instrument, "model", None)
            low_current_mode = bool(config.get("low_current_speed_mode", False))
            self.instrument.set_low_current_mode(low_current_mode)
            self.instrument.current_range_override = config.get("current_range_override")
            try:
                nplc = float(self.integration_time_var.get())
            except Exception:
                nplc = 0.0

            model_upper = (model or "").upper()
            if nplc <= 0:
                nplc = 0.01 if low_current_mode else (0.01 if model_upper == "2636B" else 0.1)

            try:
                self.instrument.set_nplc(nplc)
            except Exception:
                pass

            self.current_mode = mode
            # 只在 IV 模式下记录源模式，其它模式用 None
            self.current_source_mode = config.get("source_mode") if mode == "IV" else None
            self.current_data = []
            self.total_points = config.get("total_points", 0)
            self.completed_points = 0
            self.start_time = time.time()

            self._reset_plot()

            if self.total_points > 0:
                self.progress.config(mode="determinate", maximum=self.total_points)
                self.progress["value"] = 0
            else:
                self.progress.config(mode="indeterminate")
                self.progress.start(50)

            self.stop_event.clear()
            self.start_button.config(state="disabled")
            self.stop_button.config(state="normal")
            self._log(f"开始 {mode} 测量（模式: {self.instrument.conn_type}）")

            self.measurement_thread = threading.Thread(
                target=self._run_measurement,
                args=(mode, config),
                daemon=True,
            )
            self.measurement_thread.start()
            self.is_measuring = True
            success = True
            return True
        finally:
            if not success and self._measurement_lock_acquired:
                try:
                    self.thread_semaphore.release()
                except Exception:
                    pass
                self._measurement_lock_acquired = False

    def _run_measurement(self, mode, config):
        try:
            if mode == "IV":
                self._run_iv_measurement(config)
            elif mode in ("It", "Rt", "Pt"):
                source_mode = config.get("source_mode", "Voltage") if isinstance(config, dict) else "Voltage"
                self._run_time_measurement(config, source_mode=source_mode, mode=mode)
            else:
                self._run_time_measurement(config, source_mode="Current", mode=mode)
        except Exception as exc:
            self.queue.put(("error", f"{exc.__class__.__name__}: {exc}"))
        finally:
            try:
                self.instrument.output_off()
            except Exception:
                pass
            self.queue.put(("finished", None))

    def _run_iv_measurement(self, cfg):
        start = cfg["start"]
        stop = cfg["stop"]
        points = cfg["points"]
        cycles = cfg["cycles"]
        back_and_forth = cfg["back_and_forth"]
        triangle_from_zero = cfg.get("triangle_from_zero", False)
        delay = cfg["delay"]
        compliance = cfg["compliance"]
        source_mode = cfg["source_mode"]
        buffer_mode = bool(cfg.get("buffer_mode", False))
        levels_from_cfg = cfg.get("levels") if buffer_mode else None

        if buffer_mode and isinstance(levels_from_cfg, (list, tuple)):
            seq = list(levels_from_cfg)
        else:
            base_forward = self.instrument.sweep_points(start, stop, points)
            if triangle_from_zero:
                seg1 = self.instrument.sweep_points(0, stop, points)
                seg2 = self.instrument.sweep_points(stop, start, points)[1:]
                seg3 = self.instrument.sweep_points(start, 0, points)[1:]
                one_cycle = np.concatenate([seg1, seg2, seg3])
            elif back_and_forth:
                if len(base_forward) > 1:
                    backward = base_forward[-2::-1]
                else:
                    backward = base_forward
                one_cycle = np.concatenate([base_forward, backward])
            else:
                one_cycle = base_forward
            seq = np.tile(one_cycle, cycles)
        is_2636b = (getattr(self.instrument, "model", "") or "").upper() == "2636B"
        if is_2636b:
            self.instrument.prepare_source_2636(source_mode, compliance)
        else:
            self.instrument.configure_source(source_mode, float(seq[0]), compliance)

        if buffer_mode and not self.instrument.simulated and self.instrument.session is not None:
            try:
                if is_2636b:
                    readings = self.instrument.buffer_sweep_2636(source_mode, compliance, seq, delay)
                else:
                    readings = self.instrument.buffer_sweep_2400(source_mode, compliance, seq, delay)
                for idx, data in enumerate(readings):
                    sp = float(seq[idx]) if idx < len(seq) else 0.0
                    data.update({"index": idx, "setpoint": sp})
                    self.queue.put(("data", data, self.total_points))
                return
            except Exception as exc:
                self._log(f"缓存模式失败，回退到逐点: {exc}")

        for idx, level in enumerate(seq):
            if self.stop_event.is_set():
                break
            if is_2636b:
                self.instrument.set_level_2636(source_mode, float(level))
            else:
                self.instrument.configure_source(source_mode, float(level), compliance)
            if delay and delay > 0:
                time.sleep(delay)
            data = self.instrument.measure_once()
            data.update({"index": idx, "setpoint": float(level)})
            self.queue.put(("data", data, self.total_points))

    def _run_time_measurement(self, cfg, source_mode, mode=None):
        mode = mode or self.current_mode
        bias = cfg["bias"]
        delay = cfg["delay"]
        points = cfg["points"]
        infinite = cfg["infinite"]
        compliance = cfg["compliance"]
        buffer_mode = bool(cfg.get("buffer_mode", False))

        is_2636b = (getattr(self.instrument, "model", "") or "").upper() == "2636B"
        if is_2636b:
            self.instrument.prepare_source_2636(source_mode, compliance)
            self.instrument.set_level_2636(source_mode, bias)
        else:
            self.instrument.configure_source(source_mode, bias, compliance)

        if infinite:
            idx = 0
            while not self.stop_event.is_set():
                data = self.instrument.measure_once()
                data.update({"index": idx, "setpoint": bias})
                self.queue.put(("data", data, 0))
                idx += 1
                if delay and delay > 0:
                    time.sleep(delay)
        else:
            seq = [bias] * max(0, points)
            if buffer_mode and not self.instrument.simulated and self.instrument.session is not None:
                try:
                    if is_2636b:
                        readings = self.instrument.buffer_sweep_2636(source_mode, compliance, seq, delay)
                    else:
                        readings = self.instrument.buffer_sweep_2400(source_mode, compliance, seq, delay)
                    for idx, data in enumerate(readings):
                        data.update({"index": idx, "setpoint": bias})
                        self.queue.put(("data", data, self.total_points))
                    return
                except Exception as exc:
                    self._log(f"缓存模式失败，回退到逐点: {exc}")

            for idx in range(points):
                if self.stop_event.is_set():
                    break
                data = self.instrument.measure_once()
                data.update({"index": idx, "setpoint": bias})
                self.queue.put(("data", data, self.total_points))
                if delay and delay > 0:
                    time.sleep(delay)

    # ---- 参数收集 ----

    def _get_current_range_override_value(self):
        raw = (self.current_range_override_var.get() or "").strip()
        if not raw:
            return 1e-6
        try:
            val = float(raw)
            if val <= 0:
                raise ValueError("range must be positive")
            return val
        except Exception:
            self._log("电流量程输入无效，已回退到 1e-6 A")
            self.current_range_override_var.set("1e-6")
            return 1e-6

    def _build_iv_levels(self, start, stop, step, points, cycles, back_and_forth, triangle_from_zero=False):
        try:
            step_val = float(step)
        except Exception:
            step_val = 0.0
        if step_val == 0:
            self._log("步长为 0，已回退到点数线性生成")
            base = list(self.instrument.sweep_points(start, stop, 2))
        else:
            direction = 1 if stop >= start else -1
            actual_step = abs(step_val) * direction
            base = []
            val = float(start)
            guard = 0
            while (
                (direction > 0 and val <= stop + 1e-12)
                or (direction < 0 and val >= stop - 1e-12)
            ):
                base.append(val)
                val += actual_step
                guard += 1
                if guard > 200000:
                    self._log("步长设置导致点数过多，已提前截断")
                    break
            if len(base) < 2:
                base = [start, stop]

        if triangle_from_zero:
            seg1 = list(self.instrument.sweep_points(0, stop, points))
            seg2 = list(self.instrument.sweep_points(stop, start, points))[1:]
            seg3 = list(self.instrument.sweep_points(start, 0, points))[1:]
            segment = list(np.concatenate([seg1, seg2, seg3]))
        elif back_and_forth:
            if len(base) > 1:
                segment = base + base[-2::-1]
            else:
                segment = base
        else:
            segment = base
        seq = []
        for _ in range(max(1, int(cycles))):
            seq.extend(segment)
        return seq

    def _collect_iv_config(self):
        try:
            start = self.iv_start_var.get()
            stop = self.iv_stop_var.get()
            step = self.iv_step_var.get()
            points = self.iv_points_var.get()
            cycles = self.iv_cycles_var.get()
            delay = self.iv_delay_var.get()
            compliance = self.iv_compliance_var.get()
            source_mode = self.iv_source_mode_var.get()
        except tk.TclError:
            messagebox.showwarning("输入错误", "IV 参数无效")
            return None
        if step <= 0:
            messagebox.showwarning("输入错误", "步长必须为正数")
            return None
        if delay < 0:
            messagebox.showwarning("输入错误", "间隔时间不能为负")
            return None
        if compliance <= 0:
            messagebox.showwarning("输入错误", "保护值必须为正数")
            return None
        if cycles < 1:
            messagebox.showwarning("输入错误", "循环次数至少为 1")
            return None
        if points < 2:
            messagebox.showwarning("输入错误", "点数至少为 2")
            return None
        if self.iv_triangle_from_zero_var.get():
            per_cycle = points * 3 - 2 if points > 1 else points
        elif self.iv_backforth_var.get():
            if points > 1:
                per_cycle = points * 2 - 1
            else:
                per_cycle = points
        else:
            per_cycle = points
        buffer_mode = bool(self.buffer_mode_var.get())
        triangle_from_zero = self.iv_triangle_from_zero_var.get()
        levels = None
        if buffer_mode:
            levels = self._build_iv_levels(
                start,
                stop,
                step,
                points,
                cycles,
                self.iv_backforth_var.get(),
                triangle_from_zero,
            )
            total_points = len(levels)
        else:
            total_points = max(0, per_cycle * max(1, cycles))
            try:
                expected_step = (stop - start) / (points - 1)
                if points > 1 and abs(step - expected_step) > max(1e-9, abs(expected_step) * 0.01):
                    self._log("提示: 步长仅在缓存模式/内置 sweep 时生效，当前按点数生成扫描。")
            except Exception:
                pass
        return dict(
            start=start,
            stop=stop,
            step=step,
            points=points,
            cycles=cycles,
            back_and_forth=self.iv_backforth_var.get(),
            triangle_from_zero=triangle_from_zero,
            delay=delay,
            compliance=compliance,
            source_mode="Voltage" if source_mode == "Voltage" else "Current",
            total_points=total_points,
            low_current_speed_mode=self.low_current_speed_mode_var.get(),
            current_range_override=self._get_current_range_override_value(),
            buffer_mode=buffer_mode,
            levels=levels if buffer_mode else None,
        )

    def _collect_it_config(self):
        try:
            bias = self.it_bias_var.get()
            delay = self.it_delay_var.get()
            points = self.it_points_var.get()
            infinite = self.it_infinite_var.get()
            compliance = self.it_compliance_var.get()
        except tk.TclError:
            messagebox.showwarning("输入错误", "I-t 参数无效")
            return None
        if delay < 0:
            messagebox.showwarning("输入错误", "间隔时间不能为负")
            return None
        if not infinite and points < 1:
            messagebox.showwarning("输入错误", "点数至少为 1")
            return None
        if compliance <= 0:
            messagebox.showwarning("输入错误", "保护值必须为正数")
            return None
        buffer_mode = bool(self.buffer_mode_var.get())
        if infinite and buffer_mode:
            self._log("提示: 不限时模式下无法启用缓存模式，已回退逐点采集。")
            buffer_mode = False
        total_points = 0 if infinite else max(0, points)
        return dict(
            bias=bias,
            delay=delay,
            points=points,
            infinite=infinite,
            compliance=compliance,
            total_points=total_points,
            low_current_speed_mode=self.low_current_speed_mode_var.get(),
            current_range_override=self._get_current_range_override_value(),
            buffer_mode=buffer_mode,
        )

    def _collect_vt_config(self):
        try:
            bias = self.vt_bias_var.get()
            delay = self.vt_delay_var.get()
            points = self.vt_points_var.get()
            infinite = self.vt_infinite_var.get()
            compliance = self.vt_compliance_var.get()
        except tk.TclError:
            messagebox.showwarning("输入错误", "V-t 参数无效")
            return None
        if delay < 0:
            messagebox.showwarning("输入错误", "间隔时间不能为负")
            return None
        if not infinite and points < 1:
            messagebox.showwarning("输入错误", "点数至少为 1")
            return None
        if compliance <= 0:
            messagebox.showwarning("输入错误", "保护值必须为正数")
            return None
        buffer_mode = bool(self.buffer_mode_var.get())
        if infinite and buffer_mode:
            self._log("提示: 不限时模式下无法启用缓存模式，已回退逐点采集。")
            buffer_mode = False
        total_points = 0 if infinite else max(0, points)
        return dict(
            bias=bias,
            delay=delay,
            points=points,
            infinite=infinite,
            compliance=compliance,
            total_points=total_points,
            low_current_speed_mode=self.low_current_speed_mode_var.get(),
            current_range_override=self._get_current_range_override_value(),
            buffer_mode=buffer_mode,
        )

    def _collect_rt_config(self):
        try:
            bias = self.rt_bias_var.get()
            delay = self.rt_delay_var.get()
            points = self.rt_points_var.get()
            infinite = self.rt_infinite_var.get()
            compliance = self.rt_compliance_var.get()
        except tk.TclError:
            messagebox.showwarning("输入错误", "R-t 参数无效")
            return None
        if delay < 0:
            messagebox.showwarning("输入错误", "间隔时间不能为负")
            return None
        if not infinite and points < 1:
            messagebox.showwarning("输入错误", "点数至少为 1")
            return None
        if compliance <= 0:
            messagebox.showwarning("输入错误", "保护值必须为正数")
            return None
        buffer_mode = bool(self.buffer_mode_var.get())
        if infinite and buffer_mode:
            self._log("提示: 不限时模式下无法启用缓存模式，已回退逐点采集。")
            buffer_mode = False
        total_points = 0 if infinite else max(0, points)
        return dict(
            bias=bias,
            delay=delay,
            points=points,
            infinite=infinite,
            compliance=compliance,
            total_points=total_points,
            source_mode="Voltage",
            low_current_speed_mode=self.low_current_speed_mode_var.get(),
            current_range_override=self._get_current_range_override_value(),
            buffer_mode=buffer_mode,
        )

    def _collect_pt_config(self):
        try:
            bias = self.pt_bias_var.get()
            delay = self.pt_delay_var.get()
            points = self.pt_points_var.get()
            infinite = self.pt_infinite_var.get()
            compliance = self.pt_compliance_var.get()
        except tk.TclError:
            messagebox.showwarning("输入错误", "P-t 参数无效")
            return None
        if delay < 0:
            messagebox.showwarning("输入错误", "间隔时间不能为负")
            return None
        if not infinite and points < 1:
            messagebox.showwarning("输入错误", "点数至少为 1")
            return None
        if compliance <= 0:
            messagebox.showwarning("输入错误", "保护值必须为正数")
            return None
        buffer_mode = bool(self.buffer_mode_var.get())
        if infinite and buffer_mode:
            self._log("提示: 不限时模式下无法启用缓存模式，已回退逐点采集。")
            buffer_mode = False
        total_points = 0 if infinite else max(0, points)
        return dict(
            bias=bias,
            delay=delay,
            points=points,
            infinite=infinite,
            compliance=compliance,
            total_points=total_points,
            source_mode="Voltage",
            low_current_speed_mode=self.low_current_speed_mode_var.get(),
            current_range_override=self._get_current_range_override_value(),
            buffer_mode=buffer_mode,
        )

    # ---- 队列 & 进度 ----

    def _poll_queue(self):
        try:
            while True:
                item = self.queue.get_nowait()
                kind = item[0]
                if kind == "data":
                    data, total = item[1], item[2]
                    self._handle_data(data, total)
                elif kind == "error":
                    msg = item[1]
                    self._log("错误: " + msg)
                    messagebox.showerror("测量错误", msg)
                elif kind == "log":
                    self._log(item[1])
                elif kind == "finished":
                    self._finish_measurement()
                self.queue.task_done()
        except queue.Empty:
            pass
        self.root.after(100, self._poll_queue)

    def _format_seconds(self, sec: int) -> str:
        sec = int(max(0, sec))
        h = sec // 3600
        m = (sec % 3600) // 60
        s = sec % 60
        if h > 0:
            return f"{h:d}:{m:02d}:{s:02d}"
        else:
            return f"{m:02d}:{s:02d}"

    def _handle_data(self, data, total_points):
        try:
            idx = int(data.get("index", 0))
            setpoint = float(data.get("setpoint", 0.0))
            v = float(data.get("voltage", 0.0))
            c = float(data.get("current", 0.0))
        except Exception as exc:
            self._log(f"忽略无效数据: {exc}")
            return

        line = f"[{idx:04d}] set={setpoint:.5g}, V={v:.5g}, I={c:.5g}"
        self._log(line)
        resistance = ""
        if c != 0:
            try:
                resistance = v / c
            except Exception:
                resistance = ""
        power = ""
        try:
            power = v * c
        except Exception:
            power = ""

        data_copy = dict(data)
        data_copy.update({
            "voltage": v,
            "current": c,
            "resistance": resistance,
            "power": power,
        })

        self.current_data.append(data_copy)

        self.completed_points = idx + 1

        # 更新曲线
        if self.current_mode == "IV":
            # IV：根据源模式决定横轴
            src_mode = getattr(self, "current_source_mode", "Voltage")

            if src_mode == "Voltage":
                # —— 电压源：横轴用“实测电压”，但按扫偏方向分段 + 排序，避免往回连线 ——
                data_list = list(self.current_data)
                xs, ys = [], []

                if data_list:
                    segments = []
                    segment = [data_list[0]]
                    direction = 0  # 1: setpoint 递增；-1: setpoint 递减；0: 尚未确定

                    for cur in data_list[1:]:
                        prev = segment[-1]
                        sp_prev = float(prev.get("setpoint", 0.0))
                        sp_cur = float(cur.get("setpoint", 0.0))
                        diff = sp_cur - sp_prev

                        # 当前这一步的方向
                        if diff > 0:
                            new_dir = 1
                        elif diff < 0:
                            new_dir = -1
                        else:
                            new_dir = 0

                        if direction == 0:
                            # 第一次确定方向
                            direction = new_dir
                            segment.append(cur)
                            continue

                        # 方向没变或 diff=0：继续当前段
                        if new_dir == 0 or new_dir == direction:
                            segment.append(cur)
                        else:
                            # 扫偏方向发生反转：结束上一段，开启新一段
                            segments.append((direction, segment))
                            segment = [cur]
                            direction = new_dir

                    # 最后一段也要加进去
                    segments.append((direction, segment))

                    # 对每一段按“实测电压”排序：正向段升序，反向段降序
                    for dir_sign, seg in segments:
                        seg_sorted = sorted(
                            seg,
                            key=lambda d: float(d.get("voltage", 0.0)),
                            reverse=(dir_sign < 0),  # 反向扫：电压从大到小
                        )
                        xs.extend(float(d.get("voltage", 0.0)) for d in seg_sorted)
                        ys.extend(float(d.get("current", 0.0)) for d in seg_sorted)

                x_label = "Voltage (V)"

            else:
                # 源为电流时，仍然画标准 I-V：横轴用实测电压，纵轴电流
                xs = [float(d.get("voltage", 0.0)) for d in self.current_data]
                ys = [float(d.get("current", 0.0)) for d in self.current_data]
                x_label = "Voltage (V)"

            self.voltage_line.set_data([], [])
            self.current_line.set_data(xs, ys)
            self.current_line.set_label("Current (A)")
            self.ax.set_title("I-V sweep")
            self.ax.set_xlabel(x_label)
            self.ax.set_ylabel("Current (A)")


        elif self.current_mode == "It":
            # I-t：横轴为时间，纵轴为电流
            base_ts = self.current_data[0].get("timestamp", data.get("timestamp", 0.0))
            xs = [d.get("timestamp", base_ts) - base_ts for d in self.current_data]
            ys = [d.get("current", 0.0) for d in self.current_data]

            self.current_line.set_data(xs, ys)
            self.current_line.set_label("Current (A)")
            self.voltage_line.set_data([], [])
            self.ax.set_title("I-t")
            self.ax.set_xlabel("Time (s)")
            self.ax.set_ylabel("Current (A)")

        elif self.current_mode == "Rt":
            base_ts = self.current_data[0].get("timestamp", data.get("timestamp", 0.0))
            xs = [d.get("timestamp", base_ts) - base_ts for d in self.current_data]
            ys = []
            for d in self.current_data:
                val = d.get("resistance")
                try:
                    ys.append(float(val))
                except Exception:
                    ys.append(np.nan)

            self.current_line.set_data(xs, ys)
            self.current_line.set_label("Resistance (Ohm)")
            self.voltage_line.set_data([], [])
            self.ax.set_title("R-t")
            self.ax.set_xlabel("Time (s)")
            self.ax.set_ylabel("Resistance (Ohm)")

        elif self.current_mode == "Pt":
            base_ts = self.current_data[0].get("timestamp", data.get("timestamp", 0.0))
            xs = [d.get("timestamp", base_ts) - base_ts for d in self.current_data]
            ys = []
            for d in self.current_data:
                val = d.get("power")
                try:
                    ys.append(float(val))
                except Exception:
                    ys.append(np.nan)

            self.current_line.set_data(xs, ys)
            self.current_line.set_label("Power (W)")
            self.voltage_line.set_data([], [])
            self.ax.set_title("P-t")
            self.ax.set_xlabel("Time (s)")
            self.ax.set_ylabel("Power (W)")

        else:
            # V-t：横轴为时间，纵轴为电压
            base_ts = self.current_data[0].get("timestamp", data.get("timestamp", 0.0))
            xs = [d.get("timestamp", base_ts) - base_ts for d in self.current_data]
            ys = [d.get("voltage", 0.0) for d in self.current_data]

            self.voltage_line.set_data(xs, ys)
            self.voltage_line.set_label("Voltage (V)")
            self.current_line.set_data([], [])
            self.ax.set_title("V-t")
            self.ax.set_xlabel("Time (s)")
            self.ax.set_ylabel("Voltage (V)")

        # 确保没有图例
        leg = self.ax.get_legend()
        if leg is not None:
            try:
                leg.remove()
            except Exception:
                pass

        # 自动范围 + 当前绘图样式
        self.ax.relim()
        self.ax.autoscale_view()
        try:
            self._apply_plot_style()
        except Exception:
            pass

        self.canvas.draw_idle()

        # 进度条 + 点数 / 剩余时间
        if total_points > 0:
            self.progress.config(mode="determinate", maximum=total_points)
            done = min(self.completed_points, total_points)
            self.progress["value"] = done
            self.points_label.config(text=f"点数: {done}/{total_points}")

            elapsed = time.time() - (self.start_time or time.time())
            if done > 0 and elapsed > 0 and done < total_points:
                avg = elapsed / done
                remain_sec = int((total_points - done) * avg)
                eta_str = self._format_seconds(remain_sec)
                self.eta_label.config(text=f"剩余时间: {eta_str}")
            else:
                self.eta_label.config(text="剩余时间: 00:00")
        else:
            # 无限模式
            self.points_label.config(text=f"点数: {self.completed_points}/∞")
            self.eta_label.config(text="剩余时间: --")

    def _apply_plot_style(self):
        """根据 self.plot_style_var 调整曲线为 线 / 点 / 线+点；线=蓝色，点=红色"""
        style = getattr(self, "plot_style_var", None)
        if style is None:
            return
        style = style.get()

        if style == "点":
            linestyle = "None"
            marker = "o"
        elif style == "线+点":
            linestyle = "-"
            marker = "o"
        else:  # 默认：线
            linestyle = "-"
            marker = ""

        line_color = "blue"   # 线：蓝色
        marker_color = "red"  # 点：红色

        plot_lines = [self.voltage_line, self.current_line]
        if getattr(self, "ofr_line", None) is not None:
            plot_lines.append(self.ofr_line)

        for line in plot_lines:
            line.set_color(line_color)
            line.set_linestyle(linestyle)
            line.set_marker(marker)
            if marker:
                line.set_markerfacecolor(marker_color)
                line.set_markeredgecolor(marker_color)

        # 再次确保没有图例
        leg = self.ax.get_legend()
        if leg is not None:
            try:
                leg.remove()
            except Exception:
                pass

        self.canvas.draw_idle()

    def _handle_multi_pressure_iv_completion(self, pressure_value: float) -> bool:
        voltages = []
        currents = []
        for row in self.current_data:
            try:
                voltages.append(float(row.get("voltage", 0.0)))
                currents.append(float(row.get("current", 0.0)))
            except Exception:
                continue

        quality_enabled = self.iv_quality_enabled_var.get()
        if quality_enabled:
            is_bad, metrics = self.check_iv_quality(voltages, currents)
            self._log(
                f"[{pressure_value:g}g] 质量检测: jump_ratio={metrics['jump_ratio']:.4f}, "
                f"flip_count={metrics['flip_count']}, base_slope={metrics['base_slope']:.4g}"
            )
        else:
            is_bad = False
            metrics = {"jump_ratio": 0.0, "flip_count": 0, "base_slope": 0.0}

        max_retry = max(0, int(self.iv_quality_max_retry_var.get() or 0)) if quality_enabled else 0
        if is_bad and self.multi_tcp_retry_used < max_retry:
            cfg = copy.deepcopy(self.multi_tcp_last_iv_config) if self.multi_tcp_last_iv_config else None
            if cfg is not None:
                self.multi_tcp_retry_used += 1
                self._log(
                    f"质量检测不通过，自动重测（{self.multi_tcp_retry_used}/{max_retry}）…"
                )
                if self._initiate_measurement("IV", cfg, show_dialog=False):
                    return True
                self._log("自动重测启动失败，保存当前数据为 BAD")

        if is_bad and quality_enabled:
            self._log("质量检测仍不通过，保留最后一次数据并标记 BAD")

        path = self._save_pressure_iv_file(pressure_value, mark_bad=is_bad)
        self.multi_tcp_records.append((pressure_value, path, is_bad))
        status = "BAD" if is_bad else "OK"
        self._log(f"{pressure_value:g}g 测量完成 {status}（{path}）")
        self.multi_tcp_pending_pressure = None
        self.multi_tcp_retry_used = 0
        return False

    def _finish_measurement(self):
        self.instrument.output_off()
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.progress.stop()
        if self.total_points > 0:
            self.progress.config(mode="determinate", maximum=self.total_points)
            self.progress["value"] = min(self.completed_points, self.total_points)
            self.points_label.config(
                text=f"点数: {min(self.completed_points, self.total_points)}/{self.total_points}"
            )
            self.eta_label.config(text="剩余时间: 00:00")
        else:
            self.progress.config(mode="determinate", maximum=100)
            self.progress["value"] = 100
            self.points_label.config(text=f"点数: {self.completed_points}/∞")
            self.eta_label.config(text="剩余时间: --")

        self._log("测量结束")
        self.is_measuring = False
        if self._measurement_lock_acquired:
            try:
                self.thread_semaphore.release()
            except Exception:
                pass
            self._measurement_lock_acquired = False

        if self.auto_save_var.get() and self.multi_tcp_pending_pressure is None:
            try:
                self._auto_save_current()
            except Exception as exc:
                self._log(f"自动保存失败: {exc}")

        pending_pressure = self.multi_tcp_pending_pressure
        if pending_pressure is not None and self.current_mode == "IV":
            try:
                retrying = self._handle_multi_pressure_iv_completion(pending_pressure)
                if retrying:
                    return
            except Exception as exc:  # noqa: BLE001
                self._log(f"处理 {pending_pressure:g}g 数据时出错: {exc}")
                self.multi_tcp_pending_pressure = None
        else:
            self.multi_tcp_pending_pressure = None
            self.multi_tcp_retry_used = 0

        self._notify_tcp_waiters()

    def _reset_plot(self):
        self.voltage_line.set_data([], [])
        self.current_line.set_data([], [])
        if self.ofr_line is not None:
            self.ofr_line.set_data([], [])
        if hasattr(self, "ofr_pressures"):
            self.ofr_pressures.clear()
            self.ofr_onoff_values.clear()
        self.ax.set_title("Live measurement")
        self.ax.set_xlabel("Point index")
        self.ax.set_ylabel("Value")

        # 移除可能存在的图例
        leg = self.ax.get_legend()
        if leg is not None:
            try:
                leg.remove()
            except Exception:
                pass

        self.ax.relim()
        self.ax.autoscale_view()
        # 应用当前绘图样式
        try:
            self._apply_plot_style()
        except Exception:
            pass
        self.canvas.draw_idle()

        if self.total_points > 0:
            self.points_label.config(text=f"点数: 0/{self.total_points}")
        else:
            self.points_label.config(text="点数: 0/∞")
        self.eta_label.config(text="剩余时间: --")

    # ---- 导出 & 自动保存 ----

    def export_data(self):
        if not self.current_data:
            messagebox.showinfo("无数据", "当前没有可导出的数据")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        self._save_data_to_csv(path)
        self._log(f"数据已导出到 {path}")

    def export_log(self):
        text = self.log_text.get("1.0", "end").strip()
        if not text:
            messagebox.showinfo("无日志", "当前没有日志内容")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text", "*.txt"), ("All files", "*.*")],
        )
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        self._log(f"日志已导出到 {path}")

    def _save_data_to_csv(self, path, *, extra_comments=None):
        keys = ["index", "timestamp", "setpoint", "voltage", "current", "resistance", "power"]
        with open(path, "w", newline="") as f:
            mode = self.current_mode or ""
            wiring = "4-wire" if getattr(self, "four_wire_var", tk.BooleanVar(value=False)).get() else "2-wire"
            comments = [f"# mode: {mode}"]
            if extra_comments:
                comments.extend(extra_comments)
            comments.append(f"# wiring: {wiring}")
            for line in comments:
                f.write(f"{line}\n")

            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for row in self.current_data:
                writer.writerow({k: row.get(k, "") for k in keys})

    def get_run_directory(self, mode: str) -> str:
        """
        根据测量模式统一决定保存目录: 根目录 / MODE / YYYY-MM-DD
        """

        base = self.save_root_var.get().strip() or os.getcwd()
        date_str = datetime.datetime.now().strftime("%Y-%m-%d")
        mode = mode.upper()

        dir_path = os.path.join(base, mode, date_str)
        os.makedirs(dir_path, exist_ok=True)
        return dir_path

    def make_output_path(self, mode: str, suffix: str = ".csv", extra: str = "") -> str:
        """统一生成测量结果文件路径"""

        run_dir = self.get_run_directory(mode)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        mode = mode.upper()

        if extra:
            filename = f"{mode}_{extra}_{ts}{suffix}"
        else:
            filename = f"{mode}_{ts}{suffix}"

        return os.path.join(run_dir, filename)

    def _auto_save_current(self):
        if not self.current_data:
            return
        if not self.save_root_var.get().strip():
            return
        mode = (self.current_mode or "IV").upper()
        path = self.make_output_path(mode)
        self._save_data_to_csv(path)
        self._log(f"自动保存到 {path}")

    def _save_pressure_iv_file(self, pressure_g: float, *, mark_bad: bool = False) -> str:
        if not self.current_data:
            raise ValueError("当前没有可保存的数据")

        pressure_str = f"{pressure_g:g}"
        label = f"{pressure_str}g_BAD" if mark_bad else f"{pressure_str}g"
        path = self.make_output_path("MULTI_PRESS", extra=label)

        comments = [f"# pressure_g: {pressure_str}"]
        if mark_bad:
            comments.append("# quality: BAD")
        self._save_data_to_csv(path, extra_comments=comments)
        return path

    def check_iv_quality(self, V_list, I_list):
        if not V_list or not I_list or len(V_list) != len(I_list) or len(V_list) < 2:
            return False, {"jump_ratio": 0.0, "flip_count": 0, "base_slope": 0.0, "threshold": 0.0}

        dI = [I_list[i + 1] - I_list[i] for i in range(len(I_list) - 1)]
        abs_dI = [abs(x) for x in dI]
        if not abs_dI:
            return False, {"jump_ratio": 0.0, "flip_count": 0, "base_slope": 0.0, "threshold": 0.0}

        v_min, v_max = min(V_list), max(V_list)
        v_range = v_max - v_min
        mid_abs = []
        if v_range <= 0:
            mid_abs = list(abs_dI)
        else:
            v_low = v_min + 0.1 * v_range
            v_high = v_max - 0.1 * v_range
            for idx, delta in enumerate(abs_dI):
                v_mid = 0.5 * (V_list[idx] + V_list[idx + 1])
                if v_low <= v_mid <= v_high:
                    mid_abs.append(delta)
            if not mid_abs:
                mid_abs = list(abs_dI)

        base_slope = float(np.median(mid_abs)) if mid_abs else 0.0
        k = float(self.iv_quality_k_var.get() or 0.0)
        jump_threshold = k * base_slope
        jump_ratio = sum(1 for val in abs_dI if val > jump_threshold) / len(abs_dI)

        small_eps = max(1e-9, 0.01 * base_slope)
        flip_count = 0
        last_sign = 0
        for delta in dI:
            if delta > small_eps:
                sign = 1
            elif delta < -small_eps:
                sign = -1
            else:
                continue
            if last_sign and sign != last_sign:
                flip_count += 1
            last_sign = sign

        max_jump_ratio = float(self.iv_quality_jump_ratio_var.get() or 0.0)
        try:
            max_flip_count = int(self.iv_quality_flip_count_var.get())
        except Exception:
            max_flip_count = 0
        is_bad = jump_ratio > max_jump_ratio and flip_count > max_flip_count
        return is_bad, {
            "jump_ratio": jump_ratio,
            "flip_count": flip_count,
            "base_slope": base_slope,
            "threshold": jump_threshold,
        }

    def _ensure_multi_pressure_folder(self) -> str:
        return self.get_run_directory("MULTI_PRESS")

    def _apply_pressure_integration(self, pressure: float) -> float:
        """
        当前实现直接返回最新压力值，保留接口以兼容旧逻辑。
        """
        now = time.time()
        self._filtered_pressure = pressure
        self._filtered_pressure_ts = now
        return pressure

    def read_pressure(self):
        """
        读取 40001 (地址 0x0000) 的“测量显示值”，按手册为 16 位有符号数。
        返回值单位与设备当前单位一致（你的 UI 按 g 展示，维持现状）。
        """
        pressure = self.current_pressure
        try:
            if not self.modbus1:
                # 没有传感器连接，直接对现有值做一次“更新”，避免滤波状态发散
                return self._apply_pressure_integration(pressure)

            # 优先按“读 1 个寄存器，返回 2 字节数据”的规范读取
            resp = self.modbus1.read_registers(0x0000, 1)  # 40001
            if resp and len(resp) >= 5 and resp[1] == 0x03 and resp[2] == 0x02:
                hi, lo = resp[3], resp[4]
                val = (hi << 8) | lo
                if val >= 0x8000:  # 16 位有符号
                    val -= 0x10000
                scaled_val = val * self.pressure_scale
                pressure = scaled_val - self.tare_value
                return self._apply_pressure_integration(pressure)

            # 兼容某些固件返回 2 寄存器（4 字节）的旧逻辑（极少用到）
            if resp and len(resp) >= 7 and resp[1] == 0x03 and resp[2] == 0x04:
                data = resp[3:7]
                low_word = int.from_bytes(data[0:2], 'big', signed=True)
                high_word = int.from_bytes(data[2:4], 'big', signed=True)
                val32 = (high_word << 16) | (low_word & 0xFFFF)
                if val32 >= 0x80000000:
                    val32 -= 0x100000000
                scaled_val = val32 * self.pressure_scale
                pressure = scaled_val - self.tare_value
                return self._apply_pressure_integration(pressure)

        except Exception as e:  # noqa: PERF203
            self._log(f"读取压力数据出错: {e}")

        # 出错或无数据时，用当前值进滤波器
        return self._apply_pressure_integration(pressure)

    # ---- 日志 & 退出 ----

    def _log(self, msg):
        ts = time.strftime("%H:%M:%S")
        self.log_text.insert("end", f"[{ts}] {msg}\n")
        self.log_text.see("end")

    def on_close(self):
        if self.measurement_thread is not None and self.measurement_thread.is_alive():
            if not messagebox.askyesno("退出", "测量正在进行，确认要退出吗？"):
                return
        self.stop_event.set()
        if self.measurement_thread is not None and self.measurement_thread.is_alive():
            self.measurement_thread.join(timeout=2.0)
        self._save_settings()
        self._stop_tcp_server()
        self._clear_tcp_waiters()
        try:
            self.instrument.set_low_current_mode(False)
        except Exception:
            pass
        try:
            self.instrument.output_off()
        except Exception:
            pass
        try:
            self.instrument.close()
        except Exception:
            pass
        self.root.destroy()

    # ---- 参数保存 ----

    def _iter_persistable_variables(self):
        runtime_only = {"ofr_pressure_var", "ofr_current_var", "ofr_onoff_var", "ofr_ioff_var"}
        for name, value in vars(self).items():
            if isinstance(value, tk.Variable) and name not in runtime_only:
                yield name, value

    def _save_settings(self):
        cfg = {"variables": {}}
        for name, var in self._iter_persistable_variables():
            try:
                cfg["variables"][name] = var.get()
            except Exception:
                continue
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _apply_variable_settings(self, mapping: dict):
        for name, value in (mapping or {}).items():
            var = getattr(self, name, None)
            if isinstance(var, tk.Variable):
                try:
                    var.set(value)
                except Exception:
                    continue

    def _load_settings(self):
        if not os.path.exists(self.config_path):
            return
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception:
            return

        if "variables" in cfg:
            self._apply_variable_settings(cfg.get("variables", {}))
        else:
            self._load_settings_legacy(cfg)

        self._post_settings_load()

    def _load_settings_legacy(self, cfg: dict):
        self.save_root_var.set(cfg.get("save_root", ""))
        self.auto_save_var.set(cfg.get("auto_save", False))
        self.model_select_var.set(cfg.get("model_select", "自动识别"))
        self.channel_select_var.set(cfg.get("channel_select", "A"))
        self.source_channel_var.set(cfg.get("source_channel_var", self.channel_select_var.get()))
        self.measure_channel_var.set(cfg.get("measure_channel_var", self.channel_select_var.get()))
        self.buffer_mode_var.set(cfg.get("buffer_mode", False))
        self.baud_rate_var.set(str(cfg.get("baud_rate", "9600")))

        if hasattr(self, "integration_time_var"):
            try:
                tau = float(cfg.get("integration_nplc", cfg.get("pressure_integration_seconds", 0.0)))
            except Exception:
                tau = 0.0
            if tau < 0:
                tau = 0.0
            self.integration_time_var.set(tau)

        self.low_current_speed_mode_var.set(cfg.get("low_current_speed_mode", False))
        self.current_range_override_var.set(str(cfg.get("current_range_override", "1e-6")))

        if hasattr(self, "four_wire_var"):
            self.four_wire_var.set(cfg.get("four_wire", False))
        if hasattr(self, "plot_style_var"):
            self.plot_style_var.set(cfg.get("plot_style", "线"))

        iv = cfg.get("iv", {})
        self.iv_source_mode_var.set(iv.get("source_mode", "Voltage"))
        self.iv_start_var.set(iv.get("start", -1.0))
        self.iv_stop_var.set(iv.get("stop", 1.0))
        self.iv_step_var.set(iv.get("step", 0.02))
        self.iv_points_var.set(iv.get("points", 101))
        self.iv_cycles_var.set(iv.get("cycles", 1))
        self.iv_backforth_var.set(iv.get("back_and_forth", False))
        self.iv_triangle_from_zero_var.set(iv.get("triangle_from_zero", False))
        self.iv_delay_var.set(iv.get("delay", 0.0))
        self.iv_compliance_var.set(iv.get("compliance", 0.1))

        iv_quality = cfg.get("iv_quality", {})
        try:
            self.iv_quality_k_var.set(float(iv_quality.get("k", 8.0)))
        except Exception:
            self.iv_quality_k_var.set(8.0)
        try:
            self.iv_quality_jump_ratio_var.set(float(iv_quality.get("max_jump_ratio", 0.02)))
        except Exception:
            self.iv_quality_jump_ratio_var.set(0.02)
        try:
            self.iv_quality_flip_count_var.set(int(iv_quality.get("max_flip_count", 20)))
        except Exception:
            self.iv_quality_flip_count_var.set(20)
        try:
            self.iv_quality_max_retry_var.set(int(iv_quality.get("max_retry", 2)))
        except Exception:
            self.iv_quality_max_retry_var.set(2)

        it = cfg.get("it", {})
        self.it_bias_var.set(it.get("bias", 0.0))
        self.it_delay_var.set(it.get("delay", 0.0))
        self.it_points_var.set(it.get("points", 50))
        self.it_infinite_var.set(it.get("infinite", False))
        self.it_compliance_var.set(it.get("compliance", 0.1))

        vt = cfg.get("vt", {})
        self.vt_bias_var.set(vt.get("bias", 0.0))
        self.vt_delay_var.set(vt.get("delay", 0.0))
        self.vt_points_var.set(vt.get("points", 50))
        self.vt_infinite_var.set(vt.get("infinite", False))
        self.vt_compliance_var.set(vt.get("compliance", 10.0))

        rt = cfg.get("rt", {})
        self.rt_bias_var.set(rt.get("bias", 0.0))
        self.rt_delay_var.set(rt.get("delay", 0.0))
        self.rt_points_var.set(rt.get("points", 50))
        self.rt_infinite_var.set(rt.get("infinite", False))
        self.rt_compliance_var.set(rt.get("compliance", 0.1))

        pt = cfg.get("pt", {})
        self.pt_bias_var.set(pt.get("bias", 0.0))
        self.pt_delay_var.set(pt.get("delay", 0.0))
        self.pt_points_var.set(pt.get("points", 50))
        self.pt_infinite_var.set(pt.get("infinite", False))
        self.pt_compliance_var.set(pt.get("compliance", 0.1))

        ofr = cfg.get("ofr", {})
        self.ofr_voltage_var.set(ofr.get("voltage", 0.1))
        self.ofr_zero_tol_var.set(ofr.get("zero_tol", 5.0))
        self.ofr_bin_step_var.set(ofr.get("bin_step", 10.0))
        self.ofr_off_min_points_var.set(ofr.get("off_min", 5))

        tcp = cfg.get("tcp", {})
        self.tcp_host_var.set(tcp.get("host", "127.0.0.1"))
        self.tcp_port_var.set(tcp.get("port", 50000))

    def _post_settings_load(self):
        try:
            self._apply_channel_selection_to_instrument()
        except Exception:
            pass
        try:
            self._sync_model_channel_controls()
            self._sync_baud_control()
        except Exception:
            pass
        try:
            self._on_low_current_toggle()
        except Exception:
            pass
        try:
            if hasattr(self, "plot_style_var"):
                self._apply_plot_style()
        except Exception:
            pass

    def run(self):
        self.root.mainloop()

    # ---- TCP 从机 ----

    def apply_tcp_settings(self):
        self._save_settings()
        self._start_tcp_server()

    def _start_tcp_server(self):
        self._stop_tcp_server()
        host = (self.tcp_host_var.get() or "127.0.0.1").strip()
        try:
            port = int(self.tcp_port_var.get())
        except Exception:
            port = 50000
            self.tcp_port_var.set(port)
        self.tcp_host_var.set(host or "127.0.0.1")

        self.tcp_stop_event.clear()
        self.tcp_server_thread = threading.Thread(
            target=self._tcp_server_loop, args=(self.tcp_host_var.get(), port), daemon=True
        )
        self.tcp_server_thread.start()

    def _stop_tcp_server(self):
        self.tcp_stop_event.set()
        try:
            with socket.create_connection((self.tcp_host_var.get(), int(self.tcp_port_var.get())), timeout=0.2):
                pass
        except Exception:
            pass
        if self.tcp_server_thread is not None and self.tcp_server_thread.is_alive():
            self.tcp_server_thread.join(timeout=1.0)

    def _tcp_server_loop(self, host, port):
        sock = None
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.settimeout(1.0)

            try:
                sock.bind((host, port))
                sock.listen(5)
            except OSError as exc:
                self.queue.put(("log", f"TCP 绑定失败 {host}:{port}: {exc}"))
                return

            self.queue.put(("log", f"TCP 从机监听 {host}:{port}"))

            while not self.tcp_stop_event.is_set():
                try:
                    conn, addr = sock.accept()
                    conn.settimeout(10.0)
                    threading.Thread(
                        target=self._handle_tcp_client, args=(conn, addr), daemon=True
                    ).start()
                    self.queue.put(("log", f"TCP 客户端连接: {addr}"))
                except socket.timeout:
                    continue
                except OSError as exc:
                    if not self.tcp_stop_event.is_set():
                        self.queue.put(("log", f"TCP 接受连接错误: {exc}"))
                    continue
        except Exception as exc:  # noqa: BLE001
            self.queue.put(("log", f"TCP 服务器错误: {exc}"))
        finally:
            if sock:
                try:
                    sock.close()
                except Exception:
                    pass

    def _handle_tcp_client(self, conn, addr):
        with conn:
            try:
                conn.settimeout(1.0)
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except Exception:
                pass

            buffer = b""
            while not self.tcp_stop_event.is_set():
                try:
                    chunk = conn.recv(1024)
                except socket.timeout:
                    continue
                except Exception:
                    return

                if not chunk:
                    return

                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    raw = line.strip()
                    if not raw:
                        continue
                    handled = False
                    try:
                        payload = json.loads(raw.decode("utf-8"))
                    except Exception:
                        payload = None
                    if isinstance(payload, dict) and payload.get("cmd"):
                        if not self._process_tcp_json(conn, payload):
                            return
                        handled = True
                    if handled:
                        continue
                    command = raw.decode(errors="ignore").lower()
                    if not command:
                        continue
                    if not self._process_tcp_command(conn, command):
                        return

                if len(buffer) > 4096:
                    # 防止异常数据把线程拖住
                    return

    def _safe_send_tcp(self, conn, payload: str) -> bool:
        try:
            conn.sendall(payload.encode())
            return True
        except Exception as exc:
            summary = payload.strip().split("\n", 1)[0]
            self.queue.put(("log", f"TCP 发送失败: {summary}, 异常: {exc}"))
            return False

    def _safe_send_tcp_json(self, conn, payload: dict) -> bool:
        try:
            line = json.dumps(payload, ensure_ascii=False) + "\n"
            conn.sendall(line.encode("utf-8"))
            return True
        except Exception as exc:
            self.queue.put(("log", f"TCP 发送 JSON 失败: {payload}, 异常: {exc}"))
            return False

    def _process_tcp_json(self, conn, payload: dict) -> bool:
        cmd = payload.get("cmd")
        if cmd == "OFR_TEST_START":
            self.handle_ofr_start(payload, conn)
            return True
        if cmd == "PRESSURE_UPDATE":
            self.handle_ofr_pressure_update(payload)
            return True
        if cmd == "OFR_TEST_STOP":
            self.handle_ofr_stop(payload)
            return True
        if cmd == "OFR_TEST_ABORT":
            self.handle_ofr_abort(payload)
            return True
        self._safe_send_tcp_json(conn, {"cmd": "OFR_TEST_ERROR", "error": f"unknown cmd {cmd}"})
        return True

    def _process_tcp_command(self, conn, command: str) -> bool:
        if command == "start":
            self._begin_multi_pressure_session()
            return True

        if command.startswith("pressure"):
            parts = command.split()
            if len(parts) == 2:
                try:
                    value = float(parts[1])
                except ValueError:
                    self.queue.put(("log", f"忽略无效压力指令: {command}"))
                    return True
                self._set_multi_pressure_value(value)
            else:
                self.queue.put(("log", f"忽略无效压力指令: {command}"))
            return True

        if command == "run":
            return self._handle_tcp_run(conn)

        if command == "done":
            self._finalize_multi_pressure_session()
            return True

        self._safe_send_tcp(conn, "unknown\n")
        return True

    def _handle_tcp_run(self, conn) -> bool:
        if not self.multi_tcp_active:
            self.queue.put(("log", "TCP run 被忽略：未收到 start 指令"))
            self._safe_send_tcp(conn, "error\n")
            return True

        if self.multi_tcp_pressure is None:
            self.queue.put(("log", "TCP run 被忽略：尚未提供 pressure 指令"))
            self._safe_send_tcp(conn, "error\n")
            return True

        pressure_value = self.multi_tcp_pressure

        ack_event = threading.Event()
        done_event = threading.Event()
        ack_event.started = False

        def start_from_main():
            cfg = self._collect_iv_config()
            if cfg is None:
                ack_event.started = False
                ack_event.set()
                return
            self.multi_tcp_last_iv_config = copy.deepcopy(cfg)
            self.multi_tcp_retry_used = 0
            started = self._initiate_measurement("IV", cfg, show_dialog=False)
            ack_event.started = started
            ack_event.set()
            if started:
                self.multi_tcp_pending_pressure = pressure_value
                with self.tcp_waiters_lock:
                    self.tcp_waiters.append(done_event)

        self.root.after(0, start_from_main)
        ack_event.wait(timeout=5.0)
        if not ack_event.is_set() or not getattr(ack_event, "started", False):
            self._safe_send_tcp(conn, "error\n")
            return True

        while not done_event.wait(timeout=0.5):
            if self.tcp_stop_event.is_set():
                self._safe_send_tcp(conn, "error\n")
                return False

        self._safe_send_tcp(conn, "next\n")
        return True

    def _begin_multi_pressure_session(self):
        self.multi_tcp_active = True
        self.multi_tcp_pressure = None
        self.multi_tcp_pending_pressure = None
        self.multi_tcp_records.clear()
        self.multi_tcp_session_start = time.strftime("%Y%m%d_%H%M%S")
        self.multi_tcp_retry_used = 0
        self.multi_tcp_last_iv_config = None
        self.queue.put(("log", "TCP 多压力会话已重置"))

    def _set_multi_pressure_value(self, value: float):
        if not self.multi_tcp_active:
            self.queue.put(("log", "pressure 指令被忽略：请先发送 start"))
            return
        self.multi_tcp_pressure = value
        self.queue.put(("log", f"当前压力设定为 {value:g}g"))

    def _load_iv_file(self, path: str):
        voltages = []
        currents = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(row for row in f if not row.lstrip().startswith("#"))
            for row in reader:
                try:
                    voltages.append(float(row.get("voltage", 0.0)))
                    currents.append(float(row.get("current", 0.0)))
                except Exception:
                    continue
        return voltages, currents

    def _generate_multi_pressure_summary(self):
        if not self.multi_tcp_records:
            self.queue.put(("log", "本轮无压力数据，跳过汇总"))
            return None

        records = {}
        quality_flags = {}
        for pressure_g, path, is_bad in self.multi_tcp_records:
            if not os.path.exists(path):
                self.queue.put(("log", f"跳过不存在的文件: {path}"))
                continue
            try:
                records[pressure_g] = self._load_iv_file(path)
                quality_flags[pressure_g] = is_bad
            except Exception as exc:  # noqa: BLE001
                self.queue.put(("log", f"读取 {path} 失败: {exc}"))

        if not records:
            self.queue.put(("log", "没有可用的多压力数据，无法汇总"))
            return None

        sorted_pressures = sorted(records.keys())
        base_voltages = records[sorted_pressures[0]][0]

        header = ["Voltage(V)"] + [
            f"{p:g}g{'(BAD)' if quality_flags.get(p) else ''}" for p in sorted_pressures
        ]
        rows = []
        for idx, v in enumerate(base_voltages):
            row = [v]
            for p in sorted_pressures:
                volts, currents = records[p]
                value = currents[idx] if idx < len(currents) else ""
                if idx < len(volts) and abs(volts[idx] - v) > 1e-6:
                    self.queue.put(("log", f"警告: {p:g}g 第 {idx} 点电压不匹配"))
                row.append(value)
            rows.append(row)

        min_p, max_p = min(sorted_pressures), max(sorted_pressures)
        pressures_label = "_".join(f"{p:g}g" for p in sorted_pressures)
        if len(pressures_label) > 80:
            pressures_label = f"{min_p:g}g_to_{max_p:g}g"
        summary_path = self.make_output_path("MULTI_PRESS", extra=pressures_label)
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([f"# Pressures: {', '.join(f'{p:g}g' for p in sorted_pressures)}"])
            writer.writerow(header)
            writer.writerows(rows)

        self.queue.put(("log", f"多压力汇总已生成: {summary_path}"))
        return summary_path

    def _finalize_multi_pressure_session(self):
        try:
            self._generate_multi_pressure_summary()
        except Exception as exc:  # noqa: BLE001
            self.queue.put(("log", f"汇总失败: {exc}"))
        finally:
            self.multi_tcp_active = False
            self.multi_tcp_pressure = None
            self.multi_tcp_pending_pressure = None
            self.multi_tcp_records.clear()
            self.multi_tcp_retry_used = 0
            self.multi_tcp_last_iv_config = None

    # ---- OFR 测试处理 ----

    def configure_2400_for_ofr(self, voltage: float):
        compliance = 0.1
        self.instrument.configure_source("Voltage", voltage, compliance=compliance)

    def read_current_once(self) -> float:
        data = self.instrument.measure_once()
        return float(data.get("current", 0.0))

    def _read_voltage_and_current(self) -> tuple[float, float]:
        data = self.instrument.measure_once()
        return float(data.get("voltage", 0.0)), float(data.get("current", 0.0))

    def _prepare_ofr_plot(self):
        self.ofr_pressures.clear()
        self.ofr_onoff_values.clear()

        if self.ofr_line is None:
            self.ofr_line, = self.ax.plot([], [], "o-", label="ON/OFF", color="#e67e22")
        else:
            self.ofr_line.set_data([], [])

        self.voltage_line.set_data([], [])
        self.current_line.set_data([], [])
        self.ax.set_title("ON/OFF vs Pressure")
        self.ax.set_xlabel("Pressure (g)")
        self.ax.set_ylabel("Switching ratio (I/I_off)")
        self.ax.relim()
        self.ax.autoscale_view()
        self._apply_plot_style()
        self.canvas.draw_idle()

    def _simulate_ofr_current(self, pressure_g: float, v_test: float) -> tuple[float, float]:
        """
        开关比测试仿真模型：
        给定压力（g）和测试电压，返回 (V_meas, I_meas)。

        模型特点：
        - P < ~几百 g：电流接近 I_off，nA 级别
        - P ~ 800 g 左右开始陡升
        - P -> 10000 g：电流趋于 mA 级饱和，并随压力略有增加
        - 叠加相对噪声 + 绝对噪声
        """
        V = float(v_test)
        P = max(0.0, float(pressure_g))

        V_abs = max(abs(V), 1e-3)

        # 关态 / 开态电流（随电压线性放大一点）
        I_off = 2e-9 * V_abs      # ~ nA 级
        I_on_max = 2e-3 * V_abs   # ~ mA 级

        # 物流斯型压力响应
        P0 = 800.0     # 转折点（接近器件“导通”压力）
        width = 1200.0 # 过渡宽度
        x = (P - P0) / width
        frac = 1.0 / (1.0 + math.exp(-x))

        I = I_off + (I_on_max - I_off) * frac

        # 高压区再加一点线性增强，模拟进一步压实导电通道
        I *= (1.0 + 0.2 * (P / 10000.0))

        # 噪声：相对噪声 + 绝对噪声
        sigma = 0.03 * abs(I) + 5e-10
        I_noisy = I + random.gauss(0.0, sigma)

        # 电流符号跟随电压符号
        if V < 0:
            I_noisy = -I_noisy

        # 电压读数也给一点小抖动
        V_meas = V + random.uniform(-0.001, 0.001)

        return V_meas, I_noisy

    def start_ofr_simulation(self):
        """
        在“仿真模式”下，从 0→10000 g、步进 1 g 自动执行一次开关比测试仿真。
        - 仅在左上角勾选“仿真模式”时可用；
        - 复用 handle_ofr_start / finalize_and_export_ofr_results 的逻辑；
        - 结果会自动按 OFR 规则导出两份 CSV（raw + binned）。
        """
        # 必须在仿真模式下才允许运行
        if not self.sim_var.get():
            messagebox.showinfo("提示", "请先勾选左上角的“仿真模式”再使用开关比测试仿真。")
            return

        # 如果已有 OFR 正在跑，避免重入
        if self.ofr_active:
            messagebox.showinfo("提示", "当前已有开关比测试在进行中。")
            return

        if self.ofr_sim_thread is not None and self.ofr_sim_thread.is_alive():
            messagebox.showinfo("提示", "开关比测试仿真线程正在运行。")
            return

        # 用现有逻辑初始化 OFR 状态 & 配置 2400
        sim_test_id = time.strftime("SIM_OFR_%Y%m%d_%H%M%S")
        msg = {"cmd": "OFR_TEST_START", "test_id": sim_test_id}
        # conn=None 时 _safe_send_tcp_json 会安静失败，不会影响本地逻辑
        self.handle_ofr_start(msg, conn=None)

        if not self.ofr_active:
            # 如果初始化失败（比如配置 2400 出错），直接退出
            return

        self.queue.put(("log", f"[OFR] 启动仿真开关比测试: test_id={sim_test_id} (0–10000 g, ΔP=1 g)"))

        self.ofr_sim_stop.clear()
        self.ofr_sim_thread = threading.Thread(target=self._ofr_sim_loop, daemon=True)
        self.ofr_sim_thread.start()

    def stop_ofr_simulation(self):
        """手动停止仿真开关比测试。"""
        if self.ofr_sim_thread is None or not self.ofr_sim_thread.is_alive():
            messagebox.showinfo("提示", "当前没有正在运行的仿真开关比测试。")
            return

        self.queue.put(("log", "[OFR] 收到仿真停止指令，正在结束…"))
        self.ofr_sim_stop.set()
        thread = self.ofr_sim_thread
        self.ofr_active = False
        try:
            self.instrument.output_off()
        except Exception:
            pass
        if thread:
            thread.join(timeout=1.0)
        self.ofr_sim_thread = None
        self.finalize_and_export_ofr_results(aborted=True)

    def _ofr_sim_loop(self):
        """
        在后台线程中执行：
        - P: 0 → 10000 g, step = 1 g
        - 使用 _simulate_ofr_current 生成电流
        - 复用开关比统计逻辑（add_ofr_sample_for_stats / get_ratio_for_pressure）
        """
        try:
            v_test = float(self.ofr_voltage_var.get())
        except Exception:
            v_test = 0.1

        test_id = self.ofr_test_id
        t0 = time.time()
        dt = 0.002  # 每个点约 2 ms，可根据体验调整

        for idx, p in enumerate(range(0, 10001)):
            # 支持外部中止（未来如果加“停止仿真”按钮可以用 ofr_sim_stop）
            if self.ofr_sim_stop.is_set():
                break
            # 如果在仿真过程中被别的 OFR 测试覆盖，也退出
            if not self.ofr_active or test_id != self.ofr_test_id:
                break

            timestamp = t0 + idx * dt

            if self.ofr_t0 is None:
                self.ofr_t0 = timestamp
            t_rel = timestamp - self.ofr_t0

            # 用仿真模型生成当前点的 (V, I)
            v_meas, i_meas = self._simulate_ofr_current(p, v_test)

            # 完全复用 handle_ofr_pressure_update 内部的统计逻辑
            self.ofr_samples.append((t_rel, v_meas, i_meas, float(p)))
            self.add_ofr_sample_for_stats(float(p), i_meas)

            onoff = self.get_ratio_for_pressure(float(p))
            if onoff is None:
                floor, _, _ = self._calc_off_stats()
                if floor not in (None, 0):
                    onoff = abs(i_meas) / floor

            self.update_ofr_gui_async(float(p), i_meas, onoff)

            # 控制节奏，避免占用过高 CPU，同时让曲线有“实时感”
            if dt > 0:
                time.sleep(dt)

        # 正常扫完 0–10000 g，且未被外部终止：自动收尾并导出结果
        if (
            test_id == self.ofr_test_id
            and self.ofr_active
            and not self.ofr_sim_stop.is_set()
        ):
            self.ofr_active = False
            self.finalize_and_export_ofr_results(aborted=False)
            self.queue.put(("log", "[OFR] 仿真开关比测试完成 (0–10000 g, ΔP=1 g)"))

    def show_ofr_help(self):
        text = (
            "开关比 (On/Off Ratio) 计算说明：\n\n"
            "1. 定义：\n"
            "   本软件中开关比定义为：\n"
            "   On/Off = |I_on| / max(|I_off_mean|, k·σ_off, I_instr_floor)\n"
            "   其中 I_on 为各压力点下的平均电流(取绝对值)，\n"
            "   I_off_mean 和 σ_off 来自 0g 附近关态电流的统计，\n"
            "   I_instr_floor 为仪器电流下限。\n\n"
            "2. 负关电流的处理：\n"
            "   关态电流测得为负值时，通常是测量噪声或零点漂移所致，\n"
            "   软件会对电流取绝对值，并结合噪声统计与电流下限来计算开关比，\n"
            "   避免出现物理上无意义的巨大或负的开关比。\n\n"
            "3. 同一压力点多次采样：\n"
            "   在同一压力附近多次采样时，软件会按压力分组求平均电流，\n"
            "   再用平均电流计算开关比，并绘制压力–开关比曲线，\n"
            "   以减小噪声影响。\n\n"
            "4. 自动保存文件：\n"
            "   OFR 自动保存结果包含列：t(s), V(V), I(A), Pressure(g), OnOffRatio。\n"
            "   文件头会注明计算公式和所用的 I_instr_floor、k 等参数。\n"
        )
        messagebox.showinfo("OFR 开关比说明", text)

    def _calc_off_stats(self):
        if not self.ofr_stats:
            return None, None, None

        # 先刷新压力量化后的均值表，保证关态统计使用最新均值
        self.compute_ofr_mean_curve()
        p_threshold = float(self.ofr_zero_tol_var.get())
        off_currents = [
            i_mean
            for p_bin, i_mean in self.ofr_I_mean_by_pressure.items()
            if abs(p_bin) <= p_threshold
        ]

        mean_off = statistics.mean(off_currents) if off_currents else None
        if off_currents and len(off_currents) > 1:
            sigma_off = statistics.pstdev(off_currents)
        elif off_currents:
            sigma_off = 0.0
        else:
            sigma_off = None

        floor = self.compute_off_effective(
            off_currents=off_currents,
            I_instr_floor=self.ofr_instr_floor,
            k=self.ofr_noise_k,
        )
        return floor, mean_off, sigma_off

    def bin_pressure(self, pressure: float) -> float:
        dP = float(self.ofr_bin_step_var.get())
        return round(pressure / dP) * dP if dP > 0 else pressure

    def add_ofr_sample_for_stats(self, pressure: float, current: float):
        p_bin = self.bin_pressure(pressure)
        cnt, s = self.ofr_stats[p_bin]
        cnt += 1
        s += current
        self.ofr_stats[p_bin] = [cnt, s]

    def compute_ofr_mean_curve(self):
        pressures = []
        I_mean_list = []
        self.ofr_I_mean_by_pressure.clear()

        for p_bin in sorted(self.ofr_stats.keys()):
            cnt, s = self.ofr_stats[p_bin]
            if cnt <= 0:
                continue
            i_mean = s / cnt
            pressures.append(p_bin)
            I_mean_list.append(i_mean)
            self.ofr_I_mean_by_pressure[p_bin] = i_mean

        return pressures, I_mean_list

    def compute_off_effective(self, off_currents: list[float], I_instr_floor: float, k: float = 3.0) -> float:
        """
        从关态样本中估计有效关态电流，允许关态电流为负，分母统一使用绝对值 + 噪声下限：
        I_off_eff = max(|mu_off|, k * sigma_off, I_instr_floor)
        """
        if not off_currents:
            return I_instr_floor

        mu = statistics.mean(off_currents)
        sigma = statistics.pstdev(off_currents) if len(off_currents) > 1 else 0.0
        return max(abs(mu), k * sigma, I_instr_floor)

    def compute_on_off_curve(self, I_mean_by_pressure: dict[float, float], I_off_eff: float):
        pressures = []
        ratios = []

        for p_bin in sorted(I_mean_by_pressure.keys()):
            i_mean = I_mean_by_pressure[p_bin]
            i_on_eff = abs(i_mean)  # 关态电流可能为负，分子统一取绝对值
            ratio = i_on_eff / I_off_eff if I_off_eff > 0 else float("inf")
            pressures.append(p_bin)
            ratios.append(ratio)

        return pressures, ratios

    def get_ratio_for_pressure(self, pressure: float):
        floor, _, _ = self._calc_off_stats()
        if floor in (None, 0):
            return None

        p_bin = self.bin_pressure(pressure)
        i_mean = self.ofr_I_mean_by_pressure.get(p_bin)
        if i_mean is None:
            stats = self.ofr_stats.get(p_bin)
            if not stats:
                return None
            cnt, s = stats
            if cnt <= 0:
                return None
            i_mean = s / cnt
        return abs(i_mean) / floor

    def handle_ofr_start(self, msg: dict, conn):
        voltage = float(self.ofr_voltage_var.get())
        self.ofr_test_id = msg.get("test_id", time.strftime("OFR_%Y%m%d_%H%M%S"))

        self.ofr_raw_points = []
        self.ofr_off_points = []
        self.ofr_I_off = None
        self.ofr_stats = defaultdict(lambda: [0, 0.0])
        self.ofr_I_mean_by_pressure = {}
        self.ofr_active = False
        self.ofr_samples = []
        self.ofr_t0 = None

        self._prepare_ofr_plot()

        if self.instrument.simulated:
            self.queue.put(("log", "[OFR] 仿真模式下启动开关比测试"))
        elif self.instrument.session is None:
            error_msg = "SMU 未连接或会话无效"
            self._safe_send_tcp_json(
                conn,
                {
                    "cmd": "OFR_TEST_ERROR",
                    "test_id": self.ofr_test_id,
                    "error": error_msg,
                },
            )
            self.queue.put(("log", f"[OFR] {error_msg}"))
            return

        try:
            self.configure_2400_for_ofr(voltage)
        except Exception as exc:  # noqa: BLE001
            self.ofr_active = False
            try:
                self.instrument.output_off()
            except Exception:
                pass
            self._safe_send_tcp_json(
                conn,
                {
                    "cmd": "OFR_TEST_ERROR",
                    "test_id": self.ofr_test_id,
                    "error": str(exc),
                },
            )
            self.queue.put(("log", f"[OFR] 配置 SMU 失败: {exc}"))
            return

        if conn is None:
            ready_sent = True
        else:
            ready_sent = self._safe_send_tcp_json(
                conn, {"cmd": "OFR_TEST_READY", "test_id": self.ofr_test_id}
            )

        if not ready_sent:
            self.ofr_active = False
            try:
                self.instrument.output_off()
            except Exception:
                pass
            self.queue.put(("log", "[OFR] READY 回包发送失败，终止本次测试"))
            return

        self.ofr_active = True
        self.queue.put(("log", f"[OFR] 已进入开关比测试模式，V_test={voltage}"))

    def handle_ofr_pressure_update(self, msg: dict):
        if not self.ofr_active:
            return
        if msg.get("test_id") != self.ofr_test_id:
            return

        timestamp = float(msg.get("t", time.time()))
        pressure = float(msg.get("pressure", 0.0))

        if self.ofr_t0 is None:
            self.ofr_t0 = timestamp
        t_rel = timestamp - self.ofr_t0

        try:
            voltage, current = self._read_voltage_and_current()
        except Exception as exc:  # noqa: BLE001
            self.queue.put(("log", f"[OFR] 读电流失败: {exc}"))
            return

        self.ofr_samples.append((t_rel, voltage, current, pressure))
        self.add_ofr_sample_for_stats(pressure, current)

        onoff = self.get_ratio_for_pressure(pressure)
        if onoff is None:
            floor, _, _ = self._calc_off_stats()
            if floor not in (None, 0):
                onoff = abs(current) / floor

        self.update_ofr_gui_async(pressure, current, onoff)

    def handle_ofr_stop(self, msg: dict):
        if msg.get("test_id") != self.ofr_test_id:
            return
        self.queue.put(("log", "[OFR] 收到 OFR_TEST_STOP，结束测试"))
        self.ofr_active = False
        try:
            self.instrument.output_off()
        except Exception:
            pass
        self.finalize_and_export_ofr_results(aborted=False)
        self.ofr_test_id = ""

    def handle_ofr_abort(self, msg: dict):
        if msg.get("test_id") != self.ofr_test_id:
            return
        self.queue.put(("log", f"[OFR] 收到 OFR_TEST_ABORT: {msg.get('reason')}"))
        self.ofr_active = False
        try:
            self.instrument.output_off()
        except Exception:
            pass
        self.finalize_and_export_ofr_results(aborted=True)
        self.ofr_test_id = ""

    def update_ofr_gui_async(self, pressure: float, current: float, onoff):
        def _update():
            self.ofr_pressure_var.set(f"P: {pressure:.3f}")
            self.ofr_current_var.set(f"I: {current:.6e} A")
            floor, mean_off, sigma_off = self._calc_off_stats()
            if mean_off is not None:
                self.ofr_ioff_var.set(f"I_off: {mean_off:.6e} A, σ={sigma_off:.3e}")
            if onoff is not None:
                self.ofr_onoff_var.set(f"ON/OFF: {onoff:.3e}")
            if floor is not None:
                self._update_ofr_plot_ui()

        self.root.after(0, _update)

    def _update_ofr_plot_ui(self):
        if self.ofr_line is None:
            return

        floor, _, _ = self._calc_off_stats()
        pressures, I_mean_list = self.compute_ofr_mean_curve()
        if floor in (None, 0):
            onoffs = []
            pressures = []
        else:
            pressures, onoffs = self.compute_on_off_curve(self.ofr_I_mean_by_pressure, floor)

        self.ofr_pressures = pressures
        self.ofr_onoff_values = onoffs
        self.ofr_line.set_data(self.ofr_pressures, self.ofr_onoff_values)
        self.ax.set_title("ON/OFF vs Pressure")
        self.ax.set_xlabel("Pressure (g)")
        self.ax.set_ylabel("Switching ratio (I/I_off)")
        self.ax.relim()
        self.ax.autoscale_view()
        self._apply_plot_style()
        self.canvas.draw_idle()

    def save_dict_list_to_csv(self, path: str, rows: list[dict]):
        if not rows:
            return
        fieldnames = list(rows[0].keys())
        dir_path = os.path.dirname(path) or "."
        os.makedirs(dir_path, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def finalize_and_export_ofr_results(self, aborted: bool):
        if not self.ofr_samples:
            return

        self.compute_ofr_mean_curve()
        floor, mean_off, sigma_off = self._calc_off_stats()
        ratios_by_bin = {}
        if floor not in (None, 0):
            pressures_ratio, ratios = self.compute_on_off_curve(self.ofr_I_mean_by_pressure, floor)
            ratios_by_bin = dict(zip(pressures_ratio, ratios))

        rows_raw = []
        for (t_rel, v, i, p) in self.ofr_samples:
            p_bin = self.bin_pressure(p)
            onoff = ratios_by_bin.get(p_bin)
            if onoff is None and floor not in (None, 0):
                onoff = abs(i) / floor
            rows_raw.append(
                {
                    "t(s)": t_rel,
                    "V(V)": v,
                    "I(A)": i,
                    "Pressure(g)": p,
                    "OnOffRatio": onoff if onoff is not None else "",
                    "I_off_mean": mean_off if mean_off is not None else "",
                    "I_off_sigma": sigma_off if sigma_off is not None else "",
                    "denominator": floor if floor is not None else "",
                }
            )

        rows_bin = []
        for p_bin in sorted(self.ofr_stats.keys()):
            cnt, s = self.ofr_stats[p_bin]
            if cnt <= 0:
                continue
            i_mean = s / cnt
            onoff_mean = ratios_by_bin.get(p_bin, "") if floor not in (None, 0) else ""
            rows_bin.append(
                {
                    "P_bin": p_bin,
                    "I_mean": i_mean,
                    "ON_OFF_mean": onoff_mean,
                    "N_points": cnt,
                    "I_off_mean": mean_off if mean_off is not None else "",
                    "I_off_sigma": sigma_off if sigma_off is not None else "",
                    "denominator": floor if floor is not None else "",
                }
            )

        extra_id = (self.ofr_test_id or "").replace(" ", "_")
        raw_extra = f"{extra_id}_raw" if extra_id else "raw"
        bin_extra = f"{extra_id}_binned" if extra_id else "binned"
        if aborted:
            raw_extra += "_aborted"
            bin_extra += "_aborted"

        export_raw_path = self.make_output_path("OFR", extra=raw_extra)
        export_bin_path = self.make_output_path("OFR", extra=bin_extra)

        try:
            self.save_dict_list_to_csv(export_raw_path, rows_raw)
            self.save_dict_list_to_csv(export_bin_path, rows_bin)
            self.queue.put(("log", f"[OFR] 结果导出完成: {export_raw_path}, {export_bin_path}"))
        except Exception as exc:  # noqa: BLE001
            self.queue.put(("log", f"[OFR] 导出失败: {exc}"))

    def _notify_tcp_waiters(self):
        with self.tcp_waiters_lock:
            waiters = list(self.tcp_waiters)
            self.tcp_waiters.clear()
        for ev in waiters:
            ev.set()

    def _clear_tcp_waiters(self):
        self._notify_tcp_waiters()


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    app.run()
