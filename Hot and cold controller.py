# -*- coding: utf-8 -*-
import json
import os
import csv
import time
import struct
import serial
import socket
import threading
from tkinter import filedialog
from math import isnan, sqrt
from pathlib import Path
from datetime import datetime
from collections import deque

import tkinter as tk
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinter import messagebox
import serial.tools.list_ports

from tcp_utils import JSONLineServer

# ---- 可选依赖：鼠标/键盘 ----
try:
    import pyautogui
    pyautogui.FAILSAFE = False
except Exception:
    pyautogui = None

try:
    import keyboard   # 全局热键库（可选）
except Exception:
    keyboard = None

import sys, ctypes
# ---- Windows 鼠标坐标结构体 ----
if sys.platform.startswith("win"):
    class _POINT(ctypes.Structure):
        _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

if getattr(sys, "frozen", False):
    pyautogui = None
    keyboard = None

import sys
import ctypes
import time as _time

# Matplotlib for live plots
import matplotlib
matplotlib.use("TkAgg")
# --- Matplotlib 加速设置 ---
matplotlib.rcParams['path.simplify'] = True
matplotlib.rcParams['path.simplify_threshold'] = 1.0
matplotlib.rcParams['agg.path.chunksize'] = 10000

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# =========================
# 常量 & 默认参数
# =========================
APP_NAME = "PID 冷热台控温（Modbus-RTU）"
CONFIG_DIRNAME = "PID_冷热台"
CONFIG_FILENAME = "config.json"

TCP_HOST = "127.0.0.1"
TCP_PORT = 50010

PSU_BAUD = 9600   # 8N1
TCM_BAUD = 57600  # 8N1
PSU_ADDR_DEFAULT = 0x01
TCM_ADDR_DEFAULT = 0x01

# PSU 寄存器
REG_REMOTE_MODE = 0x0000
REG_V_SET      = 0x0001
REG_A_SET      = 0x0003
REG_OUTPUT     = 0x001B
REG_V_OUT      = 0x001D
REG_A_OUT      = 0x001F
REG_CVCC       = 0x0021

# TCM 寄存器（温度=值/100）
REG_T1   = 0x1018  # int32 BE
REG_TENV = 0x1028

# 控制参数
SAMPLE_HZ = 10.0       # 两设备统一 10Hz 采样
VOLT_LIMIT_MAX = 45.0
CURRENT_LIMIT_A = 1.0
SLEW_RATE_V_PER_S = 2.0
TEMP_LIMIT_CUTOFF = 160.0
TEMP_LIMIT_RESUME = 150.0
CSV_AUTOSAVE = True

PID_KP = 2.0
PID_KI = 0.2
PID_KD = 0.0
INTEGRAL_CLAMP = 200.0

# ====== 循环泵（Modbus-RTU, 0x01/0x03/0x06/0x10，单位见协议）======
PUMP_BAUD = 9600
PUMP_ADDR_DEFAULT = 0x01
PUMP_V_MIN = 13.0
PUMP_V_MAX = 26.0
PUMP_DEADBAND_C = 0.30        # 温度死区，避免冷热两路抢控制
PUMP_SLEW_V_PER_S = 2.5       # 泵电压斜率限幅（V/s）
PUMP_PWM_MIN_DUTY = 5.0


# =========================
# 实用：文档目录
# =========================
def resolve_documents_dir() -> Path:
    home = Path.home()
    candidates = [
        home / "Documents",
        home / "文档",
        home / "OneDrive" / "Documents",
        home / "OneDrive" / "文档",
    ]
    for p in candidates:
        if p.exists():
            return p
    return home

def get_config_path() -> Path:
    root = resolve_documents_dir() / CONFIG_DIRNAME
    root.mkdir(parents=True, exist_ok=True)
    return root / CONFIG_FILENAME

def get_logs_dir() -> Path:
    d = resolve_documents_dir() / CONFIG_DIRNAME / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d

def safe_float(s, default=None):
    """字符串->float，失败返回default；去除空白。"""
    try:
        if s is None:
            return default
        ss = str(s).strip()
        if ss in ("", "-", "+", ".", "+.", "-."):
            return default
        return float(ss)
    except Exception:
        return default


# =========================
# 简易 Modbus RTU（更稳：读够期望长度、重试）
# =========================
class ModbusRTU:
    def __init__(self, port, baudrate, timeout=0.20, retries=2):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.retries = retries
        self.ser = None
        self.lock = threading.Lock()
        self.last_comm_time = 0.0
        self.comm_interval = 0.02  # 20ms：两个设备顺序读也能跑满 10Hz

    def open(self):
        self.ser = serial.Serial(
            self.port, self.baudrate,
            bytesize=8, parity=serial.PARITY_NONE, stopbits=1,
            timeout=self.timeout, write_timeout=self.timeout,
            inter_byte_timeout=self.timeout * 0.5
        )
        self.last_comm_time = time.perf_counter()

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
        self.ser = None

    @staticmethod
    def crc16(data: bytes) -> int:
        crc = 0xFFFF
        for ch in data:
            crc ^= ch
            for _ in range(8):
                if crc & 1:
                    crc = (crc >> 1) ^ 0xA001
                else:
                    crc >>= 1
        return crc

    def _read_exact(self, expected_len: int) -> bytes:
        """逐步读取直到达到期望长度或超时。"""
        buf = bytearray()
        deadline = time.perf_counter() + self.timeout
        while len(buf) < expected_len and time.perf_counter() < deadline:
            chunk = self.ser.read(expected_len - len(buf))
            if chunk:
                buf.extend(chunk)
        return bytes(buf)

    def _txrx(self, req: bytes, expected_len: int) -> bytes:
        last_err = None
        for _ in range(self.retries + 1):
            try:
                with self.lock:
                    # 通信节流：避免背靠背写串口
                    elapsed = time.perf_counter() - self.last_comm_time
                    if elapsed < self.comm_interval:
                        time.sleep(self.comm_interval - elapsed)

                    self.ser.reset_input_buffer()
                    self.ser.write(req); self.ser.flush()
                    self.last_comm_time = time.perf_counter()
                    resp = self._read_exact(expected_len)

                if len(resp) < expected_len:
                    raise IOError(f"Modbus响应过短: {resp.hex(' ')}")

                crc_calc = self.crc16(resp[:-2])
                crc_resp = (resp[-1] << 8) | resp[-2]
                if crc_calc != crc_resp:
                    raise IOError(f"CRC校验失败: 计算 {crc_calc:04X} / 响应 {crc_resp:04X}")
                return resp
            except Exception as e:
                last_err = e
                time.sleep(0.005)
        raise last_err if last_err else IOError("Modbus通信失败")

    def read_holding_registers(self, slave_addr: int, start_addr: int, count: int) -> list:
        payload = struct.pack('>B B H H', slave_addr, 0x03, start_addr, count)
        crc = self.crc16(payload)
        req = payload + struct.pack('<H', crc)
        expected_len = 5 + 2 * count
        resp = self._txrx(req, expected_len)
        if resp[1] != 0x03:
            raise IOError(f"功能码异常: {resp[1]}")
        if resp[2] != 2 * count:
            raise IOError("字节数异常")
        regs = []
        off = 3
        for i in range(count):
            hi = resp[off + 2*i]
            lo = resp[off + 2*i + 1]
            regs.append((hi << 8) | lo)
        return regs

    def write_single_register(self, slave_addr: int, reg_addr: int, value_u16: int):
        payload = struct.pack('>B B H H', slave_addr, 0x06, reg_addr, value_u16 & 0xFFFF)
        crc = self.crc16(payload)
        req = payload + struct.pack('<H', crc)
        _ = self._txrx(req, 8)

    def write_multiple_registers(self, slave_addr: int, start_addr: int, values_u16: list):
        count = len(values_u16)
        header = struct.pack('>B B H H B', slave_addr, 0x10, start_addr, count, count*2)
        data = b''.join(struct.pack('>H', v & 0xFFFF) for v in values_u16)
        payload = header + data
        crc = self.crc16(payload)
        req = payload + struct.pack('<H', crc)
        _ = self._txrx(req, 8)

    def read_float_be(self, slave_addr: int, start_addr: int) -> float:
        regs = self.read_holding_registers(slave_addr, start_addr, 2)
        return struct.unpack('>f', struct.pack('>HH', regs[0], regs[1]))[0]

    def write_float_be(self, slave_addr: int, start_addr: int, val: float):
        hi, lo = struct.unpack('>HH', struct.pack('>f', float(val)))
        self.write_multiple_registers(slave_addr, start_addr, [hi, lo])

    def read_int32_be(self, slave_addr: int, start_addr: int) -> int:
        regs = self.read_holding_registers(slave_addr, start_addr, 2)
        raw = (regs[0] << 16) | regs[1]
        if raw & 0x80000000:
            raw -= 0x100000000
        return raw


# =========================
# 继电器（YK2006, A0 协议）
# =========================
class YK2006Relay:
    """
    简化实现：A0 | addr(1-254) | op | checksum(low8)
    op:
      0x00 关(不反馈)  0x01 开(不反馈)
      0x02 关(反馈)    0x03 开(反馈)
      0x05 查询状态(反馈，返回 0x00/0x01)
    参考：用户提供的 YK2006 串口通信协议(A0版)
    """
    def __init__(self):
        self.ser = None
        self.connected = False
        self.lock = threading.Lock()

    def connect(self, port: str, baud: int = 9600, timeout=0.20):
        self.ser = serial.Serial(
            port=port, baudrate=baud, bytesize=8, parity=serial.PARITY_NONE,
            stopbits=1, timeout=timeout, write_timeout=timeout
        )
        self.connected = True

    def disconnect(self):
        try:
            if self.ser and self.ser.is_open:
                self.ser.close()
        finally:
            self.ser = None
            self.connected = False

    @staticmethod
    def _frame(addr: int, op: int) -> bytes:
        b1 = 0xA0
        addr = max(1, min(254, int(addr)))
        op = op & 0xFF
        chk = (b1 + addr + op) & 0xFF
        return bytes([b1, addr, op, chk])

    def _write(self, data: bytes):
        if not (self.connected and self.ser):
            raise IOError("继电器未连接")
        with self.lock:
            self.ser.reset_input_buffer()
            self.ser.write(data)
            self.ser.flush()

    def _read_exact(self, n: int) -> bytes:
        buf = bytearray()
        deadline = time.perf_counter() + (self.ser.timeout or 0.2)
        while len(buf) < n and time.perf_counter() < deadline:
            ch = self.ser.read(n - len(buf))
            if ch:
                buf.extend(ch)
        return bytes(buf)

    def set(self, channel: int, on: bool, feedback: bool = False) -> bool:
        """返回 True 表示认为已到位；feedback=False 时直接返回 True。"""
        op = 0x03 if (feedback and on) else (0x02 if (feedback and not on) else (0x01 if on else 0x00))
        frm = self._frame(channel, op)
        self._write(frm)
        if not feedback:
            return True
        # 期望 4 字节回包
        resp = self._read_exact(4)
        if len(resp) != 4:
            return False
        # 校验
        if (resp[0] != 0xA0) or ((resp[0] + resp[1] + resp[2]) & 0xFF) != resp[3]:
            return False
        # resp[2]: 0x00 关 / 0x01 开
        return bool(resp[2]) == bool(on)

    def query(self, channel: int) -> bool | None:
        frm = self._frame(channel, 0x05)
        self._write(frm)
        resp = self._read_exact(4)
        if len(resp) != 4:
            return None
        if (resp[0] != 0xA0) or ((resp[0] + resp[1] + resp[2]) & 0xFF) != resp[3]:
            return None
        return bool(resp[2])

# =========================
# 设备封装
# =========================
class PowerSupply:
    def __init__(self):
        self.mb = None
        self.addr = PSU_ADDR_DEFAULT
        self.connected = False
        self.last_check = 0.0
        self.check_interval = 2.0

    def connect(self, port: str, baud: int = PSU_BAUD, addr: int = PSU_ADDR_DEFAULT):
        self.addr = addr
        self.mb = ModbusRTU(port, baudrate=baud, timeout=0.10)  # 原 0.25 -> 0.10
        self.mb.open()
        self.connected = True
        self.last_check = time.perf_counter()
        self.set_remote(True)
        self.set_current_limit(CURRENT_LIMIT_A)

    def disconnect(self):
        try:
            if self.connected and self.mb:
                try:
                    self.set_output(False)
                except Exception:
                    pass
                self.mb.close()
        finally:
            self.mb = None
            self.connected = False

    def _ensure(self):
        if not self.connected or not self.mb:
            raise IOError("电源未连接")
        now = time.perf_counter()
        if now - self.last_check >= self.check_interval:
            _ = self.mb.read_holding_registers(self.addr, REG_REMOTE_MODE, 1)
            self.last_check = now

    def set_remote(self, remote: bool):
        self._ensure()
        self.mb.write_single_register(self.addr, REG_REMOTE_MODE, 1 if remote else 0)

    def set_output(self, on: bool):
        self._ensure()
        self.mb.write_single_register(self.addr, REG_OUTPUT, 1 if on else 0)

    def set_voltage(self, v: float):
        self._ensure()
        self.mb.write_float_be(self.addr, REG_V_SET, v)

    def set_current_limit(self, a: float):
        self._ensure()
        self.mb.write_float_be(self.addr, REG_A_SET, a)

    def read_meas(self):
        self._ensure()
        regs = self.mb.read_holding_registers(self.addr, REG_V_OUT, 5)
        v = struct.unpack('>f', struct.pack('>HH', regs[0], regs[1]))[0]
        a = struct.unpack('>f', struct.pack('>HH', regs[2], regs[3]))[0]
        cvcc = regs[4] & 0xFFFF
        return v, a, cvcc

class CoolingPump:
    """
    循环泵（协议：0x03/0x06/0x10；寄存器/单位参见用户给定规范）
    V-SET: 0x0000 (0.01 V)  I-SET: 0x0001 (0.001 A)
    VOUT : 0x0002 (0.01 V)  IOUT : 0x0003 (0.001 A)  POWER: 0x0004 (0.01 W)
    ONOFF: 0x0012 (0/1)
    """
    REG_V_SET = 0x0000
    REG_I_SET = 0x0001
    REG_V_OUT = 0x0002
    REG_I_OUT = 0x0003
    REG_P_OUT = 0x0004
    REG_ONOFF = 0x0012

    def __init__(self):
        self.mb = None
        self.addr = PUMP_ADDR_DEFAULT
        self.connected = False
        self.last_check = 0.0
        self.check_interval = 2.0
        self.last_vset_cmd = 0.0

    def connect(self, port: str, baud: int = PUMP_BAUD, addr: int = PUMP_ADDR_DEFAULT):
        self.addr = addr
        self.mb = ModbusRTU(port, baudrate=baud, timeout=0.10)  # 原 0.25 -> 0.10
        self.mb.open()
        self.connected = True
        self.last_check = time.perf_counter()

    def disconnect(self):
        try:
            if self.mb:
                try:
                    self.off()
                except Exception:
                    pass
                self.mb.close()
        finally:
            self.mb = None
            self.connected = False

    def _ensure(self):
        if not self.connected or not self.mb:
            raise IOError("循环泵未连接")
        now = time.perf_counter()
        if now - self.last_check >= self.check_interval:
            # 读一个寄存器当保活
            _ = self.mb.read_holding_registers(self.addr, self.REG_V_OUT, 1)
            self.last_check = now

    # -------- 控制 --------
    def on(self):
        self._ensure()
        self.mb.write_single_register(self.addr, self.REG_ONOFF, 1)

    def off(self):
        self._ensure()
        self.mb.write_single_register(self.addr, self.REG_ONOFF, 0)

    def set_voltage(self, v: float):
        """写单寄存器 V-SET，单位 0.01 V；不在这里强制 13~26，外部做策略更灵活。"""
        self._ensure()
        v_enc = max(0, int(round(float(v) * 100.0))) & 0xFFFF
        self.mb.write_single_register(self.addr, self.REG_V_SET, v_enc)
        self.last_vset_cmd = float(v)

    def set_v_i(self, v: float, i: float):
        """一次写入 V+I（功能码 0x10）。"""
        self._ensure()
        v_enc = max(0, int(round(float(v) * 100.0))) & 0xFFFF
        i_enc = max(0, int(round(float(i) * 1000.0))) & 0xFFFF
        self.mb.write_multiple_registers(self.addr, self.REG_V_SET, [v_enc, i_enc])
        self.last_vset_cmd = float(v)

    # -------- 采集 --------
    def read_vi(self):
        self._ensure()
        regs = self.mb.read_holding_registers(self.addr, self.REG_V_OUT, 2)
        v = regs[0] * 0.01
        a = regs[1] * 0.001
        return v, a

    def read_vip(self):
        self._ensure()
        regs = self.mb.read_holding_registers(self.addr, self.REG_V_OUT, 3)
        v = regs[0] * 0.01
        a = regs[1] * 0.001
        p = regs[2] * 0.01
        return v, a, p

    # 兼容旧调用：返回 (V, A, W)
    def read_meas(self):
        return self.read_vip()

    # 兼容旧调用：set_output(True/False) -> on/off
    def set_output(self, on: bool):
        if on:
            self.on()
        else:
            self.off()


class TCMTemperature:
    def __init__(self):
        self.mb = None
        self.addr = TCM_ADDR_DEFAULT
        self.connected = False
        self.last_check = 0.0
        self.check_interval = 2.0

    def connect(self, port: str, baud: int = TCM_BAUD, addr: int = TCM_ADDR_DEFAULT):
        self.addr = addr
        self.mb = ModbusRTU(port, baudrate=baud, timeout=0.10)  # 原 0.25 -> 0.10
        self.mb.open()
        self.connected = True
        self.last_check = time.perf_counter()

    def disconnect(self):
        try:
            if self.mb:
                self.mb.close()
        finally:
            self.mb = None
            self.connected = False

    def _ensure(self):
        if not self.connected or not self.mb:
            raise IOError("温度仪未连接")
        now = time.perf_counter()
        if now - self.last_check >= self.check_interval:
            _ = self.mb.read_holding_registers(self.addr, REG_TENV, 2)
            self.last_check = now

    def read_T1(self) -> float:
        self._ensure()
        raw = self.mb.read_int32_be(self.addr, REG_T1)
        return raw / 100.0

    def read_Tenv(self) -> float:
        self._ensure()
        raw = self.mb.read_int32_be(self.addr, REG_TENV)
        return raw / 100.0


# =========================
# PID 控制器
# =========================
class PID:
    def __init__(self, kp, ki, kd, u_min=0.0, u_max=VOLT_LIMIT_MAX, integral_clamp=INTEGRAL_CLAMP):
        self.kp = kp; self.ki = ki; self.kd = kd
        self.u_min = u_min; self.u_max = u_max
        self.integral_clamp = integral_clamp
        self.reset()

    def reset(self):
        self.integral = 0.0
        self.prev_meas = None

    def step(self, setpoint, meas, dt, cc_mode=False, at_upper_limit=False):
        e = setpoint - meas
        # 抗饱和：上限&CC时且误差为正，不继续积分
        if not ((at_upper_limit or cc_mode) and e > 0):
            self.integral += e * dt
            self.integral = max(-self.integral_clamp, min(self.integral, self.integral_clamp))
        d_meas = 0.0 if (self.prev_meas is None) else (self.prev_meas - meas) / max(1e-6, dt)
        self.prev_meas = meas
        u = self.kp * e + self.ki * self.integral + self.kd * d_meas
        return max(self.u_min, min(self.u_max, u)), e

# ========= 新增：轻量异步采样器 =========
# 改进的异步采样器，避免线程阻塞
# ========= 轻量异步采样器（修正版：支持 token，同步接口 latest()） =========
class _AsyncSampler:
    def __init__(self, name, read_fn):
        self.name = name
        self.read_fn = read_fn
        self._lock = threading.Lock()
        self._running = False
        self._req_event = threading.Event()   # 有请求就置位
        self._data_ready = threading.Event()  # 有新数据就置位

        self._last_val = None
        self._last_ts = 0.0
        self._last_ok = False

        self._pending_tok = 0   # 本次请求的 token
        self._done_tok = 0      # 已完成的 token

        self._thread = None

    def start(self):
        if self._running:
            return
        self._running = True
        self._req_event.clear()
        self._data_ready.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        self._req_event.set()  # 唤醒线程以便退出
        if self._thread:
            self._thread.join(timeout=1.0)
        self._thread = None

    def request(self, token=None):
        """请求一次采样；可携带 token 用于对齐帧。"""
        with self._lock:
            self._pending_tok = 0 if token is None else int(token)
            self._data_ready.clear()   # 新请求开始前清空“完成”标志
            self._req_event.set()      # 唤醒采样线程

    def latest(self):
        """返回 (value, timestamp, ok, done_token)。与你主循环一致。"""
        with self._lock:
            return self._last_val, self._last_ts, self._last_ok, self._done_tok

    def get_latest(self, timeout=0.1):
        """可选：阻塞等待到新数据，用于需要阻塞的场景。"""
        if self._data_ready.wait(timeout=timeout):
            with self._lock:
                return self._last_val, self._last_ts, self._last_ok
        return None, 0.0, False

    def _run(self):
        while True:
            self._req_event.wait()  # 等待一次请求
            if not self._running:
                break
            # 拿到这次要处理的 token，并清除请求位
            with self._lock:
                tok = self._pending_tok
                self._req_event.clear()

            # 执行读取
            val, ok = None, False
            try:
                val = self.read_fn()
                ok = True
            except Exception:
                ok = False

            ts = time.perf_counter()
            # 更新共享结果
            with self._lock:
                self._last_val = val
                self._last_ts = ts
                self._last_ok = ok
                self._done_tok = tok
                self._data_ready.set()


# ========= 新增：按设备包装一次读取（与原 read_* 完全兼容） =========
def _read_tcm_once(self):
    # 返回 float 温度；保持与原来语义一致
    return self.tcm.read_T1()

def _read_psu_once(self):
    # 返回 (V_out, A_out, CV/CC标志)
    return self.psu.read_meas()

def _read_pump_once(self):
    # 返回 (V, I, P)
    return self.pump.read_meas()


# ========= 新增：启动/停止三个采样器 =========
def _start_async_samplers(self):
    """在 start_acquisition 里调用。"""
    # 温度
    self._sampler_tcm = _AsyncSampler("tcm", lambda: _read_tcm_once(self))
    self._sampler_tcm.start()
    # 电源
    self._sampler_psu = _AsyncSampler("psu", lambda: _read_psu_once(self))
    self._sampler_psu.start()
    # 循环泵（可能没连）
    if hasattr(self, "_stop_pump_sampler"):
        self._stop_pump_sampler()
    if hasattr(self, "_ensure_pump_sampler_running"):
        self._ensure_pump_sampler_running()
    elif hasattr(self, "pump") and getattr(self.pump, "connected", False):
        self._sampler_pump = _AsyncSampler("pump", lambda: _read_pump_once(self))
        self._sampler_pump.start()
    else:
        self._sampler_pump = None

def _stop_async_samplers(self):
    for s in (getattr(self, "_sampler_tcm", None),
              getattr(self, "_sampler_psu", None),
              getattr(self, "_sampler_pump", None)):
        try:
            if s: s.stop()
        except Exception:
            pass
    self._sampler_tcm = self._sampler_psu = self._sampler_pump = None


# =========================
# 主应用（含显示三位小数、线性程序、拟合）
# =========================
class App:
    def __init__(self, root):
        self.root = root
        self.style = ttk.Style("cosmo")
        self.root.title(APP_NAME)
        self.root.geometry("1320x860")

        # 状态对象
        self.psu = PowerSupply()
        self.tcm = TCMTemperature()
        self.acq_running = False
        self.control_enabled = False

        # =========================
        # Tkinter 变量初始化
        # =========================
        # PID 拟合功能参数
        self.fit_window_s = tk.DoubleVar(value=60.0)  # 拟合窗口 (秒)
        self.fit_slope = tk.DoubleVar(value=0.0)  # 拟合斜率 (°C/s)

        # 控制参数
        self.target_temp = tk.DoubleVar(value=25.0)
        self.kp = tk.DoubleVar(value=PID_KP)
        self.ki = tk.DoubleVar(value=PID_KI)
        self.kd = tk.DoubleVar(value=PID_KD)

        # 串口配置相关
        self.psu_port = tk.StringVar()
        self.tcm_port = tk.StringVar()
        self.psu_baud = tk.IntVar(value=PSU_BAUD)
        self.tcm_baud = tk.IntVar(value=TCM_BAUD)
        self.addr_psu = tk.IntVar(value=PSU_ADDR_DEFAULT)
        self.addr_tcm = tk.IntVar(value=TCM_ADDR_DEFAULT)

        # 限制 & 采样
        self.v_limit = tk.DoubleVar(value=VOLT_LIMIT_MAX)
        self.slew_vps = tk.DoubleVar(value=SLEW_RATE_V_PER_S)
        self.sample_hz = tk.DoubleVar(value=SAMPLE_HZ)
        self.temp_tau = tk.DoubleVar(value=1.0)  # 温度滤波 τ

        # 精度统计
        self.acc_window_s = tk.DoubleVar(value=30.0)  # 统计窗口 (秒)
        self.acc_tol = tk.DoubleVar(value=0.5)  # 容差 (±°C)

        # 实时读数
        self.cur_temp = tk.DoubleVar(value=0.0)
        self.cur_vout = tk.DoubleVar(value=0.0)
        self.cur_aout = tk.DoubleVar(value=0.0)
        self.cur_pout = tk.DoubleVar(value=0.0)
        self.cvcc_state = tk.StringVar(value="—")
        self.status_text = tk.StringVar(value="未连接")
        self.psu_status = tk.StringVar(value="未连接")
        self.tcm_status = tk.StringVar(value="未连接")
        self.pump_status = tk.StringVar(value="未连接")

        # —— CSV 会话式记录 ——
        self.csv_dir = tk.StringVar(value=str(get_logs_dir()))  # 可配置目录（默认 logs）
        self.csv_session_op = None  # 当前会话名："启动PID" / "开始线性程序"
        self.csv_session_path = None  # 当前会话文件路径
        self._last_click_ts = {}  # 防抖用的时间戳字典

        # 实时精度显示变量
        self.cur_err = tk.DoubleVar(value=0.0)  # ε(t)
        self.mae_win = tk.DoubleVar(value=0.0)  # MAE
        self.rmse_win = tk.DoubleVar(value=0.0)  # RMSE
        self.hit_ratio = tk.DoubleVar(value=0.0)  # 命中率 %
        self.acc_info = tk.StringVar(value="窗=30 s，容差=±0.5 °C")

        # 界面刷新
        self.plot_window_s = tk.DoubleVar(value=120.0)
        self.ui_refresh_ms = tk.IntVar(value=100)

        # ========== 线程通讯锁 ==========
        self.params_lock = threading.Lock()
        self.rt_lock = threading.Lock()
        self.params = {
            "sample_hz": self.sample_hz.get(),
            "v_limit": self.v_limit.get(),
            "slew_vps": self.slew_vps.get(),
            "kp": self.kp.get(),
            "ki": self.ki.get(),
            "kd": self.kd.get(),
            "target": self.target_temp.get(),
            "temp_tau": self.temp_tau.get(),
        }
        self.rt = {"temp": float('nan'), "v_out": 0.0, "a_out": 0.0, "p_out": 0.0,
                   "cvcc": 0, "err": float('nan'), "note": ""}

        self.csv_file = None
        self.csv_writer = None

        self.thread = None
        self.stop_event = threading.Event()

        self.pid = PID(self.params["kp"], self.params["ki"], self.params["kd"],
                       u_min=0.0, u_max=self.params["v_limit"])
        self.last_vset = 0.0
        self.overtemp_shutdown = False

        # 缓冲与绘图
        self.history_seconds = 15 * 60
        maxlen = int(self.history_seconds * self.params["sample_hz"])
        self.history_lock = threading.Lock()
        self.t_buf = deque(maxlen=maxlen)
        self.pump_v_buf = deque(maxlen=maxlen)
        self.pump_i_buf = deque(maxlen=maxlen)
        self.pump_p_buf = deque(maxlen=maxlen)
        self.temp_buf = deque(maxlen=maxlen)
        self.power_buf = deque(maxlen=maxlen)
        self.err_buf = deque(maxlen=maxlen)
        self.target_buf = deque(maxlen=maxlen)  # 目标温度历史

        self.plot_window_s = tk.StringVar(value="120.000")
        self.ui_refresh_ms = tk.IntVar(value=100)
        self._frame_count = 0
        self._last_plot_update = 0.0
        self._acq_t0 = None  # perf_counter 起点
        self._status_lock = threading.Lock()
        self._status_msg = ""

        # —— 线性程序（升/降温）控制 ——
        self.ramp_active = False
        self.ramp_thread = None
        self.ramp_stop_evt = threading.Event()
        self.ramp_start = tk.StringVar(value="25.000")
        self.ramp_end = tk.StringVar(value="50.000")
        self.ramp_rate = tk.StringVar(value="1.000")  # °C/min
        self.ramp_hold_min = tk.StringVar(value="0.000")

        # ★ 新增：循环变温开关
        self.ramp_cycle_enable = tk.BooleanVar(value=False)

        # —— 判稳 & CSV 起止阈值（务必在 _build_ui() 之前定义）——
        self.mae_stable_thr = tk.DoubleVar(value=0.20)  # MAE 判稳阈值（°C）
        self.slope_start_thr_cpm = tk.DoubleVar(value=0.30)  # CSV 开始阈：|斜率|≥此值（°C/min）
        self.slope_stop_thr_cpm = tk.DoubleVar(value=0.08)  # CSV 停止阈：|斜率|≤此值（°C/min）

        # —— 模拟点击设置（用 StringVar，避免空值崩溃）——
        self.auto_click_enable = tk.BooleanVar(value=False)
        self.auto_click_x = tk.StringVar(value="0")  # ← StringVar
        self.auto_click_y = tk.StringVar(value="0")  # ← StringVar
        self.auto_click_button = tk.StringVar(value="left")
        self.auto_click_double = tk.BooleanVar(value=False)
        self.auto_click_delay_ms = tk.StringVar(value="0")  # ← StringVar

        # —— 坐标拾取运行态 ——
        self._pick_running = False
        self.pick_preview = tk.StringVar(value="X=—, Y=—")  # 实时预览
        self._kb_hotkey_id = None  # keyboard 热键句柄

        # ==== 循环泵对象 / PID（冷却） ====
        self.pump = CoolingPump()
        self.pump_status = tk.StringVar(value="未连接")
        self.pump_port = tk.StringVar()
        self.pump_baud = tk.IntVar(value=PUMP_BAUD)
        self.addr_pump = tk.IntVar(value=PUMP_ADDR_DEFAULT)

        # 冷却 PID 以及策略参数
        self.kp_cool = tk.DoubleVar(value=self.kp.get())       # 默认与加热相同
        self.ki_cool = tk.DoubleVar(value=max(0.0, self.ki.get() * 0.5))  # 积分稍小，避免抢控制
        self.kd_cool = tk.DoubleVar(value=self.kd.get())

        self.pump_vmin = tk.DoubleVar(value=PUMP_V_MIN)
        self.pump_vmax = tk.DoubleVar(value=PUMP_V_MAX)
        self.pump_deadband = tk.DoubleVar(value=PUMP_DEADBAND_C)
        self.pump_slew_vps = tk.DoubleVar(value=PUMP_SLEW_V_PER_S)

        # 泵实时读数 / 命令
        self.cur_pump_v = tk.DoubleVar(value=0.0)
        self.cur_pump_a = tk.DoubleVar(value=0.0)
        self.cur_pump_p = tk.DoubleVar(value=0.0)
        self.cur_pump_vset = tk.DoubleVar(value=0.0)

        # 冷却 PID 实例（输出 = 相对增量 -> [0, vmax-vmin]）
        self.pid_cool = PID(self.kp_cool.get(), self.ki_cool.get(), self.kd_cool.get(),
                            u_min=0.0, u_max=self.pump_vmax.get() - self.pump_vmin.get())
        self.pump_last_vset = 0.0
        self._pump_on = False

        # ===== PWM 相关变量（先定义，后使用）=====
        # 说明：
        # - pump_pwm_v_on = PWM 开(高)电压。<=0 表示“跟随泵最小电压(pump_vmin)”
        # - pump_pwm_v_off = PWM 关(低)电压
        # —— PWM 相关 Tk 变量（供 UI 与 _acq_loop 共用）——
        self.pump_pwm_enable = tk.BooleanVar(value=False)  # 手动 PWM 开关
        self.pump_pwm_freq = tk.DoubleVar(value=10.0)  # PWM 频率 Hz
        self.pump_pwm_v_on = tk.DoubleVar(value=13.0)  # PWM 开(高)电压
        self.pump_pwm_v_off = tk.DoubleVar(value=0.0)  # PWM 关(低)电压
        self.pump_pwm_min_on_ms = tk.IntVar(value=300)

        self.pump_pwm_auto_on_hotv = tk.BooleanVar(value=True)  # 自动：加热电压高时启用
        self.pump_pwm_hotv_thr = tk.DoubleVar(value=40.0)  # 加热电压阈值（V）

        self.pump_pwm_auto_on_ramp_up = tk.BooleanVar(value=True)  # 自动：升温线性段启用
        self.pump_pwm_auto_duty = tk.DoubleVar(value=0.50)  # 自动触发时等效占空比 (0~1)
        self.pump_pwm_err_thr_C = tk.DoubleVar(value=0.15)  # 仍需升温的温差阈值（°C）

        self.pump_pwm_always_on_temp = tk.BooleanVar(value=False)  # 高温恒启 PWM
        self.pump_pwm_always_temp_thr = tk.DoubleVar(value=80.0)  # 高温阈值（°C）

        # 可选：允许“必要时一直保持关”的开关（如果你在 _acq_loop 里加了逻辑）
        self.pump_pwm_force_off = tk.BooleanVar(value=False)
        # —— 温度门控：低温强制继电器闭合；高温强制进入 PWM ——
        self.pwm_low_keep_on = tk.DoubleVar(value=5.0)  # 低于此温度，强制继电器 ON（循环）
        self.pwm_high_enable_pwm = tk.DoubleVar(value=15.0)  # 高于此温度，强制进入 PWM

        # ==== 继电器（YK2006） ====
        self.relay = YK2006Relay()
        self.relay_status = tk.StringVar(value="未连接")
        self.relay_port = tk.StringVar()
        self.relay_baud = tk.IntVar(value=9600)
        self.relay_chan = tk.IntVar(value=1)  # 默认第1路
        self._relay_is_on = False             # 运行态
        self._relay_epoch = None              # 周期相位起点
        self._relay_last_switch_ts = 0.0      # 最近一次切换时间（用于最短开时长）

        # —— 继电器 PWM 显示变量 ——
        self.relay_pwm_duty = tk.StringVar(value="—")  # 占空比 %
        self.relay_pwm_state = tk.StringVar(value="—")  # ON/OFF

        # 泵实时读数 / 命令
        self.cur_pump_v = tk.DoubleVar(value=0.0)
        self.cur_pump_a = tk.DoubleVar(value=0.0)
        self.cur_pump_p = tk.DoubleVar(value=0.0)

        # ★ V_set 显示（UI用）：正常模式=实际设定；PWM模式=等效电压（duty*V_on）
        self.cur_pump_vset = tk.DoubleVar(value=0.0)

        # ★ 占空比显示
        self.pwm_duty = tk.DoubleVar(value=0.0)  # 0~100（数值）
        self.pwm_duty_str = tk.StringVar(value="—")  # 文本 "xx.x%"
        # —— 自动 PWM 入口 & 高温阈值 & 占空比下限（UI 会用到，务必在 _build_ui 前定义）——
        self.pump_pwm_auto_enable = tk.BooleanVar(value=True)  # “启用自动 PWM”总开关（UI勾选）
        self.pump_pwm_hi_temp_thr = tk.DoubleVar(value=35.0)  # 高温阈值：超过即转入 PWM 模式
        self.pump_pwm_min_duty = tk.DoubleVar(value=PUMP_PWM_MIN_DUTY)  # 占空比地板（%）

        self.cycle_times = deque(maxlen=100)
        self.gui_update_times = deque(maxlen=100)
        self.last_cycle_time = time.perf_counter()

        # UI
        self._build_ui()
        self._sync_click_vars_to_entries()
        self._load_config()
        self._refresh_ports()
        self._schedule_gui_update()

        # 启动 TCP 控制服务
        self.tcp_server = JSONLineServer(TCP_HOST, TCP_PORT, self._handle_tcp_command, name="temp-controller")
        try:
            self.tcp_server.start()
        except OSError as exc:
            self.tcp_server = None
            self._set_status(f"TCP服务启动失败: {exc}")


    # ---------- UI ----------
    def _build_ui(self):
        def create_status_indicator(parent, text, var):
            frame = ttk.Frame(parent)
            frame.pack(fill=tk.X, pady=(0, 4))
            ttk.Label(frame, text=text, width=8).pack(side=tk.LEFT)
            cv = tk.Canvas(frame, width=16, height=16, highlightthickness=0)
            cv.pack(side=tk.LEFT, padx=(0, 5))
            ttk.Label(frame, textvariable=var, font=("Arial", 9)).pack(side=tk.LEFT)
            return cv

        content = ttk.Frame(self.root)
        content.pack(fill=tk.BOTH, expand=True)

        # 左侧滚动面板（保持你原来的逻辑）
        left_container = ttk.Frame(content, width=720)
        left_container.pack(side=tk.LEFT, fill=tk.Y)
        left_container.pack_propagate(False)

        self.left_canvas = tk.Canvas(left_container, highlightthickness=0)
        left_scrollbar = ttk.Scrollbar(left_container, orient="vertical", command=self.left_canvas.yview)
        self.left_canvas.configure(yscrollcommand=left_scrollbar.set)
        self.left_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        left_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        left = ttk.Frame(self.left_canvas, padding=10)
        self.left_window_id = self.left_canvas.create_window((0, 0), window=left, anchor="nw")

        def _on_left_frame_configure(event):
            self.left_canvas.configure(scrollregion=self.left_canvas.bbox("all"))
        left.bind("<Configure>", _on_left_frame_configure)

        def _on_canvas_configure(event):
            self.left_canvas.itemconfigure(self.left_window_id, width=event.width)
        self.left_canvas.bind("<Configure>", _on_canvas_configure)

        # 鼠标滚轮绑定（略）——保留你原来的三函数绑定
        def _mw_windows(event):
            self.left_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        def _mw_linux_up(event):
            self.left_canvas.yview_scroll(-3, "units")
        def _mw_linux_down(event):
            self.left_canvas.yview_scroll(3, "units")
        def _bind_wheel(_):
            self.left_canvas.bind_all("<MouseWheel>", _mw_windows)
            self.left_canvas.bind_all("<Button-4>", _mw_linux_up)
            self.left_canvas.bind_all("<Button-5>", _mw_linux_down)
        def _unbind_wheel(_):
            self.left_canvas.unbind_all("<MouseWheel>")
            self.left_canvas.unbind_all("<Button-4>")
            self.left_canvas.unbind_all("<Button-5>")
        left.bind("<Enter>", _bind_wheel)
        left.bind("<Leave>", _unbind_wheel)

        # 右侧：并列两块图表（左：温度/加热功率；右：泵电压/功率）
        right = ttk.Frame(content, padding=8)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        charts = ttk.Frame(right)
        charts.pack(fill=tk.BOTH, expand=True)
        pane_left = ttk.Frame(charts, padding=(0,0,8,0))
        pane_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        pane_right = ttk.Frame(charts)
        pane_right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # 指示灯
        status_frame = ttk.Frame(left, padding=(0,0,0,10)); status_frame.pack(fill=tk.X)
        self.psu_indicator  = create_status_indicator(status_frame, "电源:",   self.psu_status)
        self.tcm_indicator  = create_status_indicator(status_frame, "温度仪:", self.tcm_status)
        self.pump_indicator = create_status_indicator(status_frame, "循环泵:", self.pump_status)
        self.relay_indicator = create_status_indicator(status_frame, "继电器:", self.relay_status)

        # 目标/控制（保留）
        goal = ttk.Labelframe(left, text="目标与控制", padding=10); goal.pack(fill=tk.X, pady=(0,8))
        row1 = ttk.Frame(goal); row1.pack(fill=tk.X)
        ttk.Label(row1, text="目标温度 (°C)").pack(side=tk.LEFT)
        ttk.Entry(row1, textvariable=self.target_temp, width=10).pack(side=tk.LEFT, padx=6)
        ttk.Button(row1, text="应用目标", bootstyle=PRIMARY, command=self._apply_target).pack(side=tk.LEFT)
        row2 = ttk.Frame(goal); row2.pack(fill=tk.X, pady=(8,0))
        self.btn_pid_on  = ttk.Button(row2, text="启动 PID", bootstyle=SUCCESS, command=self.enable_pid)
        self.btn_pid_on.pack(side=tk.LEFT, padx=(0,6))
        self.btn_pid_off = ttk.Button(row2, text="停止 PID", bootstyle=DANGER,  command=self.disable_pid)
        self.btn_pid_off.pack(side=tk.LEFT)
        ttk.Label(goal, textvariable=self.status_text, bootstyle=INFO).pack(anchor="w", pady=(8,0))

        # 线性程序（保留）
        ramp = ttk.Labelframe(left, text="程序升/降温（线性）", padding=10); ramp.pack(fill=tk.X, pady=8)
        r1 = ttk.Frame(ramp); r1.pack(fill=tk.X, pady=2)
        ttk.Label(r1, text="起点(°C)").pack(side=tk.LEFT); ttk.Entry(r1, textvariable=self.ramp_start, width=8).pack(side=tk.LEFT, padx=(4,10))
        ttk.Label(r1, text="终点(°C)").pack(side=tk.LEFT); ttk.Entry(r1, textvariable=self.ramp_end, width=8).pack(side=tk.LEFT, padx=(4,10))
        ttk.Label(r1, text="速率(°C/min)").pack(side=tk.LEFT); ttk.Entry(r1, textvariable=self.ramp_rate, width=8).pack(side=tk.LEFT, padx=(4,10))
        r2 = ttk.Frame(ramp);
        r2.pack(fill=tk.X, pady=2)
        ttk.Label(r2, text="末端保温(min)").pack(side=tk.LEFT)
        ttk.Entry(r2, textvariable=self.ramp_hold_min, width=8).pack(side=tk.LEFT, padx=(4, 10))

        # ★ 新增：循环变温开关
        ttk.Checkbutton(r2, text="循环变温", variable=self.ramp_cycle_enable) \
            .pack(side=tk.LEFT, padx=(6, 12))

        ttk.Button(r2, text="开始线性程序", bootstyle=SUCCESS, command=self._start_ramp) \
            .pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(r2, text="停止线性程序", bootstyle=WARNING, command=self._stop_ramp) \
            .pack(side=tk.LEFT)

        # 连接（新增循环泵一行）
        conn = ttk.Labelframe(left, text="连接", padding=10); conn.pack(fill=tk.X, pady=8)
        psu_f = ttk.Frame(conn); psu_f.pack(fill=tk.X, pady=(0,4))
        ttk.Label(psu_f, text="电源端口").pack(side=tk.LEFT)
        self.combo_psu = ttk.Combobox(psu_f, textvariable=self.psu_port, width=12)
        self.combo_psu.pack(side=tk.LEFT, padx=4)
        ttk.Label(psu_f, text="地址").pack(side=tk.LEFT, padx=(8,2))
        psu_addr_spin = ttk.Spinbox(psu_f, textvariable=self.addr_psu, from_=1, to=247, width=5)
        psu_addr_spin.pack(side=tk.LEFT)
        ttk.Label(psu_f, text="波特率").pack(side=tk.LEFT, padx=(8,2))
        psu_baud_combo = ttk.Combobox(psu_f, textvariable=self.psu_baud, width=8,
                                      values=[1200, 2400, 4800, 9600, 19200, 115200])
        psu_baud_combo.pack(side=tk.LEFT)
        self._disable_widget_scroll(psu_addr_spin, psu_baud_combo)
        ttk.Button(psu_f, text="连接电源", bootstyle=PRIMARY, command=self._connect_psu).pack(side=tk.LEFT, padx=6)
        ttk.Button(psu_f, text="断开", command=self._disconnect_psu).pack(side=tk.LEFT)
        ttk.Label(psu_f, textvariable=self.psu_status, width=10).pack(side=tk.LEFT, padx=6)

        tcm_f = ttk.Frame(conn); tcm_f.pack(fill=tk.X, pady=(0,4))
        ttk.Label(tcm_f, text="温度端口").pack(side=tk.LEFT)
        self.combo_tcm = ttk.Combobox(tcm_f, textvariable=self.tcm_port, width=12)
        self.combo_tcm.pack(side=tk.LEFT, padx=4)
        ttk.Label(tcm_f, text="地址").pack(side=tk.LEFT, padx=(8,2))
        tcm_addr_spin = ttk.Spinbox(tcm_f, textvariable=self.addr_tcm, from_=1, to=247, width=5)
        tcm_addr_spin.pack(side=tk.LEFT)
        ttk.Label(tcm_f, text="波特率").pack(side=tk.LEFT, padx=(8,2))
        tcm_baud_combo = ttk.Combobox(tcm_f, textvariable=self.tcm_baud, width=8,
                                      values=[9600, 19200, 38400, 57600, 115200])
        tcm_baud_combo.pack(side=tk.LEFT)
        self._disable_widget_scroll(tcm_addr_spin, tcm_baud_combo)
        ttk.Button(tcm_f, text="连接温度仪", bootstyle=PRIMARY, command=self._connect_tcm).pack(side=tk.LEFT, padx=6)
        ttk.Button(tcm_f, text="断开", command=self._disconnect_tcm).pack(side=tk.LEFT)
        ttk.Label(tcm_f, textvariable=self.tcm_status, width=10).pack(side=tk.LEFT, padx=6)

        pump_f = ttk.Frame(conn); pump_f.pack(fill=tk.X, pady=(0,4))
        ttk.Label(pump_f, text="循环泵端口").pack(side=tk.LEFT)
        self.combo_pump = ttk.Combobox(pump_f, textvariable=self.pump_port, width=12)
        self.combo_pump.pack(side=tk.LEFT, padx=4)
        ttk.Label(pump_f, text="地址").pack(side=tk.LEFT, padx=(8,2))
        pump_addr_spin = ttk.Spinbox(pump_f, textvariable=self.addr_pump, from_=1, to=247, width=5)
        pump_addr_spin.pack(side=tk.LEFT)
        ttk.Label(pump_f, text="波特率").pack(side=tk.LEFT, padx=(8,2))
        pump_baud_combo = ttk.Combobox(pump_f, textvariable=self.pump_baud, width=8,
                                       values=[1200, 2400, 4800, 9600, 19200, 115200])
        pump_baud_combo.pack(side=tk.LEFT)
        self._disable_widget_scroll(pump_addr_spin, pump_baud_combo)
        ttk.Button(pump_f, text="连接循环泵", bootstyle=PRIMARY, command=self._connect_pump).pack(side=tk.LEFT, padx=6)
        ttk.Button(pump_f, text="断开", command=self._disconnect_pump).pack(side=tk.LEFT)
        ttk.Label(pump_f, textvariable=self.pump_status, width=10).pack(side=tk.LEFT, padx=6)

        relay_f = ttk.Frame(conn); relay_f.pack(fill=tk.X, pady=(0,4))
        ttk.Label(relay_f, text="继电器端口").pack(side=tk.LEFT)
        self.combo_relay = ttk.Combobox(relay_f, textvariable=self.relay_port, width=12)
        self.combo_relay.pack(side=tk.LEFT, padx=4)
        ttk.Label(relay_f, text="通道").pack(side=tk.LEFT, padx=(8,2))
        relay_chan_spin = ttk.Spinbox(relay_f, textvariable=self.relay_chan, from_=1, to=254, width=5)
        relay_chan_spin.pack(side=tk.LEFT)
        ttk.Label(relay_f, text="波特率").pack(side=tk.LEFT, padx=(8,2))
        relay_baud_combo = ttk.Combobox(
            relay_f,
            textvariable=self.relay_baud,
            width=8,
            values=[1200, 2400, 4800, 9600, 19200, 38400, 57600, 115200],
        )
        relay_baud_combo.pack(side=tk.LEFT)
        self._disable_widget_scroll(relay_chan_spin, relay_baud_combo)
        ttk.Button(relay_f, text="连接继电器", bootstyle=PRIMARY, command=self._connect_relay).pack(side=tk.LEFT, padx=6)
        ttk.Button(relay_f, text="断开", command=self._disconnect_relay).pack(side=tk.LEFT)
        ttk.Label(relay_f, textvariable=self.relay_status, width=10).pack(side=tk.LEFT, padx=6)

        btn_row = ttk.Frame(conn)
        btn_row.pack(fill=tk.X, pady=(6, 0))
        ttk.Button(btn_row, text="刷新串口列表", command=self._refresh_ports).pack(side=tk.RIGHT)
        ttk.Button(btn_row, text="自动连接", bootstyle=SUCCESS, command=self._auto_connect_all).pack(
            side=tk.RIGHT, padx=(0, 8)
        )

        # 实时读数（扩展：新增泵）
        dash = ttk.Labelframe(left, text="实时读数（三位小数显示）", padding=10);
        dash.pack(fill=tk.X, pady=8)

        def make_stat(parent, title, var):
            f = ttk.Frame(parent);
            f.pack(fill=tk.X, pady=1)
            ttk.Label(f, text=title).pack(side=tk.LEFT)
            ttk.Label(f, textvariable=var, font=("Consolas", 14, "bold")).pack(side=tk.RIGHT)

        make_stat(dash, "温度 (°C)", self.cur_temp)
        make_stat(dash, "电压(热) (V)", self.cur_vout)
        make_stat(dash, "电流(热) (A)", self.cur_aout)
        make_stat(dash, "功率(热) (W)", self.cur_pout)
        make_stat(dash, "状态 (CV/CC)", self.cvcc_state)
        sep = ttk.Separator(dash, orient=tk.HORIZONTAL);
        sep.pack(fill=tk.X, pady=6)
        make_stat(dash, "误差 ε (°C)", self.cur_err)
        make_stat(dash, "MAE(窗) (°C)", self.mae_win)
        make_stat(dash, "RMSE(窗) (°C)", self.rmse_win)
        make_stat(dash, "命中率(窗) (%)", self.hit_ratio)
        sep2 = ttk.Separator(dash, orient=tk.HORIZONTAL);
        sep2.pack(fill=tk.X, pady=6)
        make_stat(dash, "泵 V_out (V)", self.cur_pump_v)
        make_stat(dash, "泵 I_out (A)", self.cur_pump_a)
        make_stat(dash, "泵 P_out (W)", self.cur_pump_p)
        make_stat(dash, "泵 V_set (V)", self.cur_pump_vset)
        make_stat(dash, "泵 Duty (%)", self.pwm_duty)  # 纯数值
        make_stat(dash, "泵 开关比", self.pwm_duty_str)  # “xx.x%”

        # —— 新增：实时精度与拟合 / 滤波 ——
        acc = ttk.Labelframe(left, text="实时精度与拟合 / 滤波", padding=10);
        acc.pack(fill=tk.X, pady=8)

        rowa = ttk.Frame(acc);
        rowa.pack(fill=tk.X)
        ttk.Label(rowa, text="统计窗口 (s)").pack(side=tk.LEFT)
        ttk.Entry(rowa, textvariable=self.acc_window_s, width=9).pack(side=tk.LEFT, padx=(6, 12))
        ttk.Label(rowa, text="容差 ±(°C)").pack(side=tk.LEFT)
        ttk.Entry(rowa, textvariable=self.acc_tol, width=9).pack(side=tk.LEFT, padx=(6, 12))
        ttk.Label(rowa, text="拟合窗口 (s)").pack(side=tk.LEFT)
        ttk.Entry(rowa, textvariable=self.fit_window_s, width=9).pack(side=tk.LEFT, padx=(6, 12))
        ttk.Label(rowa, text="温度滤波 τ (s)").pack(side=tk.LEFT)
        ttk.Entry(rowa, textvariable=self.temp_tau, width=9).pack(side=tk.LEFT, padx=(6, 0))

        ttk.Label(acc, textvariable=self.acc_info, bootstyle=INFO).pack(anchor="w", pady=(6, 0))

        rowb = ttk.Frame(acc);
        rowb.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(rowb, text="线性拟合斜率 (°C/min)").pack(side=tk.LEFT)
        ttk.Label(rowb, textvariable=self.fit_slope, font=("Consolas", 12, "bold")).pack(side=tk.LEFT, padx=8)

        rowc = ttk.Frame(acc);
        rowc.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(rowc, text="判稳 MAE ≤(°C)").pack(side=tk.LEFT)
        ttk.Entry(rowc, textvariable=self.mae_stable_thr, width=8).pack(side=tk.LEFT, padx=(6, 12))
        ttk.Label(rowc, text="CSV起止斜率阈(°C/min)  起").pack(side=tk.LEFT)
        ttk.Entry(rowc, textvariable=self.slope_start_thr_cpm, width=8).pack(side=tk.LEFT, padx=(6, 6))
        ttk.Label(rowc, text="止").pack(side=tk.LEFT)
        ttk.Entry(rowc, textvariable=self.slope_stop_thr_cpm, width=8).pack(side=tk.LEFT, padx=(6, 12))

        ttk.Label(acc, text="说明：判稳窗口用“统计窗口(s)”，斜率用“拟合窗口(s)”。", bootstyle=INFO) \
            .pack(anchor="w", pady=(6, 0))

        # PID / 采样 / 限制（保留）
        params = ttk.Labelframe(left, text="PID / 采样 / 限制", padding=10);
        params.pack(fill=tk.X, pady=8)
        grid = ttk.Frame(params);
        grid.pack(fill=tk.X)

        def add_row(col, text, var, width=9):
            ttk.Label(grid, text=text).grid(row=0, column=col * 2, sticky="w", padx=(0, 4))
            ttk.Entry(grid, textvariable=var, width=width).grid(row=0, column=col * 2 + 1, sticky="w", padx=(0, 10))

        add_row(0, "Kp(热)", self.kp);
        add_row(1, "Ki(热)", self.ki);
        add_row(2, "Kd(热)", self.kd)

        grid2 = ttk.Frame(params);
        grid2.pack(fill=tk.X, pady=(8, 0))
        add2 = [("采样(Hz)", self.sample_hz), ("V上限(热)", self.v_limit), ("斜坡(V/s)(热/泵)", self.slew_vps)]
        for i, (t, v) in enumerate(add2):
            ttk.Label(grid2, text=t).grid(row=0, column=i * 2, sticky="w", padx=(0, 4))
            ttk.Entry(grid2, textvariable=v, width=9).grid(row=0, column=i * 2 + 1, sticky="w", padx=(0, 12))

        # 循环泵参数
        pump_param = ttk.Labelframe(left, text="循环泵参数（冷却 PID）", padding=10); pump_param.pack(fill=tk.X, pady=8)
        g1 = ttk.Frame(pump_param); g1.pack(fill=tk.X)
        ttk.Label(g1, text="Kp(冷)").grid(row=0, column=0, sticky="w"); ttk.Entry(g1, textvariable=self.kp_cool, width=9).grid(row=0, column=1, padx=(4,12))
        ttk.Label(g1, text="Ki(冷)").grid(row=0, column=2, sticky="w"); ttk.Entry(g1, textvariable=self.ki_cool, width=9).grid(row=0, column=3, padx=(4,12))
        ttk.Label(g1, text="Kd(冷)").grid(row=0, column=4, sticky="w"); ttk.Entry(g1, textvariable=self.kd_cool, width=9).grid(row=0, column=5, padx=(4,12))
        g2 = ttk.Frame(pump_param); g2.pack(fill=tk.X, pady=(6,0))
        ttk.Label(g2, text="泵最小/最大电压(V)").grid(row=0, column=0, sticky="w")
        ttk.Entry(g2, textvariable=self.pump_vmin, width=9).grid(row=0, column=1, padx=(4,12))
        ttk.Entry(g2, textvariable=self.pump_vmax, width=9).grid(row=0, column=2, padx=(4,12))
        ttk.Label(g2, text="死区(°C)").grid(row=0, column=3, sticky="w")
        ttk.Entry(g2, textvariable=self.pump_deadband, width=9).grid(row=0, column=4, padx=(4,12))
        ttk.Label(g2, text="泵斜坡(V/s)").grid(row=0, column=5, sticky="w")
        ttk.Entry(g2, textvariable=self.pump_slew_vps, width=9).grid(row=0, column=6, padx=(4,12))
        g3 = ttk.Frame(pump_param);
        g3.pack(fill=tk.X, pady=(6, 0))

        # UI（放到 _build_ui 里，替换你现有的“循环泵 PWM 参数”那一块）
        pwmf = ttk.Labelframe(left, text="循环泵 PWM（自动）", padding=10)
        pwmf.pack(fill=tk.X, pady=8)

        rowA = ttk.Frame(pwmf);
        rowA.pack(fill=tk.X, pady=2)
        ttk.Checkbutton(rowA, text="启用自动 PWM", variable=self.pump_pwm_auto_enable) \
            .pack(side=tk.LEFT, padx=(0, 12))
        ttk.Label(rowA, text="频率(Hz)").pack(side=tk.LEFT)
        ttk.Entry(rowA, textvariable=self.pump_pwm_freq, width=8) \
            .pack(side=tk.LEFT, padx=(4, 12))
        ttk.Label(rowA, text="(机械继电器建议 ≤2Hz)").pack(side=tk.LEFT)

        rowB = ttk.Frame(pwmf);
        rowB.pack(fill=tk.X, pady=2)
        ttk.Label(rowB, text="PWM 开(高)电压 V_on (V)").pack(side=tk.LEFT)
        ttk.Entry(rowB, textvariable=self.pump_pwm_v_on, width=8) \
            .pack(side=tk.LEFT, padx=(6, 12))
        ttk.Label(rowB, text="最短开时长 (ms)").pack(side=tk.LEFT)
        ttk.Entry(rowB, textvariable=self.pump_pwm_min_on_ms, width=8) \
            .pack(side=tk.LEFT, padx=(6, 12))

        rowC = ttk.Frame(pwmf);
        rowC.pack(fill=tk.X, pady=2)
        ttk.Label(rowC, text="高温阈值(°C)  (低于此阈值=正常PID, 高于=PWM)").pack(side=tk.LEFT)
        ttk.Entry(rowC, textvariable=self.pump_pwm_hi_temp_thr, width=8) \
            .pack(side=tk.LEFT, padx=(6, 12))

        # 在_build_ui方法中的PWM参数部分添加输入框
        rowX = ttk.Frame(pwmf);
        rowX.pack(fill=tk.X, pady=2)
        ttk.Label(rowX, text="占空比下限(%)").pack(side=tk.LEFT)
        ttk.Entry(rowX, textvariable=self.pump_pwm_min_duty, width=8).pack(side=tk.LEFT, padx=(6, 12))
        ttk.Label(rowX, text="(低于此值不闭合)").pack(side=tk.LEFT)


        rowD = ttk.Frame(pwmf);
        rowD.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(rowD, text="当前占空比:").pack(side=tk.LEFT)
        ttk.Label(rowD, textvariable=self.relay_pwm_duty, font=("Consolas", 12, "bold")) \
            .pack(side=tk.LEFT, padx=(6, 16))
        ttk.Label(rowD, text="等效V_set:").pack(side=tk.LEFT)
        ttk.Label(rowD, textvariable=self.cur_pump_vset, font=("Consolas", 12, "bold")) \
            .pack(side=tk.LEFT, padx=(6, 0))
        ttk.Label(rowD, text="V").pack(side=tk.LEFT, padx=(4, 0))

        # === 模拟点击（开始线性时触发） ===
        clickf = ttk.Labelframe(left, text="模拟点击（开始线性时）", padding=10)
        clickf.pack(fill=tk.X, pady=8)

        r0 = ttk.Frame(clickf);
        r0.pack(fill=tk.X, pady=2)
        ttk.Checkbutton(r0, text="启用", variable=self.auto_click_enable).pack(side=tk.LEFT)
        ttk.Label(r0, text="延迟(ms)").pack(side=tk.LEFT, padx=(12, 4))
        ttk.Entry(r0, textvariable=self.auto_click_delay_ms, width=8).pack(side=tk.LEFT)

        r1 = ttk.Frame(clickf);
        r1.pack(fill=tk.X, pady=2)
        ttk.Label(r1, text="屏幕X").pack(side=tk.LEFT)
        self.ent_auto_x = ttk.Entry(r1, textvariable=self.auto_click_x, width=8)
        self.ent_auto_x.pack(side=tk.LEFT, padx=(4, 12))

        ttk.Label(r1, text="屏幕Y").pack(side=tk.LEFT)
        self.ent_auto_y = ttk.Entry(r1, textvariable=self.auto_click_y, width=8)
        self.ent_auto_y.pack(side=tk.LEFT, padx=(4, 12))

        ttk.Label(r1, text="按键").pack(side=tk.LEFT)
        ttk.Combobox(r1, textvariable=self.auto_click_button, width=8,
                     values=["left", "right", "middle"]).pack(side=tk.LEFT, padx=(4, 12))
        ttk.Checkbutton(r1, text="双击", variable=self.auto_click_double).pack(side=tk.LEFT)

        # —— 坐标拾取 ——
        r2 = ttk.Frame(clickf);
        r2.pack(fill=tk.X, pady=2)
        ttk.Label(r2, textvariable=self.pick_preview, width=18).pack(side=tk.LEFT)
        ttk.Button(r2, text="按 Enter 取点", command=self._start_pick_coord_enter).pack(side=tk.LEFT, padx=(6, 6))
        ttk.Button(r2, text="停止拾取", command=self._stop_pick_coord).pack(side=tk.LEFT, padx=(6, 0))

        ttk.Button(clickf, text="立即测试点击",
                   command=lambda: self._simulate_click(
                       int(self.auto_click_x.get() or 0),
                       int(self.auto_click_y.get() or 0),
                       self.auto_click_button.get(),
                       bool(self.auto_click_double.get()),
                       int(self.auto_click_delay_ms.get() or 0)
                   ),
                   bootstyle=SECONDARY).pack(anchor="e", pady=(6, 0))

        # 数据记录
        rec = ttk.Labelframe(left, text="数据记录（CSV 导出）", padding=10);
        rec.pack(fill=tk.X, pady=8)
        rowr = ttk.Frame(rec);
        rowr.pack(fill=tk.X)
        ttk.Label(rowr, text="保存目录").pack(side=tk.LEFT)
        self.ent_csv_dir = ttk.Entry(rowr, textvariable=self.csv_dir, width=48)
        self.ent_csv_dir.pack(side=tk.LEFT, padx=(6, 8))
        ttk.Button(rowr, text="选择目录", command=self._choose_csv_dir, bootstyle=SECONDARY).pack(side=tk.LEFT)
        ttk.Label(rec, text="规则：点击「启动PID/开始线性程序」新建；点击「停止…」保存并关闭。",
                  bootstyle=INFO, wraplength=560, justify="left").pack(anchor="w", pady=(6, 0))

        plot_ctrl = ttk.Labelframe(left, text="绘图显示", padding=10);
        plot_ctrl.pack(fill=tk.X, pady=8)
        rowp = ttk.Frame(plot_ctrl);
        rowp.pack(fill=tk.X)
        ttk.Label(rowp, text="窗口 (s)").pack(side=tk.LEFT)
        ttk.Entry(rowp, textvariable=self.plot_window_s, width=10).pack(side=tk.LEFT, padx=(6, 12))
        ttk.Label(rowp, text="刷新间隔 (ms)").pack(side=tk.LEFT)
        ttk.Entry(rowp, textvariable=self.ui_refresh_ms, width=10).pack(side=tk.LEFT, padx=(6, 0))

        # 右侧图表（三面板：上1、下2）
        fig = Figure(figsize=(10, 12), dpi=100)

        # 上面一个、下面两个：上行跨两列
        gs = fig.add_gridspec(
            2, 2,
            height_ratios=[1.2, 1.0],  # 上面略高
            wspace=0.35, hspace=0.40  # 面板间距
        )

        # 顶部：温度（跨两列）
        self.ax_temp = fig.add_subplot(gs[0, :])

        # 左下：加热功率
        self.ax_power = fig.add_subplot(gs[1, 0])

        # 右下：循环泵
        self.ax_pump = fig.add_subplot(gs[1, 1])

        # -------- 顶部温度曲线 --------
        self.ax_temp.set_title("Temperature vs Time")
        self.ax_temp.set_ylabel("Temperature (°C)")
        self.line_temp, = self.ax_temp.plot([], [], lw=1.6, label="Measured")
        self.line_target, = self.ax_temp.plot([], [], lw=1.2, ls="--", label="Target")
        self.line_fit, = self.ax_temp.plot([], [], lw=1.2, ls=":", label="Linear Fit")
        self.ax_temp.legend(loc="upper right", fontsize=8)
        self.ax_temp.grid(True, alpha=0.3)

        # -------- 左下功率（加热端）--------
        self.ax_power.set_title("Heater Power vs Time")
        self.ax_power.set_ylabel("Power (W)")
        self.ax_power.set_xlabel("Time (s)")
        self.line_power, = self.ax_power.plot([], [], lw=1.2, label="Heater Power")
        self.ax_power.grid(True, alpha=0.3)
        self.ax_power.legend(loc="upper right", fontsize=8)

        # -------- 右下：循环泵功率（仅功率 vs 时间）--------
        self.ax_pump.set_title("Pump Power vs Time")
        self.ax_pump.set_ylabel("Power (W)")
        self.ax_pump.set_xlabel("Time (s)")
        self.line_pump, = self.ax_pump.plot([], [], lw=1.2, label="Pump Power")
        self.ax_pump.grid(True, alpha=0.3)
        self.ax_pump.legend(loc="upper right", fontsize=8)

        # 挂到 Tk 画布
        self.canvas = FigureCanvasTkAgg(fig, master=right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


    @staticmethod
    def _disable_widget_scroll(*widgets):
        """禁用滚轮改变给定 Spinbox/Combobox 的值。"""
        def _block(event):
            return "break"

        for widget in widgets:
            if not widget:
                continue
            try:
                widget.bind("<MouseWheel>", _block)
                widget.bind("<Shift-MouseWheel>", _block)
                widget.bind("<Button-4>", _block)
                widget.bind("<Button-5>", _block)
            except Exception:
                pass


    def _sync_click_vars_to_entries(self):
        """把变量值强制刷新到 Entry（启动与加载配置后各调一次）。"""
        try:
            if hasattr(self, "ent_auto_x"):
                self.ent_auto_x.delete(0, "end");
                self.ent_auto_x.insert(0, str(int(self.auto_click_x.get() or 0)))
            if hasattr(self, "ent_auto_y"):
                self.ent_auto_y.delete(0, "end");
                self.ent_auto_y.insert(0, str(int(self.auto_click_y.get() or 0)))
        except Exception:
            pass

    # ---------- 串口 ----------
    def _refresh_ports(self):
        ports = [p.device for p in serial.tools.list_ports.comports()]
        self.combo_psu["values"] = ports
        self.combo_tcm["values"] = ports
        self.combo_pump["values"] = ports
        # ★ 新增：继电器端口列表
        self.combo_relay["values"] = ports

    def _auto_connect_all(self):
        try:
            ports = [p.device for p in serial.tools.list_ports.comports()]
        except Exception as e:
            msg = f"自动连接失败：无法列出串口（{e}）。请检查 USB 插口。"
            self._set_status(msg)
            messagebox.showwarning("提示", msg)
            return

        if not ports:
            msg = "自动连接失败：未检测到串口，请检查 USB 插口。"
            self._set_status(msg)
            messagebox.showwarning("提示", msg)
            return

        used_ports = set()
        for var, device in (
            (self.psu_port, getattr(self.psu, "connected", False)),
            (self.tcm_port, getattr(self.tcm, "connected", False)),
            (self.pump_port, getattr(self.pump, "connected", False)),
            (self.relay_port, getattr(self.relay, "connected", False)),
        ):
            if device:
                cur = (var.get() or "").strip()
                if cur:
                    used_ports.add(cur)

        def _int_from_var(var, default):
            try:
                val = var.get()
            except Exception:
                val = default
            try:
                if isinstance(val, str):
                    val = val.strip()
                    if val == "":
                        return int(default)
                    return int(float(val))
                return int(val)
            except Exception:
                return int(default)

        failures: list[tuple[str, Exception | None]] = []
        successes: list[str] = []

        def candidates(var):
            cur = (var.get() or "").strip()
            ordered = []
            if cur:
                ordered.append(cur)
            for port in ports:
                if port not in ordered:
                    ordered.append(port)
            preferred = [p for p in ordered if p not in used_ports]
            fallback = [p for p in ordered if p in used_ports and p not in preferred]
            return preferred + fallback

        def try_device(name, is_connected, port_var, connect_call, on_success, status_var):
            if is_connected:
                return
            last_err: Exception | None = None
            for port in candidates(port_var):
                try:
                    connect_call(port)
                    used_ports.add(port)
                    on_success(port)
                    successes.append(name)
                    return
                except Exception as exc:  # noqa: PERF203
                    last_err = exc
            failures.append((name, last_err))
            status_var.set("连接失败")

        try_device(
            "电源",
            self.psu.connected,
            self.psu_port,
            lambda port: self.psu.connect(
                port,
                _int_from_var(self.psu_baud, PSU_BAUD),
                _int_from_var(self.addr_psu, PSU_ADDR_DEFAULT),
            ),
            lambda port: self._auto_connect_success_psu(port),
            self.psu_status,
        )
        try_device(
            "温度仪",
            self.tcm.connected,
            self.tcm_port,
            lambda port: self.tcm.connect(
                port,
                _int_from_var(self.tcm_baud, TCM_BAUD),
                _int_from_var(self.addr_tcm, TCM_ADDR_DEFAULT),
            ),
            lambda port: self._auto_connect_success_tcm(port),
            self.tcm_status,
        )
        try_device(
            "循环泵",
            getattr(self.pump, "connected", False),
            self.pump_port,
            lambda port: self.pump.connect(
                port,
                _int_from_var(self.pump_baud, PUMP_BAUD),
                _int_from_var(self.addr_pump, PUMP_ADDR_DEFAULT),
            ),
            lambda port: self._auto_connect_success_pump(port),
            self.pump_status,
        )
        try_device(
            "继电器",
            getattr(self.relay, "connected", False),
            self.relay_port,
            lambda port: self.relay.connect(port, baud=_int_from_var(self.relay_baud, 9600)),
            lambda port: self._auto_connect_success_relay(port),
            self.relay_status,
        )

        if failures:
            detail = []
            for name, err in failures:
                if err is None:
                    detail.append(f"{name}: 未能在任何串口上连接")
                else:
                    detail.append(f"{name}: {err}")
            msg = "以下设备自动连接失败，请检查 USB 插口：\n" + "\n".join(detail)
            self._set_status(msg.replace("\n", " "))
            messagebox.showwarning("提示", msg)
        elif successes:
            summary = f"自动连接完成（{', '.join(successes)}）。"
            self._set_status(summary)
        else:
            self._set_status("自动连接：所有设备均已连接。")

    def _auto_connect_success_psu(self, port: str):
        self.psu_port.set(port)
        self.psu_status.set("已连接")
        self._set_status("电源已连接（自动）")
        try:
            self._save_config()
        except Exception:
            pass
        self._maybe_start_acquisition()

    def _auto_connect_success_tcm(self, port: str):
        self.tcm_port.set(port)
        self.tcm_status.set("已连接")
        self._set_status("温度仪已连接（自动）")
        try:
            self._save_config()
        except Exception:
            pass
        self._maybe_start_acquisition()

    def _auto_connect_success_pump(self, port: str):
        self.pump_port.set(port)
        self.pump_status.set("已连接")
        self._set_status("循环泵已连接（自动）")
        try:
            self._save_config()
        except Exception:
            pass
        self._maybe_start_acquisition()
        if self.acq_running:
            try:
                self._ensure_pump_sampler_running()
            except Exception:
                pass

    def _auto_connect_success_relay(self, port: str):
        self.relay_port.set(port)
        self.relay_status.set("已连接")
        self._set_status("继电器已连接（自动）")
        try:
            self._save_config()
        except Exception:
            pass
        self._maybe_start_acquisition()

    # ---------- 连接/断开 ----------
    def _connect_psu(self):
        try:
            if self.psu.connected:
                messagebox.showinfo("提示", "电源已连接")
                return
            self.psu.connect(self.psu_port.get(), self.psu_baud.get(), self.addr_psu.get())
            self.psu_status.set("已连接")
            self._set_status("电源已连接（远程）")
            self._save_config()
            self._maybe_start_acquisition()
        except Exception as e:
            self.psu_status.set("连接失败")
            self._set_status(f"连接电源失败: {e}")
            messagebox.showerror("错误", f"连接电源失败：\n{e}")

    def _disconnect_psu(self):
        self.psu.disconnect()
        self.psu_status.set("未连接")
        self._set_status("电源已断开")
        self.disable_pid()
        self._save_config()

    def _connect_tcm(self):
        try:
            if self.tcm.connected:
                messagebox.showinfo("提示", "温度仪已连接")
                return
            self.tcm.connect(self.tcm_port.get(), self.tcm_baud.get(), self.addr_tcm.get())
            self.tcm_status.set("已连接")
            self._set_status("温度仪已连接")
            self._save_config()
            self._maybe_start_acquisition()
        except Exception as e:
            self.tcm_status.set("连接失败")
            self._set_status(f"连接温度仪失败: {e}")
            messagebox.showerror("错误", f"连接温度仪失败：\n{e}")

    def _disconnect_tcm(self):
        self.tcm.disconnect()
        self.tcm_status.set("未连接")
        self._set_status("温度仪已断开")
        self.disable_pid()
        self._save_config()

    def _choose_csv_dir(self):
        """让用户选择 CSV 保存目录，并落盘到 config。"""
        try:
            cur = self.csv_dir.get().strip() or str(get_logs_dir())
            path = filedialog.askdirectory(initialdir=cur, title="选择 CSV 保存目录")
            if path:
                p = Path(path)
                p.mkdir(parents=True, exist_ok=True)
                self.csv_dir.set(str(p))
                self._set_status(f"CSV 保存目录已设置：{p}")
                self._save_config()
        except Exception as e:
            messagebox.showerror("错误", f"选择目录失败：\n{e}")

    @staticmethod
    def _csv_sanitize_name(name: str) -> str:
        """Windows/Unix 通用的安全文件名替换。"""
        bad = '<>:"/\\|?*'
        s = "".join(('_' if c in bad else c) for c in name.strip())
        return s if s else "op"

    def _debounce(self, key: str, min_interval=0.4) -> bool:
        """按钮防抖：min_interval 秒内重复点击忽略，返回 False 表示拦截。"""
        now = time.perf_counter()
        last = self._last_click_ts.get(key, 0.0)
        if now - last < min_interval:
            return False
        self._last_click_ts[key] = now
        return True

    def _csv_open_session(self, op_name: str):
        """
        打开新的 CSV 会话文件：
          - 如果已有打开的会话，先安全关闭（相当于切文件）
          - 文件名：操作_YYYYmmdd_HHMMSS.csv
        """
        try:
            # 若当前有会话，先关闭（切换）
            if self.csv_writer:
                old = self.csv_session_path
                try:
                    self.csv_file.flush();
                    self.csv_file.close()
                except Exception:
                    pass
                self.csv_file = None;
                self.csv_writer = None
                self._set_status(f"已保存：{old}")

            op = self._csv_sanitize_name(op_name)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            dirp = Path(self.csv_dir.get().strip() or str(get_logs_dir()))
            dirp.mkdir(parents=True, exist_ok=True)
            fpath = dirp / f"{op}_{ts}.csv"

            self.csv_file = open(fpath, "w", newline="", encoding="utf-8")
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(["time_s", "temp_C", "target_C",
                                      "V_set", "V_out", "A_out", "P_out",
                                      "Kp", "Ki", "Kd", "e", "CC", "note"])
            self.csv_session_op = op_name
            self.csv_session_path = fpath
            self._set_status(f"开始记录：{fpath.name}")
        except Exception as e:
            self.csv_file = None;
            self.csv_writer = None
            self.csv_session_op = None;
            self.csv_session_path = None
            messagebox.showerror("错误", f"创建 CSV 失败：\n{e}")

    def _csv_close_session(self, end_reason: str):
        """
        关闭当前 CSV 会话（若有）：写入落盘，清空句柄。
        end_reason 用于日志展示；不往文件尾写注释，避免 CSV 不标准。
        """
        if not self.csv_writer:
            # 没会话也不报错，提升鲁棒性
            self._set_status("没有进行中的 CSV 会话（跳过保存）")
            return
        try:
            path = self.csv_session_path
            # 关闭
            try:
                self.csv_file.flush();
                self.csv_file.close()
            except Exception:
                pass
            finally:
                self.csv_file = None;
                self.csv_writer = None
                self.csv_session_op = None;
                self.csv_session_path = None
            self._set_status(f"已保存（{end_reason}）：{path}")
        except Exception as e:
            self._set_status(f"保存 CSV 失败：{e}")

    def _compute_mae_recent(self, window_s: float) -> float | None:
        """
        在最近 window_s 秒内，用 err_buf 计算 MAE（|误差|的平均）。
        没有足够点则返回 None。
        """
        try:
            with self.history_lock:
                t_list = list(self.t_buf)
                e_list = list(self.err_buf)
            if not t_list or not e_list:
                return None
            t_now = t_list[-1]
            t_min = t_now - max(1.0, float(window_s))
            errors = [
                e for t, e in zip(t_list, e_list)
                if t >= t_min and e is not None and not isnan(e)
            ]
            if len(errors) < 5:
                return None
            return sum(abs(x) for x in errors) / len(errors)
        except Exception:
            return None

    def _connect_relay(self):
        try:
            if self.relay.connected:
                messagebox.showinfo("提示", "继电器已连接"); return
            port = self.relay_port.get().strip()
            baud = int(self.relay_baud.get() or 9600)
            if not port:
                raise ValueError("未选择继电器端口")
            self.relay.connect(port, baud=baud)
            self.relay_status.set("已连接")
            self._set_status("继电器已连接")
            # 设置相位起点
            self._relay_epoch = time.perf_counter()
            self._save_config()
            # 不强制作为采集启动条件：继电器可选
            self._maybe_start_acquisition()
        except Exception as e:
            self.relay_status.set("连接失败")
            self._set_status(f"连接继电器失败: {e}")
            messagebox.showerror("错误", f"连接继电器失败：\n{e}")

    def _disconnect_relay(self):
        try:
            if self.relay.connected and self._relay_is_on:
                # 断开前尽量关掉
                try:
                    self.relay.set(self.relay_chan.get(), False, feedback=False)
                except Exception:
                    pass
        finally:
            self.relay.disconnect()
            self.relay_status.set("未连接")
            self._set_status("继电器已断开")
            self._save_config()

    def _connect_pump(self):
        try:
            if self.pump.connected:
                messagebox.showinfo("提示", "循环泵已连接")
                return
            self.pump.connect(self.pump_port.get(), self.pump_baud.get(), self.addr_pump.get())
            self.pump_status.set("已连接")
            self._set_status("循环泵已连接")
            self._save_config()
            self._maybe_start_acquisition()
            if self.acq_running:
                try:
                    self._ensure_pump_sampler_running()
                except Exception:
                    pass
        except Exception as e:
            self.pump_status.set("连接失败")
            self._set_status(f"连接循环泵失败: {e}")
            messagebox.showerror("错误", f"连接循环泵失败：\n{e}")

    def _disconnect_pump(self):
        self.pump.disconnect()
        try:
            self._stop_pump_sampler()
        except Exception:
            pass
        self.pump_status.set("未连接")
        self._set_status("循环泵已断开")
        self.disable_pid()
        self._save_config()


    def _compute_recent_slope_cpm(self, window_s: float) -> float | None:
        """
        用最近 window_s 秒的 (t,temp) 做最小二乘拟合，返回斜率（°C/分钟）。
        点太少或退化返回 None。
        """
        try:
            t_list = list(self.t_buf)
            y_list = list(self.temp_buf)
            if not t_list or not y_list:
                return None
            t_now = t_list[-1]
            t_min = t_now - max(5.0, float(window_s))
            i0 = 0
            for i in range(len(t_list) - 1, -1, -1):
                if t_list[i] < t_min:
                    i0 = i + 1
                    break
            xs = [];
            ys = []
            for tx, ty in zip(t_list[i0:], y_list[i0:]):
                if (ty is not None) and (not isnan(ty)):
                    xs.append(tx);
                    ys.append(ty)
            if len(xs) < 8:
                return None
            # 去相对时间，防止病态
            t0 = xs[0]
            xr = [x - t0 for x in xs]
            n = float(len(xr))
            sx = sum(xr);
            sy = sum(ys)
            sxx = sum(xx * xx for xx in xr)
            sxy = sum(xx * yy for xx, yy in zip(xr, ys))
            denom = n * sxx - sx * sx
            if abs(denom) < 1e-9:
                return None
            a = (n * sxy - sx * sy) / denom  # °C/秒
            return a * 60.0  # 转 °C/分钟
        except Exception:
            return None

    # ---------- 令牌工具 ----------
    def _next_tok(self):
        # 首次调用时初始化
        if not hasattr(self, "_tok_lock"):
            self._tok_lock = threading.Lock()
            self._tok = 0
        with self._tok_lock:
            self._tok += 1
            return self._tok

    # ---------- 线程安全的 UI 工具 ----------
    def _tk_call(self, fn, *args, **kwargs):
        """在主线程执行任意 fn(*args, **kwargs)。"""
        try:
            # 若已在主线程，直接调用；否则投递到主线程
            if threading.current_thread() is threading.main_thread():
                fn(*args, **kwargs)
            else:
                self.root.after(0, lambda: fn(*args, **kwargs))
        except Exception:
            pass

    def _tk_call_sync(self, fn, *args, **kwargs):
        """在主线程同步执行函数并返回结果。"""
        if threading.current_thread() is threading.main_thread():
            return fn(*args, **kwargs)

        result = {}
        event = threading.Event()

        def wrapper():
            try:
                result["value"] = fn(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                result["error"] = exc
            finally:
                event.set()

        try:
            self.root.after(0, wrapper)
            if not event.wait(timeout=3.0):
                raise TimeoutError("UI thread unresponsive")
        except Exception:
            raise

        if "error" in result:
            raise result["error"]
        return result.get("value")

    def _tk_set(self, tkvar, value):
        """线程安全地设置 Tk 变量."""
        self._tk_call(lambda v: tkvar.set(v), value)

    def _gf(self, tkvar, default, lo=None, hi=None):
        """
        get-float：从 Tk 变量取 float，空/非法一律回落 default，并可选夹紧到 [lo, hi]。
        """
        try:
            s = tkvar.get() if hasattr(tkvar, "get") else tkvar
            if s is None or str(s).strip() == "":
                val = float(default)
            else:
                val = float(s)
            if val != val:  # NaN
                val = float(default)
        except Exception:
            val = float(default)
        if lo is not None:
            val = max(lo, val)
        if hi is not None:
            val = min(hi, val)
        return val

    def _gi(self, tkvar, default, lo=None, hi=None):
        """
        get-int：从 Tk 变量取 int，空/非法回落 default，并可选夹紧。
        """
        v = int(round(self._gf(tkvar, default)))
        if lo is not None:
            v = max(lo, v)
        if hi is not None:
            v = min(hi, v)
        return v

    def _set_status(self, msg: str):
        """线程安全的状态栏更新。"""
        # 先更新后端缓存
        with self._status_lock:
            self._status_msg = msg
        # 再让主线程改 Tk StringVar
        self._tk_set(self.status_text, msg)

    # ---------- 采样器管理 ----------
    def _stop_pump_sampler(self):
        """停止循环泵采样线程，并清理句柄。"""
        sampler = getattr(self, "_sampler_pump", None)
        if not sampler:
            self._sampler_pump = None
            return
        try:
            sampler.stop()
        except Exception:
            pass
        self._sampler_pump = None

    def _ensure_pump_sampler_running(self):
        """在采集运行时根据连接状态启动/停止循环泵采样器。"""
        if not getattr(self, "acq_running", False):
            return
        if not (hasattr(self, "pump") and getattr(self.pump, "connected", False)):
            self._stop_pump_sampler()
            return
        if getattr(self, "_sampler_pump", None):
            return
        self._sampler_pump = _AsyncSampler("pump", lambda: _read_pump_once(self))
        self._sampler_pump.start()

    # ---------- 采集/控制 ----------
    def _maybe_start_acquisition(self):
        # 三个设备都连上就启动采集；按钮不再在这里禁用/启用
        if (self.psu.connected and self.tcm.connected and self.pump.connected
                and not self.acq_running):
            self.start_acquisition()
        # 保持按钮始终可点，具体能否执行由 enable_pid/disable_pid 自己判断

    def start_acquisition(self):
        if self.acq_running:
            return
        self.stop_event.clear()
        # 重置 token 计数器，保证之后单调递增
        self._tok = 0
        if not hasattr(self, "_tok_lock"):
            self._tok_lock = threading.Lock()

        self.acq_running = True
        self._acq_t0 = time.perf_counter()
        self._set_status("采集中…")
        _start_async_samplers(self)
        self.thread = threading.Thread(target=self._acq_loop, daemon=True)
        self.thread.start()

    def stop_acquisition(self):
        if not self.acq_running:
            return
        self.stop_event.set()
        # ★ 新增：先停采样器，再等主环退出，避免悬挂
        _stop_async_samplers(self)
        if self.thread:
            self.thread.join(timeout=2.0)
        self.acq_running = False
        self._set_status("采集已停止")

    def enable_pid(self, origin="user", notify=True):
        # 防抖，避免双击反复开文件
        if origin == "user" and not self._debounce("enable_pid"):
            return False

        if not self.acq_running:
            if notify:
                messagebox.showwarning("提示", "请先连接设备，采集会自动开始。")
            return False
        try:
            # 同步参数
            self._sync_params_to_background()

            # 读一次温度（限时 0.6s），避免阻塞 UI
            ok_temp = False
            try:
                # 若还没采样器，兜底启动一下（不影响主状态）
                if not getattr(self, "_sampler_tcm", None):
                    _start_async_samplers(self)
                # 发一个快速请求并等待到期或完成
                tok = self._next_tok()
                self._sampler_tcm.request(tok)
                deadline = time.perf_counter() + 0.6
                while time.perf_counter() < deadline:
                    _, _, ok, done = self._sampler_tcm.latest()
                    if done == tok and ok:  # 必须是“本次”的回包
                        ok_temp = True
                        break
                    time.sleep(0.01)
            except Exception:
                ok_temp = False

            if not ok_temp and notify:
                messagebox.showwarning("提示", "温度仪暂未返回有效读数，将在采集循环中继续等待（不阻塞）。")

            # 打开输出并把当前设定电压明确写回（0 或上次值）
            self.psu.set_output(True)
            self.last_vset = max(0.0, float(getattr(self, "last_vset", 0.0)))
            try:
                self.psu.set_voltage(self.last_vset)
            except Exception:
                pass

            with self.params_lock:
                if "ctrl_dir" not in self.params:
                    self.params["ctrl_dir"] = 1.0
                self.pid = PID(self.params["kp"], self.params["ki"], self.params["kd"],
                               u_min=0.0, u_max=self.params["v_limit"])

            self.control_enabled = True
            self._set_status("PID 已启动（输出已开启）")

            # ★ 只有用户点按钮（origin="user"）才开新 CSV，会话名“启动PID”
            if origin == "user":
                # 若 PID 已在运行，这里等价“切换新文件”，不会重置控制
                self._csv_open_session("启动PID")

            return True

        except Exception as e:
            if notify:
                messagebox.showerror("错误", f"开启输出失败：\n{e}")
            return False

    def disable_pid(self):
        # 防抖
        if not self._debounce("disable_pid"):
            return

        # 任何“停止 PID”都会尝试收尾当前 CSV
        self._csv_close_session("停止PID")

        self.control_enabled = False
        try:
            if self.psu.connected:
                self.psu.set_output(False)
        except Exception:
            pass

        # ★ 停止温控后：确保循环泵处于 ON 且回到基线 vmin（默认 13 V）
        try:
            if hasattr(self, "pump") and self.pump and self.pump.connected:
                vmin = float(self.pump_vmin.get() or 13.0)
                self.pump.set_output(True)
                self.pump.set_voltage(vmin)
                self.pump_last_vset = vmin
        except Exception as e:
            self._set_status(f"停止PID时设置泵基线失败：{e}")

        self._set_status("PID 已停止（输出已关闭；泵已回到基线）")

    def _apply_target(self):
        """手动应用目标（从输入框解析，非法则保持旧值）"""
        t = safe_float(self.target_temp.get(), None)
        # 确保存在 _last_valid_target
        if not hasattr(self, "_last_valid_target"):
            self._last_valid_target = safe_float(self.target_temp.get(), 25.0) or 25.0

        if t is None:
            self._set_status(f"目标输入无效，保持原目标 {self._last_valid_target:.3f} °C")
        else:
            self._last_valid_target = float(t)
            with self.params_lock:
                self.params["target"] = self._last_valid_target
            self.pid.reset()
            self._set_status(f"目标已应用: {self._last_valid_target:.3f} °C")
        self._save_config()

    # ---------- 线性程序 ----------
    def _start_ramp(self, origin="user", notify=True):
        if self.ramp_active:
            if notify:
                messagebox.showinfo("提示", "线性程序已在运行。")
            return False
        if not self.acq_running:
            if notify:
                messagebox.showwarning("提示", "未在采集状态，无法执行线性程序。请先连接设备。")
            return False

        # 防抖
        if origin == "user" and not self._debounce("start_ramp"):
            return False

        t0 = safe_float(self.ramp_start.get(), None)
        t1 = safe_float(self.ramp_end.get(), None)
        rate = safe_float(self.ramp_rate.get(), None)  # °C/min
        holdm = max(0.0, safe_float(self.ramp_hold_min.get(), 0.0) or 0.0)
        if None in (t0, t1, rate) or rate <= 0:
            if notify:
                messagebox.showerror("错误", "请正确填写 起点/终点/速率(>0)。")
            return False

        # 若未启 PID，这里自动启（但标记为 ramp 来源，避免重复开 CSV）
        if not self.control_enabled:
            self.enable_pid(origin="ramp", notify=notify)
            if not self.control_enabled:
                return False

        # ★ 不在这里开CSV（等真正进入线性段时再开）
        with self.params_lock:
            self.params["target"] = float(t0)
        self._last_valid_target = float(t0)
        self.target_temp.set(f"{float(t0):.3f}")

        self.ramp_stop_evt.clear()
        self.ramp_active = True
        self._set_status("线性程序运行中…（先到起点，再开跑）")
        loop_flag = bool(self.ramp_cycle_enable.get())
        self.ramp_thread = threading.Thread(
            target=self._ramp_worker,
            args=(float(t0), float(t1), float(rate), float(holdm), True, loop_flag),  # wait_to_start=True, loop
            daemon=True
        )
        self.ramp_thread.start()
        return True

    def _ramp_worker(self, t0, t1, rate_c_per_min, hold_min, wait_to_start=True, loop=False):
        """
        线性升/降温工作线程（支持往返循环）：
          - 第一次按 MAE 判稳（可选），随后直接往返；
          - 每一段：按设定速率线性推进 → 末端保温(可选)；
          - CSV：斜率跨过起止阈则开/停，段与段之间自动收尾/重开；
          - “循环变温”勾选关闭后，本段结束即停止；“停止线性程序”立即停止。
        """
        setattr(self, "_auto_click_done_this_ramp", False)

        def _apply_target_now(val: float):
            with self.params_lock:
                self.params["target"] = float(val)
            self._last_valid_target = float(val)
            self._tk_set(self.target_temp, f"{float(val):.3f}")

        # ===== 首段：把目标拉到起点，并按需判稳 =====
        _apply_target_now(t0)

        if wait_to_start and not self.ramp_stop_evt.is_set():
            # 用你的“统计窗口/判稳阈值”
            try:
                win_s = max(5.0, safe_float(self.acc_window_s.get(), 30.0) or 30.0)
                mae_thr = max(0.0, safe_float(self.mae_stable_thr.get(), 0.20) or 0.20)
            except Exception:
                win_s, mae_thr = 30.0, 0.20

            timeout_s = 600.0
            deadline = time.perf_counter() + timeout_s
            while not self.ramp_stop_evt.is_set() and time.perf_counter() < deadline:
                mae = self._compute_mae_recent(win_s)
                if (mae is not None) and (mae <= mae_thr):
                    break
                self._tk_set(self.target_temp, f"{float(t0):.3f}")
                time.sleep(0.3)

            if self.ramp_stop_evt.is_set():
                self.ramp_active = False
                self._set_status("线性程序已停止（未进入线性段）")
                return

            if time.perf_counter() >= deadline:
                self._set_status(f"未在 {int(timeout_s)} s 内达到 MAE≤{mae_thr:.3f}，仍开始线性段")

        # ===== 往返循环：每次跑一段（start->end）=====
        cur_start, cur_end = float(t0), float(t1)
        first_segment = True

        while not self.ramp_stop_evt.is_set():
            # （运行中允许动态读“循环变温”开关，便于中途关闭）
            loop_now = bool(self.ramp_cycle_enable.get()) if loop else False

            # --- 一段的参数 ---
            sign = 1.0 if cur_end >= cur_start else -1.0
            self._ramp_last_sign = sign
            rate_c_per_s = abs(rate_c_per_min) / 60.0

            # 斜率阈值（用于 CSV 开/停）
            try:
                fit_win = max(5.0, safe_float(self.fit_window_s.get(), 40.0) or 40.0)
            except Exception:
                fit_win = 40.0
            thr_start = safe_float(self.slope_start_thr_cpm.get(), 0.0)
            thr_stop = safe_float(self.slope_stop_thr_cpm.get(), 0.0)
            if (thr_start is None) or (thr_start <= 0):
                thr_start = max(0.02, 0.5 * abs(rate_c_per_min))
            if (thr_stop is None) or (thr_stop <= 0):
                thr_stop = max(0.01, 0.2 * abs(rate_c_per_min))

            # 起点对齐（首段已对齐过；后续段直接设目标为新起点即可）
            _apply_target_now(cur_start)

            # --- 线性推进本段 ---
            t_seg0 = time.perf_counter()
            csv_started = False
            below_stop_since = None

            while not self.ramp_stop_evt.is_set():
                elapsed = time.perf_counter() - t_seg0
                target = cur_start + sign * rate_c_per_s * elapsed
                reached = (target >= cur_end) if (sign > 0) else (target <= cur_end)
                if reached:
                    target = cur_end

                _apply_target_now(target)

                # CSV 判定：用最近 fit_win 秒拟合得到的斜率
                slope_cpm = self._compute_recent_slope_cpm(fit_win)

                # 开始：跨上起始阈
                if (not csv_started) and (slope_cpm is not None) and (abs(slope_cpm) >= thr_start):
                    self._csv_open_session("开始线性程序")
                    self._set_status(f"线性程序：斜率 |{slope_cpm:.3f}|≥{thr_start:.3f} °C/min，开始记录 CSV")
                    csv_started = True
                    below_stop_since = None
                    if hasattr(self, "_try_auto_click_start") and first_segment and not getattr(self,
                                                                                                "_auto_click_done_this_ramp",
                                                                                                False):
                        self._try_auto_click_start()
                        setattr(self, "_auto_click_done_this_ramp", True)

                # 停止：低于停止阈且持续≥2 s
                if csv_started and (slope_cpm is not None) and (abs(slope_cpm) <= thr_stop):
                    if below_stop_since is None:
                        below_stop_since = time.perf_counter()
                    elif time.perf_counter() - below_stop_since >= 2.0:
                        self._csv_close_session("线性斜率回落（停止记录）")
                        csv_started = False
                        below_stop_since = None
                else:
                    below_stop_since = None

                if reached:
                    break
                time.sleep(0.1)

            # 末端保温
            if not self.ramp_stop_evt.is_set():
                hold_s = max(0.0, float(hold_min) * 60.0)
                t_hold_end = time.perf_counter() + hold_s
                while not self.ramp_stop_evt.is_set() and time.perf_counter() < t_hold_end:
                    _apply_target_now(cur_end)
                    slope_cpm = self._compute_recent_slope_cpm(fit_win)
                    if csv_started and (slope_cpm is not None) and (abs(slope_cpm) <= thr_stop):
                        if below_stop_since is None:
                            below_stop_since = time.perf_counter()
                        elif time.perf_counter() - below_stop_since >= 2.0:
                            self._csv_close_session("线性斜率回落（停止记录）")
                            csv_started = False
                            below_stop_since = None
                            break
                    else:
                        below_stop_since = None
                    time.sleep(0.1)

            # 段收尾
            if csv_started:
                self._csv_close_session("线性段完成（兜底关闭）")

            if self.ramp_stop_evt.is_set():
                break

            # 若不循环，或循环开关已被关闭，则结束
            if not loop_now:
                break

            # 往返：交换起终点，继续下一段；后续段不再做 MAE 判稳
            cur_start, cur_end = cur_end, cur_start
            first_segment = False
            self._set_status(f"循环变温：切换至下一段（{cur_start:.3f} → {cur_end:.3f} °C）")

        # 全部结束
        self.ramp_active = False
        self._ramp_last_sign = None
        self._set_status("线性程序完成")

    def _stop_ramp(self):
        # 防抖
        if not self._debounce("stop_ramp"):
            return

        if not self.ramp_active:
            # 即使没在跑，也尝试收尾（提升鲁棒性）
            self._csv_close_session("停止线性程序")
            self._set_status("线性程序未运行（已忽略）")
            return
        self.ramp_stop_evt.set()
        if self.ramp_thread:
            self.ramp_thread.join(timeout=2.0)
        self.ramp_active = False
        self._ramp_last_sign = None
        self._set_status("线性程序已停止")
        # 停止线性程序 => 保存 CSV
        self._csv_close_session("停止线性程序")

    # ---------- TCP 控制接口 ----------
    def _handle_tcp_command(self, request):
        cmd = str(request.get("cmd", "")).strip().lower()
        if not cmd:
            raise ValueError("missing cmd")

        if cmd == "ping":
            return {"timestamp": time.time()}

        if cmd == "status":
            return {"status": self.get_tcp_status()}

        if cmd == "set_target":
            if "value" not in request:
                raise ValueError("missing value")
            allow = bool(request.get("allow_ramp", False))
            self.set_target_temperature(request["value"], allow_during_ramp=allow)
            if request.get("start", False):
                self.start_pid_remote()
            return {"result": "target-updated"}

        if cmd == "start_pid":
            ok = self.start_pid_remote()
            return {"result": ok}

        if cmd == "stop_pid":
            self.stop_pid_remote()
            return {"result": True}

        if cmd == "start_ramp":
            params = request.get("params", {})
            start = params.get("start", request.get("start"))
            end = params.get("end", request.get("end"))
            rate = params.get("rate", request.get("rate"))
            if start is None or end is None or rate is None:
                raise ValueError("start/end/rate required")
            hold = params.get("hold", request.get("hold", 0.0))
            loop = params.get("loop", request.get("loop", False))
            ok = self.start_ramp_remote(start, end, rate, hold=hold, loop=loop)
            return {"result": ok}

        if cmd == "stop_ramp":
            self.stop_ramp_remote()
            return {"result": True}

        raise ValueError(f"unknown cmd: {cmd}")

    def get_tcp_status(self):
        metrics = self._compute_error_metrics()
        with self.rt_lock:
            rt = dict(self.rt)
        with self.params_lock:
            target = float(self.params.get("target", 0.0))
        try:
            display_target = float(self.target_temp.get())
            if display_target == display_target:
                target = float(display_target)
        except Exception:
            pass

        def _safe(value):
            try:
                v = float(value)
            except Exception:
                return None
            if v != v:
                return None
            return v

        status = {
            "timestamp": time.time(),
            "acq_running": bool(self.acq_running),
            "pid_enabled": bool(self.control_enabled),
            "ramp_active": bool(getattr(self, "ramp_active", False)),
            "target": target,
            "temperature": _safe(rt.get("temp")),
            "voltage": _safe(rt.get("v_out")),
            "current": _safe(rt.get("a_out")),
            "power": _safe(rt.get("p_out")),
            "error": _safe(rt.get("err")),
            "mae": metrics.get("mae"),
            "rmse": metrics.get("rmse"),
            "hit_ratio": metrics.get("hit_ratio"),
            "window": metrics.get("window"),
            "samples": metrics.get("samples"),
            "note": rt.get("note"),
        }
        if self._acq_t0 and metrics.get("last_time") is not None:
            status["last_sample_ts"] = self._acq_t0 + metrics["last_time"]
        return status

    def set_target_temperature(self, value, allow_during_ramp=False):
        value = float(value)
        if value > TEMP_LIMIT_CUTOFF:
            raise ValueError(f"目标温度 {value:.2f} 超过安全上限 {TEMP_LIMIT_CUTOFF:.1f}°C")
        if not allow_during_ramp and getattr(self, "ramp_active", False):
            raise RuntimeError("ramp active")
        with self.params_lock:
            self.params["target"] = float(value)
        self._last_valid_target = float(value)
        self._tk_set(self.target_temp, f"{float(value):.3f}")
        return True

    def start_pid_remote(self):
        return self.enable_pid(origin="tcp", notify=False)

    def stop_pid_remote(self):
        self.disable_pid()
        return True

    def start_ramp_remote(self, start, end, rate, hold=0.0, loop=False):
        def runner():
            self.ramp_start.set(f"{float(start):.3f}")
            self.ramp_end.set(f"{float(end):.3f}")
            self.ramp_rate.set(f"{float(rate):.3f}")
            self.ramp_hold_min.set(f"{float(max(0.0, hold)):.3f}")
            self.ramp_cycle_enable.set(bool(loop))
            return self._start_ramp(origin="tcp", notify=False)

        return bool(self._tk_call_sync(runner))

    def stop_ramp_remote(self):
        self._tk_call(self._stop_ramp)
        return True

    # ---------- Tk 参数同步 ----------
    def _sync_params_to_background(self):
        with self.params_lock:
            # 采样率≥0.5 Hz，防止 dt=∞
            self.params["sample_hz"] = self._gf(self.sample_hz, default=self.params.get("sample_hz", 2.0), lo=0.5)

            # 电压限幅≥0；slew≥0
            self.params["v_limit"] = self._gf(self.v_limit, default=self.params.get("v_limit", 28.0), lo=0.0)
            self.params["slew_vps"] = self._gf(self.slew_vps, default=self.params.get("slew_vps", 2.0), lo=0.0)

            # PID 参数：空就保持原值；负值夹成 0
            self.params["kp"] = max(0.0, self._gf(self.kp, self.params.get("kp", 1.0)))
            self.params["ki"] = max(0.0, self._gf(self.ki, self.params.get("ki", 0.0)))
            self.params["kd"] = max(0.0, self._gf(self.kd, self.params.get("kd", 0.0)))

            # 温度滤波常数≥0
            self.params["temp_tau"] = self._gf(self.temp_tau, default=self.params.get("temp_tau", 3.0), lo=0.0)

            # ★ ramp 期间不覆写 target，其余情况空/非法自动回落到上次合法目标
            if not getattr(self, "ramp_active", False):
                t = None
                try:
                    t = float(self.target_temp.get())
                    if t != t:  # NaN
                        t = None
                except Exception:
                    t = None
                if t is None:
                    # 沿用上次合法 or 25℃
                    self._last_valid_target = float(getattr(self, "_last_valid_target", 25.0))
                else:
                    self._last_valid_target = float(t)
                self.params["target"] = self._last_valid_target

    def _snapshot_params(self):
        with self.params_lock:
            return dict(self.params)

    # ---------- 采集线程 ----------
    def _acq_loop(self):
        """
        异步采样 + 同步频率守约：
          - 每帧（1/sample_hz）发 token，等待至本帧截止；
          - 未按时返回则沿用上一帧值，不阻塞节拍；
          - 晚到的数据自动用于下一帧。

        本版增强：
          - 继电器在 PWM↔连续切换、占空比 0↔>0 边界时强制同步；
          - PWM/连续模式各自的心跳重发，修复偶发丢包或设备未响应；
          - 最短开时长（ON→OFF 防抖）；
          - 模式切换的泵电压 & 相位平滑。
        """

        # ==== 小工具：线程安全取数 ====
        def _gbool(v, default=False):
            try:
                return bool(v.get())
            except Exception:
                return bool(default)

        def _gfloat(v, default):
            try:
                x = float(v.get())
                return x if x == x else float(default)
            except Exception:
                return float(default)

        # ==== 继电器小工具：应用状态（支持强制/心跳、最短ON保持） ====
        def _relay_apply(want_closed: bool, *, force: bool = False):
            """
            want_closed: 希望泵侧“回路闭合”（语义层）
            force: True 无视缓存强制下发
            """
            if not getattr(self.relay, "connected", False):
                self._tk_set(self.relay_pwm_state, "— (未连)")
                return

            # 你的接法是反相：逻辑闭合==物理 OFF
            cmd_on = (not want_closed) if RELAY_INVERT else want_closed
            now = time.perf_counter()

            # 初始化运行态
            if not hasattr(self, "_relay_cmd_on"):            self._relay_cmd_on = None
            if not hasattr(self, "_relay_last_switch_ts"):     self._relay_last_switch_ts = 0.0

            # 最短开时长（仅限制 ON->OFF）
            min_on_ms = self._gi(getattr(self, "pump_pwm_min_on_ms", 120), 120, lo=20, hi=1200)
            if (self._relay_cmd_on is True) and (cmd_on is False):
                if (now - self._relay_last_switch_ts) * 1000.0 < float(min_on_ms):
                    return  # 未到最短开时长

            # 相同状态且非强制：跳过
            if (not force) and (self._relay_cmd_on is not None) and (cmd_on == self._relay_cmd_on):
                return

            try:
                ch = int(self.relay_chan.get() or 1)
                self.relay.set(ch, cmd_on, feedback=False)
                self._relay_cmd_on = cmd_on
                self._relay_last_switch_ts = now
            except Exception as e:
                self._set_status(f"写继电器失败: {e}")

        # ==== 初值 ====
        params = self._snapshot_params()
        sample_hz = max(0.5, float(params["sample_hz"]))
        dt = 1.0 / sample_hz

        v_limit = float(params["v_limit"])
        slew = float(params["slew_vps"])
        temp_tau = max(0.0, float(params["temp_tau"]))
        temp_filt = None

        self.last_vset = 0.0
        self.overtemp_shutdown = False
        t_loop0 = self._acq_t0 or time.perf_counter()

        # 丢温度计数（用于安全关断）
        no_temp_count = 0
        max_no_temp = max(1, int(2.0 / dt))  # ≥2s 未获得新温度就关加热

        # 泵运行态（兜底）
        if not hasattr(self, "pump_last_vset"):
            self.pump_last_vset = float(self.pump_vmin.get() or 13.0)

        # 上一帧观测（用于超时沿用）
        last_temp_raw = float('nan')
        last_psu = (0.0, 0.0, 0)  # (V, A, CVCC)
        last_pump = (0.0, 0.0, 0.0)

        # 模式/相位/心跳初始化
        RELAY_INVERT = True
        if not hasattr(self, "_relay_epoch"):          self._relay_epoch = time.perf_counter()
        if not hasattr(self, "_relay_last_refresh"):   self._relay_last_refresh = 0.0
        if not hasattr(self, "_prev_duty_positive"):   self._prev_duty_positive = None
        if not hasattr(self, "_pwm_mode"):             self._pwm_mode = False
        if not hasattr(self, "_duty_filt"):            self._duty_filt = 0.0
        if not hasattr(self, "_last_veq"):             self._last_veq = 0.0

        pump_pid_reset_needed = False
        prev_pump_mode = self._pwm_mode

        token = 0
        next_deadline = time.perf_counter()

        while not self.stop_event.is_set():
            # ==== 动态读取参数（每秒刷一次） ====
            cycle_ts = time.perf_counter()
            t_rel = cycle_ts - t_loop0
            if int(t_rel) != int(getattr(self, "_last_dyn_upd_t", -1)):
                self._last_dyn_upd_t = t_rel
                params = self._snapshot_params()

                sample_hz = max(0.5, float(params["sample_hz"]))
                dt = 1.0 / sample_hz
                max_no_temp = max(1, int(2.0 / dt))

                v_limit = float(params["v_limit"])
                slew = float(params["slew_vps"])
                temp_tau = max(0.0, float(params["temp_tau"]))
                ctrl_dir = float(params.get("ctrl_dir", 1.0))

                # 加热 PID
                if hasattr(self, "pid") and self.pid:
                    self.pid.kp = self._gf(self.kp, self.pid.kp, lo=0.0)
                    self.pid.ki = self._gf(self.ki, self.pid.ki, lo=0.0)
                    self.pid.kd = self._gf(self.kd, self.pid.kd, lo=0.0)
                    self.pid.u_max = v_limit

                # 泵参数
                try:
                    vmin = self._gf(self.pump_vmin, 13.0, lo=0.0)
                    vmax = self._gf(self.pump_vmax, 26.0, lo=vmin)
                    pslew = self._gf(self.pump_slew_vps, 2.5, lo=0.0)
                    deadband = self._gf(self.pump_deadband, 0.30, lo=0.0)
                except Exception:
                    vmin, vmax, pslew, deadband = 13.0, 26.0, 2.5, 0.30

                # 冷却 PID
                if hasattr(self, "pid_cool") and self.pid_cool:
                    self.pid_cool.kp = self._gf(self.kp_cool, self.pid_cool.kp, lo=0.0)
                    self.pid_cool.ki = self._gf(self.ki_cool, self.pid_cool.ki, lo=0.0)
                    self.pid_cool.kd = self._gf(self.kd_cool, self.pid_cool.kd, lo=0.0)
                    self.pid_cool.u_min = 0.0
                    self.pid_cool.u_max = max(0.0, vmax - vmin)
            else:
                ctrl_dir = float(params.get("ctrl_dir", 1.0))
                try:
                    vmin = self._gf(self.pump_vmin, 13.0, lo=0.0)
                    vmax = self._gf(self.pump_vmax, 26.0, lo=vmin)
                    pslew = self._gf(self.pump_slew_vps, 2.5, lo=0.0)
                    deadband = self._gf(self.pump_deadband, 0.30, lo=0.0)
                except Exception:
                    vmin, vmax, pslew, deadband = 13.0, 26.0, 2.5, 0.30

            pump_connected = hasattr(self, "pump") and getattr(self.pump, "connected", False)

            # ==== 发令牌 ====
            token = self._next_tok()
            if getattr(self, "_sampler_tcm", None) is not None:   self._sampler_tcm.request(token)
            if getattr(self, "_sampler_psu", None) is not None:   self._sampler_psu.request(token)
            if pump_connected and getattr(self, "_sampler_pump", None) is not None:
                self._sampler_pump.request(token)

            frame_deadline = cycle_ts + dt

            # 等待（或超时继续）
            while time.perf_counter() < frame_deadline:
                ready_t = (getattr(self, "_sampler_tcm", None) is None) or (self._sampler_tcm.latest()[3] >= token)
                ready_p = (getattr(self, "_sampler_psu", None) is None) or (self._sampler_psu.latest()[3] >= token)
                ready_u = True
                if pump_connected and getattr(self, "_sampler_pump", None) is not None:
                    ready_u = self._sampler_pump.latest()[3] >= token
                if ready_t and ready_p and ready_u:
                    break
                time.sleep(0.001)

            # ==== 取数（未就绪沿用上一帧） ====
            # 温度
            temp_raw = last_temp_raw
            if getattr(self, "_sampler_tcm", None):
                val_t, _, ok_t, done_tok = self._sampler_tcm.latest()
                if done_tok == token and ok_t:
                    temp_raw = val_t
                    last_temp_raw = temp_raw
                    no_temp_count = 0
                else:
                    no_temp_count += 1
            else:
                temp_raw = float('nan')
                no_temp_count += 1

            # 电源
            v_out, a_out, cvcc = last_psu
            if getattr(self, "_sampler_psu", None):
                val_p, _, ok_p, done_tok = self._sampler_psu.latest()
                if done_tok >= token and ok_p and isinstance(val_p, tuple) and len(val_p) == 3:
                    v_out, a_out, cvcc = val_p
                    last_psu = (v_out, a_out, cvcc)

            # 循环泵
            pump_v, pump_i, pump_p = last_pump
            if pump_connected and getattr(self, "_sampler_pump", None):
                val_u, _, ok_u, done_tok = self._sampler_pump.latest()
                if done_tok >= token and ok_u and isinstance(val_u, tuple) and len(val_u) == 3:
                    pump_v, pump_i, pump_p = val_u
                    last_pump = (pump_v, pump_i, pump_p)

            # ==== 温度滤波 ====
            if temp_raw == temp_raw:  # 非 NaN
                if temp_tau <= 1e-6 or temp_filt is None:
                    temp_filt = float(temp_raw)
                else:
                    alpha = max(0.0, min(1.0, dt / temp_tau))
                    temp_filt += alpha * (float(temp_raw) - temp_filt)

            # 误差
            note = ""
            err_phys = float('nan')
            if temp_filt is not None and temp_filt == temp_filt:
                err_phys = (params["target"] - temp_filt)

            # ==== 失联关断 ====
            if self.control_enabled and no_temp_count >= max_no_temp:
                try:
                    self.psu.set_output(False)
                    note = (note + "; " if note else "") + "sensor_lost_shutdown"
                    self.control_enabled = False
                    self.pid.reset()
                    self._set_status("温度新数据超时>2s，已关闭加热输出")
                except Exception as e:
                    self._set_status(f"温度丢失关机失败: {e}")

            # ==== 超温保护 ====
            if self.control_enabled and (temp_filt is not None) and (temp_filt == temp_filt):
                if temp_filt >= TEMP_LIMIT_CUTOFF and not self.overtemp_shutdown:
                    try:
                        self.psu.set_output(False)
                        self.overtemp_shutdown = True
                        note = (note + "; " if note else "") + "overtemp_shutdown"
                        self.pid.reset()
                    except Exception as e:
                        self._set_status(f"超温关机失败: {e}")
                if temp_filt <= TEMP_LIMIT_RESUME and self.overtemp_shutdown:
                    try:
                        self.psu.set_output(True)
                        self.overtemp_shutdown = False
                        note = (note + "; " if note else "") + "overtemp_resume"
                        self.pid.reset()
                    except Exception as e:
                        self._set_status(f"恢复输出失败: {e}")

            # ==== 加热侧 PID ====
            if self.control_enabled and (temp_filt is not None) and (temp_filt == temp_filt) and (
            not self.overtemp_shutdown):
                sp_eff = params["target"] * ctrl_dir
                meas_eff = temp_filt * ctrl_dir
                at_upper = self.last_vset >= v_limit - 1e-6
                v_pid_eff, _ = self.pid.step(sp_eff, meas_eff, dt, cc_mode=(cvcc == 1), at_upper_limit=at_upper)
                dv_max = slew * dt
                dv = max(-dv_max, min(dv_max, v_pid_eff - self.last_vset))
                v_set_cmd = max(0.0, min(v_limit, self.last_vset + dv))
                if abs(v_set_cmd - self.last_vset) > 0.01:
                    try:
                        self.psu.set_voltage(v_set_cmd)
                        self.last_vset = v_set_cmd
                    except Exception as e:
                        self._set_status(f"写加热电压失败: {e}")

            # ==== 循环泵（自动 PWM：PID -> duty；V_eq = duty*V_on）====
            # 先确保泵上电
            if pump_connected:
                try:
                    self.pump.set_output(True)
                except Exception:
                    pass

                # 取泵参数
                try:
                    vmin = self._gf(self.pump_vmin, 13.0, lo=0.0)
                    vmax = self._gf(self.pump_vmax, 26.0, lo=vmin)
                    pslew = self._gf(self.pump_slew_vps, 2.5, lo=0.0)
                    deadband = self._gf(self.pump_deadband, 0.30, lo=0.0)
                except Exception:
                    vmin, vmax, pslew, deadband = 13.0, 26.0, 2.5, 0.30

                span = max(1e-6, (vmax - vmin))
                temp_ok = (temp_filt is not None) and (temp_filt == temp_filt)

                # 由误差→冷却 PID → duty_raw
                if (not self.control_enabled) or (not temp_ok):
                    self.pid_cool.reset()
                    duty_raw = 0.0
                else:
                    hot_err = (temp_filt - params["target"])  # >0 偏热 → 需要冷
                    if hot_err <= deadband:
                        self.pid_cool.reset()
                        duty_raw = 0.0
                    else:
                        e_eff = hot_err - deadband
                        at_upper = (self.pump_last_vset >= (vmax - 1e-6))
                        u_rel, _ = self.pid_cool.step(0.0, -e_eff, dt, cc_mode=False, at_upper_limit=at_upper)
                        u_rel = max(0.0, min(self.pid_cool.u_max, u_rel))  # 0 ~ (vmax-vmin)
                        duty_raw = max(0.0, min(1.0, u_rel / span))

                # duty 平滑（限速）
                tau_s = 0.6
                alpha = max(0.0, min(1.0, dt / max(1e-6, tau_s)))
                duty_lp = self._duty_filt + alpha * (duty_raw - self._duty_filt)
                duty_eff = self._duty_filt + max(-0.06, min(0.06, duty_lp - self._duty_filt))
                duty_eff = max(0.0, min(1.0, duty_eff))
                self._duty_filt = duty_eff

                # 模式判定：手动 PWM 优先，其次自动 PWM（带回差）
                manual_pwm = _gbool(getattr(self, "pump_pwm_enable", False), False)
                auto_pwm = _gbool(getattr(self, "pump_pwm_auto_enable", True), True)
                hi_thr = _gfloat(getattr(self, "pump_pwm_hi_temp_thr", 80.0), 80.0)
                hyst = 2.0

                prev_pwm_mode = self._pwm_mode
                if manual_pwm:
                    self._pwm_mode = True
                else:
                    if not self._pwm_mode:
                        self._pwm_mode = bool(auto_pwm and temp_ok and (temp_filt >= hi_thr - 1e-9))
                    else:
                        self._pwm_mode = not (temp_ok and (temp_filt <= hi_thr - hyst + 1e-9))

                # 占空比下限
                min_duty_pct = self._gf(self.pump_pwm_min_duty, 5.0, lo=0.0, hi=100.0)
                min_duty = max(0.0, min(1.0, min_duty_pct / 100.0))

                # PWM 公共参数
                pwm_hz = max(0.2, min(20.0, _gfloat(getattr(self, "pump_pwm_freq", 1.0), 1.0)))
                period = 1.0 / pwm_hz
                V_on_user = _gfloat(getattr(self, "pump_pwm_v_on", vmax), vmax)
                V_on = max(vmin, min(vmax, V_on_user if V_on_user > 0 else vmax))

                # ==== 模式切换：做平滑过渡 & 强制继电器同步 ====
                mode_changed = (prev_pwm_mode != self._pwm_mode)
                if mode_changed:
                    if prev_pwm_mode and not self._pwm_mode:
                        # PWM -> 连续：以上次等效电压为起点
                        V_start = max(vmin, min(vmax, self._last_veq))
                        try:
                            self.pump.set_voltage(V_start)
                            self.pump_last_vset = V_start
                        except Exception as e:
                            self._set_status(f"切换模式时设置泵电压失败: {e}")
                        # 继电器立刻闭合一次（强制）
                        _relay_apply(True, force=True)
                        # 冷却 PID 重置
                        self.pid_cool.reset()
                        note = (note + "; " if note else "") + "mode_switch:pwm->cont"

                    elif (not prev_pwm_mode) and self._pwm_mode:
                        # 连续 -> PWM：以当前 vset 映射 duty 起步，重置相位为“现在”
                        current_v = self.pump_last_vset
                        duty_cmd = max(0.0, min(1.0, (current_v - vmin) / span))
                        self._duty_filt = duty_cmd
                        self._relay_epoch = time.perf_counter()
                        # 先把泵电压锁到 V_on（关由继电器实现）
                        try:
                            self.pump.set_voltage(V_on)
                            self.pump_last_vset = V_on
                        except Exception as e:
                            self._set_status(f"切换模式时设置泵电压失败: {e}")
                        # 第一拍强制闭合，防止卡错相位
                        _relay_apply(True, force=True)
                        note = (note + "; " if note else "") + "mode_switch:cont->pwm"

                # ==== 等效 Vset 对外统一发布 ====
                veq_to_publish = None

                # ---------- PWM 分支 ----------
                if self._pwm_mode:
                    duty_calc = max(0.0, min(1.0, duty_eff))  # 显示用
                    duty_use = duty_calc if (duty_calc >= min_duty) else 0.0  # 动作用（<min则不闭合）
                    Ton = duty_use * period
                    V_eq = duty_use * V_on
                    veq_to_publish = V_eq

                    # 电源锁到 V_on（即使 duty_use=0 也锁住；开关交给继电器）
                    if abs(self.pump_last_vset - V_on) > 0.05:
                        try:
                            self.pump.set_voltage(V_on)
                            self.pump_last_vset = V_on
                        except Exception as e:
                            self._set_status(f"PWM模式设泵电压失败: {e}")

                    # 相位与继电器
                    if getattr(self.relay, "connected", False):
                        now_ts = time.perf_counter()
                        if self._relay_epoch is None:
                            self._relay_epoch = now_ts
                        phase = (now_ts - self._relay_epoch) % period
                        want_closed = (phase < Ton) and (duty_use > 0.0)
                        # 正常随相位切
                        _relay_apply(want_closed, force=False)

                        # 0↔>0 边沿强制一次
                        duty_pos = (duty_use > 0.0)
                        if (self._prev_duty_positive is not None) and (duty_pos != self._prev_duty_positive):
                            _relay_apply(want_closed, force=True)
                        self._prev_duty_positive = duty_pos

                        # 心跳：每1s 强制重发
                        if now_ts - self._relay_last_refresh >= 1.0:
                            _relay_apply(want_closed, force=True)
                            self._relay_last_refresh = now_ts

                        self._tk_set(self.relay_pwm_state, "ON" if want_closed else "OFF")
                    else:
                        self._tk_set(self.relay_pwm_state, "— (未连)")

                    # UI 展示
                    def _fmt_pct(x):
                        return (f"{x * 100:.3f}%" if x < 0.01 else f"{x * 100:.1f}%")

                    disp_txt = _fmt_pct(duty_calc)
                    if duty_calc < min_duty:
                        disp_txt = f"{disp_txt} (<min {min_duty * 100:.1f}%)"

                    self._tk_set(self.pwm_duty, round(duty_calc * 100.0, 3))
                    self._tk_set(self.pwm_duty_str, _fmt_pct(duty_calc))
                    self._tk_set(self.relay_pwm_duty, disp_txt)

                    note = (note + "; " if note else "") + (
                        f"pwm(T={period:.2f}s,Ton={Ton:.4f}s,V_on={V_on:.2f}V,"
                        f"duty_calc={duty_calc:.4f},duty_use={duty_use:.4f},Veq={V_eq:.3f}V)"
                    )

                # ---------- 非 PWM（连续） ----------
                else:
                    # 继电器保持闭合；心跳2s强制一次
                    _relay_apply(True, force=mode_changed)  # 刚切换时强制闭合
                    now_ts = time.perf_counter()
                    if now_ts - self._relay_last_refresh >= 2.0:
                        _relay_apply(True, force=True)
                        self._relay_last_refresh = now_ts

                    # 连续电压 = vmin + duty_eff * span，带泵斜率限制
                    V_eq_cont = vmin + duty_eff * span
                    dv_max_pump = max(0.0, float(pslew)) * dt
                    v_cmd = self.pump_last_vset + max(-dv_max_pump, min(dv_max_pump, V_eq_cont - self.pump_last_vset))
                    v_cmd = max(vmin, min(vmax, v_cmd))
                    if abs(v_cmd - self.pump_last_vset) > 0.02:
                        try:
                            self.pump.set_voltage(v_cmd)
                            self.pump_last_vset = v_cmd
                        except Exception as e:
                            self._set_status(f"写循环泵电压失败: {e}")

                    veq_to_publish = v_cmd
                    self._tk_set(self.relay_pwm_state, "ON")
                    self._tk_set(self.pwm_duty, 0.0)
                    self._tk_set(self.pwm_duty_str, "—")
                    self._tk_set(self.relay_pwm_duty, "—")

                # 记录等效电压（兜底）
                try:
                    self._last_veq = veq_to_publish if (veq_to_publish is not None) \
                        else (self.pump_last_vset if not self._pwm_mode else duty_eff * V_on)
                except Exception:
                    self._last_veq = self.pump_last_vset

            # ==== 实时共享/缓冲/CSV ====
            p_out = v_out * a_out
            pvset_pub = float(
                (self._last_veq if hasattr(self, "_last_veq") else getattr(self, "pump_last_vset", 0.0))
            )

            with self.rt_lock:
                self.rt.update({
                    "temp": 0.0 if temp_filt is None or not (temp_filt == temp_filt) else float(temp_filt),
                    "v_out": float(v_out),
                    "a_out": float(a_out),
                    "p_out": float(p_out),
                    "cvcc": int(cvcc),
                    "err": float('nan') if not (err_phys == err_phys) else float(err_phys),
                    "note": note,
                    "pump_v": float(pump_v),
                    "pump_a": float(pump_i),
                    "pump_p": float(pump_p),
                    "pump_vset": pvset_pub,  # 统一的“等效 Vset”
                })

            with self.history_lock:
                self.t_buf.append(t_rel)
                self.temp_buf.append(None if (temp_filt is None or not (temp_filt == temp_filt)) else float(temp_filt))
                self.power_buf.append(p_out)
                with self.params_lock:
                    cur_target = float(self.params["target"])
                self.target_buf.append(cur_target)
                self.err_buf.append(None if not (err_phys == err_phys) else float(err_phys))
                if hasattr(self, "pump_v_buf"): self.pump_v_buf.append(pump_v if pump_connected else None)
                if hasattr(self, "pump_i_buf"): self.pump_i_buf.append(pump_i if pump_connected else None)
                if hasattr(self, "pump_p_buf"): self.pump_p_buf.append(pump_p if pump_connected else None)

            if self.csv_writer:
                try:
                    self.csv_writer.writerow([
                        f"{t_rel:.3f}",
                        "" if (temp_filt is None or not (temp_filt == temp_filt)) else f"{temp_filt:.3f}",
                        f"{cur_target:.3f}",
                        f"{self.last_vset:.4f}",
                        f"{v_out:.4f}",
                        f"{a_out:.4f}",
                        f"{p_out:.4f}",
                        f"{self.pid.kp:.4f}",
                        f"{self.pid.ki:.4f}",
                        f"{self.pid.kd:.4f}",
                        "" if not (err_phys == err_phys) else f"{err_phys:.4f}",
                        "CC" if cvcc == 1 else "CV",
                        note + (f"; pumpV={self.pump_last_vset:.2f},pumpP={pump_p:.2f}W" if pump_connected else "")
                    ])
                except Exception:
                    pass

            # ==== 对齐节拍 ====
            next_deadline += dt
            sleep_time = next_deadline - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_deadline = time.perf_counter()

            # 记录本帧耗时（可用于Hz显示）
            try:
                if hasattr(self, "cycle_times"):
                    self.cycle_times.append(time.perf_counter() - cycle_ts)
            except Exception:
                pass

    # ---------- GUI 定时刷新 ----------
    def _schedule_gui_update(self):
        update_start = time.perf_counter()

        # 同步一次可修改参数（保持与原逻辑一致）
        self._sync_params_to_background()
        self._frame_count += 1

        # 频率策略：仪表每帧更新，绘图 3 帧一更（约 3~5 Hz，取决于 ui_refresh_ms）
        dash_every = 1
        plot_every = 3

        if self._frame_count % dash_every == 0:
            self._update_indicators()
            self._update_dashboard()
        if self._frame_count % plot_every == 0:
            self._update_plot()

        # 适度提高 GUI 刷新频率下限（40ms），更跟手
        self.root.after(max(40, int(self.ui_refresh_ms.get() or 80)), self._schedule_gui_update)

        # 记录一次 GUI 更新耗时
        self.gui_update_times.append(time.perf_counter() - update_start)

    def _update_indicators(self):
        def paint(canvas, status_text):
            canvas.delete("all")
            if "已连接" in status_text:
                color = "#2ecc71"
            elif "连接失败" in status_text:
                color = "#e67e22"
            else:
                color = "#e74c3c"
            canvas.create_oval(2, 2, 14, 14, fill=color, width=0)

        paint(self.psu_indicator, self.psu_status.get())
        paint(self.tcm_indicator, self.tcm_status.get())
        paint(self.pump_indicator, self.pump_status.get())
        paint(self.relay_indicator, self.relay_status.get())

    def _update_dashboard(self):
        # 基本读数（三位小数格式化）
        with self.rt_lock:
            rt = dict(self.rt)

        def fmt3(x, dash="—"):
            try:
                if x is None: return dash
                if isinstance(x, float) and isnan(x): return dash
                return f"{float(x):.3f}"
            except Exception:
                return dash

        self.cur_temp.set(fmt3(rt.get("temp")))
        self.cur_vout.set(fmt3(rt.get("v_out")))
        self.cur_aout.set(fmt3(rt.get("a_out")))
        self.cur_pout.set(fmt3(rt.get("p_out")))
        self.cvcc_state.set("CC" if rt.get("cvcc", 0) == 1 else "CV")

        # 新增：循环泵四项
        self.cur_pump_v.set(fmt3(rt.get("pump_v")))
        self.cur_pump_a.set(fmt3(rt.get("pump_a")))
        self.cur_pump_p.set(fmt3(rt.get("pump_p")))
        self.cur_pump_vset.set(fmt3(rt.get("pump_vset")))

        # —— 实时精度计算 ——
        win = safe_float(self.acc_window_s.get(), 30.0) or 30.0
        tol = max(0.0, safe_float(self.acc_tol.get(), 0.5) or 0.5)
        self.acc_info.set(f"窗={win:.0f} s，容差=±{tol:.2f} °C")
        self.cur_err.set(fmt3(rt.get("err"), dash="0.000"))

        with self.history_lock:
            t_list = list(self.t_buf)
            e_list = list(self.err_buf)
        if not t_list:
            self.mae_win.set("0.000");
            self.rmse_win.set("0.000");
            self.hit_ratio.set("0.000")
            return

        t_now = t_list[-1];
        t_min = t_now - win
        i0 = 0
        for i in range(len(t_list) - 1, -1, -1):
            if t_list[i] < t_min:
                i0 = i + 1
                break
        ew = [e for e in e_list[i0:] if (e is not None and not isnan(e))]
        if len(ew) == 0:
            self.mae_win.set("0.000");
            self.rmse_win.set("0.000");
            self.hit_ratio.set("0.000")
        else:
            mae = sum(abs(x) for x in ew) / len(ew)
            rmse = sqrt(sum(x * x for x in ew) / len(ew))
            hit = sum(1 for x in ew if abs(x) <= tol) / len(ew) * 100.0
            self.mae_win.set(f"{mae:.3f}")
            self.rmse_win.set(f"{rmse:.3f}")
            self.hit_ratio.set(f"{hit:.3f}")

    def _compute_error_metrics(self, window_s=None):
        if window_s is None:
            try:
                window_s = max(1.0, safe_float(self.acc_window_s.get(), 30.0) or 30.0)
            except Exception:
                window_s = 30.0

        with self.history_lock:
            t_list = list(self.t_buf)
            e_list = list(self.err_buf)

        if not t_list:
            return {
                "window": float(window_s),
                "samples": 0,
                "mae": None,
                "rmse": None,
                "hit_ratio": None,
                "last_time": None,
            }

        t_now = t_list[-1]
        t_min = t_now - window_s
        errors = [e for t, e in zip(t_list, e_list) if t >= t_min and e is not None and not isnan(e)]
        if not errors:
            return {
                "window": float(window_s),
                "samples": 0,
                "mae": None,
                "rmse": None,
                "hit_ratio": None,
                "last_time": t_now,
            }

        mae = sum(abs(x) for x in errors) / len(errors)
        rmse = sqrt(sum(x * x for x in errors) / len(errors))
        try:
            tol = max(0.0, safe_float(self.acc_tol.get(), 0.5) or 0.5)
        except Exception:
            tol = 0.5
        hit = sum(1 for x in errors if abs(x) <= tol) / len(errors) * 100.0
        return {
            "window": float(window_s),
            "samples": len(errors),
            "mae": mae,
            "rmse": rmse,
            "hit_ratio": hit,
            "last_time": t_now,
        }

    def _update_plot(self):
        try:
            now = time.perf_counter()
            # 仍然做个限频，避免过密重绘（这里放宽到 >= 120ms 才重绘一帧）
            if now - getattr(self, "_last_plot_update", 0.0) < 0.12:
                return
            self._last_plot_update = now

            # —— 小工具 —— #
            def _smooth_ma(seq, k=5):
                if k <= 1:
                    return list(seq)
                from collections import deque
                out, dq = [], deque(maxlen=k)
                for v in seq:
                    if v is None or (isinstance(v, float) and isnan(v)):
                        out.append(None)
                    else:
                        dq.append(float(v))
                        out.append(sum(dq) / len(dq))
                return out

            def _robust_ylim(vals, pad_frac=0.08, pad_abs=0.30, min_span=0.50, floor0=False):
                vv = [float(v) for v in vals if (v is not None and not isnan(v))]
                if not vv:
                    return None
                vv.sort()
                n = len(vv)
                lo = vv[int(0.02 * (n - 1))]
                hi = vv[int(0.98 * (n - 1))]
                if hi - lo < 1e-12:
                    lo -= pad_abs;
                    hi += pad_abs
                else:
                    pad = max(pad_abs, (hi - lo) * pad_frac)
                    lo -= pad;
                    hi += pad
                if hi - lo < min_span:
                    mid = 0.5 * (hi + lo)
                    half = 0.5 * min_span
                    lo, hi = mid - half, mid + half
                if floor0:
                    lo = max(0.0, lo)
                    if hi - lo < min_span:
                        hi = lo + min_span
                return (lo, hi)

            # —— 无数据 —— #
            if len(self.t_buf) < 1:
                self.canvas.draw_idle()
                return

            # 展示窗口（秒）
            win_s = max(10.0, safe_float(self.plot_window_s.get(), 120.0) or 120.0)

            # 缓冲 -> 列表
            with self.history_lock:
                t_all = list(self.t_buf)
            yT_all = list(self.temp_buf)
            yHeP_all = list(self.power_buf)
            yTar_all = list(self.target_buf)
            yPumpP_all = list(getattr(self, "pump_p_buf", []))

            # 时间对齐到“当前”
            # —— 使用采集起点为零点的时间轴（单调递增，不会被窗口移动“重置”）——
            x = list(t_all)  # t_buf 里存的就是从采集起点累计的相对秒 t_rel

            if self.acq_running and self._acq_t0 is not None and x:
                # 让线条“跟手”：若当前时刻已超过最后一个样本，则补一个“到现在”的点
                x_now_rel = (now - self._acq_t0)
                if x_now_rel > x[-1] + 1e-9:
                    x_disp = x + [x_now_rel]
                    yT_disp = yT_all + [yT_all[-1]]
                    yHeP_disp = yHeP_all + [yHeP_all[-1]]
                    yTar_disp = yTar_all + [yTar_all[-1]]
                    if yPumpP_all:
                        yPumpP_all = yPumpP_all + [yPumpP_all[-1]]
                else:
                    x_disp, yT_disp, yHeP_disp, yTar_disp = x, yT_all, yHeP_all, yTar_all
            else:
                x_disp, yT_disp, yHeP_disp, yTar_disp = x, yT_all, yHeP_all, yTar_all

            # 截窗口
            xmin_win = max(0.0, (x_disp[-1] if x_disp else 0.0) - win_s)
            i0 = 0
            for i in range(len(x_disp) - 1, -1, -1):
                if x_disp[i] < xmin_win:
                    i0 = i + 1
                    break
            xw = x_disp[i0:];
            yTw = yT_disp[i0:];
            yHePw = yHeP_disp[i0:];
            yTarw = yTar_disp[i0:]
            pumpPw_raw = yPumpP_all[i0:] if yPumpP_all else []

            # 顶部：温度/目标 + 最近窗口线性拟合
            self.line_temp.set_data(xw, yTw)
            self.line_target.set_data(xw, yTarw)

            fit_win = max(5.0, safe_float(self.fit_window_s.get(), 40.0) or 40.0)
            if xw:
                t_now_rel = xw[-1]
                t_min = t_now_rel - fit_win
                j0 = 0
                for j in range(len(xw) - 1, -1, -1):
                    if xw[j] < t_min:
                        j0 = j + 1
                        break
                xs, ys = [], []
                for tx, ty in zip(xw[j0:], yTw[j0:]):
                    if ty is not None and not isnan(ty):
                        xs.append(tx);
                        ys.append(ty)
                if len(xs) >= 2:
                    n = float(len(xs))
                    sx = sum(xs);
                    sy = sum(ys)
                    sxx = sum(xx * xx for xx in xs)
                    sxy = sum(xx * yy for xx, yy in zip(xs, ys))
                    denom = n * sxx - sx * sx
                    if abs(denom) > 1e-9:
                        a = (n * sxy - sx * sy) / denom  # °C/s
                        b = (sy - a * sx) / n
                        xfit = [xs[0], xs[-1]]
                        yfit = [a * xfit[0] + b, a * xfit[1] + b]
                        self.line_fit.set_data(xfit, yfit)
                        try:
                            self.fit_slope.set(f"{a * 60.0:.3f}")  # °C/min
                        except Exception:
                            pass
                    else:
                        self.line_fit.set_data([], [])
                        try:
                            self.fit_slope.set("0.000")
                        except Exception:
                            pass

            # 左下：加热功率
            self.line_power.set_data(xw, yHePw)

            # 右下：泵功率（≥0，轻度平滑）
            pumpPw = []
            for v in pumpPw_raw:
                if v is None or (isinstance(v, float) and isnan(v)):
                    pumpPw.append(None)
                else:
                    vv = float(v)
                    if vv < 0.0 or vv > 1000.0:
                        pumpPw.append(None)
                    else:
                        pumpPw.append(vv)
            pumpPw = _smooth_ma(pumpPw, k=5)

            n = min(len(xw), len(pumpPw))
            if n > 0:
                self.line_pump.set_data(xw[:n], pumpPw[:n])
                self.line_pump.set_visible(True)
            else:
                self.line_pump.set_data([], [])
                self.line_pump.set_visible(True)

            # 统一 X 轴范围
            xmax = xw[-1] if xw else win_s
            xmin = max(0.0, xmax - win_s)
            self.ax_temp.set_xlim(xmin, xmax)
            self.ax_power.set_xlim(xmin, xmax)
            self.ax_pump.set_xlim(xmin, xmax)
            if hasattr(self, "ax_pump2") and self.ax_pump2:
                try:
                    self.ax_pump2.set_xlim(xmin, xmax)
                    self.ax_pump2.set_ylabel("")
                    self.ax_pump2.get_yaxis().set_visible(False)
                except Exception:
                    pass

            # —— Y 轴范围计算“限频”：最多 0.30s 改一次 ylim —— #
            if now - getattr(self, "_last_ylim_upd", 0.0) > 0.30:
                # 顶部：温度与目标线
                vals_t = []
                vals_t += [v for v in yTw if (v is not None and not isnan(v))]
                vals_t += [v for v in yTarw if (v is not None and not isnan(v))]
                r = _robust_ylim(vals_t, pad_frac=0.10, pad_abs=0.30, min_span=0.80, floor0=False)
                if r:
                    self.ax_temp.set_ylim(*r)

                # 左下：加热功率（≥0）
                r = _robust_ylim([v for v in yHePw if (v is not None and not isnan(v))],
                                 pad_frac=0.12, pad_abs=1.0, min_span=5.0, floor0=True)
                if r:
                    self.ax_power.set_ylim(*r)

                # 右下：泵功率（≥0）
                r = _robust_ylim(pumpPw, pad_frac=0.12, pad_abs=1.0, min_span=5.0, floor0=True)
                if r:
                    self.ax_pump.set_ylim(*r)

                self._last_ylim_upd = now

            try:
                self.ax_pump.legend(loc="upper right", fontsize=8)
            except Exception:
                pass

            self.canvas.draw_idle()

        except Exception as e:
            try:
                self._set_status(f"绘图异常：{e}")
            except Exception:
                pass

    def _clear_plot(self):
        with self.history_lock:
            self.t_buf.clear();
            self.temp_buf.clear();
            self.power_buf.clear()
            self.err_buf.clear();
            self.target_buf.clear()
            if hasattr(self, "pump_v_buf"): self.pump_v_buf.clear()
            if hasattr(self, "pump_i_buf"): self.pump_i_buf.clear()
            if hasattr(self, "pump_p_buf"): self.pump_p_buf.clear()

        self.line_temp.set_data([], [])
        self.line_power.set_data([], [])
        self.line_target.set_data([], [])
        self.line_fit.set_data([], [])
        if hasattr(self, "line_pump_v"): self.line_pump_v.set_data([], [])
        if hasattr(self, "line_pump_i"): self.line_pump_i.set_data([], [])
        if hasattr(self, "line_pump_p"): self.line_pump_p.set_data([], [])

        self.ax_temp.relim();
        self.ax_temp.autoscale_view()
        self.ax_power.relim();
        self.ax_power.autoscale_view()
        if hasattr(self, "ax_pump"): self.ax_pump.relim(); self.ax_pump.autoscale_view()
        if hasattr(self, "ax_pump2"): self.ax_pump2.relim(); self.ax_pump2.autoscale_view()
        self.canvas.draw_idle()

    # ---------- 配置保存/读取 ----------
    def _save_config(self):
        def s2i(v, default=0):
            try:
                s = v.get() if hasattr(v, "get") else v
            except Exception:
                s = v
            s = "" if s is None else str(s).strip()
            if s == "":
                return int(default)
            try:
                return int(float(s))
            except Exception:
                return int(default)

        def sval(v):
            try:
                s = v.get() if hasattr(v, "get") else v
            except Exception:
                s = v
            s = "" if s is None else str(s)
            return s.strip()

        cfg = {
            "psu_port": sval(self.psu_port),
            "psu_baud": s2i(self.psu_baud, PSU_BAUD),
            "psu_addr": s2i(self.addr_psu, PSU_ADDR_DEFAULT),
            "tcm_port": sval(self.tcm_port),
            "tcm_baud": s2i(self.tcm_baud, TCM_BAUD),
            "tcm_addr": s2i(self.addr_tcm, TCM_ADDR_DEFAULT),
            "relay_port": sval(self.relay_port),
            "relay_baud": s2i(self.relay_baud, 9600),
            "relay_chan": s2i(self.relay_chan, 1),

            "kp": self.kp.get(), "ki": self.ki.get(), "kd": self.kd.get(),
            "sample_hz": self.sample_hz.get(),
            "v_limit": self.v_limit.get(),
            "slew_vps": self.slew_vps.get(),
            "plot_window_s": self.plot_window_s.get(),
            "ui_refresh_ms": self.ui_refresh_ms.get(),
            "target_temp_str": self.target_temp.get(),
            "target_temp": getattr(self, "_last_valid_target", safe_float(self.target_temp.get(), 25.0) or 25.0),
            "temp_tau": self.temp_tau.get(),
            "acc_window_s": self.acc_window_s.get(),
            "acc_tol": self.acc_tol.get(),
            "fit_window_s": self.fit_window_s.get(),
            "ramp_start": self.ramp_start.get(),
            "ramp_end": self.ramp_end.get(),
            "ramp_rate": self.ramp_rate.get(),
            "ramp_hold_min": self.ramp_hold_min.get(),
            "csv_dir": self.csv_dir.get(),
            "mae_stable_thr": self.mae_stable_thr.get(),
            "slope_start_thr_cpm": self.slope_start_thr_cpm.get(),
            "slope_stop_thr_cpm": self.slope_stop_thr_cpm.get(),

            "auto_click_enable": bool(self.auto_click_enable.get()),
            "auto_click_x": s2i(self.auto_click_x, 0),
            "auto_click_y": s2i(self.auto_click_y, 0),
            "auto_click_button": self.auto_click_button.get(),
            "auto_click_double": bool(self.auto_click_double.get()),
            "auto_click_delay_ms": s2i(self.auto_click_delay_ms, 0),

            # === 新增：循环泵持久化 ===
            "pump_port": sval(self.pump_port),
            "pump_baud": s2i(self.pump_baud, PUMP_BAUD),
            "pump_addr": s2i(self.addr_pump, PUMP_ADDR_DEFAULT),
            "pump_vmin": self.pump_vmin.get(),
            "pump_vmax": self.pump_vmax.get(),
            "pump_deadband": self.pump_deadband.get(),
            "pump_slew_vps": self.pump_slew_vps.get(),
            "kp_cool": self.kp_cool.get(),
            "ki_cool": self.ki_cool.get(),
            "kd_cool": self.kd_cool.get(),

            # —— PWM 相关 ——
            "pump_pwm_enable": bool(self.pump_pwm_enable.get()),
            "pump_pwm_auto_on_hotv": bool(self.pump_pwm_auto_on_hotv.get()),
            "pump_pwm_hotv_thr": self.pump_pwm_hotv_thr.get(),
            "pump_pwm_auto_on_ramp_up": bool(self.pump_pwm_auto_on_ramp_up.get()),
            "pump_pwm_auto_duty": self.pump_pwm_auto_duty.get(),
            "pump_pwm_err_thr_C": self.pump_pwm_err_thr_C.get(),
            "pump_pwm_freq": self.pump_pwm_freq.get(),
            "pump_pwm_v_on": self.pump_pwm_v_on.get(),
            "pump_pwm_v_off": self.pump_pwm_v_off.get(),
            "pump_pwm_always_on_temp": bool(self.pump_pwm_always_on_temp.get()),
            "pump_pwm_always_temp_thr": self.pump_pwm_always_temp_thr.get(),
            "pump_pwm_min_duty": self.pump_pwm_min_duty.get(),
        }

        try:
            path = get_config_path()
            path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _load_config(self):
        try:
            path = get_config_path()
            if not path.exists():
                return
            cfg = json.loads(path.read_text(encoding="utf-8"))

            def _get_str(key, default=""):
                val = cfg.get(key, default)
                if val is None:
                    return ""
                return str(val).strip()

            def _get_int(key, default):
                val = cfg.get(key, default)
                try:
                    if isinstance(val, str):
                        val = val.strip()
                        if val == "":
                            return int(default)
                    return int(float(val))
                except Exception:
                    return int(default)

            self.psu_port.set(_get_str("psu_port", self.psu_port.get()))
            self.psu_baud.set(_get_int("psu_baud", self.psu_baud.get()))
            self.addr_psu.set(_get_int("psu_addr", self.addr_psu.get()))
            self.tcm_port.set(_get_str("tcm_port", self.tcm_port.get()))
            self.tcm_baud.set(_get_int("tcm_baud", self.tcm_baud.get()))
            self.addr_tcm.set(_get_int("tcm_addr", self.addr_tcm.get()))
            self.relay_port.set(_get_str("relay_port", self.relay_port.get()))
            self.relay_baud.set(_get_int("relay_baud", self.relay_baud.get()))
            self.relay_chan.set(_get_int("relay_chan", self.relay_chan.get()))

            self.kp.set(cfg.get("kp", self.kp.get()))
            self.ki.set(cfg.get("ki", self.ki.get()))
            self.kd.set(cfg.get("kd", self.kd.get()))
            self.sample_hz.set(cfg.get("sample_hz", self.sample_hz.get()))
            self.v_limit.set(cfg.get("v_limit", self.v_limit.get()))
            self.slew_vps.set(cfg.get("slew_vps", self.slew_vps.get()))
            self.plot_window_s.set(cfg.get("plot_window_s", self.plot_window_s.get()))
            self.ui_refresh_ms.set(cfg.get("ui_refresh_ms", self.ui_refresh_ms.get()))
            self.temp_tau.set(cfg.get("temp_tau", self.temp_tau.get()))
            self.acc_window_s.set(cfg.get("acc_window_s", self.acc_window_s.get()))
            self.acc_tol.set(cfg.get("acc_tol", self.acc_tol.get()))
            self.fit_window_s.set(cfg.get("fit_window_s", self.fit_window_s.get()))
            self.csv_dir.set(cfg.get("csv_dir", self.csv_dir.get()))

            self.mae_stable_thr.set(cfg.get("mae_stable_thr", self.mae_stable_thr.get()))
            self.slope_start_thr_cpm.set(cfg.get("slope_start_thr_cpm", self.slope_start_thr_cpm.get()))
            self.slope_stop_thr_cpm.set(cfg.get("slope_stop_thr_cpm", self.slope_stop_thr_cpm.get()))

            self.auto_click_enable.set(cfg.get("auto_click_enable", self.auto_click_enable.get()))
            self.auto_click_x.set(str(cfg.get("auto_click_x", int(self.auto_click_x.get() or "0"))))
            self.auto_click_y.set(str(cfg.get("auto_click_y", int(self.auto_click_y.get() or "0"))))
            self.auto_click_button.set(cfg.get("auto_click_button", self.auto_click_button.get()))
            self.auto_click_double.set(cfg.get("auto_click_double", self.auto_click_double.get()))
            self.auto_click_delay_ms.set(
                str(cfg.get("auto_click_delay_ms", int(self.auto_click_delay_ms.get() or "0"))))

            # —— PWM 相关 ——
            self.pump_pwm_enable.set(cfg.get("pump_pwm_enable", self.pump_pwm_enable.get()))
            self.pump_pwm_auto_on_hotv.set(cfg.get("pump_pwm_auto_on_hotv", self.pump_pwm_auto_on_hotv.get()))
            self.pump_pwm_hotv_thr.set(cfg.get("pump_pwm_hotv_thr", self.pump_pwm_hotv_thr.get()))
            self.pump_pwm_auto_on_ramp_up.set(cfg.get("pump_pwm_auto_on_ramp_up", self.pump_pwm_auto_on_ramp_up.get()))
            self.pump_pwm_auto_duty.set(cfg.get("pump_pwm_auto_duty", self.pump_pwm_auto_duty.get()))
            self.pump_pwm_err_thr_C.set(cfg.get("pump_pwm_err_thr_C", self.pump_pwm_err_thr_C.get()))
            self.pump_pwm_freq.set(cfg.get("pump_pwm_freq", self.pump_pwm_freq.get()))
            self.pump_pwm_v_on.set(cfg.get("pump_pwm_v_on", self.pump_pwm_v_on.get()))
            self.pump_pwm_v_off.set(cfg.get("pump_pwm_v_off", self.pump_pwm_v_off.get()))
            self.pump_pwm_always_on_temp.set(cfg.get("pump_pwm_always_on_temp",
                                                     self.pump_pwm_always_on_temp.get()))
            self.pump_pwm_always_temp_thr.set(cfg.get("pump_pwm_always_temp_thr",
                                                      self.pump_pwm_always_temp_thr.get()))
            self.pump_pwm_min_duty.set(cfg.get("pump_pwm_min_duty", self.pump_pwm_min_duty.get()))
            # 目标温度恢复
            t_str = cfg.get("target_temp_str", None)
            t_val = cfg.get("target_temp", None)
            if t_str is not None:
                self.target_temp.set(t_str)
                t = safe_float(t_str, None)
                if t is not None:
                    self._last_valid_target = float(t)
            elif t_val is not None:
                self._last_valid_target = float(t_val)
                self.target_temp.set(f"{self._last_valid_target:.3f}")
            self.acc_info.set(
                f"窗={safe_float(self.acc_window_s.get(), 30):.0f} s，容差=±{safe_float(self.acc_tol.get(), 0.5):.2f} °C")

            # ramp
            self.ramp_start.set(cfg.get("ramp_start", self.ramp_start.get()))
            self.ramp_end.set(cfg.get("ramp_end", self.ramp_end.get()))
            self.ramp_rate.set(cfg.get("ramp_rate", self.ramp_rate.get()))
            self.ramp_hold_min.set(cfg.get("ramp_hold_min", self.ramp_hold_min.get()))

            # === 新增：循环泵恢复 ===
            self.pump_port.set(_get_str("pump_port", self.pump_port.get()))
            self.pump_baud.set(_get_int("pump_baud", self.pump_baud.get()))
            self.addr_pump.set(_get_int("pump_addr", self.addr_pump.get()))
            self.pump_vmin.set(cfg.get("pump_vmin", self.pump_vmin.get()))
            self.pump_vmax.set(cfg.get("pump_vmax", self.pump_vmax.get()))
            self.pump_deadband.set(cfg.get("pump_deadband", self.pump_deadband.get()))
            self.pump_slew_vps.set(cfg.get("pump_slew_vps", self.pump_slew_vps.get()))
            self.kp_cool.set(cfg.get("kp_cool", self.kp_cool.get()))
            self.ki_cool.set(cfg.get("ki_cool", self.ki_cool.get()))
            self.kd_cool.set(cfg.get("kd_cool", self.kd_cool.get()))

            # 后台 target 同步
            with self.params_lock:
                self.params["target"] = self._last_valid_target

            self._sync_click_vars_to_entries()
        except Exception:
            pass

    # ---------- 退出清理 ----------
    def cleanup(self):
        """退出清理：安全收尾线程、热键、CSV 与设备。"""
        try:
            # —— 坐标拾取与 Enter 解绑 ——
            try:
                if getattr(self, "_pick_running", False) and hasattr(self, "_stop_pick_coord"):
                    self._stop_pick_coord()
            except Exception:
                pass

            # 兜底：全局 Enter 热键与 Tk 的 <Return>
            try:
                if 'keyboard' in globals() and keyboard:
                    if getattr(self, "_kb_enter_id", None) is not None:
                        try:
                            keyboard.remove_hotkey(self._kb_enter_id)
                        except Exception:
                            pass
            except Exception:
                pass
            try:
                if getattr(self, "_tk_enter_bound", False):
                    self.root.unbind_all('<Return>')
            except Exception:
                pass
            # 清理标志位
            try:
                self._kb_enter_id = None
                self._tk_enter_bound = False
                self._pick_running = False
            except Exception:
                pass

            # —— 停止线性程序（含线程 join 与 CSV 收尾）——
            try:
                if hasattr(self, "_stop_ramp"):
                    self._stop_ramp()
                else:
                    if hasattr(self, "ramp_stop_evt"):
                        self.ramp_stop_evt.set()
                    if getattr(self, "ramp_thread", None):
                        self.ramp_thread.join(timeout=2.0)
            except Exception:
                pass

            # —— 关闭 PID 输出、采集线程 ——
            try:
                self.disable_pid()
            except Exception:
                pass
            try:
                self.stop_acquisition()
            except Exception:
                pass

            # —— 断开设备 ——
            try:
                if hasattr(self, "psu") and self.psu:
                    self.psu.disconnect()
            except Exception:
                pass
            try:
                if hasattr(self, "tcm") and self.tcm:
                    self.tcm.disconnect()
            except Exception:
                pass

            # —— 兜底关闭 CSV 会话 ——
            try:
                self._csv_close_session("程序退出")
            except Exception:
                pass

        finally:
            # —— 保存配置 ——
            try:
                self._save_config()
            except Exception:
                pass
            try:
                if getattr(self, "tcp_server", None):
                    self.tcp_server.stop()
            except Exception:
                pass

    def _start_pick_coord_enter(self):
        """
        开始坐标拾取（按 Enter 确认）：
        - keyboard 可用则全局热键；
        - 否则用 Tk 的 bind_all（无论焦点在哪个控件都能触发）。
        """
        if getattr(self, "_pick_running", False):
            return

        self._pick_running = True
        self._set_status("坐标拾取中：把鼠标移到目标处，按 Enter 确认。")

        # 清理旧的绑定
        try:
            if getattr(self, "_tk_enter_bound", False):
                self.root.unbind_all('<Return>')
        except Exception:
            pass
        self._tk_enter_bound = False

        # keyboard 优先（打包成 exe 时多半不可用）
        self._kb_enter_id = None
        if 'keyboard' in globals() and keyboard:
            try:
                self._kb_enter_id = keyboard.add_hotkey('enter', lambda: self._finish_pick_coord())
            except Exception:
                self._kb_enter_id = None

        # 退回到 Tk：用 bind_all，确保在任何控件焦点下都响应
        if self._kb_enter_id is None:
            self.root.bind_all('<Return>', self._on_return_pick)
            self._tk_enter_bound = True
            try:
                self.root.focus_force()
            except Exception:
                pass

        # 开启实时坐标预览
        self._update_pick_preview()

    def _on_return_pick(self, event=None):
        """Tk 的 <Return> 回调：完成坐标拾取。"""
        try:
            self._finish_pick_coord()
        finally:
            return "break"

    def _stop_pick_coord(self):
        """手动停止拾取（不写入坐标），解绑热键/事件。"""
        try:
            if 'keyboard' in globals() and keyboard and getattr(self, "_kb_enter_id", None) is not None:
                keyboard.remove_hotkey(self._kb_enter_id)
        except Exception:
            pass
        self._kb_enter_id = None

        try:
            if getattr(self, "_tk_enter_bound", False):
                self.root.unbind_all('<Return>')
        except Exception:
            pass
        self._tk_enter_bound = False

        self._pick_running = False
        self._set_status("已停止坐标拾取")

    def _finish_pick_coord(self):
        """结束拾取并写入 X/Y（由 Enter 触发）。"""
        # —— 解绑 Enter（keyboard / Tk）——
        try:
            if 'keyboard' in globals() and keyboard and getattr(self, "_kb_enter_id", None) is not None:
                try:
                    keyboard.remove_hotkey(self._kb_enter_id)
                finally:
                    self._kb_enter_id = None
        except Exception:
            pass
        try:
            if getattr(self, "_tk_enter_bound", False):
                self.root.unbind_all('<Return>')
        except Exception:
            pass
        self._tk_enter_bound = False
        self._pick_running = False

        # —— 读取坐标 ——
        try:
            x, y = self._get_cursor_pos()
        except Exception as e:
            self._set_status(f"取坐标失败：{e}")
            return

        # —— 变量与 Entry 双路同步（防重复定义导致的“绑定丢失”）——
        try:
            self.auto_click_x.set(int(x))
            self.auto_click_y.set(int(y))
        except Exception:
            pass
        try:
            if hasattr(self, "ent_auto_x"):
                self.ent_auto_x.delete(0, "end");
                self.ent_auto_x.insert(0, str(int(x)))
            if hasattr(self, "ent_auto_y"):
                self.ent_auto_y.delete(0, "end");
                self.ent_auto_y.insert(0, str(int(y)))
        except Exception:
            pass

        # 自动勾选启用，避免“看见坐标但没启用”
        try:
            self.auto_click_enable.set(True)
        except Exception:
            pass

        # 预览与状态
        self.pick_preview.set(f"X={int(x)}, Y={int(y)}")
        self._set_status(f"已记录坐标：({int(x)},{int(y)})，已启用自动点击")

    def _simulate_click(self, x: int, y: int, button: str = "left", double: bool = False, delay_ms: int = 0):
        """非阻塞模拟点击：后台线程执行点击；UI 更新回主线程。"""
        # 最终再做一次坐标保护
        if not self._coords_ok(int(x), int(y)):
            self._set_status("模拟点击被取消：坐标无效。")
            return

        def _do_click_bg():
            try:
                # 可选延时
                if delay_ms and delay_ms > 0:
                    _time.sleep(delay_ms / 1000.0)

                # 执行点击（优先 pyautogui）
                if pyautogui is not None:
                    clicks = 2 if double else 1
                    btn = button.lower().strip()
                    if btn not in ("left", "right", "middle"):
                        btn = "left"
                    pyautogui.click(x=int(x), y=int(y), clicks=clicks, button=btn)
                else:
                    if not sys.platform.startswith("win"):
                        raise RuntimeError("缺少 pyautogui 且非 Windows，无法模拟点击。")
                    user32 = ctypes.windll.user32
                    user32.SetCursorPos(int(x), int(y))
                    MOUSEEVENTF_LEFTDOWN = 0x0002;
                    MOUSEEVENTF_LEFTUP = 0x0004
                    MOUSEEVENTF_RIGHTDOWN = 0x0008;
                    MOUSEEVENTF_RIGHTUP = 0x0010
                    MOUSEEVENTF_MIDDLEDOWN = 0x0020;
                    MOUSEEVENTF_MIDDLEUP = 0x0040
                    btn = button.lower().strip()
                    if btn == "right":
                        down, up = MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP
                    elif btn == "middle":
                        down, up = MOUSEEVENTF_MIDDLEDOWN, MOUSEEVENTF_MIDDLEUP
                    else:
                        down, up = MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP
                    user32.mouse_event(down, 0, 0, 0, 0)
                    user32.mouse_event(up, 0, 0, 0, 0)
                    if double:
                        _time.sleep(0.03)
                        user32.mouse_event(down, 0, 0, 0, 0)
                        user32.mouse_event(up, 0, 0, 0, 0)

                # 成功：回主线程更新状态
                self._set_status(f"已模拟点击：({int(x)},{int(y)}) {button}{' 双击' if double else ''}")
            except Exception as e:
                self._set_status(f"模拟点击失败：{e}")

        # 后台线程执行，确保 UI 不被阻塞
        th = threading.Thread(target=_do_click_bg, daemon=True)
        th.start()

    def _try_auto_click_start(self):
        """在线性段开始（斜率跨阈）时尝试点击；无效坐标则跳过并自动关闭该功能。"""
        try:
            if not bool(self.auto_click_enable.get()):
                return
        except Exception:
            return

        # 解析坐标（空值会变成 0）；这里做严格校验
        try:
            x = int(self.auto_click_x.get() or 0)
            y = int(self.auto_click_y.get() or 0)
        except Exception:
            x, y = 0, 0

        if not self._coords_ok(x, y):
            # 自动关闭，避免下次再触发
            try:
                self.auto_click_enable.set(False)
            except Exception:
                pass
            self._set_status("自动点击已跳过：未选择有效坐标（已自动关闭）。")
            return

        # 同一段只点一次
        if getattr(self, "_auto_click_done_this_ramp", False):
            return
        self._auto_click_done_this_ramp = True

        btn = str(self.auto_click_button.get()).strip().lower()
        dbl = bool(self.auto_click_double.get())
        try:
            dly = int(self.auto_click_delay_ms.get() or 0)
        except Exception:
            dly = 0

        self._simulate_click(x, y, btn or "left", dbl, dly)

    def _get_cursor_pos(self):
        """返回全局屏幕坐标 (x, y)。优先 pyautogui；否则 Windows 用 ctypes 回退。"""
        try:
            if pyautogui is not None:
                pos = pyautogui.position()
                return int(pos[0]), int(pos[1])
            if sys.platform.startswith("win"):
                pt = _POINT()
                ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
                return int(pt.x), int(pt.y)
            raise RuntimeError("无法获取鼠标位置：缺少 pyautogui 且非 Windows 平台")
        except Exception as e:
            raise RuntimeError(f"获取鼠标位置失败：{e}")

    def _screen_size(self):
        """返回屏幕宽高 (w,h)。优先 Tk，Windows 退回 ctypes。"""
        try:
            w = int(self.root.winfo_screenwidth())
            h = int(self.root.winfo_screenheight())
            if w > 0 and h > 0:
                return w, h
        except Exception:
            pass
        try:
            if sys.platform.startswith("win"):
                user32 = ctypes.windll.user32
                return int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1))
        except Exception:
            pass
        return 1920, 1080  # 兜底

    def _coords_ok(self, x: int, y: int) -> bool:
        """坐标有效性检查：非零、在屏幕内，且离边缘至少 2px。"""
        try:
            w, h = self._screen_size()
            return (isinstance(x, int) and isinstance(y, int)
                    and 2 <= x < w - 2 and 2 <= y < h - 2)
        except Exception:
            return False

    def _update_pick_preview(self):
        """拾取期间每 50ms 刷新一次预览坐标。"""
        if not self._pick_running:
            return
        try:
            x, y = self._get_cursor_pos()
            self.pick_preview.set(f"X={x}, Y={y}")
        except Exception:
            pass
        # 继续轮询
        self.root.after(50, self._update_pick_preview)

    def _start_pick_coord(self):
        """
        开始坐标拾取：
        - 若安装了 keyboard：注册 F8 全局热键确认；
        - 同时启动 50ms 预览刷新；可随时点“停止拾取”或再次按 F8 结束。
        """
        if self._pick_running:
            return
        self._pick_running = True
        self._set_status("坐标拾取中：把鼠标移到目标处，按 F8 确认。（若 F8 不可用，请用“3秒后取点”）")

        # 注册 F8 热键（可选）
        self._kb_hotkey_id = None
        if keyboard:
            try:
                self._kb_hotkey_id = keyboard.add_hotkey('F8', lambda: self._finish_pick_coord())
            except Exception:
                self._kb_hotkey_id = None

        # 启动预览刷新
        self._update_pick_preview()

    def _pick_coord_in_3s(self):
        """3秒后读取一次坐标（无需热键/依赖）。"""
        if self._pick_running:
            # 若当前在“开始拾取”模式中，先停止，避免冲突
            self._stop_pick_coord()
        self._set_status("3 秒后取坐标：请把鼠标移到目标位置…")
        self.root.after(3000, self._finish_pick_coord)

# =========================
# 入口
# =========================
def main():
    root = tk.Tk()
    app = App(root)

    def on_closing():
        try:
            app.cleanup()
        finally:
            root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()
