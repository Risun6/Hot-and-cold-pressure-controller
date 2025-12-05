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
        self.lock = threading.Lock()
        self.last_setpoint = 0.0  # 用于仿真模型中的电压
        self.conn_type = "仿真"    # 连接类型描述（仿真 / RS-232 / GPIB / USB / VISA）
        self.remote_sense = False  # 是否开启四线制（远端感测）
        self.model = None          # 根据 *IDN? 粗略判断机型（"2400" / "2636B"/ 其他）

    def list_resources(self):
        if self.rm is None:
            return []
        try:
            return list(self.rm.list_resources())
        except Exception:
            return []

    def connect(self, address, simulate=False):
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
                return "仿真模式（未连接仪器）"

            self.simulated = False
            try:
                self.session = self.rm.open_resource(address, timeout=5000)

                addr_upper = address.upper()
                # 串口 RS-232
                if "ASRL" in addr_upper:
                    self.conn_type = "RS-232"
                    try:
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
            if self.session is not None:
                try:
                    self.session.close()
                except Exception:
                    pass
                self.session = None

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
                    if mode == "Voltage":
                        self.session.write("smua.source.func = smua.OUTPUT_DCVOLTS")
                        self.session.write(f"smua.source.levelv = {level_val}")
                        self.session.write(f"smua.source.limiti = {comp_val}")
                    else:
                        self.session.write("smua.source.func = smua.OUTPUT_DCAMPS")
                        self.session.write(f"smua.source.leveli = {level_val}")
                        self.session.write(f"smua.source.limitv = {comp_val}")
                    self.session.write("smua.source.output = smua.OUTPUT_ON")
                else:
                    # 默认路径：保留原 2400 SCPI 行为
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
                if mode == "Voltage":
                    self.session.write("smua.source.func = smua.OUTPUT_DCVOLTS")
                    self.session.write(f"smua.source.limiti = {comp_val}")
                else:
                    self.session.write("smua.source.func = smua.OUTPUT_DCAMPS")
                    self.session.write(f"smua.source.limitv = {comp_val}")
                self.session.write("smua.source.output = smua.OUTPUT_ON")
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
                    self.session.write(f"smua.source.levelv = {level_val}")
                else:
                    self.session.write(f"smua.source.leveli = {level_val}")
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
                    if enable:
                        self.session.write("smua.sense = smua.SENSE_REMOTE")
                    else:
                        self.session.write("smua.sense = smua.SENSE_LOCAL")
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
                    self.session.write(f"smua.measure.nplc = {nplc_val}")
                else:
                    # 默认路径：保留原 2400 行为
                    self.session.write(f"SENS:CURR:NPLC {nplc_val}")
                    self.session.write(f"SENS:VOLT:NPLC {nplc_val}")
            except Exception:
                pass

    def output_off(self):
        with self.lock:
            if self.simulated or self.session is None:
                return

            model = getattr(self, "model", None)
            try:
                if model == "2636B":
                    self.session.write("smua.source.output = smua.OUTPUT_OFF")
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
                    raw = self.session.query("print(smua.measure.iv())").strip()
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

        self._setup_style()

        self.instrument = KeithleyInstrument()
        self.queue = queue.Queue()
        self.measurement_thread = None
        self.stop_event = threading.Event()
        self.tcp_stop_event = threading.Event()
        self.tcp_server_thread = None
        self.integration_time_var = tk.DoubleVar(value=0.0)  # 硬件积分时间（NPLC）
        self._filtered_pressure = None                       # 压力最新值（保留原接口）
        self._filtered_pressure_ts = None                    # 压力更新时间戳
        self.current_mode = None  # "IV", "It", "Vt", "Rt", "Pt"
        self.current_data = []
        self.total_points = 0
        self.completed_points = 0
        self.start_time = None
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

    def _setup_style(self):
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except Exception:
            pass

        style.configure("TLabel", font=("Microsoft YaHei", 9))
        style.configure("TButton", font=("Microsoft YaHei", 9))
        style.configure("TCheckbutton", font=("Microsoft YaHei", 9))
        style.configure("TNotebook.Tab", font=("Microsoft YaHei", 9))
        style.configure("TLabelframe.Label", font=("Microsoft YaHei", 9, "bold"))

        style.configure("TProgressbar", thickness=12)

    def _build_ui(self):
        # 顶部：连接 + 保存设置
        top_lf = ttk.Labelframe(self.root, text="连接 & 保存设置", padding=8)
        top_lf.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))
        top_lf.columnconfigure(2, weight=1)

        ttk.Label(top_lf, text="仪器地址:").grid(row=0, column=0, sticky="w")
        self.resource_combo = ttk.Combobox(top_lf, width=28, state="readonly")
        self.resource_combo.grid(row=0, column=1, sticky="w")
        ttk.Button(top_lf, text="刷新", command=self.refresh_resources).grid(row=0, column=2, sticky="w", padx=(6, 0))

        self.sim_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(top_lf, text="仿真模式", variable=self.sim_var, command=self.on_sim_toggle).grid(
            row=0, column=3, sticky="w", padx=(10, 0)
        )

        # 四线制开关
        self.four_wire_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            top_lf,
            text="四线制",
            variable=self.four_wire_var,
            command=self.on_four_wire_toggle,
        ).grid(row=0, column=4, sticky="w", padx=(10, 0))

        ttk.Button(top_lf, text="连接", command=self.connect_instrument).grid(row=0, column=5, sticky="w", padx=(10, 0))
        self.status_label = ttk.Label(top_lf, text="未连接（仿真）")
        self.status_label.grid(row=0, column=6, sticky="w", padx=(6, 0))

        ttk.Label(top_lf, text="保存根文件夹:").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.save_root_var = tk.StringVar()
        self.save_root_entry = ttk.Entry(top_lf, textvariable=self.save_root_var, width=40)
        self.save_root_entry.grid(row=1, column=1, columnspan=3, sticky="ew", pady=(6, 0))
        ttk.Button(top_lf, text="浏览...", command=self.choose_save_root).grid(
            row=1, column=4, sticky="w", padx=(6, 0), pady=(6, 0)
        )
        self.auto_save_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(top_lf, text="自动保存", variable=self.auto_save_var).grid(
            row=1, column=5, sticky="w", padx=(6, 0), pady=(6, 0)
        )

        interval_frame = ttk.Frame(top_lf)
        interval_frame.grid(row=2, column=0, columnspan=7, sticky="w", pady=(6, 0))
        ttk.Label(interval_frame, text="积分时间(NPLC):").pack(side=tk.LEFT, padx=5, pady=2)
        self.integration_time_entry = ttk.Entry(
            interval_frame,
            width=8,
            textvariable=self.integration_time_var,
        )
        self.integration_time_entry.pack(side=tk.LEFT, padx=5, pady=2)

        # 中间：左窄（参数+日志），右宽（图表）
        mid_lf = ttk.Labelframe(self.root, text="模式参数 & 实时显示", padding=8)
        mid_lf.grid(row=1, column=0, sticky="nsew", padx=8, pady=4)
        mid_lf.rowconfigure(0, weight=1)
        mid_lf.rowconfigure(1, weight=1)
        mid_lf.columnconfigure(0, weight=1)  # 左侧
        mid_lf.columnconfigure(1, weight=3)  # 右侧

        left_col = ttk.Frame(mid_lf)
        left_col.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(0, 4))
        left_col.rowconfigure(0, weight=0)
        left_col.rowconfigure(1, weight=1)
        left_col.columnconfigure(0, weight=1)

        self.notebook = ttk.Notebook(left_col)
        self.notebook.grid(row=0, column=0, sticky="ew")
        self._build_iv_tab()
        self._build_it_tab()
        self._build_vt_tab()
        self._build_rt_tab()
        self._build_pt_tab()
        self._build_ofr_tab()

        log_frame = ttk.Labelframe(left_col, text="日志", padding=6)
        log_frame.grid(row=1, column=0, sticky="nsew", pady=(6, 0))
        log_frame.rowconfigure(1, weight=1)
        log_frame.columnconfigure(0, weight=1)

        ttk.Label(log_frame, text="输出:").grid(row=0, column=0, sticky="w")
        self.log_text = tk.Text(log_frame, height=10, wrap="word")
        self.log_text.grid(row=1, column=0, sticky="nsew")

        chart_frame = ttk.Labelframe(mid_lf, text="实时曲线", padding=6)
        chart_frame.grid(row=0, column=1, rowspan=2, sticky="nsew", padx=(4, 0))
        chart_frame.rowconfigure(0, weight=1)
        chart_frame.columnconfigure(0, weight=1)

        self.fig = Figure(figsize=(5, 4))
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
        self.canvas_widget.grid(row=0, column=0, sticky="nsew")

        # 曲线样式选择：线 / 点 / 线+点
        style_frame = ttk.Frame(chart_frame)
        style_frame.grid(row=1, column=0, sticky="w", pady=(4, 0))
        ttk.Label(style_frame, text="曲线样式:").grid(row=0, column=0, sticky="w")

        self.plot_style_var = tk.StringVar(value="线")
        style_combo = ttk.Combobox(
            style_frame,
            textvariable=self.plot_style_var,
            values=["线", "点", "线+点"],
            state="readonly",
            width=8,
        )
        style_combo.grid(row=0, column=1, sticky="w", padx=(4, 0))
        style_combo.bind("<<ComboboxSelected>>", lambda e: self._apply_plot_style())

        bottom = ttk.Frame(self.root, padding=8)
        bottom.grid(row=2, column=0, sticky="ew", padx=8, pady=(0, 8))
        bottom.columnconfigure(2, weight=1)

        self.start_button = ttk.Button(bottom, text="开始测量", command=self.start_measurement)
        self.start_button.grid(row=0, column=0, padx=(0, 6))
        self.stop_button = ttk.Button(bottom, text="停止", command=self.stop_measurement, state="disabled")
        self.stop_button.grid(row=0, column=1, padx=(0, 6))

        self.progress = ttk.Progressbar(bottom, mode="determinate", maximum=100)
        self.progress.grid(row=0, column=2, sticky="ew", padx=(0, 6))

        ttk.Button(bottom, text="导出数据", command=self.export_data).grid(row=0, column=3, padx=(0, 6))
        ttk.Button(bottom, text="导出日志", command=self.export_log).grid(row=0, column=4)

        self.points_label = ttk.Label(bottom, text="点数: 0/0")
        self.points_label.grid(row=0, column=5, padx=(10, 0))
        self.eta_label = ttk.Label(bottom, text="剩余时间: --")
        self.eta_label.grid(row=0, column=6, padx=(10, 0))

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # TCP 从机区域
        tcp_lf = ttk.Labelframe(self.root, text="TCP 从机", padding=8)
        tcp_lf.grid(row=3, column=0, sticky="ew", padx=8, pady=(0, 8))
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

    # ---- 各模式参数区 ----

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
        self.iv_delay_var = tk.DoubleVar(value=0.0)
        self.iv_compliance_var = tk.DoubleVar(value=0.1)
        self.iv_quality_k_var = tk.DoubleVar(value=8.0)
        self.iv_quality_jump_ratio_var = tk.DoubleVar(value=0.02)
        self.iv_quality_flip_count_var = tk.IntVar(value=20)
        self.iv_quality_max_retry_var = tk.IntVar(value=2)
        self.iv_quality_enabled_var = tk.BooleanVar(value=False)
        self._iv_updating = False

        row = 0
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

    def refresh_resources(self):
        resources = self.instrument.list_resources()
        self.resource_combo["values"] = resources
        if resources:
            self.resource_combo.current(0)
            self._log(f"找到地址: {resources}")
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
        simulate = self.sim_var.get()
        if simulate:
            status = self.instrument.connect(address=None, simulate=True)
        else:
            addr = self.resource_combo.get().strip()
            if not addr:
                messagebox.showwarning("未选择地址", "请先在下拉框中选择一个仪器地址，或勾选仿真模式。")
                return
            status = self.instrument.connect(address=addr, simulate=False)

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
        if self.measurement_thread is not None and self.measurement_thread.is_alive():
            if show_dialog:
                messagebox.showwarning("忙碌", "测量正在进行中")
            else:
                self._log("TCP 请求被忽略：测量正在进行中")
            return False

        if not self.instrument.simulated and self.instrument.session is None:
            if show_dialog:
                messagebox.showwarning("未连接", "请先连接仪器或勾选仿真模式")
            else:
                self._log("TCP 请求被忽略：未连接仪器")
            return False

        model = getattr(self.instrument, "model", None)
        try:
            nplc = float(self.integration_time_var.get())
        except Exception:
            nplc = 0.0

        if nplc <= 0:
            nplc = 0.01 if model == "2636B" else 0.1

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
        return True

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
        delay = cfg["delay"]
        compliance = cfg["compliance"]
        source_mode = cfg["source_mode"]

        base_forward = self.instrument.sweep_points(start, stop, points)
        if back_and_forth:
            if len(base_forward) > 1:
                backward = base_forward[-2::-1]
            else:
                backward = base_forward
            one_cycle = np.concatenate([base_forward, backward])
        else:
            one_cycle = base_forward

        seq = np.tile(one_cycle, cycles)
        is_2636b = self.instrument.model == "2636B"
        if is_2636b:
            self.instrument.prepare_source_2636(source_mode, compliance)
        else:
            self.instrument.configure_source(source_mode, float(seq[0]), compliance)

        for idx, level in enumerate(seq):
            if self.stop_event.is_set():
                break
            if is_2636b:
                self.instrument.set_level_2636(source_mode, float(level))
            else:
                self.instrument.configure_source(source_mode, float(level), compliance)
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

        is_2636b = self.instrument.model == "2636B"
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
                time.sleep(delay)
        else:
            for idx in range(points):
                if self.stop_event.is_set():
                    break
                data = self.instrument.measure_once()
                data.update({"index": idx, "setpoint": bias})
                self.queue.put(("data", data, self.total_points))
                time.sleep(delay)

    # ---- 参数收集 ----

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
        if self.iv_backforth_var.get():
            if points > 1:
                per_cycle = points * 2 - 1
            else:
                per_cycle = points
        else:
            per_cycle = points
        total_points = max(0, per_cycle * max(1, cycles))
        return dict(
            start=start,
            stop=stop,
            step=step,
            points=points,
            cycles=cycles,
            back_and_forth=self.iv_backforth_var.get(),
            delay=delay,
            compliance=compliance,
            source_mode="Voltage" if source_mode == "Voltage" else "Current",
            total_points=total_points,
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
        total_points = 0 if infinite else max(0, points)
        return dict(
            bias=bias,
            delay=delay,
            points=points,
            infinite=infinite,
            compliance=compliance,
            total_points=total_points,
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
        total_points = 0 if infinite else max(0, points)
        return dict(
            bias=bias,
            delay=delay,
            points=points,
            infinite=infinite,
            compliance=compliance,
            total_points=total_points,
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
        total_points = 0 if infinite else max(0, points)
        return dict(
            bias=bias,
            delay=delay,
            points=points,
            infinite=infinite,
            compliance=compliance,
            total_points=total_points,
            source_mode="Voltage",
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
        total_points = 0 if infinite else max(0, points)
        return dict(
            bias=bias,
            delay=delay,
            points=points,
            infinite=infinite,
            compliance=compliance,
            total_points=total_points,
            source_mode="Voltage",
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
            self.instrument.output_off()
        except Exception:
            pass
        try:
            self.instrument.close()
        except Exception:
            pass
        self.root.destroy()

    # ---- 参数保存 ----

    def _save_settings(self):
        cfg = {
            "save_root": self.save_root_var.get(),
            "auto_save": self.auto_save_var.get(),
            "four_wire": bool(getattr(self, "four_wire_var", tk.BooleanVar(value=False)).get()),
            "plot_style": getattr(self, "plot_style_var", tk.StringVar(value="线")).get(),
            "integration_nplc": float(self.integration_time_var.get())
            if hasattr(self, "integration_time_var")
            else 0.0,
            "iv": {
                "source_mode": self.iv_source_mode_var.get(),
                "start": self.iv_start_var.get(),
                "stop": self.iv_stop_var.get(),
                "step": self.iv_step_var.get(),
                "points": self.iv_points_var.get(),
                "cycles": self.iv_cycles_var.get(),
                "back_and_forth": self.iv_backforth_var.get(),
                "delay": self.iv_delay_var.get(),
                "compliance": self.iv_compliance_var.get(),
            },
            "iv_quality": {
                "k": self.iv_quality_k_var.get(),
                "max_jump_ratio": self.iv_quality_jump_ratio_var.get(),
                "max_flip_count": self.iv_quality_flip_count_var.get(),
                "max_retry": self.iv_quality_max_retry_var.get(),
            },
            "it": {
                "bias": self.it_bias_var.get(),
                "delay": self.it_delay_var.get(),
                "points": self.it_points_var.get(),
                "infinite": self.it_infinite_var.get(),
                "compliance": self.it_compliance_var.get(),
            },
            "vt": {
                "bias": self.vt_bias_var.get(),
                "delay": self.vt_delay_var.get(),
                "points": self.vt_points_var.get(),
                "infinite": self.vt_infinite_var.get(),
                "compliance": self.vt_compliance_var.get(),
            },
            "rt": {
                "bias": self.rt_bias_var.get(),
                "delay": self.rt_delay_var.get(),
                "points": self.rt_points_var.get(),
                "infinite": self.rt_infinite_var.get(),
                "compliance": self.rt_compliance_var.get(),
            },
            "pt": {
                "bias": self.pt_bias_var.get(),
                "delay": self.pt_delay_var.get(),
                "points": self.pt_points_var.get(),
                "infinite": self.pt_infinite_var.get(),
                "compliance": self.pt_compliance_var.get(),
            },
            "ofr": {
                "voltage": self.ofr_voltage_var.get(),
                "zero_tol": self.ofr_zero_tol_var.get(),
                "bin_step": self.ofr_bin_step_var.get(),
                "off_min": self.ofr_off_min_points_var.get(),
            },
            "tcp": {
                "host": self.tcp_host_var.get(),
                "port": self.tcp_port_var.get(),
            },
        }
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _load_settings(self):
        if not os.path.exists(self.config_path):
            return
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception:
            return
        self.save_root_var.set(cfg.get("save_root", ""))
        self.auto_save_var.set(cfg.get("auto_save", False))

        if hasattr(self, "integration_time_var"):
            try:
                tau = float(cfg.get("integration_nplc", cfg.get("pressure_integration_seconds", 0.0)))
            except Exception:
                tau = 0.0
            if tau < 0:
                tau = 0.0
            self.integration_time_var.set(tau)

        # 读取四线制与曲线样式
        if hasattr(self, "four_wire_var"):
            self.four_wire_var.set(cfg.get("four_wire", False))
        if hasattr(self, "plot_style_var"):
            self.plot_style_var.set(cfg.get("plot_style", "线"))
            try:
                self._apply_plot_style()
            except Exception:
                pass

        iv = cfg.get("iv", {})
        self.iv_source_mode_var.set(iv.get("source_mode", "Voltage"))
        self.iv_start_var.set(iv.get("start", -1.0))
        self.iv_stop_var.set(iv.get("stop", 1.0))
        self.iv_step_var.set(iv.get("step", 0.02))
        self.iv_points_var.set(iv.get("points", 101))
        self.iv_cycles_var.set(iv.get("cycles", 1))
        self.iv_backforth_var.set(iv.get("back_and_forth", False))
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
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((host, port))
                sock.listen()
            except Exception as exc:
                self.queue.put(("log", f"TCP 从机启动失败: {exc}"))
                return

            self.queue.put(("log", f"TCP 从机监听 {host}:{port}"))
            sock.settimeout(1.0)
            while not self.tcp_stop_event.is_set():
                try:
                    conn, addr = sock.accept()
                except socket.timeout:
                    continue
                threading.Thread(target=self._handle_tcp_client, args=(conn, addr), daemon=True).start()

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
