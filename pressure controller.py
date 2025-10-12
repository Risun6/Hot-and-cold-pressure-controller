import tkinter as tk
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinter import scrolledtext, messagebox, filedialog, Menu, simpledialog
import serial
import serial.tools.list_ports
import threading
import time
import struct
import datetime
import queue
import math
import matplotlib
import numpy as np
import os
import json
import csv
import openpyxl
from collections import deque
from openpyxl.drawing.image import Image as XLImage
from PIL import Image, ImageChops
import pyautogui
import atexit, signal
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from tcp_utils import JSONLineServer
from sim_click import capture_click_point, perform_click_async

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

from pathlib import Path

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
    p = home / "Documents"
    p.mkdir(parents=True, exist_ok=True)
    return p


def calculate_crc(data: bytes) -> bytes:
    """计算Modbus CRC校验码"""
    crc = 0xFFFF
    for pos in data:
        crc ^= pos
        for _ in range(8):
            if crc & 0x0001:
                crc >>= 1
                crc ^= 0xA001
            else:
                crc >>= 1
    return struct.pack('<H', crc)


TCP_HOST = "127.0.0.1"
TCP_PORT = 50020

LEFT_PANEL_WIDTH = 760


class ModbusController:
    """
    线程安全、可自愈重连的 Modbus 串口封装：
    - 所有收发在同一个原子区（RLock）内完成，避免并发读写
    - 失败触发指数退避重连（带互斥，防止并发多次重连）
    - 关闭时先停，再缓释，给 Windows 驱动时间释放句柄
    """
    def __init__(self, port, baudrate=9600, slave_id=1, write_timeout=0.2):
        self.port_name = port
        self.baudrate = baudrate
        self.slave_id = slave_id

        self._io_lock = threading.RLock()       # 收发原子区
        self._reopen_mutex = threading.Lock()   # 防止并发重连
        self._closing = False
        self.serial = None
        self._fail_cnt = 0

        self._open_port(write_timeout=write_timeout)

    # ---------- 端口打开 ----------
    def _open_port(self, write_timeout=0.2, max_retry=6):
        backoff = 0.2
        last_exc = None
        for _ in range(max_retry):
            try:
                ser = serial.Serial(
                    self.port_name,
                    self.baudrate,
                    timeout=0.03,
                    write_timeout=write_timeout,
                    bytesize=serial.EIGHTBITS,
                    parity=serial.PARITY_NONE,
                    stopbits=serial.STOPBITS_ONE,
                    inter_byte_timeout=0.03
                )
                try:
                    ser.reset_input_buffer()
                    ser.reset_output_buffer()
                except Exception:
                    pass
                with self._io_lock:
                    self.serial = ser
                time.sleep(0.05)  # 给驱动一个呼吸
                return
            except (serial.SerialException, OSError) as e:
                last_exc = e
                time.sleep(backoff)
                backoff = min(backoff * 2, 1.5)
        raise ConnectionError(f"无法打开串口 {self.port_name}: {last_exc}")

    # ---------- 端口关闭（可重复调用、容错） ----------
    def close(self):
        with self._io_lock:
            self._closing = True
            ser = self.serial
            self.serial = None
        if ser:
            try:
                try:
                    ser.reset_input_buffer()
                    ser.reset_output_buffer()
                except Exception:
                    pass
                try:
                    ser.dtr = False
                    ser.rts = False
                except Exception:
                    pass
                # Windows 上直接 close 有时仍占句柄，给 200–300ms 释放窗口
                ser.close()
            except Exception:
                pass
            finally:
                time.sleep(0.25)
        self._closing = False

    # ---------- 内部：阻塞式重连（带互斥、指数退避） ----------
    def _reopen_blocking(self, max_retry=8):
        if not self._reopen_mutex.acquire(blocking=False):
            return  # 另一线程已在重连
        try:
            self.close()
            backoff = 0.2
            for _ in range(max_retry):
                try:
                    self._open_port()
                    self._fail_cnt = 0
                    return
                except Exception:
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 2.0)
        finally:
            self._reopen_mutex.release()

    def _reopen_async(self):
        threading.Thread(target=self._reopen_blocking, daemon=True).start()

    # ---------- 统一的原子“发→等→收” ----------
    def send_command(self, cmd: bytes, delay: float = 0.01) -> bytes:
        """
        - 在一个 RLock 原子区内完成 write → sleep → read，避免其他线程插入
        - 任何 SerialException / OSError（含 OSError(22)）都会触发自愈重连
        """
        full_cmd = cmd + calculate_crc(cmd)
        try:
            with self._io_lock:
                ser = self.serial
                if not ser or not ser.is_open or self._closing:
                    return b''

                # 清理残留输入，避免旧帧干扰
                try:
                    ser.reset_input_buffer()
                except Exception:
                    pass

                ser.write(full_cmd)      # 受 write_timeout 保护
                time.sleep(delay)        # 等待从站回包
                resp = ser.read(256)     # 受 timeout 保护
                self._fail_cnt = 0
                return resp

        except (serial.SerialTimeoutException, serial.SerialException, OSError):
            # 包含 Windows 的 OSError(22, '函数不正确')
            self._fail_cnt += 1
            # 少量瞬时失败不立刻重连，连续失败触发自愈
            if self._fail_cnt >= 3:
                self._reopen_async()
            return b''
        except Exception:
            # 兜底：不让异常向上炸掉采集线程
            self._fail_cnt += 1
            if self._fail_cnt >= 3:
                self._reopen_async()
            return b''

    # ---------- Modbus 功能封装 ----------
    def write_registers(self, start_address: int, values: list, delay: float = 0.05) -> bytes:
        reg_count = len(values)
        byte_count = reg_count * 2
        cmd = bytearray()
        cmd.append(self.slave_id)
        cmd.append(0x10)
        cmd += start_address.to_bytes(2, byteorder='big')
        cmd += reg_count.to_bytes(2, byteorder='big')
        cmd.append(byte_count)
        for v in values:
            cmd += int(v).to_bytes(2, byteorder='big', signed=False)
        return self.send_command(bytes(cmd), delay)

    def read_registers(self, start_address: int, reg_count: int) -> bytes:
        cmd = bytearray()
        cmd.append(self.slave_id)
        cmd.append(0x03)
        cmd += start_address.to_bytes(2, byteorder='big')
        cmd += reg_count.to_bytes(2, byteorder='big')
        return self.send_command(bytes(cmd), delay=0.01)

    def write_single_register(self, address: int, value: int) -> bytes:
        cmd = bytearray()
        cmd.append(self.slave_id)
        cmd.append(0x06)
        cmd += address.to_bytes(2, byteorder='big')
        cmd += int(value).to_bytes(2, byteorder='big', signed=False)
        return self.send_command(bytes(cmd), delay=0.01)

    # —— 你原来的若干便捷方法保持不变 —— #
    def hardware_tare(self):           return self.write_single_register(0x0015, 1)
    def zero_calibration(self):        return self.write_single_register(0x0016, 1)
    def weight_calibration(self, weight_value: int):
        try:
            self.write_single_register(0x0018, 0)  # 部分设备需要先进入模式
        except Exception:
            pass
        return self.write_single_register(0x0006, int(weight_value))
    def toggle_write_protection(self, enable: bool):
        return self.write_single_register(0x0019, 1 if enable else 0)



class App(ttk.Frame):
    def __init__(self, master=None):
        if master is None:
            window = ttk.Window(themename="cosmo")
            master_widget = window
            self._owns_window = True
        else:
            master_widget = master
            if isinstance(master, (tk.Tk, tk.Toplevel, ttk.Window)):
                window = master
            else:
                window = master.winfo_toplevel()
            self._owns_window = False

        super().__init__(master_widget)
        self.pack(fill=tk.BOTH, expand=True)

        self._window = window
        self._embedded = not self._owns_window
        self._external_log = None
        self._use_internal_log = self._owns_window
        self.log_text = None

        if self._owns_window:
            try:
                ttk.Style().theme_use("cosmo")
            except Exception:
                pass

        if hasattr(self._window, "title"):
            self._window.title("压力控制综合程序")
        if hasattr(self._window, "minsize"):
            self._window.minsize(1600, 900)
        if hasattr(self._window, "geometry"):
            self._window.geometry("1600x900")
        if hasattr(self._window, "protocol"):
            self._window.protocol("WM_DELETE_WINDOW", self.on_closing)

        if self._owns_window:
            try:
                self._window.iconbitmap("icon.ico")
            except Exception:
                pass

        self.create_menu()

        # 初始化变量
        self.modbus1 = None  # 压力传感器
        self.modbus2 = None  # 运动控制器
        self.running = False
        self.jump_running = False
        self.goto_mode_set = False
        self.jump_values = {i: None for i in range(1, 9)}
        self.jump_vars = {i: tk.DoubleVar() for i in range(1, 9)}
        self.jump_check_vars = {i: tk.BooleanVar(value=True) for i in range(1, 9)}
        self.params_modified = False
        self.last_direction = "上"
        self.last_speed = 5.0
        self.last_distance = 20.0
        self.jump_mode_var = tk.StringVar(value="按实际位置")
        self.tare_value = 0
        self.pressure_running = False
        self.current_session_data = []
        self.session_start_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_stop_time = ""
        self.time_data = []
        self.pos_data = []
        self.pressure_data = []
        self._pressure_history_lock = threading.Lock()
        self._pressure_history = deque()
        self._auto_tare_prompted = False
        self._history_max_seconds = 1800.0
        self.time_start = time.time()
        self.min_pressure = float('inf')
        self.max_pressure = float('-inf')
        docs = resolve_documents_dir()
        self.CONFIG_DIR = str(docs)
        self.CONFIG_FILE = "pressure_monitor_config.json"
        self.CONFIG_PATH = os.path.join(self.CONFIG_DIR, self.CONFIG_FILE)
        # 不要在此处 self.log（此时日志控件未创建）

        self.loaded_config = {}

        self._ui_refresh_default_ms = 60.0
        self._plot_refresh_default_ms = 80.0

        self.default_params = {
            "pos_interval": "0.01",
            "press_interval": "0.01",
            "base_jump": 1000.0,
            "jump_interval": 1000.0,
            "all_jump_direction": "正向",
            "jump_settings": {
                str(i): {
                    "threshold": 1000.0 + (i - 1) * 1000.0,
                    "direction": "正向",
                    "auto_close": False
                } for i in range(1, 9)
            },
            "serial_port1": "",
            "baud_rate1": "9600",
            "serial_port2": "",
            "baud_rate2": "9600",
            "high_speed": 0.01,
            "low_speed": 0.001,
            "pressure_tolerance": 10.0,
            "ui_refresh_interval_ms": self._ui_refresh_default_ms,
            "plot_refresh_interval_ms": self._plot_refresh_default_ms,
        }
        self.send_jump_enabled = {i: False for i in range(1, 9)}
        self.jump_signal_sent = {i: False for i in range(1, 9)}
        self.jump_last_state = {i: None for i in range(1, 9)}
        self.last_jump_sent = None
        self.last_sent_signal = None
        self.last_pressure = 0
        self.connection_status_var1 = tk.StringVar(value="未连接")
        self.connection_status_var2 = tk.StringVar(value="未连接")
        self.pressure_display_var = tk.StringVar(value="压力: 0g")
        self.current_pos_var = tk.StringVar(value="0.0")
        self.current_speed_var = tk.StringVar(value="0.0")
        self.ui_refresh_interval_ms_var = tk.StringVar(
            value=self._format_interval(self._ui_refresh_default_ms)
        )
        self.plot_refresh_interval_ms_var = tk.StringVar(
            value=self._format_interval(self._plot_refresh_default_ms)
        )
        self.ui_refresh_rate_var = tk.StringVar(value="")
        self.plot_refresh_rate_var = tk.StringVar(value="")
        self.pressure_queue = queue.Queue()
        self.position_queue = queue.Queue()
        self.last_press_read = time.time()
        self.last_pos_read = time.time()
        self.last_plot_update = time.time()
        self.current_position = 0.0
        self.current_pressure = 0
        self.MIN_INTERVAL = 0.01
        self.main_canvas = None
        self.scrollbar = None
        self.scrollable_frame = None
        self._ui_update_job = None
        self._plot_update_job = None
        self.pressure_control_thread = None
        self.pressure_control_running = False
        self.target_pressure = 0.0
        self.pressure_tolerance = 10.0
        self.low_speed = 0.001
        self.jog_step_mm = 0.001  # 点动步长固定为 1μm，去除可编辑项
        self.high_speed = 5.0
        self.write_protection_enabled = True
        self.debug_mode = True
        self.device_thread = None
        self.pixel_detection_paused_until = 0
        self._periodic_jobs_enabled = False
        self._last_pixel_check_ts = time.time()

        # 多压力测试相关
        self.multi_pressure_mode_var = tk.BooleanVar(value=False)
        self.stable_time_var = tk.DoubleVar(value=10.0)
        self.sim_click_enabled = tk.BooleanVar(value=False)
        self.sim_click_pos = None
        self.pressure_points_var = tk.StringVar(value="1000,2000,3000")
        self.loop_mode_var = tk.StringVar(value="顺序")
        self.multi_pressure_running = False
        self.pixel_detection_enabled = tk.BooleanVar(value=False)
        self.pixel_detection_region = None
        self.pixel_sensitivity_var = tk.DoubleVar(value=5.0)
        self.pixel_timeout_var = tk.DoubleVar(value=300.0)
        self.initial_pixel_snapshot = None

        # 三个时间参数
        self.detection_interval_var = tk.IntVar(value=1000)       # ms
        self.pressure_step_interval_var = tk.DoubleVar(value=60.0) # s
        self.timeout_var = tk.IntVar(value=8000)                  # ms
        self.pixel_click_cooldown_var = tk.IntVar(value=2000)
        self.last_pixel_click_time = 0
        self.prev_pixel_snapshot = None
        self.pixel_log_enabled = tk.BooleanVar(value=False)

        self.current_unit = "g"
        self.unit_area = 1.0  # Pa/MPa 换算面积(m²)

        # 创建界面
        self.create_widgets()
        self.log(f"配置文件路径: {self.CONFIG_PATH}")
        self.refresh_ports()

        self.ui_refresh_interval_ms_var.trace_add(
            "write", self._on_ui_refresh_interval_changed
        )
        self.plot_refresh_interval_ms_var.trace_add(
            "write", self._on_plot_refresh_interval_changed
        )
        self._on_ui_refresh_interval_changed()
        self._on_plot_refresh_interval_changed()

        # 加载配置
        self.load_config()
        self.validate_intervals()

        self._periodic_jobs_enabled = True
        self._reschedule_ui_update()
        self._reschedule_plot_update()

        self.flashing = False
        self.flash_state = False

        self.last_pressure_record_time = 0
        self.pressure_thread_running = False
        self.position_thread_running = False

        self.after(500, self._maximize_window)
        self.pressure_control_paused_until = 0

        # —— 运动命令节流（避免每50ms狂发）——
        self._mc_last_cmd_ts = 0.0
        self._mc_last_dir = None
        self._mc_last_speed = None
        self._mc_last_distance = None

        self._jump_pos_offsets = {}  # {idx: offset_mm}
        self.LOG_MAX_LINES = 200

        self.tcp_server = JSONLineServer(TCP_HOST, TCP_PORT, self._handle_tcp_command, name="pressure-controller")
        try:
            self.tcp_server.start()
        except OSError as exc:
            self.tcp_server = None
            self.log(f"TCP 服务启动失败: {exc}")

        atexit.register(self._cleanup_ports)
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda *a: self.shutdown())

    def create_menu(self):
        if self._embedded:
            toolbar = ttk.Frame(self, padding=(0, 6))
            toolbar.pack(fill=tk.X)
            ttk.Button(toolbar, text="保存设置", command=self.save_config, bootstyle="secondary-outline").pack(
                side=tk.LEFT, padx=4
            )
            ttk.Button(toolbar, text="加载设置", command=self.load_config, bootstyle="secondary-outline").pack(
                side=tk.LEFT, padx=4
            )
            ttk.Button(toolbar, text="停止全部", command=self.stop_all, bootstyle="warning-outline").pack(
                side=tk.LEFT, padx=12
            )
            return

        menubar = Menu(self._window)

        file_menu = Menu(menubar, tearoff=0)
        file_menu.add_command(label="保存设置", command=self.save_config)
        file_menu.add_command(label="加载设置", command=self.load_config)
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self.on_closing)
        menubar.add_cascade(label="文件", menu=file_menu)

        self._window.config(menu=menubar)

    def call_in_ui(self, fn, *args, **kwargs):
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

        self.after(0, wrapper)
        if not event.wait(timeout=3.0):
            raise TimeoutError("UI thread unresponsive")
        if "error" in result:
            raise result["error"]
        return result.get("value")

    def _maximize_window(self):
        if not getattr(self, "_owns_window", False):
            return
        window = getattr(self, "_window", None)
        if not window:
            return
        for attr in ("state", "wm_state"):
            method = getattr(window, attr, None)
            if callable(method):
                try:
                    method("zoomed")
                    return
                except tk.TclError:
                    continue
        attributes = getattr(window, "attributes", None)
        if callable(attributes):
            try:
                attributes("-zoomed", True)
            except tk.TclError:
                pass

    def create_widgets(self):
        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        left_panel_frame = ttk.LabelFrame(main_frame, text="控制面板", width=LEFT_PANEL_WIDTH)
        left_panel_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10), pady=0)
        left_panel_frame.pack_propagate(False)

        pressure_display_frame = ttk.LabelFrame(left_panel_frame, text="当前压力")
        pressure_display_frame.pack(fill=tk.X, padx=5, pady=10)

        self.big_pressure_var = tk.StringVar(value="0 g")
        self.big_pressure_label = ttk.Label(
            pressure_display_frame,
            textvariable=self.big_pressure_var,
            font=("Arial", 50, "bold"),
            anchor="center"
        )
        self.big_pressure_label.pack(fill=tk.X, padx=10, pady=20)
        self.big_pressure_label.config(background="white", foreground="black")

        device_frame = ttk.LabelFrame(left_panel_frame, text="设备连接")
        device_frame.pack(fill=tk.X, padx=5, pady=5)

        sensor_frame = ttk.Frame(device_frame)
        sensor_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(sensor_frame, text="压力传感器:").pack(side=tk.LEFT, padx=5, pady=2)
        self.port_cb1 = ttk.Combobox(sensor_frame, width=12)
        self.port_cb1.pack(side=tk.LEFT, padx=5, pady=2)
        self._disable_mouse_wheel(self.port_cb1)
        self.baud_entry1 = ttk.Entry(sensor_frame, width=8)
        self.baud_entry1.pack(side=tk.LEFT, padx=5, pady=2)
        self.baud_entry1.insert(0, "9600")
        self._disable_mouse_wheel(self.baud_entry1)
        self.sensor_btn = ttk.Button(sensor_frame, text="连接", command=self.toggle_sensor_connection,
                                     bootstyle="outline-success")
        self.sensor_btn.pack(side=tk.LEFT, padx=5, pady=2)
        self.connection_status_var1 = tk.StringVar(value="未连接")
        ttk.Label(sensor_frame, textvariable=self.connection_status_var1).pack(side=tk.LEFT, padx=5, pady=2)

        controller_frame = ttk.Frame(device_frame)
        controller_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(controller_frame, text="运动控制器:").pack(side=tk.LEFT, padx=5, pady=2)
        self.port_cb2 = ttk.Combobox(controller_frame, width=12)
        self.port_cb2.pack(side=tk.LEFT, padx=5, pady=2)
        self._disable_mouse_wheel(self.port_cb2)
        self.baud_entry2 = ttk.Entry(controller_frame, width=8)
        self.baud_entry2.pack(side=tk.LEFT, padx=5, pady=2)
        self.baud_entry2.insert(0, "9600")
        self._disable_mouse_wheel(self.baud_entry2)
        self.controller_btn = ttk.Button(controller_frame, text="连接", command=self.toggle_controller_connection,
                                         bootstyle="outline-success")
        self.controller_btn.pack(side=tk.LEFT, padx=5, pady=2)
        self.connection_status_var2 = tk.StringVar(value="未连接")
        ttk.Label(controller_frame, textvariable=self.connection_status_var2).pack(side=tk.LEFT, padx=5, pady=2)

        self.sensor_connected = False
        self.controller_connected = False

        device_btn_frame = ttk.Frame(device_frame)
        device_btn_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(device_btn_frame, text="刷新端口", command=self.refresh_ports,
                   bootstyle="outline-secondary").pack(side=tk.LEFT, padx=5)
        ttk.Button(device_btn_frame, text="自动连接", command=self.auto_connect_devices,
                   bootstyle="outline-primary").pack(side=tk.LEFT, padx=5)

        interval_frame = ttk.Frame(device_frame)
        interval_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(interval_frame, text="位置读取间隔:").pack(side=tk.LEFT, padx=5, pady=2)
        self.pos_interval_entry = ttk.Entry(interval_frame, width=8)
        self.pos_interval_entry.insert(0, "0.01")
        self.pos_interval_entry.pack(side=tk.LEFT, padx=5, pady=2)
        ttk.Label(interval_frame, text="压力读取间隔:").pack(side=tk.LEFT, padx=5, pady=2)
        self.press_interval_entry = ttk.Entry(interval_frame, width=8)
        self.press_interval_entry.insert(0, "0.01")
        self.press_interval_entry.pack(side=tk.LEFT, padx=5, pady=2)

        left_canvas = tk.Canvas(left_panel_frame)
        scrollbar = ttk.Scrollbar(left_panel_frame, orient="vertical", command=left_canvas.yview)
        self.scrollable_frame = ttk.Frame(left_canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: left_canvas.configure(scrollregion=left_canvas.bbox("all"))
        )

        left_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw", width=LEFT_PANEL_WIDTH - 20)
        left_canvas.configure(yscrollcommand=scrollbar.set)

        self.scrollable_frame.bind("<Enter>", lambda e: self.scrollable_frame.bind_all("<MouseWheel>", lambda
            e: left_canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")))
        self.scrollable_frame.bind("<Leave>", lambda e: self.scrollable_frame.unbind_all("<MouseWheel>"))

        left_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        motion_ctrl_frame = ttk.LabelFrame(self.scrollable_frame, text="运动控制")
        motion_ctrl_frame.pack(fill=tk.X, padx=5, pady=5)

        direction_frame = ttk.Frame(motion_ctrl_frame)
        direction_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(direction_frame, text="运动方向:").pack(side=tk.LEFT, padx=5, pady=2)
        self.direction = tk.StringVar(value="上")
        self.direction.trace_add("write", lambda *args: self.set_params_modified())
        ttk.Radiobutton(direction_frame, text="上", variable=self.direction, value="上",
                        bootstyle="info-toolbutton").pack(side=tk.LEFT, pady=2)
        ttk.Radiobutton(direction_frame, text="下", variable=self.direction, value="下",
                        bootstyle="info-toolbutton").pack(side=tk.LEFT, pady=2)

        speed_set_frame = ttk.Frame(motion_ctrl_frame)
        speed_set_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(speed_set_frame, text="速度 (mm/s):").pack(side=tk.LEFT, padx=5, pady=2)
        self.speed_entry = ttk.Entry(speed_set_frame, width=8)
        self.speed_entry.pack(side=tk.LEFT, padx=5, pady=2)
        self.speed_entry.insert(0, "5.0")
        self.speed_entry.bind("<KeyRelease>", lambda e: self.set_params_modified())

        distance_frame = ttk.Frame(motion_ctrl_frame)
        distance_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(distance_frame, text="距离 (mm):").pack(side=tk.LEFT, padx=5, pady=2)
        self.distance_entry = ttk.Entry(distance_frame, width=8)
        self.distance_entry.pack(side=tk.LEFT, padx=5, pady=2)
        self.distance_entry.insert(0, "20.0")
        self.distance_entry.bind("<KeyRelease>", lambda e: self.set_params_modified())

        target_pos_frame = ttk.Frame(motion_ctrl_frame)
        target_pos_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(target_pos_frame, text="目标位置 (mm):").pack(side=tk.LEFT, padx=5, pady=2)
        self.target_entry = ttk.Entry(target_pos_frame, width=8)
        self.target_entry.pack(side=tk.LEFT, padx=5, pady=2)
        self.target_entry.insert(0, "0.0")

        ctrl_btn_frame = ttk.Frame(motion_ctrl_frame)
        ctrl_btn_frame.pack(fill=tk.X, padx=5, pady=8)
        btn_width = 8

        self.set_btn = ttk.Button(ctrl_btn_frame, text="设置参数", width=btn_width,
                                  command=lambda: threading.Thread(target=self.set_parameters, daemon=True).start(),
                                  state="disabled", bootstyle="warning")
        self.set_btn.grid(row=0, column=0, padx=5, pady=2, sticky='ew')

        self.start_btn = ttk.Button(ctrl_btn_frame, text="启动运动", width=btn_width,
                                    command=self.start_machine,
                                    state="disabled", bootstyle="success")
        self.start_btn.grid(row=0, column=1, padx=5, pady=2, sticky='ew')

        self.zero_btn = ttk.Button(ctrl_btn_frame, text="回零", width=btn_width,
                                   command=self.return_zero,
                                   state="disabled", bootstyle="primary")
        self.zero_btn.grid(row=1, column=1, padx=5, pady=2, sticky='ew')

        self.move_btn = ttk.Button(ctrl_btn_frame, text="运动到", width=btn_width,
                                   command=self.move_to,
                                   state="disabled", bootstyle="primary")
        self.move_btn.grid(row=1, column=0, padx=5, pady=2, sticky='ew')

        self.jump_btn = ttk.Button(ctrl_btn_frame, text="跳变", width=btn_width,
                                   command=self.start_jump_mode,
                                   state="disabled", bootstyle="primary")
        self.jump_btn.grid(row=1, column=2, padx=5, pady=2, sticky='ew')

        self.stop_btn = ttk.Button(ctrl_btn_frame, text="停止运动", width=btn_width,
                                    command=self.stop_all,
                                    state="disabled", bootstyle="danger")
        self.stop_btn.grid(row=0, column=2, padx=5, pady=2, sticky='ew')

        for i in range(3):
            ctrl_btn_frame.grid_columnconfigure(i, weight=1, uniform="btn")

        # ===== 压力控制区 =====
        pressure_ctrl_frame = ttk.LabelFrame(self.scrollable_frame, text="压力控制")
        pressure_ctrl_frame.pack(fill=tk.X, padx=5, pady=5)

        row1_frame = ttk.Frame(pressure_ctrl_frame)
        row1_frame.pack(fill=tk.X, padx=5, pady=4)
        ttk.Label(row1_frame, text="目标压力(g):").pack(side=tk.LEFT, padx=5)
        self.target_pressure_var = tk.DoubleVar()
        ttk.Entry(row1_frame, textvariable=self.target_pressure_var, width=10).pack(side=tk.LEFT, padx=2)

        ttk.Label(row1_frame, text="容差(g):").pack(side=tk.LEFT, padx=10)
        self.tolerance_var = tk.DoubleVar(value=10.0)
        ttk.Entry(row1_frame, textvariable=self.tolerance_var, width=8).pack(side=tk.LEFT, padx=2)

        self.jump_bias_gate_var = tk.DoubleVar(value=100.0)  # 闭环偏置门槛 g

        row2_frame = ttk.Frame(pressure_ctrl_frame)
        row2_frame.pack(fill=tk.X, padx=5, pady=4)
        ttk.Label(row2_frame, text="高速(mm/s):").pack(side=tk.LEFT, padx=5)
        self.high_speed_var = tk.DoubleVar(value=5.0)
        ttk.Entry(row2_frame, textvariable=self.high_speed_var, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Label(row2_frame, text="低速(mm/s):").pack(side=tk.LEFT, padx=10)
        self.low_speed_var = tk.DoubleVar(value=0.001)
        ttk.Entry(row2_frame, textvariable=self.low_speed_var, width=8).pack(side=tk.LEFT, padx=2)

        row3_frame = ttk.Frame(pressure_ctrl_frame)
        row3_frame.pack(fill=tk.X, padx=5, pady=4)
        ttk.Label(row3_frame, text="渐进区间(g):").pack(side=tk.LEFT, padx=5)
        self.progressive_zone_var = tk.DoubleVar(value=100.0)
        ttk.Entry(row3_frame, textvariable=self.progressive_zone_var, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Label(row3_frame, text="保护压力(g):").pack(side=tk.LEFT, padx=10)
        self.safety_pressure_var = tk.DoubleVar(value=30000.0)
        ttk.Entry(row3_frame, textvariable=self.safety_pressure_var, width=10).pack(side=tk.LEFT, padx=2)

        row4_frame = ttk.Frame(pressure_ctrl_frame)
        row4_frame.pack(fill=tk.X, padx=5, pady=4)
        self.motion_mode_var = tk.StringVar(value="点动")
        ttk.Radiobutton(row4_frame, text="点动模式", variable=self.motion_mode_var, value="点动",
                        bootstyle="info-toolbutton").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(row4_frame, text="持续模式", variable=self.motion_mode_var, value="持续",
                        bootstyle="info-toolbutton").pack(side=tk.LEFT, padx=5)
        ttk.Label(
            row4_frame,
            text=f"点动步长：{self.jog_step_mm:.3f} mm（固定）"
        ).pack(side=tk.LEFT, padx=12)

        unit_frame = ttk.Frame(pressure_ctrl_frame)
        unit_frame.pack(fill=tk.X, padx=5, pady=4)
        for unit in ["g", "N", "Pa", "MPa"]:
            btn = ttk.Button(unit_frame, text=unit, width=6,
                             command=lambda u=unit: self.change_pressure_unit(u))
            btn.pack(side=tk.LEFT, padx=5)

        btn_frame = ttk.Frame(pressure_ctrl_frame)
        btn_frame.pack(fill=tk.X, padx=5, pady=8)
        self.start_pressure_btn = ttk.Button(btn_frame, text="启动压力控制", command=self.start_pressure_control,
                                              state="disabled", bootstyle="success")
        self.start_pressure_btn.pack(side=tk.LEFT, padx=10)
        self.stop_pressure_btn = ttk.Button(btn_frame, text="停止压力控制", command=self.stop_pressure_control,
                                            state="disabled", bootstyle="danger")
        self.stop_pressure_btn.pack(side=tk.LEFT, padx=10)

        # ===== 实时数据区 =====
        data_frame = ttk.LabelFrame(self.scrollable_frame, text="实时数据")
        data_frame.pack(fill=tk.X, padx=5, pady=5)

        pos_frame = ttk.Frame(data_frame)
        pos_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(pos_frame, text="当前位置 (mm):").pack(side=tk.LEFT, padx=5, pady=2)
        ttk.Label(pos_frame, textvariable=self.current_pos_var, font=("Arial", 12, "bold")).pack(side=tk.LEFT, pady=2)

        speed_frame = ttk.Frame(data_frame)
        speed_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(speed_frame, text="当前速度 (mm/s):").pack(side=tk.LEFT, padx=5, pady=2)
        ttk.Label(speed_frame, textvariable=self.current_speed_var, font=("Arial", 12, "bold")).pack(side=tk.LEFT, padx=5, pady=2)

        # 压力显示
        press_frame = ttk.Frame(data_frame)
        press_frame.pack(fill=tk.X, padx=5, pady=2)

        ttk.Label(press_frame, text="当前压力 (g):").pack(side=tk.LEFT, padx=5, pady=2)
        ttk.Label(press_frame, textvariable=self.pressure_display_var, font=("Arial", 12, "bold"),
                  bootstyle="danger").pack(side=tk.LEFT, padx=5, pady=2)

        # 右侧按钮（右对齐）：最右是“去皮”，它左边是“自动去皮”
        ttk.Button(press_frame, text="去皮", command=self.tare_pressure, bootstyle="warning") \
            .pack(side=tk.RIGHT, padx=5, pady=2)
        ttk.Button(press_frame, text="自动去皮", command=self.start_auto_tare, bootstyle="success") \
            .pack(side=tk.RIGHT, padx=5, pady=2)

        refresh_frame = ttk.LabelFrame(self.scrollable_frame, text="界面刷新设置")
        refresh_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(refresh_frame, text="数据显示刷新间隔(ms):").grid(
            row=0, column=0, padx=5, pady=4, sticky='e'
        )
        ui_interval_entry = ttk.Entry(
            refresh_frame, textvariable=self.ui_refresh_interval_ms_var, width=8
        )
        ui_interval_entry.grid(row=0, column=1, padx=5, pady=4, sticky='w')
        self._disable_mouse_wheel(ui_interval_entry)
        ttk.Label(
            refresh_frame,
            textvariable=self.ui_refresh_rate_var,
            bootstyle="secondary"
        ).grid(row=0, column=2, padx=5, pady=4, sticky='w')

        ttk.Label(refresh_frame, text="曲线刷新间隔(ms):").grid(
            row=1, column=0, padx=5, pady=4, sticky='e'
        )
        plot_interval_entry = ttk.Entry(
            refresh_frame, textvariable=self.plot_refresh_interval_ms_var, width=8
        )
        plot_interval_entry.grid(row=1, column=1, padx=5, pady=4, sticky='w')
        self._disable_mouse_wheel(plot_interval_entry)
        ttk.Label(
            refresh_frame,
            textvariable=self.plot_refresh_rate_var,
            bootstyle="secondary"
        ).grid(row=1, column=2, padx=5, pady=4, sticky='w')

        ttk.Label(refresh_frame, text="有效范围: 20–1000 ms", bootstyle="secondary") \
            .grid(row=0, column=3, rowspan=2, padx=5, pady=4, sticky='w')
        refresh_frame.grid_columnconfigure(2, weight=1)

        # ===== 自动多压力测试 =====
        multi_pressure_frame = ttk.LabelFrame(self.scrollable_frame, text="自动多压力测试")
        multi_pressure_frame.pack(fill=tk.X, padx=5, pady=5)

        points_frame = ttk.Frame(multi_pressure_frame)
        points_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(points_frame, text="压力点(g):").pack(side=tk.LEFT, padx=5, pady=2)
        ttk.Entry(points_frame, textvariable=self.pressure_points_var, width=25).pack(side=tk.LEFT, padx=5, pady=2)
        ttk.Label(points_frame, text="(用逗号分隔)").pack(side=tk.LEFT, padx=5, pady=2)

        mode_stable_frame = ttk.Frame(multi_pressure_frame)
        mode_stable_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(mode_stable_frame, text="循环模式:").pack(side=tk.LEFT, padx=5, pady=2)
        self.loop_mode_combo = ttk.Combobox(mode_stable_frame, textvariable=self.loop_mode_var, width=8)
        self.loop_mode_combo['values'] = ("顺序", "倒序", "顺序+倒序")
        self.loop_mode_combo.pack(side=tk.LEFT, padx=(0, 15), pady=2)
        ttk.Label(mode_stable_frame, text="判稳时长(s):").pack(side=tk.LEFT, padx=5, pady=2)
        ttk.Entry(mode_stable_frame, textvariable=self.stable_time_var, width=4).pack(side=tk.LEFT, padx=5, pady=2)
        ttk.Label(mode_stable_frame, text="循环次数:").pack(side=tk.LEFT, padx=10, pady=2)
        self.loop_count_var = tk.IntVar(value=1)
        ttk.Entry(mode_stable_frame, textvariable=self.loop_count_var, width=3).pack(side=tk.LEFT, padx=2, pady=2)

        time_params_frame = ttk.Frame(multi_pressure_frame)
        time_params_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(time_params_frame, text="像素检测间隔(ms):").grid(row=0, column=0, padx=5, pady=2, sticky='w')
        ttk.Entry(time_params_frame, textvariable=self.detection_interval_var, width=8).grid(row=0, column=1, padx=2, pady=2)
        ttk.Label(time_params_frame, text="压力步进间隔(s):").grid(row=0, column=2, padx=(20, 5), pady=2, sticky='w')
        ttk.Entry(time_params_frame, textvariable=self.pressure_step_interval_var, width=8).grid(row=0, column=3, padx=2, pady=2)

        pixel_frame = ttk.Frame(multi_pressure_frame)
        pixel_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Checkbutton(pixel_frame, text="开启像素点检测", variable=self.pixel_detection_enabled).pack(side=tk.LEFT, padx=5, pady=2)
        ttk.Button(pixel_frame, text="设置检测区域", command=self.set_pixel_detection_region, bootstyle="outline-info").pack(side=tk.LEFT, padx=5, pady=2)

        self.indicator_canvas = tk.Canvas(pixel_frame, width=18, height=18, highlightthickness=0)
        self.indicator_circle = self.indicator_canvas.create_oval(2, 2, 16, 16, fill="red")
        self.indicator_canvas.pack(side=tk.LEFT, padx=6)

        pixel_param_frame = ttk.Frame(multi_pressure_frame)
        pixel_param_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Checkbutton(pixel_param_frame, text="开启检测日志", variable=self.pixel_log_enabled).pack(side=tk.LEFT, padx=5, pady=2)
        ttk.Label(pixel_param_frame, text="灵敏度(%):").pack(side=tk.LEFT, padx=5, pady=2)
        ttk.Entry(pixel_param_frame, textvariable=self.pixel_sensitivity_var, width=8).pack(side=tk.LEFT, padx=2, pady=2)
        ttk.Label(pixel_param_frame, text="超时(s):").pack(side=tk.LEFT, padx=10, pady=2)
        ttk.Entry(pixel_param_frame, textvariable=self.pixel_timeout_var, width=8).pack(side=tk.LEFT, padx=2, pady=2)

        click_frame = ttk.Frame(multi_pressure_frame)
        click_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Checkbutton(click_frame, text="开启模拟点击", variable=self.sim_click_enabled).pack(side=tk.LEFT, padx=5, pady=2)
        ttk.Button(click_frame, text="设置点击点", command=self.set_sim_click_point, bootstyle="outline-info").pack(side=tk.LEFT, padx=5, pady=2)
        ttk.Label(click_frame, text="点击冷却(ms):").pack(side=tk.LEFT, padx=10, pady=2)
        ttk.Entry(click_frame, textvariable=self.pixel_click_cooldown_var, width=8).pack(side=tk.LEFT, padx=2, pady=2)

        button_frame = ttk.Frame(multi_pressure_frame)
        button_frame.pack(padx=5, pady=5)
        ttk.Button(button_frame, text="启动多压力测试", command=self.start_multi_pressure_test, bootstyle="success").pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="停止多压力测试", command=self.stop_multi_pressure_test, bootstyle="danger").pack(side=tk.LEFT)

        # 跳变设置区  ————  用本段完整替换你原来的“跳变设置”区域  ————
        jump_frame = ttk.LabelFrame(self.scrollable_frame, text="跳变设置")
        jump_frame.pack(fill=tk.X, padx=5, pady=5)

        # 基础阈值和间隔
        base_jump_frame = ttk.Frame(jump_frame)
        base_jump_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(base_jump_frame, text="基础阈值 (g):").pack(side=tk.LEFT, padx=5, pady=2)
        self.base_jump_var = tk.DoubleVar(value=1000.0)
        self.base_jump_entry = ttk.Entry(base_jump_frame, textvariable=self.base_jump_var, width=8)
        self.base_jump_entry.pack(side=tk.LEFT, padx=5, pady=2)
        ttk.Label(base_jump_frame, text="间隔 (g):").pack(side=tk.LEFT, padx=5, pady=2)
        self.jump_interval_var = tk.DoubleVar(value=1000.0)
        self.jump_interval_entry = ttk.Entry(base_jump_frame, textvariable=self.jump_interval_var, width=8)
        self.jump_interval_entry.pack(side=tk.LEFT, padx=5, pady=2)
        ttk.Button(base_jump_frame, text="应用", command=self.apply_jump_interval, bootstyle="outline-primary").pack(
            side=tk.LEFT, padx=5, pady=2)

        # 统一方向
        direction_set_frame = ttk.Frame(jump_frame)
        direction_set_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(direction_set_frame, text="统一方向:").pack(side=tk.LEFT, padx=5, pady=2)
        self.all_jump_direction_var = tk.StringVar(value="正向")
        self.all_jump_direction_combo = ttk.Combobox(direction_set_frame, textvariable=self.all_jump_direction_var,
                                                     width=8)
        self.all_jump_direction_combo['values'] = ("正向", "反向", "双向")
        self.all_jump_direction_combo.pack(side=tk.LEFT, padx=5, pady=2)
        ttk.Button(direction_set_frame, text="应用所有", command=self.toggle_all_jump_direction,
                   bootstyle="outline-primary").pack(side=tk.LEFT, padx=5, pady=2)

        # 跳变间隔 & 循环次数
        interval_frame = ttk.Frame(jump_frame)
        interval_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(interval_frame, text="跳变间隔 (秒):").pack(side=tk.LEFT, padx=5, pady=2)
        self.interval_entry = ttk.Entry(interval_frame, width=8)
        self.interval_entry.pack(side=tk.LEFT, padx=5, pady=2)
        self.interval_entry.insert(0, "1.0")

        ttk.Label(interval_frame, text="循环次数:").pack(side=tk.LEFT, padx=10, pady=2)
        self.jump_loop_count_var = tk.IntVar(value=0)  # 0=一直循环
        ttk.Entry(interval_frame, textvariable=self.jump_loop_count_var, width=6).pack(side=tk.LEFT, padx=2, pady=2)
        ttk.Label(interval_frame, text="(0 = 一直循环)").pack(side=tk.LEFT, padx=6, pady=2)

        # 采集速度 / 跳变速度
        speed_frame = ttk.Frame(jump_frame)
        speed_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(speed_frame, text="采集速度(mm/s):").pack(side=tk.LEFT, padx=5, pady=2)
        self.jump_acq_speed_var = tk.DoubleVar(value=0.001)  # 默认 0.001
        ttk.Entry(speed_frame, textvariable=self.jump_acq_speed_var, width=10).pack(side=tk.LEFT, padx=5, pady=2)
        ttk.Label(speed_frame, text="跳变速度(mm/s):").pack(side=tk.LEFT, padx=10, pady=2)
        self.jump_run_speed_var = tk.DoubleVar(value=0.1)  # 默认 0.1
        ttk.Entry(speed_frame, textvariable=self.jump_run_speed_var, width=10).pack(side=tk.LEFT, padx=5, pady=2)
        bias_gate_frame = ttk.Frame(jump_frame)
        bias_gate_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(bias_gate_frame, text="闭环偏置门槛 (g):").pack(side=tk.LEFT, padx=5, pady=2)
        ttk.Entry(bias_gate_frame, textvariable=self.jump_bias_gate_var, width=8).pack(side=tk.LEFT, padx=5, pady=2)
        ttk.Label(bias_gate_frame, text=">该门槛才±0.001 mm").pack(side=tk.LEFT, padx=8, pady=2)

        # 跳变模式选择（如果你想在UI上切换；若已有别处控制可删）
        mode_frame = ttk.Frame(jump_frame)
        mode_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(mode_frame, text="跳变模式:").pack(side=tk.LEFT, padx=5, pady=2)
        # 复用已有 self.jump_mode_var
        mode_combo = ttk.Combobox(mode_frame, textvariable=self.jump_mode_var, width=10)
        mode_combo['values'] = ("按实际压力", "按实际位置")
        mode_combo.pack(side=tk.LEFT, padx=5, pady=2)

        # 跳变阈值设置（每一行： 阈值 + 方向 + 自动关闭 + 开启按钮 + 实测位置显示）
        self.jump_entries = {}
        self.jump_buttons = {}
        self.jump_directions = {}
        self.jump_auto_close_vars = {}
        self.jump_auto_close_checks = {}
        self.jump_pos_vars = {}  # 新增：每个跳变点的“已采集位置”显示
        self.jump_pos_labels = {}

        for i in range(1, 9):
            jump_subframe = ttk.Frame(jump_frame)
            jump_subframe.pack(fill=tk.X, padx=5, pady=2)

            ttk.Label(jump_subframe, text=f"跳变{i}:").pack(side=tk.LEFT, padx=2, pady=2)

            self.jump_vars[i] = tk.DoubleVar() if i not in self.jump_vars else self.jump_vars[i]
            self.jump_entries[i] = ttk.Entry(jump_subframe, textvariable=self.jump_vars[i], width=8)
            self.jump_entries[i].pack(side=tk.LEFT, padx=2, pady=2)

            self.jump_directions[i] = tk.StringVar(value="正向") if i not in self.jump_directions else \
            self.jump_directions[i]
            direction_combo = ttk.Combobox(jump_subframe, textvariable=self.jump_directions[i], width=8)
            direction_combo['values'] = ("正向", "反向", "双向")
            direction_combo.pack(side=tk.LEFT, padx=2, pady=2)

            self.jump_auto_close_vars[i] = tk.BooleanVar() if i not in self.jump_auto_close_vars else \
            self.jump_auto_close_vars[i]
            self.jump_auto_close_checks[i] = ttk.Checkbutton(jump_subframe, text="自动关闭",
                                                             variable=self.jump_auto_close_vars[i])
            self.jump_auto_close_checks[i].pack(side=tk.LEFT, padx=2, pady=2)

            btn = ttk.Button(jump_subframe, text="关闭", command=lambda i=i: self.toggle_send_jump_signal(i),
                             width=6, bootstyle="outline-danger")
            btn.pack(side=tk.LEFT, padx=2, pady=2)
            self.jump_buttons[i] = btn

            # —— 新增：已采集位置显示 —— #
            self.jump_pos_vars[i] = tk.StringVar(value="—")  # 初始未知
            lbl = ttk.Label(jump_subframe, textvariable=self.jump_pos_vars[i])
            lbl.pack(side=tk.LEFT, padx=8, pady=2)
            self.jump_pos_labels[i] = lbl

        # 一键打开所有跳变信号按钮
        ttk.Button(jump_frame, text="一键打开所有跳变信号", command=self.enable_all_jump_signals,
                   bootstyle="outline-info").pack(padx=5, pady=5)

        # 跳变 启动/停止（放在“跳变设置”里）
        jump_ctrl_btns = ttk.Frame(jump_frame)
        jump_ctrl_btns.pack(fill=tk.X, padx=5, pady=6)
        ttk.Button(jump_ctrl_btns, text="启动跳变", command=self.start_jump_mode,
                   bootstyle="success").pack(side=tk.LEFT, padx=5)
        ttk.Button(jump_ctrl_btns, text="停止跳变", command=lambda: self.stop_jump(manual=True),
                   bootstyle="danger").pack(side=tk.LEFT, padx=5)

        # 右侧面板
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.fig = Figure(figsize=(10, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax2 = self.ax.twinx()
        self.init_plot_style()

        self.line_pos, = self.ax.plot([], [], color='#268bd2', linewidth=2.1, label='Position (mm)')
        self.line_press, = self.ax2.plot([], [], color='#e74c3c', linewidth=2.1, label=f'Pressure ({self.current_unit})')

        self.ax.legend(loc='upper left', fontsize=10, frameon=False, borderpad=0.3, labelspacing=0.2)
        self.ax2.legend(loc='upper right', fontsize=10, frameon=False, borderpad=0.3, labelspacing=0.2)

        self.canvas = FigureCanvasTkAgg(self.fig, master=right_panel)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # —— 图表按钮与显示时长设置 —— #
        chart_btn_frame = ttk.Frame(right_panel)
        chart_btn_frame.pack(fill=tk.X, padx=5, pady=5)

        # 新增：显示时长(s) 输入框（0 = 不限制）
        ttk.Label(chart_btn_frame, text="Display Duration (s):").pack(side=tk.LEFT, padx=(0, 4))
        self.plot_window_seconds_var = tk.DoubleVar(value=30.0)  # 默认 30 秒，只显示最近 30s 数据
        self.plot_window_entry = ttk.Entry(chart_btn_frame, textvariable=self.plot_window_seconds_var, width=7)
        self.plot_window_entry.pack(side=tk.LEFT, padx=(0, 10))

        self.export_btn = ttk.Button(
            chart_btn_frame, text="Export Chart",
            command=self.export_chart, state="disabled", bootstyle="info"
        )
        self.export_btn.pack(side=tk.LEFT, padx=5, pady=2)

        self.clear_chart_btn = ttk.Button(
            chart_btn_frame, text="Clear Chart",
            command=self.clear_chart, bootstyle="warning"
        )
        self.clear_chart_btn.pack(side=tk.LEFT, padx=5, pady=2)

        if self._use_internal_log:
            log_frame = ttk.LabelFrame(right_panel, text="日志")
            log_frame.pack(fill=tk.BOTH, padx=5, pady=5)
            self.log_text = scrolledtext.ScrolledText(log_frame, state="disabled", height=10)
            self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.refresh_ports()

    # ====== 新增函数：设置指示灯颜色 ======
    def set_indicator_color(self, color):
        self.indicator_canvas.itemconfig(self.indicator_circle, fill=color)
        if color == "green":
            self.after(500, lambda: self.set_indicator_color("red"))

    def set_sim_click_point(self):
        pos = capture_click_point(
            self,
            title="设置点击点",
            hint="移动鼠标到目标软件按钮处，按 Enter 键记录",
            reporter=self.log,
        )
        if pos:
            self.sim_click_pos = pos

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

        coords = {'x1': 0, 'y1': 0, 'x2': 0, 'y2': 0, 'rect': None}

        def on_press(event):
            coords['x1'], coords['y1'] = event.x, event.y
            if coords['rect']:
                canvas.delete(coords['rect'])
            coords['rect'] = canvas.create_rectangle(coords['x1'], coords['y1'], event.x, event.y, outline="red", width=2)

        def on_drag(event):
            canvas.coords(coords['rect'], coords['x1'], coords['y1'], event.x, event.y)

        def on_release(event):
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
                    screen_w, screen_h = pyautogui.size()
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
                snapshot = pyautogui.screenshot(region=region)
                if snapshot.mode != 'RGB':
                    snapshot = snapshot.convert('RGB')

            except ValueError:
                self.log('检测区域无效：选取的区域太小')
                messagebox.showwarning('无效区域', '选取的检测区域太小，请拖拽一个更大的区域。')
            except Exception as e:
                self.log(f'设置检测区域失败: {e}')
                messagebox.showerror('错误', f'无法截取屏幕区域: {e}')
            else:
                self.pixel_detection_region = region
                self.initial_pixel_snapshot = snapshot
                self.prev_pixel_snapshot = None
                self.log(f"已设置检测区域: {self.pixel_detection_region}")
            finally:
                overlay.destroy()
            return "break"

        canvas.bind("<ButtonPress-1>", on_press)
        canvas.bind("<B1-Motion>", on_drag)
        canvas.bind("<ButtonRelease-1>", on_release)
        overlay.bind("<Return>", on_enter)

    def wait_for_pixel_change(self, timeout: float, interval: float) -> bool:
        if not self.pixel_detection_region or self.initial_pixel_snapshot is None:
            return False

        baseline = self.initial_pixel_snapshot
        start_time = time.time()

        while time.time() - start_time < timeout and self.multi_pressure_running:
            try:
                current = pyautogui.screenshot(region=self.pixel_detection_region)
                diff = ImageChops.difference(baseline, current)
                if diff.getbbox():
                    self.set_indicator_color("green")
                    return True
            except Exception:
                pass
            time.sleep(interval)
        return False

    def check_pixel_change(self):
        if not self.pixel_detection_region:
            return False, "检测区域未设置"
        try:
            curr = pyautogui.screenshot(region=self.pixel_detection_region)
            base = self.prev_pixel_snapshot or self.initial_pixel_snapshot
            diff = ImageChops.difference(base, curr)
            self.prev_pixel_snapshot = curr
            diff_data = diff.getdata()
            diff_pixels = sum(1 for px in diff_data if px != (0, 0, 0))
            total = curr.width * curr.height
            pct = diff_pixels / total * 100
            threshold = self.pixel_sensitivity_var.get()
            if pct >= threshold:
                return True, f"检测到像素变化: {pct:.2f}%"
            else:
                return False, f"无显著变化: {pct:.2f}%"
        except Exception as e:
            return False, f"检测失败: {e}"

    def start_multi_pressure_test(self):
        if not (self.modbus1 and self.modbus2):
            messagebox.showerror("错误", "请先连接压力传感器和运动控制器")
            return

        if self.sim_click_enabled.get() and not self.sim_click_pos:
            messagebox.showerror("错误", "请先设置模拟点击点")
            return
        if self.pixel_detection_enabled.get() and not self.pixel_detection_region:
            messagebox.showerror("错误", "请先设置像素检测区域")
            return

        try:
            points = [float(p) for p in self.pressure_points_var.get().split(",") if p.strip()]
            if not points:
                raise ValueError
        except ValueError:
            messagebox.showerror("错误", "压力点格式错误，请使用逗号分隔数字")
            return

        try:
            tolerance          = float(self.tolerance_var.get())
            stable_time        = float(self.stable_time_var.get())
            loop_mode          = self.loop_mode_var.get()
            loop_count         = max(1, int(self.loop_count_var.get()))
            detection_interval = self.detection_interval_var.get() / 1000.0
            pressure_step_int  = float(self.pressure_step_interval_var.get())
            pixel_sense        = float(self.pixel_sensitivity_var.get())
            pixel_timeout      = float(self.pixel_timeout_var.get())
        except ValueError:
            messagebox.showerror("错误", "参数格式错误，请确认数值")
            return

        if loop_mode == "顺序":
            seq = points
        elif loop_mode == "倒序":
            seq = list(reversed(points))
        else:
            seq = points + list(reversed(points))

        self._maybe_prompt_auto_tare()
        self.multi_pressure_running = True
        threading.Thread(
            target=self.multi_pressure_test_loop,
            args=(seq, tolerance, stable_time,
                  detection_interval, pressure_step_int,
                  pixel_sense, pixel_timeout, loop_count),
            daemon=True
        ).start()
        self.log(f"启动多压力测试: {seq}  循环次数={loop_count}  判稳={stable_time}s")

    def multi_pressure_test_loop(
            self,
            pressure_sequence,
            tolerance,
            stable_time,
            detection_interval,
            pressure_step_interval,
            pixel_sensitivity,
            pixel_timeout,
            loop_count
    ):
        ori_target = self.target_pressure_var.get()

        try:
            for round_id in range(1, loop_count + 1):
                if not self.multi_pressure_running:
                    break
                self.log(f"======== 第 {round_id}/{loop_count} 轮 ========")
                for target in pressure_sequence:
                    if not self.multi_pressure_running:
                        break

                    self.target_pressure_var.set(target)
                    if not self.pressure_control_running:
                        self.start_pressure_control()

                    in_range_t0 = None
                    while self.multi_pressure_running:
                        cur_p = self.read_pressure()
                        if abs(cur_p - target) <= tolerance:
                            if in_range_t0 is None:
                                in_range_t0 = time.time()
                            if time.time() - in_range_t0 >= stable_time:
                                break
                        else:
                            in_range_t0 = None
                        time.sleep(0.05)

                    if not self.multi_pressure_running:
                        break

                    self.stop_pressure_control()
                    time.sleep(0.10)

                    if self.sim_click_enabled.get():
                        self.log("执行模拟点击…")
                        self.perform_sim_click()
                        self.perform_sim_click()
                        try:
                            self.initial_pixel_snapshot = pyautogui.screenshot(region=self.pixel_detection_region)
                            self.log("已刷新基准快照")
                        except Exception as e:
                            self.log(f"刷新基准快照失败: {e}")

                        self.log("冻结 5 秒，开始倒计时…")
                        for remaining in range(5, 0, -1):
                            self.log(f"倒计时：{remaining} s")
                            time.sleep(1)
                        self.log("倒计时结束，开始像素检测…")

                    if self.pixel_detection_enabled.get():
                        self.log(f"开始像素检测 (超时 {pixel_timeout}s，间隔 {detection_interval}s)…")
                        detected = self.wait_for_pixel_change(pixel_timeout, detection_interval)
                        if detected:
                            self.log(f"✅ {target}g：检测到像素变化，继续下一步")
                        else:
                            self.log(f"⚠️ {target}g：像素检测超时，跳过当前压力点")
                    else:
                        time.sleep(pressure_step_interval)

                    self.log(f"✓ {target} g 完成")

                    self.log("冷却 5 秒，开始倒计时…")
                    for remaining in range(5, 0, -1):
                        self.log(f"倒计时：{remaining} s")
                        time.sleep(1)
                    self.log("冷却结束，前往下一个压力点…")

            self.log("多压力测试完成")
        except Exception as e:
            self.log(f"多压力测试异常: {e}")
        finally:
            self.target_pressure_var.set(ori_target)
            self.multi_pressure_running = False

    def precise_pressure_control(self, target, tolerance, timeout=60.0):
        self.log(f"开始精准控制压力到: {target}g")
        start_time = time.time()
        direction_changed = False
        self.direction.set("下")

        self.speed_entry.delete(0, tk.END); self.speed_entry.insert(0, str(self.high_speed))
        self.distance_entry.delete(0, tk.END); self.distance_entry.insert(0, "20.0")
        self.set_parameters(); self.start_machine()

        while self.multi_pressure_running:
            if time.time() - start_time > timeout:
                self.log(f"压力控制超时 ({timeout}s)，终止控制")
                self.stop_motion()
                return False

            current_pressure = self.read_pressure()
            diff = target - current_pressure
            if diff < 100:
                break
            if current_pressure > target + tolerance:
                self.direction.set("上")
                self.speed_entry.delete(0, tk.END); self.speed_entry.insert(0, str(self.high_speed))
                self.set_parameters(); self.start_machine()
                direction_changed = True
                self.log("压力超过目标值，切换为上移")
            time.sleep(0.1)

        self.speed_entry.delete(0, tk.END); self.speed_entry.insert(0, str(self.low_speed))
        self.set_parameters()

        while self.multi_pressure_running:
            if time.time() - start_time > timeout:
                self.log(f"压力控制超时 ({timeout}s)，终止控制")
                self.stop_motion()
                return False

            current_pressure = self.read_pressure()
            if abs(current_pressure - target) <= tolerance:
                self.stop_motion()
                self.log(f"压力达标并停止，当前压力: {current_pressure:.1f}g")
                return True

            if current_pressure < target:
                if self.direction.get() != "下" or direction_changed:
                    self.direction.set("下")
                    self.speed_entry.delete(0, tk.END); self.speed_entry.insert(0, str(self.low_speed))
                    self.set_parameters()
                    direction_changed = True
                self.start_machine()
            else:
                if self.direction.get() != "上" or direction_changed:
                    self.direction.set("上")
                    self.speed_entry.delete(0, tk.END); self.speed_entry.insert(0, str(self.low_speed))
                    self.set_parameters()
                    direction_changed = True
                self.start_machine()
            time.sleep(0.1)
        return False

    def connect_sensor(self, port: str, baudrate: int, suppress_error: bool = False) -> bool:
        if not port:
            if not suppress_error:
                messagebox.showerror("错误", "请选择压力传感器的串口")
            return False

        try:
            modbus = ModbusController(port, baudrate, slave_id=1)
        except Exception as e:
            self.log(f"连接压力传感器失败: {e}")
            self.connection_status_var1.set("未连接")
            if not suppress_error:
                messagebox.showerror("错误", str(e))
            return False

        if self.modbus1:
            try:
                self.modbus1.close()
            except Exception:
                pass

        self.modbus1 = modbus
        if hasattr(self, 'port_cb1'):
            self.port_cb1.set(port)
        if hasattr(self, 'baud_entry1'):
            self.baud_entry1.delete(0, tk.END)
            self.baud_entry1.insert(0, str(baudrate))
        self.log(f"连接到压力传感器: {port} @ {baudrate} bps")
        self.connection_status_var1.set("已连接")
        if hasattr(self, 'sensor_btn'):
            self.sensor_btn.config(text="断开", bootstyle="danger")
        self.sensor_connected = True

        if isinstance(self.loaded_config, dict):
            self.loaded_config['serial_port1'] = port
            self.loaded_config['baud_rate1'] = str(baudrate)

        if not hasattr(self, 'pressure_thread') or not self.pressure_thread or not self.pressure_thread.is_alive():
            self.pressure_thread_running = True
            self.pressure_thread = threading.Thread(target=self.poll_pressure_data, daemon=True)
            self.pressure_thread.start()

        if self.controller_connected and (not hasattr(self, 'position_thread') or not self.position_thread or not self.position_thread.is_alive()):
            self.position_thread_running = True
            self.position_thread = threading.Thread(target=self.poll_position_data, daemon=True)
            self.position_thread.start()

        self._refresh_connection_dependent_controls()
        return True

    def disconnect_sensor(self):
        self.pressure_thread_running = False
        if self.modbus1:
            try:
                self.modbus1.close()
            except Exception:
                pass
            self.modbus1 = None
        self.log("压力传感器已断开")
        self.connection_status_var1.set("未连接")
        if hasattr(self, 'sensor_btn'):
            self.sensor_btn.config(text="连接", bootstyle="outline-success")
        self.sensor_connected = False
        self._refresh_connection_dependent_controls()

    def connect_controller(self, port: str, baudrate: int, suppress_error: bool = False) -> bool:
        if not port:
            if not suppress_error:
                messagebox.showerror("错误", "请选择运动控制器的串口")
            return False

        try:
            modbus = ModbusController(port, baudrate, slave_id=1)
        except Exception as e:
            self.log(f"连接运动控制器失败: {e}")
            self.connection_status_var2.set("未连接")
            if not suppress_error:
                messagebox.showerror("错误", str(e))
            return False

        if self.modbus2:
            try:
                self.modbus2.close()
            except Exception:
                pass

        self.modbus2 = modbus
        if hasattr(self, 'port_cb2'):
            self.port_cb2.set(port)
        if hasattr(self, 'baud_entry2'):
            self.baud_entry2.delete(0, tk.END)
            self.baud_entry2.insert(0, str(baudrate))
        self.log(f"连接到运动控制器: {port} @ {baudrate} bps")
        self.connection_status_var2.set("已连接")
        if hasattr(self, 'controller_btn'):
            self.controller_btn.config(text="断开", bootstyle="danger")
        self.controller_connected = True

        if isinstance(self.loaded_config, dict):
            self.loaded_config['serial_port2'] = port
            self.loaded_config['baud_rate2'] = str(baudrate)

        if not hasattr(self, 'position_thread') or not self.position_thread or not self.position_thread.is_alive():
            self.position_thread_running = True
            self.position_thread = threading.Thread(target=self.poll_position_data, daemon=True)
            self.position_thread.start()

        if self.sensor_connected and (not hasattr(self, 'pressure_thread') or not self.pressure_thread or not self.pressure_thread.is_alive()):
            self.pressure_thread_running = True
            self.pressure_thread = threading.Thread(target=self.poll_pressure_data, daemon=True)
            self.pressure_thread.start()

        self._refresh_connection_dependent_controls()
        return True

    def disconnect_controller(self):
        self.position_thread_running = False
        if self.modbus2:
            try:
                self.modbus2.close()
            except Exception:
                pass
            self.modbus2 = None
        self.log("运动控制器已断开")
        self.connection_status_var2.set("未连接")
        if hasattr(self, 'controller_btn'):
            self.controller_btn.config(text="连接", bootstyle="outline-success")
        self.controller_connected = False
        self._refresh_connection_dependent_controls()

    def toggle_sensor_connection(self):
        if not hasattr(self, 'port_cb1'):
            return

        try:
            baudrate = int(self.baud_entry1.get())
        except ValueError:
            messagebox.showerror("错误", "压力传感器波特率必须为数字")
            return

        if not self.sensor_connected:
            port = self.port_cb1.get()
            self.connect_sensor(port, baudrate, suppress_error=False)
        else:
            self.disconnect_sensor()

    def toggle_controller_connection(self):
        if not hasattr(self, 'port_cb2'):
            return

        try:
            baudrate = int(self.baud_entry2.get())
        except ValueError:
            messagebox.showerror("错误", "运动控制器波特率必须为数字")
            return

        if not self.controller_connected:
            port = self.port_cb2.get()
            self.connect_controller(port, baudrate, suppress_error=False)
        else:
            self.disconnect_controller()

    def show_serial_settings(self):
        dialog = ttk.Toplevel(self)
        dialog.title("串口设置")
        dialog.geometry("400x300")
        dialog.transient(self)
        dialog.grab_set()

        frame = ttk.Frame(dialog, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, text="压力传感器端口:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        port_cb1 = ttk.Combobox(frame, width=15)
        port_cb1.grid(row=0, column=1, padx=5, pady=5)
        port_cb1['values'] = [port.device for port in serial.tools.list_ports.comports()]
        port_cb1.set(self.port_cb1.get() if hasattr(self, 'port_cb1') else "COM3")
        self._disable_mouse_wheel(port_cb1)

        ttk.Label(frame, text="波特率:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        baud_entry1 = ttk.Entry(frame, width=8)
        baud_entry1.grid(row=1, column=1, padx=5, pady=5)
        baud_entry1.insert(0, self.baud_entry1.get() if hasattr(self, 'baud_entry1') else "9600")
        self._disable_mouse_wheel(baud_entry1)

        ttk.Label(frame, text="运动控制器端口:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        port_cb2 = ttk.Combobox(frame, width=15)
        port_cb2.grid(row=2, column=1, padx=5, pady=5)
        port_cb2['values'] = [port.device for port in serial.tools.list_ports.comports()]
        port_cb2.set(self.port_cb2.get() if hasattr(self, 'port_cb2') else "COM24")
        self._disable_mouse_wheel(port_cb2)

        ttk.Label(frame, text="波特率:").grid(row=3, column=0, padx=5, pady=5, sticky="w")
        baud_entry2 = ttk.Entry(frame, width=8)
        baud_entry2.grid(row=3, column=1, padx=5, pady=5)
        baud_entry2.insert(0, self.baud_entry2.get() if hasattr(self, 'baud_entry2') else "9600")
        self._disable_mouse_wheel(baud_entry2)

        def save_settings():
            if hasattr(self, 'port_cb1'):
                self.port_cb1.set(port_cb1.get())
            if hasattr(self, 'baud_entry1'):
                self.baud_entry1.delete(0, tk.END)
                self.baud_entry1.insert(0, baud_entry1.get())
            if hasattr(self, 'port_cb2'):
                self.port_cb2.set(port_cb2.get())
            if hasattr(self, 'baud_entry2'):
                self.baud_entry2.delete(0, tk.END)
                self.baud_entry2.insert(0, baud_entry2.get())
            self.log("串口设置已更新")
            dialog.destroy()

        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=4, column=0, columnspan=2, pady=10)
        ttk.Button(btn_frame, text="保存", command=save_settings, bootstyle="success").pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="取消", command=dialog.destroy, bootstyle="secondary").pack(side=tk.LEFT, padx=5)

    def show_calibration_settings(self):
        dialog = ttk.Toplevel(self)
        dialog.title("校准设置")
        dialog.geometry("300x200")
        dialog.transient(self)
        dialog.grab_set()

        frame = ttk.Frame(dialog, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, text="砝码值 (g):").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        weight_entry = ttk.Entry(frame, width=10)
        weight_entry.grid(row=0, column=1, padx=5, pady=5)
        weight_entry.insert(0, "1000")

        def calibrate():
            try:
                weight = int(weight_entry.get())
                self.weight_calibration(weight)
                self.log(f"砝码校准已执行，砝码值: {weight}g")
            except ValueError:
                messagebox.showerror("错误", "请输入有效的砝码值")

        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=1, column=0, columnspan=2, pady=10)
        ttk.Button(btn_frame, text="执行砝码校准", command=calibrate, bootstyle="primary").pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="关闭", command=dialog.destroy, bootstyle="secondary").pack(side=tk.LEFT, padx=5)

    def show_advanced_settings(self):
        dialog = ttk.Toplevel(self)
        dialog.title("高级设置")
        dialog.geometry("400x300")
        dialog.transient(self)
        dialog.grab_set()

        frame = ttk.Frame(dialog, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, text="位置读取间隔:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        pos_interval_entry = ttk.Entry(frame, width=8)
        pos_interval_entry.grid(row=0, column=1, padx=5, pady=5)
        pos_interval_entry.insert(0, self.pos_interval_entry.get() if hasattr(self, 'pos_interval_entry') else "0.01")

        ttk.Label(frame, text="压力读取间隔:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        press_interval_entry = ttk.Entry(frame, width=8)
        press_interval_entry.grid(row=1, column=1, padx=5, pady=5)
        press_interval_entry.insert(0, self.press_interval_entry.get() if hasattr(self,'press_interval_entry') else "0.01")

        ttk.Label(frame, text="高速 (mm/s):").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        high_speed_entry = ttk.Entry(frame, width=8)
        high_speed_entry.grid(row=2, column=1, padx=5, pady=5)
        high_speed_entry.insert(0, str(self.high_speed))

        ttk.Label(frame, text="低速 (mm/s):").grid(row=3, column=0, padx=5, pady=5, sticky="w")
        low_speed_entry = ttk.Entry(frame, width=8)
        low_speed_entry.grid(row=3, column=1, padx=5, pady=5)
        low_speed_entry.insert(0, str(self.low_speed))

        ttk.Label(frame, text="压力容差 (g):").grid(row=4, column=0, padx=5, pady=5, sticky="w")
        tolerance_entry = ttk.Entry(frame, width=8)
        tolerance_entry.grid(row=4, column=1, padx=5, pady=5)
        tolerance_entry.insert(0, str(self.pressure_tolerance))

        def save_settings():
            if hasattr(self, 'pos_interval_entry'):
                self.pos_interval_entry.delete(0, tk.END); self.pos_interval_entry.insert(0, pos_interval_entry.get())
            if hasattr(self, 'press_interval_entry'):
                self.press_interval_entry.delete(0, tk.END); self.press_interval_entry.insert(0, press_interval_entry.get())
            self.high_speed = float(high_speed_entry.get())
            self.low_speed = float(low_speed_entry.get())
            self.pressure_tolerance = float(tolerance_entry.get())
            self.log("高级设置已更新")
            dialog.destroy()
            self.save_config()

        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=5, column=0, columnspan=2, pady=10)
        ttk.Button(btn_frame, text="保存", command=save_settings, bootstyle="success").pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="取消", command=dialog.destroy, bootstyle="secondary").pack(side=tk.LEFT, padx=5)

    def show_about(self):
        messagebox.showinfo("关于", "压力控制综合程序\n版本: 4.0\n开发日期: 2025-06-12")

    def show_instructions(self):
        instructions = """
=========================== 压力控制综合程序 使用说明 ===========================

1. 快速开始
   1.1 串口设置：
       - 菜单 → 设置 → 串口设置，选择压力传感器和运动控制器的端口与波特率。
   1.2 连接设备：
       - 在“设备连接”区域分别点击“连接”。

2. 界面布局
   - 左侧控制面板：运动控制、压力控制、多压力测试、跳变设置…
   - 右侧图表区：时间—位置/压力曲线，可导出或清除。
   - 底部日志区：记录所有操作和设备反馈。

3. 运动控制
   - 选“上/下”，输入速度/距离，先“设置参数”再“启动运动”。

4. 压力与保护
   - 大字显示当前压力；触发保护压力后红黄闪烁报警。
   - “启动压力控制”后：高速靠近 → 低速微调 → 容差内保持。

5. 自动多压力测试
   - 填写压力点（逗号分隔）与循环模式，可选像素检测与模拟点击。
   - 三个核心参数：
     • 像素检测间隔(ms)
     • 压力步进间隔(s)     ← 已统一为“秒”
     • 超时(s)

6. 跳变功能
   - 为 1–8 跳变点设阈值、方向(正/反/双向)、是否自动关闭。

7. 校准 & 去皮
   - 菜单 → 工具：硬件去皮 / 零点校准 / 砝码校准 / 写保护。

8. 导出
   - “导出图表”生成带图的 Excel；“清除图表”重置历史。
"""
        dialog = ttk.Toplevel(self)
        dialog.title("使用说明")
        dialog.geometry("1000x900")
        dialog.transient(self)
        dialog.grab_set()

        text = scrolledtext.ScrolledText(dialog, wrap=tk.WORD, font=("微软雅黑", 12))
        text.pack(expand=True, fill='both', padx=12, pady=12)
        text.insert("1.0", instructions)
        text.config(state='disabled')

        btn = ttk.Button(dialog, text="关闭", command=dialog.destroy, bootstyle="secondary")
        btn.pack(pady=8)

    def set_params_modified(self):
        self.params_modified = True

    @staticmethod
    def _disable_mouse_wheel(widget):
        """禁止鼠标滚轮改变特定输入部件的值。"""
        if not widget:
            return

        def _block(event):
            return "break"

        widget.bind("<MouseWheel>", _block)
        widget.bind("<Button-4>", _block)
        widget.bind("<Button-5>", _block)

    def _get_baudrate(self, attr_name: str, default: int = 9600) -> int:
        entry = getattr(self, attr_name, None)
        if entry is not None:
            try:
                return int(entry.get())
            except (ValueError, tk.TclError):
                pass
        return default

    def _serial_auto_candidates(self, combo_attr: str, config_key: str, port_list):
        candidates = []
        combo = getattr(self, combo_attr, None)
        if combo is not None:
            current = combo.get()
            if current and current in port_list:
                candidates.append(current)

        if isinstance(self.loaded_config, dict):
            saved = self.loaded_config.get(config_key, '') or ''
            if saved and saved in port_list and saved not in candidates:
                candidates.append(saved)

        for port in port_list:
            if port not in candidates:
                candidates.append(port)

        return candidates

    def refresh_ports(self):
        try:
            ports = list(serial.tools.list_ports.comports())
        except Exception as e:
            ports = []
            self.log(f"刷新串口列表失败: {e}")

        port_list = [port.device for port in ports]

        def _update_combo(attr_name: str, config_key: str):
            if not hasattr(self, attr_name):
                return
            combo = getattr(self, attr_name)
            combo['values'] = port_list
            preferred = ''
            if isinstance(self.loaded_config, dict):
                preferred = self.loaded_config.get(config_key, '') or ''

            current = combo.get()
            if current in port_list:
                new_value = current
            elif preferred in port_list:
                new_value = preferred
            else:
                new_value = ''

            combo.set(new_value)

        _update_combo('port_cb1', 'serial_port1')
        _update_combo('port_cb2', 'serial_port2')

        if port_list:
            self.log("串口列表已刷新: {}".format(', '.join(port_list)))
        else:
            self.log("未检测到可用串口")

    def auto_connect_devices(self):
        try:
            ports = list(serial.tools.list_ports.comports())
        except Exception as e:
            self.log(f"自动连接失败：无法获取串口列表 ({e})")
            messagebox.showerror("错误", f"无法获取串口列表：{e}")
            return

        port_list = [port.device for port in ports]
        if not port_list:
            self.log("自动连接失败：未检测到可用串口")
            messagebox.showwarning("自动连接", "未检测到可用串口，请检查USB插口。")
            return

        if self.sensor_connected and self.controller_connected:
            self.log("自动连接：设备已连接")
            return

        used_ports = set()
        if self.sensor_connected and self.modbus1:
            used_ports.add(getattr(self.modbus1, 'port_name', self.port_cb1.get()))
        if self.controller_connected and self.modbus2:
            used_ports.add(getattr(self.modbus2, 'port_name', self.port_cb2.get()))

        sensor_connected = self.sensor_connected
        controller_connected = self.controller_connected

        sensor_baud = self._get_baudrate('baud_entry1')
        controller_baud = self._get_baudrate('baud_entry2')

        if not sensor_connected:
            for port in self._serial_auto_candidates('port_cb1', 'serial_port1', port_list):
                if not port or port in used_ports:
                    continue
                self.log(f"自动连接尝试压力传感器: {port}")
                if self.connect_sensor(port, sensor_baud, suppress_error=True):
                    used_ports.add(port)
                    sensor_connected = True
                    break

        if not controller_connected:
            for port in self._serial_auto_candidates('port_cb2', 'serial_port2', port_list):
                if not port or port in used_ports:
                    continue
                self.log(f"自动连接尝试运动控制器: {port}")
                if self.connect_controller(port, controller_baud, suppress_error=True):
                    used_ports.add(port)
                    controller_connected = True
                    break

        failures = []
        if not sensor_connected:
            failures.append("压力传感器")
        if not controller_connected:
            failures.append("运动控制器")

        if failures:
            hint = "、".join(failures)
            self.log(f"自动连接失败：{hint}")
            messagebox.showwarning("自动连接", f"未能连接到{hint}，请检查USB插口或设备状态。")
        else:
            self.log("自动连接成功")

    def log(self, message):
        if callable(self._external_log):
            try:
                self._external_log(message)
            except Exception:
                pass
        if not self._use_internal_log or not self.log_text:
            return

        import threading, time
        if threading.current_thread() is not threading.main_thread():
            self.after(0, lambda m=message: self.log(m))
            return

        ts = time.strftime('%H:%M:%S')
        self.log_text.config(state="normal")
        self.log_text.insert("end", f"{ts} - {message}\n")

        try:
            max_lines = int(getattr(self, "LOG_MAX_LINES", 200))
        except Exception:
            max_lines = 200

        line_count = int(self.log_text.index('end-1c').split('.')[0])
        if line_count > max_lines:
            excess = line_count - max_lines
            self.log_text.delete('1.0', f'{excess + 1}.0')

        self.log_text.see("end")
        self.log_text.config(state="disabled")

    def _maybe_prompt_auto_tare(self):
        if self._auto_tare_prompted:
            return

        self._auto_tare_prompted = True

        import threading

        def _ask_and_handle():
            answer = messagebox.askyesno("自动去皮", "是否自动去皮？")
            if answer:
                self.after(0, self.start_auto_tare)
            if event:
                event.set()

        event = None
        if threading.current_thread() is threading.main_thread():
            _ask_and_handle()
        else:
            event = threading.Event()

            def _wrapped():
                try:
                    _ask_and_handle()
                finally:
                    if event and not event.is_set():
                        event.set()

            self.after(0, _wrapped)
            event.wait()

    def set_parameters(self):
        if not self.modbus2:
            return
        try:
            speed = float(self.speed_entry.get())
            distance = float(self.distance_entry.get())
        except ValueError:
            messagebox.showerror("错误", "速度和距离必须为数字")
            return

        speed_val = int(speed * 1000)
        distance_val = int(distance * 1000)
        direction = self.direction.get()
        op_mode = 1 if direction == "上" else 0

        self.log(f"设置参数: 方向={direction}({op_mode}), 速度={speed_val}, 距离={distance_val}")

        try:
            resp2 = self.modbus2.write_registers(170, [op_mode, 0], delay=0.005)
            self.log(f"运动控制器设置工作方式: {resp2.hex()}")
            resp2 = self.modbus2.write_registers(174, [distance_val, 0], delay=0.005)
            self.log(f"运动控制器设置距离: {resp2.hex()}")
            resp2 = self.modbus2.write_registers(176, [speed_val, 0], delay=0.005)
            self.log(f"运动控制器设置速度: {resp2.hex()}")
        except Exception as e:
            self.log(f"设置参数失败: {e}")

        self.params_modified = False
        self.last_direction = direction
        self.last_speed = speed
        self.last_distance = distance

    def start_machine(self):
        if self.params_modified:
            messagebox.showwarning("提示", "您修改了控制参数但尚未点击'设置'，请先点击'设置'后再启动。")
            return
        if not self.modbus2:
            return

        self._maybe_prompt_auto_tare()
        direction = self.direction.get()
        speed = float(self.speed_entry.get())
        if direction == "下" and speed > 0.5:
            if not messagebox.askyesno("高速下压警告",
                                       f"您即将以 {speed} mm/s 的速度下压，这可能导致设备损坏！\n是否继续？"):
                return

        self.log("发送启动命令")
        try:
            resp2 = self.modbus2.write_registers(178, [1, 0], delay=0.005)
            self.log(f"运动控制器启动命令: {resp2.hex()}")
        except Exception as e:
            self.log(f"启动命令失败: {e}")

    def return_zero(self):
        if not self.modbus2:
            return

        self._maybe_prompt_auto_tare()
        self.log("发送回零命令")
        try:
            resp2 = self.modbus2.write_registers(170, [3, 0], delay=0.005)
            self.log(f"运动控制器设置回零模式: {resp2.hex()}")
            resp2 = self.modbus2.write_registers(178, [1, 0], delay=0.005)
            self.log(f"运动控制器启动回零: {resp2.hex()}")
        except Exception as e:
            self.log(f"回零命令失败: {e}")

    def move_to(self):
        if not self.modbus2:
            return
        try:
            target = float(self.target_entry.get())
        except ValueError:
            messagebox.showerror("错误", "目标位置必须为数字")
            return

        self._maybe_prompt_auto_tare()
        target_val = int(target * 1000)
        self.log(f"运动到目标位置: {target_val}")
        try:
            resp2 = self.modbus2.write_registers(174, [target_val, 0], delay=0.005)
            self.log(f"运动控制器设置目标位置: {resp2.hex()}")
            resp2 = self.modbus2.write_registers(178, [1, 0], delay=0.005)
            self.log(f"运动控制器启动转到: {resp2.hex()}")
        except Exception as e:
            self.log(f"运动到目标位置失败: {e}")

    def stop_motion(self):
        """异步下发停止命令，不阻塞 Tk 主线程。"""
        if not self.modbus2:
            return
        threading.Thread(target=self._stop_motion_worker, daemon=True).start()

    def _stop_motion_worker(self):
        try:
            resp2 = self.modbus2.write_registers(180, [1, 0], delay=0.005)
            self.log(f"运动控制器停止命令: {resp2.hex() if resp2 else ''}")
        except Exception as e:
            self.log(f"停止命令失败: {e}")
        finally:
            self._no_autostart_until = time.time() + 0.6
            self._mc_last_cmd_ts = 0.0
            self._mc_last_dir = self._mc_last_speed = self._mc_last_distance = None
            self.set_var_safe(self.current_speed_var, "0.000")

    def stop_all(self):
        self.stop_motion()
        self.jump_running = False
        self.goto_mode_set = False
        self.pressure_control_running = False
        self.multi_pressure_running = False
        self.pixel_detection_enabled.set(False)
        self.sim_click_enabled.set(False)
        self.auto_tare_running = False
        if (hasattr(self, 'pressure_control_thread') and
                self.pressure_control_thread and
                self.pressure_control_thread.is_alive()):
            for _ in range(5):
                time.sleep(0.05)
        self.log("所有控制已停止（包含压力控制、像素检测、模拟点击）")

    def parse_pressure_response(self, response):
        """
        将 03 功能码回复解析为压力值（优先 2 字节，兼容 4 字节）。
        """
        try:
            # 标准：01 03 02 [HI] [LO] CRC CRC
            if response and len(response) >= 5 and response[1] == 0x03 and response[2] == 0x02:
                hi, lo = response[3], response[4]
                val = (hi << 8) | lo
                if val >= 0x8000:
                    val -= 0x10000
                return val - self.tare_value

            # 兼容：返回 4 字节
            if response and len(response) >= 7 and response[1] == 0x03 and response[2] == 0x04:
                data = response[3:7]
                low_word = int.from_bytes(data[0:2], 'big', signed=True)
                high_word = int.from_bytes(data[2:4], 'big', signed=True)
                val32 = (high_word << 16) | (low_word & 0xFFFF)
                if val32 >= 0x80000000:
                    val32 -= 0x100000000
                return val32 - self.tare_value

        except Exception as e:
            self.log(f"解析压力数据错误: {e}")

        # 回退：沿用当前值，避免 UI 抖动
        return self.current_pressure

    def poll_pressure_data(self):
        self.pressure_thread_running = True
        while getattr(self, 'pressure_thread_running', True) and self.sensor_connected:
            try:
                press_interval = max(float(self.press_interval_entry.get()), self.MIN_INTERVAL)
            except Exception:
                press_interval = self.MIN_INTERVAL
            try:
                if self.modbus1:
                    pressure = self.read_pressure()
                    ts = time.time()
                    self.current_pressure = pressure
                    self.pressure_queue.put((ts, pressure))
                    self._record_pressure_history(ts, pressure)
                    try:
                        safety_val = float(self.safety_pressure_var.get())
                    except Exception:
                        safety_val = 30000
                    if pressure >= safety_val:
                        self.log(f"⚠️ 保护压力触发！当前压力: {pressure}g > {safety_val}g")
                        self.stop_motion()
                        self.current_speed_var.set("0.000")
                        if not hasattr(self, '_protect_warned') or not self._protect_warned:
                            self._protect_warned = True
                            self.after(0, lambda: messagebox.showwarning("保护压力",
                                                                         f"压力已超过保护值 ({safety_val}g)，已急停！"))
            except Exception as e:
                self.log(f"压力数据读取错误: {e}")
            time.sleep(press_interval)
        self.log("压力线程已结束")

    def _record_pressure_history(self, timestamp, pressure):
        target = float(getattr(self, "target_pressure", 0.0))
        with self._pressure_history_lock:
            self._pressure_history.append((float(timestamp), float(pressure), target))
            cutoff = float(timestamp) - float(self._history_max_seconds)
            while self._pressure_history and self._pressure_history[0][0] < cutoff:
                self._pressure_history.popleft()

    def _compute_pressure_metrics(self, window_s=None):
        if window_s is None:
            window_s = 60.0
        now = time.time()
        cutoff = now - float(max(window_s, 1.0))
        with self._pressure_history_lock:
            samples = [item for item in self._pressure_history if item[0] >= cutoff]
        if not samples:
            return {
                "window": float(window_s),
                "samples": 0,
                "mae": None,
                "rmse": None,
                "last": None,
            }
        errors = [p - t for _, p, t in samples]
        mae = sum(abs(e) for e in errors) / len(errors)
        rmse = math.sqrt(sum(e * e for e in errors) / len(errors))
        last = samples[-1]
        return {
            "window": float(window_s),
            "samples": len(samples),
            "mae": mae,
            "rmse": rmse,
            "last": {
                "timestamp": last[0],
                "pressure": last[1],
                "target": last[2],
                "error": last[1] - last[2],
            },
        }

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
            self.set_target_pressure(request["value"])
            if request.get("start", False):
                self.start_pressure_remote()
            return {"result": "target-updated"}

        if cmd == "start_control":
            ok = self.start_pressure_remote()
            return {"result": ok}

        if cmd == "stop_control":
            self.stop_pressure_remote()
            return {"result": True}

        if cmd == "stop_all":
            self.stop_all()
            return {"result": True}

        raise ValueError(f"unknown cmd: {cmd}")

    def get_tcp_status(self):
        metrics = self._compute_pressure_metrics()
        status = {
            "timestamp": time.time(),
            "pressure": float(self.current_pressure),
            "target": float(getattr(self, "target_pressure", 0.0)),
            "running": bool(self.pressure_control_running),
            "multi_pressure_running": bool(self.multi_pressure_running),
            "mae": metrics.get("mae"),
            "rmse": metrics.get("rmse"),
            "window": metrics.get("window"),
            "samples": metrics.get("samples"),
            "last": metrics.get("last"),
            "tolerance": float(getattr(self, "pressure_tolerance", 0.0)),
        }
        return status

    def set_target_pressure(self, value):
        value = float(value)

        def setter():
            self.target_pressure_var.set(value)

        self.call_in_ui(setter)
        self.target_pressure = value
        return True

    def start_pressure_remote(self):
        def starter():
            return self.start_pressure_control(notify=False)

        return bool(self.call_in_ui(starter))

    def stop_pressure_remote(self):
        self.call_in_ui(self.stop_pressure_control)
        return True

    def poll_position_data(self):
        self.position_thread_running = True
        while getattr(self, 'position_thread_running', True) and self.controller_connected:
            try:
                pos_interval = max(float(self.pos_interval_entry.get()), self.MIN_INTERVAL)
            except Exception:
                pos_interval = self.MIN_INTERVAL
            try:
                if self.modbus2:
                    resp = self.modbus2.read_registers(100, 2)
                    if resp and len(resp) >= 7 and resp[2] >= 4:
                        data = resp[3:7]
                        low_word = int.from_bytes(data[0:2], byteorder='big', signed=True)
                        high_word = int.from_bytes(data[2:4], byteorder='big', signed=True)
                        pos_raw = (high_word << 16) | (low_word & 0xFFFF)
                        if pos_raw >= 0x80000000:
                            pos_raw -= 0x100000000
                        position = pos_raw / 1000.0
                        self.current_position = position
                        self.position_queue.put((time.time(), position))
                    speed = self.read_speed()
                    self.current_speed_var.set(f"{speed:.3f}")
            except Exception as e:
                self.log(f"位置或速度数据读取错误: {e}")
            time.sleep(pos_interval)
        self.log("位置线程已结束")

    def poll_device_data(self):
        try:
            press_interval = max(float(self.press_interval_entry.get()), self.MIN_INTERVAL)
        except Exception:
            press_interval = self.MIN_INTERVAL
        try:
            pos_interval = max(float(self.pos_interval_entry.get()), self.MIN_INTERVAL)
        except Exception:
            pos_interval = self.MIN_INTERVAL

        last_press_time = 0
        last_pos_time = 0

        while self.sensor_connected or self.controller_connected:
            now = time.time()
            if self.sensor_connected and (now - last_press_time >= press_interval):
                try:
                    if self.modbus1:
                        pressure = self.read_pressure()
                        self.current_pressure = pressure
                        self.pressure_queue.put((now, pressure))
                        self._record_pressure_history(now, pressure)
                        try:
                            safety_val = float(self.safety_pressure_var.get())
                        except Exception:
                            safety_val = 2000
                        if pressure >= safety_val:
                            self.log(f"⚠️ 保护压力触发！当前压力: {pressure}g > {safety_val}g")
                            self.stop_motion()
                            self.current_speed_var.set("0.000")
                            messagebox.showwarning("保护压力", f"压力已超过保护值 ({safety_val}g)，已急停！")
                except Exception as e:
                    self.log(f"压力数据读取错误: {e}")
                last_press_time = now

            if self.controller_connected and (now - last_pos_time >= pos_interval):
                try:
                    if self.modbus2:
                        resp = self.modbus2.read_registers(100, 2)
                        if resp and len(resp) >= 7 and resp[2] >= 4:
                            data = resp[3:7]
                            low_word = int.from_bytes(data[0:2], byteorder='big', signed=True)
                            high_word = int.from_bytes(data[2:4], byteorder='big', signed=True)
                            pos_raw = (high_word << 16) | (low_word & 0xFFFF)
                            if pos_raw >= 0x80000000:
                                pos_raw -= 0x100000000
                            position = pos_raw / 1000.0
                            self.current_position = position
                            self.position_queue.put((now, position))
                except Exception as e:
                    self.log(f"位置数据读取错误: {e}")
                last_pos_time = now
            time.sleep(0.005)

    def _normalize_interval_value(self, raw_value, default):
        try:
            value = float(raw_value)
        except (TypeError, ValueError, tk.TclError):
            value = default
        if not math.isfinite(value) or value <= 0:
            value = default
        return max(20.0, min(1000.0, value))

    def _format_interval(self, value: float) -> str:
        try:
            value = float(value)
        except (TypeError, ValueError):
            return "0"
        if abs(value - round(value)) < 1e-6:
            return str(int(round(value)))
        return f"{value:.1f}"

    def _get_interval_ms(self, var: tk.StringVar, default: float) -> float:
        return self._normalize_interval_value(var.get(), default)

    @staticmethod
    def _update_refresh_rate_label(label_var: tk.StringVar, interval_ms: float) -> None:
        hz = 1000.0 / interval_ms if interval_ms > 0 else 0.0
        label_var.set(f"≈ {hz:.1f} Hz")

    def _on_ui_refresh_interval_changed(self, *_args):
        interval_ms = self._get_interval_ms(
            self.ui_refresh_interval_ms_var, self._ui_refresh_default_ms
        )
        self._update_refresh_rate_label(self.ui_refresh_rate_var, interval_ms)
        if getattr(self, "_periodic_jobs_enabled", False):
            self._reschedule_ui_update(interval_ms)

    def _on_plot_refresh_interval_changed(self, *_args):
        interval_ms = self._get_interval_ms(
            self.plot_refresh_interval_ms_var, self._plot_refresh_default_ms
        )
        self._update_refresh_rate_label(self.plot_refresh_rate_var, interval_ms)
        if getattr(self, "_periodic_jobs_enabled", False):
            self._reschedule_plot_update(interval_ms)

    def _reschedule_ui_update(self, interval_ms: float | None = None) -> None:
        if self._ui_update_job is not None:
            try:
                self.after_cancel(self._ui_update_job)
            except Exception:
                pass
            self._ui_update_job = None
        if not getattr(self, "_periodic_jobs_enabled", False):
            return
        if interval_ms is None:
            interval_ms = self._get_interval_ms(
                self.ui_refresh_interval_ms_var, self._ui_refresh_default_ms
            )
        interval_ms = max(1, int(round(interval_ms)))
        self._ui_update_job = self.after(interval_ms, self.update_ui_periodically)

    def _reschedule_plot_update(self, interval_ms: float | None = None) -> None:
        if self._plot_update_job is not None:
            try:
                self.after_cancel(self._plot_update_job)
            except Exception:
                pass
            self._plot_update_job = None
        if not getattr(self, "_periodic_jobs_enabled", False):
            return
        if interval_ms is None:
            interval_ms = self._get_interval_ms(
                self.plot_refresh_interval_ms_var, self._plot_refresh_default_ms
            )
        interval_ms = max(1, int(round(interval_ms)))
        self._plot_update_job = self.after(interval_ms, self.update_plot_periodically)

    def _cancel_periodic_jobs(self) -> None:
        self._periodic_jobs_enabled = False
        for attr in ("_ui_update_job", "_plot_update_job"):
            job = getattr(self, attr, None)
            if job is not None:
                try:
                    self.after_cancel(job)
                except Exception:
                    pass
                setattr(self, attr, None)

    def update_ui_periodically(self):
        self._ui_update_job = None
        if not getattr(self, "_periodic_jobs_enabled", False):
            return

        try:
            self.process_queued_data()
            self.current_pos_var.set(f"{self.current_position:.3f}")
            self.refresh_pressure_display()

            now = time.time()
            if (not self.pixel_detection_enabled.get()
                    or self.pressure_control_running
                    or now < self.pixel_detection_paused_until):
                self._last_pixel_check_ts = now
            else:
                try:
                    interval_s = float(self.detection_interval_var.get()) / 1000.0
                except Exception:
                    interval_s = 1.0
                interval_s = max(interval_s, 0.05)
                if now - getattr(self, "_last_pixel_check_ts", 0.0) >= interval_s:
                    changed, msg = self.check_pixel_change()
                    if self.pixel_log_enabled.get():
                        self.log(f"像素检测: {msg}")
                    if changed:
                        self.set_indicator_color("green")
                    self._last_pixel_check_ts = now

            try:
                safety_val = float(self.safety_pressure_var.get())
            except Exception:
                safety_val = 2000.0

            if self.current_pressure >= safety_val:
                if not self.flashing:
                    self.flashing = True
                    self.start_pressure_flash()
            else:
                if self.flashing:
                    self.flashing = False
                self.big_pressure_label.config(background="white", foreground="black")

        except Exception as exc:  # noqa: BLE001
            try:
                self.log(f"UI 刷新异常: {exc}")
            except Exception:
                pass
        finally:
            if getattr(self, "_periodic_jobs_enabled", False):
                self._reschedule_ui_update()

    def start_pressure_flash(self):
        if not self.flashing:
            self.big_pressure_label.config(background="white", foreground="black")
            return
        if self.flash_state:
            self.big_pressure_label.config(background="yellow", foreground="red")
        else:
            self.big_pressure_label.config(background="red", foreground="yellow")
        self.flash_state = not self.flash_state
        self.after(350, self.start_pressure_flash)

    def update_plot_periodically(self):
        """
        根据 self.plot_window_seconds_var 的时间窗仅绘制最近一段数据：
        - 避免 X 轴上下限相同导致的 singular 警告（只有一个点时自动加余量）
        - 过滤 NaN/Inf
        - 抽稀到最多 ~3000 点，降低重绘负担
        """
        import bisect

        self._plot_update_job = None
        if not getattr(self, "_periodic_jobs_enabled", False):
            return

        MAX_POINTS = 3000  # 每条曲线最多绘制的点数

        try:
            n = min(len(self.time_data), len(self.pos_data), len(self.pressure_data))
            if n < 1:
                return

            t = list(self.time_data[-n:])
            pos = list(self.pos_data[-n:])
            pre = list(self.pressure_data[-n:])

            filtered = [
                (ti, xi, yi)
                for ti, xi, yi in zip(t, pos, pre)
                if (
                    ti is not None
                    and xi is not None
                    and yi is not None
                    and math.isfinite(ti)
                    and math.isfinite(xi)
                    and math.isfinite(yi)
                )
            ]
            if not filtered:
                return
            t, pos, pre = map(list, zip(*filtered))

            try:
                window_s = float(self.plot_window_seconds_var.get())
                if window_s < 0:
                    window_s = 0.0
            except Exception:
                window_s = 0.0

            if window_s > 0 and len(t) > 1:
                t_end = t[-1]
                t_start = t_end - window_s
                i0 = bisect.bisect_left(t, t_start)
                t, pos, pre = t[i0:], pos[i0:], pre[i0:]

                if not t:
                    t, pos, pre = [t_end], [pos[-1]], [pre[-1]]

            n = len(t)
            if n > MAX_POINTS:
                step = max(1, n // MAX_POINTS)
                t = t[::step]
                pos = pos[::step]
                pre = pre[::step]
                n = len(t)

            self.line_pos.set_data(t, pos)
            self.line_press.set_data(t, pre)

            if n >= 1:
                x0, x1 = t[0], t[-1]
                if x1 == x0:
                    pad = max(1e-6, (0.02 * window_s) if window_s > 0 else 1.0)
                    self.ax.set_xlim(x0 - pad, x1 + pad)
                else:
                    self.ax.set_xlim(x0, x1)

            if pos:
                ymin, ymax = min(pos), max(pos)
                pad = max(1e-9, (ymax - ymin) * 0.05 if ymax != ymin else abs(ymin) * 0.05 or 1e-6)
                self.ax.set_ylim(ymin - pad, ymax + pad)

            if pre:
                ymin2, ymax2 = min(pre), max(pre)
                pad2 = max(1e-9, (ymax2 - ymin2) * 0.05 if ymax2 != ymin2 else abs(ymin2) * 0.05 or 1e-6)
                self.ax2.set_ylim(ymin2 - pad2, ymax2 + pad2)

            self.canvas.draw_idle()

        except Exception as e:  # noqa: BLE001
            try:
                self.log(f"绘图更新异常：{e}")
            except Exception:
                pass
        finally:
            if getattr(self, "_periodic_jobs_enabled", False):
                self._reschedule_plot_update()

    def process_queued_data(self):
        while not self.position_queue.empty():
            timestamp, position = self.position_queue.get()
            current_time = timestamp - self.time_start
            self.time_data.append(current_time)
            self.pos_data.append(position)
            self.pressure_data.append(self.current_pressure)

        pressure_interval_threshold = 1.0
        while not self.pressure_queue.empty():
            timestamp, pressure = self.pressure_queue.get()
            current_time_rel = timestamp - self.time_start
            if not self.time_data or timestamp - self.last_pressure_record_time >= pressure_interval_threshold:
                self.time_data.append(current_time_rel)
                self.pos_data.append(self.current_position)
                self.pressure_data.append(pressure)
                self.last_pressure_record_time = timestamp

    def read_position(self):
        pos = 0.0
        try:
            if self.modbus2:
                resp_pos = self.modbus2.read_registers(100, 2)
                if resp_pos and len(resp_pos) >= 7 and resp_pos[2] >= 4:
                    data = resp_pos[3:7]
                    low_word = int.from_bytes(data[0:2], byteorder='big', signed=True)
                    high_word = int.from_bytes(data[2:4], byteorder='big', signed=True)
                    pos_raw = (high_word << 16) | (low_word & 0xFFFF)
                    if pos_raw >= 0x80000000:
                        pos_raw -= 0x100000000
                    pos = pos_raw / 1000.0
        except Exception as e:
            self.log(f"读取位置数据出错: {e}")
        return pos

    def read_speed(self):
        speed = 0.0
        try:
            if self.modbus2:
                resp = self.modbus2.read_registers(110, 2)
                if resp and len(resp) >= 7 and resp[2] == 4:
                    data = resp[3:7]
                    low_word = int.from_bytes(data[0:2], byteorder='big', signed=True)
                    high_word = int.from_bytes(data[2:4], byteorder='big', signed=True)
                    speed_raw = (high_word << 16) | (low_word & 0xFFFF)
                    if speed_raw >= 0x80000000:
                        speed_raw -= 0x100000000
                    speed = speed_raw / 1000.0
        except Exception as e:
            self.log(f"读取速度数据出错: {e}")
        return speed

    def read_pressure(self):
        """
        读取 40001 (地址 0x0000) 的“测量显示值”，按手册为 16 位有符号数。
        返回值单位与设备当前单位一致（你的 UI 按 g 展示，维持现状）。
        """
        pressure = self.current_pressure
        try:
            if not self.modbus1:
                return pressure

            # 优先按“读 1 个寄存器，返回 2 字节数据”的规范读取
            resp = self.modbus1.read_registers(0x0000, 1)  # 40001
            if resp and len(resp) >= 5 and resp[1] == 0x03 and resp[2] == 0x02:
                hi, lo = resp[3], resp[4]
                val = (hi << 8) | lo
                if val >= 0x8000:  # 16 位有符号
                    val -= 0x10000
                pressure = val - self.tare_value
                return pressure

            # 兼容某些固件返回 2 寄存器（4 字节）的旧逻辑（极少用到）
            if resp and len(resp) >= 7 and resp[1] == 0x03 and resp[2] == 0x04:
                data = resp[3:7]
                low_word = int.from_bytes(data[0:2], 'big', signed=True)
                high_word = int.from_bytes(data[2:4], 'big', signed=True)
                val32 = (high_word << 16) | (low_word & 0xFFFF)
                if val32 >= 0x80000000:
                    val32 -= 0x100000000
                pressure = val32 - self.tare_value
                return pressure

        except Exception as e:
            self.log(f"读取压力数据出错: {e}")

        return pressure

    def start_jump_mode(self):
        """根据选择的模式启动跳变：按实际压力 / 按实际位置"""
        if self.jump_running:
            self.log("跳变已在运行中")
            return

        if not (self.modbus1 and self.modbus2):
            messagebox.showerror("错误", "请先连接压力传感器和运动控制器")
            return

        # 只取“开启”的跳变点
        enabled_ids = [i for i in range(1, 9) if self.send_jump_enabled.get(i, False)]
        if not enabled_ids:
            self.log("没有'开启'的跳变点，无法启动")
            return

        mode = (self.jump_mode_var.get() or "").strip()
        target_fn = self._jump_by_position_loop if mode == "按实际位置" else self._jump_by_pressure_loop
        self._maybe_prompt_auto_tare()
        try:
            t = threading.Thread(target=target_fn, args=(enabled_ids,), daemon=True)
            self.jump_running = True
            t.start()
        except Exception as e:
            self.jump_running = False
            self.log(f"启动跳变线程失败: {e}")

    def stop_jump_mode(self):
        """停止跳变"""
        if not self.jump_running:
            return
        self.jump_running = False
        self.stop_motion()
        self.log("跳变被手动停止")

    # ====== 模式 A：按实际压力 ======
    def _jump_by_pressure_sequence(self, enabled_pairs, tolerance_g, jump_speed, interval_s):
        """
        依次把压力跑到每个阈值（g），达到容差即急停；间隔等待，再到下一个阈值
        """
        try:
            for idx, target_g in enabled_pairs:
                if not self.jump_running:
                    break

                ok = self._move_to_pressure_once(target_g, tolerance_g, jump_speed, timeout_s=120)
                if not ok:
                    self.log(f"跳变{idx}: 未能在超时时间内到达 {target_g} g，已跳过")
                else:
                    self.log(f"跳变{idx}: 达到 {target_g} g，已急停")

                # 间隔
                t0 = time.time()
                while self.jump_running and time.time() - t0 < interval_s:
                    time.sleep(0.05)
            self.log("按实际压力跳变完成")
        except Exception as e:
            self.log(f"按实际压力跳变异常: {e}")
        finally:
            self.stop_motion()
            self.current_speed_var.set("0.000")
            self.jump_running = False

    # ====== 模式 B：按实际位置 ======
    def _jump_by_position_sequence(self, enabled_pairs, tolerance_g, collect_speed, jump_speed, interval_s):
        """
        第一步：按压力逐点采集 -> 记录对应的当前位置(mm)
        第二步：切到转到模式，按这些位置做目标位置运动
        """
        positions = []
        try:
            # 1) 逐点到各压力，记录位置
            for idx, target_g in enabled_pairs:
                if not self.jump_running:
                    break
                ok = self._move_to_pressure_once(target_g, tolerance_g, collect_speed, timeout_s=180)
                if not ok:
                    self.log(f"采集{idx}: 未能到达 {target_g} g，跳过该点")
                    continue
                pos = self.read_position()
                positions.append((idx, target_g, pos))
                self.log(f"采集{idx}: 压力 {target_g} g 对应位置 {pos:.3f} mm")

                # 小间隔
                t0 = time.time()
                while self.jump_running and time.time() - t0 < interval_s:
                    time.sleep(0.05)

            if not positions:
                self.log("未采集到任何位置，按实际位置跳变终止")
                return

            # 2) 切换到“转到模式”，用跳变速度逐个跑位置
            self._ensure_goto_mode()
            self._set_goto_speed(jump_speed)

            for idx, target_g, target_pos in positions:
                if not self.jump_running:
                    break
                self.log(f"跳变到记录位置：#{idx} 目标压力 {target_g} g → 目标位置 {target_pos:.3f} mm")

                # 发目标位置并启动
                try:
                    target_val = int(round(target_pos * 1000))
                    self.modbus2.write_registers(174, [target_val, 0], delay=0.005)
                    self.modbus2.write_registers(178, [1, 0], delay=0.005)
                except Exception as e:
                    self.log(f"下发目标位置失败：{e}")
                    continue

                # 等待到位（根据速度/位置双条件）
                ok = self._wait_until_reach_position(target_pos, timeout_s=180)
                if ok:
                    self.log(f"位置到位：{target_pos:.3f} mm")
                else:
                    self.log(f"到位超时：{target_pos:.3f} mm（已继续后续点）")

                # 间隔
                t0 = time.time()
                while self.jump_running and time.time() - t0 < interval_s:
                    time.sleep(0.05)

            self.log("按实际位置跳变完成")
        except Exception as e:
            self.log(f"按实际位置跳变异常: {e}")
        finally:
            self.stop_motion()
            self.current_speed_var.set("0.000")
            self.jump_running = False

    # ====== 通用辅助：在给定速度下把压力跑到 target_g 并急停 ======
    def _move_to_pressure_once(self, target_g, tolerance_g, move_speed, timeout_s=120.0, stable_hold_s=0.2):
        """
        在给定速度 move_speed 下，把压力跑到 target_g：
        - 只在方向/参数变化、或疑似停住时，才下发一次参数+启动（节流去抖）
        - 进入容差并保持 stable_hold_s 秒后急停
        - 超时返回 False
        """
        # 保护阈值
        try:
            safety_g = float(self.safety_pressure_var.get())
        except Exception:
            safety_g = 30000.0

        # 清一次节流记忆，避免上次状态影响
        self._mc_last_dir = None
        self._mc_last_speed = None
        self._mc_last_distance = None
        self._mc_last_cmd_ts = 0.0

        t0 = time.time()
        stable_t0 = None

        while self.jump_running and (time.time() - t0 <= timeout_s):
            cur = self.read_pressure()
            diff = target_g - cur

            # 保护判断
            if cur >= safety_g:
                self.log(f"⚠️ 保护压力触发（{cur:.1f} ≥ {safety_g:.1f} g），已急停")
                self.stop_motion()
                return False

            # 已在容差内 → 计时稳定
            if abs(diff) <= tolerance_g:
                if stable_t0 is None:
                    stable_t0 = time.time()
                if time.time() - stable_t0 >= stable_hold_s:
                    self.stop_motion()
                    self.current_speed_var.set("0.000")
                    return True
            else:
                stable_t0 = None  # 离开容差，重置稳定计时

            # 根据压力差决定方向（若你的硬件方向相反，把 "下"/"上" 互换即可）
            desired_dir = "下" if diff > 0 else "上"

            # 仅在必要时下发一次（节流）
            self._issue_motion_if_needed(desired_dir, move_speed, 20.0, min_interval=0.30)

            time.sleep(0.05)

        # 超时
        self.stop_motion()
        self.current_speed_var.set("0.000")
        return False

    # ====== 低层：转到模式 / 速度 / 到位判断 ======
    def _ensure_goto_mode(self):
        """把控制器切换到“转到模式”（170寄存器=2），只做一次"""
        if not self.goto_mode_set:
            try:
                resp = self.modbus2.write_registers(170, [2, 0], delay=0.005)
                self.log(f"运动控制器设置为转到模式: {resp.hex()}")
                self.goto_mode_set = True
            except Exception as e:
                self.log(f"设置转到模式失败：{e}")

    def _set_goto_speed(self, speed_mm_s: float):
        try:
            val = int(round(speed_mm_s * 1000))
            resp = self.modbus2.write_registers(176, [val, 0], delay=0.005)
            self.log(f"转到模式速度设置为 {speed_mm_s} mm/s: {resp.hex()}")
        except Exception as e:
            self.log(f"设置转到速度失败：{e}")

    def _wait_until_reach_position(self, target_pos_mm: float, timeout_s: float = 15.0) -> bool:
        t0 = time.time()
        last_pos = self.read_position()
        last_ts = time.time()
        no_prog_t0 = None

        while self.jump_running and (time.time() - t0 <= timeout_s):
            pos = self.read_position()
            now = time.time()
            dt = max(1e-3, now - last_ts)
            v_est = abs(pos - last_pos) / dt  # 用位置差估计实际速度

            # 到位判据：靠近到 0.01 mm 或 实际速度很小
            if abs(pos - target_pos_mm) <= 0.01 or v_est < 0.0003:
                self.stop_motion()
                return True

            # 进度看门狗：若 1.5 s 内几乎没靠近目标，就“踢一下”重新启动
            approaching = abs(pos - target_pos_mm) < abs(last_pos - target_pos_mm) - 1e-4
            if not approaching:
                no_prog_t0 = no_prog_t0 or now
                if now - no_prog_t0 > 1.5:
                    self._set_motion_params_quiet("下" if target_pos_mm > pos else "上",
                                                  float(self.jump_run_speed_var.get()),
                                                  0.2)  # 小步长踢动
                    self.start_machine_quiet()
                    no_prog_t0 = None
            else:
                no_prog_t0 = None

            last_pos, last_ts = pos, now
            time.sleep(0.05)

        self.stop_motion()
        self.log(f"到位超时：目标 {target_pos_mm:.3f} mm，当前 {pos:.3f} mm")
        return False

    def jump_mode_loop(self, interval, selected_jumps, mode):
        if not selected_jumps:
            self.log("没有选中的跳变位置，退出跳变模式")
            self.jump_running = False
            return

        if mode == "顺序循环":
            while self.jump_running:
                for jump in selected_jumps:
                    if not self.jump_running:
                        break
                    jump_val = self.jump_values[jump]
                    if jump_val is not None:
                        jump_val = int(jump_val * 1000)
                        resp2 = self.modbus2.write_registers(174, [jump_val, 0], delay=0.005)
                        self.log(f"运动控制器设置跳变位置 {jump}: {resp2.hex()}")
                        resp2 = self.modbus2.write_registers(178, [1, 0], delay=0.005)
                        self.log(f"运动控制器启动跳变 {jump}: {resp2.hex()}")
                        time.sleep(interval)
        elif mode == "顺序加倒序循环":
            while self.jump_running:
                for jump in selected_jumps:
                    if not self.jump_running:
                        break
                    jump_val = self.jump_values[jump]
                    if jump_val is not None:
                        jump_val = int(jump_val * 1000)
                        resp2 = self.modbus2.write_registers(174, [jump_val, 0], delay=0.005)
                        self.log(f"运动控制器设置跳变位置 {jump}: {resp2.hex()}")
                        resp2 = self.modbus2.write_registers(178, [1, 0], delay=0.005)
                        self.log(f"运动控制器启动跳变 {jump}: {resp2.hex()}")
                        time.sleep(interval)
                for jump in reversed(selected_jumps):
                    if not self.jump_running:
                        break
                    jump_val = self.jump_values[jump]
                    if jump_val is not None:
                        jump_val = int(jump_val * 1000)
                        resp2 = self.modbus2.write_registers(174, [jump_val, 0], delay=0.005)
                        self.log(f"运动控制器设置跳变位置 {jump}: {resp2.hex()}")
                        resp2 = self.modbus2.write_registers(178, [1, 0], delay=0.005)
                        self.log(f"运动控制器启动跳变 {jump}: {resp2.hex()}")
                        time.sleep(interval)
        elif mode == "倒序循环":
            while self.jump_running:
                for jump in reversed(selected_jumps):
                    if not self.jump_running:
                        break
                    jump_val = self.jump_values[jump]
                    if jump_val is not None:
                        jump_val = int(jump_val * 1000)
                        resp2 = self.modbus2.write_registers(174, [jump_val, 0], delay=0.005)
                        self.log(f"运动控制器设置跳变位置 {jump}: {resp2.hex()}")
                        resp2 = self.modbus2.write_registers(178, [1, 0], delay=0.005)
                        self.log(f"运动控制器启动跳变 {jump}: {resp2.hex()}")
                        time.sleep(interval)

    def pressure_based_jump_loop(self, interval, enabled_indices):
        """
        按压力阈值触发跳变。
        正向=上升沿（低->高跨过阈值）
        反向=下降沿（高->低跨过阈值）
        双向=任一沿
        """
        last_pressure = self.read_pressure()
        sent_once = {i: False for i in enabled_indices}  # 每点节流标记

        while self.jump_running:
            cur = self.read_pressure()
            for i in list(enabled_indices):
                # 若过程中被手动禁用，则跳过并复位节流标记
                if not self.send_jump_enabled[i]:
                    sent_once[i] = False
                    continue

                try:
                    threshold = float(self.jump_vars[i].get())
                except Exception:
                    continue

                mode = self.jump_directions[i].get()  # "正向" / "反向" / "双向"
                rise = (last_pressure < threshold) and (cur >= threshold)
                fall = (last_pressure > threshold) and (cur <= threshold)
                should_fire = (
                        (mode == "正向" and rise) or
                        (mode == "反向" and fall) or
                        (mode == "双向" and (rise or fall))
                )

                if should_fire and not sent_once[i]:
                    self.send_jump_signal(i)
                    if self.jump_auto_close_vars[i].get():
                        self.stop_motion()
                    sent_once[i] = True  # 本次越阈只触发一次
                    time.sleep(interval)  # 简单节流/反抖

                # 远离阈值后，解除节流以便下次跨越再触发
                if (cur < threshold and last_pressure < threshold) or (cur > threshold and last_pressure > threshold):
                    sent_once[i] = False

            last_pressure = cur
            time.sleep(0.01)

        self.log("跳变已停止")

    def check_pressure_threshold(self, current_pressure, threshold, jump_direction, trend):
        if jump_direction == "正向":
            return current_pressure >= threshold and trend == "正向"
        elif jump_direction == "反向":
            return current_pressure <= threshold and trend == "反向"
        elif jump_direction == "双向":
            return (current_pressure >= threshold and trend == "正向") or \
                   (current_pressure <= threshold and trend == "反向")
        return False

    def send_jump_signal(self, jump_number):
        if not self.modbus2:
            return

        def _worker():
            try:
                self.log(f"发送跳变{jump_number}信号")
                resp2 = self.modbus2.write_registers(182, [jump_number, 0], delay=0.005)
                self.log(f"运动控制器发送跳变{jump_number}信号: {resp2.hex() if resp2 else '无响应'}")
            except Exception as e:
                self.log(f"发送跳变{jump_number}信号失败: {e}")

        threading.Thread(target=_worker, daemon=True).start()

    def toggle_send_jump_signal(self, jump_number):
        """每行“开启/关闭”按钮：真正控制该点是否参与触发"""
        self.send_jump_enabled[jump_number] = not self.send_jump_enabled[jump_number]
        on = self.send_jump_enabled[jump_number]
        self.jump_buttons[jump_number].config(
            text=("开启" if on else "关闭"),
            bootstyle=("success" if on else "outline-danger")
        )
        self.log(f"跳变{jump_number}已{'使能' if on else '禁用'}")

    def enable_all_jump_signals(self):
        for i in range(1, 9):
            self.send_jump_enabled[i] = True
            self.jump_buttons[i].config(text="开启", bootstyle="success")
        self.log("已一键使能所有跳变")

    def apply_jump_interval(self):
        try:
            base_jump = self.base_jump_var.get()
            jump_interval = self.jump_interval_var.get()
            for i in range(1, 9):
                self.jump_vars[i].set(base_jump + (i - 1) * jump_interval)
                self.jump_values[i] = base_jump + (i - 1) * jump_interval
            self.log(f"已应用基础阈值 {base_jump}g 和间隔 {jump_interval}g")
        except Exception as e:
            self.log(f"应用跳变间隔失败: {e}")

    def toggle_all_jump_direction(self):
        try:
            all_direction = self.all_jump_direction_var.get()
            for i in range(1, 9):
                self.jump_directions[i].set(all_direction)
            self.log(f"已设置所有跳变方向为: {all_direction}")
        except Exception as e:
            self.log(f"设置所有跳变方向失败: {e}")

    def update_plot(self):
        if not self.time_data:
            return
        try:
            n = min(len(self.time_data), len(self.pos_data), len(self.pressure_data))
            t = self.time_data[:n]
            y1 = self.pos_data[:n]
            y2 = self.pressure_data[:n]

            self.line_pos.set_data(t, y1)
            self.line_press.set_data(t, y2)

            self.ax.relim(); self.ax.autoscale_view()
            self.ax2.relim(); self.ax2.autoscale_view()
            self.ax2.set_ylabel(f"Pressure ({self.current_unit})", color='#e74c3c')
            self.canvas.draw_idle()
        except Exception as e:
            self.log(f"Chart update error: {e}")

    def save_session_data(self):
        if not self.time_data or not self.pos_data or not self.pressure_data:
            return

        file_name = f"pressure_data_{self.session_start_time.replace(':', '-')}.csv"
        file_path = os.path.join(os.getcwd(), file_name)
        try:
            with open(file_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Time (s)", "Position (mm)", "Pressure (g)"])
                for i in range(len(self.time_data)):
                    writer.writerow([self.time_data[i], self.pos_data[i], self.pressure_data[i]])
            self.log(f"会话数据已保存到 {file_path}")
        except Exception as e:
            self.log(f"保存会话数据失败: {e}")

    def export_chart(self):
        if not self.time_data or not self.pos_data or not self.pressure_data:
            messagebox.showerror("Error", "No data available for export.")
            return
        try:
            chart_path = "temp_chart.png"
            self.fig.savefig(chart_path, dpi=150)
            workbook = openpyxl.Workbook()
            sheet = workbook.active
            img = XLImage(chart_path)
            sheet.add_image(img, 'A1')
            file_path = filedialog.asksaveasfilename(
                defaultextension=".xlsx",
                filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")],
                title="Save Chart to Excel"
            )
            if file_path:
                workbook.save(file_path)
                self.log(f"Chart exported to {file_path}")
            if os.path.exists(chart_path):
                os.remove(chart_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export chart: {e}")

    def reset_chart(self):
        self.time_data.clear()
        self.pos_data.clear()
        self.pressure_data.clear()
        self.line_pos.set_data([], [])
        self.line_press.set_data([], [])
        self.ax.relim(); self.ax.autoscale_view()
        self.ax2.relim(); self.ax2.autoscale_view()
        self.canvas.draw_idle()
        self.log("Chart cleared")

    def clear_chart(self):
        if messagebox.askyesno("Confirm", "Clear all chart data?"):
            self.reset_chart()

    def tare_pressure(self):
        if not self.modbus1:
            return
        try:
            offset = self.read_pressure()  # 这是“旧去皮后”的读数
            self.tare_value += offset  # 叠加到总去皮值
            self.log(f"软件去皮：本次偏移 {offset:.1f}g，累计去皮 = {self.tare_value:.1f}g")
        except Exception as e:
            self.log(f"设置压力去皮值失败: {e}")

    def hardware_tare(self):
        if not self.modbus1:
            messagebox.showerror("错误", "压力传感器未连接")
            return
        try:
            self.modbus1.toggle_write_protection(False)
            self.log("写保护已关闭")
            resp = self.modbus1.hardware_tare()
            if resp:
                self.log("硬件去皮命令已发送")
                self.tare_value = 0
            else:
                self.log("硬件去皮命令失败")
            self.modbus1.toggle_write_protection(True)
            self.log("写保护已启用")
        except Exception as e:
            self.log(f"硬件去皮失败: {e}")

    def zero_calibration(self):
        if not self.modbus1:
            messagebox.showerror("错误", "压力传感器未连接")
            return
        try:
            self.modbus1.toggle_write_protection(False)
            self.log("写保护已关闭")
            resp = self.modbus1.zero_calibration()
            if resp:
                self.log("零点校准命令已发送")
            else:
                self.log("零点校准命令失败")
            self.modbus1.toggle_write_protection(True)
            self.log("写保护已启用")
        except Exception as e:
            self.log(f"零点校准失败: {e}")

    def weight_calibration(self, weight_value=None):
        if not self.modbus1:
            messagebox.showerror("错误", "压力传感器未连接")
            return
        if weight_value is None:
            try:
                weight_value = int(self.target_pressure_var.get())
            except:
                messagebox.showerror("错误", "请输入有效的砝码值")
                return
        try:
            self.modbus1.toggle_write_protection(False)
            self.log("写保护已关闭")
            resp = self.modbus1.weight_calibration(weight_value)
            if resp:
                self.log(f"砝码校准命令已发送，砝码值: {weight_value}g")
            else:
                self.log("砝码校准命令失败")
            self.modbus1.toggle_write_protection(True)
            self.log("写保护已启用")
        except Exception as e:
            self.log(f"砝码校准失败: {e}")

    def toggle_write_protection(self):
        if not self.modbus1:
            messagebox.showerror("错误", "压力传感器未连接")
            return
        try:
            self.write_protection_enabled = not self.write_protection_enabled
            resp = self.modbus1.toggle_write_protection(self.write_protection_enabled)
            if resp:
                status = "启用" if self.write_protection_enabled else "禁用"
                self.log(f"写保护已{status}")
            else:
                self.log("写保护切换失败")
        except Exception as e:
            self.log(f"写保护切换失败: {e}")

    def load_config(self):
        self.loaded_config = {}
        if os.path.exists(self.CONFIG_PATH):
            try:
                with open(self.CONFIG_PATH, 'r') as f:
                    config = json.load(f)
                    if isinstance(config, dict):
                        self.loaded_config = config

                    # 串口设置
                    if hasattr(self, 'port_cb1'):
                        self.port_cb1.set(config.get("serial_port1", "") or "")
                    if hasattr(self, 'baud_entry1'):
                        self.baud_entry1.delete(0, tk.END)
                        self.baud_entry1.insert(0, config.get("baud_rate1", "9600"))
                    if hasattr(self, 'port_cb2'):
                        self.port_cb2.set(config.get("serial_port2", "") or "")
                    if hasattr(self, 'baud_entry2'):
                        self.baud_entry2.delete(0, tk.END)
                        self.baud_entry2.insert(0, config.get("baud_rate2", "9600"))

                    # 其他设置
                    if hasattr(self, 'pos_interval_entry'):
                        self.pos_interval_entry.delete(0, tk.END)
                        self.pos_interval_entry.insert(0, config.get("pos_interval", "0.01"))
                    if hasattr(self, 'press_interval_entry'):
                        self.press_interval_entry.delete(0, tk.END)
                        self.press_interval_entry.insert(0, config.get("press_interval", "0.01"))

                    self.base_jump_var.set(config.get("base_jump", 1000.0))
                    self.jump_interval_var.set(config.get("jump_interval", 1000.0))
                    self.all_jump_direction_var.set(config.get("all_jump_direction", "正向"))

                    for i in range(1, 9):
                        jump_settings = config.get("jump_settings", {}).get(str(i), {})
                        self.jump_vars[i].set(jump_settings.get("threshold", 1000.0 + (i - 1) * 1000.0))
                        self.jump_directions[i].set(jump_settings.get("direction", "正向"))
                        self.jump_auto_close_vars[i].set(jump_settings.get("auto_close", False))

                    # 压力控制
                    hs = float(config.get("high_speed", 5.0))
                    ls = float(config.get("low_speed", 0.001))
                    tol = float(config.get("pressure_tolerance", 10.0))
                    self.high_speed_var.set(hs)
                    self.low_speed_var.set(ls)
                    self.tolerance_var.set(tol)
                    self.high_speed = hs
                    self.low_speed = ls
                    self.pressure_tolerance = tol

                    # 多压力测试
                    self.stable_time_var.set(float(config.get("stable_time", 10.0)))
                    self.pixel_sensitivity_var.set(float(config.get("pixel_sensitivity", 5.0)))
                    self.pixel_timeout_var.set(float(config.get("pixel_timeout", 300.0)))
                    self.pressure_points_var.set(config.get("pressure_points", "1000,2000,3000"))
                    self.loop_mode_var.set(config.get("loop_mode", "顺序"))

                    # 时间参数
                    self.detection_interval_var.set(int(config.get("detection_interval", 1000)))  # ms
                    self.pressure_step_interval_var.set(float(config.get("pressure_step_interval", 60.0)))  # s
                    self.timeout_var.set(int(config.get("timeout", 8000)))  # ms

                    ui_refresh = self._normalize_interval_value(
                        config.get("ui_refresh_interval_ms", self._ui_refresh_default_ms),
                        self._ui_refresh_default_ms,
                    )
                    plot_refresh = self._normalize_interval_value(
                        config.get("plot_refresh_interval_ms", self._plot_refresh_default_ms),
                        self._plot_refresh_default_ms,
                    )
                    self.ui_refresh_interval_ms_var.set(self._format_interval(ui_refresh))
                    self.plot_refresh_interval_ms_var.set(self._format_interval(plot_refresh))

                    # —— 新增：跳变设置 —— #
                    if hasattr(self, 'jump_loop_count_var'):
                        self.jump_loop_count_var.set(int(config.get("jump_loop_count", 0)))
                    if hasattr(self, 'jump_acq_speed_var'):
                        self.jump_acq_speed_var.set(float(config.get("jump_acq_speed", 0.001)))
                    if hasattr(self, 'jump_run_speed_var'):
                        self.jump_run_speed_var.set(float(config.get("jump_run_speed", 0.1)))
                    self.jump_mode_var.set(config.get("jump_mode", "按实际位置"))
                    self.jump_bias_gate_var.set(float(config.get("jump_bias_gate", 100.0)))



            except Exception as e:
                self.log(f"加载配置文件出错: {e}")
        else:
            self.log("未找到配置文件，使用默认参数")
            if hasattr(self, 'port_cb1'):
                self.port_cb1.set('')
            if hasattr(self, 'port_cb2'):
                self.port_cb2.set('')

        self.refresh_ports()

    def save_config(self):
        config = {
            "pos_interval": self.pos_interval_entry.get() if hasattr(self, 'pos_interval_entry') else "0.01",
            "press_interval": self.press_interval_entry.get() if hasattr(self, 'press_interval_entry') else "0.01",

            "base_jump": float(self.base_jump_var.get()),
            "jump_interval": float(self.jump_interval_var.get()),
            "all_jump_direction": self.all_jump_direction_var.get(),
            "jump_settings": {
                str(i): {
                    "threshold": float(self.jump_vars[i].get()),
                    "direction": self.jump_directions[i].get(),
                    "auto_close": bool(self.jump_auto_close_vars[i].get())
                } for i in range(1, 9)
            },

            "serial_port1": self.port_cb1.get() if hasattr(self, 'port_cb1') else "",
            "baud_rate1": self.baud_entry1.get() if hasattr(self, 'baud_entry1') else "9600",
            "serial_port2": self.port_cb2.get() if hasattr(self, 'port_cb2') else "",
            "baud_rate2": self.baud_entry2.get() if hasattr(self, 'baud_entry2') else "9600",

            # 压力控制区
            "high_speed": float(self.high_speed_var.get()),
            "low_speed": float(self.low_speed_var.get()),
            "pressure_tolerance": float(self.tolerance_var.get()),

            # 多压力测试
            "stable_time": float(self.stable_time_var.get()),
            "pixel_sensitivity": float(self.pixel_sensitivity_var.get()),
            "pixel_timeout": float(self.pixel_timeout_var.get()),
            "pressure_points": self.pressure_points_var.get(),
            "loop_mode": self.loop_mode_var.get(),

            # 新增时间参数（单位保持一致：ms / s）
            "detection_interval": int(self.detection_interval_var.get()),  # ms
            "pressure_step_interval": float(self.pressure_step_interval_var.get()),  # s
            "timeout": int(self.timeout_var.get()),  # ms
            "ui_refresh_interval_ms": self._get_interval_ms(
                self.ui_refresh_interval_ms_var, self._ui_refresh_default_ms
            ),
            "plot_refresh_interval_ms": self._get_interval_ms(
                self.plot_refresh_interval_ms_var, self._plot_refresh_default_ms
            ),

            # —— 新增：跳变设置持久化 —— #
            "jump_loop_count": int(self.jump_loop_count_var.get()) if hasattr(self, 'jump_loop_count_var') else 0,
            "jump_acq_speed": float(self.jump_acq_speed_var.get()) if hasattr(self, 'jump_acq_speed_var') else 0.001,
            "jump_run_speed": float(self.jump_run_speed_var.get()) if hasattr(self, 'jump_run_speed_var') else 0.1,
            "jump_mode": self.jump_mode_var.get(),
            "jump_bias_gate": float(self.jump_bias_gate_var.get()),

        }

        self.loaded_config = dict(config)

        try:
            if not os.path.exists(self.CONFIG_DIR):
                os.makedirs(self.CONFIG_DIR)
            with open(self.CONFIG_PATH, 'w') as f:
                json.dump(config, f, indent=4)
            self.log("配置文件保存成功")
        except Exception as e:
            self.log(f"保存配置文件出错: {e}")
            messagebox.showerror("错误", f"保存配置文件失败：{e}\n路径：{self.CONFIG_PATH}")



    def send_test_command(self):
        """同时向两个设备发一条多寄存器写入做连通性测试"""
        if not self.modbus1 or not self.modbus2:
            self.log("请先连接两台设备再测试")
            return
        try:
            test_command = [1234, 0]
            resp1 = self.modbus1.write_registers(190, test_command, delay=0.005)
            self.log(f"压力传感器发送测试命令: {resp1.hex() if resp1 else '无响应'}")
            resp2 = self.modbus2.write_registers(190, test_command, delay=0.005)
            self.log(f"运动控制器发送测试命令: {resp2.hex() if resp2 else '无响应'}")
        except Exception as e:
            self.log(f"发送测试命令失败: {e}")

    def validate_intervals(self):
        try:
            if hasattr(self, 'pos_interval_entry'):
                pos_interval = max(float(self.pos_interval_entry.get()), self.MIN_INTERVAL)
                self.pos_interval_entry.delete(0, tk.END)
                self.pos_interval_entry.insert(0, str(pos_interval))

            if hasattr(self, 'press_interval_entry'):
                press_interval = max(float(self.press_interval_entry.get()), self.MIN_INTERVAL)
                self.press_interval_entry.delete(0, tk.END)
                self.press_interval_entry.insert(0, str(press_interval))
        except ValueError:
            if hasattr(self, 'pos_interval_entry'):
                self.pos_interval_entry.delete(0, tk.END)
                self.pos_interval_entry.insert(0, str(self.MIN_INTERVAL))
            if hasattr(self, 'press_interval_entry'):
                self.press_interval_entry.delete(0, tk.END)
                self.press_interval_entry.insert(0, str(self.MIN_INTERVAL))

    def start_pressure_control(self, notify=True):
        if not (self.sensor_connected and self.controller_connected):
            if notify:
                messagebox.showerror("错误", "请先连接压力传感器和运动控制器")
            else:
                self.log("压力控制启动失败：设备未全部连接")
            return False
        try:
            self.target_pressure = float(self.target_pressure_var.get())
            self.pressure_tolerance = float(self.tolerance_var.get())
            self.high_speed = float(self.high_speed_var.get())
            self.low_speed = float(self.low_speed_var.get())
        except ValueError:
            if notify:
                messagebox.showerror("错误", "请输入有效的数值")
            return False

        if self.pressure_control_running:
            if notify:
                self.log("压力控制已在运行中")
                return False
            # 远程/静默调用视为幂等成功，避免重复启动时报错
            return True

        if self.high_speed > 0.5:
            if notify:
                if not messagebox.askyesno("高速下压警告",
                                           f"您即将以 {self.high_speed} mm/s 的速度下压，这可能导致设备损坏！\n是否继续？"):
                    return False

        self._maybe_prompt_auto_tare()
        self.pressure_control_running = True
        self.log(f"启动压力控制: 目标压力={self.target_pressure}g, 容差={self.pressure_tolerance}g")
        self.pressure_control_thread = threading.Thread(target=self.pressure_control_loop, daemon=True)
        self.pressure_control_thread.start()
        self._refresh_pressure_button_states()
        return True

    def _finalize_pressure_control(self, *, mark_stopped: bool = True, clear_thread: bool = False):
        """集中处理停止压力控制时的公共清理逻辑。"""
        if mark_stopped:
            self.pressure_control_running = False
        if clear_thread:
            self.pressure_control_thread = None
        self.stop_motion()
        self.current_speed_var.set("0.000")
        self._refresh_pressure_button_states()

    def stop_pressure_control(self):
        if not self.pressure_control_running:
            self.log("压力控制当前未运行")
            self._finalize_pressure_control(mark_stopped=False)
            return

        self.log("压力控制已停止")
        thread = getattr(self, "pressure_control_thread", None)
        thread_alive = bool(thread and thread.is_alive())
        self._finalize_pressure_control(clear_thread=not thread_alive)

    def pressure_control_loop(self):
        self.log("压力控制线程启动")
        try:
            try:
                target_pressure = float(self.target_pressure_var.get())
                high_speed = float(self.high_speed_var.get())
                low_speed = float(self.low_speed_var.get())
                progressive_zone = float(self.progressive_zone_var.get())
                safety_pressure = float(self.safety_pressure_var.get())
                mode = self.motion_mode_var.get()
            except Exception as e:
                self.log(f"参数读取错误: {e}")
                return

            jog_step = self.jog_step_mm

            highspeed_active = False
            continuous_active = False
            stable_cnt = 0
            stable_need = 5
            in_stable_state = False
            adjust_dir = None

            while self.pressure_control_running:
                try:
                    current_pressure = self.read_pressure()
                    diff = target_pressure - current_pressure
                    abs_diff = abs(diff)
                    try:
                        tolerance = float(self.tolerance_var.get())
                    except:
                        tolerance = 10.0

                    if current_pressure >= safety_pressure:
                        self.log(f"⚠️ 保护压力触发！{current_pressure:.1f} ≥ {safety_pressure:.1f} g")
                        self.stop_motion()
                        self.current_speed_var.set("0.000")
                        self.after(0, lambda: messagebox.showwarning("保护压力", f"压力超过保护值 ({safety_pressure} g)，已急停"))
                        break

                    if abs_diff > tolerance:
                        if in_stable_state:
                            self.log("压力漂移出容差，重新调节")
                            in_stable_state = False
                            stable_cnt = 0

                        if abs_diff > progressive_zone:
                            desired_dir = "下" if diff > 0 else "上"
                            if (not highspeed_active) or (self.direction.get() != desired_dir):
                                self.stop_motion()
                                self.direction.set(desired_dir)
                                self.speed_entry.delete(0, tk.END); self.speed_entry.insert(0, str(high_speed))
                                self.distance_entry.delete(0, tk.END); self.distance_entry.insert(0, "20.0")
                                self.set_parameters(); self.start_machine()
                                self.current_speed_var.set(f"{high_speed:.3f}")
                                self.log(f"区间外高速{'下压' if diff > 0 else '抬升'}开始 ΔP={abs_diff:.1f} g")
                                highspeed_active = True
                                continuous_active = False
                            time.sleep(0.05)
                            continue

                        if highspeed_active:
                            self.stop_motion()
                            highspeed_active = False

                        if mode == "点动":
                            desired_dir = "下" if diff > 0 else "上"
                            if adjust_dir != desired_dir:
                                self.direction.set(desired_dir)
                                self.speed_entry.delete(0, tk.END); self.speed_entry.insert(0, str(low_speed))
                                self.distance_entry.delete(0, tk.END); self.distance_entry.insert(0, str(jog_step))
                                self.set_parameters()
                                adjust_dir = desired_dir
                                self.log(f"渐进区 → 设置 {desired_dir} 点动参数：{low_speed} mm/s, {jog_step} mm")

                            self.start_machine()
                            self.current_speed_var.set(f"{low_speed:.3f}")
                            self.stop_motion()
                            current_pressure = self.read_pressure()
                            diff = target_pressure - current_pressure
                            abs_diff = abs(diff)
                            crossed = (diff > 0 and adjust_dir == "上") or (diff < 0 and adjust_dir == "下")
                            if crossed:
                                self.log("已越过目标压力，切换方向精调")
                                adjust_dir = None
                            continue
                        else:
                            desired_dir = "下" if diff > 0 else "上"
                            if (not continuous_active) or self.direction.get() != desired_dir or self.params_modified:
                                self.stop_motion()
                                self.direction.set(desired_dir)
                                self.speed_entry.delete(0, tk.END); self.speed_entry.insert(0, str(low_speed))
                                self.distance_entry.delete(0, tk.END); self.distance_entry.insert(0, "20.0")
                                self.set_parameters(); self.start_machine()
                                self.current_speed_var.set(f"{low_speed:.3f}")
                                self.log(f"渐进区低速持续{'下压' if diff > 0 else '抬升'}")
                                continuous_active = True
                            time.sleep(0.05)
                            continue

                    stable_cnt += 1
                    if stable_cnt >= stable_need and not in_stable_state:
                        self.stop_motion()
                        self.current_speed_var.set("0.000")
                        self.log(f"压力进入恒压保持 ±{tolerance} g")
                        in_stable_state = True
                        highspeed_active = False
                        continuous_active = False
                    time.sleep(0.05)

                except Exception as e:
                    self.log(f"压力控制异常: {e}")
                    time.sleep(0.2)
        finally:
            self._finalize_pressure_control(clear_thread=True)
            self.log("压力控制线程结束")

    def update_button_states(self):
        start_btn = getattr(self, "start_pressure_btn", None)
        stop_btn = getattr(self, "stop_pressure_btn", None)
        if not start_btn or not stop_btn:
            return

        if self.sensor_connected and self.controller_connected:
            start_btn.config(state="normal" if not self.pressure_control_running else "disabled")
            stop_btn.config(state="normal" if self.pressure_control_running else "disabled")
        else:
            start_btn.config(state="disabled")
            stop_btn.config(state="disabled")

    def _refresh_pressure_button_states(self):
        try:
            self.call_in_ui(self.update_button_states)
        except Exception as exc:  # noqa: BLE001
            self.log(f"刷新压力控制按钮状态失败: {exc}")

    def _refresh_connection_dependent_controls(self):
        controls = [
            'set_btn', 'start_btn', 'zero_btn', 'move_btn',
            'jump_btn', 'stop_btn', 'export_btn'
        ]
        enabled = self.sensor_connected and self.controller_connected
        state = "normal" if enabled else "disabled"
        for attr in controls:
            btn = getattr(self, attr, None)
            if btn is not None:
                btn.config(state=state)
        self._refresh_pressure_button_states()

    def shutdown(self, destroy_window: bool | None = None):
        """统一的关闭流程：可在嵌入/独立模式中复用。"""
        if destroy_window is None:
            destroy_window = self._owns_window

        try:
            self._cancel_periodic_jobs()
        except Exception:
            pass

        try:
            try:
                self.save_config()
            except Exception as e:
                self.log(f"关闭前保存配置失败: {e}")

            try:
                self.stop_all()
            except Exception:
                pass

            try:
                if getattr(self, "tcp_server", None):
                    self.tcp_server.stop()
            except Exception:
                pass

            self._cleanup_ports()
        finally:
            target = self._window if destroy_window else self
            try:
                target.destroy()
            except Exception:
                pass

    def on_closing(self):
        self.shutdown(destroy_window=True)

    def change_pressure_unit(self, unit):
        from tkinter import Toplevel, Label, Entry, Button

        old_unit = self.current_unit
        self.current_unit = unit

        if unit in ["Pa", "MPa"]:
            def confirm_area():
                try:
                    length = float(length_entry.get())
                    width = float(width_entry.get())
                    if length <= 0 or width <= 0:
                        raise ValueError("长度/宽度需为正数")
                    area = (length / 1000) * (width / 1000)
                    self.unit_area = area
                    messagebox.showinfo(
                        "面积确认",
                        f"已设置受力面：{length:.2f} mm × {width:.2f} mm\n面积 = {area:.6f} m²，单位 {unit}"
                    )
                    top.destroy()
                    self.refresh_pressure_display()
                except Exception as e:
                    messagebox.showerror("输入错误", f"输入无效：{e}")

            top = tk.Toplevel(self)
            top.title("设置受力面尺寸")
            top.geometry("400x200")
            top.transient(self)
            top.grab_set()
            Label(top, text="请输入受力面尺寸（单位:mm）", font=("微软雅黑", 11, "bold")).pack(pady=20)
            frame = tk.Frame(top)
            frame.pack(pady=5)
            Label(frame, text="长:").grid(row=0, column=0, padx=4, pady=4, sticky='e')
            length_entry = Entry(frame, width=8)
            length_entry.grid(row=0, column=1, padx=4)
            length_entry.insert(0, "100")
            Label(frame, text="宽:").grid(row=0, column=2, padx=4, pady=4, sticky='e')
            width_entry = Entry(frame, width=8)
            width_entry.grid(row=0, column=3, padx=4)
            width_entry.insert(0, "100")
            btn = Button(top, text="确定", command=confirm_area, width=8)
            btn.pack(pady=10)
            top.bind('<Return>', lambda event: confirm_area())
            top.wait_window()
            self.refresh_pressure_display()
            return
        else:
            messagebox.showinfo("单位切换", f"已切换单位为 {unit}")

    def refresh_pressure_display(self):
        p = self.current_pressure
        # 大字与小字统一格式，保留 1 位小数；负值如 -23.4 g
        try:
            if self.current_unit == "g":
                txt = f"{p:.1f} g"
            elif self.current_unit == "N":
                txt = f"{p / 101.97:.3f} N"  # 示例换算：按 1 kgf ≈ 9.80665 N，若你已有准确面积换算逻辑可替换
            elif self.current_unit == "Pa":
                txt = f"{p / self.unit_area:.1f} Pa"
            elif self.current_unit == "MPa":
                txt = f"{p / self.unit_area / 1e6:.6f} MPa"
            else:
                txt = f"{p:.1f} g"
        except Exception:
            txt = f"{p} g"

        # 大字显示
        self.big_pressure_var.set(txt)
        # 实时数据区的小字显示（保持你原有变量名）
        self.pressure_display_var.set(f"{p:.1f}")

    def test_pixel_detection(self):
        if not self.pixel_detection_region:
            messagebox.showwarning("提示", "请先设置检测区域后再测试")
            return
        changed, msg = self.check_pixel_change()
        self.log(f"像素检测测试: {msg}")
        if changed:
            self.set_indicator_color("green")
        else:
            self.indicator_canvas.itemconfig(self.indicator_circle, fill="orange")
            self.after(500, lambda: self.indicator_canvas.itemconfig(self.indicator_circle, fill="red"))

    def init_plot_style(self):
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Position (mm)", color='#268bd2')
        self.ax.tick_params(axis='y', labelcolor='#268bd2')
        self.ax2.set_ylabel(f"Pressure ({self.current_unit})", color='#e74c3c')
        self.ax2.tick_params(axis='y', labelcolor='#e74c3c')
        self.ax.grid(True, linestyle='--', alpha=0.6)

    def stop_multi_pressure_test(self):
        if self.multi_pressure_running:
            self.multi_pressure_running = False
            self.log("多压力测试已手动停止")
        self.pixel_detection_enabled.set(False)
        self.sim_click_enabled.set(False)

    def perform_sim_click(self, *, double: bool = False, delay_ms: int = 0):
        if not self.sim_click_pos:
            self.log("模拟点击点未设置，跳过点击")
            return
        x, y = self.sim_click_pos
        perform_click_async(int(x), int(y), double=double, delay_ms=delay_ms, reporter=self.log)

    def set_external_logger(self, callback):
        self._external_log = callback

    def start_machine_quiet(self):
        """直接下发启动寄存器（无弹窗），带禁启窗口。"""
        if not self.modbus2:
            return
        # —— 新增：禁止在刚停止后的窗口内重新点火 —— #
        if time.time() < getattr(self, "_no_autostart_until", 0):
            self.log("抑制自动重启（刚刚停止）")
            return
        self._maybe_prompt_auto_tare()
        try:
            resp2 = self.modbus2.write_registers(178, [1, 0], delay=0.005)
            self.log(f"运动控制器启动命令(quiet): {resp2.hex() if resp2 else ''}")
        except Exception as e:
            self.log(f"启动命令失败(quiet): {e}")

    def _set_motion_params_quiet(self, desired_dir: str, speed_mm_s: float, distance_mm: float):
        """直接写寄存器设置方向/距离/速度，不改UI输入框。"""
        if not self.modbus2:
            return
        op_mode = 1 if desired_dir == "上" else 0
        speed_val = int(round(speed_mm_s * 1000))
        dist_val = int(round(distance_mm * 1000))
        try:
            resp = self.modbus2.write_registers(170, [op_mode, 0], delay=0.005)
            self.log(f"运动控制器设置工作方式(op={op_mode}): {resp.hex() if resp else ''}")
            resp = self.modbus2.write_registers(174, [dist_val, 0], delay=0.005)
            self.log(f"运动控制器设置距离(mm={distance_mm}): {resp.hex() if resp else ''}")
            resp = self.modbus2.write_registers(176, [speed_val, 0], delay=0.005)
            self.log(f"运动控制器设置速度(mm/s={speed_mm_s}): {resp.hex() if resp else ''}")
        except Exception as e:
            self.log(f"设置运动参数失败(quiet): {e}")

    def _issue_motion_if_needed(self, desired_dir: str, speed_mm_s: float, distance_mm: float, min_interval: float = 0.30):
        """
        仅在需要时才下发：方向变了 / 速度或距离变了 / 机器疑似停住 且 距上次发命令超过 min_interval 秒。
        返回 True 表示这次确实下发了启动。
        """
        now = time.time()
        try:
            cur_spd = self.read_speed()
        except Exception:
            cur_spd = 0.0

        changed_params = (
            desired_dir != self._mc_last_dir or
            self._mc_last_speed is None or abs(speed_mm_s - self._mc_last_speed) > 1e-6 or
            self._mc_last_distance is None or abs(distance_mm - self._mc_last_distance) > 1e-6
        )
        looks_idle = abs(cur_spd) < 0.0008  # 速度几乎为0，认为停住
        enough_gap = (now - self._mc_last_cmd_ts) >= min_interval

        if changed_params:
            # 方向改变先停一下，防止反向顶撞
            if self._mc_last_dir is not None and desired_dir != self._mc_last_dir:
                self.stop_motion()
                time.sleep(0.03)
            self._set_motion_params_quiet(desired_dir, speed_mm_s, distance_mm)
            self.start_machine_quiet()
            self._mc_last_cmd_ts = time.time()
            self._mc_last_dir = desired_dir
            self._mc_last_speed = speed_mm_s
            self._mc_last_distance = distance_mm
            self.current_speed_var.set(f"{speed_mm_s:.3f}")
            return True

        # 参数没变：只有在“像是停住了”且时间间隔满足时，才重发一次启动
        if looks_idle and enough_gap:
            self.start_machine_quiet()
            self._mc_last_cmd_ts = time.time()
            self.current_speed_var.set(f"{speed_mm_s:.3f}")
            return True

        return False

    # —— 停止跳变 —— #
    def stop_jump(self, manual=False):
        """只置位并异步发停机；收尾由各工作线程 finally 自行完成。"""
        if not self.jump_running:
            return
        self.jump_running = False
        self.stop_motion()  # 异步，不阻塞 UI
        if manual:
            self.log("跳变被手动停止")

    # —— 压力到点（使用之前修复的“节流”逻辑） —— #
    def _move_to_pressure_once(self, target_g, tolerance_g=None, move_speed=0.1, timeout_s=120.0, stable_hold_s=0.2):
        """
        把压力跑到 target_g 并保持 stable_hold_s 秒后急停。
        仅在方向/速度/距离变更或疑似停住时才下发命令（节流）。
        """
        if tolerance_g is None:
            try:
                tolerance_g = float(self.tolerance_var.get())
            except Exception:
                tolerance_g = 10.0

        try:
            safety_g = float(self.safety_pressure_var.get())
        except Exception:
            safety_g = 30000.0

        # 节流记忆重置
        self._mc_last_dir = None
        self._mc_last_speed = None
        self._mc_last_distance = None
        self._mc_last_cmd_ts = 0.0

        t0 = time.time()
        stable_t0 = None

        while self.jump_running and (time.time() - t0 <= timeout_s):
            cur = self.read_pressure()
            diff = target_g - cur

            # 保护
            if cur >= safety_g:
                self.log(f"⚠️ 保护压力触发（{cur:.1f} ≥ {safety_g:.1f} g），已急停")
                self.stop_motion()
                return False

            # 容差判断
            if abs(diff) <= tolerance_g:
                if stable_t0 is None:
                    stable_t0 = time.time()
                if time.time() - stable_t0 >= stable_hold_s:
                    self.stop_motion()
                    self.current_speed_var.set("0.000")
                    return True
            else:
                stable_t0 = None

            desired_dir = "下" if diff > 0 else "上"
            # 仅必要时下发
            self._issue_motion_if_needed(desired_dir, move_speed, 20.0, min_interval=0.30)

            time.sleep(0.05)

        self.stop_motion()
        self.current_speed_var.set("0.000")
        return False

    # —— 位置到点 —— #
    def _move_to_position_once(self, pos_mm, speed_mm_s, timeout_s=90.0):
        if not self.modbus2:
            return False
        try:
            # 设置“转到”模式
            self.modbus2.write_registers(170, [2, 0], delay=0.005)
            # 速度
            speed_val = int(round(speed_mm_s * 1000))
            self.modbus2.write_registers(176, [speed_val, 0], delay=0.005)
            # 目标位置
            target_val = int(round(pos_mm * 1000))
            self.modbus2.write_registers(174, [target_val, 0], delay=0.005)
            # 启动
            self.start_machine_quiet()

            t0 = time.time()
            while self.jump_running and (time.time() - t0 <= timeout_s):
                cur_pos = self.read_position()
                cur_spd = self.read_speed()
                # 到位判据：位置误差很小或速度≈0
                if abs(cur_pos - pos_mm) <= 0.01 or abs(cur_spd) < 0.0008:
                    self.stop_motion()
                    return True
                time.sleep(0.05)
        except Exception as e:
            self.log(f"位置到点失败: {e}")

        self.stop_motion()
        return False

    # —— 校准：按各“压力点”采集一次实际位置（用于“按实际位置”跳变） —— #
    def _calibrate_jump_positions(self, seq_pairs, acq_speed):
        """
        采集阶段：按压力逐点逼近并记录位置。
        使用渐进区逻辑：
          - 区间外：高速连续靠近
          - 区间内：低速点动/持续（由“点动/持续”单选决定）
        速度取值：
          - 高速 = “采集速度(mm/s)”（jump_acq_speed_var）
          - 低速 = “压力控制”区的低速(low_speed_var)
        """
        pos_map = {}
        try:
            tol = float(self.tolerance_var.get())
        except Exception:
            tol = 10.0
        try:
            prog = float(self.progressive_zone_var.get())
        except Exception:
            prog = 100.0

        # 采集用的高/低速与点动步长、模式
        try:
            high_speed = float(self.jump_acq_speed_var.get())  # 采集速度 = 高速
        except Exception:
            high_speed = max(0.003, float(acq_speed))  # 兜底
        try:
            low_speed = float(self.low_speed_var.get())  # 来自“压力控制”区
        except Exception:
            low_speed = 0.001
        jog_step = self.jog_step_mm
        mode = self.motion_mode_var.get() if hasattr(self, "motion_mode_var") else "点动"

        self.log("开始按压力点采集位置（带渐进区逻辑）…")
        for i, target_g in seq_pairs:
            if not self.jump_running:
                break

            self.log(f"采集{i}: 目标压力 {target_g} g（高速={high_speed} mm/s，低速={low_speed} mm/s，渐进区={prog} g）")

            ok = self._approach_pressure_like_control(
                target_g,
                tolerance_g=tol,
                progressive_zone_g=prog,
                high_speed=high_speed,
                low_speed=low_speed,
                jog_step_mm=jog_step,
                mode=mode,
                timeout_s=None,  # 采集阶段通常不超时
            )
            if not (self.jump_running and ok):
                self.log(f"采集{i}: 未能到达 {target_g} g，跳过该点")
                continue

            pos = self.read_position()
            pos_map[i] = pos
            if hasattr(self, "jump_pos_vars"):
                self.jump_pos_vars[i].set(f"{pos:.3f} mm")
            self.log(f"采集{i}: 压力 {target_g:.1f} g → 位置 {pos:.3f} mm")

            # 小停顿，防机械回弹
            import time
            time.sleep(0.15)

        if not pos_map:
            self.log("未采集到任何位置")
        else:
            self.log(f"采集完成，共 {len(pos_map)} 个点（已使用渐进区逻辑）")
        return pos_map

    # —— 顺序：按实际压力跳变 —— #
    def _run_jump_sequence_pressure(self, seq_pairs, run_speed, loop_cnt, interval_s):
        """
        seq_pairs: [(i, pressure_g), ...] 已经按方向排好序
        """
        loops_done = 0
        while self.jump_running and (loop_cnt == 0 or loops_done < loop_cnt):
            for i, target_g in seq_pairs:
                if not self.jump_running:
                    break
                ok = self._move_to_pressure_once(target_g, move_speed=run_speed, timeout_s=180.0, stable_hold_s=0.25)
                if ok:
                    self.log(f"✓ 跳变{i}: 到达 {target_g} g")
                else:
                    self.log(f"✗ 跳变{i}: 未到达 {target_g} g")
                # 间隔
                t_end = time.time() + max(0.0, interval_s)
                while self.jump_running and time.time() < t_end:
                    time.sleep(0.05)
            loops_done += 1

        self.stop_jump(manual=False)
        self.log("按实际压力跳变结束")

    # —— 顺序：按实际位置跳变（先采集位置，再按位置跑） —— #
    # --- App 内：精准到“间隔最后一刻”取样的按实际位置跳变 ---
    def _run_jump_sequence_position(self, seq_pairs, acq_speed, run_speed, loop_cnt, interval_s):
        """
        闭环版“按实际位置”跳变（末尾单点取样）：
          1) 先按各压力点采集对应位置（渐进区逻辑）
          2) 跳变执行阶段：每到一个记录位置 -> 立刻触发跳变信号
             -> 进入点间“间隔” -> 在间隔的“最后一刻”精确读取一次压力
             -> 若 |误差| > 偏置门槛(bias_gate) 则记录 ±0.001 mm 偏置（下次生效）
        """
        import time

        # 1) 采集阶段：得到每个跳变点的“基准位置”
        pos_map = self._calibrate_jump_positions(seq_pairs, acq_speed)
        if not pos_map:
            self.log("未采集到任何位置，按实际位置跳变终止")
            self.stop_jump(manual=False)
            return

        # 偏置表（跨循环累积）
        if not hasattr(self, "_jump_pos_offsets"):
            self._jump_pos_offsets = {}  # {idx: offset_mm}

        # 参数读取
        try:
            tol_g = float(self.tolerance_var.get())
        except Exception:
            tol_g = 10.0
        try:
            safety_g = float(self.safety_pressure_var.get())
        except Exception:
            safety_g = 30000.0
        try:
            bias_gate_g = float(self.jump_bias_gate_var.get())  # >该门槛才±0.001mm
        except Exception:
            bias_gate_g = 100.0

        OFFSET_STEP_MM = 0.001  # 单步偏置
        MAX_OFFSET_ABS = 0.50  # 偏置限幅
        LAST_GUARD_S = 0.02  # “最后一刻”保护时间

        # 2) 执行阶段：转到模式 + 速度 —— 仅一次；若速度未变则跳过
        # 缓存键：_turn_mode_armed (bool), _turn_mode_speed (float)
        need_prepare = (not getattr(self, "_turn_mode_armed", False)) \
                       or (float(getattr(self, "_turn_mode_speed", float("nan"))) != float(run_speed))
        if need_prepare:
            try:
                # 确保当前停止一次，以免切模式时在动
                self.stop_motion()
            except Exception:
                pass
            try:
                if self.modbus2:
                    # 2 = 转到（绝对位置）模式
                    self.modbus2.write_registers(170, [2, 0], delay=0.005)
                    # 设置速度（单位：0.001 mm/s）
                    self.modbus2.write_registers(176, [int(round(float(run_speed) * 1000)), 0], delay=0.005)
                self._turn_mode_armed = True
                self._turn_mode_speed = float(run_speed)
                self.log(f"执行阶段：转到模式已配置一次，速度 {run_speed} mm/s")
            except Exception as e:
                self.log(f"设置转到模式/速度失败: {e}")

        loops_done = 0
        prev_target = None

        while self.jump_running and (loop_cnt == 0 or loops_done < loop_cnt):
            for idx, target_g in seq_pairs:
                if not self.jump_running:
                    break
                if idx not in pos_map:
                    continue

                base_pos = float(pos_map[idx])
                offset = float(self._jump_pos_offsets.get(idx, 0.0))
                target_pos = base_pos + offset

                # 到位（不再在这里重复配置模式/速度）
                ok = self._move_to_position_once(target_pos, speed_mm_s=run_speed, timeout_s=180.0)
                if not ok:
                    self.log(
                        f"跳变{idx}: 到位等待超时，目标 {target_pos:.3f} mm（基准 {base_pos:.3f} + 偏置 {offset:+.3f}）")
                    # 进入点间间隔但不做偏置评估
                    t_end = time.time() + max(0.0, float(interval_s))
                    self._sleep_until(t_end)
                    continue

                # 刚到位就触发跳变信号（如果配置允许）
                try:
                    self._fire_jump_if_configured(idx, prev_target, target_g)
                except Exception:
                    pass
                prev_target = target_g

                # 点间间隔：监控 + “最后一刻”取样
                t_end = time.time() + max(0.0, float(interval_s))
                while self.jump_running and time.time() < t_end - LAST_GUARD_S:
                    p_now = float(self.read_pressure())
                    if p_now >= safety_g:
                        self.log(f"⚠️ 跳变{idx}: 触发保护压力 {p_now:.1f}g ≥ {safety_g:.1f}g，急停")
                        self.stop_motion()
                        self.jump_running = False
                        return
                    time.sleep(0.05)

                self._sleep_until(t_end)
                p_last = float(self.read_pressure())  # ★ 末刻单点
                if p_last >= safety_g:
                    self.log(f"⚠️ 跳变{idx}: 触发保护压力 {p_last:.1f}g ≥ {safety_g:.1f}g，急停")
                    self.stop_motion()
                    self.jump_running = False
                    return

                # 偏置评估
                err = p_last - float(target_g)
                aerr = abs(err)

                if aerr <= tol_g:
                    self.log(
                        f"跳变{idx}: 末刻压力 {p_last:.1f} g 在容差 ±{tol_g:.0f} g 内（目标 {target_g:.1f} g），偏置保持 {offset:+.3f} mm")
                elif aerr > bias_gate_g:
                    if err < 0:
                        new_offset = offset + OFFSET_STEP_MM  # 压力偏小 → 再多压一点
                        action = f"+{OFFSET_STEP_MM:.3f} mm"
                    else:
                        new_offset = offset - OFFSET_STEP_MM  # 压力偏大 → 少压一点
                        action = f"-{OFFSET_STEP_MM:.3f} mm"
                    new_offset = max(-MAX_OFFSET_ABS, min(MAX_OFFSET_ABS, new_offset))
                    self._jump_pos_offsets[idx] = new_offset
                    self.log(
                        f"跳变{idx}: 末刻压力 {p_last:.1f} g 与目标 {target_g:.1f} g 偏差 {err:+.1f} g > ±{bias_gate_g:.0f} g → "
                        f"记录偏置 {action}（由 {offset:+.3f} → {new_offset:+.3f} mm，下次生效）"
                    )
                else:
                    self.log(
                        f"跳变{idx}: 末刻压力 {p_last:.1f} g 与目标 {target_g:.1f} g 偏差 {err:+.1f} g，"
                        f"在(±{tol_g:.0f} g, ±{bias_gate_g:.0f} g] 区间 → 不调整偏置（保持 {offset:+.3f} mm）"
                    )

            if loop_cnt != 0:
                loops_done += 1

        self.stop_jump(manual=False)
        self.log("按实际位置跳变结束（末刻取样 & 闭环偏置）")

    def _sleep_until(self, t_end: float):
        """
        将当前线程休眠到 t_end（基于 time.time()）。
        使用两阶段睡眠：先粗睡，再短自旋，尽量贴近目标时刻。
        """
        import time
        while True:
            now = time.time()
            dt = t_end - now
            if dt <= 0:
                break
            if dt > 0.01:
                time.sleep(dt - 0.008)  # 粗睡，留少许余量
            else:
                # 最后 10ms 内短自旋，尽量贴近“最后一刻”
                time.sleep(0.0005)

    def start_auto_tare(self):
        """
        自动去皮：
        - 以 0.001 mm/s 向上慢速运动；
        - 监测最近 5 s 的压力波动（max-min），若 ≤ 1 g 则认为稳定；
        - 用该窗口的平均值做软件去皮（置 tare_value），然后急停。
        """
        if not self.modbus1 or not self.sensor_connected:
            messagebox.showerror("错误", "压力传感器未连接")
            return
        if not self.modbus2 or not self.controller_connected:
            messagebox.showerror("错误", "运动控制器未连接")
            return

        # 避免重复启动
        if not hasattr(self, "auto_tare_running"):
            self.auto_tare_running = False
        if self.auto_tare_running:
            self.log("自动去皮已在进行中")
            return

        # 若当前有恒压控制等，先停掉
        if getattr(self, "pressure_control_running", False):
            self.stop_pressure_control()

        # 设置向上 0.001 mm/s 慢速，给出较大的行程
        try:
            self.direction.set("上")
            self.speed_entry.delete(0, tk.END)
            self.speed_entry.insert(0, "0.001")
            self.distance_entry.delete(0, tk.END)
            self.distance_entry.insert(0, "20.0")  # 足够的位移
            self.set_parameters()
            self.start_machine()
            self.current_speed_var.set(f"{0.001:.3f}")
        except Exception as e:
            self.log(f"自动去皮启动运动失败: {e}")
            return

        self.auto_tare_running = True
        threading.Thread(target=self._auto_tare_loop, daemon=True).start()
        self.log("自动去皮开始：向上 0.001 mm/s，检测 5 s 波动 ≤ 1 g 即取零")

    def _auto_tare_loop(self):
        """
        简化版自动去皮：
        - 以“当前读取压力”为基准 baseline
        - 只要连续保持在 baseline ± 1 g 内满 5 s，就触发 3/2/1 倒计时并去皮
        - 一旦越界，立刻把 baseline 重置为最新值并重新计时
        """
        window_sec = 5.0  # 连续 5s
        tolerance_g = 1.0  # ±1 g
        timeout_s = 180.0  # 可选超时

        # 上行 0.001 mm/s，设置一次即可，不要在循环里反复写寄存器
        try:
            self.direction.set("上")
            self.speed_entry.delete(0, tk.END)
            self.speed_entry.insert(0, "0.001")
            self.distance_entry.delete(0, tk.END)
            self.distance_entry.insert(0, "20.0")  # 给足行程即可
            self.set_parameters()
            self.start_machine()
            self.current_speed_var.set("0.001")
            self.log("自动去皮：向上 0.001 mm/s，进入稳定检测（±1 g 连续 5 s）")
        except Exception as e:
            self.log(f"自动去皮启动失败: {e}")
            self.auto_tare_running = False
            return

        baseline = None
        stable_t0 = None
        last_countdown_logged = None
        t_begin = time.time()

        try:
            while self.auto_tare_running and (time.time() - t_begin <= timeout_s):
                p = float(self.read_pressure())
                now = time.time()

                if baseline is None:
                    baseline = p
                    stable_t0 = now
                    last_countdown_logged = None
                else:
                    # 在 ±1 g 内
                    if abs(p - baseline) <= tolerance_g:
                        elapsed = now - stable_t0

                        # 剩余 3/2/1 秒时打印倒计时（只打一遍）
                        remain = int(max(0, window_sec - elapsed) + 0.999)  # 天花板取整
                        if 1 <= remain <= 3 and remain != last_countdown_logged:
                            self.log(f"自动去皮：倒计时 {remain}")
                            last_countdown_logged = remain

                        if elapsed >= window_sec:
                            # 达标：停止并去皮（以 baseline 作为零点）
                            self.stop_motion()
                            self.current_speed_var.set("0.000")
                            self.tare_value += baseline
                            self.log(f"自动去皮成功：基准 {baseline:.1f} g，±{tolerance_g:.1f} g 持续 {window_sec:.0f}s → 已置零（累计去皮 {self.tare_value:.1f} g）")
                            self.after(0, lambda: messagebox.showinfo("自动去皮", "自动去皮完成，已取零。"))
                            return
                    else:
                        # 越界：以当前读数为新基准，重新计时
                        baseline = p
                        stable_t0 = now
                        last_countdown_logged = None

                time.sleep(0.05)

            # 结束（可能是手动停止或超时）
            self.stop_motion()
            self.current_speed_var.set("0.000")
            if self.auto_tare_running:
                self.log("自动去皮超时：一直未能连续 5 s 稳定在 ±1 g 内")
                self.after(0, lambda: messagebox.showwarning("自动去皮", "超时：未达到稳定条件。"))
        finally:
            self.auto_tare_running = False

    def _approach_pressure_like_control(
            self,
            target_g: float,
            tolerance_g: float,
            progressive_zone_g: float,
            high_speed: float,
            low_speed: float,
            jog_step_mm: float,
            mode: str = None,
            stable_need: int = 5,
            safety_pressure: float = None,
            timeout_s: float | None = None,  # None = 不超时
    ) -> bool:
        """
        逼近到 target_g 并稳定保持：
        - 区间外高速连续；渐进区低速点动/持续；
        - 进入容差累计 stable_need 次即成功；
        - timeout_s=None 则不因为超时退出（仍保留保护压力和手动停止）。
        """
        if mode is None:
            mode = self.motion_mode_var.get() if hasattr(self, "motion_mode_var") else "点动"
        if safety_pressure is None:
            try:
                safety_pressure = float(self.safety_pressure_var.get())
            except Exception:
                safety_pressure = 30000.0

        t0 = time.time()
        highspeed_active = False
        continuous_active = False
        adjust_dir = None
        stable_cnt = 0

        while self.jump_running and (timeout_s is None or time.time() - t0 <= timeout_s):
            cur_p = float(self.read_pressure())
            diff = target_g - cur_p
            adiff = abs(diff)

            # 保护
            if cur_p >= safety_pressure:
                self.log(f"⚠️ 保护压力触发：{cur_p:.1f} ≥ {safety_pressure:.1f} g，急停")
                self.stop_motion()
                return False

            # 未达容差：调整
            if adiff > tolerance_g:
                if stable_cnt:
                    stable_cnt = 0

                # 区间外：高速持续
                if adiff > progressive_zone_g:
                    desired = "下" if diff > 0 else "上"
                    if (not highspeed_active) or self.direction.get() != desired:
                        self.stop_motion()
                        self.direction.set(desired)
                        self.speed_entry.delete(0, tk.END);
                        self.speed_entry.insert(0, f"{high_speed}")
                        self.distance_entry.delete(0, tk.END);
                        self.distance_entry.insert(0, "20.0")
                        self.set_parameters();
                        self.start_machine()
                        highspeed_active = True
                        continuous_active = False
                        self.current_speed_var.set(f"{high_speed:.3f}")
                        self.log(f"高速靠近：{desired}，ΔP={adiff:.1f} g")
                    time.sleep(0.05)
                    continue

                # 渐进区：低速
                if highspeed_active:
                    self.stop_motion()
                    highspeed_active = False

                if mode == "点动":
                    desired = "下" if diff > 0 else "上"
                    if adjust_dir != desired:
                        self.direction.set(desired)
                        self.speed_entry.delete(0, tk.END);
                        self.speed_entry.insert(0, f"{low_speed}")
                        self.distance_entry.delete(0, tk.END);
                        self.distance_entry.insert(0, f"{jog_step_mm}")
                        self.set_parameters()
                        adjust_dir = desired
                        self.log(f"渐进区点动：{desired} @ {low_speed} mm/s，步长 {jog_step_mm} mm")
                    self.start_machine();
                    self.stop_motion()
                    time.sleep(0.05)
                    continue
                else:
                    desired = "下" if diff > 0 else "上"
                    if (not continuous_active) or self.direction.get() != desired:
                        self.stop_motion()
                        self.direction.set(desired)
                        self.speed_entry.delete(0, tk.END);
                        self.speed_entry.insert(0, f"{low_speed}")
                        self.distance_entry.delete(0, tk.END);
                        self.distance_entry.insert(0, "20.0")
                        self.set_parameters();
                        self.start_machine()
                        continuous_active = True
                        self.current_speed_var.set(f"{low_speed:.3f}")
                        self.log(f"渐进区持续：{desired} @ {low_speed} mm/s")
                    time.sleep(0.05)
                    continue

            # 已在容差内：累计稳定
            stable_cnt += 1
            if stable_cnt >= stable_need:
                self.stop_motion()
                self.current_speed_var.set("0.000")
                self.log(f"已在容差 ±{tolerance_g} g 内稳定，目标 {target_g:.1f} g 达成")
                return True
            time.sleep(0.05)

        # 只有在被手动停止（或有超时值且确实超时）才会走到这里
        self.stop_motion()
        return False

    def _jump_by_position_loop(self, enabled_ids):
        """读取UI参数与启用点，转交给闭环执行函数。"""
        try:
            collect_speed = float(self.jump_acq_speed_var.get())  # 采集速度(mm/s)
            run_speed = float(self.jump_run_speed_var.get())  # 跳变速度(mm/s)
            interval_s = float(self.interval_entry.get())  # 点间等待(s)
            loop_cnt = int(self.jump_loop_count_var.get())  # 0=无限
        except Exception as e:
            self.log(f"跳变参数错误: {e}")
            self.jump_running = False
            return

        # 只取有效的 (idx, target_g)
        seq_pairs = []
        for i in enabled_ids:
            try:
                target_g = float(self.jump_vars[i].get())
                seq_pairs.append((i, target_g))
            except Exception:
                self.log(f"跳变{i}: 压力阈值无效，跳过")

        if not seq_pairs:
            self.log("未找到有效的跳变点")
            self.jump_running = False
            return

        # 交由闭环序列函数执行（含采集+闭环微调+循环）
        self._run_jump_sequence_position(seq_pairs, collect_speed, run_speed, loop_cnt, interval_s)

    def _jump_by_pressure_loop(self, enabled_ids):
        """
        按“实际压力”直跑：
        - 只用一档跳变速度 run_speed；
        - 达到或越过阈值立即急停（不做低速/渐进调节）；
        - 下压使用“看门狗行程 step_mm”分段推进，防一次长行程砸坏工件；
        - 任何停机后 600ms 内禁止自动重启；
        - 支持循环次数(0=无限)与点间间隔；
        """
        # 读取参数
        try:
            run_speed = float(self.jump_run_speed_var.get())  # 跳变速度
            interval_s = float(self.interval_entry.get())  # 点间间隔
            loop_cnt = int(self.jump_loop_count_var.get())  # 0=一直循环
            try:
                safety = float(self.safety_pressure_var.get())  # 保护压力
            except Exception:
                safety = 30000.0
        except Exception as e:
            self.log(f"跳变参数错误: {e}")
            self.jump_running = False
            return

        # —— 下压看门狗行程（一次只跑这么长，没到阈值再续跑） —— #
        step_mm_down = 0.20  # 建议 0.10~0.50 之间
        step_mm_up = 2.00  # 上抬可以稍大些

        # 收集开启的点
        seq = []
        for i in enabled_ids:
            try:
                p = float(self.jump_vars[i].get())
                seq.append((i, p))
            except Exception:
                pass
        if not seq:
            self.log("无有效压力点，退出")
            self.jump_running = False
            return

        # 先按当前压力决定方向顺序（先去“近端”）
        cur_p = float(self.read_pressure())
        seq.sort(key=lambda x: x[1])
        seq = seq if cur_p < seq[0][1] else list(reversed(seq))

        loops_done = 0
        prev_target = None

        while self.jump_running and (loop_cnt == 0 or loops_done < loop_cnt):
            for idx, target in seq:
                if not self.jump_running:
                    break

                hit = False
                t0 = time.time()
                timeout_s = 180.0

                while self.jump_running and (time.time() - t0 <= timeout_s):
                    cur_p = float(self.read_pressure())
                    # 保护
                    if cur_p >= safety:
                        self.log(f"⚠️ 保护压力触发：{cur_p:.1f} ≥ {safety:.1f} g，急停")
                        self.stop_motion()
                        self.jump_running = False
                        return

                    desired_dir = "下" if target > cur_p else "上"
                    # 本次只跑一个小步长（看门狗行程）
                    step_mm = step_mm_down if desired_dir == "下" else step_mm_up

                    # 下发参数并启动（带禁启窗口）
                    self._set_motion_params_quiet(desired_dir, run_speed, step_mm)
                    self.start_machine_quiet()
                    self.current_speed_var.set(f"{run_speed:.3f}")

                    # 在这个小步长内，盯着阈值，达到/越过立即急停
                    inner_t0 = time.time()
                    while self.jump_running and time.time() - inner_t0 <= 2.0:  # 一个小步不超过2秒
                        cur_p = float(self.read_pressure())

                        if cur_p >= safety:
                            self.log(f"⚠️ 保护压力触发：{cur_p:.1f} ≥ {safety:.1f} g，急停")
                            self.stop_motion()
                            self.jump_running = False
                            return

                        if (desired_dir == "下" and cur_p >= target) or (desired_dir == "上" and cur_p <= target):
                            hit = True
                            break

                        time.sleep(0.02)

                    # 无论命中与否，小步结束都先停一下，再决定是否续跑
                    self.stop_motion()

                    if hit:
                        break
                    # 没命中且还在超时时间内 → 继续下一小步

                # 一个目标点收尾
                if hit:
                    self.log(f"✓ 跳变{idx}: {'增压' if target > cur_p else '卸压'}至 {target} g（到/越过阈值，急停）")
                    self._fire_jump_if_configured(idx, prev_target, target)
                else:
                    self.log(f"✗ 跳变{idx}: 超时未到达 {target} g")

                prev_target = target

                # 点间间隔
                t_end = time.time() + max(0.0, interval_s)
                while self.jump_running and time.time() < t_end:
                    time.sleep(0.05)

            if loop_cnt != 0:
                loops_done += 1
            # 如需往返，可在此处： seq = list(reversed(seq))

        self.stop_motion()
        self.jump_running = False
        self.log("按实际压力跳变结束")

    def _fire_jump_if_configured(self, idx: int, prev_target: float, cur_target: float):
        """根据跳变方向配置决定是否发送跳变信号"""
        if prev_target is None:
            return
        if not self.send_jump_enabled.get(idx, False):
            return

        trend = "正向" if cur_target > prev_target else ("反向" if cur_target < prev_target else "双向")
        cfg = self.jump_directions[idx].get() if hasattr(self, "jump_directions") else "双向"

        should = (cfg == "双向") or (cfg == trend)
        if not should:
            self.log(f"跳变{idx}: 方向配置为[{cfg}]，与趋势[{trend}]不符，不触发")
            return

        try:
            self.send_jump_signal(idx)
            self.log(f"跳变{idx}: 已发送（配置[{cfg}]，趋势[{trend}]）")
            if hasattr(self, "jump_auto_close_vars") and self.jump_auto_close_vars[idx].get():
                self.stop_motion()
                self.log(f"跳变{idx}: 自动关闭已启用，已停止运动")
        except Exception as e:
            self.log(f"跳变{idx}: 发送信号失败: {e}")

    def set_var_safe(self, tk_var, value):
        if threading.current_thread() is threading.main_thread():
            tk_var.set(value)
        else:
            self.after(0, lambda: tk_var.set(value))

    def _read_pressure_avg(self, duration_s=0.25):
        t0 = time.time();
        vals = []
        while time.time() - t0 < duration_s:
            vals.append(float(self.read_pressure()))
            time.sleep(0.02)
        return sum(vals) / len(vals) if vals else float(self.read_pressure())

    def _tune_position_for_pressure(self, idx: int, target_g: float, start_pos_mm: float,
                                    tol_g: float, step_mm: float = 0.001,
                                    max_steps: int = 60, settle_s: float = 0.25,
                                    timeout_each: float = 15.0):
        """
        以 start_pos_mm 为起点，按 ±0.001 mm 微调，直到 |P-P*| ≤ tol_g
        返回 (final_pos_mm, success)
        """
        try:
            safety_g = float(self.safety_pressure_var.get())
        except Exception:
            safety_g = 30000.0

        pos = float(start_pos_mm)
        for k in range(max_steps):
            ok = self._move_to_position_once(pos, speed_mm_s=float(self.jump_run_speed_var.get()),
                                             timeout_s=timeout_each)
            if not ok:
                self.log(f"跳变{idx}: 到位等待超时，已停止在 {pos:.3f} mm")
                return pos, False

            time.sleep(settle_s)
            p = self._read_pressure_avg(0.25)
            if p >= safety_g:
                self.log(f"⚠️ 跳变{idx}: 触发保护压力 {p:.1f}g ≥ {safety_g:.1f}g，急停")
                self.stop_motion()
                return pos, False

            err = p - target_g
            if abs(err) <= tol_g:
                self.log(f"跳变{idx}: 压力收敛 {p:.1f}g (目标 {target_g:.1f}g, |误差|≤{tol_g}g)")
                return pos, True

            # 你要求：“误差>100g 时按 ±0.001mm 调”，这里兼容小于100g仍未进容差的情况——仍用同样步长收敛
            if err < 0:  # 压力偏小 → 往下（增大位置）
                pos += step_mm
            else:  # 压力偏大 → 往上（减小位置）
                pos -= step_mm

        self.log(f"跳变{idx}: 微调步数用尽仍未进容差（最后压力 {p:.1f}g）")
        return pos, False

    # --- 放在 class App 内部 ---
    def _cleanup_ports(self):
        """兜底释放：先停循环与线程，再让驱动释放串口句柄。可安全重复调用。"""
        try:
            self._cancel_periodic_jobs()
        except Exception:
            pass

        # 防重复清理
        if getattr(self, "_cleaning", False):
            return
        self._cleaning = True
        try:
            # 1) 先拉闸：让所有后台循环看到 False 自行退出
            for flag in ("pressure_thread_running",
                         "position_thread_running",
                         "pressure_control_running",
                         "multi_pressure_running",
                         "jump_running",
                         "auto_tare_running"):
                if hasattr(self, flag):
                    try:
                        setattr(self, flag, False)
                    except Exception:
                        pass

            # 给循环一个时间片感知退出
            try:
                time.sleep(0.05)
            except Exception:
                pass

            # 2) 尽量等待线程结束（非阻塞或有限时）
            def _join(name, timeout_daemon=0.2, timeout_nondaemon=0.8):
                t = getattr(self, name, None)
                if t and hasattr(t, "is_alive") and t.is_alive():
                    try:
                        # 后台线程（daemon）短等一下；非后台线程多等一会
                        to = timeout_daemon if getattr(t, "daemon", False) else timeout_nondaemon
                        t.join(timeout=to)
                    except Exception:
                        pass

            for th_name in ("pressure_thread", "position_thread",
                            "pressure_control_thread", "plot_update_thread",
                            "ui_update_thread"):
                _join(th_name)

            # 3) 关闭设备（走 ModbusController.close()，内部自带锁＆释放等待）
            def _close_dev(dev_attr):
                try:
                    dev = getattr(self, dev_attr, None)
                    if dev:
                        try:
                            dev.close()  # 使用你控制器类的 close，避免自己去动 dev.serial
                        except Exception:
                            pass
                        # 置空，避免后续误用
                        try:
                            setattr(self, dev_attr, None)
                        except Exception:
                            pass
                except Exception:
                    pass

            _close_dev("modbus1")
            _close_dev("modbus2")

            # 4) 再给 Windows 一点点时间真正释放串口句柄
            try:
                time.sleep(0.25)
            except Exception:
                pass

        finally:
            self._cleaning = False

    def panic_reset_motor(self):
        self.position_thread_running = False
        time.sleep(0.1)
        if self.modbus2:
            try:
                self.modbus2.close()
            except:
                pass
            self.modbus2 = None
        time.sleep(0.2)
        try:
            self.modbus2 = ModbusController(self.port_cb2.get(), int(self.baud_entry2.get()), 1)
            self.controller_connected = True
            self.connection_status_var2.set("已连接")
            self.log("✓ 电机串口已重置并重连")
        except Exception as e:
            self.controller_connected = False
            self.connection_status_var2.set("未连接")
            self.log(f"× 重置失败: {e}")


if __name__ == "__main__":
    app = App()
    app._window.mainloop()
