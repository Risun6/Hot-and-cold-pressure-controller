"""统一的多程序启动器：以标签页形式承载温控、压力与多序列调度界面。"""

from __future__ import annotations

import importlib.util
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from types import ModuleType

import ttkbootstrap as ttk
from tkinter import scrolledtext

ROOT_DIR = Path(__file__).resolve().parent


def _load_module(module_name: str, file_name: str) -> ModuleType:
    """加载带空格文件名的 GUI 模块，避免重复导入。"""
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, ROOT_DIR / file_name)
    if spec is None or spec.loader is None:  # pragma: no cover - 极端情况
        raise ImportError(f"无法加载模块 {file_name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


temperature_module = _load_module("temperature_controller_app", "Hot and cold controller.py")
pressure_module = _load_module("pressure_controller_app", "pressure controller.py")

from multi_sequence_controller import MultiSequenceApp  # noqa: E402  # 该文件命名规范

TemperatureApp = temperature_module.App  # type: ignore[attr-defined]
PressureApp = pressure_module.App  # type: ignore[attr-defined]


class UnifiedController(ttk.Window):
    def __init__(self) -> None:
        super().__init__(themename="cosmo")
        self.title("冷热压力一体化控制台")
        self.geometry("1720x980")

        paned = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        left_panel = ttk.Frame(paned)
        right_panel = ttk.Labelframe(paned, text="日志", padding=8)

        paned.add(left_panel, weight=6)
        paned.add(right_panel, weight=1)

        self.notebook = ttk.Notebook(left_panel, padding=8)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.log_text = scrolledtext.ScrolledText(right_panel, state="disabled", width=36)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        self._log_lock = threading.Lock()

        temp_tab = ttk.Frame(self.notebook)
        press_tab = ttk.Frame(self.notebook)
        seq_tab = ttk.Frame(self.notebook)

        for tab, text in (
            (temp_tab, "温度控制"),
            (press_tab, "压力控制"),
            (seq_tab, "多序列执行"),
        ):
            self.notebook.add(tab, text=text)

        self.temperature_app = TemperatureApp(temp_tab)
        self.pressure_app = PressureApp(press_tab)
        self.sequence_app = MultiSequenceApp(
            seq_tab,
            temp_controller=self.temperature_app,
            pressure_controller=self.pressure_app,
        )

        for child in (self.temperature_app, self.pressure_app, self.sequence_app):
            setter = getattr(child, "set_external_logger", None)
            if callable(setter):
                setter(self.append_log)

        # 统一填充，使三个子界面都自适应可用空间
        for child in (self.temperature_app.container, self.pressure_app, self.sequence_app):
            try:
                child.pack_configure(fill=tk.BOTH, expand=True)
            except Exception:
                pass

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(100, self._maximize_window)

    def append_log(self, message: str) -> None:
        if threading.current_thread() is not threading.main_thread():
            self.after(0, lambda m=message: self.append_log(m))
            return
        timestamp = time.strftime("%H:%M:%S")
        text = f"{timestamp} - {message}\n"
        with self._log_lock:
            self.log_text.configure(state="normal")
            self.log_text.insert(tk.END, text)
            self.log_text.see(tk.END)
            self.log_text.configure(state="disabled")

    def _on_close(self) -> None:
        for cleanup in (
            getattr(self.temperature_app, "cleanup", None),
            lambda: self.pressure_app.shutdown(destroy_window=False),
            lambda: self.sequence_app.shutdown(destroy_window=False),
        ):
            if cleanup is None:
                continue
            try:
                cleanup()
            except Exception:
                pass
        self.destroy()

    def _maximize_window(self) -> None:
        for attr in ("state", "wm_state"):
            method = getattr(self, attr, None)
            if callable(method):
                try:
                    method("zoomed")
                    return
                except tk.TclError:
                    continue
        attributes = getattr(self, "attributes", None)
        if callable(attributes):
            try:
                attributes("-zoomed", True)
            except tk.TclError:
                pass


def main() -> None:
    app = UnifiedController()
    app.mainloop()


if __name__ == "__main__":
    main()
