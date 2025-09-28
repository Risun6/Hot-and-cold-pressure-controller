"""统一的多程序启动器：以标签页形式承载温控、压力与多序列调度界面。"""

from __future__ import annotations

import importlib.util
import sys
import tkinter as tk
from pathlib import Path
from types import ModuleType

import ttkbootstrap as ttk

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

        self.notebook = ttk.Notebook(self, padding=8)
        self.notebook.pack(fill=tk.BOTH, expand=True)

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
        self.sequence_app = MultiSequenceApp(seq_tab)

        # 统一填充，使三个子界面都自适应可用空间
        for child in (self.temperature_app.container, self.pressure_app, self.sequence_app):
            try:
                child.pack_configure(fill=tk.BOTH, expand=True)
            except Exception:
                pass

        self.protocol("WM_DELETE_WINDOW", self._on_close)

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


def main() -> None:
    app = UnifiedController()
    app.mainloop()


if __name__ == "__main__":
    main()
