"""Shared helpers for simulated mouse clicks and coordinate capture."""

from __future__ import annotations

import sys
import threading
import time as _time
from typing import Callable, Optional, Tuple

try:  # pragma: no cover - optional dependency
    import pyautogui  # type: ignore
except Exception:  # pragma: no cover - fallback handled in helpers
    pyautogui = None  # type: ignore

try:  # pragma: no cover - Windows specific fallback
    import ctypes
except Exception:  # pragma: no cover - non-Windows environments do not need ctypes
    ctypes = None  # type: ignore

import tkinter as tk
from tkinter import messagebox


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


def capture_click_point(master: tk.Misc, *, title: str = "设置点击点", hint: str = "移动鼠标到目标处，按 Enter 键记录", reporter: Reporter = None) -> Optional[Point]:
    """Prompt the user to move the cursor and press Enter to capture screen coordinates."""

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
            except Exception:  # pragma: no cover - safe fallback
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


def perform_click_async(x: int, y: int, *, button: str = "left", double: bool = False, delay_ms: int = 0, reporter: Reporter = None) -> threading.Thread:
    """Perform a simulated click on a background thread and report the result."""

    def _worker() -> None:
        try:
            if delay_ms > 0:
                _time.sleep(delay_ms / 1000.0)
            _click_once(int(x), int(y), button)
            if double:
                _time.sleep(0.03)
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


__all__ = ["capture_click_point", "perform_click_async"]
