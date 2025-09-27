"""启动温控、压力控制与多序列调度界面的统一入口。

运行本脚本会启动三个独立的子进程，每个子进程负责一个原有的
Tk 图形界面程序。按 Ctrl+C 或关闭任意窗口时会自动清理剩余进程。
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

ROOT_DIR = Path(__file__).resolve().parent


@dataclass
class ManagedProcess:
    name: str
    path: Path
    process: Optional[subprocess.Popen] = None

    def launch(self) -> None:
        if not self.path.exists():
            raise FileNotFoundError(f"找不到 {self.name} 程序: {self.path}")
        cmd = [sys.executable, str(self.path)]
        env = os.environ.copy()
        self.process = subprocess.Popen(cmd, cwd=str(self.path.parent), env=env)

    def terminate(self, timeout: float = 3.0) -> None:
        if not self.process:
            return
        proc = self.process
        if proc.poll() is not None:
            return
        try:
            proc.terminate()
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
        finally:
            self.process = None

    def is_running(self) -> bool:
        return bool(self.process) and self.process.poll() is None


class Launcher:
    def __init__(self):
        self.apps: List[ManagedProcess] = [
            ManagedProcess("温度控制", ROOT_DIR / "Hot and cold controller.py"),
            ManagedProcess("压力控制", ROOT_DIR / "pressure controller.py"),
            ManagedProcess("多序列调度", ROOT_DIR / "multi_sequence_controller.py"),
        ]
        self._register_signals()

    def _register_signals(self) -> None:
        def handler(signum, frame):  # noqa: ARG001 - 回调签名要求
            print(f"\n收到信号 {signum}，准备退出…")
            self.shutdown()
            sys.exit(0)

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, handler)
            except Exception:
                # 某些平台（如 Windows）不支持 SIGTERM
                pass

    def start_all(self) -> None:
        for app in self.apps:
            try:
                app.launch()
            except Exception as exc:
                print(f"× 无法启动 {app.name}: {exc}")
            else:
                print(f"✓ 已启动 {app.name} (PID={app.process.pid})")

    def monitor(self) -> None:
        try:
            while True:
                alive = [app for app in self.apps if app.is_running()]
                if not alive:
                    print("所有子程序已退出，主程序结束。")
                    break
                for app in list(self.apps):
                    if app.process and app.process.poll() is not None:
                        code = app.process.returncode
                        status = "正常退出" if code == 0 else f"异常退出 (code={code})"
                        print(f"- {app.name} {status}")
                        app.process = None
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("\n收到键盘中断，正在关闭所有子程序…")
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        for app in self.apps:
            if app.is_running():
                print(f"→ 关闭 {app.name}…")
            app.terminate()


def main() -> None:
    launcher = Launcher()
    launcher.start_all()
    launcher.monitor()


if __name__ == "__main__":
    main()
