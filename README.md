# Hot and Cold Pressure Controller

该仓库包含多个使用 Tkinter/ttkbootstrap 构建的控制界面脚本（温控、压力控制、多序列调度以及 Keithley 2400/2636B 仪器演示）。

## 依赖安装

运行界面前请先安装所需的 Python 库：

```bash
pip install -r requirements.txt
```

如果只需要部分功能，可按需安装（均通过 `pip install <包名>`）：

- `numpy`：数据处理与仿真曲线计算。
- `pyvisa`、`pyvisa-py`：与 Keithley 2400/2636B 等仪器通信（无硬件可选装）。
- `pyserial`：串口通讯（压力控制及设备列表）。
- `ttkbootstrap`：Tkinter 主题与组件库。
- `Pillow`：图像处理（报表截图、图片叠加）。
- `matplotlib`：绘制实时曲线并嵌入 Tkinter。
- `openpyxl`：生成/写入 Excel 报表。
- `pyautogui`、`keyboard`：可选的自动化/全局热键支持（缺失时相关功能将被跳过）。

> 注意：Tkinter 随大多数 Python 发行版自带；在精简环境下请确保已安装系统级 Tk 支持。
