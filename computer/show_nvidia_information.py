import tkinter as tk
from tkinter import ttk
import pynvml  # nvidia-ml-py 提供的库
import threading
import time

class NvidiaSmiGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("实时 nvidia-smi 监控")
        self.root.geometry("900x600")
        self.root.resizable(True, True)

        # 尝试初始化 NVML
        try:
            pynvml.nvmlInit()
            self.nvml_available = True
            self.device_count = pynvml.nvmlDeviceGetCount()
        except pynvml.NVMLError as e:
            self.nvml_available = False
            self.device_count = 0
            print(f"NVML 初始化失败: {e}")

        # 创建主框架
        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 如果无可用 GPU，显示错误信息
        if not self.nvml_available or self.device_count == 0:
            label = ttk.Label(main_frame, text="未检测到 NVIDIA GPU 或 NVML 不可用", foreground="red")
            label.pack(pady=20)
            return

        # 创建滚动条
        canvas = tk.Canvas(main_frame, borderwidth=0)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill=tk.BOTH, expand=True)
        scrollbar.pack(side="right", fill="y")

        # 存储每个 GPU 对应的控件引用，以便更新
        self.gui_elements = []

        # 为每个 GPU 创建一个信息卡片
        for i in range(self.device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle).decode('utf-8') if isinstance(pynvml.nvmlDeviceGetName(handle), bytes) else pynvml.nvmlDeviceGetName(handle)

            # 卡片框架
            card = ttk.LabelFrame(scrollable_frame, text=f"GPU {i}: {name}", padding="10")
            card.pack(fill=tk.X, pady=5, padx=5)

            # 使用网格布局放置信息
            # 温度
            ttk.Label(card, text="温度:").grid(row=0, column=0, sticky=tk.W, pady=2)
            temp_label = ttk.Label(card, text="-- °C")
            temp_label.grid(row=0, column=1, sticky=tk.W, padx=5)

            # 功率
            ttk.Label(card, text="功率:").grid(row=1, column=0, sticky=tk.W, pady=2)
            power_label = ttk.Label(card, text="-- / -- W")
            power_label.grid(row=1, column=1, sticky=tk.W, padx=5)

            # 显存
            ttk.Label(card, text="显存:").grid(row=2, column=0, sticky=tk.W, pady=2)
            mem_label = ttk.Label(card, text="-- / -- MiB")
            mem_label.grid(row=2, column=1, sticky=tk.W, padx=5)
            mem_progress = ttk.Progressbar(card, length=200, mode='determinate')
            mem_progress.grid(row=2, column=2, padx=5)

            # GPU 利用率
            ttk.Label(card, text="GPU 利用率:").grid(row=3, column=0, sticky=tk.W, pady=2)
            util_label = ttk.Label(card, text="-- %")
            util_label.grid(row=3, column=1, sticky=tk.W, padx=5)
            util_progress = ttk.Progressbar(card, length=200, mode='determinate')
            util_progress.grid(row=3, column=2, padx=5)

            # 风扇速度（如果支持）
            ttk.Label(card, text="风扇:").grid(row=4, column=0, sticky=tk.W, pady=2)
            fan_label = ttk.Label(card, text="-- %")
            fan_label.grid(row=4, column=1, sticky=tk.W, padx=5)

            # 保存该 GPU 的所有控件引用
            self.gui_elements.append({
                'handle': handle,
                'temp': temp_label,
                'power': power_label,
                'mem': mem_label,
                'mem_progress': mem_progress,
                'util': util_label,
                'util_progress': util_progress,
                'fan': fan_label
            })

        # 启动定时更新
        self.update_stats()

        # 关闭窗口时清理 NVML
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def update_stats(self):
        """从 NVML 获取最新数据并更新界面"""
        if not self.nvml_available:
            return

        for elem in self.gui_elements:
            handle = elem['handle']

            try:
                # 温度 (通常为 GPU 热点温度，此处用核心温度)
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                elem['temp'].config(text=f"{temp} °C")

                # 功率 (毫瓦 -> 瓦)
                power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
                elem['power'].config(text=f"{power_usage:.1f} / {power_limit:.1f} W")

                # 显存 (字节 -> MiB)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                used_mib = mem_info.used / 1024 / 1024
                total_mib = mem_info.total / 1024 / 1024
                elem['mem'].config(text=f"{used_mib:.0f} / {total_mib:.0f} MiB")
                mem_percent = (used_mib / total_mib) * 100
                elem['mem_progress']['value'] = mem_percent

                # GPU 利用率
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = util.gpu
                elem['util'].config(text=f"{gpu_util} %")
                elem['util_progress']['value'] = gpu_util

                # 风扇速度 (如果有多个风扇，取第一个)
                try:
                    fan_speed = pynvml.nvmlDeviceGetFanSpeed(handle)  # 返回百分比
                    elem['fan'].config(text=f"{fan_speed} %")
                except pynvml.NVMLError_NotSupported:
                    elem['fan'].config(text="N/A")
                except pynvml.NVMLError_InvalidArgument:
                    # 某些 GPU 可能没有风扇
                    elem['fan'].config(text="N/A")

            except pynvml.NVMLError as e:
                print(f"读取 GPU 信息出错: {e}")

        # 每 1000 毫秒 (1 秒) 更新一次
        self.root.after(1000, self.update_stats)

    def on_close(self):
        """关闭窗口时的清理工作"""
        if self.nvml_available:
            pynvml.nvmlShutdown()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = NvidiaSmiGUI(root)
    root.mainloop()