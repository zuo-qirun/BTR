# BTR（Battery Thermal Runaway）电池热失控监测与预测系统

本仓库包含一个**端云协同**的电池热失控监测方案：
- **ESP32 端**负责多传感器采集、状态判定、声光告警和 MQTT 上报；
- **Computer 端（Python）**负责模型训练与在线推理，对 ESP32 状态进行二次校验与纠偏。

## 项目结构

```text
BTR/
├── ESP32/BTR/                         # PlatformIO 工程（嵌入式固件）
│   ├── src/main.cpp                   # 主程序：采样、状态机、MQTT 收发、OLED/蜂鸣器/灯控
│   ├── platformio.ini                 # 板卡与依赖配置
│   └── examples/                      # 各传感器/模块示例代码
│
├── computer/                          # Python 端（训练 + 在线预测）
│   ├── main.py                        # MQTT 监听 + 模型推理 + 状态纠偏回传
│   ├── train_modelLSTM.py             # LSTM 模型训练脚本
│   ├── train_model_lstm_60s.py        # LSTM 模型（60 秒序列）
│   ├── train_model_cnnlstm.py         # CNN+LSTM 混合模型训练脚本
│   ├── train_model_transformer.py     # Transformer 模型训练脚本
│   ├── train_model_random_forest.py   # 随机森林模型训练脚本
│   ├── requirements.txt               # Python 依赖
│   ├── data/                          # 训练数据（CSV）
│   ├── models_LSTM/                   # 已训练 LSTM 产物
│   ├── models_lstm_60s/               # 已训练 LSTM(60s) 产物
│   ├── models_cnn_lstm/               # 已训练 CNN-LSTM 产物
│   ├── models_transformer/            # 已训练 Transformer 产物
│   └── models_random_forest/          # 已训练随机森林产物
│
└── README.md
```

---

## 1. ESP32 端能力概览

`ESP32/BTR/src/main.cpp` 已实现：

- **多传感器采集**：
  - 温湿度：SHT4x (I2C)
  - 内部温度：MAX31865 + PT100 (SPI)
  - 气体传感器：SGP41 (VOC/NOx, I2C)、MQ-2/4/7/8 (ADC)
  - 烟雾检测：MAX30105 (I2C)
- **2Hz 采样**（每 500ms 一次）并维护最近 10 秒历史窗口；
- **本地状态分级**：`NORMAL` / `WARNING` / `DANGER`；
- **声光与屏幕联动**：WS2812、蜂鸣器、OLED（5 个界面：状态、温湿度、气体、VOC、MAX30105）；
- **MQTT 通信**：
  - 发布主题：`sensor`
  - 订阅主题：`statue`<sup>†</sup>
- **远程纠偏覆盖逻辑**：当云端下发状态覆盖后，本地连续正常 60 秒再自动解除。

> <sup>†</sup> **已知问题**：MQTT 订阅主题拼写为 `statue` 而非 `status`，代码中已保持一致，暂不修正以避免破坏兼容性。

> 注意：WiFi、MQTT 私钥、各类阈值都在 `main.cpp` 内以常量定义，请按你的硬件环境修改。

---

## 2. Python 端能力概览

### 2.1 在线推理（`computer/main.py`）

- 连接 MQTT Broker（`bemfa.com:9501`）；
- 订阅 `sensor` 主题，解析 ESP32 上报 JSON；
- 使用 `models_LSTM/` 中模型与 scaler 对**内部温度**和**环境温度**分别推理；
- 当 AI 结论与 ESP32 本地状态不一致时，向 `statue`<sup>†</sup> 主题回发 `normal` 或 `danger`，用于纠偏。

### 2.2 模型训练脚本

- `train_modelLSTM.py`：LSTM 时序模型（带差分特征与阈值优化）；
- `train_model_lstm_60s.py`：LSTM 模型（60 秒序列输入）；
- `train_model_cnnlstm.py`：CNN + LSTM 混合模型；
- `train_model_transformer.py`：Transformer 编码器时序模型；
- `train_model_random_forest.py`：统计特征 + 随机森林模型。

三个训练脚本都基于 `computer/data/*.csv` 构建样本，默认将模型产物输出到各自目录。

---

## 3. 快速开始

### 3.1 Python 环境

```bash
cd computer
pip install -r requirements.txt
```

### 3.2 运行在线推理服务

```bash
cd computer
python main.py
```

运行前请确认：
- `computer/models_LSTM/config.pkl`、模型文件和 scaler 文件存在；
- MQTT 配置与 ESP32 端一致（Broker / 主题 / ClientID）。

### 3.3 训练模型（按需）

```bash
cd computer
python train_modelLSTM.py
# 或
python train_model_lstm_60s.py
python train_model_cnnlstm.py
python train_model_transformer.py
python train_model_random_forest.py
```

---

## 4. 硬件清单与引脚配置

### 4.1 硬件组件

| 组件 | 型号 | 数量 |
|------|------|------|
| 主控板 | ESP32-S3 (ESP32-S3-DevKitM-1) | 1 |
| 温湿度传感器 | Sensirion SHT4x | 1 |
| RTD 温度传感器 | Adafruit MAX31865 + PT100 | 1 |
| 气体传感器 | Sensirion SGP41 (VOC/NOx) | 1 |
| MQ 气体传感器 | MQ-2, MQ-4, MQ-7, MQ-8 | 各 1 |
| 烟雾传感器 | SparkFun MAX30105 | 1 |
| OLED 显示屏 | SSD1306 128x64 (I2C) | 1 |
| RGB 灯 | WS2812 NeoPixel | 1 |
| 蜂鸣器 | 有源蜂鸣器 (5V) | 1 |

### 4.2 引脚配置

| 传感器/模块 | ESP32-S3 引脚 |
|-------------|---------------|
| WS2812 RGB 灯 | GPIO 48 |
| 蜂鸣器 | GPIO 18 |
| 按键 | GPIO 17 |
| MAX31865 CS | GPIO 10 |
| MAX31865 MOSI | GPIO 11 |
| MAX31865 MISO | GPIO 13 |
| MAX31865 SCK | GPIO 12 |
| MQ-2 | GPIO 1 (ADC) |
| MQ-4 | GPIO 2 (ADC) |
| MQ-8 | GPIO 3 (ADC) |
| MQ-7 | GPIO 4 (ADC) |
| SHT4x / SGP41 / OLED / MAX30105 | I2C (GPIO 21/22) |

---

## 5. 数据格式

### 5.1 训练数据格式

训练数据采用 CSV，至少包含以下列：

```csv
Time,Temperature
0.0,31.03
0.5,31.05
1.0,31.08
```

命名约定：
- 正常样本：`temp_data_xxx.csv`
- 异常样本：`temp_data_xxx_exception.csv`

脚本会根据文件名中的 `_exception` 自动识别标签。

### 5.2 MQTT 数据格式

ESP32 上报 JSON 格式：

```json
{
  "timestamp_ms": 1234567890,
  "temp_ambient_c": 25.5,
  "temp_internal_c": 32.1,
  "humidity_percent": 45.2,
  "mq2_ppm": 15.3,
  "mq4_ppm": 12.1,
  "mq8_ppm": 8.5,
  "mq7_ppm": 22.0,
  "voc_index": 85,
  "max30105_smoke": 120,
  "max30105_temp_c": 33.5,
  "status": "NORMAL"
}
```

---

## 6. 依赖

### 6.1 Python 环境

- Python >= 3.8
- `computer/requirements.txt`:

```
paho-mqtt
numpy
pandas
scikit-learn
tensorflow
matplotlib
seaborn
```

### 6.2 ESP32 固件

- PlatformIO
- 框架：arduino
- 板卡：esp32-s3-devkitm-1

---

## 7. 说明

- 仓库中已包含部分历史训练产物（`models_*`）与备份数据（`data_backup_*`）；
- 如需复现实验，建议从 `computer/data/` 重新训练并更新对应模型目录；
- ESP32 示例代码位于 `ESP32/BTR/examples/`，可用于单模块联调。
