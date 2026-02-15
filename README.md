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
│   ├── train_model_transformer.py     # Transformer 模型训练脚本
│   ├── train_model_random_forest.py   # 随机森林模型训练脚本
│   ├── requirements.txt               # Python 依赖
│   ├── data/                          # 训练数据（CSV）
│   ├── models_LSTM/                   # 已训练 LSTM 产物
│   ├── models_transformer/            # 已训练 Transformer 产物
│   └── models_random_forest/          # 已训练随机森林产物
│
└── README.md
```

---

## 1. ESP32 端能力概览

`ESP32/BTR/src/main.cpp` 已实现：

- **多传感器采集**（温湿度、气体、PT100/MAX31865、MAX30105 等）；
- **2Hz 采样**（每 500ms 一次）并维护最近 10 秒历史窗口；
- **本地状态分级**：`NORMAL` / `WARNING` / `DANGER`；
- **声光与屏幕联动**：WS2812、蜂鸣器、OLED；
- **MQTT 通信**：
  - 发布主题：`sensor`
  - 订阅主题：`statue`
- **远程纠偏覆盖逻辑**：当云端下发状态覆盖后，本地连续正常一段时间再自动解除。

> 注意：WiFi、MQTT 私钥、各类阈值都在 `main.cpp` 内以常量定义，请按你的硬件环境修改。

---

## 2. Python 端能力概览

### 2.1 在线推理（`computer/main.py`）

- 连接 MQTT Broker（`bemfa.com:9501`）；
- 订阅 `sensor` 主题，解析 ESP32 上报 JSON；
- 使用 `models_LSTM/` 中模型与 scaler 对**内部温度**和**环境温度**分别推理；
- 当 AI 结论与 ESP32 本地状态不一致时，向 `statue` 主题回发 `normal` 或 `danger`，用于纠偏。

### 2.2 模型训练脚本

- `train_modelLSTM.py`：LSTM 时序模型（带差分特征与阈值优化）；
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
python train_model_transformer.py
# 或
python train_model_random_forest.py
```

---

## 4. 数据格式约定

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

---

## 5. 依赖

`computer/requirements.txt` 当前依赖：
- paho-mqtt
- numpy
- pandas
- scikit-learn
- tensorflow
- matplotlib
- seaborn

---

## 6. 说明

- 仓库中已包含部分历史训练产物（`models_*`）与备份数据（`data_backup_*`）；
- 如需复现实验，建议从 `computer/data/` 重新训练并更新对应模型目录；
- ESP32 示例代码位于 `ESP32/BTR/examples/`，可用于单模块联调。
