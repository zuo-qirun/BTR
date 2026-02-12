# 电池热失控预测系统 (BTR - Battery Thermal Runaway)

基于深度学习的电池热失控实时监测与预测系统。

## 🎯 项目概述

本项目包含两个主要部分：

1. **ESP32硬件监测系统** - 实时采集电池温度、湿度、气体等传感器数据
2. **AI预测系统** - 使用LSTM深度学习模型预测电池热失控风险

## 📁 项目结构

```
BTR/
├── computer/                    # AI预测系统（Python）
│   ├── train_model.py          # 模型训练脚本
│   ├── predict_realtime.py     # 实时预测脚本
│   ├── quick_start.py          # 快速开始脚本
│   ├── analyze_data.py         # 数据分析脚本
│   ├── examples.py             # 使用示例
│   ├── 使用指南.md              # 快速使用指南
│   ├── README_MODEL.md         # 详细技术文档
│   ├── requirements.txt        # Python依赖
│   ├── data/                   # 训练数据
│   └── models/                 # 训练好的模型（训练后生成）
│
├── ESP32/                      # ESP32硬件系统
│   └── BTR/                    # 固件代码
│       └── platformio.ini      # PlatformIO配置
│
└── README.md                   # 本文件
```

## 🚀 快速开始

### AI预测系统

```bash
# 1. 进入目录
cd computer

# 2. 安装依赖
pip install -r requirements.txt

# 3. 运行快速开始脚本（自动检查环境、训练模型、运行演示）
python quick_start.py
```

详细使用说明请查看：
- **快速入门**: `computer/使用指南.md`
- **技术文档**: `computer/README_MODEL.md`

### 核心功能

#### 1. 训练模型
```bash
python train_model.py
```
- 自动加载 `data/` 目录下的所有训练数据
- 使用LSTM神经网络训练
- 生成模型文件到 `models/` 目录

#### 2. 实时预测
```bash
# 模拟模式（使用测试数据）
python predict_realtime.py

# 手动输入模式
python predict_realtime.py --manual
```

#### 3. 数据分析
```bash
python analyze_data.py
```
- 分析数据质量
- 生成可视化报告
- 检查数据问题

#### 4. 查看示例
```bash
python examples.py
```
- 基本使用示例
- 热失控检测示例
- 连续监控示例
- 批量预测示例

## 💡 使用场景

### 场景1：实时监控

```python
from predict_realtime import RealTimePredictor

predictor = RealTimePredictor()

while True:
    # 每0.5秒获取温度
    temperature = get_sensor_temperature()
    predictor.add_temperature(temperature)
    
    if predictor.can_predict():
        risk_prob, risk_level, warning = predictor.predict()
        
        if warning:
            print(f"⚠️ 警告: 热失控风险 {risk_prob:.2%}")
            trigger_alarm()
    
    time.sleep(0.5)
```

### 场景2：批量分析

```python
from predict_realtime import RealTimePredictor

predictor = RealTimePredictor()

# 加载历史数据
for temp in historical_temperatures:
    predictor.add_temperature(temp)

# 预测
risk_prob, risk_level, warning = predictor.predict()
print(f"风险评估: {risk_prob:.2%} ({risk_level})")
```

## 📊 系统特点

### AI预测系统

- ✅ **深度学习**: 使用LSTM神经网络捕捉温度时间序列模式
- ✅ **实时预测**: 每0.5秒接收数据，实时预测热失控风险
- ✅ **多级预警**: 低、中、高、极高四级风险评估
- ✅ **高准确率**: 基于大量训练数据，准确识别热失控模式
- ✅ **易于集成**: 简单的API接口，易于集成到现有系统

### 技术参数

- **输入**: 60个温度数据点（过去30秒，每0.5秒一个点）
- **输出**: 热失控风险概率（0-100%）
- **预测范围**: 未来10秒
- **模型**: 3层LSTM + 全连接层
- **训练数据**: 80个CSV文件（正常+异常数据）

## 📈 风险等级

| 风险概率 | 等级 | 说明 | 建议 |
|---------|------|------|------|
| 0-30% | 低 🟢 | 正常运行 | 继续监控 |
| 30-60% | 中 🟡 | 温度异常 | 加强监控 |
| 60-80% | 高 🟠 | 可能热失控 | 准备应急 |
| 80-100% | 极高 🔴 | 热失控风险 | 立即处理 |

## 🛠️ 技术栈

### AI系统
- Python 3.8+
- TensorFlow/Keras
- NumPy/Pandas
- Scikit-learn
- Matplotlib

### 硬件系统
- ESP32
- 温度传感器
- 湿度传感器
- 气体传感器（MQ系列）

## 📖 文档

- **快速使用指南**: `computer/使用指南.md`
- **详细技术文档**: `computer/README_MODEL.md`
- **代码示例**: `computer/examples.py`

## 🔧 配置

主要配置参数在 `train_model.py` 中：

```python
SEQUENCE_LENGTH = 60      # 时间序列长度（30秒）
PREDICTION_HORIZON = 20   # 预测未来时间步（10秒）
BATCH_SIZE = 32          # 训练批次大小
EPOCHS = 50              # 训练轮数
```

## 📝 数据格式

训练数据CSV格式：
```csv
Time,Temperature
0.0,31.03
0.5,31.05
1.0,31.08
...
```

- 正常数据: `temp_data_XXX.csv`
- 异常数据: `temp_data_XXX_exception.csv`

## 🤝 贡献

欢迎提交问题和改进建议！

## 📄 许可证

MIT License

## 📞 联系

如有问题，请查看文档或提交Issue。

---

**开发日期**: 2026年2月9日
