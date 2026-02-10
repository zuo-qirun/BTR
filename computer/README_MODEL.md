# 电池热失控预测系统

基于深度学习的电池热失控实时预测系统，使用LSTM神经网络分析温度时间序列数据。

## 功能特点

- 🔥 **实时预测**: 每0.5秒接收温度数据并预测热失控风险
- 🧠 **深度学习**: 使用LSTM神经网络捕捉温度变化模式
- 📊 **可视化**: 训练过程可视化和模型性能评估
- ⚠️ **智能预警**: 多级风险评估（低、中、高、极高）

## 系统架构

```
computer/
├── train_model.py          # 模型训练脚本
├── predict_realtime.py     # 实时预测脚本
├── data/                   # 训练数据目录
│   ├── temp_data_001.csv   # 正常数据
│   ├── temp_data_036_exception.csv  # 异常数据（热失控）
│   └── ...
├── models/                 # 模型保存目录（训练后生成）
│   ├── thermal_runaway_model.h5
│   ├── scaler.pkl
│   └── config.pkl
└── requirements.txt        # Python依赖
```

## 安装依赖

```bash
cd computer
pip install -r requirements.txt
```

## 使用方法

### 1. 训练模型

首次使用需要先训练模型：

```bash
python train_model.py
```

训练过程：
- 自动加载 `data/` 目录下的所有CSV文件
- 正常数据文件：`temp_data_XXX.csv`
- 异常数据文件：`temp_data_XXX_exception.csv`
- 训练完成后模型保存在 `models/` 目录

训练参数（可在脚本中修改）：
- `SEQUENCE_LENGTH = 60`: 使用过去30秒的数据（每0.5秒一个点）
- `PREDICTION_HORIZON = 20`: 预测未来10秒
- `EPOCHS = 50`: 训练轮数
- `BATCH_SIZE = 32`: 批次大小

### 2. 实时预测

#### 模拟模式（使用测试数据）

```bash
python predict_realtime.py
```

这将使用测试数据模拟实时预测过程。

#### 手动输入模式

```bash
python predict_realtime.py --manual
```

手动输入温度值进行预测：
```
温度 #1: 31.5
温度 #2: 31.6
温度 #3: 31.7
...
```

输入至少60个温度值后开始预测。

### 3. 集成到实时系统

在你的实时监控系统中集成预测器：

```python
from predict_realtime import RealTimePredictor

# 初始化预测器
predictor = RealTimePredictor()

# 每0.5秒接收温度数据
while True:
    temperature = get_temperature_from_sensor()  # 从传感器获取温度
    predictor.add_temperature(temperature)
    
    # 进行预测
    if predictor.can_predict():
        risk_prob, risk_level, warning = predictor.predict()
        
        print(f"热失控风险: {risk_prob:.2%} ({risk_level})")
        
        if warning:
            print("⚠️ 警告: 检测到热失控风险！")
            # 触发报警系统
            trigger_alarm()
    
    time.sleep(0.5)
```

## 数据格式

CSV文件格式：
```csv
Time,Temperature
0.0,31.03
0.5,31.05
1.0,31.08
1.5,31.12
...
```

- `Time`: 时间戳（秒）
- `Temperature`: 温度值（摄氏度）

## 模型说明

### 网络架构

```
输入层: (60, 1) - 60个时间步的温度序列
  ↓
LSTM层1: 128单元 + Dropout(0.3)
  ↓
LSTM层2: 64单元 + Dropout(0.3)
  ↓
LSTM层3: 32单元 + Dropout(0.3)
  ↓
全连接层: 16单元 + ReLU + Dropout(0.2)
  ↓
输出层: 1单元 + Sigmoid（热失控概率）
```

### 风险等级

| 风险概率 | 风险等级 | 是否警告 |
|---------|---------|---------|
| < 30%   | 低      | ❌      |
| 30-60%  | 中      | ❌      |
| 60-80%  | 高      | ⚠️      |
| > 80%   | 极高    | ⚠️      |

## 性能指标

训练完成后会显示：
- **准确率 (Accuracy)**: 整体预测准确度
- **精确率 (Precision)**: 预测为热失控的样本中真正热失控的比例
- **召回率 (Recall)**: 实际热失控样本中被正确识别的比例
- **混淆矩阵**: 详细的分类结果

## 可视化

训练完成后会生成 `models/training_history.png`，包含：
- 训练/验证损失曲线
- 训练/验证准确率曲线
- 训练/验证精确率曲线
- 训练/验证召回率曲线

## 注意事项

1. **数据质量**: 确保训练数据包含足够的正常和异常样本
2. **时间间隔**: 默认假设数据每0.5秒采集一次
3. **温度单位**: 确保所有温度数据使用相同单位（摄氏度）
4. **模型更新**: 收集到新数据后可重新训练以提高准确性

## 故障排除

### 问题：训练时内存不足
**解决方案**: 减小 `BATCH_SIZE` 或 `SEQUENCE_LENGTH`

### 问题：预测准确率低
**解决方案**: 
- 增加训练数据量
- 调整 `SEQUENCE_LENGTH` 和 `PREDICTION_HORIZON`
- 增加训练轮数 `EPOCHS`

### 问题：模型文件未找到
**解决方案**: 先运行 `train_model.py` 训练模型

## 技术栈

- **Python 3.8+**
- **TensorFlow/Keras**: 深度学习框架
- **NumPy/Pandas**: 数据处理
- **Scikit-learn**: 数据预处理和评估
- **Matplotlib**: 可视化

## 许可证

MIT License

## 联系方式

如有问题或建议，请联系开发团队。
