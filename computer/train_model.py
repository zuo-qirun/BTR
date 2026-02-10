"""
电池热失控预测模型训练脚本
使用LSTM神经网络基于时间序列温度数据预测热失控
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle
from datetime import datetime

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ====== 配置参数 ======
DATA_DIR = "data"  # 数据目录
MODEL_DIR = "models"  # 模型保存目录
SEQUENCE_LENGTH = 60  # 时间序列长度（使用过去30秒的数据，每0.5秒一个点）
PREDICTION_HORIZON = 40  # 预测未来10秒（20个时间步）
BATCH_SIZE = 128
EPOCHS = 30
LEARNING_RATE = 0.001
NORMAL_SAMPLE_STRIDE = 30  # 正常数据采样步长（每30个取一个）
EXCEPTION_SAMPLE_STRIDE = 5  # 异常数据采样步长（每5个取一个，增加异常样本）
SAMPLE_STRIDE = 10  # 采样步长（每隔N个样本取一个）
CLASS_WEIGHT = {0: 1.0, 1: 50.0}  # 类别权重：热失控样本权重提高50倍

class BatteryThermalRunawayPredictor:
    """电池热失控预测器"""
    
    def __init__(self, sequence_length=SEQUENCE_LENGTH, prediction_horizon=PREDICTION_HORIZON):
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.model = None
        self.scaler = StandardScaler()
        self.history = None
        
    def load_data(self, data_dir):
        """加载所有CSV数据文件"""
        print("正在加载数据...")
        
        all_sequences = []
        all_labels = []
        
        # 获取所有CSV文件
        csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
        
        for csv_file in csv_files:
            try:
                # 判断是否为异常文件（热失控）
                is_exception = "_exception" in os.path.basename(csv_file)
                label = 1 if is_exception else 0
                
                # 读取数据
                df = pd.read_csv(csv_file)
                
                # 确保数据按时间排序
                df = df.sort_values('Time')
                
                # 提取温度数据
                temperatures = df['Temperature'].values
                
                # 根据文件类型选择采样步长
                sample_stride = EXCEPTION_SAMPLE_STRIDE if is_exception else NORMAL_SAMPLE_STRIDE
                
                # 创建时间序列样本（简单采样：每隔sample_stride个取一个）
                max_sequences = len(temperatures) - self.sequence_length - self.prediction_horizon
                
                if max_sequences <= 0:
                    print(f"  ⚠️  数据点不足，跳过")
                    continue
                
                # 简单采样策略：根据文件类型使用不同步长
                sample_count = 0
                for i in range(0, max_sequences, sample_stride):
                        
                    sequence = temperatures[i:i + self.sequence_length]
                    # 检查未来是否会发生热失控
                    future_temps = temperatures[i + self.sequence_length:i + self.sequence_length + self.prediction_horizon]
                    
                    # 如果是异常文件，且未来温度急剧上升，标记为热失控
                    if is_exception:
                        temp_increase = np.max(future_temps) - sequence[-1]
                        if temp_increase > 5:  # 温度上升超过5度
                            label = 1
                        else:
                            label = 0
                    
                    all_sequences.append(sequence)
                    all_labels.append(label)
                    sample_count += 1
                
                print(f"已加载: {os.path.basename(csv_file)} - {'异常' if is_exception else '正常'} - {len(temperatures)} 个数据点 (采样 {sample_count} 个样本)")
                
            except Exception as e:
                print(f"加载文件 {csv_file} 时出错: {e}")
                continue
        
        # 转换为numpy数组
        X = np.array(all_sequences)
        y = np.array(all_labels)
        
        print(f"\n数据加载完成:")
        print(f"总样本数: {len(X)}")
        print(f"正常样本: {np.sum(y == 0)}")
        print(f"异常样本: {np.sum(y == 1)}")
        
        return X, y
    
    def preprocess_data(self, X, y):
        """数据预处理和标准化"""
        print("\n正在预处理数据...")
        
        # 重塑数据以进行标准化
        X_reshaped = X.reshape(-1, 1)
        X_scaled = self.scaler.fit_transform(X_reshaped)
        X_scaled = X_scaled.reshape(X.shape)
        
        # 添加特征维度
        X_scaled = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"训练集大小: {len(X_train)}")
        print(f"测试集大小: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def build_model(self, input_shape):
        """构建LSTM模型"""
        print("\n构建模型...")
        
        model = keras.Sequential([
            # 第一层LSTM
            layers.LSTM(128, return_sequences=True, input_shape=input_shape),
            layers.Dropout(0.3),
            
            # 第二层LSTM
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.3),
            
            # 第三层LSTM
            layers.LSTM(32),
            layers.Dropout(0.3),
            
            # 全连接层
            layers.Dense(16, activation='relu'),
            layers.Dropout(0.2),
            
            # 输出层（二分类）
            layers.Dense(1, activation='sigmoid')
        ])
        
        # 编译模型
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )
        
        self.model = model
        print(model.summary())
        
        return model
    
    def train(self, X_train, y_train, X_test, y_test):
        """训练模型"""
        print("\n开始训练模型...")
        
        # 回调函数
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            )
        ]
        
        # 训练模型
        self.history = self.model.fit(
            X_train, y_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            class_weight=CLASS_WEIGHT,  # 添加类别权重
            verbose=1
        )
        
        print("\n模型训练完成!")
        
    def evaluate(self, X_test, y_test):
        """评估模型"""
        print("\n评估模型...")
        
        # 评估
        results = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"\n测试集结果:")
        print(f"Loss: {results[0]:.4f}")
        print(f"Accuracy: {results[1]:.4f}")
        print(f"Precision: {results[2]:.4f}")
        print(f"Recall: {results[3]:.4f}")
        
        # 预测
        y_pred_prob = self.model.predict(X_test)
        y_pred = (y_pred_prob > 0.3).astype(int)  # 降低阈值到0.3以提高召回率
        
        # 混淆矩阵
        from sklearn.metrics import confusion_matrix, classification_report
        cm = confusion_matrix(y_test, y_pred)
        print("\n混淆矩阵:")
        print(cm)
        print("\n分类报告:")
        print(classification_report(y_test, y_pred, target_names=['正常', '热失控']))
        
        return results
    
    def plot_training_history(self):
        """绘制训练历史"""
        if self.history is None:
            print("没有训练历史可绘制")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(self.history.history['loss'], label='训练损失')
        axes[0, 0].plot(self.history.history['val_loss'], label='验证损失')
        axes[0, 0].set_title('模型损失')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy
        axes[0, 1].plot(self.history.history['accuracy'], label='训练准确率')
        axes[0, 1].plot(self.history.history['val_accuracy'], label='验证准确率')
        axes[0, 1].set_title('模型准确率')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Precision
        axes[1, 0].plot(self.history.history['precision'], label='训练精确率')
        axes[1, 0].plot(self.history.history['val_precision'], label='验证精确率')
        axes[1, 0].set_title('模型精确率')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Recall
        axes[1, 1].plot(self.history.history['recall'], label='训练召回率')
        axes[1, 1].plot(self.history.history['val_recall'], label='验证召回率')
        axes[1, 1].set_title('模型召回率')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # 保存图表
        os.makedirs(MODEL_DIR, exist_ok=True)
        plt.savefig(os.path.join(MODEL_DIR, 'training_history.png'))
        print(f"\n训练历史图表已保存到: {os.path.join(MODEL_DIR, 'training_history.png')}")
        plt.show()
    
    def save_model(self, model_dir=MODEL_DIR):
        """保存模型和预处理器"""
        os.makedirs(model_dir, exist_ok=True)
        
        # 保存模型
        model_path = os.path.join(model_dir, 'thermal_runaway_model.h5')
        self.model.save(model_path)
        print(f"\n模型已保存到: {model_path}")
        
        # 保存标准化器
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"标准化器已保存到: {scaler_path}")
        
        # 保存配置
        config = {
            'sequence_length': self.sequence_length,
            'prediction_horizon': self.prediction_horizon,
            'model_path': model_path,
            'scaler_path': scaler_path,
            'trained_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        config_path = os.path.join(model_dir, 'config.pkl')
        with open(config_path, 'wb') as f:
            pickle.dump(config, f)
        print(f"配置已保存到: {config_path}")
    
    def load_model(self, model_dir=MODEL_DIR):
        """加载已训练的模型"""
        # 加载配置
        config_path = os.path.join(model_dir, 'config.pkl')
        with open(config_path, 'rb') as f:
            config = pickle.load(f)
        
        self.sequence_length = config['sequence_length']
        self.prediction_horizon = config['prediction_horizon']
        
        # 加载模型
        model_path = config['model_path']
        self.model = keras.models.load_model(model_path)
        print(f"模型已加载: {model_path}")
        
        # 加载标准化器
        scaler_path = config['scaler_path']
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        print(f"标准化器已加载: {scaler_path}")
        
        return self.model
    
    def predict(self, temperature_sequence):
        """
        预测热失控风险
        
        参数:
            temperature_sequence: 温度序列数组，长度应为 sequence_length
        
        返回:
            风险概率 (0-1之间)
        """
        if len(temperature_sequence) != self.sequence_length:
            raise ValueError(f"输入序列长度必须为 {self.sequence_length}")
        
        # 标准化
        sequence = np.array(temperature_sequence).reshape(-1, 1)
        sequence_scaled = self.scaler.transform(sequence)
        sequence_scaled = sequence_scaled.reshape(1, self.sequence_length, 1)
        
        # 预测
        risk_prob = self.model.predict(sequence_scaled, verbose=0)[0][0]
        
        return risk_prob


def main():
    """主函数"""
    print("=" * 60)
    print("电池热失控预测模型训练")
    print("=" * 60)
    
    # 创建预测器
    predictor = BatteryThermalRunawayPredictor(
        sequence_length=SEQUENCE_LENGTH,
        prediction_horizon=PREDICTION_HORIZON
    )
    
    # 加载数据
    X, y = predictor.load_data(DATA_DIR)
    
    # 预处理数据
    X_train, X_test, y_train, y_test = predictor.preprocess_data(X, y)
    
    # 构建模型
    predictor.build_model(input_shape=(SEQUENCE_LENGTH, 1))
    
    # 训练模型
    predictor.train(X_train, y_train, X_test, y_test)
    
    # 评估模型
    predictor.evaluate(X_test, y_test)
    
    # 绘制训练历史
    predictor.plot_training_history()
    
    # 保存模型
    predictor.save_model()
    
    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)
    
    # 测试预测
    print("\n测试实时预测功能...")
    test_sequence = X_test[0].flatten()
    risk = predictor.predict(test_sequence)
    print(f"测试样本热失控风险: {risk:.2%}")
    print(f"实际标签: {'热失控' if y_test[0] == 1 else '正常'}")


if __name__ == "__main__":
    main()
