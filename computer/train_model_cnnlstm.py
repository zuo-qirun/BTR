"""
电池热失控预测 - CNN + LSTM 混合模型
使用卷积层提取局部特征，LSTM 捕捉时序依赖
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_score, recall_score, f1_score,
                             roc_auc_score, accuracy_score)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle
from datetime import datetime
from scipy.signal import savgol_filter

# ====== 配置参数 ======
DATA_DIR = "data"                          # 数据目录
MODEL_DIR = "models_cnn_lstm"               # 模型保存目录
SEQUENCE_LENGTH = 60                        # 序列长度（60个点，对应30秒）
PREDICTION_HORIZON = 40                      # 预测未来步长（用于标签生成）
BATCH_SIZE = 32
EPOCHS = 100
INITIAL_LEARNING_RATE = 0.001
TIME_INTERVAL = 0.5                          # 重采样间隔（秒）
MAX_GAP = 70.0                               # 最大允许原始数据间隔
CLASS_WEIGHT = {0: 1.0, 1: 200.0}            # 正类权重

# 特征选项
USE_DIFF_FEATURES = True                      # 是否添加差分特征（温度的一阶/二阶导）
USE_SMOOTHING = True                          # 是否对温度平滑后再计算差分
SMOOTH_WINDOW = 11                             # 平滑窗口
SMOOTH_POLYORDER = 3                           # 多项式阶数

# CNN + LSTM 超参数
CONV_FILTERS = 64                              # 卷积核数量
CONV_KERNEL_SIZE = 5                           # 卷积核大小
POOL_SIZE = 2                                   # 池化大小
LSTM_UNITS = 64                                 # LSTM 单元数
DROPOUT = 0.3                                   # Dropout 比率

class BatteryThermalCnnLstm:
    """基于 CNN + LSTM 的热失控预测器"""
    
    def __init__(self, sequence_length=SEQUENCE_LENGTH, prediction_horizon=PREDICTION_HORIZON):
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.model = None
        self.scaler = StandardScaler()
        self.history = None
        self.best_threshold = 0.3
        
    def load_data(self, data_dir):
        """加载所有 CSV 文件，生成时间序列样本（与之前版本一致）"""
        print("正在加载数据...")
        all_sequences = []
        all_labels = []
        csv_files = glob.glob(os.path.join(data_dir, "*.csv"))

        for csv_file in csv_files:
            try:
                is_exception = "_exception" in os.path.basename(csv_file)
                df = pd.read_csv(csv_file)
                df = df.sort_values('Time').reset_index(drop=True)
                time_values = df['Time'].values
                temp_values = df['Temperature'].values

                # 重采样到统一时间间隔
                start_time, end_time = time_values[0], time_values[-1]
                uniform_times = np.arange(start_time, end_time, TIME_INTERVAL)
                uniform_temps = np.interp(uniform_times, time_values, temp_values)

                # 异常文件裁剪：保留温度上升段
                if is_exception:
                    baseline = np.percentile(uniform_temps[:min(1000, len(uniform_temps))], 50)
                    threshold = baseline + 5.0
                    rising_idx = np.where(uniform_temps > threshold)[0]
                    if len(rising_idx) > 0:
                        start_idx = max(0, rising_idx[0] - 200)
                        uniform_temps = uniform_temps[start_idx:]
                        uniform_times = uniform_times[start_idx:]

                # 生成滑动窗口样本
                max_sequences = len(uniform_temps) - self.sequence_length - self.prediction_horizon
                if max_sequences <= 0:
                    continue

                # 对正常文件随机采样
                if is_exception:
                    sample_indices = range(max_sequences)
                else:
                    max_normal = 2000
                    num_samples = min(max_normal, max_sequences)
                    sample_indices = np.random.choice(max_sequences, num_samples, replace=False) if max_sequences > max_normal else range(max_sequences)

                for i in sample_indices:
                    seq = uniform_temps[i:i+self.sequence_length]
                    future = uniform_temps[i+self.sequence_length:i+self.sequence_length+self.prediction_horizon]
                    if is_exception:
                        increase = np.max(future) - seq[-1]
                        label = 1 if increase > 5 else 0
                    else:
                        label = 0
                    all_sequences.append(seq)
                    all_labels.append(label)

                print(f"已加载: {os.path.basename(csv_file)} -> {len(sample_indices)} 序列")
            except Exception as e:
                print(f"加载文件 {csv_file} 出错: {e}")

        X = np.array(all_sequences)
        y = np.array(all_labels)
        print(f"\n总样本数: {len(X)}，正常: {np.sum(y==0)}，异常: {np.sum(y==1)}")
        return X, y

    def add_derivative_features(self, X):
        """添加一阶和二阶导数特征（可选平滑）"""
        n_samples, seq_len = X.shape
        if USE_SMOOTHING and seq_len >= SMOOTH_WINDOW:
            # 平滑后的一阶导
            deriv1 = savgol_filter(X, window_length=SMOOTH_WINDOW, polyorder=SMOOTH_POLYORDER, deriv=1, axis=1)
            # 平滑后的二阶导
            deriv2 = savgol_filter(X, window_length=SMOOTH_WINDOW, polyorder=SMOOTH_POLYORDER, deriv=2, axis=1)
        else:
            # 简单差分（非平滑）
            diff1 = np.diff(X, n=1, axis=1)
            deriv1 = np.concatenate([np.zeros((n_samples, 1)), diff1], axis=1)
            diff2 = np.diff(X, n=2, axis=1)
            deriv2 = np.concatenate([np.zeros((n_samples, 2)), diff2], axis=1)

        # 堆叠特征：温度, 一阶导, 二阶导 -> (n_samples, seq_len, 3)
        X_multi = np.stack([X, deriv1, deriv2], axis=2)
        print(f"特征维度: {X_multi.shape[2]} (温度, 一阶导, 二阶导)")
        return X_multi

    def preprocess_data(self, X, y):
        """预处理：添加特征、标准化、划分数据集"""
        print("\n预处理数据...")
        if USE_DIFF_FEATURES:
            X_multi = self.add_derivative_features(X)
        else:
            X_multi = X.reshape(X.shape[0], X.shape[1], 1)

        # 标准化（每个特征通道独立）
        n_samples, seq_len, n_features = X_multi.shape
        X_flat = X_multi.reshape(-1, n_features)
        X_scaled_flat = self.scaler.fit_transform(X_flat)
        X_scaled = X_scaled_flat.reshape(n_samples, seq_len, n_features)

        # 划分训练/测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"训练集: {len(X_train)}，测试集: {len(X_test)}")
        print(f"训练集异常: {np.sum(y_train)}，测试集异常: {np.sum(y_test)}")
        return X_train, X_test, y_train, y_test

    def build_model(self, input_shape):
        """构建 CNN + LSTM 模型"""
        inputs = layers.Input(shape=input_shape)

        # CNN 部分：提取局部特征
        x = layers.Conv1D(filters=CONV_FILTERS, kernel_size=CONV_KERNEL_SIZE, 
                          activation='relu', padding='same')(inputs)
        x = layers.MaxPooling1D(pool_size=POOL_SIZE)(x)
        x = layers.Dropout(DROPOUT)(x)

        # 可选的第二层卷积
        x = layers.Conv1D(filters=CONV_FILTERS*2, kernel_size=CONV_KERNEL_SIZE,
                          activation='relu', padding='same')(x)
        x = layers.MaxPooling1D(pool_size=POOL_SIZE)(x)
        x = layers.Dropout(DROPOUT)(x)

        # LSTM 部分：捕捉时序依赖
        x = layers.LSTM(LSTM_UNITS, return_sequences=False)(x)
        x = layers.Dropout(DROPOUT)(x)

        # 输出层
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(DROPOUT)(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)

        model = keras.Model(inputs, outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=INITIAL_LEARNING_RATE),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )
        return model

    def train(self, X_train, y_train, X_test, y_test):
        """训练模型"""
        print("\n开始训练 CNN + LSTM...")
        
        # 确保模型已编译
        if not hasattr(self.model, 'optimizer') or self.model.optimizer is None:
            print("模型未编译，正在重新编译...")
            self.model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=INITIAL_LEARNING_RATE),
                loss='binary_crossentropy',
                metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
            )
        
        callbacks = [
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)
        ]

        self.history = self.model.fit(
            X_train, y_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            class_weight=CLASS_WEIGHT,
            verbose=1
        )
        print("训练完成！")

    def find_best_threshold(self, X_val, y_val, target_recall=0.95):
        """在验证集上寻找最佳阈值"""
        y_pred_prob = self.model.predict(X_val, verbose=0).flatten()
        thresholds = np.arange(0.05, 0.8, 0.01)
        best_thresh = 0.3
        best_precision = 0.0

        for thresh in thresholds:
            y_pred = (y_pred_prob > thresh).astype(int)
            recall = recall_score(y_val, y_pred, zero_division=0)
            if recall >= target_recall:
                precision = precision_score(y_val, y_pred, zero_division=0)
                if precision > best_precision:
                    best_precision = precision
                    best_thresh = thresh

        if best_thresh == 0.3:
            # 未达到目标，选择召回率最高的阈值
            recalls = [recall_score(y_val, (y_pred_prob > t).astype(int), zero_division=0) for t in thresholds]
            best_thresh = thresholds[np.argmax(recalls)]

        self.best_threshold = best_thresh
        print(f"最佳阈值: {best_thresh:.3f}")
        return best_thresh

    def evaluate(self, X_test, y_test):
        """评估模型，输出详细报告"""
        y_pred_prob = self.model.predict(X_test, verbose=0).flatten()
        y_pred = (y_pred_prob > 0.5).astype(int)

        # 默认阈值结果
        print("\n" + "="*60)
        print("Evaluation Report - CNN + LSTM")
        print("="*60)
        print(f"Default threshold (0.5):")
        print(f"  Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
        print(f"  Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
        print(f"  Recall:    {recall_score(y_test, y_pred, zero_division=0):.4f}")
        print(f"  F1:        {f1_score(y_test, y_pred, zero_division=0):.4f}")
        print(f"  AUC-ROC:   {roc_auc_score(y_test, y_pred_prob):.4f}")
        print(f"\nConfusion matrix (threshold=0.5):")
        print(confusion_matrix(y_test, y_pred))

        # 优化阈值结果
        y_pred_opt = (y_pred_prob > self.best_threshold).astype(int)
        print(f"\nOptimized threshold ({self.best_threshold:.3f}):")
        print(f"  Precision: {precision_score(y_test, y_pred_opt, zero_division=0):.4f}")
        print(f"  Recall:    {recall_score(y_test, y_pred_opt, zero_division=0):.4f}")
        print(f"  F1:        {f1_score(y_test, y_pred_opt, zero_division=0):.4f}")
        print(f"\nConfusion matrix (optimized):")
        print(confusion_matrix(y_test, y_pred_opt))
        print("\nClassification Report (optimized):")
        print(classification_report(y_test, y_pred_opt, target_names=['正常', '热失控'], zero_division=0))

        # 不同阈值下的性能
        print("\nPerformance at different thresholds:")
        for thresh in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
            y_pred_thresh = (y_pred_prob > thresh).astype(int)
            prec = precision_score(y_test, y_pred_thresh, zero_division=0)
            rec = recall_score(y_test, y_pred_thresh, zero_division=0)
            print(f"  threshold={thresh:.1f}: Precision={prec:.4f}, Recall={rec:.4f}")

        # 保存报告到文件
        os.makedirs(MODEL_DIR, exist_ok=True)
        report_path = os.path.join(MODEL_DIR, "evaluation_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("Evaluation Report - CNN + LSTM\n")
            f.write(f"Best threshold: {self.best_threshold:.3f}\n")
            f.write(f"Confusion matrix (optimized):\n{confusion_matrix(y_test, y_pred_opt)}\n")
            f.write(classification_report(y_test, y_pred_opt, target_names=['正常', '热失控'], zero_division=0))
        print(f"\n评估报告已保存到: {report_path}")

    def plot_training_history(self):
        """绘制训练曲线"""
        if self.history is None:
            return
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes[0,0].plot(self.history.history['loss'], label='Train Loss')
        axes[0,0].plot(self.history.history['val_loss'], label='Val Loss')
        axes[0,0].set_title('Loss')
        axes[0,0].legend()
        axes[0,1].plot(self.history.history['accuracy'], label='Train Acc')
        axes[0,1].plot(self.history.history['val_accuracy'], label='Val Acc')
        axes[0,1].set_title('Accuracy')
        axes[0,1].legend()
        axes[1,0].plot(self.history.history['precision'], label='Train Precision')
        axes[1,0].plot(self.history.history['val_precision'], label='Val Precision')
        axes[1,0].set_title('Precision')
        axes[1,0].legend()
        axes[1,1].plot(self.history.history['recall'], label='Train Recall')
        axes[1,1].plot(self.history.history['val_recall'], label='Val Recall')
        axes[1,1].set_title('Recall')
        axes[1,1].legend()
        plt.tight_layout()
        plt.savefig(os.path.join(MODEL_DIR, 'training_history.png'))
        plt.show()

    def save_model(self):
        """保存模型、标准化器和配置"""
        os.makedirs(MODEL_DIR, exist_ok=True)
        self.model.save(os.path.join(MODEL_DIR, 'cnn_lstm_model.h5'))
        with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'wb') as f:
            pickle.dump(self.scaler, f)

        config = {
            'sequence_length': self.sequence_length,
            'prediction_horizon': self.prediction_horizon,
            'best_threshold': self.best_threshold,
            'use_diff_features': USE_DIFF_FEATURES,
            'use_smoothing': USE_SMOOTHING,
            'smooth_window': SMOOTH_WINDOW if USE_SMOOTHING else None,
            'smooth_polyorder': SMOOTH_POLYORDER if USE_SMOOTHING else None,
            'conv_filters': CONV_FILTERS,
            'lstm_units': LSTM_UNITS,
            'trained_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        with open(os.path.join(MODEL_DIR, 'config.pkl'), 'wb') as f:
            pickle.dump(config, f)
        print(f"\n模型已保存到: {MODEL_DIR}")

    def load_model(self, model_dir=MODEL_DIR):
        """加载已训练模型"""
        with open(os.path.join(model_dir, 'config.pkl'), 'rb') as f:
            config = pickle.load(f)
        self.sequence_length = config['sequence_length']
        self.prediction_horizon = config['prediction_horizon']
        self.best_threshold = config.get('best_threshold', 0.3)
        self.model = keras.models.load_model(os.path.join(model_dir, 'cnn_lstm_model.h5'))
        with open(os.path.join(model_dir, 'scaler.pkl'), 'rb') as f:
            self.scaler = pickle.load(f)
        print(f"模型加载完成，最佳阈值: {self.best_threshold}")
        return self.model

    def predict(self, temperature_sequence):
        """对新温度序列进行预测"""
        if len(temperature_sequence) != self.sequence_length:
            raise ValueError(f"输入序列长度必须为 {self.sequence_length}")

        # 转换为numpy并添加batch维度
        temp_seq = np.array(temperature_sequence).reshape(1, -1)

        # 添加差分特征（如果训练时使用）
        if USE_DIFF_FEATURES:
            if USE_SMOOTHING and self.sequence_length >= SMOOTH_WINDOW:
                deriv1 = savgol_filter(temp_seq, window_length=SMOOTH_WINDOW, polyorder=SMOOTH_POLYORDER, deriv=1, axis=1)
                deriv2 = savgol_filter(temp_seq, window_length=SMOOTH_WINDOW, polyorder=SMOOTH_POLYORDER, deriv=2, axis=1)
            else:
                diff1 = np.diff(temp_seq, n=1, axis=1)
                deriv1 = np.concatenate([np.zeros((1, 1)), diff1], axis=1)
                diff2 = np.diff(temp_seq, n=2, axis=1)
                deriv2 = np.concatenate([np.zeros((1, 2)), diff2], axis=1)
            X_multi = np.stack([temp_seq, deriv1, deriv2], axis=2)
        else:
            X_multi = temp_seq.reshape(1, self.sequence_length, 1)

        # 标准化
        n_samples, seq_len, n_features = X_multi.shape
        X_flat = X_multi.reshape(-1, n_features)
        X_scaled_flat = self.scaler.transform(X_flat)
        X_scaled = X_scaled_flat.reshape(n_samples, seq_len, n_features)

        # 预测
        prob = self.model.predict(X_scaled, verbose=0)[0, 0]
        pred = 1 if prob > self.best_threshold else 0
        return prob, pred


def main():
    print("="*60)
    print("电池热失控预测 - CNN + LSTM 模型")
    print("="*60)

    # 初始化
    predictor = BatteryThermalCnnLstm()

    # 加载数据
    X, y = predictor.load_data(DATA_DIR)

    # 预处理
    X_train, X_test, y_train, y_test = predictor.preprocess_data(X, y)

    # 构建模型
    input_shape = (SEQUENCE_LENGTH, X_train.shape[2])
    predictor.model = predictor.build_model(input_shape)
    print(predictor.model.summary())

    # 训练
    predictor.train(X_train, y_train, X_test, y_test)

    # 从训练集中划分验证集寻找最佳阈值
    X_train_part, X_val, y_train_part, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    predictor.find_best_threshold(X_val, y_val)

    # 评估
    predictor.evaluate(X_test, y_test)

    # 绘制训练历史
    predictor.plot_training_history()

    # 保存模型
    predictor.save_model()

    print("\n训练完成！")


if __name__ == "__main__":
    main()