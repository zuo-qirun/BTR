"""
电池热失控预测 - LSTM 模型（预测未来60秒）
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
DATA_DIR = "data"
MODEL_DIR = "models_lstm_60s"               # 新模型保存目录
SEQUENCE_LENGTH = 60                         # 过去30秒（60个点）
PREDICTION_HORIZON = 120                      # 未来60秒（120个点）← 修改点1
BATCH_SIZE = 16
EPOCHS = 100
INITIAL_LEARNING_RATE = 0.001
TIME_INTERVAL = 0.5                           # 重采样间隔（秒）
MAX_GAP = 70.0
CLASS_WEIGHT = {0: 1.0, 1: 200.0}             # 正类权重

# 特征选项（可保持原样）
USE_DIFF_FEATURES = True
USE_SMOOTHING = True
SMOOTH_WINDOW = 11
SMOOTH_POLYORDER = 3

class BatteryThermalLSTM:
    def __init__(self, sequence_length=SEQUENCE_LENGTH, prediction_horizon=PREDICTION_HORIZON):
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.model = None
        self.scaler = StandardScaler()
        self.history = None
        self.best_threshold = 0.3

    def load_data(self, data_dir):
        """加载数据，生成标签时使用新的预测步长"""
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

                # 重采样到 TIME_INTERVAL
                start_time, end_time = time_values[0], time_values[-1]
                uniform_times = np.arange(start_time, end_time, TIME_INTERVAL)
                uniform_temps = np.interp(uniform_times, time_values, temp_values)

                # 异常文件裁剪
                if is_exception:
                    baseline = np.percentile(uniform_temps[:min(1000, len(uniform_temps))], 50)
                    threshold = baseline + 5.0
                    rising_idx = np.where(uniform_temps > threshold)[0]
                    if len(rising_idx) > 0:
                        start_idx = max(0, rising_idx[0] - 200)
                        uniform_temps = uniform_temps[start_idx:]

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
                        # 未来窗口内温度上升超过5°C → 正样本
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
        """添加差分特征（与之前相同）"""
        n_samples, seq_len = X.shape
        if USE_SMOOTHING and seq_len >= SMOOTH_WINDOW:
            deriv1 = savgol_filter(X, window_length=SMOOTH_WINDOW, polyorder=SMOOTH_POLYORDER, deriv=1, axis=1)
            deriv2 = savgol_filter(X, window_length=SMOOTH_WINDOW, polyorder=SMOOTH_POLYORDER, deriv=2, axis=1)
        else:
            diff1 = np.diff(X, n=1, axis=1)
            deriv1 = np.concatenate([np.zeros((n_samples, 1)), diff1], axis=1)
            diff2 = np.diff(X, n=2, axis=1)
            deriv2 = np.concatenate([np.zeros((n_samples, 2)), diff2], axis=1)
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

        # 标准化
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
        """构建LSTM模型（与之前相同）"""
        model = keras.Sequential([
            layers.LSTM(128, return_sequences=True, input_shape=input_shape),
            layers.Dropout(0.3),
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.3),
            layers.LSTM(32),
            layers.Dropout(0.3),
            layers.Dense(16, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1, activation='sigmoid')
        ])
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=INITIAL_LEARNING_RATE),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )
        return model

    def train(self, X_train, y_train, X_test, y_test):
        """训练模型"""
        print("\n开始训练 LSTM...")
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

    def find_best_threshold(self, X_val, y_val, target_recall=0.95):
        """在验证集上寻找最佳阈值（以召回率为导向）"""
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
            recalls = [recall_score(y_val, (y_pred_prob > t).astype(int), zero_division=0) for t in thresholds]
            best_thresh = thresholds[np.argmax(recalls)]
        self.best_threshold = best_thresh
        print(f"最佳阈值: {best_thresh:.3f}")
        return best_thresh

    def evaluate(self, X_test, y_test):
        """评估模型并输出报告"""
        y_pred_prob = self.model.predict(X_test, verbose=0).flatten()
        y_pred = (y_pred_prob > 0.5).astype(int)

        print("\n" + "="*60)
        print("Evaluation Report - LSTM (60s prediction)")
        print("="*60)
        print(f"Default threshold (0.5):")
        print(f"  Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
        print(f"  Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
        print(f"  Recall:    {recall_score(y_test, y_pred, zero_division=0):.4f}")
        print(f"  F1:        {f1_score(y_test, y_pred, zero_division=0):.4f}")
        print(f"  AUC-ROC:   {roc_auc_score(y_test, y_pred_prob):.4f}")
        print(f"\nConfusion matrix (threshold=0.5):")
        print(confusion_matrix(y_test, y_pred))

        y_pred_opt = (y_pred_prob > self.best_threshold).astype(int)
        print(f"\nOptimized threshold ({self.best_threshold:.3f}):")
        print(f"  Precision: {precision_score(y_test, y_pred_opt, zero_division=0):.4f}")
        print(f"  Recall:    {recall_score(y_test, y_pred_opt, zero_division=0):.4f}")
        print(f"  F1:        {f1_score(y_test, y_pred_opt, zero_division=0):.4f}")
        print(f"\nConfusion matrix (optimized):")
        print(confusion_matrix(y_test, y_pred_opt))
        print("\nClassification Report (optimized):")
        print(classification_report(y_test, y_pred_opt, target_names=['正常', '热失控'], zero_division=0))

        # 不同阈值性能
        print("\nPerformance at different thresholds:")
        for thresh in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
            y_pred_thresh = (y_pred_prob > thresh).astype(int)
            prec = precision_score(y_test, y_pred_thresh, zero_division=0)
            rec = recall_score(y_test, y_pred_thresh, zero_division=0)
            print(f"  threshold={thresh:.1f}: Precision={prec:.4f}, Recall={rec:.4f}")

        # 保存报告
        os.makedirs(MODEL_DIR, exist_ok=True)
        report_path = os.path.join(MODEL_DIR, "evaluation_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("Evaluation Report - LSTM (60s prediction)\n")
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
        """保存模型"""
        os.makedirs(MODEL_DIR, exist_ok=True)
        self.model.save(os.path.join(MODEL_DIR, 'lstm_60s_model.h5'))
        with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'wb') as f:
            pickle.dump(self.scaler, f)
        config = {
            'sequence_length': self.sequence_length,
            'prediction_horizon': self.prediction_horizon,
            'best_threshold': self.best_threshold,
            'use_diff_features': USE_DIFF_FEATURES,
            'use_smoothing': USE_SMOOTHING,
            'trained_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        with open(os.path.join(MODEL_DIR, 'config.pkl'), 'wb') as f:
            pickle.dump(config, f)
        print(f"\n模型已保存到: {MODEL_DIR}")

    def load_model(self, model_dir=MODEL_DIR):
        """加载模型"""
        with open(os.path.join(model_dir, 'config.pkl'), 'rb') as f:
            config = pickle.load(f)
        self.sequence_length = config['sequence_length']
        self.prediction_horizon = config['prediction_horizon']
        self.best_threshold = config.get('best_threshold', 0.3)
        self.model = keras.models.load_model(os.path.join(model_dir, 'lstm_60s_model.h5'))
        with open(os.path.join(model_dir, 'scaler.pkl'), 'rb') as f:
            self.scaler = pickle.load(f)
        print(f"模型加载完成，最佳阈值: {self.best_threshold}")
        return self.model

    def predict(self, temperature_sequence):
        """实时预测"""
        if len(temperature_sequence) != self.sequence_length:
            raise ValueError(f"输入序列长度必须为 {self.sequence_length}")
        temp_seq = np.array(temperature_sequence).reshape(1, -1)
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

        n_samples, seq_len, n_features = X_multi.shape
        X_flat = X_multi.reshape(-1, n_features)
        X_scaled_flat = self.scaler.transform(X_flat)
        X_scaled = X_scaled_flat.reshape(n_samples, seq_len, n_features)

        prob = self.model.predict(X_scaled, verbose=0)[0, 0]
        pred = 1 if prob > self.best_threshold else 0
        return prob, pred


def main():
    print("="*60)
    print("电池热失控预测 - LSTM (预测未来60秒)")
    print("="*60)

    predictor = BatteryThermalLSTM()

    X, y = predictor.load_data(DATA_DIR)
    X_train, X_test, y_train, y_test = predictor.preprocess_data(X, y)

    input_shape = (SEQUENCE_LENGTH, X_train.shape[2])
    predictor.model = predictor.build_model(input_shape)
    print(predictor.model.summary())

    predictor.train(X_train, y_train, X_test, y_test)

    # 从训练集中划分验证集找阈值
    X_train_part, X_val, y_train_part, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    predictor.find_best_threshold(X_val, y_val)

    predictor.evaluate(X_test, y_test)
    predictor.plot_training_history()
    predictor.save_model()

    print("\n训练完成！")

if __name__ == "__main__":
    main()