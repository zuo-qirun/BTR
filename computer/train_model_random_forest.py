"""
电池热失控预测 - 随机森林模型（带详细评估报告）
基于温度时间序列提取统计特征，使用随机森林分类
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix, 
                             precision_score, recall_score, f1_score, roc_auc_score,
                             precision_recall_curve, auc, accuracy_score)
from sklearn.preprocessing import StandardScaler
import pickle
import joblib
from datetime import datetime
from scipy.signal import savgol_filter

# ====== 配置参数 ======
DATA_DIR = "data"                         # 数据目录
MODEL_DIR = "models_random_forest"         # 模型保存目录
SEQUENCE_LENGTH = 60                       # 序列长度（与LSTM一致）
PREDICTION_HORIZON = 40                     # 预测未来步长
TIME_INTERVAL = 0.5                         # 重采样间隔
MAX_GAP = 70.0                              # 最大允许间隔
RANDOM_STATE = 42

# 特征提取参数
USE_SMOOTHING = True                        # 是否对温度进行平滑后再提取导数特征
SMOOTH_WINDOW = 11                           # 平滑窗口（奇数）
SMOOTH_POLYORDER = 3                         # 多项式阶数

def load_data(data_dir):
    """加载所有CSV文件，返回温度序列列表和标签"""
    print("正在加载数据...")
    all_sequences = []
    all_labels = []
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))

    for csv_file in csv_files:
        try:
            is_exception = "_exception" in os.path.basename(csv_file)
            label = 1 if is_exception else 0

            df = pd.read_csv(csv_file)
            df = df.sort_values('Time').reset_index(drop=True)
            time_values = df['Time'].values
            temp_values = df['Temperature'].values

            # 重采样到 TIME_INTERVAL
            start_time, end_time = time_values[0], time_values[-1]
            uniform_times = np.arange(start_time, end_time, TIME_INTERVAL)
            uniform_temps = np.interp(uniform_times, time_values, temp_values)

            # 异常文件裁剪（保留上升段）
            if is_exception:
                baseline_temp = np.percentile(uniform_temps[:min(1000, len(uniform_temps))], 50)
                threshold_temp = baseline_temp + 5.0
                rising_indices = np.where(uniform_temps > threshold_temp)[0]
                if len(rising_indices) > 0:
                    start_idx = max(0, rising_indices[0] - 200)
                    uniform_temps = uniform_temps[start_idx:]
                    uniform_times = uniform_times[start_idx:]

            # 生成滑动窗口样本
            max_sequences = len(uniform_temps) - SEQUENCE_LENGTH - PREDICTION_HORIZON
            if max_sequences <= 0:
                print(f"  ⚠️  {os.path.basename(csv_file)} 数据点不足，跳过")
                continue

            # 对正常文件随机采样
            if is_exception:
                sample_indices = range(max_sequences)
            else:
                max_normal_samples = 2000
                num_samples = min(max_normal_samples, max_sequences)
                sample_indices = np.random.choice(max_sequences, num_samples, replace=False) if max_sequences > max_normal_samples else range(max_sequences)

            for i in sample_indices:
                sequence = uniform_temps[i:i + SEQUENCE_LENGTH]
                future_temps = uniform_temps[i + SEQUENCE_LENGTH:i + SEQUENCE_LENGTH + PREDICTION_HORIZON]

                # 标签：如果未来温度比窗口末尾上升超过5°C，则视为正样本
                if is_exception:
                    temp_increase = np.max(future_temps) - sequence[-1]
                    lbl = 1 if temp_increase > 5 else 0
                else:
                    lbl = 0

                all_sequences.append(sequence)
                all_labels.append(lbl)

            print(f"已加载: {os.path.basename(csv_file)} - 生成 {len(sample_indices)} 个序列")
        except Exception as e:
            print(f"加载文件 {csv_file} 时出错: {e}")

    X = np.array(all_sequences)
    y = np.array(all_labels)
    print(f"\n总样本数: {len(X)}，正常: {np.sum(y==0)}，异常: {np.sum(y==1)}")
    return X, y

def extract_features_from_sequence(seq):
    """
    从单个温度序列（60点）提取特征向量
    返回一个包含多种统计特征的列表
    """
    features = []
    # 1. 基本统计量
    features.append(np.mean(seq))
    features.append(np.std(seq))
    features.append(np.min(seq))
    features.append(np.max(seq))
    features.append(np.ptp(seq))          # 峰峰值
    features.append(np.median(seq))
    features.append(np.percentile(seq, 25))
    features.append(np.percentile(seq, 75))
    features.append(np.percentile(seq, 90))

    # 2. 趋势特征：线性拟合的斜率和截距
    x = np.arange(len(seq))
    slope, intercept = np.polyfit(x, seq, 1)
    features.append(slope)
    features.append(intercept)

    # 3. 差分特征（一阶差分）
    diff1 = np.diff(seq)
    features.append(np.mean(diff1))
    features.append(np.std(diff1))
    features.append(np.max(diff1))
    features.append(np.min(diff1))
    features.append(np.sum(diff1 > 0) / len(diff1))   # 正向变化比例
    features.append(np.sum(diff1 < 0) / len(diff1))   # 负向变化比例

    # 4. 二阶差分特征（加速度）
    diff2 = np.diff(diff1)
    if len(diff2) > 0:
        features.append(np.mean(diff2))
        features.append(np.std(diff2))
        features.append(np.max(diff2))
        features.append(np.min(diff2))
    else:
        features.extend([0, 0, 0, 0])

    # 5. 与起始温度的相对变化
    features.append(seq[-1] - seq[0])
    features.append((seq[-1] - seq[0]) / len(seq))   # 平均每步变化

    # 6. 最后10个点的趋势
    last10 = seq[-10:]
    if len(last10) >= 2:
        slope_last10 = np.polyfit(np.arange(len(last10)), last10, 1)[0]
    else:
        slope_last10 = 0
    features.append(slope_last10)

    # 7. 如果启用平滑，则添加平滑后的一阶/二阶导数的统计量
    if USE_SMOOTHING:
        # 对序列进行平滑
        if len(seq) >= SMOOTH_WINDOW:
            smoothed = savgol_filter(seq, window_length=SMOOTH_WINDOW, polyorder=SMOOTH_POLYORDER)
            # 平滑后的一阶导
            deriv1 = savgol_filter(seq, window_length=SMOOTH_WINDOW, polyorder=SMOOTH_POLYORDER, deriv=1)
            # 平滑后的二阶导
            deriv2 = savgol_filter(seq, window_length=SMOOTH_WINDOW, polyorder=SMOOTH_POLYORDER, deriv=2)

            # 添加导数的统计量
            features.append(np.mean(deriv1))
            features.append(np.std(deriv1))
            features.append(np.max(deriv1))
            features.append(np.min(deriv1))
            features.append(np.mean(deriv2))
            features.append(np.std(deriv2))
            features.append(np.max(deriv2))
            features.append(np.min(deriv2))
        else:
            # 窗口不足则填充0
            features.extend([0]*8)

    return features

def build_feature_matrix(X):
    """为所有序列提取特征，返回特征矩阵"""
    print("提取特征...")
    feature_list = []
    for i, seq in enumerate(X):
        if i % 5000 == 0:
            print(f"  已处理 {i}/{len(X)} 个序列")
        feat = extract_features_from_sequence(seq)
        feature_list.append(feat)
    return np.array(feature_list)

def print_evaluation_report(y_true, y_pred, y_prob, best_threshold, model_name="RandomForest"):
    """打印结构化的评估报告，并保存到文件"""
    report_lines = []
    report_lines.append("="*60)
    report_lines.append(f"Evaluation Report - {model_name}")
    report_lines.append("="*60)

    # 基础指标（使用默认阈值0.5）
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc_roc = roc_auc_score(y_true, y_prob)

    report_lines.append(f"Default threshold (0.5):")
    report_lines.append(f"  Accuracy:  {accuracy:.4f}")
    report_lines.append(f"  Precision: {precision:.4f}")
    report_lines.append(f"  Recall:    {recall:.4f}")
    report_lines.append(f"  F1:        {f1:.4f}")
    report_lines.append(f"  AUC-ROC:   {auc_roc:.4f}")

    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    report_lines.append(f"\nConfusion matrix (threshold=0.5):")
    report_lines.append(f"{cm}")

    # 优化阈值下的结果
    y_pred_opt = (y_prob > best_threshold).astype(int)
    precision_opt = precision_score(y_true, y_pred_opt, zero_division=0)
    recall_opt = recall_score(y_true, y_pred_opt, zero_division=0)
    f1_opt = f1_score(y_true, y_pred_opt, zero_division=0)
    cm_opt = confusion_matrix(y_true, y_pred_opt)

    report_lines.append(f"\nOptimized threshold ({best_threshold:.3f}):")
    report_lines.append(f"  Precision: {precision_opt:.4f}")
    report_lines.append(f"  Recall:    {recall_opt:.4f}")
    report_lines.append(f"  F1:        {f1_opt:.4f}")
    report_lines.append(f"\nConfusion matrix (optimized):")
    report_lines.append(f"{cm_opt}")

    # 分类报告
    report_lines.append(f"\nClassification Report (optimized):")
    report_lines.append(classification_report(y_true, y_pred_opt, target_names=['正常', '热失控'], zero_division=0))

    # 不同阈值下的性能
    thresholds = np.arange(0.1, 0.8, 0.1)
    report_lines.append(f"\nPerformance at different thresholds:")
    for thresh in thresholds:
        y_pred_thresh = (y_prob > thresh).astype(int)
        prec = precision_score(y_true, y_pred_thresh, zero_division=0)
        rec = recall_score(y_true, y_pred_thresh, zero_division=0)
        report_lines.append(f"  threshold={thresh:.1f}: Precision={prec:.4f}, Recall={rec:.4f}")

    # 打印到控制台
    for line in report_lines:
        print(line)

    # 保存到文件
    report_path = os.path.join(MODEL_DIR, "evaluation_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"\n评估报告已保存到: {report_path}")

def main():
    print("="*60)
    print("电池热失控预测 - 随机森林模型（带详细评估报告）")
    print("="*60)

    # 1. 加载原始序列数据
    X_seq, y = load_data(DATA_DIR)

    # 2. 特征工程
    X_feat = build_feature_matrix(X_seq)
    print(f"特征矩阵形状: {X_feat.shape}")

    # 3. 标准化（随机森林通常不需要，但为了后续可解释性可以保留）
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_feat)

    # 4. 划分训练集和测试集（保持类别比例）
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"训练集: {len(X_train)}，测试集: {len(X_test)}")

    # 5. 训练随机森林（处理类别不平衡）
    print("\n训练随机森林...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1
    )
    rf.fit(X_train, y_train)

    # 6. 交叉验证（可选）
    cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='recall')
    print(f"5折交叉验证召回率: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # 7. 测试集预测
    y_pred_prob = rf.predict_proba(X_test)[:, 1]
    y_pred = rf.predict(X_test)

    # 8. 寻找最佳阈值（以召回率为导向）
    from sklearn.metrics import precision_recall_curve
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_prob)
    # 选择召回率 >= 0.95 的最高精确率阈值
    target_recall = 0.95
    best_thresh = 0.5
    best_precision = 0.0
    for i in range(len(thresholds)):
        if recalls[i] >= target_recall:
            if precisions[i] > best_precision:
                best_precision = precisions[i]
                best_thresh = thresholds[i]
    if best_thresh == 0.5:
        # 未达到目标，选择召回率最高的阈值
        best_thresh = thresholds[np.argmax(recalls)]
    print(f"\n最佳阈值 (目标召回率≥{target_recall}): {best_thresh:.3f}")

    # 9. 输出详细评估报告
    print_evaluation_report(y_test, y_pred, y_pred_prob, best_thresh, model_name="RandomForest")

    # 10. 特征重要性
    importances = rf.feature_importances_
    feature_names = [f'feat_{i}' for i in range(X_feat.shape[1])]
    indices = np.argsort(importances)[::-1]
    print("\n特征重要性前十:")
    for i in range(10):
        print(f"{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

    # 11. 保存模型及预处理器
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, 'random_forest_model.pkl')
    joblib.dump(rf, model_path)
    print(f"\n模型已保存到: {model_path}")

    scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"标准化器已保存到: {scaler_path}")

    # 保存配置
    config = {
        'sequence_length': SEQUENCE_LENGTH,
        'prediction_horizon': PREDICTION_HORIZON,
        'best_threshold': best_thresh,
        'use_smoothing': USE_SMOOTHING,
        'smooth_window': SMOOTH_WINDOW if USE_SMOOTHING else None,
        'smooth_polyorder': SMOOTH_POLYORDER if USE_SMOOTHING else None,
        'feature_names': feature_names,
        'trained_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    with open(os.path.join(MODEL_DIR, 'config.pkl'), 'wb') as f:
        pickle.dump(config, f)
    print("配置已保存。")

    # 绘制PR曲线
    plt.figure(figsize=(8,6))
    plt.plot(recalls, precisions, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.savefig(os.path.join(MODEL_DIR, 'pr_curve.png'))
    plt.show()

    print("\n训练完成！")

if __name__ == "__main__":
    main()