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
# 路径基于当前脚本目录，确保从仓库根或其它位置运行时能找到 data 和 models 文件夹
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")  # 数据目录
MODEL_DIR = os.path.join(BASE_DIR, "models")  # 模型保存目录
SEQUENCE_LENGTH = 60  # 时间序列长度（使用过去30秒的数据，每0.5秒一个点）
PREDICTION_HORIZON = 40  # 预测未来20秒（40个时间步）
BATCH_SIZE = 128
EPOCHS = 2
# 初始学习率，训练过程中使用 ReduceLROnPlateau 自适应调整学习率
INITIAL_LEARNING_RATE = 0.001
TIME_INTERVAL = 0.5  # 重采样时间间隔（秒）- 统一为0.5秒
MAX_GAP = 70.0  # 最大允许的原始数据时间间隔（秒），超过此值跳过该区间
CLASS_WEIGHT = {0: 1.0, 1: 50.0}  # 类别权重：热失控样本权重提高50倍
# 实验配置：多个候选类别权重（默认列表）
CLASS_WEIGHT_LIST = [
    {0: 1.0, 1: 5.0},
    {0: 1.0, 1: 10.0},
    {0: 1.0, 1: 20.0},
    {0: 1.0, 1: 50.0},
    {0: 1.0, 1: 100.0},
]
# 是否为每次试验保存完整模型
SAVE_MODELS_PER_EXPERIMENT = True
# 目标 GPU 名称中包含的子串（用于选择特定 GPU，例如 '5060'）
GPU_NAME_SUBSTR = '5060'

class BatteryThermalRunawayPredictor:
    """电池热失控预测器"""
    
    def __init__(self, sequence_length=SEQUENCE_LENGTH, prediction_horizon=PREDICTION_HORIZON):
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.model = None
        self.scaler = StandardScaler()
        self.history = None
        
    def load_data(self, data_dir):
        """加载所有CSV数据文件，统一使用0.5秒间隔重采样"""
        print("正在加载数据...")
        print(f"重采样间隔: {TIME_INTERVAL}秒")
        
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
                df = df.sort_values('Time').reset_index(drop=True)
                
                # 基于时间的重采样：使用线性插值生成等间隔时间序列
                time_values = df['Time'].values
                temp_values = df['Temperature'].values
                
                # 检查原始数据的时间间隔
                time_diffs = np.diff(time_values)
                max_gap = np.max(time_diffs)
                avg_gap = np.mean(time_diffs)
                
                # 创建等间隔时间点（统一为0.5秒）
                start_time = time_values[0]
                end_time = time_values[-1]
                uniform_times = np.arange(start_time, end_time, TIME_INTERVAL)
                
                # 线性插值（正常文件从5秒插值到0.5秒，异常文件从0.02秒降采样到0.5秒）
                uniform_temps = np.interp(uniform_times, time_values, temp_values)
                
                # 对异常文件：删除前面温度平稳的部分，只保留开始上升的部分
                if is_exception:
                    # 找到温度开始显著上升的位置（基线温度+5°C）
                    baseline_temp = np.percentile(uniform_temps[:min(1000, len(uniform_temps))], 50)  # 前500秒的中位数作为基线
                    threshold_temp = baseline_temp + 5.0  # 基线+5°C
                    
                    # 找到首次超过阈值的位置
                    rising_indices = np.where(uniform_temps > threshold_temp)[0]
                    if len(rising_indices) > 0:
                        start_idx = max(0, rising_indices[0] - 200)  # 保留上升前100秒（200个点）的数据
                        uniform_temps = uniform_temps[start_idx:]
                        uniform_times = uniform_times[start_idx:]
                
                # 标记哪些插值点是不可靠的（原始数据间隔过大，如60秒）
                valid_mask = np.ones(len(uniform_temps), dtype=bool)
                if max_gap > MAX_GAP:
                    for i, t in enumerate(uniform_times):
                        # 找到 t 所在的原始数据区间
                        idx = np.searchsorted(time_values, t)
                        if idx > 0 and idx < len(time_values):
                            gap = time_values[idx] - time_values[idx-1]
                            if gap > MAX_GAP:
                                valid_mask[i] = False
                
                # 创建时间序列样本
                max_sequences = len(uniform_temps) - self.sequence_length - self.prediction_horizon
                
                if max_sequences <= 0:
                    print(f"  ⚠️  {os.path.basename(csv_file)} 数据点不足，跳过")
                    continue
                
                # 对正常文件进行随机采样，减少数据量但保持多样性
                if is_exception:
                    # 异常文件：使用所有点（已经删除了前面平稳部分）
                    sample_indices = range(max_sequences)
                else:
                    # 正常文件：每个文件最多取2000个序列
                    max_normal_samples = 2000
                    num_samples = min(max_normal_samples, max_sequences)
                    if max_sequences > max_normal_samples:
                        # 随机采样
                        sample_indices = np.random.choice(max_sequences, num_samples, replace=False)
                        sample_indices = sorted(sample_indices)
                    else:
                        # 全部使用
                        sample_indices = range(max_sequences)
                
                sample_count = 0
                skipped_count = 0
                for i in sample_indices:
                    # 检查这个序列是否包含不可靠的插值点
                    sequence_mask = valid_mask[i:i + self.sequence_length + self.prediction_horizon]
                    if not np.all(sequence_mask):
                        skipped_count += 1
                        continue  # 跳过包含超大间隔的序列
                        
                    sequence = uniform_temps[i:i + self.sequence_length]
                    # 检查未来是否会发生热失控
                    future_temps = uniform_temps[i + self.sequence_length:i + self.sequence_length + self.prediction_horizon]
                    
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
                
                skip_info = f" (跳过{skipped_count}个大间隔序列)" if skipped_count > 0 else ""
                print(f"已加载: {os.path.basename(csv_file)} - {'异常' if is_exception else '正常'} - 原始{len(temp_values)}点 → 重采样{len(uniform_temps)}点 → {sample_count}个序列{skip_info}")
                
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
            optimizer=keras.optimizers.Adam(learning_rate=INITIAL_LEARNING_RATE),
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
                min_lr=1e-7,
                verbose=1
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
        from sklearn.metrics import confusion_matrix, classification_report, f1_score, roc_auc_score
        cm = confusion_matrix(y_test, y_pred)
        print("\n混淆矩阵:")
        print(cm)
        print("\n分类报告:")
        print(classification_report(y_test, y_pred, target_names=['正常', '热失控']))
        
        # 额外指标
        f1 = f1_score(y_test, y_pred)
        try:
            auc = roc_auc_score(y_test, y_pred_prob)
            print(f"\n额外指标:")
            print(f"F1-Score: {f1:.4f}")
            print(f"AUC-ROC: {auc:.4f}")
        except:
            print(f"\n额外指标:")
            print(f"F1-Score: {f1:.4f}")
        
        # 分析不同阈值下的性能
        print("\n不同阈值下的召回率和精确率:")
        for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
            y_pred_thresh = (y_pred_prob > threshold).astype(int)
            from sklearn.metrics import precision_score, recall_score
            precision = precision_score(y_test, y_pred_thresh, zero_division=0)
            recall = recall_score(y_test, y_pred_thresh, zero_division=0)
            print(f"  阈值={threshold:.1f}: Precision={precision:.4f}, Recall={recall:.4f}")
        
        print(f"\n推荐阈值: 0.3 (当前使用)")
        print(f"说明: 对于热失控预警，优先保证高召回率（不漏报），可接受一定误报")
        
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


def select_gpu_by_name(substr=GPU_NAME_SUBSTR):
    """尝试选择名称包含给定子串的 GPU 并限制 TensorFlow 仅使用它。"""
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            print("未检测到 GPU，使用 CPU 运行。")
            return None

        selected = None
        for gpu in gpus:
            # 物理设备对象名可能包含信息，尝试匹配子串
            if substr in gpu.name or substr in str(gpu):
                selected = gpu
                break

        if selected is None:
            # 未找到匹配项，默认使用第0个
            selected = gpus[0]
            print(f"未找到名称含 '{substr}' 的 GPU，使用第一块 GPU: {selected}")
        else:
            print(f"选择 GPU: {selected}")

        tf.config.set_visible_devices(selected, 'GPU')
        tf.config.experimental.set_memory_growth(selected, True)
        return selected
    except Exception as e:
        print(f"设置 GPU 时出错: {e}\n将使用默认设备。")
        return None


def run_experiments():
    """按 `CLASS_WEIGHT_LIST` 顺序运行多个试验，保存模型和汇总结果到 CSV。"""
    # 选择 GPU（如果可用）
    select_gpu_by_name()

    # 加载数据一次，供所有试验复用
    predictor = BatteryThermalRunawayPredictor(
        sequence_length=SEQUENCE_LENGTH,
        prediction_horizon=PREDICTION_HORIZON
    )
    X, y = predictor.load_data(DATA_DIR)
    if X.size == 0 or len(X) == 0:
        print("\n未找到可用样本 (数据集为空)。请将 CSV 数据放到 data 目录或检查数据加载逻辑。实验已中止。")
        return
    X_train, X_test, y_train, y_test = predictor.preprocess_data(X, y)

    # 结果 CSV
    os.makedirs(MODEL_DIR, exist_ok=True)
    results_csv = os.path.join(MODEL_DIR, 'experiment_results.csv')
    if not os.path.exists(results_csv):
        with open(results_csv, 'w') as f:
            f.write('exp_name,class_weight,val_loss,val_accuracy,precision,recall,f1,auc,trained_date,model_path\n')

    for idx, cw in enumerate(CLASS_WEIGHT_LIST, start=1):
        exp_name = f'exp_{idx}_w{int(list(cw.values())[1])}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        print('\n' + '='*60)
        print(f"开始试验: {exp_name} - class_weight={cw}")

        # 构建并训练新模型实例
        predictor = BatteryThermalRunawayPredictor(
            sequence_length=SEQUENCE_LENGTH,
            prediction_horizon=PREDICTION_HORIZON
        )
        predictor.build_model(input_shape=(SEQUENCE_LENGTH, 1))

        # 训练时传入当前权重
        predictor.history = predictor.model.fit(
            X_train, y_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=(X_test, y_test),
            callbacks=[
                keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1)
            ],
            class_weight=cw,
            verbose=1
        )

        # 评估
        results = predictor.model.evaluate(X_test, y_test, verbose=0)
        y_pred_prob = predictor.model.predict(X_test)
        y_pred = (y_pred_prob > 0.3).astype(int)
        from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y_test, y_pred_prob)
        except:
            auc = ''

        model_path = ''
        if SAVE_MODELS_PER_EXPERIMENT:
            exp_dir = os.path.join(MODEL_DIR, exp_name)
            os.makedirs(exp_dir, exist_ok=True)
            model_path = os.path.join(exp_dir, 'thermal_runaway_model.h5')
            predictor.model.save(model_path)
            # 保存 scaler
            scaler_path = os.path.join(exp_dir, 'scaler.pkl')
            with open(scaler_path, 'wb') as f:
                pickle.dump(predictor.scaler, f)
            # 保存 config
            config = {
                'sequence_length': predictor.sequence_length,
                'prediction_horizon': predictor.prediction_horizon,
                'class_weight': cw,
                'model_path': model_path,
                'scaler_path': scaler_path,
                'trained_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            with open(os.path.join(exp_dir, 'config.pkl'), 'wb') as f:
                pickle.dump(config, f)

        # 写入结果 CSV
        with open(results_csv, 'a') as f:
            f.write(f"{exp_name},{cw},{results[0]:.6f},{results[1]:.6f},{precision:.6f},{recall:.6f},{f1:.6f},{auc},{datetime.now().strftime('%Y-%m-%d %H:%M:%S')},{model_path}\n")

        print(f"试验 {exp_name} 完成: val_loss={results[0]:.6f}, val_acc={results[1]:.6f}, precision={precision:.4f}, recall={recall:.4f}, f1={f1:.4f}, auc={auc}")



def main():
    """主函数"""
    print("=" * 60)
    print("电池热失控预测模型训练")
    print("=" * 60)
    # 运行批量实验（每个 class_weight 在 CLASS_WEIGHT_LIST 中）
    run_experiments()


if __name__ == "__main__":
    main()
