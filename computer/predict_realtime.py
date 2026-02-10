"""
电池热失控实时预测脚本
每0.5秒接收温度数据并实时预测热失控风险
"""

import os
import numpy as np
import pickle
from collections import deque
from datetime import datetime
import tensorflow as tf
from tensorflow import keras

class RealTimePredictor:
    """实时热失控预测器"""
    
    def __init__(self, model_dir="models"):
        """初始化预测器"""
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.sequence_length = None
        self.temperature_buffer = None
        self.load_model()
        
    def load_model(self):
        """加载训练好的模型"""
        print("正在加载模型...")
        
        # 加载配置
        config_path = os.path.join(self.model_dir, 'config.pkl')
        with open(config_path, 'rb') as f:
            config = pickle.load(f)
        
        self.sequence_length = config['sequence_length']
        print(f"序列长度: {self.sequence_length}")
        print(f"训练日期: {config['trained_date']}")
        
        # 加载模型
        model_path = config['model_path']
        self.model = keras.models.load_model(model_path)
        print(f"✓ 模型已加载")
        
        # 加载标准化器
        scaler_path = config['scaler_path']
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        print(f"✓ 标准化器已加载")
        
        # 初始化温度缓冲区
        self.temperature_buffer = deque(maxlen=self.sequence_length)
        print(f"✓ 缓冲区已初始化 (容量: {self.sequence_length})")
        
    def add_temperature(self, temperature):
        """
        添加新的温度数据点
        
        参数:
            temperature: 温度值
        """
        self.temperature_buffer.append(temperature)
        
    def can_predict(self):
        """检查是否有足够的数据进行预测"""
        return len(self.temperature_buffer) >= self.sequence_length
    
    def predict(self):
        """
        预测当前热失控风险
        
        返回:
            risk_prob: 风险概率 (0-1)
            risk_level: 风险等级 ('低', '中', '高', '极高')
            warning: 是否需要警告
        """
        if not self.can_predict():
            return None, None, False
        
        # 获取最近的温度序列
        sequence = np.array(list(self.temperature_buffer))
        
        # 标准化
        sequence_scaled = self.scaler.transform(sequence.reshape(-1, 1))
        sequence_scaled = sequence_scaled.reshape(1, self.sequence_length, 1)
        
        # 预测
        risk_prob = self.model.predict(sequence_scaled, verbose=0)[0][0]
        
        # 确定风险等级
        if risk_prob < 0.3:
            risk_level = "低"
            warning = False
        elif risk_prob < 0.6:
            risk_level = "中"
            warning = False
        elif risk_prob < 0.8:
            risk_level = "高"
            warning = True
        else:
            risk_level = "极高"
            warning = True
        
        return risk_prob, risk_level, warning
    
    def get_temperature_stats(self):
        """获取温度统计信息"""
        if len(self.temperature_buffer) == 0:
            return None
        
        temps = np.array(list(self.temperature_buffer))
        return {
            'current': temps[-1],
            'mean': np.mean(temps),
            'max': np.max(temps),
            'min': np.min(temps),
            'std': np.std(temps),
            'trend': temps[-1] - temps[0] if len(temps) > 1 else 0
        }


def simulate_real_time_prediction():
    """模拟实时预测（用于测试）"""
    import time
    import pandas as pd
    
    print("=" * 70)
    print("电池热失控实时预测系统")
    print("=" * 70)
    
    # 初始化预测器
    predictor = RealTimePredictor()
    
    # 加载测试数据
    print("\n加载测试数据...")
    test_file = "data/temp_data_036_exception.csv"  # 使用一个异常文件进行测试
    df = pd.read_csv(test_file)
    df = df.sort_values('Time')
    temperatures = df['Temperature'].values
    
    print(f"测试文件: {test_file}")
    print(f"数据点数: {len(temperatures)}")
    print("\n开始实时预测...\n")
    
    # 模拟实时数据流
    prediction_count = 0
    warning_count = 0
    
    for i, temp in enumerate(temperatures):
        # 添加温度数据
        predictor.add_temperature(temp)
        
        # 每0.5秒（模拟）进行一次预测
        if predictor.can_predict() and i % 5 == 0:  # 每5个数据点显示一次
            risk_prob, risk_level, warning = predictor.predict()
            stats = predictor.get_temperature_stats()
            
            prediction_count += 1
            if warning:
                warning_count += 1
            
            # 显示预测结果
            timestamp = datetime.now().strftime('%H:%M:%S')
            print(f"[{timestamp}] 数据点 #{i+1}")
            print(f"  当前温度: {stats['current']:.2f}°C")
            print(f"  平均温度: {stats['mean']:.2f}°C")
            print(f"  温度趋势: {stats['trend']:+.2f}°C")
            print(f"  热失控风险: {risk_prob:.2%} ({risk_level})")
            
            if warning:
                print(f"  ⚠️  警告: 检测到热失控风险！")
            
            print("-" * 70)
            
            # 模拟延迟
            time.sleep(0.1)
    
    print("\n" + "=" * 70)
    print("预测完成")
    print(f"总预测次数: {prediction_count}")
    print(f"警告次数: {warning_count}")
    print(f"警告率: {warning_count/prediction_count*100:.1f}%")
    print("=" * 70)


def predict_from_input():
    """从用户输入获取温度数据并预测"""
    print("=" * 70)
    print("电池热失控实时预测系统 - 手动输入模式")
    print("=" * 70)
    
    # 初始化预测器
    predictor = RealTimePredictor()
    
    print(f"\n请输入温度数据（需要至少 {predictor.sequence_length} 个数据点）")
    print("输入格式: 每行一个温度值，输入 'q' 退出\n")
    
    count = 0
    while True:
        try:
            user_input = input(f"温度 #{count+1}: ").strip()
            
            if user_input.lower() == 'q':
                break
            
            # 解析温度值
            temp = float(user_input)
            predictor.add_temperature(temp)
            count += 1
            
            # 如果有足够的数据，进行预测
            if predictor.can_predict():
                risk_prob, risk_level, warning = predictor.predict()
                stats = predictor.get_temperature_stats()
                
                print(f"\n--- 预测结果 ---")
                print(f"当前温度: {stats['current']:.2f}°C")
                print(f"平均温度: {stats['mean']:.2f}°C")
                print(f"温度范围: {stats['min']:.2f}°C ~ {stats['max']:.2f}°C")
                print(f"温度趋势: {stats['trend']:+.2f}°C")
                print(f"热失控风险: {risk_prob:.2%} ({risk_level})")
                
                if warning:
                    print(f"⚠️  警告: 检测到热失控风险！")
                
                print("-" * 70 + "\n")
            else:
                remaining = predictor.sequence_length - len(predictor.temperature_buffer)
                print(f"需要再输入 {remaining} 个数据点才能开始预测\n")
                
        except ValueError:
            print("错误: 请输入有效的数字\n")
        except KeyboardInterrupt:
            print("\n\n程序已停止")
            break


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--manual":
        # 手动输入模式
        predict_from_input()
    else:
        # 模拟模式
        simulate_real_time_prediction()
