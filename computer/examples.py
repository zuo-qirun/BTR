"""
使用示例 - 展示如何在实际应用中使用预测模型
"""

import time
import numpy as np
from predict_realtime import RealTimePredictor


def example_1_basic_usage():
    """示例1: 基本使用"""
    print("=" * 70)
    print("示例1: 基本使用")
    print("=" * 70)
    print()
    
    # 初始化预测器
    predictor = RealTimePredictor()
    
    # 模拟温度数据（正常情况）
    print("模拟正常温度数据...")
    base_temp = 30.0
    
    for i in range(70):
        # 生成轻微波动的温度
        temp = base_temp + np.random.normal(0, 0.5)
        predictor.add_temperature(temp)
        
        if predictor.can_predict():
            risk_prob, risk_level, warning = predictor.predict()
            print(f"时间步 {i+1}: 温度={temp:.2f}°C, 风险={risk_prob:.2%} ({risk_level})")
            
            if warning:
                print("  ⚠️  警告!")
        
        time.sleep(0.1)
    
    print()


def example_2_thermal_runaway_detection():
    """示例2: 热失控检测"""
    print("=" * 70)
    print("示例2: 热失控检测")
    print("=" * 70)
    print()
    
    # 初始化预测器
    predictor = RealTimePredictor()
    
    # 模拟温度数据（热失控情况）
    print("模拟热失控场景...")
    base_temp = 30.0
    
    for i in range(100):
        # 前60步正常，之后温度快速上升
        if i < 60:
            temp = base_temp + np.random.normal(0, 0.3)
        else:
            # 温度开始快速上升
            temp = base_temp + (i - 60) * 0.5 + np.random.normal(0, 0.5)
        
        predictor.add_temperature(temp)
        
        if predictor.can_predict() and i % 5 == 0:
            risk_prob, risk_level, warning = predictor.predict()
            stats = predictor.get_temperature_stats()
            
            print(f"时间步 {i+1}:")
            print(f"  当前温度: {stats['current']:.2f}°C")
            print(f"  温度趋势: {stats['trend']:+.2f}°C")
            print(f"  风险评估: {risk_prob:.2%} ({risk_level})")
            
            if warning:
                print("  🚨 警告: 检测到热失控风险!")
            print()
        
        time.sleep(0.05)
    
    print()


def example_3_continuous_monitoring():
    """示例3: 连续监控"""
    print("=" * 70)
    print("示例3: 连续监控系统集成")
    print("=" * 70)
    print()
    
    # 初始化预测器
    predictor = RealTimePredictor()
    
    # 模拟连续监控
    print("模拟连续监控（按 Ctrl+C 停止）...")
    print()
    
    base_temp = 31.0
    step = 0
    warning_count = 0
    
    try:
        while True:
            # 模拟从传感器读取温度
            # 在实际应用中，这里应该是: temp = read_from_sensor()
            temp = base_temp + np.random.normal(0, 1.0)
            
            # 添加温度数据
            predictor.add_temperature(temp)
            step += 1
            
            # 每秒进行一次预测（假设每0.5秒采集一次数据）
            if predictor.can_predict() and step % 2 == 0:
                risk_prob, risk_level, warning = predictor.predict()
                stats = predictor.get_temperature_stats()
                
                # 显示监控信息
                timestamp = time.strftime('%H:%M:%S')
                print(f"[{timestamp}] 温度: {stats['current']:.2f}°C | "
                      f"平均: {stats['mean']:.2f}°C | "
                      f"风险: {risk_prob:.2%} ({risk_level})", end="")
                
                if warning:
                    print(" | ⚠️  警告!")
                    warning_count += 1
                    
                    # 在实际应用中，这里应该触发报警
                    # trigger_alarm()
                    # send_notification()
                else:
                    print()
            
            # 模拟0.5秒采样间隔
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print(f"\n\n监控已停止")
        print(f"总监控步数: {step}")
        print(f"警告次数: {warning_count}")


def example_4_batch_prediction():
    """示例4: 批量预测"""
    print("=" * 70)
    print("示例4: 批量预测")
    print("=" * 70)
    print()
    
    # 初始化预测器
    predictor = RealTimePredictor()
    
    # 准备多组测试数据
    test_cases = [
        {
            'name': '正常情况',
            'data': [30.0 + np.random.normal(0, 0.3) for _ in range(60)]
        },
        {
            'name': '轻微升温',
            'data': [30.0 + i * 0.05 + np.random.normal(0, 0.2) for i in range(60)]
        },
        {
            'name': '快速升温',
            'data': [30.0 + i * 0.2 + np.random.normal(0, 0.3) for i in range(60)]
        },
        {
            'name': '热失控',
            'data': [30.0 + i * 0.5 + np.random.normal(0, 0.5) for i in range(60)]
        }
    ]
    
    print("批量预测结果:\n")
    
    for case in test_cases:
        # 清空缓冲区
        predictor.temperature_buffer.clear()
        
        # 添加数据
        for temp in case['data']:
            predictor.add_temperature(temp)
        
        # 预测
        risk_prob, risk_level, warning = predictor.predict()
        
        print(f"{case['name']}:")
        print(f"  温度范围: {min(case['data']):.2f}°C ~ {max(case['data']):.2f}°C")
        print(f"  温度变化: {case['data'][-1] - case['data'][0]:+.2f}°C")
        print(f"  风险评估: {risk_prob:.2%} ({risk_level})")
        print(f"  是否警告: {'是 ⚠️' if warning else '否 ✓'}")
        print()


def main():
    """主函数"""
    print("\n电池热失控预测系统 - 使用示例\n")
    
    examples = [
        ("基本使用", example_1_basic_usage),
        ("热失控检测", example_2_thermal_runaway_detection),
        ("连续监控", example_3_continuous_monitoring),
        ("批量预测", example_4_batch_prediction)
    ]
    
    print("可用示例:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    print(f"  0. 运行所有示例")
    print()
    
    try:
        choice = input("请选择示例 (0-4): ").strip()
        
        if choice == '0':
            # 运行所有示例（除了连续监控）
            for i, (name, func) in enumerate(examples):
                if i != 2:  # 跳过连续监控
                    func()
                    input("\n按 Enter 继续下一个示例...")
        elif choice in ['1', '2', '3', '4']:
            examples[int(choice) - 1][1]()
        else:
            print("无效选择")
            
    except KeyboardInterrupt:
        print("\n\n程序已停止")
    except Exception as e:
        print(f"\n错误: {e}")


if __name__ == "__main__":
    main()
