"""
分析异常文件的温度变化模式
检查是否存在温度上升后又下降的情况
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = "data"
TEMP_THRESHOLD = 100  # 温度阈值（摄氏度）

print("=" * 70)
print("分析异常文件的温度变化模式")
print("=" * 70)

# 获取所有异常文件
exception_files = glob.glob(os.path.join(DATA_DIR, "*_exception.csv"))
exception_files.sort()

for csv_file in exception_files:
    try:
        df = pd.read_csv(csv_file)
        df = df.sort_values('Time').reset_index(drop=True)
        
        temps = df['Temperature'].values
        times = df['Time'].values
        
        filename = os.path.basename(csv_file)
        
        # 统计信息
        max_temp = np.max(temps)
        min_temp = np.min(temps)
        mean_temp = np.mean(temps)
        
        # 检查是否超过阈值
        above_threshold = temps > TEMP_THRESHOLD
        above_count = np.sum(above_threshold)
        
        print(f"\n📄 {filename}")
        print(f"   数据点数: {len(temps)}")
        print(f"   温度范围: {min_temp:.2f}°C ~ {max_temp:.2f}°C")
        print(f"   平均温度: {mean_temp:.2f}°C")
        print(f"   超过{TEMP_THRESHOLD}°C的点数: {above_count} ({above_count/len(temps)*100:.1f}%)")
        
        if above_count > 0:
            # 找到超过阈值的区间
            above_indices = np.where(above_threshold)[0]
            first_above = above_indices[0]
            last_above = above_indices[-1]
            
            print(f"   首次超过阈值: 第{first_above}点 (时间{times[first_above]:.2f}秒)")
            print(f"   最后超过阈值: 第{last_above}点 (时间{times[last_above]:.2f}秒)")
            
            # 检查峰值后是否有明显下降
            max_temp_idx = np.argmax(temps)
            print(f"   峰值温度: {max_temp:.2f}°C (第{max_temp_idx}点, 时间{times[max_temp_idx]:.2f}秒)")
            
            # 检查峰值后的温度变化
            if max_temp_idx < len(temps) - 10:
                after_peak = temps[max_temp_idx:]
                temp_drop = max_temp - after_peak[-1]
                
                print(f"   峰值后温度变化: {max_temp:.2f}°C → {after_peak[-1]:.2f}°C (下降{temp_drop:.2f}°C)")
                
                # 判断是否有显著下降
                if temp_drop > 50:
                    print(f"   ⚠️  存在显著温度下降 (>{50}°C)")
                    
                    # 找到下降段的起点
                    # 定义下降段：从峰值开始，温度持续低于峰值-10°C
                    decline_start = None
                    for i in range(max_temp_idx, len(temps)):
                        if temps[i] < max_temp - 10:
                            decline_start = i
                            break
                    
                    if decline_start:
                        print(f"   下降段起点: 第{decline_start}点 (时间{times[decline_start]:.2f}秒, 温度{temps[decline_start]:.2f}°C)")
                        print(f"   建议: 考虑删除或重新标注第{decline_start}点之后的数据")
                else:
                    print(f"   ✅ 温度保持高位，符合真实热失控特征")
            else:
                print(f"   ⚠️  峰值出现在数据末尾，无法判断后续趋势")
        else:
            print(f"   ⚠️  温度未超过{TEMP_THRESHOLD}°C，可能不是真正的热失控")
        
    except Exception as e:
        print(f"\n❌ 读取 {csv_file} 失败: {e}")

print("\n" + "=" * 70)
print("分析完成")
print("=" * 70)
print("\n建议:")
print("1. 如果存在显著温度下降，考虑删除下降段数据")
print("2. 或者将下降段重新标注为正常数据")
print("3. 保留上升段和高温段作为热失控特征")
print("=" * 70)
