"""
分析正常文件，查找可能被遗漏标记的异常数据
检测标准：
1. 温度超过阈值（如100°C）
2. 温度快速上升（短时间内上升超过阈值）
3. 温度异常波动
"""

import os
import glob
import pandas as pd
import numpy as np

DATA_DIR = "data"
TEMP_THRESHOLD = 80  # 温度阈值（摄氏度）- 降低阈值以发现潜在异常
RAPID_RISE_THRESHOLD = 20  # 快速上升阈值（度/秒）
RAPID_RISE_WINDOW = 10  # 检测窗口（秒）

print("=" * 70)
print("分析正常文件，查找可能的异常数据")
print("=" * 70)
print(f"检测标准:")
print(f"  1. 温度超过 {TEMP_THRESHOLD}°C")
print(f"  2. {RAPID_RISE_WINDOW}秒内温度上升超过 {RAPID_RISE_THRESHOLD}°C")
print("=" * 70)

# 获取所有正常文件（不包含_exception的文件）
normal_files = [f for f in glob.glob(os.path.join(DATA_DIR, "*.csv")) 
                if "_exception" not in os.path.basename(f)]
normal_files.sort()

suspicious_files = []

for csv_file in normal_files:
    try:
        filename = os.path.basename(csv_file)
        
        # 读取数据
        df = pd.read_csv(csv_file)
        df = df.sort_values('Time').reset_index(drop=True)
        
        temps = df['Temperature'].values
        times = df['Time'].values
        
        # 统计信息
        max_temp = np.max(temps)
        min_temp = np.min(temps)
        mean_temp = np.mean(temps)
        std_temp = np.std(temps)
        
        # 检测1: 温度超过阈值
        high_temp_count = np.sum(temps > TEMP_THRESHOLD)
        high_temp_flag = high_temp_count > 0
        
        # 检测2: 快速温度上升
        rapid_rise_flag = False
        max_rise_rate = 0
        max_rise_location = None
        
        # 计算温度变化率
        for i in range(len(temps) - 1):
            # 找到时间窗口内的数据点
            window_end_time = times[i] + RAPID_RISE_WINDOW
            window_indices = np.where((times >= times[i]) & (times <= window_end_time))[0]
            
            if len(window_indices) > 1:
                temp_rise = temps[window_indices[-1]] - temps[window_indices[0]]
                time_span = times[window_indices[-1]] - times[window_indices[0]]
                
                if time_span > 0:
                    rise_rate = temp_rise / time_span  # 度/秒
                    
                    if rise_rate > max_rise_rate:
                        max_rise_rate = rise_rate
                        max_rise_location = i
                    
                    if temp_rise > RAPID_RISE_THRESHOLD:
                        rapid_rise_flag = True
        
        # 检测3: 异常波动（标准差过大）
        high_volatility_flag = std_temp > 5.0
        
        # 判断是否可疑
        is_suspicious = high_temp_flag or rapid_rise_flag
        
        if is_suspicious:
            print(f"\n⚠️  {filename}")
            print(f"   温度范围: {min_temp:.2f}°C ~ {max_temp:.2f}°C")
            print(f"   平均温度: {mean_temp:.2f}°C")
            print(f"   标准差: {std_temp:.2f}°C")
            
            if high_temp_flag:
                print(f"   🔥 检测到高温: {high_temp_count}个点超过{TEMP_THRESHOLD}°C (最高{max_temp:.2f}°C)")
                # 找到高温点的位置
                high_temp_indices = np.where(temps > TEMP_THRESHOLD)[0]
                first_high = high_temp_indices[0]
                print(f"      首次高温: 第{first_high}点 (时间{times[first_high]:.2f}秒)")
            
            if rapid_rise_flag:
                print(f"   📈 检测到快速上升: 最大上升速率 {max_rise_rate:.2f}°C/秒")
                if max_rise_location is not None:
                    print(f"      位置: 第{max_rise_location}点 (时间{times[max_rise_location]:.2f}秒)")
            
            if high_volatility_flag:
                print(f"   📊 温度波动较大: 标准差 {std_temp:.2f}°C")
            
            suspicious_files.append({
                'filename': filename,
                'max_temp': max_temp,
                'high_temp_count': high_temp_count,
                'max_rise_rate': max_rise_rate,
                'std_temp': std_temp,
                'high_temp': high_temp_flag,
                'rapid_rise': rapid_rise_flag,
                'high_volatility': high_volatility_flag
            })
        
    except Exception as e:
        print(f"\n❌ 读取 {csv_file} 失败: {e}")

print("\n" + "=" * 70)
print("分析完成")
print("=" * 70)
print(f"总正常文件数: {len(normal_files)}")
print(f"可疑文件数: {len(suspicious_files)}")

if suspicious_files:
    print(f"\n可疑文件列表（按最高温度排序）:")
    suspicious_files.sort(key=lambda x: x['max_temp'], reverse=True)
    
    for i, info in enumerate(suspicious_files, 1):
        flags = []
        if info['high_temp']:
            flags.append(f"高温{info['max_temp']:.1f}°C")
        if info['rapid_rise']:
            flags.append(f"快速上升{info['max_rise_rate']:.1f}°C/s")
        if info['high_volatility']:
            flags.append(f"高波动{info['std_temp']:.1f}°C")
        
        print(f"  {i}. {info['filename']}: {', '.join(flags)}")
    
    print(f"\n建议:")
    print(f"  1. 检查这些文件的温度曲线，确认是否为真实热失控")
    print(f"  2. 如果确认是异常，将文件重命名为 *_exception.csv")
    print(f"  3. 重新运行数据清洗和模型训练")
else:
    print(f"\n✅ 未发现可疑的正常文件")

print("=" * 70)
