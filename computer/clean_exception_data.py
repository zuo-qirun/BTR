"""
清洗异常数据文件
删除温度峰值后的下降段，只保留上升段和高温段
"""

import os
import glob
import pandas as pd
import numpy as np
import shutil
from datetime import datetime

DATA_DIR = "data"
BACKUP_DIR = f"data_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
TEMP_THRESHOLD = 100  # 温度阈值（摄氏度）
DROP_THRESHOLD = 10  # 温度下降阈值（摄氏度），超过此值认为开始下降

print("=" * 70)
print("清洗异常数据文件")
print("=" * 70)

# 创建备份目录
os.makedirs(BACKUP_DIR, exist_ok=True)
print(f"\n📁 备份目录: {BACKUP_DIR}")

# 获取所有异常文件
exception_files = glob.glob(os.path.join(DATA_DIR, "*_exception.csv"))
exception_files.sort()

cleaned_count = 0
skipped_count = 0

for csv_file in exception_files:
    try:
        filename = os.path.basename(csv_file)
        print(f"\n{'='*70}")
        print(f"📄 处理: {filename}")
        
        # 读取数据
        df = pd.read_csv(csv_file)
        df = df.sort_values('Time').reset_index(drop=True)
        
        temps = df['Temperature'].values
        times = df['Time'].values
        original_len = len(df)
        
        # 统计信息
        max_temp = np.max(temps)
        max_temp_idx = np.argmax(temps)
        
        print(f"   原始数据点数: {original_len}")
        print(f"   峰值温度: {max_temp:.2f}°C (第{max_temp_idx}点, 时间{times[max_temp_idx]:.2f}秒)")
        
        # 检查峰值后是否有显著下降
        should_clean = False
        cut_point = None
        
        if max_temp_idx < len(temps) - 10:
            # 从峰值开始向后查找，找到温度下降超过阈值的点
            for i in range(max_temp_idx + 1, len(temps)):
                if temps[i] < max_temp - DROP_THRESHOLD:
                    cut_point = i
                    temp_drop = max_temp - temps[-1]
                    should_clean = temp_drop > 50  # 总下降超过50度才清洗
                    break
        
        if should_clean and cut_point:
            # 备份原始文件
            backup_path = os.path.join(BACKUP_DIR, filename)
            shutil.copy2(csv_file, backup_path)
            print(f"   ✅ 已备份到: {backup_path}")
            
            # 截断数据：保留到下降段起点
            df_cleaned = df.iloc[:cut_point].copy()
            
            # 保存清洗后的数据
            df_cleaned.to_csv(csv_file, index=False)
            
            removed_points = original_len - len(df_cleaned)
            removed_percent = removed_points / original_len * 100
            
            print(f"   🔧 清洗完成:")
            print(f"      - 截断点: 第{cut_point}点 (时间{times[cut_point]:.2f}秒, 温度{temps[cut_point]:.2f}°C)")
            print(f"      - 保留数据: {len(df_cleaned)}点")
            print(f"      - 删除数据: {removed_points}点 ({removed_percent:.1f}%)")
            print(f"      - 新的温度范围: {df_cleaned['Temperature'].min():.2f}°C ~ {df_cleaned['Temperature'].max():.2f}°C")
            
            cleaned_count += 1
        else:
            if max_temp_idx >= len(temps) - 10:
                reason = "峰值在末尾"
            elif not cut_point:
                reason = "无明显下降"
            else:
                reason = "下降幅度小"
            
            print(f"   ⏭️  跳过清洗 (原因: {reason})")
            skipped_count += 1
        
    except Exception as e:
        print(f"\n❌ 处理 {csv_file} 失败: {e}")
        skipped_count += 1

print("\n" + "=" * 70)
print("清洗完成")
print("=" * 70)
print(f"✅ 已清洗文件数: {cleaned_count}")
print(f"⏭️  跳过文件数: {skipped_count}")
print(f"📁 备份位置: {BACKUP_DIR}")
print("\n提示:")
print("  - 原始文件已备份，如需恢复可从备份目录复制")
print("  - 清洗后的文件已覆盖原文件")
print("  - 建议检查清洗结果后再进行模型训练")
print("=" * 70)
