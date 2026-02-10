"""
检查数据文件中的时间间隔
找出存在大于2秒间隔的文件
"""

import os
import glob
import pandas as pd
import numpy as np

DATA_DIR = "data"
MAX_GAP_THRESHOLD = 2.0  # 2秒阈值

print("=" * 70)
print("检查数据文件时间间隔")
print("=" * 70)

csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
csv_files.sort()

files_with_large_gaps = []
all_stats = []

for csv_file in csv_files:
    try:
        df = pd.read_csv(csv_file)
        df = df.sort_values('Time').reset_index(drop=True)
        
        time_values = df['Time'].values
        time_diffs = np.diff(time_values)
        
        max_gap = np.max(time_diffs)
        min_gap = np.min(time_diffs)
        avg_gap = np.mean(time_diffs)
        median_gap = np.median(time_diffs)
        
        # 统计大于2秒的间隔数量
        large_gaps = time_diffs[time_diffs > MAX_GAP_THRESHOLD]
        large_gap_count = len(large_gaps)
        
        filename = os.path.basename(csv_file)
        is_exception = "_exception" in filename
        
        stats = {
            'filename': filename,
            'is_exception': is_exception,
            'total_points': len(time_values),
            'min_gap': min_gap,
            'avg_gap': avg_gap,
            'median_gap': median_gap,
            'max_gap': max_gap,
            'large_gap_count': large_gap_count
        }
        all_stats.append(stats)
        
        if max_gap > MAX_GAP_THRESHOLD:
            files_with_large_gaps.append(stats)
            print(f"\n⚠️  {filename}")
            print(f"   类型: {'异常' if is_exception else '正常'}")
            print(f"   总数据点: {len(time_values)}")
            print(f"   最小间隔: {min_gap:.3f}秒")
            print(f"   平均间隔: {avg_gap:.3f}秒")
            print(f"   中位间隔: {median_gap:.3f}秒")
            print(f"   最大间隔: {max_gap:.3f}秒 ⚠️")
            print(f"   >2秒间隔数: {large_gap_count}")
            
            # 显示大间隔的位置
            if large_gap_count <= 5:
                large_gap_indices = np.where(time_diffs > MAX_GAP_THRESHOLD)[0]
                print(f"   大间隔位置: {large_gap_indices.tolist()}")
        
    except Exception as e:
        print(f"\n❌ 读取 {csv_file} 失败: {e}")

print("\n" + "=" * 70)
print("统计摘要")
print("=" * 70)
print(f"总文件数: {len(csv_files)}")
print(f"存在大间隔(>2秒)的文件数: {len(files_with_large_gaps)}")

if files_with_large_gaps:
    print(f"\n存在大间隔的文件列表:")
    for stats in files_with_large_gaps:
        print(f"  - {stats['filename']}: 最大间隔 {stats['max_gap']:.2f}秒, {stats['large_gap_count']}处大间隔")
else:
    print("\n✅ 所有文件的时间间隔都小于2秒")

# 统计正常文件和异常文件的间隔情况
normal_files = [s for s in all_stats if not s['is_exception']]
exception_files = [s for s in all_stats if s['is_exception']]

if normal_files:
    avg_normal = np.mean([s['avg_gap'] for s in normal_files])
    max_normal = np.max([s['max_gap'] for s in normal_files])
    print(f"\n正常文件 ({len(normal_files)}个):")
    print(f"  平均间隔: {avg_normal:.3f}秒")
    print(f"  最大间隔: {max_normal:.3f}秒")

if exception_files:
    avg_exception = np.mean([s['avg_gap'] for s in exception_files])
    max_exception = np.max([s['max_gap'] for s in exception_files])
    print(f"\n异常文件 ({len(exception_files)}个):")
    print(f"  平均间隔: {avg_exception:.3f}秒")
    print(f"  最大间隔: {max_exception:.3f}秒")

print("\n" + "=" * 70)
