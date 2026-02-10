"""
数据预处理和分析脚本
检查数据质量、可视化温度趋势、生成数据报告
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

DATA_DIR = "data"
REPORT_DIR = "data_reports"


def analyze_single_file(file_path):
    """分析单个CSV文件"""
    try:
        df = pd.read_csv(file_path)
        
        # 基本信息
        info = {
            'filename': os.path.basename(file_path),
            'is_exception': '_exception' in os.path.basename(file_path),
            'total_points': len(df),
            'duration': df['Time'].max() - df['Time'].min(),
            'temp_mean': df['Temperature'].mean(),
            'temp_std': df['Temperature'].std(),
            'temp_min': df['Temperature'].min(),
            'temp_max': df['Temperature'].max(),
            'temp_range': df['Temperature'].max() - df['Temperature'].min(),
            'missing_values': df.isnull().sum().sum(),
            'duplicates': df.duplicated().sum()
        }
        
        # 温度变化率
        df_sorted = df.sort_values('Time')
        temp_diff = df_sorted['Temperature'].diff()
        info['max_temp_increase'] = temp_diff.max()
        info['max_temp_decrease'] = temp_diff.min()
        
        return info, df
        
    except Exception as e:
        print(f"分析文件 {file_path} 时出错: {e}")
        return None, None


def generate_data_report():
    """生成数据质量报告"""
    print("=" * 70)
    print("数据质量分析报告")
    print("=" * 70)
    print()
    
    # 获取所有CSV文件
    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    
    if len(csv_files) == 0:
        print(f"错误: 在 {DATA_DIR} 目录中没有找到CSV文件")
        return
    
    print(f"找到 {len(csv_files)} 个数据文件\n")
    
    # 分析所有文件
    all_info = []
    normal_files = []
    exception_files = []
    
    for file_path in csv_files:
        info, df = analyze_single_file(file_path)
        if info:
            all_info.append(info)
            if info['is_exception']:
                exception_files.append((file_path, df))
            else:
                normal_files.append((file_path, df))
    
    # 创建报告目录
    os.makedirs(REPORT_DIR, exist_ok=True)
    
    # 生成统计报告
    print("数据集概览:")
    print(f"  正常数据文件: {len(normal_files)}")
    print(f"  异常数据文件: {len(exception_files)}")
    print()
    
    # 详细统计
    df_report = pd.DataFrame(all_info)
    
    print("温度统计:")
    print(f"  平均温度: {df_report['temp_mean'].mean():.2f}°C")
    print(f"  温度标准差: {df_report['temp_std'].mean():.2f}°C")
    print(f"  最低温度: {df_report['temp_min'].min():.2f}°C")
    print(f"  最高温度: {df_report['temp_max'].max():.2f}°C")
    print()
    
    print("数据质量:")
    print(f"  总数据点: {df_report['total_points'].sum():,}")
    print(f"  平均每文件数据点: {df_report['total_points'].mean():.0f}")
    print(f"  缺失值总数: {df_report['missing_values'].sum()}")
    print(f"  重复值总数: {df_report['duplicates'].sum()}")
    print()
    
    # 保存详细报告
    report_path = os.path.join(REPORT_DIR, 'data_quality_report.csv')
    df_report.to_csv(report_path, index=False, encoding='utf-8-sig')
    print(f"✓ 详细报告已保存: {report_path}")
    
    # 可视化
    visualize_data(normal_files, exception_files, df_report)
    
    return df_report


def visualize_data(normal_files, exception_files, df_report):
    """可视化数据"""
    print("\n生成可视化图表...")
    
    # 创建图表
    fig = plt.figure(figsize=(20, 12))
    
    # 1. 温度分布对比
    ax1 = plt.subplot(2, 3, 1)
    normal_temps = df_report[~df_report['is_exception']]['temp_mean']
    exception_temps = df_report[df_report['is_exception']]['temp_mean']
    
    ax1.hist(normal_temps, bins=20, alpha=0.7, label='正常', color='green')
    ax1.hist(exception_temps, bins=20, alpha=0.7, label='异常', color='red')
    ax1.set_xlabel('平均温度 (°C)')
    ax1.set_ylabel('文件数量')
    ax1.set_title('温度分布对比')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 温度范围对比
    ax2 = plt.subplot(2, 3, 2)
    normal_ranges = df_report[~df_report['is_exception']]['temp_range']
    exception_ranges = df_report[df_report['is_exception']]['temp_range']
    
    ax2.boxplot([normal_ranges, exception_ranges], labels=['正常', '异常'])
    ax2.set_ylabel('温度范围 (°C)')
    ax2.set_title('温度变化范围对比')
    ax2.grid(True, alpha=0.3)
    
    # 3. 数据点数量分布
    ax3 = plt.subplot(2, 3, 3)
    ax3.bar(['正常', '异常'], 
            [df_report[~df_report['is_exception']]['total_points'].sum(),
             df_report[df_report['is_exception']]['total_points'].sum()],
            color=['green', 'red'], alpha=0.7)
    ax3.set_ylabel('数据点总数')
    ax3.set_title('数据量对比')
    ax3.grid(True, alpha=0.3)
    
    # 4. 示例正常温度曲线
    ax4 = plt.subplot(2, 3, 4)
    if len(normal_files) > 0:
        sample_file, sample_df = normal_files[0]
        sample_df_sorted = sample_df.sort_values('Time')
        ax4.plot(sample_df_sorted['Time'], sample_df_sorted['Temperature'], 
                color='green', alpha=0.7, linewidth=0.5)
        ax4.set_xlabel('时间 (秒)')
        ax4.set_ylabel('温度 (°C)')
        ax4.set_title(f'正常数据示例\n{os.path.basename(sample_file)}')
        ax4.grid(True, alpha=0.3)
    
    # 5. 示例异常温度曲线
    ax5 = plt.subplot(2, 3, 5)
    if len(exception_files) > 0:
        sample_file, sample_df = exception_files[0]
        sample_df_sorted = sample_df.sort_values('Time')
        ax5.plot(sample_df_sorted['Time'], sample_df_sorted['Temperature'], 
                color='red', alpha=0.7, linewidth=0.5)
        ax5.set_xlabel('时间 (秒)')
        ax5.set_ylabel('温度 (°C)')
        ax5.set_title(f'异常数据示例\n{os.path.basename(sample_file)}')
        ax5.grid(True, alpha=0.3)
    
    # 6. 温度变化率对比
    ax6 = plt.subplot(2, 3, 6)
    normal_increase = df_report[~df_report['is_exception']]['max_temp_increase']
    exception_increase = df_report[df_report['is_exception']]['max_temp_increase']
    
    ax6.boxplot([normal_increase, exception_increase], labels=['正常', '异常'])
    ax6.set_ylabel('最大温度上升率 (°C/步)')
    ax6.set_title('温度变化率对比')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    chart_path = os.path.join(REPORT_DIR, 'data_visualization.png')
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    print(f"✓ 可视化图表已保存: {chart_path}")
    
    plt.show()


def check_data_quality():
    """检查数据质量问题"""
    print("\n" + "=" * 70)
    print("数据质量检查")
    print("=" * 70)
    print()
    
    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    issues = []
    
    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)
            filename = os.path.basename(file_path)
            
            # 检查缺失值
            if df.isnull().sum().sum() > 0:
                issues.append(f"⚠️  {filename}: 包含 {df.isnull().sum().sum()} 个缺失值")
            
            # 检查重复值
            if df.duplicated().sum() > 0:
                issues.append(f"⚠️  {filename}: 包含 {df.duplicated().sum()} 个重复行")
            
            # 检查数据点数量
            if len(df) < 100:
                issues.append(f"⚠️  {filename}: 数据点过少 ({len(df)} 个)")
            
            # 检查温度异常值
            temp_mean = df['Temperature'].mean()
            temp_std = df['Temperature'].std()
            outliers = df[(df['Temperature'] < temp_mean - 3*temp_std) | 
                         (df['Temperature'] > temp_mean + 3*temp_std)]
            if len(outliers) > 0:
                issues.append(f"⚠️  {filename}: 包含 {len(outliers)} 个温度异常值")
            
        except Exception as e:
            issues.append(f"❌ {filename}: 读取失败 - {e}")
    
    if len(issues) == 0:
        print("✓ 所有数据文件质量良好")
    else:
        print(f"发现 {len(issues)} 个问题:\n")
        for issue in issues:
            print(f"  {issue}")
    
    print()


def main():
    """主函数"""
    print("=" * 70)
    print("电池热失控数据预处理和分析")
    print("=" * 70)
    print()
    
    # 检查数据目录
    if not os.path.exists(DATA_DIR):
        print(f"错误: 数据目录不存在: {DATA_DIR}")
        return
    
    # 生成报告
    df_report = generate_data_report()
    
    # 检查数据质量
    check_data_quality()
    
    print("=" * 70)
    print("分析完成！")
    print("=" * 70)
    print(f"\n报告已保存到: {REPORT_DIR}/")


if __name__ == "__main__":
    main()
