import os
import pandas as pd
import glob

def process_files():
    directory = 'computer/data'
    # 获取所有以007和008打头的csv文件
    patterns = [os.path.join(directory, '007*.csv'), os.path.join(directory, '008*.csv')]
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern))
    
    target_columns = ['Test_Time(s)', 'Temperature (C)_1', 'Temperature (C)_2']
    
    for file_path in files:
        print(f"正在处理: {file_path}")
        try:
            # 读取CSV文件
            df = pd.read_csv(file_path)
            
            # 检查目标列是否存在
            existing_cols = [col for col in target_columns if col in df.columns]
            if len(existing_cols) < len(target_columns):
                missing = set(target_columns) - set(existing_cols)
                print(f"警告: 文件 {file_path} 缺少列: {missing}")
            
            # 只保留存在的列
            df_filtered = df[existing_cols]
            
            # 保存回原文件
            df_filtered.to_csv(file_path, index=False)
            print(f"已完成: {file_path}")
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")

if __name__ == "__main__":
    process_files()