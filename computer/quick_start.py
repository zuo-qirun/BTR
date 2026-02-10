"""
快速开始脚本 - 电池热失控预测系统
自动检查环境、训练模型（如果需要）并运行演示
"""

import os
import sys

def check_dependencies():
    """检查依赖是否安装"""
    print("检查依赖...")
    required_packages = [
        'numpy',
        'pandas',
        'sklearn',
        'tensorflow',
        'matplotlib'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} (未安装)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n缺少依赖包: {', '.join(missing_packages)}")
        print("请运行: pip install -r requirements.txt")
        return False
    
    print("✓ 所有依赖已安装\n")
    return True


def check_data():
    """检查数据文件"""
    print("检查数据文件...")
    data_dir = "data"
    
    if not os.path.exists(data_dir):
        print(f"  ✗ 数据目录不存在: {data_dir}")
        return False
    
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    if len(csv_files) == 0:
        print(f"  ✗ 数据目录中没有CSV文件")
        return False
    
    normal_files = [f for f in csv_files if '_exception' not in f]
    exception_files = [f for f in csv_files if '_exception' in f]
    
    print(f"  ✓ 找到 {len(csv_files)} 个数据文件")
    print(f"    - 正常数据: {len(normal_files)} 个")
    print(f"    - 异常数据: {len(exception_files)} 个")
    
    if len(exception_files) == 0:
        print("  ⚠️  警告: 没有异常数据文件，模型可能无法学习热失控模式")
    
    print()
    return True


def check_model():
    """检查模型是否存在"""
    print("检查模型文件...")
    model_dir = "models"
    
    if not os.path.exists(model_dir):
        print(f"  ✗ 模型目录不存在: {model_dir}")
        return False
    
    required_files = ['thermal_runaway_model.h5', 'scaler.pkl', 'config.pkl']
    missing_files = []
    
    for file in required_files:
        file_path = os.path.join(model_dir, file)
        if os.path.exists(file_path):
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} (不存在)")
            missing_files.append(file)
    
    if missing_files:
        print("\n模型文件不完整，需要训练模型")
        return False
    
    print("✓ 模型文件完整\n")
    return True


def train_model():
    """训练模型"""
    print("=" * 70)
    print("开始训练模型...")
    print("=" * 70)
    print()
    
    try:
        from train_model import main
        main()
        return True
    except Exception as e:
        print(f"\n训练失败: {e}")
        return False


def run_demo():
    """运行演示"""
    print("\n" + "=" * 70)
    print("运行实时预测演示...")
    print("=" * 70)
    print()
    
    try:
        from predict_realtime import simulate_real_time_prediction
        simulate_real_time_prediction()
        return True
    except Exception as e:
        print(f"\n演示失败: {e}")
        return False


def main():
    """主函数"""
    print("=" * 70)
    print("电池热失控预测系统 - 快速开始")
    print("=" * 70)
    print()
    
    # 1. 检查依赖
    if not check_dependencies():
        print("\n请先安装依赖后再运行此脚本")
        sys.exit(1)
    
    # 2. 检查数据
    if not check_data():
        print("\n请确保 data/ 目录中有训练数据")
        sys.exit(1)
    
    # 3. 检查模型
    model_exists = check_model()
    
    if not model_exists:
        print("需要训练模型")
        response = input("是否现在训练模型？(y/n): ").strip().lower()
        
        if response == 'y':
            if not train_model():
                print("\n模型训练失败")
                sys.exit(1)
        else:
            print("\n请先运行 train_model.py 训练模型")
            sys.exit(1)
    
    # 4. 运行演示
    print("\n准备运行实时预测演示...")
    response = input("按 Enter 继续，或输入 'q' 退出: ").strip().lower()
    
    if response != 'q':
        run_demo()
    
    print("\n" + "=" * 70)
    print("快速开始完成！")
    print("=" * 70)
    print("\n下一步:")
    print("  1. 查看 README_MODEL.md 了解详细使用方法")
    print("  2. 运行 python predict_realtime.py --manual 手动输入测试")
    print("  3. 集成到你的实时监控系统中")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程序已停止")
        sys.exit(0)
