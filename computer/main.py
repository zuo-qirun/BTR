import json
import os
from collections import deque

import paho.mqtt.client as mqtt
import numpy as np
import pickle
from tensorflow import keras

# ====== 配置信息 (需与 ESP32 代码一致) ======
MQTT_BROKER = "bemfa.com"
MQTT_PORT = 9501
# 巴法云使用私钥作为 ClientID
MQTT_CLIENT_ID = "84810b9b5f5245fdbc1e1738837f27a9"
# 订阅的主题
MQTT_TOPIC = "sensor"
# 发布状态的主题
MQTT_STATUS_TOPIC = "statue"

# 模型目录（与训练脚本一致）
MODEL_DIR = "models"


def load_predictor(model_dir=MODEL_DIR):
    """加载 config、模型与 scaler，返回 (model, scaler, sequence_length)"""
    config_path = os.path.join(model_dir, "config.pkl")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"找不到配置文件: {config_path}")

    with open(config_path, 'rb') as f:
        config = pickle.load(f)

    model_path = config.get('model_path')
    scaler_path = config.get('scaler_path')
    sequence_length = config.get('sequence_length', 60)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到模型文件: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"找不到 scaler 文件: {scaler_path}")

    model = keras.models.load_model(model_path)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    print(f"已加载模型: {model_path}")
    print(f"已加载 scaler: {scaler_path}")

    return model, scaler, sequence_length


# 载入模型与 scaler，准备缓冲区
try:
    MODEL, SCALER, SEQ_LEN = load_predictor()
except Exception as e:
    print(f"加载模型失败: {e}")
    MODEL = None
    SCALER = None
    SEQ_LEN = 60

# 温度缓冲区（分别保存内部和外部温度）
# 注意：假设 MQTT 数据以固定间隔（约0.5秒）到达
# 如果实际间隔不均匀，需要添加时间戳检查和插值逻辑
temp_internal_buffer = deque(maxlen=SEQ_LEN)
temp_ambient_buffer = deque(maxlen=SEQ_LEN)
time_buffer = deque(maxlen=SEQ_LEN)  # 存储时间戳用于检查间隔


def on_connect(client, userdata, flags, rc):
    """连接成功的回调函数"""
    if rc == 0:
        print(f"成功连接到巴法云! 正在订阅主题: {MQTT_TOPIC}...")
        client.subscribe(MQTT_TOPIC)
    else:
        print(f"连接失败，错误码: {rc}")


def process_and_predict(esp32_status):
    """当缓冲区满时调用模型进行预测并打印概率"""
    if MODEL is None or SCALER is None:
        print("模型或 scaler 未加载，无法预测")
        return None

    if len(temp_internal_buffer) < SEQ_LEN or len(temp_ambient_buffer) < SEQ_LEN:
        return None

    # 预测内部温度
    seq_internal = np.array(list(temp_internal_buffer)).reshape(-1, 1)
    seq_internal_scaled = SCALER.transform(seq_internal)
    seq_internal_scaled = seq_internal_scaled.reshape(1, SEQ_LEN, 1)
    prob_internal = float(MODEL.predict(seq_internal_scaled, verbose=0)[0][0])
    level_internal = "高风险" if prob_internal >= 0.5 else "低风险"

    # 预测外部温度
    seq_ambient = np.array(list(temp_ambient_buffer)).reshape(-1, 1)
    seq_ambient_scaled = SCALER.transform(seq_ambient)
    seq_ambient_scaled = seq_ambient_scaled.reshape(1, SEQ_LEN, 1)
    prob_ambient = float(MODEL.predict(seq_ambient_scaled, verbose=0)[0][0])
    level_ambient = "高风险" if prob_ambient >= 0.5 else "低风险"

    print(f"\n=== 预测结果 ===")
    print(f"内部温度热失控概率: {prob_internal:.2%} -> {level_internal}")
    print(f"外部温度热失控概率: {prob_ambient:.2%} -> {level_ambient}")
    
    # 判断AI预测状态
    ai_is_danger = prob_internal >= 0.5 or prob_ambient >= 0.5
    if ai_is_danger:
        print(f"⚠️ AI预测状态: 危险")
    else:
        print(f"✓ AI预测状态: 正常")
    
    print(f"ESP32状态: {esp32_status}")
    print(f"================\n")
    
    # 双向纠正逻辑：
    # 1. ESP32说DANGER，但AI预测正常 → 发送normal
    # 2. ESP32说NORMAL/WARNING，但AI预测危险 → 发送danger
    if esp32_status == "DANGER" and not ai_is_danger:
        return "normal"
    elif esp32_status in ["NORMAL", "WARNING"] and ai_is_danger:
        return "danger"
    
    return None


def on_message(client, userdata, msg):
    """收到消息的回调函数，解析并用于预测"""
    try:
        payload = msg.payload.decode('utf-8')
        data = json.loads(payload)

        ts = data.get('timestamp_ms')
        temp_ambient = data.get('temp_ambient_c')
        temp_internal = data.get('temp_internal_c')
        esp32_status = data.get('status', 'UNKNOWN')

        # 检查是否有有效的温度数据
        if temp_internal is None or temp_ambient is None:
            print("收到数据但缺少温度字段，跳过")
            return

        # 添加到各自的缓冲区
        temp_internal_buffer.append(float(temp_internal))
        temp_ambient_buffer.append(float(temp_ambient))
        if ts is not None:
            time_buffer.append(ts / 1000.0)  # 转换为秒

        print("\n--- 收到传感器数据 ---")
        print(f"时间戳: {ts} ms")
        print(f"内部温度: {temp_internal} °C | 环境温度: {temp_ambient} °C")
        print(f"缓冲区: {len(temp_internal_buffer)}/{SEQ_LEN}")
        
        # 检查时间间隔是否均匀（可选）
        if len(time_buffer) >= 2:
            avg_interval = (time_buffer[-1] - time_buffer[0]) / (len(time_buffer) - 1)
            print(f"平均时间间隔: {avg_interval:.2f}秒")

        # 当缓冲区填满时进行预测
        if len(temp_internal_buffer) == SEQ_LEN and len(temp_ambient_buffer) == SEQ_LEN:
            status = process_and_predict(esp32_status)
            if status:
                # 发送状态到 statue 主题
                client.publish(MQTT_STATUS_TOPIC, status)
                print(f"✉️ 已发送纠正状态到 {MQTT_STATUS_TOPIC}: {status}")
            else:
                print(f"ℹ️ ESP32与AI预测一致，无需纠正")

    except json.JSONDecodeError:
        print(f"收到非 JSON 格式消息: {msg.payload.decode('utf-8')}")
    except Exception as e:
        print(f"处理消息时出错: {e}")


def start_mqtt_listener():
    """启动 MQTT 监听主函数"""
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, client_id=MQTT_CLIENT_ID)
    client.on_connect = on_connect
    client.on_message = on_message

    print(f"正在尝试连接到 {MQTT_BROKER}...")
    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.loop_forever()
    except KeyboardInterrupt:
        print("\n程序已手动停止")
        client.disconnect()
    except Exception as e:
        print(f"发生异常: {e}")


if __name__ == "__main__":
    start_mqtt_listener()
