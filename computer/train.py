import json
import paho.mqtt.client as mqtt

# ====== 配置信息 (需与 ESP32 代码一致) ======
MQTT_BROKER = "bemfa.com"
MQTT_PORT = 9501
# 巴法云使用私钥作为 ClientID
MQTT_CLIENT_ID = "84810b9b5f5245fdbc1e1738837f27a9" 
# 订阅的主题
MQTT_TOPIC = "sensor"

def on_connect(client, userdata, flags, rc):
    """连接成功的回调函数"""
    if rc == 0:
        print(f"成功连接到巴法云! 正在订阅主题: {MQTT_TOPIC}...")
        client.subscribe(MQTT_TOPIC)
    else:
        print(f"连接失败，错误码: {rc}")

def on_message(client, userdata, msg):
    """收到消息的回调函数"""
    try:
        # 解析收到的 JSON 字符串
        payload = msg.payload.decode('utf-8')
        data = json.loads(payload)
        
        # 打印解析后的数据
        print("\n--- 收到传感器数据 ---")
        print(f"时间戳: {data.get('timestamp_ms')} ms")
        print(f"环境温度: {data.get('temp_ambient_c')} °C")
        print(f"内部温度: {data.get('temp_internal_c')} °C")
        print(f"环境湿度: {data.get('humidity_percent')} %")
        print(f"MQ2(烟雾): {data.get('mq2_ppm')} ppm")
        print(f"MQ4(甲烷): {data.get('mq4_ppm')} ppm")
        print(f"MQ8(氢气): {data.get('mq8_ppm')} ppm")
        print(f"MQ7(一氧化碳): {data.get('mq7_ppm')} ppm")
        print(f"VOC 指数: {data.get('voc_index')}")
        print(f"系统状态: {data.get('status')}")
        
    except json.JSONDecodeError:
        print(f"收到非 JSON 格式消息: {msg.payload.decode('utf-8')}")
    except Exception as e:
        print(f"处理消息时出错: {e}")

def start_mqtt_listener():
    """启动 MQTT 监听主函数"""
    # 适配 paho-mqtt 2.0+ 版本，必须指定 CallbackAPIVersion
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, client_id=MQTT_CLIENT_ID)
    
    # 设置回调函数
    client.on_connect = on_connect
    client.on_message = on_message

    print(f"正在尝试连接到 {MQTT_BROKER}...")
    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        # loop_forever() 会阻塞进程，持续处理网络循环、自动重连等
        client.loop_forever()
    except KeyboardInterrupt:
        print("\n程序已手动停止")
        client.disconnect()
    except Exception as e:
        print(f"发生异常: {e}")

if __name__ == "__main__":
    start_mqtt_listener()