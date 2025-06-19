import serial
import time
import json
import base64
import logging
import paho.mqtt.client as mqtt
from picamera2 import Picamera2
from typing import Optional, Dict, Any

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sensor_data.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SensorDataCollector:
    def __init__(self, config_file: str = 'config.json'):
        """初始化传感器数据采集器"""
        self.config = self._load_config(config_file)
        self.pico_serial = None
        self.mqtt_client = None
        self.picam2 = None
        self._setup_connections()
        self._setup_camera()
    
    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"配置文件 {config_file} 不存在")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"配置文件格式错误: {e}")
            raise
    
    def _setup_connections(self):
        """建立连接"""
        # 初始化串口
        try:
            serial_config = self.config['serial']
            self.pico_serial = serial.Serial(
                serial_config['port'],
                serial_config['baudrate'],
                timeout=serial_config['timeout']
            )
            logger.info("串口连接成功")
        except Exception as e:
            logger.error(f"串口连接失败: {e}")
            raise
        
        # 初始化MQTT客户端
        try:
            mqtt_config = self.config['mqtt']
            self.mqtt_client = mqtt.Client()
            self.mqtt_client.username_pw_set(
                mqtt_config['username'],
                mqtt_config['password']
            )
            self.mqtt_client.connect(
                mqtt_config['broker'],
                mqtt_config['port'],
                60
            )
            logger.info("MQTT连接成功")
        except Exception as e:
            logger.error(f"MQTT连接失败: {e}")
            raise
    
    def _setup_camera(self):
        """初始化摄像头（只在程序启动时执行一次）"""
        try:
            camera_config = self.config['camera']
            self.picam2 = Picamera2()
            self.picam2.configure(self.picam2.create_still_configuration(
                main={'size': (camera_config['width'], camera_config['height'])}
            ))
            self.picam2.start()
            logger.info("摄像头初始化成功")
        except Exception as e:
            logger.error(f"摄像头初始化失败: {e}")
            self.picam2 = None
    
    def read_sensor_data(self, timeout: int = 2) -> str:
        """读取传感器数据"""
        if not self.pico_serial:
            return ""
        
        try:
            # 清空输入缓冲区
            self.pico_serial.reset_input_buffer()
            
            data = b""
            start_time = time.time()
            
            while True:
                if self.pico_serial.in_waiting:
                    data += self.pico_serial.read(self.pico_serial.in_waiting)
                    if b'\r\n' in data:
                        break
                
                if time.time() - start_time > timeout:
                    logger.warning("读取传感器数据超时")
                    break
                
                time.sleep(0.1)
            
            decoded = data.decode('utf-8').strip()
            result = decoded.splitlines()[0] if decoded else ""
            logger.info(f"传感器数据: {result}")
            return result
            
        except Exception as e:
            logger.error(f"读取传感器数据失败: {e}")
            return ""
    
    def capture_image(self) -> str:
        """拍摄照片并返回Base64编码"""
        if not self.picam2:
            logger.error("摄像头未初始化，无法拍照")
            return ""
        
        try:
            image_path = "capture.jpg"
            self.picam2.capture_file(image_path)
            
            with open(image_path, "rb") as img_file:
                encoded_string = base64.b64encode(img_file.read()).decode('utf-8')
            
            logger.info("图片拍摄成功")
            return encoded_string
            
        except Exception as e:
            logger.error(f"拍照失败: {e}")
            return ""
    
    def save_to_js_file(self, data: Dict[str, Any], filename: str = "sensorData.js") -> bool:
        """保存数据到JavaScript文件"""
        try:
            with open(filename, "w", encoding='utf-8') as f:
                f.write("var sensorData = " + json.dumps(data, indent=2, ensure_ascii=False) + ";")
            logger.info(f"数据已保存到 {filename}")
            return True
        except Exception as e:
            logger.error(f"保存文件失败: {e}")
            return False
    
    def publish_mqtt(self, data: Dict[str, Any]) -> bool:
        """通过MQTT发布数据"""
        if not self.mqtt_client:
            return False
        
        try:
            topic = self.config['mqtt']['topic']
            self.mqtt_client.publish(topic, json.dumps(data, ensure_ascii=False))
            logger.info("MQTT数据发布成功")
            return True
        except Exception as e:
            logger.error(f"MQTT发布失败: {e}")
            return False
    
    def collect_and_send_data(self):
        """采集并发送数据"""
        logger.info("开始数据采集")
        
        # 读取传感器数据
        soil_data = self.read_sensor_data()
        
        # 拍摄照片
        image_base64 = self.capture_image()
        
        # 组合数据
        data = {
            "timestamp": time.time(),
            "soil_data": soil_data,
            "image_base64": image_base64
        }
        
        # 保存到文件
        self.save_to_js_file(data)
        
        # 发布到MQTT
        self.publish_mqtt(data)
        
        logger.info("数据采集完成")
    
    def run(self):
        """主运行循环"""
        logger.info("传感器数据采集器启动")
        
        try:
            while True:
                self.collect_and_send_data()
                
                # 等待指定间隔
                interval = self.config.get('interval', 30)
                logger.info(f"等待 {interval} 秒后进行下次采集")
                time.sleep(interval)
                
        except KeyboardInterrupt:
            logger.info("检测到用户中断，正在关闭...")
        except Exception as e:
            logger.error(f"运行时错误: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """清理资源"""
        if self.picam2:
            try:
                self.picam2.stop()
                logger.info("摄像头已关闭")
            except Exception as e:
                logger.error(f"关闭摄像头时出错: {e}")
        
        if self.pico_serial:
            try:
                self.pico_serial.close()
                logger.info("串口已关闭")
            except Exception as e:
                logger.error(f"关闭串口时出错: {e}")
        
        if self.mqtt_client:
            try:
                self.mqtt_client.disconnect()
                logger.info("MQTT连接已断开")
            except Exception as e:
                logger.error(f"断开MQTT连接时出错: {e}")

def main():
    """主函数"""
    try:
        collector = SensorDataCollector()
        collector.run()
    except Exception as e:
        logger.error(f"程序启动失败: {e}")

if __name__ == "__main__":
    main()