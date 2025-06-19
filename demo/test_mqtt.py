import json
import base64
import time
import os
import wave
from io import BytesIO
import paho.mqtt.client as mqtt
import numpy as np
import soundfile as sf
from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
import logging
import re
from datetime import datetime, timezone

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# === 配置区域 ===
# MQTT配置
MQTT_CONFIG = {
    "broker": "mqtt.iksns.net",
    "port": 1883,
    "topic": "smart/mattress/data",
    "username": "iotua",
    "password": "Vhwwhfy48vqHv6"
}

# TTS配置
REF_AUDIO_PATH = r"C:\Users\wx\Desktop\All\GPT-SoVIT\data\slicer_opt\1.wav"
TTS_CONFIG_PATH = "GPT_SoVITS/configs/tts_infer.yaml"

# 保存目录
SAVE_DIR = r"C:\Users\wx\Desktop\All\demo\saved_audio"
# === 配置结束 ===

def initialize_tts():
    """初始化TTS语音合成"""
    logger.info("初始化TTS语音合成...")
    tts_config = TTS_Config(TTS_CONFIG_PATH)
    tts_pipeline = TTS(tts_config)
    return tts_config, tts_pipeline

def generate_audio(text, tts_pipeline, tts_config):
    """生成音频并返回PCM数据"""
    logger.info(f"开始语音合成: {text[:20]}...")
    
    # 分割句子
    segments = re.split(r'(?<=[。？！\n])', text)
    pcm_accum = []
    
    for seg in segments:
        seg = seg.strip()
        if not seg: 
            continue
            
        tts_req = {
            "text": seg,
            "text_lang": "zh",
            "ref_audio_path": REF_AUDIO_PATH,
            "aux_ref_audio_paths": [],
            "prompt_text": "",
            "prompt_lang": "zh",
            "top_k": 5,
            "top_p": 1.0,
            "temperature": 1.0,
            "text_split_method": "cut5",
            "batch_size": 1,
            "batch_threshold": 0.75,
            "split_bucket": True,
            "return_fragment": False,
            "speed_factor": 1.0,
            "streaming_mode": False,
            "seed": 20250603,
            "parallel_infer": True,
            "repetition_penalty": 1.35
        }
        
        for sr, audio in tts_pipeline.run(tts_req):
            pcm_accum.append(audio)
    
    # 合并音频
    if pcm_accum:
        full_pcm = np.concatenate(pcm_accum, axis=0)
        return full_pcm
    
    logger.warning("未生成音频数据")
    return None

def pcm_to_wav_bytes(pcm_data, sampling_rate):
    """将PCM数据转换为WAV字节"""
    buf = BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit audio
        wf.setframerate(sampling_rate)
        wf.writeframes(pcm_data.tobytes())
    
    return buf.getvalue()

def publish_qa_mqtt(qa_data):
    """通过MQTT发布单个QA数据"""
    logger.info("准备通过MQTT发送QA数据...")
    
    # 连接MQTT
    client = mqtt.Client()
    client.username_pw_set(MQTT_CONFIG["username"], MQTT_CONFIG["password"])
    
    try:
        logger.info(f"连接MQTT代理: {MQTT_CONFIG['broker']}:{MQTT_CONFIG['port']}")
        client.connect(MQTT_CONFIG["broker"], MQTT_CONFIG["port"], 60)
        client.loop_start()
        
        # 发送单个QA项
        logger.info(f"发布到主题: {MQTT_CONFIG['topic']}")
        payload = json.dumps(qa_data)
        result = client.publish(MQTT_CONFIG["topic"], payload)
        result.wait_for_publish()
        
        client.loop_stop()
        client.disconnect()
        
        logger.info(f"MQTT发布成功! 消息ID: {result.mid}")
        return True
    except Exception as e:
        logger.error(f"MQTT发布失败: {e}")
        return False

def test_send_qa_items():
    """测试发送多个QA项"""
    # 确保保存目录存在
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # 初始化TTS
    tts_config, tts_pipeline = initialize_tts()
    
    # 创建QA项列表
    qa_items = []
    
    # QA 1
    qa_id = "QA001"
    question = "刘维是小处男"
    answer = "刘维是小处男"
    
    # 生成音频
    audio_pcm = generate_audio(answer, tts_pipeline, tts_config)
    if audio_pcm is not None:
        # 转换为WAV字节
        wav_bytes = pcm_to_wav_bytes(audio_pcm, tts_config.sampling_rate)
        
        # 保存音频文件用于调试
        audio_path = os.path.join(SAVE_DIR, f"{qa_id}.wav")
        with open(audio_path, "wb") as f:
            f.write(wav_bytes)
        logger.info(f"音频已保存到: {audio_path}")
        
        # 创建QA数据
        qa_data = {
            "time": datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z'),
            "id": qa_id,
            "question": question,
            "answer": answer,
            "audio": base64.b64encode(wav_bytes).decode('utf-8')
        }
        
        # 添加到列表
        qa_items.append(qa_data)
        
        # 保存JSON文件
        json_path = os.path.join(SAVE_DIR, f"{qa_id}.json")
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(qa_data, jf, ensure_ascii=False, indent=2)
        logger.info(f"已保存QA数据: {json_path}")
        
        # 通过MQTT发送
        success = publish_qa_mqtt(qa_data)
        if success:
            logger.info(f"QA项 {qa_id} 发送成功")
        else:
            logger.error(f"QA项 {qa_id} 发送失败")
    
    # QA 2
    qa_id = "QA002"
    question = "舒文涛？"
    answer = "舒文涛是爱慕"
    
    # 生成音频
    audio_pcm = generate_audio(answer, tts_pipeline, tts_config)
    if audio_pcm is not None:
        # 转换为WAV字节
        wav_bytes = pcm_to_wav_bytes(audio_pcm, tts_config.sampling_rate)
        
        # 保存音频文件用于调试
        audio_path = os.path.join(SAVE_DIR, f"{qa_id}.wav")
        with open(audio_path, "wb") as f:
            f.write(wav_bytes)
        logger.info(f"音频已保存到: {audio_path}")
        
        # 创建QA数据
        qa_data = {
            "time": datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z'),
            "id": qa_id,
            "question": question,
            "answer": answer,
            "audio": base64.b64encode(wav_bytes).decode('utf-8')
        }
        
        # 添加到列表
        qa_items.append(qa_data)
        
        # 保存JSON文件
        json_path = os.path.join(SAVE_DIR, f"{qa_id}.json")
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(qa_data, jf, ensure_ascii=False, indent=2)
        logger.info(f"已保存QA数据: {json_path}")
        
        # 通过MQTT发送
        success = publish_qa_mqtt(qa_data)
        if success:
            logger.info(f"QA项 {qa_id} 发送成功")
        else:
            logger.error(f"QA项 {qa_id} 发送失败")
    
    # 保存所有QA项的JSON文件
    json_path = os.path.join(SAVE_DIR, f"qa_items_{int(time.time())}.json")
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(qa_items, jf, ensure_ascii=False, indent=2)
    logger.info(f"已保存所有QA项: {json_path}")

if __name__ == "__main__":
    test_send_qa_items()