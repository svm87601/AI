#!/usr/bin/env python3
# batch_tts_faq_db.py

import os
import re
import sys
import json
import pymysql
import numpy as np
import soundfile as sf
import logging
from tqdm import tqdm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("tts_batch.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 将 TTS 包路径加入 PYTHONPATH
sys.path.append(r"C:\Users\wx\Desktop\All\demo")
sys.path.append(r"C:\Users\wx\Desktop\All\demo\GPT_SoVITS")

from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config

# —— 配置区域 —— 
# 数据库配置
db_config = {
    "host": "localhost",
    "user": "root",
    "password": "123456",
    "database": "faq_database",
    "charset": "utf8mb4",
    "cursorclass": pymysql.cursors.DictCursor
}

# 输出目录
OUTPUT_DIR      = r"C:\Users\wx\Desktop\All\demo\Audio_data"
REF_AUDIO       = r"C:\Users\wx\Desktop\All\GPT-SoVIT\data\slicer_opt\1.wav"
TTS_CONFIG_PATH = r"GPT_SoVITS/configs/tts_infer.yaml"

# TTS 固定 seed
FIXED_SEED      = 20250603

# 其他 TTS 参数
TEXT_LANG          = "zh"
PROMPT_LANG        = "zh"
TOP_K              = 5
TOP_P              = 1.0
TEMPERATURE        = 1.0
TEXT_SPLIT_METHOD  = "cut0"  # 更改为更简单的分割方法
BATCH_SIZE         = 1
BATCH_THRESHOLD    = 0.75
SPLIT_BUCKET       = True
RETURN_FRAGMENT    = False
SPEED_FACTOR       = 1.0
STREAMING_MODE     = False
PARALLEL_INFER     = True
REPETITION_PENALTY = 1.35
# —— 配置结束 —— 

def normalize_special_chars(text):
    """规范化文本中的特殊符号"""
    if not text:
        return ""
    
    # 温度符号处理
    text = re.sub(r'(\d+)\s?°C', r'\1摄氏度', text)
    text = re.sub(r'(\d+)\s?℃', r'\1摄氏度', text)
    
    # 其他常见特殊符号处理
    replacements = {
        '°': '度',
        '℉': '华氏度',
        '–': '到',  # 短破折号
        '—': '到',  # 长破折号
        '×': '乘',
        '÷': '除',
        '±': '加减'
    }
    
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    
    # 移除其他非常规字符
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s，。！？；：（）【】、\"\'-]', '', text)
    
    # 替换连续空格
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def load_faq_from_db():
    """
    从 MySQL 数据库中加载 FAQ 问答对
    要求表结构包含字段：id（主键）、question、answer
    """
    try:
        conn = pymysql.connect(**db_config)
        with conn.cursor() as cursor:
            cursor.execute("SELECT id, question, answer FROM faq_data")
            rows = cursor.fetchall()
            if not rows:
                logger.warning("FAQ 表中没有数据")
                return []
            logger.info(f"从数据库加载到 {len(rows)} 条 FAQ 记录")
            return rows
    except Exception as e:
        logger.error(f"数据库连接失败: {e}")
        return []
    finally:
        if 'conn' in locals():
            conn.close()

def robust_text_preprocessing(text):
    """强健的文本预处理函数"""
    if not text:
        return ""
    
    # 1. 规范化特殊符号
    text = normalize_special_chars(text)
    
    # 2. 移除HTML标签
    text = re.sub(r'<[^>]+>', '', text)
    
    # 3. 替换连续空格和换行符
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 4. 移除纯数字的文本（如果整个文本只有数字）
    if re.fullmatch(r'[\d\s.,]+', text):
        logger.warning(f"文本仅包含数字: {text}")
        return ""  # 返回空字符串，后续会跳过
    
    # 5. 确保文本有足够的字符
    if len(text) < 3:
        logger.warning(f"文本过短: {text}")
        return ""
    
    return text

def is_valid_chinese_text(text):
    """检查文本是否包含有效的中文字符"""
    # 检查是否包含中文字符
    if re.search(r'[\u4e00-\u9fff]', text):
        return True
    
    # 检查是否包含英文单词
    if re.search(r'\b[a-zA-Z]+\b', text):
        return True
    
    return False

def initialize_tts():
    """初始化TTS引擎，带错误处理"""
    try:
        logger.info("正在初始化TTS引擎...")
        tts = TTS(TTS_Config(TTS_CONFIG_PATH))
        logger.info("TTS引擎初始化成功")
        return tts
    except Exception as e:
        logger.error(f"TTS初始化失败: {e}")
        return None

def synthesize_and_save(faqs, out_dir, ref_audio):
    os.makedirs(out_dir, exist_ok=True)
    tts = initialize_tts()
    if not tts:
        logger.error("无法初始化TTS引擎，退出")
        return {}
    
    mapping = {}
    failed_items = []
    batch_size = 20  # 每批处理20个
    
    # 分批处理避免内存问题
    for batch_start in tqdm(range(0, len(faqs), batch_size), desc="处理批次"):
        batch = faqs[batch_start:batch_start + batch_size]
        
        for item in batch:
            idx = item["id"]
            raw_text = item["answer"].strip()
            
            # 1. 检查原始文本
            if not raw_text:
                logger.warning(f"ID {idx} 文本为空，跳过")
                failed_items.append({"id": idx, "reason": "空文本", "text": raw_text})
                continue
            
            # 2. 预处理文本
            text = robust_text_preprocessing(raw_text)
            if not text:
                logger.warning(f"ID {idx} 预处理后文本为空，跳过")
                failed_items.append({"id": idx, "reason": "预处理后空文本", "text": raw_text})
                continue
                
            # 3. 验证文本是否有效
            if not is_valid_chinese_text(text):
                logger.warning(f"ID {idx} 文本不包含有效字符: {text}")
                failed_items.append({"id": idx, "reason": "无效字符", "text": text})
                continue
            
            # 特殊调试信息（针对第210条）
            if idx == 210:
                logger.info(f"⚙️ 处理 ID 210 特殊记录")
                logger.info(f"  原始文本: {raw_text}")
                logger.info(f"  处理后文本: {text}")
            
            logger.info(f"处理 ID {idx} (长度: {len(text)} 字符)")
            
            # 4. 准备TTS参数
            tts_req = {
                "text":              text,
                "text_lang":         TEXT_LANG,
                "ref_audio_path":    ref_audio,
                "aux_ref_audio_paths": [],
                "prompt_text":       "",
                "prompt_lang":       PROMPT_LANG,
                "top_k":             TOP_K,
                "top_p":             TOP_P,
                "temperature":       TEMPERATURE,
                "text_split_method": TEXT_SPLIT_METHOD,
                "batch_size":        BATCH_SIZE,
                "batch_threshold":   BATCH_THRESHOLD,
                "split_bucket":      SPLIT_BUCKET,
                "return_fragment":   RETURN_FRAGMENT,
                "speed_factor":      SPEED_FACTOR,
                "streaming_mode":    STREAMING_MODE,
                "seed":              FIXED_SEED,
                "parallel_infer":    PARALLEL_INFER,
                "repetition_penalty": REPETITION_PENALTY
            }
            
            # 5. 尝试生成音频
            try:
                generator = tts.run(tts_req)
                chunks, sr0 = [], None
                
                # 收集音频片段
                for sr, audio in generator:
                    if sr0 is None:
                        sr0 = sr
                    chunks.append(audio)
                
                if not chunks:
                    logger.warning(f"ID {idx} 没有生成任何音频")
                    failed_items.append({"id": idx, "reason": "无音频生成", "text": text})
                    continue
                
                # 合并并保存音频
                pcm = np.concatenate(chunks, axis=0)
                wav_path = os.path.join(out_dir, f"{idx}.wav")
                sf.write(wav_path, pcm, sr0, format="wav")
                logger.info(f"保存音频到 {wav_path}")
                
                mapping[idx] = {
                    "question": item["question"],
                    "answer":   raw_text,  # 保存原始文本
                    "audio":    wav_path
                }
                
            except Exception as e:
                logger.error(f"ID {idx} 合成失败: {str(e)}")
                # 捕获详细错误信息
                import traceback
                error_details = traceback.format_exc()
                logger.debug(f"错误详情:\n{error_details}")
                
                failed_items.append({
                    "id": idx, 
                    "reason": str(e), 
                    "text": text,
                    "raw_text": raw_text
                })
    
    # 保存失败记录
    if failed_items:
        failed_path = os.path.join(out_dir, "failed_items.json")
        with open(failed_path, "w", encoding="utf-8") as f:
            json.dump(failed_items, f, ensure_ascii=False, indent=2)
        logger.warning(f"有 {len(failed_items)} 个项目失败，详情见 {failed_path}")
    
    return mapping

if __name__ == "__main__":
    logger.info("1) 从数据库加载 FAQ 数据 ...")
    faqs = load_faq_from_db()
    
    if not faqs:
        logger.error("没有加载到FAQ数据，退出")
        sys.exit(1)
    
    logger.info("2) 合成 TTS 并保存 ...")
    mp = synthesize_and_save(faqs, OUTPUT_DIR, REF_AUDIO)
    
    map_path = os.path.join(OUTPUT_DIR, "faq_audio_map.json")
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(mp, f, ensure_ascii=False, indent=2)
    
    logger.info(f"✅ 完成。音频文件存放于: {OUTPUT_DIR}")
    logger.info(f"✅ 映射文件: {map_path}")