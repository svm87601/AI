# asr/test_asr.py
import time
import wave
import os
import sys
import io
import uuid
import logging
from typing import Optional, Tuple, List
import numpy as np

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 移除以下行：
# from config.logger import setup_logging

try:
    from funasr import AutoModel
    from funasr.utils.postprocess_utils import rich_transcription_postprocess
except ImportError:
    logger.error("FunASR 库未安装。请使用: pip install funasr")
    raise

TAG = __name__

class CaptureOutput:
    def __enter__(self):
        self._output = io.StringIO()
        self._orig = sys.stdout
        sys.stdout = self._output

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._orig
        out = self._output.getvalue().strip()
        if out:
            logger.info(f"模型输出: {out}")
        self._output.close()

class ASRProvider:
    def __init__(self, delete_audio_file: bool):
        self.model_dir = r"C:\Users\wx\Desktop\All\ASR-VAD\models\SenseVoiceSmall"
        self.output_dir = r"C:\Users\wx\Desktop\All\ASR-VAD\tmp"
        self.delete_audio_file = delete_audio_file

        os.makedirs(self.output_dir, exist_ok=True)

        with CaptureOutput():
            # 仅初始化 ASR 模型，不带 VAD
            self.model = AutoModel(
                model=self.model_dir,
                trust_remote_code=True,
                device="cpu",
                disable_update=True,
            )

    def save_audio_to_file(self, audio_data: bytes, session_id: str) -> str:
        """将音频数据保存为WAV文件"""
        file_name = f"asr_{session_id}_{uuid.uuid4()}.wav"
        path = os.path.join(self.output_dir, file_name)
        
        # 直接写入接收的音频数据
        with open(path, "wb") as f:
            f.write(audio_data)
            
        return path

    def asr_file(self, wav_path: str) -> str:
        """对单个 WAV 文件进行 ASR，返回文本"""
        start = time.time()
        res = self.model.generate(
            input=wav_path,
            language="auto",
            use_itn=True,
            batch_size_s=60,
        )
        text = rich_transcription_postprocess(res[0]["text"])
        logger.info(f"ASR 耗时: {time.time()-start:.3f}s | 文本: {text}")
        return text

    async def speech_to_text(self, audio_data: bytes, session_id: str) -> Tuple[Optional[str], Optional[str]]:
        file_path = None
        try:
            file_path = self.save_audio_to_file(audio_data, session_id)
            text = self.asr_file(file_path)
            return text, file_path
        except Exception as e:
            logger.error(f"ASR 失败: {e}", exc_info=True)
            return "", None
        finally:
            if self.delete_audio_file and file_path and os.path.exists(file_path):
                try: 
                    os.remove(file_path)
                except Exception as e:
                    logger.error(f"删除临时文件失败: {e}")