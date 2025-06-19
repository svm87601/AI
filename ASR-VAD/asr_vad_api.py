# server.py
import os
import io
import asyncio
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import PlainTextResponse
import torch
import torchaudio
from asr.test_asr import ASRProvider
import uvicorn
app = FastAPI()

# 本地 Silero VAD 仓库路径
VAD_REPO = r"C:\Users\wx\Desktop\All\ASR-VAD\models\snakers4_silero-vad"

# 加载 Silero VAD
vad_model, utils = torch.hub.load(
    repo_or_dir=VAD_REPO,
    model='silero_vad',
    source='local',
)
(get_speech_timestamps, _, _, _, _) = utils

# 初始化 ASR Provider（全局复用）
asr_provider = ASRProvider(delete_audio_file=False)

@app.post("/asr/", response_class=PlainTextResponse)
async def asr_endpoint(file: UploadFile = File(...), threshold: float = 0.5, min_silence_ms: int = 700):
    """
    接收一个音频文件，做 VAD 切分 + ASR，返回每段文字，格式：
    [0] 0.00-1.23s: 你好
    [1] 1.50-2.34s: 世界
    """
    # 1. 读取上传的音频
    contents = await file.read()
    wav, sr = torchaudio.load(io.BytesIO(contents), format=file.filename.split('.')[-1])
    # 2. 单声道
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    # 3. 重采样
    target_sr = 16000
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=target_sr)
        sr = target_sr
    # 4. VAD 分段
    raw_segments = get_speech_timestamps(
        wav, vad_model, sampling_rate=sr,
        threshold=threshold,
        min_silence_duration_ms=min_silence_ms
    )
    # 5. 合并小段
    merged = []
    i = 0
    while i < len(raw_segments):
        seg = raw_segments[i]
        start, end = seg['start'], seg['end']
        dur = (end - start) / sr
        while dur < 0.5 and i + 1 < len(raw_segments):
            i += 1
            end = raw_segments[i]['end']
            dur = (end - start) / sr
        merged.append({'start': start, 'end': end})
        i += 1

    # 6. 对每段调用 ASR
    results: List[str] = []
    for idx, seg in enumerate(merged):
        s, e = seg['start'], seg['end']
        chunk = wav[:, s:e]
        tmp_path = os.path.join(asr_provider.output_dir, f"seg_{idx}.wav")
        torchaudio.save(tmp_path, chunk, sr)
        text = asr_provider.asr_file(tmp_path)
        results.append(f"[{idx}] {s/sr:.2f}-{e/sr:.2f}s: {text}")

    return "\n".join(results)

if __name__ == "__main__":
    
    uvicorn.run("asr_vad_api:app", host="0.0.0.0", port=8001, reload=True)
