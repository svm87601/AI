#!/usr/bin/env python3
# client_grpc.py

import cv2
import dlib
import time
import threading
import grpc
import sys
import os
import signal
import numpy as np
import sounddevice as sd
import soundfile as sf
import requests
from io import BytesIO
import select
import uuid

# 导入 gRPC 自动生成的 stub
import chat_pb2
import chat_pb2_grpc

# ------------------- 配置区域 -------------------
# gRPC 服务地址
SERVER_GRPC_ADDR = "10.1.41.60:50051"   # 修改为你实际的 gRPC 地址
# ASR+VAD HTTP 服务地址
ASR_VAD_URL     = "http://10.1.41.60:8001/asr/"  # 修改为你实际的 ASR+VAD 接口

# 录音配置
RECORD_DURATION = 5    # 固定录音 5 秒，如果希望更灵活可以改为动态检测
SAMPLE_RATE     = 16000
CHANNELS        = 1

# 人脸检测与界面配置
FACE_DETECTION_INTERVAL = 0.5   
NO_PERSON_TIMEOUT       = 10          
USB_CAMERA_INDEX        = 1            
FRAME_WIDTH             = 640               
FRAME_HEIGHT            = 480

# ------------------------------------------------

def signal_handler(signum, frame):
    print(f"\n收到终止信号 {signum}，正在安全关闭...")
    os._exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

class ClassroomDevice:
    def __init__(self):
        # 摄像头初始化
        self.cap = None
        self._init_camera()

        self.detector = dlib.get_frontal_face_detector()
        self.last_detected_time = time.time()
        self.running = True

        # 会话状态
        self.conversation_active = False
        self.conversation_thread = None

        # gRPC channel + stub
        self.grpc_channel = grpc.insecure_channel(SERVER_GRPC_ADDR)
        self.stub = chat_pb2_grpc.ChatServiceStub(self.grpc_channel)

    def _init_camera(self):
        """初始化摄像头（带重试机制）"""
        max_retries = 3
        for i in range(max_retries):
            self.cap = cv2.VideoCapture(USB_CAMERA_INDEX, cv2.CAP_V4L2)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
                self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
                print("摄像头初始化成功")
                time.sleep(1)  # 摄像头预热
                return
            else:
                if self.cap is not None:
                    self.cap.release()
                print(f"摄像头初始化失败，第{i+1}次重试...")
                time.sleep(2)
        raise RuntimeError("无法打开 USB 摄像头")

    def start_conversation(self):
        """当检测到人脸时启动录音+对话线程"""
        if not self.conversation_active:
            self.conversation_active = True
            os.system("clear")
            self.conversation_thread = threading.Thread(target=self.conversation_loop)
            self.conversation_thread.daemon = True
            self.conversation_thread.start()

    def conversation_loop(self):
        """
        对话循环：自动录音 -> 上传 ASR+VAD 服务 -> 拿到文本 -> 调用 gRPC Chat
        当超过超时时间无人时自动结束对话
        """
        print("进入语音对话模式（说话即可，5秒后自动识别）:")
        while self.running:
            # 如果超过 NO_PERSON_TIMEOUT 未检测到人脸，则退出对话模式
            if time.time() - self.last_detected_time > NO_PERSON_TIMEOUT:
                print("\n超过10秒未检测到人脸，退出对话模式。")
                break

            # 1) 录制一段音频
            wav_data = self.record_audio(RECORD_DURATION)
            if wav_data is None:
                print("录音失败或音量过低，重新检测人脸。")
                break

            # 2) 上传给 ASR+VAD 服务，获取识别文本
            text = self.call_asr_service(wav_data)
            if not text:
                print("未识别出有效文本，重新检测人脸。")
                continue

            print(f"\n识别到文本: {text}")

            # 3) 调用 gRPC 发送识别到的文本，并处理流式返回
            self.call_grpc_and_play(text)

            # 对话结束后，如果仍有人脸继续循环，否则等待下一次检测
            print("\n如需继续对话，请保持人脸可见，系统会在 10s 后退出对话模式。")
        self.conversation_active = False
        os.system("clear")

    def record_audio(self, duration_s: float):
        """
        使用 sounddevice 录音，并返回 WAV 二进制数据
        录制 duration_s 秒钟后自动停止
        """
        try:
            print(f"开始录音 ({duration_s} 秒)...")
            recording = sd.rec(int(duration_s * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                               channels=CHANNELS, dtype='int16')
            sd.wait()  # 等待录音结束
            # 保存到 BytesIO
            buf = BytesIO()
            sf.write(buf, recording, SAMPLE_RATE, format='wav')
            wav_bytes = buf.getvalue()
            return wav_bytes
        except Exception as e:
            print(f"录音出错: {e}")
            return None

    def call_asr_service(self, wav_bytes: bytes):
        """
        将 WAV bytes 打包成 UploadFile 形式，POST 到 ASR+VAD 服务
        返回识别到的纯文本（去掉行号和时长）
        """
        try:
            files = {'file': ('audio.wav', wav_bytes, 'audio/wav')}
            response = requests.post(ASR_VAD_URL, files=files, timeout=30)
            if response.status_code != 200:
                print(f"ASR 服务返回错误: {response.status_code}")
                return ""
            result_text = response.text.strip()
            # ASR 返回格式类似： "[0] 0.00-1.23s: 你好\n[1] 1.50-2.34s: 世界"
            # 我们只保留冒号后面的文字，并用空格拼接
            lines = result_text.splitlines()
            pieces = []
            for ln in lines:
                parts = ln.split(':', 1)
                if len(parts) == 2:
                    pieces.append(parts[1].strip())
            return " ".join(pieces)
        except Exception as e:
            print(f"调用 ASR 服务失败: {e}")
            return ""

    def call_grpc_and_play(self, question: str):
        """
        使用 gRPC 调用 ChatService，将识别到的问题发给服务端，
        然后流式接收 TextChunk、AudioChunk 并分别打印/播放。
        """
        session_id = str(int(time.time())) + "_" + uuid.uuid4().hex[:6]
        request = chat_pb2.ChatRequest(session_id=session_id, question=question)

        try:
            # 发起 gRPC 流式调用
            responses = self.stub.Chat(request, timeout=300)
        except Exception as e:
            print(f"gRPC 调用失败: {e}")
            return

        # 定义播放音频时要缓冲的 BytesIO
        audio_buffer = BytesIO()
        is_streaming_audio = False

        print("模型回答（streaming）：", end="", flush=True)
        for resp in responses:
            # 文本分片
            if resp.HasField("text_chunk"):
                txt = resp.text_chunk.text
                print(txt, end="", flush=True)
                continue

            # 音频分片
            if resp.HasField("audio_chunk"):
                is_streaming_audio = True
                chunk = resp.audio_chunk.data
                if chunk:
                    audio_buffer.write(chunk)
                # 如果是最后一个音频分片，则播放
                if resp.audio_chunk.is_last:
                    # 将 buffer 定位回头，读 PCM 播放
                    audio_buffer.seek(0)
                    data, sr = sf.read(audio_buffer)
                    print("\n[开始播放 TTS 音频]")
                    sd.play(data, sr)
                    sd.wait()
                    # 重置 buffer，等待下次
                    audio_buffer = BytesIO()
                    is_streaming_audio = False
                continue

            # Done 标志
            if resp.HasField("done"):
                print("\n[会话结束]")
                break

        print("[gRPC 交互完成]")

    def run_detection(self):
        """人脸检测主循环，检测到人脸则触发对话模式"""
        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    self._reinit_camera()
                    continue

                current_time = time.time()
                # 每隔一段时间进行一次检测
                if current_time - self.last_detected_time > FACE_DETECTION_INTERVAL:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = self.detector(gray, 0)

                    if len(faces) > 0:
                        self.last_detected_time = current_time
                        if not self.conversation_active:
                            print(f"[检测] 发现 {len(faces)} 张人脸，启动语音识别对话")
                            self.start_conversation()

                    # 在画面上绘制人脸框
                    for face in faces:
                        x, y, w, h = face.left(), face.top(), face.width(), face.height()
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                cv2.putText(frame, f"状态: {'对话中' if self.conversation_active else '等待人脸'}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.imshow('Plant Monitor - Press Q to exit', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            self.cleanup()

    def _reinit_camera(self):
        """摄像头重初始化（安全版）"""
        print("尝试重启摄像头……")
        if self.cap:
            self.cap.release()
        time.sleep(1)
        self._init_camera()

    def cleanup(self):
        """资源清理"""
        self.running = False
        if self.cap and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        print("[系统] 资源已释放")


if __name__ == "__main__":
    device = None
    try:
        device = ClassroomDevice()
        device.run_detection()
    except Exception as e:
        print(f"[致命错误] {e}")
    finally:
        if device:
            device.cleanup()
