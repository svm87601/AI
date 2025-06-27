#!/usr/bin/env python3
# server.py

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
import base64
import json
import subprocess
import re

# 导入 gRPC 自动生成的 stub
import chat_pb2
import chat_pb2_grpc

from flask import Flask, jsonify, request, Response, render_template, send_from_directory
import threading

from threading import Lock
import atexit
from flask import Flask, jsonify, render_template, Response, send_from_directory
from flask_cors import CORS
# 添加全局锁
frame_lock = Lock()
app = Flask(__name__, 
           template_folder='/home/tx2/Desktop/Botanical/templates',
           static_folder='/home/tx2/Desktop/Botanical/static')
CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)
app.config['CORS_HEADERS'] = 'Content-Type'
# 共享状态
system_state = {
    "face_detected": False,
    "conversation_active": False,
    "recognized_text": "",
    "response_text": "",
    "response_audio": b"",
    "system_message": "系统就绪",
    "camera_frame": None,
    "playing_audio": False,
    "current_media": "default_grass",  # default_grass, image1, video1
    "video_playing": False
}

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
USB_CAMERA_INDEX = 1 
FRAME_WIDTH             = 640               
FRAME_HEIGHT            = 480

# 新摄像头设备ID (2bdf:0289)
TARGET_CAMERA_ID = "2bdf:0289"

# 媒体文件路径 - 修正目录结构
DEFAULT_GRASS_IMAGE = "static/images/default_grass.jpg"
IMAGE1_PATH = "static/images/image1.jpg"
VIDEO1_PATH = "static/videos/video1.mp4"

# ------------------------------------------------

# 全局设备实例
device_instance = None

def signal_handler(signum, frame):
    print(f"\n收到终止信号 {signum}，正在安全关闭...")
    if device_instance:
        device_instance.cleanup()
    # 等待所有线程完成清理
    time.sleep(1)
    os._exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# 注册退出处理函数
def cleanup_at_exit():
    print("应用程序退出，执行清理...")
    if device_instance:
        device_instance.cleanup()

atexit.register(cleanup_at_exit)

def find_camera_by_id(vendor_id, product_id):
    """根据设备ID查找对应的摄像头设备文件路径"""
    try:
        # 执行lsusb命令获取设备信息
        result = subprocess.check_output(["lsusb"]).decode("utf-8")
        lines = result.split("\n")
        
        # 查找匹配的设备
        for line in lines:
            # 解析lsusb输出，格式类似: Bus 001 Device 033: ID 2bdf:0289
            match = re.search(r"Bus \d+ Device \d+: ID (\w+):(\w+)", line)
            if match:
                vid, pid = match.groups()
                if f"{vid}:{pid}" == f"{vendor_id}:{product_id}":
                    # 获取设备序号
                    device_num = re.search(r"Device (\d+):", line).group(1)
                    
                    # 查找对应的video设备
                    video_devices = subprocess.check_output(["ls", "/dev/video*"]).decode("utf-8")
                    for video_dev in video_devices.split("\n"):
                        if video_dev:
                            # 尝试通过udevadm获取设备信息
                            try:
                                udev_info = subprocess.check_output(
                                    ["udevadm", "info", "--query=all", "--name", video_dev]
                                ).decode("utf-8")
                                if f"ID_VENDOR_ID={vendor_id}" in udev_info and f"ID_MODEL_ID={product_id}" in udev_info:
                                    return video_dev
                            except:
                                continue
        return None
    except Exception as e:
        print(f"查找摄像头设备时出错: {e}")
        return None

class ClassroomDevice:
    def __init__(self):
        global device_instance
        device_instance = self
        
        # 摄像头初始化
        self.cap = None
        self._init_camera()

        self.detector = dlib.get_frontal_face_detector()
        self.last_detected_time = time.time()
        self.running = True

        # 会话状态
        self.conversation_active = False
        self.audio_player = None 
        self.conversation_thread = None

        # gRPC channel + stub
        self.grpc_channel = grpc.insecure_channel(SERVER_GRPC_ADDR)
        self.stub = chat_pb2_grpc.ChatServiceStub(self.grpc_channel)
        
        # 启动摄像头帧处理线程
        self.frame_thread = threading.Thread(target=self.process_camera_frames)
        self.frame_thread.daemon = True
        self.frame_thread.start()
        
        # 添加退出回调
        atexit.register(self.cleanup)

    def _init_camera(self):
        """初始化摄像头（带详细日志和错误处理）"""
        max_retries = 5
        for i in range(max_retries):
            self.cap = cv2.VideoCapture(USB_CAMERA_INDEX, cv2.CAP_V4L2)
            
            if self.cap.isOpened():
                # 打印实际获取的摄像头参数（调试用）
                actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                fps = self.cap.get(cv2.CAP_PROP_FPS)
                print(f"摄像头初始化成功！实际分辨率: {actual_width}x{actual_height}, FPS: {fps}")
                    
                # 设置目标分辨率（若支持）
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
                self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
                self.cap.set(cv2.CAP_PROP_FPS, 15)
                    
                time.sleep(1)  # 摄像头预热
                return
            else:
                if self.cap is not None:
                    self.cap.release()
                # 修改错误处理逻辑，不使用getErrorStatus()
                error_msg = "无法打开摄像头设备"
                if hasattr(cv2, 'error'):
                    try:
                        # 尝试获取错误信息（适用于新版本OpenCV）
                        error_msg = cv2.error(cv2.getTickCount()).msg
                    except:
                        pass
                print(f"摄像头初始化失败 ({i+1}/5): {error_msg}，正在重试...")
                time.sleep(2)
        raise RuntimeError(f"无法打开摄像头设备 /dev/video{USB_CAMERA_INDEX}")
    
    def process_camera_frames(self):
        """处理摄像头帧并更新状态"""
        while self.running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    # 尝试重新初始化摄像头
                    self._init_camera()
                    continue
                
                # 处理帧并检测人脸
                current_time = time.time()
                if current_time - self.last_detected_time > FACE_DETECTION_INTERVAL:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = self.detector(gray, 0)
                    
                    if len(faces) > 0:
                        print("检测到人脸，准备 switch current_media=image1")
                        system_state["current_media"] = "image1"
                        self.last_detected_time = current_time
                        system_state["face_detected"] = True
                        
                        # 检测到人脸时更新媒体状态为image1
                        if system_state["current_media"] != "image1" and not system_state["playing_audio"]:
                            system_state["current_media"] = "image1"
                        
                        # 绘制人脸框
                        for face in faces:
                            x, y, w, h = face.left(), face.top(), face.width(), face.height()
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    else:
                        if time.time() - self.last_detected_time > NO_PERSON_TIMEOUT:
                            system_state["face_detected"] = False
                            # 长时间未检测到人脸时恢复到默认状态
                            if not system_state["playing_audio"]:
                                system_state["current_media"] = "default_grass"
                
                # 添加状态文本
                status_text = f"状态: {'检测到人脸' if system_state['face_detected'] else '等待中'}"
                cv2.putText(frame, status_text, (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # 转换为JPEG并更新状态
                _, jpeg = cv2.imencode('.jpg', frame)
                system_state["camera_frame"] = jpeg.tobytes()
                
                time.sleep(0.05)  # 控制帧率
            except Exception as e:
                print(f"处理摄像头帧时出错: {e}")
                if not self.running:  # 如果正在关闭，直接退出
                    return
                time.sleep(1)

    def start_conversation(self):
        """当检测到人脸时启动录音+对话线程"""
        if not self.conversation_active:
            self.conversation_active = True
            system_state["conversation_active"] = True
            system_state["system_message"] = "对话已启动，请开始说话..."
            
            print("启动对话线程...")
            self.conversation_thread = threading.Thread(target=self.conversation_loop)
            self.conversation_thread.daemon = True
            self.conversation_thread.start()

    def stop_conversation(self):
        """停止对话"""
        if self.conversation_active:
            print("停止对话线程...")
            self.conversation_active = False
            system_state["conversation_active"] = False
            system_state["system_message"] = "对话已停止"
            
            # 等待对话线程结束
            if self.conversation_thread and self.conversation_thread.is_alive():
                self.conversation_thread.join(timeout=3.0)
                if self.conversation_thread.is_alive():
                    print("警告：对话线程未在超时时间内结束")

    def conversation_loop(self):
        """
        对话循环：自动录音 -> 上传 ASR+VAD 服务 -> 拿到文本 -> 调用 gRPC Chat
        当超过超时时间无人时自动结束对话
        """
        print("对话循环开始...")
        while self.running and self.conversation_active:
            # 如果超过 NO_PERSON_TIMEOUT 未检测到人脸，则退出对话模式
            if time.time() - self.last_detected_time > NO_PERSON_TIMEOUT:
                system_state["system_message"] = "长时间未检测到人脸，退出对话模式"
                break

            # 1) 录制一段音频
            wav_data = self.record_audio(RECORD_DURATION)
            if wav_data is None:
                system_state["system_message"] = "录音失败，请稍后重试"
                break

            # 2) 上传给 ASR+VAD 服务，获取识别文本
            text = self.call_asr_service(wav_data)
            if not text:
                system_state["system_message"] = "未识别到有效语音"
                continue
            
            system_state["recognized_text"] = text
            system_state["system_message"] = "正在处理您的请求..."

            # 3) 调用 gRPC 发送识别到的文本，并处理流式返回
            self.call_grpc_and_play(text)

        self.conversation_active = False
        system_state["conversation_active"] = False
        system_state["system_message"] = "对话结束"
        print("对话循环结束")

    def record_audio(self, duration_s: float):
        """
        使用 sounddevice 录音，并返回 WAV 二进制数据
        录制 duration_s 秒钟后自动停止
        """
        try:
            system_state["system_message"] = f"正在录音 ({duration_s}秒)..."
            
            recording = sd.rec(int(duration_s * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                               channels=CHANNELS, dtype='int16')
            sd.wait()  # 等待录音结束
            
            # 保存到 BytesIO
            buf = BytesIO()
            sf.write(buf, recording, SAMPLE_RATE, format='wav')
            wav_bytes = buf.getvalue()
            return wav_bytes
        except Exception as e:
            system_state["system_message"] = f"录音出错: {e}"
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
                system_state["system_message"] = f"ASR 服务错误: {response.status_code}"
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
            system_state["system_message"] = f"ASR 调用失败: {e}"
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
            system_state["system_message"] = f"模型连接失败: {e}"
            return

        # 定义播放音频时要缓冲的 BytesIO
        audio_buffer = BytesIO()
        response_text = ""
        is_streaming_audio = False

        for resp in responses:
            # 如果对话被停止，则中断处理
            if not self.conversation_active:
                print("对话已停止，中断gRPC处理")
                break
                
            # 文本分片
            if resp.HasField("text_chunk"):
                txt = resp.text_chunk.text
                response_text += txt
                system_state["response_text"] = response_text
                continue

            # 音频分片
            if resp.HasField("audio_chunk"):
                is_streaming_audio = True
                chunk = resp.audio_chunk.data
                if chunk:
                    audio_buffer.write(chunk)
                # 如果是最后一个音频分片，则播放
                if resp.audio_chunk.is_last:
                    # 设置播放状态和视频媒体
                    system_state["playing_audio"] = True
                    system_state["current_media"] = "video1"
                    
                    # 将 buffer 定位回头，读 PCM 播放
                    audio_buffer.seek(0)
                    data, sr = sf.read(audio_buffer)
                    
                    # 保存音频数据供前端使用
                    temp_buffer = BytesIO()
                    sf.write(temp_buffer, data, sr, format='wav')
                    system_state["response_audio"] = temp_buffer.getvalue()
                    
                    # 播放音频
                    sd.play(data, sr)
                    sd.wait()
                    
                    # 重置 buffer，等待下次
                    audio_buffer = BytesIO()
                    is_streaming_audio = False
                    
                    # 播放完成后立即重置媒体状态
                    system_state["playing_audio"] = False
                    if system_state["face_detected"]:
                        system_state["current_media"] = "image1"
                    else:
                        system_state["current_media"] = "default_grass"
                continue

            # Done 标志
            if resp.HasField("done"):
                system_state["system_message"] = "交互完成"
                break

        system_state["system_message"] = "模型响应处理完成"

    def run_detection(self):
        """人脸检测主循环，检测到人脸则触发对话模式"""
        try:
            while self.running:
                time.sleep(1)  # 主循环不再处理帧，交由帧处理线程
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
        """资源清理 - 确保所有资源被正确释放"""
        global device_instance
        
        if hasattr(self, "cleaned") and self.cleaned:
            return
            
        self.cleaned = True
        print("开始资源清理...")
        
        # 停止运行标志
        self.running = False
        
        # 停止对话线程
        self.stop_conversation()
        
        # 释放摄像头资源
        print("释放摄像头资源...")
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.cap = None
            
        # 关闭gRPC通道
        print("关闭gRPC通道...")
        if hasattr(self, 'grpc_channel') and self.grpc_channel:
            try:
                self.grpc_channel.close()
                self.grpc_channel = None
            except Exception as e:
                print(f"关闭gRPC通道时出错: {e}")
        
        # 等待帧处理线程结束
        print("等待帧处理线程结束...")
        if hasattr(self, 'frame_thread') and self.frame_thread.is_alive():
            self.frame_thread.join(timeout=2.0)
            if self.frame_thread.is_alive():
                print("警告: 帧处理线程未在超时时间内结束")
        
        cv2.destroyAllWindows()
        system_state["system_message"] = "系统已关闭"
        print("资源清理完成")
        
        # 重置全局实例
        device_instance = None


# Flask 路由
@app.route('/')
def index():
    """主页面"""
    return render_template('index.html')

@app.route('/api/face_status', methods=['GET'])
def face_status():
    """获取人脸检测状态"""
    return jsonify({
        "detected": system_state["face_detected"],
        "conversation_active": system_state["conversation_active"],
        "system_message": system_state["system_message"],
        "recognized_text": system_state["recognized_text"],
        "response_text": system_state["response_text"],
        "current_media": system_state["current_media"],
        "playing_audio": system_state["playing_audio"]
    })

@app.route('/api/start_chat', methods=['POST'])
def start_chat():
    """启动对话"""
    if not system_state["conversation_active"]:
        # 触发对话逻辑
        threading.Thread(target=device.start_conversation).start()
        return jsonify({"status": "started"}), 202
    return jsonify({"status": "already_active"}), 200

@app.route('/api/stop_chat', methods=['POST'])
def stop_chat():
    """停止对话"""
    device.stop_conversation()
    return jsonify({"status": "stopped"}), 200

@app.route('/api/get_audio', methods=['GET'])
def get_audio():
    """获取生成的音频"""
    if system_state["response_audio"]:
        return Response(system_state["response_audio"], mimetype="audio/wav")
    return jsonify({"error": "No audio available"}), 404

@app.route('/api/camera_feed')
def camera_feed():
    """摄像头视频流"""
    def generate():
        while True:
            # 获取当前帧数据
            frame_data = system_state["camera_frame"]
            
            # 如果帧数据缺失，创建空白帧
            if not frame_data:
                try:
                    # 创建一个黑色背景的空白帧
                    blank = np.zeros((480, 640, 3), dtype=np.uint8)
                    # 添加提示文字
                    cv2.putText(blank, "摄像头初始化中...", (50, 240), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    # 编码为JPEG
                    _, jpeg = cv2.imencode('.jpg', blank)
                    frame_data = jpeg.tobytes()
                except Exception as e:
                    print(f"创建空白帧错误: {e}")
                    # 返回简单的错误消息
                    frame_data = b''
            
            # 构建MJPEG流格式
            try:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + 
                       frame_data + b'\r\n')
            except Exception as e:
                print(f"视频流生成错误: {e}")
                # 返回一个简单的错误帧
                yield (b'--frame\r\n'
                       b'Content-Type: text/plain\r\n\r\n' +
                       b'Error: ' + str(e).encode() + b'\r\n')
            
            # 控制帧率
            time.sleep(0.05)
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/static/<path:path>')
def send_static(path):
    """提供静态文件服务"""
    return send_from_directory('/home/tx2/Desktop/Botanical/static', path)

def run_flask():
    app.run(host='0.0.0.0', port=5000, threaded=True)

if __name__ == "__main__":
    try:
        device = ClassroomDevice()
        
        # 在新线程中启动Flask服务
        flask_thread = threading.Thread(target=run_flask, daemon=True)
        flask_thread.start()
        
        # 运行主检测循环
        device.run_detection()
    except Exception as e:
        print(f"主程序发生错误: {e}")
        if device:
            device.cleanup()
        raise
