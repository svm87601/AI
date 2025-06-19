import os, re, wave, json, logging, time
import torch, soundfile as sf
from io import BytesIO
from threading import Thread, Lock
from flask import Flask, Response, render_template, request, jsonify, stream_with_context, send_file

# ===== gRPC 导入 =====
import grpc
from concurrent import futures
import chat_pb2, chat_pb2_grpc

# ===== 其余第三方 =====
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer, BitsAndBytesConfig
from faq_retriever import FAQRetriever
from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
import numpy as np
import paho.mqtt.client as mqtt
import base64

# 设置日志级别
logging.getLogger("transformers").setLevel(logging.ERROR)

app = Flask(__name__)

# 模型和Tokenizer路径
MODEL_PATH = r"C:\Users\wx\Desktop\All\Models\DeepSeek-R1-Distill-Qwen-1___5B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# 全局历史与FAQ检索
chat_history = []
history_lock = Lock()
faq_retriever = FAQRetriever()
AUDIO_MAP_PATH = r"C:\Users\wx\Desktop\All\demo\Audio_data\faq_audio_map.json"
faq_audio_map = json.load(open(AUDIO_MAP_PATH, 'r', encoding='utf-8'))

# TTS 初始化
REF_AUDIO_PATH = r"C:\Users\wx\Desktop\All\GPT-SoVIT\data\slicer_opt\1.wav"
tts_config = TTS_Config("GPT_SoVITS/configs/tts_infer.yaml")
tts_pipeline = TTS(tts_config)

# ==== MQTT 发布函数 ====
mqtt_conf = {
    "broker":   "mqtt.iksns.net",
    "port":     1883,
    "topic":    "smart/mattress/data",
    "username": "iotua",
    "password": "Vhwwhfy48vqHv6"
}
FIXED_SEED      = 20250603

# 辅助函数
def escape_json_string(s): return s.replace('"','\\"').replace("\n","\\n")

def _wave_header(sr=None):
    sr = sr or tts_config.sampling_rate
    buf = BytesIO()
    with wave.open(buf,"wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(b"")
    return buf.getvalue()

def _pack_wav(audio, sr):
    buf = BytesIO()
    sf.write(buf, audio, sr, format='wav')
    return buf.getvalue()

def publish_dialog_mqtt(user_q, sys_a, audio_path):
    # 将音频文件 base64 编码
    with open(audio_path, "rb") as f:
        b64_audio = base64.b64encode(f.read()).decode()
    payload = {
        "timestamp": time.time(),
        "dialog": {"U": user_q, "S": sys_a},
        "audio_b64": b64_audio
    }
    client = mqtt.Client()
    client.username_pw_set(mqtt_conf["username"], mqtt_conf["password"])
    client.connect(mqtt_conf["broker"], mqtt_conf["port"], 60)
    client.loop_start()
    client.publish(mqtt_conf["topic"], json.dumps(payload)).wait_for_publish()
    client.loop_stop()
    client.disconnect()

# 文本流生成
def generate_response_stream(question):
    global chat_history
    # 1) 尝试FAQ匹配
    faq_ans = faq_retriever.search_faq(question)
    if faq_ans:
        # 找回ID
        faq_id = None
        for item in faq_retriever.faq_data:
            if item.get('answer') == faq_ans:
                faq_id = item.get('id'); break
        chat_history.append(f"用户: {question}")
        chat_history.append(f"助手: {faq_ans}")
        yield f'data:{{"type":"notice","content":"[系统] 知识库命中"}}\n\n'
        yield f'data:{{"type":"answer","content":"{escape_json_string(faq_ans)}"}}\n\n'
        yield 'data:[DONE]\n\n'
        return
    # 2) LLM 生成
    with history_lock:
        hist = "\n".join(chat_history[-10:])
    system_prompt = ("你是水稻种植知识助手，专注于农业领域的问答，" "请给出专业、详细且易懂的回答。")
    prompt = (
        f"Below is an instruction\n"
        f"Instruction: {system_prompt}\n\n"
        f"History: {hist}\n"
        f"User: {question}\n"
        f"Assistant: <think>"
    )
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    Thread(target=model.generate, kwargs={
        'input_ids': inputs.input_ids,
        'attention_mask': inputs.attention_mask,
        'max_new_tokens': 512,
        'streamer': streamer,
        'temperature': 0.7,
        'top_p': 0.9
    }).start()

    thinking = True; full_answer = ""
    for chunk in streamer:
        if thinking:
            if '</think>' in chunk:
                parts = chunk.split('</think>',1)
                yield f'data:{{"type":"think","content":"{escape_json_string(parts[0])}"}}\n\n'
                thinking = False; chunk = parts[1]
            else:
                yield f'data:{{"type":"think","content":"{escape_json_string(chunk)}"}}\n\n'
                continue
        full_answer += chunk
        yield f'data:{{"type":"answer","content":"{escape_json_string(chunk)}"}}\n\n'
    if not full_answer.strip():
        placeholder = "抱歉，未能生成回答，请稍后重试。"
        yield f'data:{{"type":"answer","content":"{escape_json_string(placeholder)}"}}\n\n'
        full_answer = placeholder
    chat_history.append(f"用户: {question}")
    chat_history.append(f"助手: {full_answer}")
    if len(chat_history)>20: chat_history[:] = chat_history[-20:]
    yield 'data:[DONE]\n\n'

# 音频流生成
def generate_audio_stream(answer, save_audio_path):
    """生成 WAV 流，同时写入 save_audio_path"""
    # 打头
    yield _wave_header(tts_config.sampling_rate)
    # 打开文件写 PCM
    pcm_accum = []
    for seg in re.split(r'(?<=[。？！\n])', answer):
        seg = seg.strip(); 
        if not seg: 
            continue
        tts_req = {
            "text":              seg,
            "text_lang":         "zh",
            "ref_audio_path":    REF_AUDIO_PATH,
            "aux_ref_audio_paths": [],
            "prompt_text":       "",
            "prompt_lang":       "zh",
            "top_k":             5,
            "top_p":             1.0,
            "temperature":       1.0,
            "text_split_method": "cut5",
            "batch_size":        1,
            "batch_threshold":   0.75,
            "split_bucket":      True,
            "return_fragment":   False,
            "speed_factor":      1.0,
            "streaming_mode":    False,
            "seed":              FIXED_SEED,
            "parallel_infer":    True,
            "repetition_penalty":1.35
        }
        for sr, audio in tts_pipeline.run(tts_req):
            pcm_accum.append(audio)
            yield _pack_wav(audio, sr)
    # 合并并保存完整 WAV
    if pcm_accum:
        full_pcm = np.concatenate(pcm_accum, axis=0)
        sf.write(save_audio_path, full_pcm, tts_config.sampling_rate, format="wav")

# 统一路由
@app.route('/stream', methods=['GET','POST'])
def unified_stream():
    data = request.get_json(silent=True) or {}
    q = request.args.get('question') or data.get('question','')
    if not q:
        return jsonify({'error':'问题不能为空'}), 400
    is_audio = request.args.get('audio')=='true' or data.get('audio', False)

    # 1) FAQ 处理
    faq_ans = faq_retriever.search_faq(q)
    if is_audio and faq_ans:
        # 找到对应 ID
        faq_id = next((item['id'] for item in faq_retriever.faq_data
                       if item.get('answer') == faq_ans), None)
        if faq_id:
            path = faq_audio_map.get(str(faq_id), {}).get('audio')
            if path and os.path.exists(path):
                return send_file(path, mimetype='audio/wav')
        # 落回到 TTS 流

    # 2) 纯文本流
    if not is_audio:
        return Response(stream_with_context(generate_response_stream(q)),
                        content_type='text/event-stream')

    # 3) Audio 流：先累计完整文本
    full_answer = ""
    for e in generate_response_stream(q):
        if e.startswith('data:{"type":"answer"'):
            try:
                full_answer += json.loads(e[5:-2])['content']
            except:
                pass

    # 4) 保存对话和音频目录
    timestamp = int(time.time())
    SAVE_DIR = r"C:\Users\wx\Desktop\All\demo\saved_audio"
    os.makedirs(SAVE_DIR, exist_ok=True)
    audio_save_path = os.path.join(SAVE_DIR, f"audio_{timestamp}.wav")
    json_path       = os.path.join(SAVE_DIR, f"dialog_{timestamp}.json")

    # 保存 JSON 文件
    record = {"timestamp":timestamp, "dialog":{"U":q,"S":full_answer}}
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(record, jf, ensure_ascii=False, indent=2)

    # 5) 音频生成 + MQTT 推送
    def audio_gen():
        yield from generate_audio_stream(full_answer, audio_save_path)
        publish_dialog_mqtt(q, full_answer, audio_save_path)

    return Response(stream_with_context(audio_gen()),
                    mimetype='audio/wav')

# ===== gRPC 部分 =====
class ChatServiceServicer(chat_pb2_grpc.ChatServiceServicer):
    def Chat(self, request, context):
        import json as _json
        question = request.question
        session  = request.session_id

        # ——— 1) 尝试 FAQ ———
        faq_ans = faq_retriever.search_faq(question)
        if faq_ans:
            # 1a) 发送 FAQ 文本（直接标最后一个文本分片）
            yield chat_pb2.ChatResponse(
                text_chunk=chat_pb2.TextChunk(text=faq_ans, is_last=True)
            )

            # 1b) 如果有预制的 FAQ 音频，分片返回
            faq_id = next((i['id'] for i in faq_retriever.faq_data
                           if i.get('answer') == faq_ans), None)
            if faq_id is not None:
                path = faq_audio_map.get(str(faq_id),{}).get('audio')
                if path and os.path.exists(path):
                    with open(path, 'rb') as f:
                        while True:
                            data = f.read(1024)
                            if not data:
                                break
                            yield chat_pb2.ChatResponse(
                                audio_chunk=chat_pb2.AudioChunk(data=data, is_last=False)
                            )
                    # 空 chunk 标记音频结束
                    yield chat_pb2.ChatResponse(
                        audio_chunk=chat_pb2.AudioChunk(data=b"", is_last=True)
                    )
                    # Done
                    yield chat_pb2.ChatResponse(done=chat_pb2.Done(info="END"))
                    return
            # 若无预制音频，后续也会走到 TTS 生成

        # —— 2) FAQ 未命中，LLM + TTS 流 —— 
        full_answer = ""
        for line in generate_response_stream(question):
            if '"type":"answer"' in line:
                # 提取 JSON 里的 content
                json_str = line.partition('{')[2].rpartition('}')[0]
                data = _json.loads("{" + json_str + "}")
                piece = data.get("content", "")
                yield chat_pb2.ChatResponse(
                    text_chunk=chat_pb2.TextChunk(text=piece, is_last=False)
                )
                full_answer += piece

        # 把生成的 WAV 放到指定目录
        tmp_dir = r"C:\Users\wx\Desktop\All\demo\tmp"
        os.makedirs(tmp_dir, exist_ok=True)
        tmp_wav = os.path.join(tmp_dir, f"{session}.wav")

        # 音频流：generate_audio_stream 会写入 tmp_wav 并分片 yield
        for chunk in generate_audio_stream(full_answer, tmp_wav):
            yield chat_pb2.ChatResponse(
                audio_chunk=chat_pb2.AudioChunk(data=chunk, is_last=False)
            )

        # 最后一个空 chunk 标记音频结束
        yield chat_pb2.ChatResponse(
            audio_chunk=chat_pb2.AudioChunk(data=b"", is_last=True)
        )

        # 完成标志
        yield chat_pb2.ChatResponse(done=chat_pb2.Done(info="END"))

def serve_grpc():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    chat_pb2_grpc.add_ChatServiceServicer_to_server(ChatServiceServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("gRPC server started on port 50051")
    server.wait_for_termination()


if __name__=='__main__':
    # 启动 gRPC 服务
    t = Thread(target=serve_grpc, daemon=True)
    t.start()
    # 启动 Flask
    app.run(host='0.0.0.0', port=9000, threaded=True)
