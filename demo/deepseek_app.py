import os, re, wave, json, logging
import torch, soundfile as sf
from io import BytesIO
from threading import Thread, Lock
from flask import Flask, Response, render_template, request, jsonify, stream_with_context
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer, BitsAndBytesConfig
from faq_retriever import FAQRetriever
# 恢复 RAG 功能，加载指令模板
from RAG import load_instruction_from_mysql
from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
import numpy as np

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

# TTS 初始化
REF_AUDIO_PATH = r"C:\Users\wx\Desktop\All\GPT-SoVIT\data\slicer_opt\1.wav"
tts_config = TTS_Config("GPT_SoVITS/configs/tts_infer.yaml")
tts_pipeline = TTS(tts_config)

# 辅助函数

def escape_json_string(s):
    return s.replace('"','\\"').replace("\n","\\n")

# WAV 头与打包

def _wave_header(sr=None):
    sr = sr or tts_config.sampling_rate
    buf = BytesIO()
    with wave.open(buf,"wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(b"")
    return buf.getvalue()

def _pack_wav(audio,sr):
    buf = BytesIO()
    sf.write(buf,audio,sr,format='wav')
    return buf.getvalue()

# 核心：流式生成文本或音频响应

def generate_response_stream(question):
    global chat_history
    # 1. FAQ检索
    faq_ans = faq_retriever.search_faq(question)
    if faq_ans:
        chat_history.append(f"用户: {question}")
        chat_history.append(f"助手: {faq_ans}")
        yield f'data:{{"type":"notice","content":"[系统] 知识库命中"}}\n\n'
        yield f'data:{{"type":"answer","content":"{escape_json_string(faq_ans)}"}}\n\n'
        yield 'data:[DONE]\n\n'
        return

    # 2. LLM 生成回答
    with history_lock:
        hist = "\n".join(chat_history[-10:])
    # 默认系统提示
    system_prompt = ("你是水稻种植知识助手，专注于农业领域的问答，" 
                     "请给出专业、详细且易懂的回答。")
    prompt = (
        f"Below is an instruction\n"                  
        f"Instruction: {system_prompt}\n\n"
        f"History: {hist}\n" 
        f"User: {question}\n"
        f"Assistant: <think>"
    )
    # 编码和流式生成
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    Thread(target=model.generate, kwargs={
        "input_ids":inputs.input_ids,
        "attention_mask":inputs.attention_mask,
        "max_new_tokens":512,
        "streamer":streamer,
        "temperature":0.7,
        "top_p":0.9
    }).start()

    thinking = True
    full_answer=""
    for chunk in streamer:
        if thinking:
            if "</think>" in chunk:
                parts = chunk.split("</think>",1)
                yield f'data:{{"type":"think","content":"{escape_json_string(parts[0])}"}}\n\n'
                thinking=False
                chunk = parts[1]
            else:
                yield f'data:{{"type":"think","content":"{escape_json_string(chunk)}"}}\n\n'
                continue
        # 输出回答内容
        full_answer += chunk
        yield f'data:{{"type":"answer","content":"{escape_json_string(chunk)}"}}\n\n'

    # 如果无回答则占位
    if not full_answer.strip():
        placeholder = "抱歉，未能生成回答，请稍后重试。"
        yield f'data:{{"type":"answer","content":"{escape_json_string(placeholder)}"}}\n\n'
        full_answer = placeholder

    # 更新历史
    chat_history.append(f"用户: {question}")
    chat_history.append(f"助手: {full_answer}")
    if len(chat_history)>20:
        chat_history=chat_history[-20:]
    yield 'data:[DONE]\n\n'

# 音频流生成
def generate_audio_stream(answer):
    yield _wave_header()
    segments = re.split(r'(?<=[。？！\n])', answer)
    for seg in segments:
        seg = seg.strip()
        if not seg: continue
        for sr,audio in tts_pipeline.run({
            'text':seg,
            'text_lang':'zh',
            'ref_audio_path':REF_AUDIO_PATH,
            'prompt_lang':'zh',
            'speed_factor':1.2
        }):
            yield _pack_wav(audio,sr)

# 统一路由
@app.route('/stream', methods=['GET','POST'])
def unified_stream():
    q = request.args.get('question','')
    data = {}
    if not q:
        data = request.get_json(silent=True) or {}
        q = data.get('question','')
    if not q:
        return jsonify({'error':'问题不能为空'}),400
    audio_flag = request.args.get('audio')=='true' or data.get('audio',False)
    if audio_flag:
        def audio_gen():
            full = ''
            for event in generate_response_stream(q):
                if event.startswith('data:{"type":"answer"'):
                    try: full+=json.loads(event[5:-2])['content']
                    except: pass
            yield from generate_audio_stream(full)
        return Response(stream_with_context(audio_gen()), mimetype='audio/wav')
    return Response(stream_with_context(generate_response_stream(q)), content_type='text/event-stream')

@app.route('/')
def home():
    return render_template('index.html')

if __name__=='__main__':
    app.run(host='0.0.0.0', port=9000, threaded=True)
