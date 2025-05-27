from flask import Flask, request, jsonify, Response, render_template
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from peft import PeftModel
from threading import Thread
import logging
import mysql.connector
from faq_retriever import FAQRetriever
import json
from RAG import load_instruction_from_mysql
# Set log level
logging.getLogger("transformers").setLevel(logging.ERROR)

app = Flask(__name__)

# Load model and tokenizer
model_path = r"D:\tensorflow\中小学教育\DeepSeek-R1-Distill-Qwen-1___5B"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Load instruction content
instruction_content = load_instruction_from_mysql()

# Global chat history
chat_history = []
faq_retriever = FAQRetriever()

def escape_json_string(s):
    """转义字符串中的特殊字符以用于JSON"""
    return s.replace('"', '\\"').replace('\n', '\\n')

def generate_response(question):
    """Generate streaming response for the given question"""
    global chat_history
    
    # 1. 先检查FAQ系统
    faq_answer = faq_retriever.search_faq(question)
    if faq_answer:
        # 添加到历史记录
        chat_history.append(f"### Question: {question}")
        chat_history.append(f"### Response: {faq_answer.strip()}")
        
        # 返回格式化的SSE响应
        escaped_answer = escape_json_string(faq_answer)
        yield f'data: {{"type": "notice", "content": "[系统] 从知识库中找到匹配答案"}}\n\n'
        yield f'data: {{"type": "answer", "content": "{escaped_answer}"}}\n\n'
        yield 'data: [DONE]\n\n'  # 结束标记
        return
    
    # 2. 如果没有FAQ匹配，调用大模型生成回答
    history_prompt = ""
    if chat_history:
        recent_history = chat_history[-10:]
        history_prompt = "\n".join(recent_history) + "\n"

    prompt_style = f"""Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that appropriately completes the request.
Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

### Instruction:
{instruction_content}
### History:
{history_prompt}

### Question:
{question}

### Response:
<think>"""

    model_inputs = tokenizer([prompt_style], return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    generation_kwargs = dict(
        input_ids=model_inputs.input_ids,
        attention_mask=model_inputs.attention_mask,
        max_new_tokens=800,
        use_cache=True,
        streamer=streamer
    )
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    response = ""
    think_output = ""
    answer_output = ""
    is_think_complete = False

    for new_text in streamer:
        if not is_think_complete:
            if "</think>" in new_text:
                think_part = new_text.split("</think>")[0]
                think_output += think_part
                is_think_complete = True
                answer_part = new_text.split("</think>")[1]
                answer_output += answer_part
                yield f'data: {{"type": "think", "content": "{escape_json_string(think_part)}"}}\n\n'
                yield f'data: {{"type": "answer", "content": "{escape_json_string(answer_part)}"}}\n\n'
            else:
                think_output += new_text
                yield f'data: {{"type": "think", "content": "{escape_json_string(new_text)}"}}\n\n'
        else:
            answer_output += new_text
            yield f'data: {{"type": "answer", "content": "{escape_json_string(new_text)}"}}\n\n'

    # 添加当前对话到历史记录
    chat_history.append(f"### Question: {question}")
    chat_history.append(f"### Response: {answer_output.strip()}")

    # 限制历史记录长度
    if len(chat_history) > 10:
        chat_history = chat_history[-10:]
    
    yield 'data: [DONE]\n\n'  # 结束标记

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')

@app.route('/ask', methods=['GET'])
def ask():
    """Handle question requests"""
    question = request.args.get('question')
    if not question:
        return jsonify({"error": "No question provided"}), 400

    return Response(generate_response(question), content_type='text/event-stream')

if __name__ == '__main__':
    app.run(debug=False, threaded=True, host='0.0.0.0')