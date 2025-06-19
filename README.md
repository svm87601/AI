
# 植物园智能观察系统

集成 ASR-VAD、TTS、FAQ 问答、RAG、GPT_SoVITS，实现语音交互＋传感器＋大模型推理的植物园智能观察系统。



<!-- PROJECT LOGO -->
<br />

<p align="center">
  <a href="https://github.com/Cjj5201314">
    <img src="https://github.com/Cjj5201314/Picture/blob/main/Data/1.png?raw=true" 
         alt="Logo" 
         width="80" 
         height="80" 
         style="border-radius: 50%;">
  </a>

  <h3 align="center">ASR-VAD: 语音识别与语音活动检测</h3>

  <h3 align="center">FAQ-RAG-TTS: 智能问答系统</h3>
  <p align="center">
  客户端 + 服务器端 + 语音交互 + 传感器
    <br />
    <a href="https://github.com/shaojintian/Best_README_template"><strong>探索本项目的文档 »</strong></a>
    <br />
    <br />
    <a href="https://github.com/shaojintian/Best_README_template">查看Demo</a>
    ·
    <a href="https://github.com/shaojintian/Best_README_template/issues">报告Bug</a>
    ·
    <a href="https://github.com/shaojintian/Best_README_template/issues">提出新特性</a>
  </p>

</p>


 本篇README.md面向开发者
 
## 目录

- [上手指南](#上手指南)  
  - [开发前的配置要求](#开发前的配置要求)  
  - [安装步骤](#安装步骤)  
- [文件目录说明](#文件目录说明)  
- [开发架构概览](#开发架构概览)  
- [部署](#部署)  
- [使用到的框架](#使用到的框架)  
- [贡献者](#贡献者)  
  - [如何参与开源项目](#如何参与开源项目)  
- [版本控制](#版本控制)  
- [作者](#作者)  
- [鸣谢](#鸣谢)  

---

### 上手指南


###### 开发前的配置要求

- 操作系统：Windows10/Ubuntu18.04/Raspberry Pi OS 
- Python：>=3.8  
- Conda：>=4.10  

###### **安装步骤**

1. **克隆仓库**  

```sh
git clone https://github.com/shaojintian/Best_README_template.git
```
2. **创建并激活conda环境**  
```sh
conda create -n plant_sys python=3.9 -y
conda activate plant_sys
```
3. **安装依赖**  
```sh
pip install -r ASR-VAD/requirements.txt
pip install -r demo/requirements.txt
pip install -r Botanical/requirements.txt
pip install -r sensor_Demo/requirements.txt
```

### 文件目录说明
eg:

```
.
├── ASR-VAD/                   # ASR + VAD API 服务端
│   ├── asr_vad_api.py
│   ├── asr/test_asr.py
│   ├── config/
│   ├── core/
│   ├── models/
│   │   ├── SenseVoiceSmall    # TODO: 下载模型放入此处
│   │   └── snakers4_silero-vad
│   └── tmp/
│
├── sensor_Demo/               # 树莓派 Pi4 边缘采集端 (C++ + Python)
│   ├── c++/
│   │   ├── src/
│   │   │   ├── Base64.cpp
│   │   │   ├── Logger.cpp
│   │   │   ├── main.cpp
│   │   │   └── SensorDataCollector.cpp
│   │   ├── include/Base64.h
│   │   └── CMakeLists.txt
│   ├── config.json
│   ├── dataDemo.py            # 获取串口数据＋拍照＋Base64 → 上传后台
│   ├── 采集器代码说明文档.md
│   └── requirements.txt
│
├── Botanical/                 # Jetson TX2 边缘客户端
│   ├── client.py              # deepseek_app.py 的 RPC 客户端
│   ├── classroom_device.py    # deepseek_app3.py 的 RPC 客户端
│   ├── chat.proto
│   ├── chat_pb2.py
│   ├── chat_pb2_grpc.py
│   └── requirements.txt
│
├── demo/                      # Web + 大模型 + FAQ + TTS 演示
│   ├── Audio_data/
│   ├── data/FAQ问答对
│   ├── GPT_SoVITS/
│   │   ├── configs/tts_infer.yaml
│   │   ├── pretrained_models/   # TODO: 下载放入此处
│   │   └── text/G2PWModel       # TODO: 下载放入此处
│   ├── Mysql/faq_data.sql
│   ├── saved_audio/
│   ├── templates/index.html
│   ├── deepseek_app.py
│   ├── deepseek_app3.py
│   ├── FAQ_data.ipynb           # FAQ 问答对导入 MySQL 脚本
│   ├── faq_retriever.py
│   ├── RAG.py
│   ├── test_mqtt.py
│   └── tts_audio.py
│
└── Models/                    # 各种大模型下载链接
    ├── DeepSeek-R1-Distill-Qwen-1___5B
    ├── DeepSeek-R1-Distill-Qwen-7B
    └── paraphrase-multilingual-MiniLM-L12-v2


```





### 开发的架构 

- ASR-VAD：负责音频转文本的 REST/gRPC 服务
- sensor_Demo：树莓派端采集传感器＋拍照，上报后台
- Botanical：Jetson TX2 上的 RPC 客户端，接入大模型推理 + TTS
- demo：Web 前端 + FAQ 缓存 + 大模型生成 + TTS 播放
- Models：存放所有大模型下载链接


### 部署

1. **初始化数据库** <br>
FAQ 数据自动导入：
```
cd demo
jupyter notebook FAQ_data.ipynb
# 运行 Notebook，即可建库建表并导入 FAQ 对
```
2. **启动 ASR-VAD 服务**
```
cd ASR-VAD
python asr_vad_api.py
```
3. **启动 sensor_Demo 采集器**
```
cd sensor_Demo
python dataDemo.py
```
4. **启动 Web Demo**
```
cd demo
python deepseek_app3.py
```

### 注意事项

### 使用到的框架

- [FastAPI](https://fastapi.tiangolo.com)
- [gRPC / Protobuf](https://grpc.io/)
- [PyTorch / ONNX Runtime](https://pytorch.org/)
- [Flask](https://flask.palletsprojects.com)
- [CMake / C++17](https://cmake.org/) 
- [MySQL](https://www.mysql.com/)
- [Jupyter Notebook](https://jupyter.org/)
- [MQTT](https://mqtt.org/)
- [TensorRT](https://developer.nvidia.com/tensorrt)

### 贡献者

请阅读**CONTRIBUTING.md** 查阅为该项目做出贡献的开发者。

#### 如何参与开源项目

贡献使开源社区成为一个学习、激励和创造的绝佳场所。你所作的任何贡献都是**非常感谢**的。


1. Fork 本项目
2. 创建 Feature 分支 (git checkout -b feature/AmazingFeature)
3. 提交改动 (git commit -m 'Add AmazingFeature')
4. 推送并发起 Pull Request
5. 等待审核，回复意见



### 版本控制

使用 Git 进行版本管理，分支策略请参考 CONTRIBUTING.md。

### 作者
<a href="https://github.com/Cjj5201314" target="_blank">
  <img src="https://github.com/Cjj5201314.png?size=100"
       width="80" height="80"
       style="border-radius:50%;"/>
</a>
<a href="https://github.com/svm87601" target="_blank">
  <img src="https://github.com/svm87601.png?size=100"
       width="80" height="80"
       style="border-radius:50%;"/>
</a>
<a href="https://github.com/swt-xim" target="_blank">
  <img src="https://github.com/swt-xim.png?size=100"
       width="80" height="80"
       style="border-radius:50%;"/>
</a>



- [GitHub: @Cjj5201314](https://github.com/Cjj5201314)
- [GitHub: @svm87601](https://github.com/svm87601)
- [GitHub: @swt-xim](https://github.com/swt-xim)


### 版权说明

无

### 鸣谢


- [Best README Template](https://github.com/shaojintian/Best_README_template)
- [Sentence-BERT](https://github.com/UKPLab/sentence-transformers)
- [Silero VAD](https://github.com/snakers4/silero-vad)
- [DeepSeek](https://github.com/deepsound-project/samplernn-pytorch)







