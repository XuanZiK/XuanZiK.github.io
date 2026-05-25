---
title: "Roboarm - AI智能机器人系统"
excerpt: "智能机器人系统,支持中英文语音指令控制的物体抓取与放置,集成 YOLO 视觉检测、SenseVoice 语音识别与机械臂串口控制。<br/>"
collection: projects
---

<video controls preload="metadata" style="width: 100%; max-width: 800px; display: block; margin: 0 auto 1.5em auto;">
  <source src="/files/videos/roboarm_demo.mp4" type="video/mp4">
  你的浏览器不支持 video 标签。
</video>

## 项目描述

Roboarm_v1 是一个智能机器人系统,支持语音控制的物体抓取和放置操作。该系统集成了计算机视觉、语音识别、自然语言处理和机械臂控制等技术,能够通过中英文语音指令控制机器人进行智能操作。


## 主要功能

- **物体检测**:使用 YOLO 模型进行实时物体检测,支持多种立方体类别的识别
- **语音识别**:使用 SenseVoiceSmall 模型,支持中英粤日韩文语音识别
- **坐标变换**:相机像素坐标到世界坐标的 2D 仿射变换
- **语义匹配**:根据识别信息,理解抓取、放置等动作意图
- **机械臂控制**:通过串口与机械臂通信,执行笛卡尔坐标移动和吸盘控制
- **实时交互**:摄像头实时画面显示,支持键盘和语音交互

## 安装说明

### 环境要求

- Python 3.8+
- CUDA(可选,用于 GPU 加速)(这里仅用 CPU)

### 依赖安装

克隆项目到本地:

```bash
git clone https://github.com/XuanZiK/AI-Roboticarm.git
```

创建 Conda 环境:

```bash
conda create -n roboarm python=3.10
conda activate roboarm
```

安装 Python 依赖:

```bash
cd version1
pip install -r requirements.txt
```

### 模型和数据准备

- 模型文件位于 `models/` 目录,包括 YOLO 检测模型和其他 AI 模型
- 标定数据位于 `calibration/` 目录
- 语音意图数据位于 `utils/data/` 目录

## 使用方法

### 基本运行

```bash
python scripts/picknplace.py
```

### 测试模式

```bash
python test/test.py
```

### 交互说明

- 按空格键开始/停止录音
- 系统会自动解析语音指令并执行相应操作
- 按 `q` 键退出程序

## 文件结构

```
version1/
├── requirements.txt              # Python 依赖列表
├── calibration/                  # 标定相关
│   ├── calib_matrix_2d.json      # 2D 变换矩阵数据
│   └── calib.py                  # 相机-机械臂标定类
├── models/                       # AI 模型文件
│   ├── yoloe-v8s-seg.pt          # YOLO 物体检测模型
│   ├── model.pt                  # 主模型
│   └── ...                       # 其他模型文件
├── scripts/                      # 主脚本
│   └── picknplace.py             # Pick-and-Place 主程序
├── test/                         # 测试脚本
│   ├── test.py                   # 交互测试
│   ├── Detection_test.py         # 检测测试
│   └── send_command.py           # 命令行发送测试
├── utils/                        # 工具模块
│   ├── __init__.py
│   ├── audio.py                  # 音频处理和语音识别
│   ├── command.py                # 机械臂串口通信
│   ├── detect.py                 # 物体检测器
│   ├── intent_command.py         # 意图到命令转换
│   ├── intent_fuzz_loader.py     # 模糊意图加载器
│   ├── intent_types.py           # 意图类型定义
│   ├── speech_intent_engine.py   # 语音意图引擎
│   ├── speech_intent_en.py       # 英文语音意图解析器
│   ├── speech_intent_zh.py       # 中文语音意图解析器
│   ├── transform.py              # 外参标定矩阵计算
│   └── data/                     # 语音意图数据
│       ├── en_speech_intent.json # 英文意图配置
│       ├── zh_speech_intent.json # 中文意图配置
│       └── intent_fuzz.json      # 模糊意图数据
└── README.md                     # 项目说明文档
```

## 核心模块说明

### 语音处理(`utils/audio.py`, `utils/speech_intent_*.py`)

- 实现语音录制、识别和意图解析
- 支持中英文语音指令
- 使用规则引擎匹配动作和目标

### 视觉检测(`utils/detect.py`)

- 基于 YOLO 的实时物体检测
- 支持多种立方体类别的识别
- 提供像素坐标到世界坐标的转换接口

### 坐标变换(`utils/transform.py`)

- 实现相机像素坐标到机械臂世界坐标的 2D 仿射变换
- 加载预计算的变换矩阵

### 机械臂控制(`utils/command.py`)

- 串口通信接口
- 支持笛卡尔坐标移动和吸盘控制
- 提供命令发送和状态查询功能

### 标定(`calibration/calib.py`)

- 相机-机械臂坐标系标定
- 计算 2D 仿射变换矩阵
- 支持手动添加标定点和自动计算

## 配置和自定义

- **语音意图配置**:修改 `utils/data/` 下的 JSON 文件
- **模型配置**:替换 `models/` 目录下的模型文件
- **串口配置**:在 `utils/command.py` 中修改默认端口和波特率
- **标定数据**:运行标定程序更新 `calibration/calib_matrix_2d.json`

## 注意事项

- 确保机械臂串口连接正确(默认 `/dev/ttyUSB0`)
- 摄像头需要正确配置(默认设备 `0`)
- 首次运行时可能需要标定相机外参

