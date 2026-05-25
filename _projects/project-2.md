---
title: "Aubo 机械臂叠螺母"
excerpt: 在 Aubo 机械臂上用 π0.5 (VLA) 的 LoRA 微调实现螺母堆叠任务。
collection: projects
---

<video controls preload="metadata" style="width: 100%; max-width: 800px; display: block; margin: 0 auto 1.5em auto;">
  <source src="/files/videos/aubo_stack_nut_demo.mp4" type="video/mp4">
  你的浏览器不支持 video 标签。
</video>

## 项目描述

本项目使用 **Aubo 机械臂** 完成"叠螺母"任务，采用 **π0.5 (Pi-0.5 VLA)** 的 **LoRA 微调** 方案：基于 π0.5 预训练权重，在少量本任务数据上做适配后部署。

## 硬件与任务

| 项 | 内容 |
|---|---|
| 机械臂 | Aubo |
| 相机 | 末端顶部Realsense相机,单目RGB，不带深度 |
| 任务 | 从分散位置拾取螺母 → 按指定顺序堆叠 |

## 训练 & 部署要点

| 项 | 值 | 来源 |
|---|---|---|
| 基础模型 | π0.5（PaliGemma-2B + Gemma-300M action expert） | `weight_loader: pi05_base` |
| 微调方式 | LoRA（关 EMA：`ema_decay=None`）
| 模型 variant | `paligemma_variant=gemma_2b_lora`<br>`action_expert_variant=gemma_300m_lora` | openpi 默认 LoRA 模板 |
| LoRA r / α | **16 / 16**（注入 `attn` + `ffn`） ||
| 任务标识 / 指令 | `aubo_stacked_nuts` / *"Stack the nuts on the table together."* | `collect_data.sh` + 推理代码常量 |
| 数据规模 | 单条 episode `TIME_STEPS=2000`，30 Hz，state/action dim = 7 | `collect_data.sh`、`ros1_infer.py` |
| 训练步数 | **30,000** | openpi 默认 |
| 优化器 | **AdamW**（`clip_gradient_norm=1.0`） | openpi 默认 |
| 学习率 / scheduler | peak `5e-5`，warmup 10,000 步，之后 cosine（`decay_lr=5e-5`，等同保持不变） | openpi 默认 |
| Batch size | 256（openpi 默认） | _（待回忆确认）_ |
| 推理 `action_horizon` / `chunk_size` | 30 / 25 | 
| 部署架构 | ROS1 客户端 → WebSocket 连远端推理服务 | `ros1_infer.py` |


