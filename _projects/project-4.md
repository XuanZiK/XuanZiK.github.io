---
title: "X1 打乒乓球 - ACT & VLASH π0.5 项目"
excerpt: 一个关于X1机器人打乒乓球的项目，使用了两种主流的vla模型训练
collection: projects
---

<video controls preload="metadata" style="width: 100%; max-width: 800px; display: block; margin: 0 auto 1.5em auto;">
  <source src="/files/videos/X1_ACT_pingpong_demo.mp4" type="video/mp4">
  你的浏览器不支持 video 标签。
</video>

## 项目描述

本项目在开展了两条路线，第一条是用**模仿学习ACT**进行训练以及部署，第二条则选择了**VLASH（基于pi05**） 的LoRA微调，针对 **X1 7-DOF 进行的打乒乓任务**，这里数据采集了（30 Hz、151 episodes）。

## 路线一：ACT 模仿学习

### 关键改动（其余使用官方默认）

- **按数据集特点调整 ACT 参数**：根据本项目数据集（30 Hz、151 episodes、7-DOF）的统计特征调整 ACT 默认超参
- **图像编码器替换**：ACT 原版用一个**预训练且共享权重的 ResNet-18** 处理所有相机图像。但本项目重点关注乒乓球的走向，ResNet-18 的总下采样倍率 要来到32× ， 故改成 **4 层带残差的 CNN**，降低降采样倍率，以保留乒乓球这类小目标的细节
- **chunk size 调参（重点）**：经测试，应设为一次关键动作（如一次完整击球所用的帧数）的 **1/4 ~ 1/3**。过大会丢失反应时机，过小则会让动作不连贯
- **batch size 与学习率**：根据显存上限确定 batch size，并相应调整学习率
- **部署时机**：当 loss 收敛、或下降速度已经非常缓慢后再部署，避免欠拟合

## 路线二：VLASH (π0.5) LoRA 微调

> 配置来源：
> - 训练（带深度）：`x1_pingpong_lora.yaml`
> - 训练（纯 RGB）：`x1_pingpong_lora_rgbonly.yaml`
> - 推理：`inference_pingpong.yaml`
> - 模型默认：`vlash/policies/pi05/configuration_pi05.py`

### 1. 模型骨架

| 参数 | 值 | 作用 |
|---|---|---|
| `paligemma_variant` | `gemma_2b` | 视觉-语言主干（PaliGemma + SigLIP），≈3B 参数 |
| `action_expert_variant` | `gemma_300m` | 动作专家小模型 |
| VLM hidden / inter / layers | 2048 / 16384 / 18 | Gemma-2B 文本侧 |
| Action expert hidden / inter / layers | 1024 / 4096 / 18 | 动作侧 |
| `image_resolution` | 224×224 | 网络输入分辨率（原始相机 480×640 / 320×240 通过 resize-with-pad 进入） |
| `max_state_dim` / `max_action_dim` | 32 / 32 | 状态/动作向量都 pad 到 32 维（X1 是 7-DOF，剩 25 维 pad） |
| `tokenizer_max_length` | 200 | 任务文本最长 token 数 |
| `dtype` | `bfloat16` | 计算精度 |
| `use_adarms`（action expert） | `True` | **π0.5 关键设计**：用 AdaRMSNorm 把状态/时间作为条件注入 |

### 2. 动作生成 / Flow Matching（π0.5 默认）

| 参数 | 值 | 作用 |
|---|---|---|
| `chunk_size` | 50 |
| `n_action_steps` | 50 |
| `num_inference_steps` | 10 |
| `time_sampling_beta_alpha / beta` | 1.5 / 1.0 | 训练时 t 服从 Beta(1.5, 1.0)，偏向高噪声 |
| `time_sampling_scale / offset` | 0.999 / 0.001 |
| `min_period / max_period` | 4e-3 / 4.0 |

### 3. 任务输入定制

| 参数 | 值 | 备注 |
|---|---|---|
| `policy.type` | `pi05` | |
| `pretrained_path` | `/mnt/data/pi05_base` |
| `state_cond` | `true` | **打开 π0.5 的状态条件 AdaRMSNorm**，作者说"显著提升稳定性"，对乒乓这种状态敏感任务尤其重要 |
| `empty_cameras` | `2` | 补 2 个空相机占位（left_wrist + right_wrist），与 pi05_base 预训练数据布局对齐，避免分布漂移 |
| `dataset.repo_id / root` | `local/x1_pingpong` / `/mnt/workspace/data/20260226` | LeRobot v3.0 本地数据：151 episodes、30 Hz、7-DOF |
| `single_task` | 不传（任务文本写在 `tasks.parquet`） | |
| `normalization` | STATE / ACTION → `MEAN_STD`，VISUAL → `IDENTITY` | π0.5 默认（注释写 `MEAN_STD` 比 `QUANTILES` 更稳） |

### 4. 训练超参（**微调**）

| 参数 | 带深度版 / RGB-only 版 | 备注 |
|---|---|---|
| `batch_size` | 1 / 2 | RGB-only 翻倍利用显存 |
| `grad_accum_steps` | 4 / 2 | 全局有效 batch 都是 16（4 卡 × 4 / 4 卡 × 2 × 2） |
| `steps` | 20000 | 比 vlash 默认 50000 缩短，因为只 43k 帧 |
| `num_workers` | 4 | |
| `seed` | 1000 | |
| `optimizer.type` | `adamw` | |
| `optimizer.lr` | `5e-5` | 比 π0.5 默认 `2.5e-5` 高 1 倍（LoRA 微调常见做法） |
| `optimizer.betas` | `[0.9, 0.95]` | π0.5 默认 |
| `optimizer.weight_decay` | `1e-10` | **几乎关掉**（避免把低秩适配器拉回零） |
| `scheduler.type` | `cosine_decay_with_warmup` | |
| `num_warmup_steps` | 500 | 比默认 1000 短，配合短训练 |
| `peak_lr → decay_lr` | `5e-5 → 2.5e-6` | 衰减 20×，标准 cosine |
| `num_decay_steps` | 20000 | 与 steps 对齐 |
| `save_freq` / `log_freq` | 2000 / 100 | 多存 checkpoint 方便挑 |

### 5. VLASH 特有设计（论文特殊设计）

| 参数 | 值 | 备注 |
|---|---|---|
| `max_delay_steps` | **8** | **时延增广**：训练时 query 区间随机往后偏移 0~8 步，让模型学会"基于陈旧观测预测未来动作"，这是异步推理无开销的前提 |
| `shared_observation` | **true** | **共享观测**：一次前向把 9 个 offset 分支并行算（自定义注意力 mask 防跨支泄露），相比朴素做法约 9× 提速 |
| 异步推理（推理 yaml） | `inference_overlap_steps: 3` | 推理重叠 3 步，覆盖 ≈38 ms 延迟 |
| 动作量化（推理 yaml） | `action_quant_ratio: 1` | 这里没开（= 1 表示不量化） |

### 6. LoRA 配置（参照官方微调）

| 参数 | 值 | 备注 |
|---|---|---|
| `enable` / `backend` | `true` / `peft` | 使用 HuggingFace PEFT |
| `r` | 16 | 低秩矩阵秩 |
| `alpha` | 16 | 缩放系数（α/r = 1.0） |
| `dropout` | 0 | LoRA 层不 dropout |
| `target_modules`（**注入 LoRA 的层**） | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, out_proj, fc1, fc2` | 覆盖 Gemma（`*_proj`）和 SigLIP（`out_proj / fc1 / fc2`）所有线性层 |
| `extra_trainable_modules`（**全量训练的层**） | `action_in_proj, action_out_proj, time_mlp_in, time_mlp_out, state_proj, state_mlp_in, state_mlp_out, embeddings, input_layernorm, post_attention_layernorm` | **核心点**：动作/状态/时间投影 + LayerNorm + embeddings 全量训。任务相关的"非 VLM 部分"留给全参更新，VLM 大头用 LoRA |
| `use_qlora` | `false`（默认） | 没开 4-bit 量化 |

### 7. 推理端关键参数

| 待续..
