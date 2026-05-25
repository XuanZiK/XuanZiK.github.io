---
title: "LeRobot ACT - 单臂抓取任务全流程"
excerpt: 基于 LeRobot 框架与 ACT 模型的单臂物体抓取项目，涵盖遥操数据采集、HDF5→LeRobot 格式转换、模型训练、模拟推理与实机部署的完整链路。
collection: projects
---

<div style="display: flex; flex-wrap: wrap; gap: 1em; margin: 0 auto 1.5em auto; max-width: 900px;">
  <video controls preload="metadata" style="flex: 1 1 360px; width: 100%; max-width: 100%;">
    <source src="/files/videos/pick_cube_demo1.mp4" type="video/mp4">
    你的浏览器不支持 video 标签。
  </video>
  <video controls preload="metadata" style="flex: 1 1 360px; width: 100%; max-width: 100%;">
    <source src="/files/videos/pick_cube_demo2.mp4" type="video/mp4">
    你的浏览器不支持 video 标签。
  </video>
</div>

## 项目描述

本项目在 **LeRobot 框架** 下完成一个单臂"抓取并放置"任务（pick_cube）的完整闭环：从遥操数据采集、格式转换、ACT 模型训练，到模拟推理验证与实机部署调参。任务相对简单，仅采集 50 条数据即可达到不错的效果。

## 1. 数据采集

任务初期发现 LeRobot 自带的遥操控制频率偏低，主从臂操作明显卡顿，会显著影响采集速率与数据质量。因此**另外开发了一套数采代码**，便于排查问题、提高采集帧率。

### Teleop 文件夹结构

```
Teleop/
├── convert_data/                      # 转换好的数据（lerobot 格式）
├── data/                              # 原始录制数据（hdf5 格式）
├── data1/                             # 备份数据目录
├── hdf5_to_lerobot_converter/         # HDF5 → LeRobot 3.0 转换工具
├── onero_description/                 # 机械臂模型文件
├── teleop_config.yml                  # 遥操作配置文件（遥操、录制共用）
├── teleop_controller.py               # 主从臂遥操函数与工具
├── teleop_main.py                     # 遥操主程序
├── teleop_record_hdf5.py              # 数据采集主程序
└── visualize_hdf5.py                  # 数据可视化检查
```

> **数据量参考**：本任务比较简单，**仅采集 50 条数据**就足以训练出可用模型。复杂任务可按需加量。

## 2. 模型训练

训练前需要安装好 LeRobot，可参考以下教程完成完整环境配置：

- 使用开源数据在云服务器训练 ACT
- LeRobot (ROS2) ACT 本地数采-本地训练-单机部署完整流程
- lerobot-ACT 本地训练部署流程

### 训练命令

参数可以直接命令行传入，也可以写到配置文件里。训练过程用 **wandb** 实时监控。

```bash
lerobot-train \
  --dataset.repo_id=test \
  --dataset.root=/home/woan/Teleop2/convert_data/pick_cube \
  --dataset.revision=v3.0 \
  --dataset.streaming=false \
  --policy.type=act \
  --output_dir=output_lerobot_train/act \
  --job_name=pick_cube \
  --policy.device=cuda \
  --wandb.enable=true \
  --wandb.project=Lerobot_xuanzi_Project \
  --policy.push_to_hub=false \
  --steps=50000 \
  --batch_size=16
```

ACT 模型完整参数详见：

```
~lerobot/src/lerobot/policies/act/configuration_act.py
```

不同任务可在此基础上调优。

### 模拟推理（部署前验证）

实机部署前先用已有视频对训练出的模型做模拟推理。**好处是部署遇到问题时，可以优先排除"模型本身是否训坏"这一项**。

```bash
python simulate_episode_video_inference.py \
  --model-path output_lerobot_train/act/checkpoints/last \
  --episode-index 5 \
  --max-frames 600 \
  --export-compare-video outputs/ep5_compare.mp4 \
  --save-actions outputs/ep5_actions.npy
```

观察导出的 compare 视频，**模型预测轨迹与真实轨迹的运动趋势一致** → 可以进入下一步部署。

## 3. 模型部署

### 3.1 模型文件结构

每 20k step 保存一次 checkpoint

### 3.2 部署配置与运行

推理用 `record_config.yml`。**ACT 模型参数量小，普通笔记本即可推理**。本任务只用了一个 RGB 相机（顶部 `head` 位置）.
```bash
lerobot-record --config_path=tmp/eval_config.yml
```

### 3.3 机械臂初始位姿

部署前必须确认初始位姿：

```yaml
home_joints_positions: [0, 0., 0, 0, 0., 0, 0]
```

默认全 0 → 机械臂初始化为**垂直向下**姿态。

```yaml
ready_waypoints:
  - [-0.523, -1.568,  0.001, -0.006,  0.127,  0.161, 0.000]
  - [-0.404, -1.451,  0.379,  1.531,  0.116, -0.150, 0.000]
  - [-0.291, -0.195,  0.066,  1.736,  0.011, -0.325, 0.000]
```

机械臂初始化后会**依次经过这三个航点**到达准备姿态。


### 3.4 部署参数微调（关键）

`eval_config.yml` 里有两行关键参数：

| 参数 | 默认 | 调优后 |
|---|---|---|
| `n_action_steps` | `80` | `1` |
| `temporal_ensemble_coeff` | `null`（不启用） | `0.01` |


**调优后**：
- 开启 ACT 的 **时序融合机制**（`temporal_ensemble_coeff = 0.01`）
- `n_action_steps` **必须设为 1**（与时序融合互斥）
- 系数也可以是 `-0.01`，但实测效果不如 `0.01`

调优后**推理异常丝滑**，机械臂动作明显更连贯。

## 4. 总结

### 任务总结

对于**任务简单、时长较短**的抓取放置任务，即使在杂乱环境下（不推荐，会拖累泛化），仅 50 条数据就能取得不错的效果。本次任务抓取的是茶水间的溜溜梅塑料桶——**反光效果较强**，推测 ResNet 对其纹理特征提取明显，间接帮助了视觉定位。

### 问题分析：位置泛化弱

实测发现：**当物块放在没采集过的位置时，机械臂无法到达**。

这是**模仿学习的通病**——在模型层面调整收益甚微。

| 解决方向 | 具体做法 |
|---|---|
| **短期** | 采集更完善的数据集，让抓取分布覆盖整个桌面；或对现有数据做增强 |
| **长期** | 增加腕部相机（**头部 + 腕部双相机**）以提供闭环视觉；或更换更强的模型（如 VLA 类） |
