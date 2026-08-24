# 可追溯工作日志

这份日志记录项目实际执行过程，为最终项目总结、README 和面试材料提供事实来源。不得用计划中的结果替代真实运行结果。

## 2026-08-09：环境检查

- 工作区起初为空，且不是 Git 仓库；已运行 `git init`，尚未创建 commit。
- 本地 GPU：NVIDIA GeForce RTX 5060 Laptop GPU，8,151 MiB。
- Base Python：Anaconda Python 3.9.13，没有 PyTorch。
- 发现已有 `env_isaaclab`：Python 3.11.14、PyTorch 2.7.0+cu128，可识别 CUDA GPU。
- WSL 查询返回 access denied，因此本地流程按 Windows 原生设计。

## 2026-08-09：Gate 0 合成闭环

创建合成三维 reach 数据、Horizon window、Transformer action queries、masked Smooth-L1 loss、checkpoint 和闭环 rollout。

真实运行结果：

```text
train_samples=4800, val_samples=1200
best_val_loss=0.002561
50 rollouts, success_rate=1.000, mean_steps=17.06
```

该任务仅验证工程链路，不作为简历主结果。

## 2026-08-09：ManiSkill 本地安装

- 在仓库内创建 `.venv`，通过 `--system-site-packages` 复用现有 PyTorch。
- 固定安装 `mani_skill==3.0.1` 和 `sapien==3.0.3`。
- Windows 上成功创建 `PickCube-v1` CPU 环境：状态观察 42 维，动作 8 维。
- 官方文档说明 Windows 支持 CPU 仿真和渲染，但不支持 GPU simulation；正式并行仿真计划放在 Linux 服务器。

## 2026-08-09：官方数据接入

- 下载 PickCube 官方演示，约 36.6 MB。
- 将 100 条 motion-planning 轨迹以 `obs_mode=state` 重放，100/100 保存成功。
- 实现 HDF5 lazy dataset，确保按完整 episode 划分训练/验证集。
- 添加 observation history、action chunk 和 episode 尾部 padding mask。

## 2026-08-09：第一次真实闭环失败

运动规划数据使用 `pd_joint_pos` 绝对关节目标：

```text
80 train episodes, 20 validation episodes
best_val_loss=0.001215
20 rollouts, success_rate=0.000
```

诊断观察：策略输出绝对关节目标时闭环振荡，未能形成稳定抓取。这个实验说明低 validation loss 不保证闭环成功。

## 2026-08-09：更换动作表示和数据来源

- 尝试 `pd_ee_delta_pose`，但 Windows CPU 控制器依赖当前缺失的 Pinocchio，环境创建失败；该控制模式留待 Linux 服务器。
- 改用官方 RL expert 的 `pd_joint_delta_pos` 数据。
- CPU 重放 200 条，199 条保存成功，1 条回放失败并被剔除。

未归一化训练结果：

```text
160 train episodes, 39 validation episodes
20 rollouts, success_rate=0.100
```

## 2026-08-09：严格训练集归一化

- 只用训练 episodes 计算 observation/action mean 和 std。
- validation 和 rollout 共用训练统计量。
- 模型输出在执行前反归一化，并裁剪到环境 action space。
- 统计量随 checkpoint 保存。

Action-Chunking Transformer 真实结果：

```text
parameters=604,168
best_val_loss=0.009097
100 rollouts, success_rate=0.900, mean_steps=24.09
failed seeds=30014, 30020, 30022, 30027, 30031,
             30033, 30052, 30076, 30088, 30092
```

## 2026-08-09：Single-step MLP 基线

使用相同数据 split、归一化和 rollout seeds，只将模型改成：

```text
[B, 2, 42] -> flatten [B, 84] -> 3-layer MLP -> [B, 1, 8]
```

真实结果：

```text
parameters=156,936
best_val_loss=0.006754
100 rollouts, success_rate=0.750, mean_steps=28.52
```

初步差异为 15 个百分点，尚需三个训练 seed 才能形成正式结论。

## 当前下一步

1. 对失败 seeds 录制视频并分类：未抓住、抓住后掉落、放置不准、超时或振荡。
2. 将 MLP 和 Transformer 分别运行三个训练 seed，汇总均值和标准差。
3. 接入 RGB 图像观察，建立视觉 BC 和视觉 action chunking 基线。

## 2026-08-09：可视化审计

- 确认此前所有策略 rollout 使用真实 ManiSkill/SAPIEN 物理环境，但 `render_mode=None`，因此没有保存可视化结果。
- 本地以 `rgb_array` 和 minimal shader 两次测试渲染；环境创建和 reset 成功，但 `render_camera.get_picture` 发生 Windows native access violation。
- 已新增 `record_rollouts.py`，使用 ManiSkill `RecordEpisode` 保存指定 seed 视频；该脚本待 Linux 服务器运行。
- 已新增状态级失败分类，记录每个失败 seed 是否曾抓住、是否曾到达目标、最终是否静止。

## 2026-08-09：云 GPU 选型

- 当前目标是 Linux/Vulkan 录像验收与后续单卡视觉策略，不需要课题组的 8 卡 4090。
- 推荐配置为单张 RTX 4090 24 GB、10–16 vCPU、64 GB 内存、100–200 GB 磁盘；仅录制当前状态策略时，RTX 3090 24 GB 已足够。
- 公开价格对比和完整租用、验收、上传、录像、下载、停机步骤已写入 `docs/cloud_gpu_guide.md`。
- 是否正式选定平台、实际主机规格、Vulkan 验收结果、实付费用和视频产物尚待执行，后续不得把本次调研写成已完成实验。

## 2026-08-10：课题组服务器只读体检

- 项目已上传至 `/home/zhm/embodied-policy-lab`；录像所需代码、配置和 `outputs/pickcube_state_delta/best.pt` 均存在，上传目录约 2.6 MB，checkpoint 约 2.4 MB。
- 节点名为 `node02`，可见 8 张 RTX 4090 24 GB。体检时 GPU 0–3 基本空闲；GPU 4–7 各被一个 Python 进程占用约 22–23 GB。此状态只是时间点快照，不等于 GPU 0–3 已被分配给本项目。
- `/usr/local/bin` 下存在 `srun`、`sbatch`、`sinfo`，说明在占用 GPU 前必须进一步确认课题组的调度规则。
- 用户 Miniconda 可用，但默认 Python 为 3.14.6；公共环境中未发现 PyTorch、ManiSkill 或 Gymnasium。本项目需要创建个人 Python 3.11 隔离环境，不能安装进公共/默认环境。
- `vulkaninfo` 工具未安装，但 NVIDIA Vulkan ICD `/usr/share/vulkan/icd.d/nvidia_icd.json` 存在。当前只能判定“具备候选驱动配置”，尚未证明渲染可用；最终以 ManiSkill 实际 RGB 帧测试为准。
- 后续实际调用确认 Slurm 版本为 24.11.6，但 `sinfo` 和 `squeue` 均无法连接 Slurm controller。因此不能把“安装了 Slurm 命令”视为调度系统可用，也不能据此自行占用空闲卡；需向课题组确认 `node02` 的实际用卡规则。
- `/home` 文件系统总计 10 TB、剩余约 7 TB。个人 Conda 环境 `embodied-policy` 已成功创建，Python 为 3.11.15，路径为 `/home/zhm/miniconda3/envs/embodied-policy/bin/python`；PyTorch 仍在下载安装中。
- 课题组确认该节点直接通过 `CUDA_VISIBLE_DEVICES` 选择 GPU，不依赖当前不可用的 Slurm controller；每次运行前仍需重新检查占用并只暴露一张空闲卡。
- PyTorch 2.7.0+cu128 官方 wheel 约 1.1 GB，服务器到 `download.pytorch.org` 的下载速度一度仅约 12.8 KB/s，已取消该下载。现有环境审计发现 `occaffogato` 使用 Python 3.10.20、PyTorch 2.7.1+cu126，可作为兼容性和应急克隆来源；其他环境分别使用 PyTorch 2.4.0/2.5.0。
- Pip 全局索引已配置为清华 PyPI 镜像。下一方案是在干净的 Python 3.11 环境中通过该镜像安装 PyTorch 2.7.1，并在安装前用 `--dry-run` 确认解析到 CUDA 12.x 依赖；尚未把此方案记录为安装成功。
- 2026-08-11：用户在 `(base)` 环境执行 Torch CUDA 检查，得到 `ModuleNotFoundError`。该结果只说明 base 环境没有 Torch，不能作为 `embodied-policy` 环境安装失败的证据；后续必须先激活目标环境并记录 `which python`。
- 2026-08-11：确认 `/home/zhm/miniconda3/envs/embodied-policy/bin/python` 已安装 Torch 2.13.0；虽然 shell 提示符仍显示 `(base)`，`which python` 已指向目标环境，因此以解释器路径为准。该 Torch 构建声明依赖 CUDA 13 系列；尚待 `torch.cuda.is_available()` 验证其对节点 4090 的可用性。
- 2026-08-11：服务器环境安装完成并通过 6 个单元测试。实际版本为 ManiSkill 3.0.1、SAPIEN 3.0.3、Gymnasium 1.3.0；测试覆盖数据集窗口/episode 边界、Transformer 与 MLP 前后向、动作 padding mask，结果为 `Ran 6 tests ... OK`。尚待 CUDA 可用性和 Linux RGB 渲染验收。
- 2026-08-11：ManiSkill 最小离屏渲染成功：在 `PickCube-v1`、`state` 观察、`pd_joint_delta_pos` 控制与 `physx_cpu` 仿真后，`env.render()` 返回 `Tensor`，形状为 `[1, 512, 512, 3]`。首维是批量环境数 1，后三维是 RGB 图像；因此服务器端 RGB 观测/视频链路的渲染部分已验证。
- 2026-08-11：首次尝试服务器闭环评测时得到 `No module named embodied_policy.evaluate`。定位到上传到服务器的 `embodied_policy/` 目录缺少 `evaluate.py`（同时也缺少未来训练会用到的 `train.py`），属于代码同步不完整，而非 Python 包、Torch 或 GPU 错误。补传源文件到 editable 项目目录后无需重装环境。
- 2026-08-11：补传 `evaluate.py`、`train.py` 与配置所需的状态轨迹 HDF5 后，服务器闭环评测成功完成。100 episodes 的成功率为 0.900、平均步数为 24.15；失败 seed 和本地结果一致（10 个），失败类别全为 `grasped_but_never_placed`。至此，环境安装、RGB 渲染和 checkpoint 闭环推理链路均已在 Linux 服务器验证；下一步为录像与人工失败归因。
- 2026-08-11：首次录像时，ManiSkill 3.0.1 的 `RecordEpisode` 在 `info_on_video=True` 路径将单环境的 batched reward 转为 Python float，触发 `TypeError`。已将 `record_rollouts.py` 的 info overlay 关闭，视频文件名仍记录 seed 和 success/failure，详细状态仍在评测 JSON 中；本地语法检查和 6 个单元测试通过。待同步到服务器后重录。
- 2026-08-11：关闭 overlay 后，成功 seed `30007` 已保存为具名 MP4；3 个失败 seed 跑满 50 步时被 `RecordEpisode.max_steps_per_video=50` 自动保存为匿名编号文件。已将自动切片禁用，改为只由每个 episode 结束后的显式 `flush_video(name=seed_outcome)` 保存；本地语法检查和 6 个单元测试通过。待同步脚本后重录 4 个具名视频。
- 2026-08-11：录像重试成功，服务器生成 4 条具名 MP4：`seed_30007_success.mp4`（16 步）以及 `seed_30014_failure.mp4`、`seed_30020_failure.mp4`、`seed_30022_failure.mp4`（各 50 步）。视频尚待下载和人工视觉归因；在完成审阅前不得将 failure category 细化为掉落、目标偏差或静止条件问题。
- 2026-08-11：本地视频元数据与关键帧审计完成。成功 MP4 实际为 17 帧/0.85 秒，失败 MP4 各为 51 帧/2.55 秒，均为 512×512、20 FPS；成功短是第 16 步早停，不是单帧编码。失败视频显示抓取与向目标搬运成功，但仅凭单相机画面不能量化目标距离、高度或夹爪约束状态；将补充逐步 telemetry 后再做细分归因。
