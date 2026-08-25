# Embodied Policy Lab

> A reproducible closed-loop imitation-learning study for robotic PickCube manipulation in ManiSkill.

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](environment.yml)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.13-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Simulator](https://img.shields.io/badge/Simulator-ManiSkill%203-5B43E7)](https://maniskill.readthedocs.io/)
[![Task](https://img.shields.io/badge/Task-PickCube-0F766E)](https://maniskill.readthedocs.io/en/latest/tasks/table_top_gripper/)

一个面向具身智能学习的可复现实验仓库：使用 ManiSkill 官方 PickCube 专家轨迹，训练状态条件行为克隆策略，并以闭环 rollout 而非离线 loss 作为主要评价标准。

**快速入口：** [项目卡片](docs/project_card.md) · [实验日志](docs/experiment_log.md) · [数据来源](docs/data_provenance.md) · [学习笔记](docs/learning_guide.md)

## Why this repository?

这个项目关注的不是“把 loss 降下来”，而是以下三个具身控制问题：

1. **数据是否可信？** 训练/验证按完整 episode 划分，避免相邻时间窗口泄露。
2. **策略是否真能控制机器人？** 在固定的 100 个未见初始条件上进行闭环 rollout。
3. **结论是否稳健？** 固定数据划分和评测任务，使用三个训练随机种子，并报告失败轨迹与负结果。

```mermaid
flowchart LR
    A[ManiSkill expert trajectories] --> B[Episode-level train / val split]
    B --> C[Window dataset<br/>2 observations → 8 actions]
    C --> D[Action-chunking Transformer]
    D --> E[Closed-loop PickCube rollout]
    E --> F[Success rate + telemetry + videos]
```

## Main result

在相同的 160/39 episode 训练/验证划分、固定 100 个闭环测试 episode 和三个训练随机种子下：

| Policy | Closed-loop success rate | Mean steps | What it shows |
|---|---:|---:|---|
| **Action-chunking Transformer (8 actions)** | **92.7% ± 3.1%** | **24.04 ± 0.22** | 主状态基线 |
| Single-step MLP (1 action) | 77.3% ± 5.5% | 27.95 ± 1.24 | 低离线 loss 不保证闭环成功 |
| Transformer, 1-action target | 91.7% ± 2.3% | 25.56 ± 0.58 | 8-step 监督略好，但证据不足以声称显著优势 |

对同一个 8-action Transformer checkpoint，仅改变部署时连续执行的动作数：

| Replan interval | 1 | 2 | 4 | 8 |
|---|---:|---:|---:|---:|
| Success rate | **92.7%** | 90.0% | 73.7% | 58.3% |

**Takeaway:** 预测动作块不代表应开环执行动作块；每执行一步就重新观测与规划，远比连续执行 4 或 8 步可靠。完整逐 seed 数据、实验条件和限制见 [实验日志](docs/experiment_log.md)。

## What is implemented

- Episode-level split、历史观测窗口、动作末端 padding mask 与仅训练集统计量归一化；
- Action-chunking Transformer 与 single-step MLP 基线；
- 训练、验证、最佳 checkpoint、固定 seed 批量实验与聚合 CSV；
- PickCube 闭环评估、失败遥测（距离、抓取状态、夹爪命令等）和指定 seed 视频录制；
- RGB+state、RGB-only、RGB+proprioception 视觉输入消融，以及预训练 ResNet-18 尝试。

## Honest scope and negative result

This is a **simulation** project using **official ManiSkill expert demonstrations**. It is not a physical-robot deployment and not a vision-language-action foundation model.

视觉部分目前没有获得可用的纯视觉策略：RGB-only 与 RGB+proprioception 在三个 seed 上均为 0% 成功率；预训练 ResNet-18 的一次试验仅为 1%，不构成有效改进。RGB + 完整 state 的 95.3% 结果含有物体/目标真值状态，不能用来证明视觉编码器有效。保留这些负结果是为了避免把特权状态带来的效果误写成视觉能力。

## Reproduce

### 1. Environment and tests

```bash
conda env create -f environment.yml
conda activate embodied-policy
python -m unittest discover -s tests -v
```

### 2. Train and evaluate the state baseline

The ManiSkill trajectory file is intentionally not committed. Follow [data provenance](docs/data_provenance.md) to obtain it, then run:

```bash
python -m embodied_policy.train --config configs/pickcube_state_delta.yaml
python -m embodied_policy.evaluate \
  --config configs/pickcube_state_delta.yaml \
  --checkpoint outputs/pickcube_state_delta/best.pt
```

### 3. Reproduce the three-seed comparison

```bash
python -m embodied_policy.run_experiments \
  --configs configs/pickcube_state_delta.yaml configs/pickcube_state_mlp.yaml \
  --train-seeds 7 17 27 \
  --output-root outputs/pickcube_seed_sweep
```

The runner saves a frozen config, checkpoint, rollout metrics and an aggregate CSV for every run. Outputs and source data are excluded from Git by design.

### Record rollout videos

```bash
python -m embodied_policy.record_rollouts \
  --config configs/pickcube_state_delta.yaml \
  --checkpoint outputs/pickcube_state_delta/best.pt \
  --seeds 30007 30014 30020 \
  --output-dir outputs/videos/transformer
```

## Repository map

```text
configs/                Experiment configurations
embodied_policy/data/   Episode-aware HDF5 datasets and normalization
embodied_policy/models/ MLP, action-chunking Transformer and vision policies
embodied_policy/train.py
embodied_policy/evaluate.py
embodied_policy/run_experiments.py
docs/                   Experiment log, data provenance and learning notes
tests/                  Unit tests for padding, episode boundaries and evaluation
```

## Project principles

1. Closed-loop task metrics matter more than training loss alone.
2. Train/validation splitting happens before temporal windowing.
3. A claim should include seeds, fixed evaluation conditions and limitations.
4. Failure cases and negative results are first-class experimental outputs.
