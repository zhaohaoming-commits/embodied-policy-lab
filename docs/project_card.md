# 项目卡片：Embodied Policy Lab

## 一句话介绍

在 ManiSkill 的 PickCube 任务中实现并评估闭环行为克隆策略，重点检验“离线动作预测好”是否真的能转化为“机器人能稳定抓取和放置方块”。

## 问题与设置

- **任务：** Panda 机械臂抓起方块并放到目标区域。
- **数据：** ManiSkill 官方 RL expert demonstrations；并非自己采集，也不是实物机器人数据。
- **输入：** 主结果使用 42 维仿真状态（关节、夹爪/末端、物体、目标及相对位置等）。
- **输出：** 8 维关节增量/夹爪动作；主模型每次预测未来 8 步。
- **评价：** 固定 100 个未见初始条件的闭环 rollout 成功率，而不只看 validation loss。

## 我做了什么

1. 将轨迹按**完整 episode**划分为训练和验证，再生成时间窗口，避免相邻状态泄露。
2. 实现历史状态输入、future action chunk、末端 padding mask 与训练集统计量归一化。
3. 实现 action-chunking Transformer，并用 single-step MLP 作为对照基线。
4. 编写训练、评估、三 seed 批量实验、失败遥测和 rollout 视频录制流程。
5. 对 action horizon、重规划周期及视觉输入进行受控消融，并保留负结果。

## 核心结果

固定 160/39 训练/验证 episode、100 个测试 rollout 和训练 seed 7/17/27：

| 方法 | 成功率 | 平均步数 |
|---|---:|---:|
| Action-chunking Transformer（8 步目标） | **92.7% ± 3.1%** | **24.04 ± 0.22** |
| Single-step MLP | 77.3% ± 5.5% | 27.95 ± 1.24 |

更重要的部署消融：对同一个 8-step checkpoint，`replan_interval=1` 时为 **92.7%**，连续开环执行 8 步时降至 **58.3%**。

## 能成立的结论

- 在该固定 PickCube 设置中，主 Transformer 的闭环表现优于 single-step MLP。
- 低 validation loss 不足以预测闭环成功率：MLP 的 validation loss 更低，但 rollout 成功率更低。
- 高频闭环反馈很关键；动作块适合作为时序监督，不代表应该完整开环执行。

## 不能夸大的结论

- 8-step 相比 1-step 仅高约 1 个百分点，不能声称 action chunk 显著更强。
- RGB-only 与 RGB+proprioception 没有训练出可用策略；这不是成功的视觉策略项目。
- 结果仅覆盖一个仿真任务和一个固定训练/验证划分，不能外推为通用机器人能力。

## 面试版 45 秒讲法

> 我做了一个 ManiSkill PickCube 的闭环模仿学习实验。数据来自官方专家轨迹，我先按完整 episode 划分训练和验证以避免时间窗口泄露。主模型是 action-chunking Transformer：输入最近两步状态，预测未来八步动作，但部署时每执行一步就重新观测并规划。在固定一百个测试任务、三个训练随机 seed 下，它的成功率是 92.7%±3.1%，高于单步 MLP 的 77.3%±5.5%。我还发现预测八步并不意味着应连续执行八步；把重规划间隔从 1 提到 8，成功率会降到 58.3%，说明闭环反馈才是关键。视觉输入消融没有成功，我把它作为负结果保留，并据此把下一阶段转向动作条件预测和失败预警。
