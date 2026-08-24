# 项目一：Embodied Policy Lab 路线图

## 最终目标

构建一套从数据到闭环评测的视觉模仿学习实验框架，并围绕动作分块策略完成可复现的基线、消融和失败分析。最终成果应包含代码、模型权重、实验表格、演示视频和一份论文式技术报告。

## Gate 0：训练闭环（已完成）

合成三维到达任务用于验证工程链路，不作为最终简历成果。

- 轨迹生成、历史观测窗口和未来动作块
- Transformer action queries
- padding mask 与 masked Smooth L1 loss
- checkpoint 与 JSONL 指标
- 未见初始状态的闭环 rollout

首次本地验证：69K 参数，4,800 个训练样本，50 次 rollout 成功率 100%。

## Gate 1：机器人状态策略（工程闭环完成；规范实验待完成）

- 安装并固定 ManiSkill 版本
- 跑通 PickCube-v1 环境及演示数据
- 编写统一 trajectory adapter
- 只用机器人本体状态训练单步 BC
- 评测至少 100 次 rollout，并保存失败种子

当前进展：已完成 PickCube 官方演示下载、CPU 重放和 HDF5 adapter；Linux 服务器已完成 RGB 离屏渲染，并保存 1 条成功和 3 条失败 MP4。固定数据划分和测试集的三训练 seed 主基线为：Action-Chunking Transformer 92.7% ± 3.1%，Single-step MLP 77.3% ± 5.5%。

尚未完成：对 `action_horizon` 与 `replan_interval` 的严格消融。三个独立训练 seed、固定且独立于训练 seed 的评测集、逐步失败 telemetry 均已完成；当前结果可作为该单任务状态基线的主结论，但不能外推为视觉策略或跨任务结论。

验收：同一条命令可完成训练和独立评测；随机策略、脚本专家和学习策略指标可对比。

## Gate 2：视觉与动作分块

- 增加 RGB 编码器和状态融合
- 实现 single-step BC 与 chunked policy 两条基线
- 加入 action normalization、temporal ensembling 和重规划周期
- 记录训练时间、峰值显存、推理延迟和任务成功率

验收：在至少三个操作任务上进行公平比较，并能解释主要失败模式。

## Gate 3：规范实验

- 每项主结果三个随机种子
- 消融 action horizon、observation horizon、视觉编码器、数据量和重规划周期
- 测试位置、纹理、光照或物理参数的分布外泛化
- 自动汇总均值、标准差和置信区间

验收：形成一张主结果表、两张消融图和一套失败案例分类。

## Gate 4：研究升级

项目二在此时根据 Gate 1--3 暴露的问题选题。候选方向：

1. action-conditioned latent world model，用于失败预测或动作块筛选；
2. 小型 VLA 的高效微调与指令/场景泛化；
3. 视觉与接触力/触觉融合，用于插入、滑移和恢复任务。

只有当候选方法能回答明确问题、拥有可靠基线和可承担的数据成本时才立项。
