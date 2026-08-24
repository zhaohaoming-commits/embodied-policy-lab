# 学习指南：你需要真正理解这个项目的什么

本文档随项目逐步更新。最终面试时，你应能脱离代码解释每一节，而不只是运行命令。

## 1. 我们在解决什么问题

任务是 PickCube：Panda 机械臂需要抓起桌面上的方块，并移动到随机目标位置。我们使用模仿学习，让模型根据专家轨迹学习从观察到动作的映射。

模仿学习的数据可以写成：

```text
(observation_t, action_t)
```

训练阶段类似监督学习；真正困难的是部署后模型产生的动作会改变下一时刻观察，因此一个小错误可能不断累积。这叫 covariate shift（协变量偏移）。所以验证集 action loss 很低，也不等于闭环任务成功。

## 2. 当前数据长什么样

数据来自 ManiSkill 官方 PickCube 演示，保存为 HDF5。每个 `traj_n` 是一条完整 episode，主要字段是：

- `obs`: `[T + 1, 42]`，机器人和任务状态；
- `actions`: `[T, 8]`，七个机械臂关节控制量和一个夹爪控制量；
- `success`: 每个时间步是否满足成功条件；
- `terminated` / `truncated`: episode 如何结束。

训练集和验证集必须按完整 episode 划分。若先把时间窗口打散再划分，相邻状态可能同时出现在训练集和验证集，造成数据泄漏。

## 3. Action Chunking 数据窗口

当前42维 state observation 的真实组成是：

| 字段 | 维度 | 含义 |
|---|---:|---|
| `agent.qpos` | 9 | 7个机械臂关节位置和2个夹爪关节位置 |
| `agent.qvel` | 9 | 对应关节速度 |
| `extra.is_grasped` | 1 | 是否抓住方块 |
| `extra.tcp_pose` | 7 | 末端位置与四元数姿态 |
| `extra.goal_pos` | 3 | 目标位置 |
| `extra.obj_pose` | 7 | 方块位置与四元数姿态 |
| `extra.tcp_to_obj_pos` | 3 | 末端到方块的相对向量 |
| `extra.obj_to_goal_pos` | 3 | 方块到目标的相对向量 |

当前8维 action 是7个机械臂关节增量控制量和1个夹爪控制量。

Transformer 数据窗口设置：

```text
最近 2 帧 observation: [batch, 2, 42]
未来 8 步 action:       [batch, 8, 8]
有效动作 mask:          [batch, 8]
```

episode 尾部可能不足八个未来动作，所以不足部分补零，并用 mask 保证补零位置不参与 loss。

## 4. 为什么必须归一化

42个观察维度具有不同量纲和数值范围。若直接训练，大范围特征可能主导梯度。我们只使用训练 episode 计算：

```text
normalized_x = (x - train_mean) / train_std
```

动作同样归一化；模型输出后再反归一化成环境需要的控制量。验证集和评测环境绝不能参与统计量计算，否则会泄漏信息。

第一次实验中，未归一化策略闭环成功率为10%，加入训练集归一化后，在同一任务的100次初步评测中达到90%。这说明数据预处理不是装饰，而是策略能否正常工作的关键部分。

## 5. 为什么还要做 MLP 基线

一个复杂模型获得高分不能证明复杂结构有效。Single-step MLP BC 使用相同的数据、episode split、归一化和评测 seed，只把模型改为：

```text
[B, 2, 42] -> flatten [B, 84] -> MLP -> [B, 1, 8]
```

比较结果能回答：Action Chunking Transformer 是否真的优于普通单步行为克隆？如果 MLP 同样优秀，我们就不能把成功归功于 Transformer。

## 6. 当前必须能回答的问题

1. 为什么 `obs` 比 `actions` 多一个时间步？
2. 为什么训练/验证要按 episode 划分？
3. `action_mask` 解决了什么问题？
4. 为什么验证 loss 很低仍可能闭环失败？
5. 为什么归一化统计量只能来自训练集？

## 7. 仿真如何判定成功与失败

PickCube 最多运行50步。ManiSkill 在每一步返回：

- `is_grasped`：夹爪是否抓住方块；
- `is_obj_placed`：方块与目标的距离是否不超过0.025米；
- `is_robot_static`：机器人关节速度是否低于0.2；
- `success = is_obj_placed AND is_robot_static`。

若50步内从未出现 `success=True`，该 rollout 记为失败。评估器还会输出 `rollout_telemetry.json`：每步包含物体—目标距离、物体/目标高度、夹爪命令、机械臂动作幅度及三个环境标志。我们据此将失败初分为：从未抓住、途中掉落（曾抓住但最终未抓住）、始终夹持却未进入放置条件、到过目标但未稳定完成。视频用于验证和细化这些自动标签。

## 8. 当前可视化状态

物理仿真和数值评测已在本地 ManiSkill/SAPIEN 中真实运行。此前为了加快100次评测使用 `render_mode=None`，没有生成视频。本地 Windows 的 `rgb_array` 测试在 SAPIEN `render_camera.get_picture` 中发生原生 access violation；视频录制工具已经实现，将在 Linux 服务器执行。这个渲染问题不能被描述成策略或物理仿真失败。
