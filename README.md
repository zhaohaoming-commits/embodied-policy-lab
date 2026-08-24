# Embodied Policy Lab

从零实现、训练和评测具身模仿学习策略。第一个里程碑使用一个可控的合成到达任务，验证完整的数据、模型、训练、checkpoint 与闭环 rollout 链路；随后接入 ManiSkill 机器人轨迹。

## 当前里程碑

- [x] 合成三维到达任务及专家轨迹
- [x] 历史观测与 action chunk 数据窗口
- [x] 从零实现 Transformer action chunk policy
- [x] masked action loss、训练、验证及 checkpoint
- [x] 闭环 rollout 成功率评测
- [ ] 接入 ManiSkill 演示数据
- [ ] 图像编码器与本体状态融合
- [ ] BC 与 action chunking 基线对比
- [ ] 多随机种子及消融实验

## 本地运行

仓库内的 `.venv` 已隔离安装 ManiSkill 3.0.1，并复用现有 Conda 环境的 PyTorch：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m embodied_policy.train --config configs/smoke.yaml
.\.venv\Scripts\python.exe -m embodied_policy.evaluate --config configs/smoke.yaml --checkpoint outputs/smoke/best.pt
```

训练产物写入 `outputs/`，该目录不提交 Git。服务器端可用 `environment.yml` 创建独立环境。

当前 PickCube 状态策略：

```powershell
.\.venv\Scripts\python.exe -m embodied_policy.train --config configs/pickcube_state_delta.yaml
.\.venv\Scripts\python.exe -m embodied_policy.evaluate --config configs/pickcube_state_delta.yaml --checkpoint outputs/pickcube_state_delta/best.pt
```

100 次闭环评测成功率为 90%，失败 seed 和汇总指标保存在 `outputs/pickcube_state_delta/eval_metrics.json`。这是单 seed 基线，不是最终主结果。

在相同数据划分、归一化和评测 seeds 下，Single-step MLP 为 75%，Action-Chunking Transformer 为 90%。详细实验条件与限制见 `docs/experiment_log.md`。

数据来源和预处理过程见 `docs/data_provenance.md`。官方数据及训练输出不会提交到 GitHub。

在支持 Vulkan 渲染的 Linux 机器上录制指定 seed：

```bash
python -m embodied_policy.record_rollouts \
  --config configs/pickcube_state_delta.yaml \
  --checkpoint outputs/pickcube_state_delta/best.pt \
  --seeds 30014 30020 30022 \
  --output-dir outputs/videos/transformer
```

## 项目原则

1. 每个模型必须有闭环任务指标，不能只报告训练 loss。
2. 每项结论至少运行三个随机种子。
3. 报告失败案例、推理延迟与显存，而不只保存成功视频。
4. 仿真器、数据集与策略模块解耦，确保基线之间公平比较。
