# Embodied Policy Lab

从零实现、训练和评测具身模仿学习策略。第一个里程碑使用一个可控的合成到达任务，验证完整的数据、模型、训练、checkpoint 与闭环 rollout 链路；随后接入 ManiSkill 机器人轨迹。

## 当前里程碑

- [x] 合成三维到达任务及专家轨迹
- [x] 历史观测与 action chunk 数据窗口
- [x] 从零实现 Transformer action chunk policy
- [x] masked action loss、训练、验证及 checkpoint
- [x] 闭环 rollout 成功率评测
- [x] 接入 ManiSkill PickCube 官方演示数据
- [ ] 图像编码器与本体状态融合
- [x] Single-step BC 与 action chunking 状态基线对比
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

归一化后的 Action-Chunking Transformer 在 100 次闭环评测中成功率为 90%，Linux 4090 服务器复现为 90%。失败 seed 和汇总指标保存在 `outputs/pickcube_state_delta/eval_metrics.json`。这是单训练 seed 基线，不是最终主结果。

固定训练/验证划分和 100 个测试 episode 后的三训练 seed 结果：Action-Chunking Transformer 为 **92.7% ± 3.1%**，Single-step MLP 为 **77.3% ± 5.5%**。这是当前主基线；详细实验条件、逐 seed 数据与限制见 `docs/experiment_log.md`。

三训练 seed 的正式比较使用批量入口。`split_seed` 与 `eval.seed` 已固定为 7；因此改变 `--train-seeds` 只会改变模型初始化与 DataLoader 顺序，两个模型也会共享数据划分和 100 个测试 episode：

```bash
python -m embodied_policy.run_experiments \
  --configs configs/pickcube_state_delta.yaml configs/pickcube_state_mlp.yaml \
  --train-seeds 7 17 27 \
  --output-root outputs/pickcube_seed_sweep_20260824
```

运行按顺序执行，避免在共享服务器上抢占多张 GPU。输出根目录会保存每次运行的冻结 `config.yaml`、checkpoint、评测 metrics 和汇总的 `run_summary.{json,csv}`、`aggregate_summary.csv`。

受控消融包括：使用 `pickcube_state_delta_h1.yaml` 训练同一 Transformer 的单步预测版本；以及对已训练的 8-step checkpoint 仅改变部署时的 `replan_interval`，无需重训。后者使用 `embodied_policy.sweep_replan`，并输出每个 interval 的均值与样本标准差。

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
