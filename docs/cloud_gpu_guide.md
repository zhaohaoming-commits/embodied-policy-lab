# 云 GPU 租用与 ManiSkill 验收指南

更新时间：2026-08-09。价格和库存会变化，租用前以平台控制台为准。

## 1. 当前项目到底需要什么

当前阶段不是训练 7B 具身大模型，而是：

1. 在 Linux 上复现 PickCube 状态策略的闭环评测；
2. 使用 ManiSkill/SAPIEN 的 `rgb_array` 渲染录制成功与失败 seed；
3. 下一阶段训练单目或多视角视觉策略。

因此目前只需要 **1 张 24 GB 显存 GPU**，不需要 4 卡或 8 卡。建议规格：

| 项目 | 最低配置 | 推荐配置 |
|---|---:|---:|
| GPU | RTX 3090 24 GB | RTX 4090 24 GB |
| CPU | 8 vCPU | 10–16 vCPU |
| 内存 | 32 GB | 64 GB |
| 磁盘 | 50 GB | 100–200 GB |
| 系统 | Ubuntu 22.04 | Ubuntu 22.04 |
| 必要能力 | NVIDIA 驱动、Vulkan、SSH | 同左 |

RTX 3090 足以完成当前状态策略评测和视频录制；4090 的优势主要体现在后续视觉编码器训练更快。A100/H100 现阶段浪费预算。若以后要对 7B VLA 做全参数微调，24 GB 可能不够，但那是另一次独立选型；LoRA/量化实验仍可能用单张 4090 完成。

## 2. 平台选择

### 当前建议

1. **零成本试运行：OpenBayes 免费 3 小时 RTX 4090。** 用它验证 Linux、Vulkan、项目安装和录像链路。免费额度和库存以注册后的控制台为准。
2. **后续按量首选：矩池云 RTX 4090。** 官网当前列出的最低价为 1.54 元/GPU·小时，页面所列机器为 24 GB 显存、10 核 CPU、64 GB 内存和 200 GB 硬盘。若只做当前录像，RTX 3090 最低 1.29 元/小时也完全够用。
3. **缺货备选：AutoDL RTX 4090。** 官网当前公开价约 1.88 元/小时；按量实例开机计费、关机停止 GPU 计费，SSH/JupyterLab 文档完整。关机不代表扩容磁盘停止收费。
4. **海外备选：RunPod/Vast.ai。** 只有在国内平台缺货，且海外支付、代码与模型下载网络都稳定时再考虑。Vast.ai 是供需定价市场，机器数量多但主机质量和价格会波动；RunPod 官网当前 4090 标价为 0.69 美元/小时。

不能根据公开首页可靠判断“此刻谁的 4090 余量最多”。正确做法是在实际租用时同时打开矩池云和 AutoDL 的主机市场，按 **4090、单卡、按量、可用** 筛选；无货就切换平台，不为“可能有卡”购买包周。

### 价格直觉（仅计算费）

| 方案 | 公开价格 | 5 小时 | 20 小时 | 适合用途 |
|---|---:|---:|---:|---|
| 矩池云 3090 24 GB | 1.29 元/时起 | 6.45 元 | 25.80 元 | 当前评测、录像 |
| 矩池云 4090 24 GB | 1.54 元/时起 | 7.70 元 | 30.80 元 | 当前任务 + 视觉策略 |
| AutoDL 4090 24 GB | 约 1.88 元/时 | 9.40 元 | 37.60 元 | 国内备选 |
| OpenBayes 4090 24 GB | 2.70 元/时 | 13.50 元 | 54.00 元 | 免费额度验收、临时任务 |

以上不含扩容磁盘、持久化存储和流量等费用。首次不要包日、包周，也不要选择竞价/可中断实例；等命令稳定、checkpoint 能续训后再用低价可中断资源。

## 3. 租用页面怎么选

1. 注册并完成平台需要的认证，只充值 10–20 元或先用免费额度。
2. 计费选择“按量/按时”，GPU 数量选择 1。
3. 优先选 RTX 4090 24 GB；如果只是今天录像且 4090 缺货，选 RTX 3090 24 GB。
4. 镜像选择 Ubuntu 22.04、Python 3.11、PyTorch 2.x、CUDA 12.x。这里的 CUDA 小版本不必与本地完全相同，重点是云主机驱动能支持镜像里的 CUDA。
5. 磁盘先用免费容量；若小于 50 GB，再扩到 100 GB。视觉数据集开始生成后才考虑 200 GB。
6. 开机后取得平台给出的 SSH 主机、端口和用户名。配置 SSH 公钥，不要把密码或私钥发到聊天、GitHub 或项目文件。

不要选 Serverless 推理产品、Windows 实例、低于 16 GB 显存的卡、没有 SSH 的“应用空间”，或不能确认 Vulkan 是否可用的实例。

## 4. 开机后先验收，不要立刻装一堆包

在云端终端运行：

```bash
nvidia-smi
python --version
df -h
```

然后检查 Vulkan：

```bash
sudo apt-get update
sudo apt-get install -y libvulkan1 vulkan-tools
vulkaninfo --summary
ls /usr/share/vulkan/icd.d/nvidia_icd.json
```

必须在 `vulkaninfo --summary` 中看到 NVIDIA GPU。ManiSkill 官方将 Linux/NVIDIA 的 CPU 仿真、GPU 仿真和渲染都列为支持；其排错文档也明确要求检查 NVIDIA Vulkan ICD。若 Vulkan 看不到显卡，先不要充值更多钱或上传大数据：换镜像、联系客服，仍失败就换主机。

## 5. 上传项目并创建环境

代码最终应从自己的 GitHub 仓库克隆。仓库尚未推送前，可以在本地 PowerShell 执行（把 SSH 参数替换成平台提供的值）：

```powershell
scp -P <端口> -r .\configs .\docs .\embodied_policy .\scripts .\tests .\pyproject.toml .\environment.yml <用户>@<主机>:/root/embodied-policy-lab/
scp -P <端口> .\outputs\pickcube_state_delta\best.pt <用户>@<主机>:/root/embodied-policy-lab/outputs/pickcube_state_delta/
```

不要上传 Windows 的 `.venv`；虚拟环境不可跨系统复用。云端安装：

```bash
cd /root/embodied-policy-lab
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[sim]'
```

若平台镜像没有可用的系统 Python 3.10/3.11，则使用其内置 Conda：

```bash
conda env create -f environment.yml
conda activate embodied-policy
python -m pip install -e . --no-deps
```

## 6. 先跑最小渲染测试，再跑项目

最小测试只创建一个环境、渲染一帧并关闭：

```bash
python - <<'PY'
import gymnasium as gym
import mani_skill.envs

env = gym.make(
    "PickCube-v1",
    num_envs=1,
    obs_mode="state",
    control_mode="pd_joint_delta_pos",
    render_mode="rgb_array",
)
env.reset(seed=0)
frame = env.render()
print("render ok:", tuple(frame.shape), frame.dtype)
env.close()
PY
```

成功后再运行测试与指定失败 seed 的录像：

```bash
python -m unittest discover -s tests -v
python -m embodied_policy.record_rollouts \
  --config configs/pickcube_state_delta.yaml \
  --checkpoint outputs/pickcube_state_delta/best.pt \
  --seeds 30014 30020 30022 \
  --output-dir outputs/videos/transformer
```

这里的三个 seed 是已知失败样例。录像脚本仍会重新在仿真器里执行完整 episode，并根据环境返回的 `info["success"]` 命名 success/failure，而不是凭肉眼或预先标签判断。

## 7. 下载、备份、关机

在本地 PowerShell 下载结果：

```powershell
scp -P <端口> -r <用户>@<主机>:/root/embodied-policy-lab/outputs/videos .\outputs\
```

确认视频能播放、日志已保存后：

1. 将代码推送 GitHub；checkpoint、原始数据和大视频不直接提交 Git。
2. 将关键指标、配置和少量展示视频另行备份。
3. 在平台控制台关机，确认状态不是“运行中”。
4. 检查扩容磁盘或持久化存储是否仍在计费；不用就释放或缩容。

AutoDL 的按量 GPU 是关机停止计算费，但扩容数据盘即使关机仍会按规则计费；其本地数据盘无冗余，重要结果必须另存。

## 8. 本阶段验收标准

租服务器的目标不是“成功登录”，而是完成以下证据链：

- `nvidia-smi` 正常；
- `vulkaninfo --summary` 识别 NVIDIA GPU；
- ManiSkill 最小脚本得到 RGB 帧；
- 项目单元测试通过；
- 至少录下 1 个成功和 1 个失败 episode；
- 视频、运行命令、软件版本和费用被记录；
- 结果下载后实例已停止计费。

## 参考链接

- [矩池云官网与公开 GPU 价格](https://matpool.com/)
- [OpenBayes 价格与免费计算时](https://app.hyper.ai/pricing/)
- [AutoDL 快速开始](https://www.autodl.com/docs/quick_start/)
- [AutoDL 计费说明](https://api.autodl.com/docs/price/)
- [AutoDL SSH 文档](https://www.autodl.com/docs/ssh/)
- [AutoDL 本地数据盘说明](https://www.autodl.com/docs/local_disk/)
- [RunPod GPU 价格](https://www.runpod.io/pricing)
- [Vast.ai 实时算力市场](https://vast.ai/)
- [ManiSkill 安装、系统支持与 Vulkan 排错](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html)
