# OpenArm 通用抓取 · ROCm 环境验证 To-Do

目标：在 AMD ROCm 上搭一套通用抓取流程——**GraspGenX（抓取生成，PyTorch/ROCm）→ PyRoki/mink（IK）→ MuJoCo 回放+双机位录制（现有 CPU 管线）**，绕开 CUDA-only 的 GraspGenX end2end（cuRobo/Newton/warp）。

基础镜像候选：`rocm/jax:rocm7.2.4-jax0.8.2-py3.12`

> 用法：每一项都是一个 gate，过了再做下一项。最关键的三个 gate：**1.2 跨版本内核兼容**、**3.3 torch+jax 共存**、**5.2 py3.12 依赖**——它们决定「单容器 vs 双容器」「是否升级宿主 ROCm / 改用 py3.11 镜像」。

## 宿主机现状（已确认）

- Docker 29.3.0
- amdgpu-dkms **6.10.5** / ROCm 用户态 **6.3.2**
- GPU：AMD Radeon PRO W7900 (48GB, **gfx1100**)
- `/dev/kfd`、`/dev/dri` 存在；当前用户在 `video` / `render` 组
- 当前 `openarm_mp_labs` 管线：纯 CPU，依赖 `mujoco / mink / daqp / numpy`，**不依赖 torch / jax / ROCm**

---

## 阶段 0｜宿主机前置

- [x] 0.1 Docker 可用（29.3.0）
- [x] 0.2 GPU 设备与权限（/dev/kfd、/dev/dri、video/render 组）
- [x] 0.3 内核驱动 amdgpu-dkms 6.10.5 / ROCm 6.3.2，卡 W7900 gfx1100
- [ ] 0.4 磁盘空间充足（镜像+checkpoint+wheel 约 30–50GB）：`df -h /DATA`

## 阶段 1｜镜像与 GPU 可见性（关键兼容性 gate）

- [ ] 1.1 拉取镜像：`docker pull rocm/jax:rocm7.2.4-jax0.8.2-py3.12`
- [ ] 1.2 **跨版本兼容冒烟测试**（ROCm 7.2 容器 on ROCm 6.3 宿主）：起容器带
      `--device=/dev/kfd --device=/dev/dri --group-add video --group-add render --security-opt seccomp=unconfined`，
      容器内跑 `rocminfo` → **通过：列出 gfx1100**。失败则升级宿主 amdgpu-dkms 到 ROCm 7.x 配套版。
- [ ] 1.3 JAX 见到 GPU：`python -c "import jax; print(jax.devices())"` → **通过：含 ROCm GPU**
- [ ] 1.4 JAX 实算：小矩阵 `jnp.dot` + `.block_until_ready()` → **通过：无 HIP 错误、结果正确**

## 阶段 2｜盘点镜像已有内容

- [ ] 2.1 版本：`python --version`（预期 3.12）、`jax.__version__`
- [ ] 2.2 **是否自带 PyTorch**：`python -c "import torch;print(torch.__version__, torch.version.hip, torch.cuda.is_available())"`
      → 自带且可用则省去阶段 3；缺失则进阶段 3
- [ ] 2.3 ROCm 用户态版本：`cat /opt/rocm/.info/version`（确认 7.2.4）、`pip list | grep -iE "torch|jax|rocm"`

## 阶段 3｜PyTorch-ROCm（GraspGenX 硬依赖）

- [ ] 3.1（仅当 2.2 缺失）安装 `torch==2.12.0+rocm7.2`（PR#1 验证、匹配镜像 ROCm 7.2.4）
- [ ] 3.2 torch GPU 验证：`torch.cuda.is_available()` / `get_device_name()` / `.to('cuda')` 运算 → **通过：识别 W7900、运算无误**
- [ ] 3.3 **torch + jax 共存**：同进程各跑一次 GPU 运算 → **通过：均可用、无 ROCm 运行时冲突**（决定单容器/双容器）

## 阶段 4｜现有 CPU 管线进容器（低风险）

- [ ] 4.1 装栈：`mujoco>=3.6 / mink / daqp / numpy` + 本地 `openarm_control / openarm_mujoco / openarm_mp_labs`（py3.12 兼容）
- [ ] 4.2 跑通现有 demo：`MUJOCO_GL=egl ... demo_pick_place --record` → **通过：生成 mp4、抓取正常**

## 阶段 5｜GraspGenX 可行性（py3.12 主要风险）

- [ ] 5.1 clone GraspGenX，合入 `alexhegit` 的 `main`(rocm) + `openarm` 分支（均改 pyproject/AGENTS，需手动合）
- [ ] 5.2 **依赖安装（py3.12 关口）**：`uv sync --extra rocm`，重点看 `scene-synthesizer / spconv / pointnet2_ops` 是否有 py3.12 wheel
      → **通过：核心推理依赖装上**（`pointnet2_ops` 缺失可接受，降级 `ptv3_vanilla`）。大面积失败则改用 py3.11 的 rocm/pytorch 镜像。
- [ ] 5.3 checkpoint：首次 import 自动从 HuggingFace 拉取，或设 `GRASPGENX_CHECKPOINT_DIR` → **通过：可加载**

## 阶段 6｜GraspGenX + OpenArm 抓取生成（核心目标）

- [ ] 6.1 夹爪发现：`python scripts/list_grippers.py` 含 `openarm`（PR#3 资产）
- [ ] 6.2 mesh 推理：`demo_object_mesh.py --gripper_name openarm --mesh_file <物体>` → **通过：有效 6-DOF 抓取、打分合理（PR#3：0.70–0.99）**
- [ ] 6.3（可选）可视化：PR#4 `demo_object_mesh_vis.py` 渲染夹爪精细网格确认姿态

## 阶段 7｜PyRoki（可选 IK 升级）

- [ ] 7.1 安装：`pip install -e pyroki`（JAX 已在镜像）
- [ ] 7.2 用 `openarm_description/output.urdf` 求 IK → **通过：收敛、与 mink 结果对比合理**

## 阶段 8｜集成闭环

- [ ] 8.1 对接：GraspGenX 输出位姿 → `openarm_mp_labs` 轨迹层（替换写死的 `prepare_targets` / `TCP_OFFSET_LOCAL` / `GRASP_GRIP`）
- [ ] 8.2 端到端：非立方体物体 → 抓取生成 → IK/轨迹 → MuJoCo 回放+双机位录制，全程 ROCm/CPU、不碰 CUDA

---

## 架构决策（待阶段 1–3 验证后定）

- **方案 A 单容器**：本 JAX 镜像 + 叠 torch-rocm，全部组件同容器。简单，但依赖 3.3 共存通过。
- **方案 B 双容器（更干净）**：容器1 = GraspGenX 抓取服务（torch-rocm，走自带 ZMQ server/client）；容器2 = 本 JAX 镜像 + PyRoki + MuJoCo + 现有管线。GPU 依赖隔离。

## 参考 PR（NVlabs/GraspGenX，作者 alexhegit）

- #1 Add AMD ROCm GPU support（pyproject `rocm` extra，torch 2.12+rocm7.2，W7900 验证）
- #3 add OpenArm pinch gripper（revolute_2f，已 onboard，省 wizard）
- #4 demo_object_mesh_vis.py（夹爪视觉网格渲染）
- #2 DGX Spark aarch64+CUDA13（与 AMD 无关，忽略）
