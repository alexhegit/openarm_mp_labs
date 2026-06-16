# OpenArm 通用抓取 · ROCm 环境验证 To-Do

目标：在 AMD ROCm 上搭一套通用抓取流程——**GraspGenX（抓取生成，PyTorch/ROCm）→ PyRoki/mink（IK）→ MuJoCo 回放+双机位录制（现有 CPU 管线）**，绕开 CUDA-only 的 GraspGenX end2end（cuRobo/Newton/warp）。

基础镜像候选：`rocm/jax:rocm7.2.4-jax0.8.2-py3.12`

> 用法：每一项都是一个 gate，过了再做下一项。最关键的三个 gate：**1.2 跨版本内核兼容**、**3.3 torch+jax 共存**、**5.2 py3.12 依赖**——它们决定「单容器 vs 双容器」「是否升级宿主 ROCm / 改用 py3.11 镜像」。

---

## 进展与关键发现（2026-06-16，venv 分支）

实测结论，已固化为 `docker/Dockerfile`（镜像 tag `openarm-rocm:unified`）：

1. **ROCm 7.2 容器可在本机 ROCm 6.3.2 宿主上跑**（gate 1.2 通过）：容器内 `rocminfo` 看到 W7900 gfx1100，无需升级宿主驱动。
2. **JAX-ROCm 在 gfx1100 上正常**：`rocm/jax`（jax 0.8.2/rocm7.2.1）与从 AMD repo 安装的 jax 0.9.2 均能 GPU 计算。jax-rocm wheel 来源：`https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/`（非公共 PyPI）。
3. **torch-rocm 必须选含 gfx1100 的构建**：`rocm/pytorch:...release_2.7.1` 的 arch list 只有 `gfx942/gfx1201/gfx950`，**在 W7900 上 GPU 计算段错误**（`is_available()` 仍返回 True，极具迷惑性）。**`rocm/pytorch:latest`（torch 2.10.0+rocm7.2.3）arch list 含 gfx1100，GPU 计算正常** → 选它做底座。
4. **torch-rocm 与 jax-rocm 不能装进同一 site-packages**：会因 ROCm 库/numpy ABI 冲突导致连 torch 单独也崩。**解法：一个镜像、两个隔离 venv** —— 系统 python 放 torch（GraspGenX），`/opt/venv-planner` 放 jax+mujoco+mink（PyRoki/规划侧）。集成时进程分离（GraspGenX 走自带 ZMQ server）。
5. **统一镜像三栈实测**（真实退出码 rc=0）：torch 2.10 GPU ✓ / jax 0.9.2 GPU ✓ / mujoco 3.9 + mink ✓。

**架构定论：单镜像 + 双 venv + 进程分离**（不是单进程，也不必双容器）。

构建与验证命令：
```bash
docker build -t openarm-rocm:unified -f docker/Dockerfile docker
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video --group-add 110 \
  --security-opt seccomp=unconfined openarm-rocm:unified bash -lc '
  python -c "import torch;print(torch.cuda.get_device_name(0), float(torch.ones(8,device=\"cuda\").sum()))"
  /opt/venv-planner/bin/python -c "import jax;print(jax.devices())"'
```

剩余待办：阶段 5（GraspGenX 装进系统 python 并跑 openarm 夹爪推理）、阶段 7（PyRoki 装进 venv-planner）、阶段 8（集成闭环）。

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
- [x] 0.4 磁盘空间充足：`/DATA` 1.3T 可用

## 阶段 1｜镜像与 GPU 可见性（关键兼容性 gate）

- [x] 1.1 镜像：本机已缓存 `rocm/jax:latest`（ROCm7.2.1/jax0.8.2）与 `rocm/pytorch:latest`（torch2.10），等价于指定 tag，直接复用
- [x] 1.2 **跨版本兼容**：ROCm 7.2 容器在 ROCm 6.3.2 宿主上 `rocminfo` 看到 W7900 gfx1100 ✓（用 `--group-add 110` 数字 GID）
- [x] 1.3 JAX 见到 GPU：`jax.devices()` → `[RocmDevice(id=0)]` ✓
- [x] 1.4 JAX 实算：`jnp.ones@jnp.ones` 结果正确 ✓

## 阶段 2｜盘点镜像已有内容

- [x] 2.1 版本：Python 3.12.3 / jax 0.8.2
- [x] 2.2 PyTorch：`rocm/jax` 不含 torch；改用 `rocm/pytorch:latest` 自带 torch 2.10.0+rocm7.2.3
- [x] 2.3 ROCm 用户态：jax 镜像 7.2.1；jax-rocm wheel 源 = `repo.radeon.com/rocm/manylinux/rocm-rel-7.2/`

## 阶段 3｜PyTorch-ROCm（GraspGenX 硬依赖）

- [x] 3.1 torch 来自 `rocm/pytorch:latest`（2.10.0+rocm7.2.3，arch list 含 gfx1100）— **不要用 release_2.7.1（无 gfx1100，会段错误）**
- [x] 3.2 torch GPU 验证：W7900 识别 + `torch.ones(...,device=cuda).sum()` 正确 ✓
- [x] 3.3 **torch + jax 共存**：单进程会段错误 → 解法 = 双 venv + 进程分离（见上「进展与关键发现」）✓

## 阶段 4｜现有 CPU 管线进容器（低风险）

- [x] 4.1 装栈：`mujoco 3.9 / mink 1.1 / daqp 0.7.2 / numpy` 已装入 venv-planner ✓
- [ ] 4.2 跑通现有 demo（需把本地 `openarm_control/openarm_mujoco/openarm_mp_labs` 挂载进容器）：`MUJOCO_GL=egl ... demo_pick_place --record` → **通过：生成 mp4、抓取正常**

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
