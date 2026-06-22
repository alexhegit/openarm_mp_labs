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

剩余待办：阶段 7（PyRoki 装进 venv-planner）、阶段 8（集成闭环）。

## 进展（2026-06-17，venv 分支）— 阶段 5/6 GraspGenX + OpenArm 推理跑通 ✅

在常驻容器 `graspgen-dev`（基于 `openarm-rocm:unified`）里完成 GraspGenX 安装与 openarm 夹爪 mesh 推理冒烟测试：

1. **clone 超集分支**：`alexhegit/GraspGenX` 的 `visual-mesh-demo` 分支（含 ROCm + openarm 夹爪资产 + 可视化 demo 三合一），夹爪资产在 `assets/proc_grippers/openarm/`（`config.json`/`gripper.urdf`/`coll_mesh.obj`/`vis_mesh.obj`）。
2. **依赖安装（py3.12 关口通过）**：用 **pip（非 uv）、不加 `--extra rocm`** → 已装的 torch 2.10/torchvision 0.25 满足 `>=2.1/>=0.16` 被保留、**未被改动**；`scene-synthesizer 1.15 / usd-core 26.5 / torch-geometric / diffusers 0.11.1` 全部 py3.12 装上。**关键坑**：不要用 constraints 钉本地版 torch（pip 会去远程找 `+rocm7.2.4` 本地版导致 ResolutionImpossible）；git 报 dubious ownership 需 `git config --global --add safe.directory`。numpy 被 GraspGenX 钉降到 1.26.4，**实测 torch 2.10 GPU 仍正常**（gpu_sum 4096，W7900）。
3. **checkpoint**：首次 `import graspgenx` 自动 clone `ext/graspgenx_checkpoints`（在挂载盘），但 `.pth` 是 **git-LFS 指针**，须 `apt install git-lfs && git lfs pull` 拉真权重 → gen `epoch_736.pth` 1.2GB / dis `epoch_1056.pth` 484MB。
4. **夹爪发现**：`python scripts/list_grippers.py` → `openarm` 在列（#1）✓。
5. **headless mesh 推理冒烟**（rc=0）：openarm 在 `box.obj` 上 GPU 推理 ~1.5–4.5s，**50 个 6-DOF 抓取，置信度 0.42–0.63**，以 `isaac_grasp` 格式（position+quaternion）存到 `output/openarm_box_grasps.yml`。
   - **后续项**：openarm 缺 `points.json`/`proc_gripper_only_pointnet_vae_repr.json`/`tsdf.npy` 缓存，当前用 dummy 值（有 WARNING），可能拉低分数（PR#3 标称 0.70–0.99）；后续按 README「Integrating a New Gripper」生成这些缓存以提升质量。

可复现命令（在常驻容器 `graspgen-dev` 内，源码/checkpoint 都在挂载盘 `/workspace/GraspGenX`）：
```bash
# 容器：openarm-rocm:unified，挂载工作区，常驻
docker run -d --name graspgen-dev --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add 110 --security-opt seccomp=unconfined \
  -v /DATA/AMD-Sim/OpenArm_Labs:/workspace -w /workspace/GraspGenX \
  openarm-rocm:unified sleep infinity

# 安装（保护 torch：不加 rocm extra、不用 constraints 钉 torch）
docker exec graspgen-dev bash -lc '
  git config --global --add safe.directory /workspace/GraspGenX
  pip install -e .'                                   # torch/torchvision 不动，numpy→1.26.4

# 拉 checkpoint 真权重（git-LFS）
docker exec graspgen-dev bash -lc '
  apt-get update -qq && apt-get install -y -qq git-lfs
  cd ext/graspgenx_checkpoints && git config --global --add safe.directory "$(pwd)"
  git lfs install && git lfs pull'

# 验证夹爪 + headless 推理
docker exec graspgen-dev bash -lc 'python scripts/list_grippers.py | grep -i openarm'
docker exec graspgen-dev bash -lc '
  python scripts/demo_object_mesh.py \
    --mesh_file assets/sample_data/object_mesh/box.obj --mesh_scale 1.0 \
    --gripper_name openarm --grasp_threshold -1.0 --return_topk --topk_num_grasps 50 \
    --no-visualization --output_file /workspace/output/openarm_box_grasps.yml'
```
> 注：pip 安装的依赖在容器可写层，**不在挂载盘**。如需跨容器复用，`docker commit graspgen-dev openarm-rocm:graspgen` 固化为镜像（或写进 Dockerfile）。

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
- [x] 4.2 **容器内跑通现有 demo** ✓：挂载工作区，`--no-deps -e` 装本地三包，`MUJOCO_GL=osmesa` 录像成功
      —— `Recorded 2150 frames (71.7s)`，`Peak cube lift 112.2 mm`，画面与宿主机一致。
      注意：mesa **EGL 在 W7900 容器内初始化失败**，改用 **osmesa 软件渲染**（已写入镜像）；容器需 `ffmpeg`+`libosmesa6`（已装）。
      运行：
      ```bash
      docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video --group-add 110 \
        --security-opt seccomp=unconfined -v /DATA/AMD-Sim/OpenArm_Labs:/workspace \
        -w /workspace/openarm_mp_labs openarm-rocm:unified bash -lc '
        /opt/venv-planner/bin/pip install -q --no-deps -e ../openarm_control -e ../openarm_mujoco -e .
        /opt/venv-planner/bin/python -m openarm_mp_labs.demo_pick_place --record output/pick_place_container.mp4'
      ```

## 阶段 5｜GraspGenX 可行性（py3.12 主要风险）

- [x] 5.1 clone `alexhegit/GraspGenX` 的 `visual-mesh-demo` 分支（rocm+openarm+vis 超集，免去手动合并）
- [x] 5.2 **依赖安装（py3.12 关口）通过**：用 **pip、不加 rocm extra** 保护 torch 2.10；`scene-synthesizer/usd-core/torch-geometric/diffusers` py3.12 全装上；spconv 已从 pyproject 移除、走 `ptv3vanilla`，无需 `pointnet2_ops`
- [x] 5.3 checkpoint：import 自动 clone，**git-LFS pull 拉真权重**（gen 1.2GB / dis 484MB）✓

## 阶段 6｜GraspGenX + OpenArm 抓取生成（核心目标）

- [x] 6.1 夹爪发现：`list_grippers.py` 含 `openarm`（#1）✓
- [x] 6.2 mesh 推理：openarm 在 `box.obj` → 50 个 6-DOF 抓取、置信度 0.42–0.63、isaac_grasp yml 输出 ✓
      （待优化：补 openarm 的 `points.json/tsdf.npy` 缓存以提升分数到 PR#3 标称 0.70–0.99）
- [ ] 6.3（可选）可视化：PR#4 `demo_object_mesh_vis.py` 渲染夹爪精细网格确认姿态

## 阶段 7｜PyRoki（可选 IK 升级）

- [ ] 7.1 安装：`pip install -e pyroki`（JAX 已在镜像）
- [ ] 7.2 用 `openarm_description/output.urdf` 求 IK → **通过：收敛、与 mink 结果对比合理**

## 阶段 8｜集成闭环 ✅（2026-06-22 跑通）

- [x] 8.1 对接：新增 `grasp_io.py` 适配层——读 GraspGenX `isaac_grasp` yml → 选 grasp → 物体帧转 MuJoCo 世界/EE 帧。`prepare_targets` 支持 `--grasp-file/--grasp-mode`，`PickPlaceTargets.from_grasp` + `build_poses` 泛化为「任意接近轴 + 抓取朝向」，无 grasp 时回退原俯抓。
- [x] 8.2 端到端：GraspGenX 位姿 → mink IK → 轨迹 → MuJoCo 双机位录像，全程 ROCm/CPU。

实测（40mm 方块 mesh 推理 → 闭环）：
- `--grasp-mode topdown`（复用已验证竖直姿态，GraspGenX 仅选 grasp）：**抬升 112.2mm**，与写死路径一致。
- `--grasp-mode full`（用 GraspGenX 真实 6-DOF 朝向）：选中 conf=0.817 的**水平侧抓**（approach=[+1,0,0]），IK 误差 0.6mm，**抬升 115.7mm**，录像 `output/pick_place_graspgenx_full.mp4`（2150 帧）。证明轨迹层已能消费 GraspGenX 任意朝向抓取。

**关键设计/坑**：grasp 的 `position`=夹爪基座，`+z`=接近方向（指尖在 +z 0.068）；与 MuJoCo「site −z 为指尖」相反。每个 grasp 的接近线都过物体质心，故指尖目标取 sim 实测 cube 中心、接近轴/朝向取自 GraspGenX。`topdown` 模式必须**直接用 home 朝向**（已校准），早期「强制 approach=[0,0,-1] 再重对齐」会引入倾斜导致抓空（lift=0）。

运行：
```bash
# 1) 生成 cube mesh 抓取（系统 python / torch）
docker exec graspgen-dev bash -lc 'cd /workspace/GraspGenX && python - <<PY
import trimesh; trimesh.creation.box(extents=[0.04]*3).export("/workspace/output/cube_40mm.obj")
PY
python scripts/demo_object_mesh.py --mesh_file /workspace/output/cube_40mm.obj \
  --gripper_name openarm --grasp_threshold -1.0 --return_topk --topk_num_grasps 50 \
  --no-visualization --output_file /workspace/output/openarm_cube40_grasps.yml'
# 2) 闭环回放（venv-planner / mujoco+mink）
docker exec graspgen-dev bash -lc 'cd /workspace/openarm_mp_labs && \
  MUJOCO_GL=osmesa /opt/venv-planner/bin/python -m openarm_mp_labs.demo_pick_place \
  --grasp-file /workspace/output/openarm_cube40_grasps.yml --grasp-mode full \
  --record /workspace/output/pick_place_graspgenx_full.mp4'
```

后续：非立方体物体（用其 mesh 推理 + 在 sim 里换成对应物体/位姿）；side-grasp 的抓取附着用接触力替代运动学 seat；阶段 6.2 夹爪缓存提分。

---

## 架构决策（待阶段 1–3 验证后定）

- **方案 A 单容器**：本 JAX 镜像 + 叠 torch-rocm，全部组件同容器。简单，但依赖 3.3 共存通过。
- **方案 B 双容器（更干净）**：容器1 = GraspGenX 抓取服务（torch-rocm，走自带 ZMQ server/client）；容器2 = 本 JAX 镜像 + PyRoki + MuJoCo + 现有管线。GPU 依赖隔离。

## 参考 PR（NVlabs/GraspGenX，作者 alexhegit）

- #1 Add AMD ROCm GPU support（pyproject `rocm` extra，torch 2.12+rocm7.2，W7900 验证）
- #3 add OpenArm pinch gripper（revolute_2f，已 onboard，省 wizard）
- #4 demo_object_mesh_vis.py（夹爪视觉网格渲染）
- #2 DGX Spark aarch64+CUDA13（与 AMD 无关，忽略）
