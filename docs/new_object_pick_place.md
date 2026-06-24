# 用 GraspGenX 给新物体生成抓取，并在 openarm_mp_labs 里跑 pick-and-place

本文说明完整流程：**为一个新物体用 GraspGenX 生成抓取候选 YAML → 在
`openarm_mp_labs` 里引用该文件，规划轨迹、仿真并录像**。

这是一个**两阶段离线管线**，两端各自待在自己的环境里，只通过一个 YAML 文件交接：

```
[GraspGenX 环境: torch+ROCm+GPU+checkpoint]
    物体 mesh ──推理──> <name>_grasps.yml   (isaac_grasp 格式，物体网格帧下的 6-DOF 抓取候选)
                            │   ← 唯一交接物：一个文件
[openarm_mp_labs 环境: jax+mujoco]
    <name>_grasps.yml + 物体 MJCF ──> 选抓取 → 规划轨迹 → 仿真/录像
```

> 为什么是文档而非代码集成：两端框架（torch vs jax，都基于 ROCm）放进同一环境易冲突；
> 抓取模型又必须有 torch+GPU+权重。用文件契约解耦最灵活：换抓取生成器、换机器、批处理都不受影响。
> 详见 ADR `openarm_labs_hub/docs/decisions/0005-graspgenx-file-contract.md`。

---

## 前置条件

- **GraspGenX 端**：已安装 GraspGenX + 下载 checkpoint（推荐用已固化容器 `openarm-rocm:graspgen`）。
  首次安装见 hub 实验记录 `2026-06-17-graspgenx-openarm-inference.md`。
  > ⚠️ **必须用 fork [`alexhegit/GraspGenX`](https://github.com/alexhegit/GraspGenX)，不是上游。**
  > 上游 `NVlabs/GraspGenX` **尚未集成 OpenArm 夹爪，也没有 ROCm 支持**，这两项都在该 fork 中实现并已向上游提 PR（截至本文未合并）：
  > - OpenArm pinch gripper：[NVlabs/GraspGenX#3](https://github.com/NVlabs/GraspGenX/pull/3)（分支 `openarm`，新增 `assets/proc_grippers/openarm/`）
  > - AMD ROCm 支持：[NVlabs/GraspGenX#1](https://github.com/NVlabs/GraspGenX/pull/1)（`pyproject.toml` 加 `rocm` extra，W7900 + ROCm 7.2 验证）
  >
  > 安装（AMD GPU）：`uv sync --extra rocm`（如需 ZMQ 服务再加 `--extra serve`）。注意 `rocm` 与 `end2end` 互斥（end2end 的 cuRobo/Newton 是 CUDA-only，见 ADR 0002）。
- **openarm_mp_labs 端**：planner venv（mujoco + jax + openarm_control），见本仓库 README。
- 跑 demo 阶段**不需要** GraspGenX；只有生成新抓取时才用它。

---

## 步骤 1 — 准备物体 mesh（米制、居中）

GraspGenX 和 MuJoCo 都按**米**工作，且本仓库约定 **mesh 原点 = 物体几何中心**
（这样运行时物体的世界位姿就能直接换算抓取）。

- **扫描资产**：用 [Scan2Sim](https://github.com/alexhegit/Scan2Sim) 把原始扫描 obj 转成
  居中、mm→m 的 MuJoCo 资产（同时得到 `visual.obj` / `collision.stl` / `<obj>.xml`）：
  ```bash
  # 在 Scan2Sim 里（容器内需 imagemagick 做贴图转换）
  python -c "from scan2sim.core.batch_converter import convert_object; \
  from pathlib import Path; convert_object(Path('assets/<obj>/<obj>.obj'), unit_scale=0.001)"
  # 产物：mjcf/<obj>/{<obj>.xml, meshes/visual.obj, collision.stl, ...}
  ```
- **自有 mesh**：`.obj/.stl/.ply` 均可，确保单位是米、且已居中到几何中心。

---

## 步骤 2 — 用 GraspGenX 生成抓取候选 YAML

对**视觉/几何 mesh**（步骤 1 的 `visual.obj` 或你的 mesh）跑推理：

```bash
docker exec graspgen-dev bash -lc '
  cd /workspace/GraspGenX
  python scripts/demo_object_mesh.py \
    --mesh_file /workspace/Scan2Sim/mjcf/<obj>/meshes/visual.obj --mesh_scale 1.0 \
    --gripper_name openarm \
    --grasp_threshold -1.0 --return_topk --topk_num_grasps 50 \
    --no-visualization --output_file /workspace/output/<obj>_grasps.yml'
```

- `--gripper_name openarm`：务必用 openarm 夹爪，抓取帧才和本仓库约定一致。该夹爪由 fork
  `alexhegit/GraspGenX` 提供（上游暂无，见 [PR #3](https://github.com/NVlabs/GraspGenX/pull/3)），
  对应 `assets/proc_grippers/openarm/`；上游仓库跑这条命令会因找不到该夹爪而失败。
- `--mesh_scale 1.0`：mesh 已是米制就用 1.0。
- 检查日志 `Inferred N grasps, scores: a — b`，确认非空、置信度合理。
- 产物：`output/<obj>_grasps.yml`（`isaac_grasp` 格式：每个 grasp 含 `confidence` +
  物体帧下的 `position` + `orientation`(四元数 w/xyz)；夹爪基座 `+z` 为接近方向）。

---

## 步骤 3 — 在 openarm_mp_labs 里引用该文件跑 demo

`--object` 指物体 MJCF（决定仿真里加载什么物体），`--grasp-file` 指步骤 2 的 YAML
（决定从哪、以什么姿态抓）。两者配合：

```bash
cd openarm_mp_labs

# 只看轨迹是否可解（IK 误差）
uv run openarm-mp-demo \
  --object /path/to/Scan2Sim/mjcf/<obj>/<obj>.xml \
  --grasp-file output/<obj>_grasps.yml --grasp-mode full --generate-only

# 物理回放（不录像），看抬升高度
uv run openarm-mp-demo \
  --object /path/to/<obj>.xml --grasp-file output/<obj>_grasps.yml \
  --grasp-mode full --simulate-only

# 录像（headless）
MUJOCO_GL=egl uv run openarm-mp-demo \
  --object /path/to/<obj>.xml --grasp-file output/<obj>_grasps.yml \
  --grasp-mode full --record output/<obj>_pick_place.mp4
```

参数说明：
- `--grasp-mode`
  - `topdown`：强制竖直接近（最稳，复用已校准的俯抓朝向）；
  - `best` / `full`：用 GraspGenX 选中的真实 6-DOF 朝向（体现任意角度抓取）。
- `--object` 还接受**内置短名**（如 `ginger`），会自动带上其默认 `--grasp-file`。
- 输出默认落在 `openarm_mp_labs/output/`（已 gitignore）。

### 可选：把小体积资产 vendored 进仓库做成内置示例

若希望像 `ginger` 那样 `--object <name>` 一键运行（自包含、无需外部路径）：
1. 把（建议抽取后的轻量）`visual.obj` + `collision.stl` 放到 `assets/<obj>/meshes/`，写一个
   `assets/<obj>/<obj>.xml`（参考 `assets/ginger/ginger.xml`，注明来源）；
2. 把 YAML 放到 `assets/grasps/<obj>_grasps.yml`；
3. 在 `src/openarm_mp_labs/demo_pick_place.py` 的 `_BUNDLED_OBJECTS` 里登记
   `"<obj>": (assets/<obj>/<obj>.xml, assets/grasps/<obj>_grasps.yml)`。
> 高模视觉网格（>~5 万面）在 osmesa 软渲染下录像很慢，建议
> `trimesh.simplify_quadric_decimation` 抽取到 ~1.5 万面再入库（几何/抓取结果不变）。

---

## 关键约定与常见坑

- **米制 + 居中**：mesh 必须米制、原点在几何中心，否则世界位姿换算错位。
- **帧契约**：YAML 是**物体网格帧**下的抓取；夹爪基座 `+z` = 接近方向。本仓库
  `grasp_io.py` 按此契约换算到 MuJoCo 控制点帧（用 `GRASP_DEPTH_M`、`TCP_OFFSET_LOCAL`）。
  换了别的抓取生成器，只要产出同格式 YAML 即可接入；约定变了要同步 `grasp_io.py`。
- **接触点 ≠ 质心**：非对称物体的抓取点取 `base + GRASP_DEPTH_M·approach`，已在 `grasp_io` 处理。
- **换位置/朝向不必重跑 GraspGenX**：抓取是物体帧下的，运行时乘物体实时世界位姿即可。
  只有**换成不同形状的物体**才需要对新 mesh 重新推理。
  （出生位置目前在 `scene_builder.py` 的 `_DEFAULT_POS`，改那里即可。）
- **可达性**：新位置/姿态要在右臂可达范围内，IK 解不出会在 `*_refine` 阶段报较大误差。

---

## 已验证示例：ginger（仓库内置）

```bash
uv run openarm-mp-demo --object ginger --grasp-mode full --record
```
- 来源：Scan2Sim 的 3D 扫描 ginger（资产 vendored 在 `assets/ginger/`）。
- 结果：topdown 抬升 ~120 mm；full（对角抓，conf 0.97）抬升 ~112 mm。
- 完整记录见 hub `docs/experiments/2026-06-24-ginger-scanned-object-pick-place.md`。
