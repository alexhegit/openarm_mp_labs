# OpenArm MuJoCo Pick-and-Place 轨迹生成方案

> 基于 `openarm_mujoco` + `openarm_control`，在 **`openarm_mp_labs`** 中实现 MuJoCo pick-and-place 轨迹生成、仿真与录制。
>
> 记录日期：2026-06-15

---

## 1. 结论

**可以完成关节空间轨迹生成与 MuJoCo 回放。** 上游仓库提供模型与 IK；本仓库 (`openarm_mp_labs`) 提供完整 pipeline 与 demo。

| 能力 | 上游 | openarm_mp_labs |
|------|------|-----------------|
| MuJoCo 模型 / demo.xml | openarm_mujoco | 使用 |
| FK / IK | openarm_control | 使用 |
| 路点 + 轨迹 + 回放 + 录制 | — | ✅ 已实现 |
| 稳定抓放 demo | — | ✅ v0.1（物理闭合 +  kinematic attach） |

---

## 2. 系统架构

```
openarm_mujoco (demo.xml)
        ↓
openarm_control (Kinematics / IK)
        ↓
openarm_mp_labs
  ├── trajectory.py    路点 + 插值 + IK
  ├── simulation.py    物理调参 + attach + 录制
  └── demo_pick_place  CLI
```

---

## 3. 运行

```bash
cd openarm_mp_labs
uv sync
MUJOCO_GL=egl uv run openarm-mp-demo --record
```

输出：`output/pick_place_demo.mp4`

---

## 4. 抓取改进（v0.1）

| 问题 | 原因 | 对策 |
|------|------|------|
| 夹爪够不到方块 | XML _spawn z=1.05，物理 settle 后 z≈1.025 | `settle_physics()` 后读取 cube 位姿 |
| 指尖与 IK 目标偏差 | `right_ee_control_point` 在腕部，指尖低 ~25 mm | `GRASP_DZ=0.025` 等偏移 |
| position 控制跟踪误差 | 默认 kp 偏低 | `tune_manipulation_physics()` |
| 纯接触抓取不稳定 | 闭合时推挤方块 | 闭合阶段物理 + **CubeAttachment** 随 EE 搬运 |

`CubeAttachment`：夹爪闭合且 cube 距 EE < 6 cm 时，lift/transport 阶段 cube 相对 EE 固定；`open_gripper` 时释放。

---

## 5. 模块说明

| 文件 | 职责 |
|------|------|
| `config.py` | 路点偏移、夹爪常量 |
| `kinematics_utils.py` | slerp、IK 迭代 |
| `trajectory.py` | 轨迹生成、夹爪 ramp |
| `simulation.py` | reset、物理、attach、record |
| `demo_pick_place.py` | CLI 入口 |

---

## 6. 验收

- [x] demo.xml 轨迹 IK 无失败
- [x] MP4 录制（离屏 EGL）
- [x] 方块随臂抬起 > 30 mm（attach 辅助）
- [ ] 纯摩擦抓取（无 attach）— 后续迭代

---

## 7. 相关仓库

```
OpenArm_Labs/
├── openarm_mujoco/
├── openarm_control/
└── openarm_mp_labs/          ← 本仓库
    ├── docs/                 ← 本文档
    ├── src/openarm_mp_labs/
    └── output/               ← demo 视频
```
