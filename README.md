# OpenArm MP Labs

Manipulation labs built on [openarm_mujoco](https://github.com/enactic/openarm_mujoco) and [openarm_control](https://github.com/enactic/openarm_control).

This repository contains:

- **Pick-and-place plan** — design notes and architecture (`docs/`)
- **Trajectory generation** — Cartesian waypoints + mink IK (`src/openarm_mp_labs/trajectory.py`)
- **Simulation replay & recording** — MuJoCo physics, offscreen MP4 (`src/openarm_mp_labs/simulation.py`)
- **Demo output** — recorded runs (`output/`)

## Prerequisites

Sibling clones in the same workspace (or edit `pyproject.toml` paths):

```
OpenArm_Labs/
├── openarm_mujoco/
├── openarm_control/
└── openarm_mp_labs/    ← this repo
```

## Install

```bash
cd openarm_mp_labs
uv sync
```

## Run pick-and-place demo

```bash
# Generate trajectory only
uv run openarm-mp-demo --generate-only

# Physics check (exit 1 if cube not lifted > 30 mm)
uv run openarm-mp-demo --simulate-only

# Record MP4 (headless, uses MUJOCO_GL=egl on Linux servers)
MUJOCO_GL=egl uv run openarm-mp-demo --record

# Interactive viewer (requires DISPLAY)
uv run openarm-mp-demo --hold-steps 80
```

Default recording path: `output/pick_place_demo.mp4`

## GraspGenX-driven grasps (beyond the cube)


flowchart LR
  A[Object mesh] --> B["GraspGenX inference<br/>ROCm + checkpoint"]
  B --> C["*_grasps.yml<br/>(isaac_grasp)"]
  C --> D[grasp_io: select grasp + transform to world frame]
  D --> E[trajectory: fingertip waypoints + site IK]
  E --> F[simulation: settle + re-IK + attach + recording]


The grasp pose can come from a GraspGenX `isaac_grasp` YAML instead of the
hardcoded top-down cube pose, and the manipuland can be a scanned mesh object
instead of the cube. A scanned **ginger** is bundled so this runs standalone:

```bash
# Self-contained: bundled ginger asset + its GraspGenX grasps
uv run openarm-mp-demo --object ginger --grasp-mode full --simulate-only
MUJOCO_GL=egl uv run openarm-mp-demo --object ginger --grasp-mode full --record

# Cube driven by a GraspGenX YAML you generated yourself
uv run openarm-mp-demo --grasp-file path/to/grasps.yml --grasp-mode topdown --simulate-only
```

- `--object` accepts a bundled name (`ginger`) or a path to any Scan2Sim object
  MJCF. A bundled name also supplies its default `--grasp-file`.
- `--grasp-mode`: `topdown` (force the validated vertical approach), `best`/`full`
  (use GraspGenX's selected 6-DOF orientation).

Verified lifts: ginger topdown ~120 mm, ginger full (diagonal grasp, conf 0.97)
~112 mm.

### Bundled assets & attribution

The ginger asset under `assets/ginger/` is a 3D scan from
[Scan2Sim](https://github.com/alexhegit/Scan2Sim), converted to a MuJoCo asset by
Scan2Sim (centered, mm→m). The vendored visual mesh is decimated to ~14k faces
(from ~148k) to keep the repo small and osmesa rendering fast; collision is the
convex hull. To use other Scan2Sim objects, convert them with Scan2Sim and point
`--object` at the generated MJCF. Grasp YAMLs in `assets/grasps/` are GraspGenX
outputs for those meshes.

## Grasp improvements (v0.1)

Compared to the initial prototype:

1. **Settle physics** after reset — cube resting height (~1.025 m) is used for waypoints, not XML spawn height.
2. **Correct EE/site height** — grasp targets account for fingertip offset below `right_ee_control_point`.
3. **Gripper ramp** — gradual close/open over many frames instead of a step command.
4. **Actuator tuning** — higher arm/finger `kp` and cube friction for contact.
5. **Kinematic attach** — when the gripper closes near the cube, the cube follows the EE through lift/transport (released on `open_gripper`). Pure contact-only grasp remains unreliable under position control; attach enables a stable demo while physics contact still drives the close phase.

## Documentation

- [docs/new_object_pick_place.md](docs/new_object_pick_place.md) — end-to-end
  workflow: generate a GraspGenX grasp YAML for a **new object** and run a
  pick-and-place demo from it.
- [docs/pick_and_place_plan.md](docs/pick_and_place_plan.md) — design notes.

## License

Apache License 2.0 — Copyright 2026 Enactic, Inc.
