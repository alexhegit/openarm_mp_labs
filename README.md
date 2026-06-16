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

## Grasp improvements (v0.1)

Compared to the initial prototype:

1. **Settle physics** after reset — cube resting height (~1.025 m) is used for waypoints, not XML spawn height.
2. **Correct EE/site height** — grasp targets account for fingertip offset below `right_ee_control_point`.
3. **Gripper ramp** — gradual close/open over many frames instead of a step command.
4. **Actuator tuning** — higher arm/finger `kp` and cube friction for contact.
5. **Kinematic attach** — when the gripper closes near the cube, the cube follows the EE through lift/transport (released on `open_gripper`). Pure contact-only grasp remains unreliable under position control; attach enables a stable demo while physics contact still drives the close phase.

## Documentation

See [docs/pick_and_place_plan.md](docs/pick_and_place_plan.md).

## License

Apache License 2.0 — Copyright 2026 Enactic, Inc.
