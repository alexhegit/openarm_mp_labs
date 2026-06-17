# Unified AMD ROCm Container — Usage

Reproducible container workflow for running the OpenArm pick-and-place pipeline
(and, later, GraspGenX + PyRoki) on a single AMD GPU host.

Verified host: **AMD Radeon PRO W7900 (gfx1100, 48 GB)**, host ROCm 6.3.2 /
amdgpu-dkms 6.10.5. A ROCm 7.2 container runs fine on this 6.3 host — no host
driver upgrade needed.

## 1. Image

Base: `rocm/pytorch:latest` (torch 2.10.0+rocm7.2.3, arch list includes
`gfx1100`). The pinned `..._pytorch_release_2.7.1` image is built only for
`gfx942/gfx1201/gfx950` and **segfaults on gfx1100 GPU compute** — do not use it.

The image carries three stacks with a deliberate split (see `Dockerfile`):

| Stack | Location | Purpose |
|-------|----------|---------|
| PyTorch-ROCm (2.10.0) | system python | GraspGenX (grasp generation) |
| JAX-ROCm (0.9.2) | `/opt/venv-planner` | PyRoki (kinematics) |
| MuJoCo 3.9 + mink + daqp | `/opt/venv-planner` | trajectory + replay |

> **Why two venvs:** torch-rocm and jax-rocm each bundle their own ROCm runtime
> and pin different numpy; in one site-packages even *torch alone* segfaults.
> They run in separate processes anyway (GraspGenX ships a ZMQ server).

System libs baked in: `ffmpeg`, `libosmesa6`, `libgl1` (headless software GL —
mesa **EGL fails to init** on the W7900 inside the container, so MuJoCo uses
`MUJOCO_GL=osmesa`).

### Build

```bash
cd openarm_mp_labs
docker build -t openarm-rocm:unified -f docker/Dockerfile docker
```

## 2. Run a container

The GPU device flags and groups are required (`render` group resolves by GID
`110` on this host — adjust if different: `getent group render`):

```bash
docker run --rm -it \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add 110 \
  --security-opt seccomp=unconfined \
  -v /DATA/AMD-Sim/OpenArm_Labs:/workspace \
  -w /workspace/openarm_mp_labs \
  openarm-rocm:unified bash
```

### Verify the three stacks (GPU)

```bash
# torch (GraspGenX side)
python -c "import torch;print(torch.cuda.get_device_name(0), float(torch.ones(8,device='cuda').sum()))"
# jax (PyRoki side)
/opt/venv-planner/bin/python -c "import jax;print(jax.devices())"
# mujoco + mink
/opt/venv-planner/bin/python -c "import mujoco,mink;print(mujoco.__version__,'mink ok')"
```

## 3. Install the OpenArm packages (planner venv)

The three local packages are mounted from the workspace and installed editable.
`--no-deps` is used because `openarm-control`/`openarm-mujoco` are not on PyPI;
their actual deps (mujoco, mink, daqp, numpy, pyyaml) are already in the venv.

```bash
/opt/venv-planner/bin/pip install --no-deps -e ../openarm_control -e ../openarm_mujoco -e .
```

## 4. Run the pick-and-place task (with recording)

```bash
/opt/venv-planner/bin/python -m openarm_mp_labs.demo_pick_place \
    --record output/pick_place_container.mp4 --fps 30 --render-every 8
```

`MUJOCO_GL=osmesa` is the image default. Software rendering is correct but slow
(~6 min for the full 71.7 s clip). Expected console tail:

```
Recorded 2150 frames (71.7s) -> /workspace/.../output/pick_place_container.mp4
Peak cube lift: 112.2 mm
```

## 5. Check results

- `Peak cube lift` should be ~112 mm (cube grasped at the curved fingertips,
  lifted, transported, placed).
- The MP4 is written under `output/` (git-ignored). Files created in-container
  are owned by `root`; fix on host with
  `sudo chown $(id -u):$(id -g) output/pick_place_container.mp4`.
- Quick visual check (host): extract a frame around the grasp/lift phase:
  ```bash
  ffmpeg -y -ss 28 -i output/pick_place_container.mp4 -frames:v 1 output/container_check.png
  ```
  Both camera views (front 45° + top-down) should show the cube pinched by the
  curved fingertips.

## One-shot (non-interactive)

```bash
docker run --rm \
  --device=/dev/kfd --device=/dev/dri --group-add video --group-add 110 \
  --security-opt seccomp=unconfined \
  -v /DATA/AMD-Sim/OpenArm_Labs:/workspace \
  -w /workspace/openarm_mp_labs openarm-rocm:unified bash -lc '
    /opt/venv-planner/bin/pip install -q --no-deps -e ../openarm_control -e ../openarm_mujoco -e .
    /opt/venv-planner/bin/python -m openarm_mp_labs.demo_pick_place \
        --record output/pick_place_container.mp4'
```
