# Unified AMD ROCm environment for OpenArm general-grasping integration.
#
# One image carrying all three stacks so the whole pipeline runs on a single
# AMD GPU host (verified: AMD Radeon PRO W7900 / gfx1100, host ROCm 6.3.2 +
# amdgpu-dkms 6.10.5 — a ROCm 7.2 container runs fine on it):
#
#   * PyTorch-ROCm  -> GraspGenX (6-DOF grasp generation)        [system python]
#   * JAX-ROCm      -> PyRoki (kinematic optimization / IK)       [/opt/venv-planner]
#   * MuJoCo + mink -> openarm_mp_labs trajectory + replay (CPU)  [/opt/venv-planner]
#
# WHY TWO VENVS: torch-rocm and jax-rocm each ship their own bundled ROCm runtime
# and pin different shared deps (e.g. numpy). Installing both into one
# site-packages makes even *torch alone* segfault. Keeping them in separate
# virtualenvs isolates the ROCm libs / numpy ABI. The integration runs GraspGenX
# (torch, system python, via its ZMQ server) and the planner (jax+PyRoki+MuJoCo,
# venv-planner) as separate processes — the natural split for this pipeline.
# NOTE: must use a torch-rocm build whose arch list includes gfx1100 (RDNA3 /
# Radeon PRO W7900). The pinned `..._pytorch_release_2.7.1` image is built only
# for gfx942/gfx1201/gfx950 and SEGFAULTS on gfx1100 GPU compute. `rocm/pytorch:latest`
# (torch 2.10.0+rocm7.2.3) lists gfx1100 and runs correctly on the W7900.
FROM rocm/pytorch:latest

ARG ROCM_JAX_REPO=https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/
ARG JAX_VERSION=0.9.2

# System libs: ffmpeg (video encode) + OSMesa (headless software GL for MuJoCo
# offscreen rendering; mesa EGL fails to init on the W7900 inside the container).
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libosmesa6 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Planner venv: JAX-ROCm (PyRoki) + MuJoCo physics + mink IK. Isolated from the
# system torch install so neither side's ROCm runtime / numpy clobbers the other.
RUN python -m venv /opt/venv-planner \
    && /opt/venv-planner/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/venv-planner/bin/pip install --no-cache-dir \
        "jax==${JAX_VERSION}" "jaxlib==${JAX_VERSION}" \
        "jax-rocm7-pjrt==${JAX_VERSION}" "jax-rocm7-plugin==${JAX_VERSION}" \
        -f "${ROCM_JAX_REPO}" \
    && /opt/venv-planner/bin/pip install --no-cache-dir \
        "mujoco>=3.6.0" "mink>=1.1.0" "daqp==0.7.2" "numpy" "pyyaml"

# Headless rendering for MuJoCo offscreen video recording (software OSMesa).
ENV MUJOCO_GL=osmesa
ENV PYOPENGL_PLATFORM=osmesa
ENV VENV_PLANNER=/opt/venv-planner/bin/python

WORKDIR /workspace
