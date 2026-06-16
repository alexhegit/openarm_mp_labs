# Copyright 2026 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""MuJoCo simulation reset, physics tuning, replay, and recording."""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from openarm_control import ArmSetup, Kinematics

from openarm_mp_labs.config import (
    ATTACH_DISTANCE_M,
    CLOSED_LOOP_INNER_ITERS,
    CLOSED_LOOP_REIK_EVERY,
    EE_BODY_NAME,
    GRIPPER_OPEN,
    GRASP_GRIP,
    PickPlaceTargets,
    TCP_OFFSET_LOCAL,
)
from openarm_mp_labs.trajectory import TrajectoryFrame

# Phases where position-control drift must be corrected by re-solving IK from the
# actual simulated pose so the fingertips track the cube precisely.
CONTACT_PHASES = (
    "descend_grasp", "close_gripper", "lift", "transport", "descend_place", "open_gripper",
)

# Phases during which the cube may be captured by the gripper. Capturing as the
# fingers descend (before close) locks the cube before it can be knocked away.
ATTACH_PHASES = ("descend_grasp", "close_gripper", "lift", "transport", "descend_place")


def _in_attach_phase(label: str) -> bool:
    return any(label.startswith(p) for p in ATTACH_PHASES)


def settle_physics(model: mujoco.MjModel, data: mujoco.MjData, steps: int = 2000) -> None:
    for _ in range(steps):
        mujoco.mj_step(model, data)


def reset_sim_to_home(setup: ArmSetup, keyframe: str = "home", settle_steps: int = 2000) -> None:
    key_id = mujoco.mj_name2id(setup.model, mujoco.mjtObj.mjOBJ_KEY, keyframe)
    if key_id < 0:
        raise ValueError(f"Keyframe '{keyframe}' not found.")
    mujoco.mj_resetDataKeyframe(setup.model, setup.data, key_id)
    for i in range(setup.model.nu):
        jid = setup.model.actuator_trnid[i, 0]
        if jid >= 0:
            setup.data.ctrl[i] = setup.data.qpos[setup.model.jnt_qposadr[jid]]
    mujoco.mj_forward(setup.model, setup.data)
    settle_physics(setup.model, setup.data, settle_steps)


def read_cube_center(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    body_id = model.body("orange_cube").id
    return data.xpos[body_id].copy()


def tune_manipulation_physics(model: mujoco.MjModel) -> None:
    """Increase tracking and contact friction for reliable pick-and-place."""
    for i in range(1, 8):
        aid = model.actuator(f"right_joint{i}_ctrl").id
        model.actuator_gainprm[aid, 0] = 600.0
        model.actuator_biasprm[aid, 1] = -600.0
    finger_id = model.actuator("right_finger1_ctrl").id
    model.actuator_gainprm[finger_id, 0] = 400.0
    model.actuator_biasprm[finger_id, 1] = -400.0
    for gi in range(model.ngeom):
        if model.body(model.geom_bodyid[gi]).name == "orange_cube":
            model.geom_friction[gi] = [4.0, 0.005, 0.0001]
            model.geom_solref[gi] = [0.002, 1.0]


@dataclass
class CubeAttachment:
    """Kinematic carry: seat the cube at the curved fingertip pinch point.

    The grasp point is a fixed location in the EE control-site frame
    (``TCP_OFFSET_LOCAL`` = the distal curved-tip pinch position). Once the tips
    reach the cube it is locked there and, as the gripper closes, smoothly ramped
    from its picked-up pose to the seated pinch point (no teleport). The carried
    cube is clamped to the table height so releasing it can never pop it through
    the floor.
    """

    model: mujoco.MjModel
    site_id: int
    cube_body_id: int
    cube_qpos_slice: slice
    cube_dof_slice: slice
    floor_z: float
    attached: bool = False
    rest_offset: np.ndarray | None = None

    @classmethod
    def create(cls, model: mujoco.MjModel, site_id: int, floor_z: float) -> CubeAttachment:
        cube_body_id = model.body("orange_cube").id
        jnt_id = model.body_jntadr[cube_body_id]
        qpos_adr = model.jnt_qposadr[jnt_id]
        dof_adr = model.jnt_dofadr[jnt_id]
        return cls(
            model=model,
            site_id=site_id,
            cube_body_id=cube_body_id,
            cube_qpos_slice=slice(qpos_adr, qpos_adr + 7),
            cube_dof_slice=slice(dof_adr, dof_adr + 6),
            floor_z=floor_z,
        )

    def _pinch_point(self, data: mujoco.MjData) -> np.ndarray:
        """World position of the curved-fingertip pinch point in the EE frame."""
        site_pos = data.site_xpos[self.site_id]
        site_mat = data.site_xmat[self.site_id].reshape(3, 3)
        return site_pos + site_mat @ TCP_OFFSET_LOCAL

    def maybe_attach(self, data: mujoco.MjData, gripper: float) -> None:
        if self.attached:
            return
        cube_pos = data.xpos[self.cube_body_id]
        if np.linalg.norm(cube_pos - self._pinch_point(data)) > ATTACH_DISTANCE_M:
            return
        # Record where the cube was picked up relative to the pinch point so it can
        # be ramped into the tips as the gripper closes.
        self.rest_offset = np.asarray(cube_pos - self._pinch_point(data), dtype=np.float64)
        self.attached = True

    def _seat_factor(self, gripper: float) -> float:
        span = GRASP_GRIP - GRIPPER_OPEN
        return float(np.clip((gripper - GRIPPER_OPEN) / span, 0.0, 1.0))

    def apply(self, data: mujoco.MjData, gripper: float) -> None:
        if not self.attached or self.rest_offset is None:
            return
        pinch = self._pinch_point(data)
        s = self._seat_factor(gripper)
        target = pinch + (1.0 - s) * self.rest_offset
        target[2] = max(target[2], self.floor_z)
        adr = self.cube_qpos_slice.start
        data.qpos[adr : adr + 3] = target
        data.qvel[self.cube_dof_slice] = 0.0
        mujoco.mj_forward(self.model, data)

    def release(self) -> None:
        self.attached = False
        self.rest_offset = None


def sync_ctrl_from_frame(setup: ArmSetup, frame: TrajectoryFrame) -> None:
    resolver = setup.joint_resolver
    cmd = frame.q16.copy()
    cmd[7] = frame.gripper
    resolver.set_ctrl(setup.data.ctrl, cmd[:8], "right")
    resolver.set_ctrl(setup.data.ctrl, cmd[8:16], "left")


def closed_loop_correct(
    setup: ArmSetup,
    kin: Kinematics | None,
    frame: TrajectoryFrame,
    left_home: np.ndarray,
    step: int,
) -> None:
    """Re-solve IK from the actual simulated pose during contact phases.

    Position actuators leave a residual tracking error (observed up to ~45 mm in
    y); re-commanding the IK solution computed from the live qpos pulls the
    fingertips back onto the planned target.
    """
    if kin is None or frame.target_pose is None:
        return
    if not any(frame.label.startswith(p) for p in CONTACT_PHASES):
        return
    if step % CLOSED_LOOP_REIK_EVERY != 0:
        return
    resolver = setup.joint_resolver
    r_j, r_g = resolver.get_driver(setup.data.qpos, "right")
    cur = np.concatenate([r_j, [r_g], left_home]).astype(np.float32)
    for _ in range(CLOSED_LOOP_INNER_ITERS):
        kin.sync(cur)
        kin.set_target("right", frame.target_pose)
        kin.set_gripper("right", frame.gripper)
        solved = kin.solve()
        if solved is not None:
            cur = solved.copy()
            cur[8:16] = left_home
            cur[7] = frame.gripper
    resolver.set_ctrl(setup.data.ctrl, cur[:8], "right")
    resolver.set_ctrl(setup.data.ctrl, cur[8:16], "left")


def simulate_frames(
    setup: ArmSetup,
    frames: list[TrajectoryFrame],
    hold_steps: int,
    attachment: CubeAttachment | None = None,
    kin: Kinematics | None = None,
) -> float:
    """Run physics for all frames. Returns peak cube lift height (m)."""
    model, data = setup.model, setup.data
    cube_start = read_cube_center(model, data)
    left_home = frames[0].q16[8:16].copy()
    peak_lift = 0.0

    for frame in frames:
        sync_ctrl_from_frame(setup, frame)
        if frame.label == "retreat" and attachment is not None:
            attachment.release()
        for step in range(hold_steps):
            closed_loop_correct(setup, kin, frame, left_home, step)
            mujoco.mj_step(model, data)
            if attachment is not None:
                if _in_attach_phase(frame.label):
                    attachment.maybe_attach(data, frame.gripper)
                if attachment.attached:
                    attachment.apply(data, frame.gripper)
            peak_lift = max(peak_lift, read_cube_center(model, data)[2] - cube_start[2])

    return float(peak_lift)


def replay_trajectory(
    setup: ArmSetup,
    frames: list[TrajectoryFrame],
    hold_steps: int,
    kin: Kinematics | None = None,
) -> None:
    model, data = setup.model, setup.data
    reset_sim_to_home(setup)
    tune_manipulation_physics(model)
    attachment = CubeAttachment.create(
        model, setup.frame_ids["right"], floor_z=read_cube_center(model, data)[2]
    )
    left_home = frames[0].q16[8:16].copy()
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0.45, 0.0, 1.05]
        viewer.cam.distance = 1.6
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -35
        for frame in frames:
            sync_ctrl_from_frame(setup, frame)
            if frame.label == "retreat":
                attachment.release()
            for step in range(hold_steps):
                if not viewer.is_running():
                    return
                step_start = time.time()
                closed_loop_correct(setup, kin, frame, left_home, step)
                mujoco.mj_step(model, data)
                if _in_attach_phase(frame.label):
                    attachment.maybe_attach(data, frame.gripper)
                if attachment.attached:
                    attachment.apply(data, frame.gripper)
                viewer.sync()
                elapsed = time.time() - step_start
                time.sleep(max(0.0, model.opt.timestep - elapsed))


@dataclass
class RenderView:
    """One camera pane in the recorded video."""

    camera: mujoco.MjvCamera | int
    track_gripper: bool


def _front_chase_camera() -> mujoco.MjvCamera:
    # Front 45-degrees-from-above 3/4 view of the right arm. The cell is enclosed
    # except its open top, so only a small orbit distance stays inside the walls.
    # The lookat is retargeted to the gripper each frame (chase) in the render loop.
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.44, 0.0, 1.05]
    cam.distance = 0.75
    cam.azimuth = 135
    cam.elevation = -45
    return cam


def _build_render_views(model: mujoco.MjModel) -> list[RenderView]:
    """Two panes: a front 45-degree chase view and the top-down ceiling view."""
    views = [RenderView(camera=_front_chase_camera(), track_gripper=True)]
    ceiling_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "camera_ceiling")
    if ceiling_id >= 0:
        views.append(RenderView(camera=ceiling_id, track_gripper=False))
    else:
        top = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(top)
        top.type = mujoco.mjtCamera.mjCAMERA_FREE
        top.lookat[:] = [0.45, 0.0, 1.03]
        top.distance = 1.5
        top.azimuth = 90
        top.elevation = -89
        views.append(RenderView(camera=top, track_gripper=False))
    return views


def _ffmpeg_path() -> str:
    for candidate in ("/usr/bin/ffmpeg", "ffmpeg"):
        try:
            proc = subprocess.run(
                [candidate, "-encoders"],
                capture_output=True,
                text=True,
                check=False,
            )
            if "libx264" in proc.stdout:
                return candidate
        except OSError:
            continue
    return "ffmpeg"


def record_trajectory(
    setup: ArmSetup,
    frames: list[TrajectoryFrame],
    hold_steps: int,
    output_path: Path,
    *,
    width: int = 960,
    height: int = 600,
    fps: int = 30,
    render_every: int = 8,
    kin: Kinematics | None = None,
) -> float:
    model, data = setup.model, setup.data
    reset_sim_to_home(setup)
    tune_manipulation_physics(model)
    cube_start = read_cube_center(model, data)
    attachment = CubeAttachment.create(
        model, setup.frame_ids["right"], floor_z=cube_start[2]
    )
    left_home = frames[0].q16[8:16].copy()
    peak_lift = 0.0

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    renderer = mujoco.Renderer(model, height=height, width=width)
    views = _build_render_views(model)
    scene_option = mujoco.MjvOption()
    ee_body_id = model.body(EE_BODY_NAME).id
    # Per-view smoothed chase target for the tracking cameras.
    cam_targets = [
        np.array(v.camera.lookat, dtype=np.float64)
        if isinstance(v.camera, mujoco.MjvCamera)
        else np.zeros(3)
        for v in views
    ]
    total_width = width * len(views)

    ffmpeg_cmd = [
        _ffmpeg_path(),
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{total_width}x{height}",
        "-pix_fmt",
        "rgb24",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    proc = subprocess.Popen(
        ffmpeg_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    if proc.stdin is None:
        raise RuntimeError("Failed to open ffmpeg stdin.")

    frame_idx = 0
    try:
        for frame in frames:
            sync_ctrl_from_frame(setup, frame)
            if frame.label == "retreat":
                attachment.release()
            for step in range(hold_steps):
                closed_loop_correct(setup, kin, frame, left_home, step)
                mujoco.mj_step(model, data)
                if _in_attach_phase(frame.label):
                    attachment.maybe_attach(data, frame.gripper)
                if attachment.attached:
                    attachment.apply(data, frame.gripper)
                peak_lift = max(peak_lift, read_cube_center(model, data)[2] - cube_start[2])
                if step % render_every == 0:
                    ee_pos = data.xpos[ee_body_id]
                    chase = np.array([ee_pos[0], ee_pos[1], ee_pos[2] - 0.05])
                    panes = []
                    for vi, view in enumerate(views):
                        if view.track_gripper and isinstance(view.camera, mujoco.MjvCamera):
                            # Smoothly chase the gripper so the grasp stays framed
                            # and the camera stays inside the enclosed cell.
                            cam_targets[vi] += 0.15 * (chase - cam_targets[vi])
                            view.camera.lookat[:] = cam_targets[vi]
                        renderer.update_scene(
                            data, camera=view.camera, scene_option=scene_option
                        )
                        panes.append(renderer.render().copy())
                    proc.stdin.write(np.concatenate(panes, axis=1).tobytes())
                    frame_idx += 1
    finally:
        proc.stdin.close()
        stderr = proc.stderr.read().decode() if proc.stderr else ""
        ret = proc.wait()
        renderer.close()
        if ret != 0:
            raise RuntimeError(f"ffmpeg failed (exit {ret}): {stderr[-2000:]}")

    duration = frame_idx / fps
    print(f"Recorded {frame_idx} frames ({duration:.1f}s) -> {output_path}")
    print(f"Peak cube lift: {peak_lift * 1000:.1f} mm")
    return peak_lift


def prepare_targets(setup: ArmSetup) -> PickPlaceTargets:
    reset_sim_to_home(setup)
    tune_manipulation_physics(setup.model)
    cube = read_cube_center(setup.model, setup.data)
    print(f"Settled cube center: [{cube[0]:.3f}, {cube[1]:.3f}, {cube[2]:.3f}]")
    return PickPlaceTargets.from_cube(cube)
