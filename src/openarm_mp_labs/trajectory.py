# Copyright 2026 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Cartesian pick-and-place trajectory generation via IK."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from openarm_control import ArmSetup, Kinematics

from openarm_mp_labs.config import (
    GRASP_DZ,
    GRIPPER_OPEN,
    GRASP_GRIP,
    LIFT_DZ,
    PLACE_ABOVE_DZ,
    PLACE_DZ,
    PRE_GRASP_DZ,
    RETREAT_DZ,
    PickPlaceTargets,
)
from openarm_mp_labs.kinematics_utils import (
    fingertip_to_site_pose,
    lerp_pose,
    pin_left_home,
    read_bimanual_driver,
    solve_to_pose,
)


@dataclass
class TrajectoryFrame:
    q16: np.ndarray
    label: str
    gripper: float
    target_pose: np.ndarray | None = None  # right-arm control-site IK target [7]


def build_poses(home_pose: np.ndarray, targets: PickPlaceTargets) -> dict[str, np.ndarray]:
    """Build control-site IK targets so the FINGERTIPS reach the cube/place points.

    All waypoints (except home) are specified as fingertip-midpoint world targets
    and converted to site poses via the calibrated TCP offset.

    When ``targets`` carries a 6-DOF grasp (e.g. from GraspGenX), the grasp
    orientation drives the EE orientation and the pre-grasp backs off along the
    grasp's approach axis; otherwise the calibrated top-down vertical grasp (home
    orientation, back off straight up) is used.
    """
    ori = targets.grasp_quat if targets.grasp_quat is not None else home_pose[3:7].copy()
    ori = np.asarray(ori, dtype=np.float32)
    # Unit direction to back off FROM the object (opposite the approach).
    if targets.approach_world is not None:
        backoff = -np.asarray(targets.approach_world, dtype=np.float64)
        backoff /= np.linalg.norm(backoff)
    else:
        backoff = np.array([0.0, 0.0, 1.0])  # top-down: back off straight up
    up = np.array([0.0, 0.0, 1.0])  # lift is always vertical
    cube = np.asarray(targets.cube_center, dtype=np.float64)
    px, py, pz = targets.place_center

    def ft(world_xyz: np.ndarray) -> np.ndarray:
        return fingertip_to_site_pose(np.asarray(world_xyz, dtype=np.float64), ori)

    return {
        "home": home_pose.copy(),
        "pre_grasp": ft(cube + backoff * PRE_GRASP_DZ),
        "grasp": ft(cube + backoff * GRASP_DZ),
        "lift": ft(cube + up * LIFT_DZ),
        "place_above": ft(np.array([px, py, pz + PLACE_ABOVE_DZ])),
        "place": ft(np.array([px, py, pz + PLACE_DZ])),
        "retreat": ft(np.array([px, py, pz + RETREAT_DZ])),
    }


def append_segment(
    frames: list[TrajectoryFrame],
    kin: Kinematics,
    start_pose: np.ndarray,
    end_pose: np.ndarray,
    q16: np.ndarray,
    left_home: np.ndarray,
    gripper: float,
    steps: int,
    label: str,
    *,
    refine_iters: int = 25,
) -> np.ndarray:
    current = q16.copy()
    for i in range(steps):
        t = (i + 1) / steps
        pose = lerp_pose(start_pose, end_pose, t)
        solved = solve_to_pose(kin, current, pose, gripper)
        if solved is None:
            raise RuntimeError(f"IK failed during '{label}' at step {i + 1}/{steps}")
        current = pin_left_home(solved, left_home)
        current[7] = gripper
        frames.append(
            TrajectoryFrame(
                q16=current.copy(), label=label, gripper=gripper,
                target_pose=pose.copy(),
            )
        )

    if refine_iters > 0:
        solved = solve_to_pose(kin, current, end_pose, gripper, outer_iters=refine_iters)
        if solved is None:
            raise RuntimeError(f"IK failed refining '{label}' at target pose")
        current = pin_left_home(solved, left_home)
        current[7] = gripper
        frames.append(
            TrajectoryFrame(
                q16=current.copy(), label=f"{label}_refine", gripper=gripper,
                target_pose=end_pose.copy(),
            )
        )
    return current


def append_gripper_ramp(
    frames: list[TrajectoryFrame],
    q16: np.ndarray,
    left_home: np.ndarray,
    gripper_start: float,
    gripper_end: float,
    steps: int,
    label: str,
    hold_pose: np.ndarray | None = None,
) -> np.ndarray:
    current = pin_left_home(q16.copy(), left_home)
    for i in range(steps):
        t = (i + 1) / steps
        grip = gripper_start + t * (gripper_end - gripper_start)
        current[7] = grip
        frames.append(
            TrajectoryFrame(
                q16=current.copy(), label=label, gripper=float(grip),
                target_pose=None if hold_pose is None else hold_pose.copy(),
            )
        )
    return current


def generate_trajectory(
    setup: ArmSetup,
    kin: Kinematics,
    targets: PickPlaceTargets,
) -> list[TrajectoryFrame]:
    q16 = read_bimanual_driver(setup.joint_resolver, setup.data.qpos)
    left_home = q16[8:16].copy()
    home_pose = kin.fk("right", q16[:8])
    poses = build_poses(home_pose, targets)

    frames: list[TrajectoryFrame] = []
    current = pin_left_home(q16.copy(), left_home)
    current[7] = GRIPPER_OPEN
    frames.append(TrajectoryFrame(q16=current.copy(), label="home", gripper=GRIPPER_OPEN))

    current = append_segment(
        frames, kin, poses["home"], poses["pre_grasp"], current, left_home,
        GRIPPER_OPEN, steps=25, label="approach",
    )
    current = append_segment(
        frames, kin, poses["pre_grasp"], poses["grasp"], current, left_home,
        GRIPPER_OPEN, steps=18, label="descend_grasp",
    )
    current = append_gripper_ramp(
        frames, current, left_home, GRIPPER_OPEN, GRASP_GRIP,
        steps=24, label="close_gripper", hold_pose=poses["grasp"],
    )
    current = append_segment(
        frames, kin, poses["grasp"], poses["lift"], current, left_home,
        GRASP_GRIP, steps=22, label="lift",
    )
    current = append_segment(
        frames, kin, poses["lift"], poses["place_above"], current, left_home,
        GRASP_GRIP, steps=35, label="transport",
    )
    current = append_segment(
        frames, kin, poses["place_above"], poses["place"], current, left_home,
        GRASP_GRIP, steps=18, label="descend_place",
    )
    current = append_gripper_ramp(
        frames, current, left_home, GRASP_GRIP, GRIPPER_OPEN,
        steps=20, label="open_gripper", hold_pose=poses["place"],
    )
    current = append_segment(
        frames, kin, poses["place"], poses["retreat"], current, left_home,
        GRIPPER_OPEN, steps=15, label="retreat",
    )
    append_segment(
        frames, kin, poses["retreat"], poses["home"], current, left_home,
        GRIPPER_OPEN, steps=30, label="return_home",
    )
    return frames


def trajectory_summary(
    frames: list[TrajectoryFrame],
    kin: Kinematics,
    targets: PickPlaceTargets,
    home_pose: np.ndarray,
) -> None:
    print(f"Generated {len(frames)} trajectory frames")
    labels: dict[str, int] = {}
    for frame in frames:
        labels[frame.label] = labels.get(frame.label, 0) + 1
    for label, count in labels.items():
        print(f"  {label}: {count}")

    target_poses = build_poses(home_pose, targets)
    for label_key, pose_name in (
        ("descend_grasp_refine", "grasp"),
        ("descend_place_refine", "place"),
    ):
        idx = next(i for i, f in enumerate(frames) if f.label == label_key)
        ee = kin.fk("right", frames[idx].q16[:8])
        err = np.linalg.norm(ee[:3] - target_poses[pose_name][:3])
        print(f"  site IK error at {label_key}: {err * 1000:.1f} mm")
