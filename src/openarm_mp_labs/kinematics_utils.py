# Copyright 2026 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Pose interpolation and IK helpers."""

from __future__ import annotations

import mujoco
import numpy as np
from openarm_control import Kinematics

from openarm_mp_labs.config import TCP_OFFSET_LOCAL


def slerp_quat(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = float(np.clip(np.dot(q0, q1), -1.0, 1.0))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        out = q0 + t * (q1 - q0)
        return (out / np.linalg.norm(out)).astype(np.float32)
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * t
    s0 = np.sin(theta_0 - theta) / sin_theta_0
    s1 = np.sin(theta) / sin_theta_0
    return (s0 * q0 + s1 * q1).astype(np.float32)


def lerp_pose(pose_a: np.ndarray, pose_b: np.ndarray, t: float) -> np.ndarray:
    pos = (1.0 - t) * pose_a[:3] + t * pose_b[:3]
    quat = slerp_quat(pose_a[3:7], pose_b[3:7], t)
    return np.concatenate([pos, quat]).astype(np.float32)


def make_pose(x: float, y: float, z: float, orientation: np.ndarray) -> np.ndarray:
    return np.array([x, y, z, *orientation], dtype=np.float32)


def _quat_to_mat(quat: np.ndarray) -> np.ndarray:
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, np.asarray(quat, dtype=np.float64))
    return mat.reshape(3, 3)


def fingertip_to_site_pose(
    fingertip_xyz: np.ndarray, orientation: np.ndarray
) -> np.ndarray:
    """Convert a desired fingertip-midpoint world target to a control-site IK pose.

    site_pos = fingertip_pos - R(orientation) @ TCP_OFFSET_LOCAL
    """
    rot = _quat_to_mat(orientation)
    site_pos = np.asarray(fingertip_xyz, dtype=np.float64) - rot @ TCP_OFFSET_LOCAL
    return np.array([*site_pos, *orientation], dtype=np.float32)


def pin_left_home(q16: np.ndarray, left_home: np.ndarray) -> np.ndarray:
    out = q16.copy()
    out[8:16] = left_home
    return out


def read_bimanual_driver(joint_resolver, qpos: np.ndarray) -> np.ndarray:
    r_j, r_g = joint_resolver.get_driver(qpos, "right")
    l_j, l_g = joint_resolver.get_driver(qpos, "left")
    return np.concatenate([np.append(r_j, r_g), np.append(l_j, l_g)]).astype(np.float32)


def solve_to_pose(
    kin: Kinematics,
    q16: np.ndarray,
    target_pose: np.ndarray,
    gripper: float,
    outer_iters: int = 15,
) -> np.ndarray | None:
    q = q16.copy()
    kin.set_gripper("right", gripper)
    for _ in range(outer_iters):
        kin.sync(q)
        kin.set_target("right", target_pose)
        result = kin.solve()
        if result is None:
            return None
        kin.set_gripper("right", gripper)
        q = result
        q[7] = gripper
    return q
