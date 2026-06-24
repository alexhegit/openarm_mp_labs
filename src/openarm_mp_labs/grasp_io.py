# Copyright 2026 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Read GraspGenX 6-DOF grasps and map them into the MuJoCo world / EE frame.

GraspGenX (``isaac_grasp`` YAML) emits grasps in the **object mesh frame**. Each
grasp pose is the gripper *base* frame (``openarm_left_ee_base_link``):

* origin ``position`` = gripper base in the object frame
* ``+z`` axis = the approach direction (pointing from the base into the object);
  the fingertips sit at ``+z * 0.068`` per ``assets/proc_grippers/openarm``.

This module converts a selected grasp into the quantities the trajectory layer
needs: a fingertip-midpoint world target and a control-site world orientation
that matches the existing ``TCP_OFFSET_LOCAL`` convention (fingertips point along
the site-local ``TCP_OFFSET_LOCAL`` direction, i.e. roughly site ``-z``).
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
import yaml

from openarm_mp_labs.config import GRASP_DEPTH_M, TCP_OFFSET_LOCAL


@dataclass(frozen=True)
class Grasp:
    """A single GraspGenX grasp in the object mesh frame."""

    confidence: float
    position: np.ndarray  # (3,) gripper base in object frame
    quat: np.ndarray  # (4,) wxyz, gripper base orientation in object frame

    def rotation(self) -> np.ndarray:
        mat = np.zeros(9)
        mujoco.mju_quat2Mat(mat, np.asarray(self.quat, dtype=np.float64))
        return mat.reshape(3, 3)

    def approach(self) -> np.ndarray:
        """Unit approach axis (object frame): gripper +z, base -> object."""
        return self.rotation() @ np.array([0.0, 0.0, 1.0])


def load_grasps(path: str) -> list[Grasp]:
    with open(path) as f:
        doc = yaml.safe_load(f)
    if not doc or "grasps" not in doc:
        raise ValueError(f"No 'grasps' in {path}")
    grasps: list[Grasp] = []
    for g in doc["grasps"].values():
        o = g["orientation"]
        quat = np.array([o["w"], *o["xyz"]], dtype=np.float64)
        quat /= np.linalg.norm(quat)
        grasps.append(
            Grasp(
                confidence=float(g["confidence"]),
                position=np.array(g["position"], dtype=np.float64),
                quat=quat,
            )
        )
    return grasps


def select_grasp(grasps: list[Grasp], mode: str = "topdown") -> Grasp:
    """Pick a grasp.

    * ``topdown``  : most downward-pointing approach (safest for the pinch
      attachment / grip tuning, which were calibrated for a vertical grasp).
    * ``best``     : highest confidence among *feasible* grasps (gripper above
      the object and approach not pointing upward — i.e. not grasping from
      underneath the support surface).
    """
    if not grasps:
        raise ValueError("No grasps to select from")

    def downward(g: Grasp) -> float:
        return float(-g.approach()[2])  # +1 == straight down

    if mode == "topdown":
        return max(grasps, key=downward)
    if mode in ("best", "full"):
        feasible = [g for g in grasps if downward(g) > -0.1 and g.position[2] > -0.02]
        pool = feasible or grasps
        return max(pool, key=lambda g: g.confidence)
    raise ValueError(f"Unknown grasp mode: {mode}")


def _rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Minimal rotation matrix taking unit vector ``a`` to unit vector ``b``."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c > 1.0 - 1e-9:
        return np.eye(3)
    if c < -1.0 + 1e-9:
        # 180 deg: rotate about any axis orthogonal to a.
        axis = np.cross(a, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, np.array([0.0, 1.0, 0.0]))
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    s = np.linalg.norm(v)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s * s))


def _mat_to_quat(mat: np.ndarray) -> np.ndarray:
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.asarray(mat, dtype=np.float64).reshape(9))
    return quat


def _quat_to_mat(quat: np.ndarray) -> np.ndarray:
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, np.asarray(quat, dtype=np.float64))
    return mat.reshape(3, 3)


@dataclass(frozen=True)
class WorldGrasp:
    """A grasp resolved into the MuJoCo world / EE-site convention."""

    fingertip_world: np.ndarray  # (3,) where the fingertip pinch should close
    site_quat: np.ndarray  # (4,) wxyz control-site orientation
    approach_world: np.ndarray  # (3,) unit, base -> object (descend direction)
    confidence: float


def resolve_grasp(
    grasp: Grasp,
    object_center_world: np.ndarray,
    object_quat_world: np.ndarray,
    home_quat: np.ndarray,
    mode: str = "topdown",
) -> WorldGrasp:
    """Map an object-frame grasp into world fingertip target + site orientation.

    The fingertip target is the grasp's world contact point
    ``base + GRASP_DEPTH_M * approach`` (object-agnostic: works for asymmetric
    objects where the contact is not the centroid). The site orientation aligns
    the site-local fingertip axis (``TCP_OFFSET_LOCAL`` direction) with the
    grasp's world approach axis via the minimal twist from the home orientation,
    so a near-vertical grasp reproduces the validated top-down pose.
    """
    home_quat = np.asarray(home_quat, dtype=np.float64)
    R_obj2world = _quat_to_mat(object_quat_world)
    object_center_world = np.asarray(object_center_world, dtype=np.float64)

    # Grasp pose -> world. The object mesh frame origin coincides with the
    # manipuland body origin (the mesh is centered on its centroid).
    base_world = object_center_world + R_obj2world @ grasp.position
    approach_world = R_obj2world @ grasp.approach()
    approach_world /= np.linalg.norm(approach_world)
    contact_world = base_world + GRASP_DEPTH_M * approach_world

    if mode == "topdown":
        # Keep the calibrated home EE orientation and descend straight down;
        # GraspGenX still picks WHERE to grasp (contact point) and selects among
        # candidates. Stays in the proven physics regime.
        return WorldGrasp(
            fingertip_world=contact_world,
            site_quat=home_quat.copy(),
            approach_world=np.array([0.0, 0.0, -1.0]),
            confidence=grasp.confidence,
        )

    # full / best: use the grasp's actual 6-DOF orientation.
    u_local = TCP_OFFSET_LOCAL / np.linalg.norm(TCP_OFFSET_LOCAL)
    R_home = _quat_to_mat(home_quat)
    a_home = R_home @ u_local  # world approach at home (~ straight down)
    R_align = _rotation_between(a_home, approach_world)
    site_quat = _mat_to_quat(R_align @ R_home)

    return WorldGrasp(
        fingertip_world=contact_world,
        site_quat=site_quat,
        approach_world=approach_world,
        confidence=grasp.confidence,
    )
