# Copyright 2026 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Pick-and-place constants and scene tuning."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Right gripper driver range in MJCF: open=-0.7854, fully closed=0.0
GRIPPER_OPEN = -0.7854
GRIPPER_CLOSED = 0.0
# Grip command used when holding the 40 mm cube at the curved fingertips. The
# tips meet (gap -> 0) at the fully-closed command, so closing all the way drives
# the (kinematic) fingers straight through the cube. At the cube's tip location
# (local z = -0.150) the tip gap equals ~40 mm at this angle, so the curved tips
# actually pinch the cube with a slight squeeze instead of overshooting it.
GRASP_GRIP = -0.21

# Nominal cube spawn from demo.xml (settled z is lower after physics).
CUBE_SPAWN = np.array([0.45, 0.0, 1.05], dtype=np.float64)
PLACE_XY = np.array([0.47, 0.05], dtype=np.float64)

# Tool-center-point / grasp point in the EE control-site local frame. The fingers
# are CURVED pincers: at a closed gripper their distal TIPS meet (gap -> 0 mm) at
# local ~[0.02, 0, -0.165], while the proximal pad region stays ~44 mm apart and
# the mid-finger ~20 mm. To grasp the 40 mm cube with the curved fingertips the
# cube must sit at the distal tip region, NOT the proximal pad. This offset places
# the cube target at the tip pinch location; the same vector is used to seat the
# carried cube in the EE frame.
TCP_OFFSET_LOCAL = np.array([0.005, 0.0, -0.150], dtype=np.float64)

# Vertical offsets relative to settled cube center, applied to the FINGERTIP target.
PRE_GRASP_DZ = 0.10
GRASP_DZ = 0.0
LIFT_DZ = 0.12
PLACE_ABOVE_DZ = 0.10
PLACE_DZ = 0.01
RETREAT_DZ = 0.10

# Closed-loop IK during contact-sensitive phases reduces position-control drift.
CLOSED_LOOP_REIK_EVERY = 20
CLOSED_LOOP_INNER_ITERS = 8


# Gripper attach when mostly closed (open=-0.7854, closed=0.0). The gripper's
# fully-closed gap (~49 mm) barely exceeds the 40 mm cube, so a kinematic attach
# stabilizes the carry once the fingers actually straddle the cube.
ATTACH_GRIPPER_THRESHOLD = -0.35
ATTACH_DISTANCE_M = 0.08
EE_BODY_NAME = "openarm_right_ee_base_link"


@dataclass(frozen=True)
class PickPlaceTargets:
    cube_center: np.ndarray
    place_center: np.ndarray

    @classmethod
    def from_cube(cls, cube_center: np.ndarray) -> PickPlaceTargets:
        place = np.array([PLACE_XY[0], PLACE_XY[1], cube_center[2]], dtype=np.float64)
        return cls(cube_center=cube_center.copy(), place_center=place)
