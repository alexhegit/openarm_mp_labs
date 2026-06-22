#!/usr/bin/env python3
# Copyright 2026 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""OpenArm MuJoCo pick-and-place demo (right arm, demo.xml)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from openarm_control import ArmSetup, IKParams, Kinematics
from openarm_mujoco.v2 import openarm_demo_xml

from openarm_mp_labs.simulation import (
    prepare_targets,
    record_trajectory,
    replay_trajectory,
    simulate_frames,
    reset_sim_to_home,
    tune_manipulation_physics,
)
from openarm_mp_labs.trajectory import generate_trajectory, trajectory_summary


def _default_output_path() -> Path:
    return Path(__file__).resolve().parents[2] / "output" / "pick_place_demo.mp4"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="OpenArm pick-and-place on demo.xml (right arm)."
    )
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--simulate-only", action="store_true", help="Physics replay, no viewer/video")
    parser.add_argument("--hold-steps", type=int, default=80)
    parser.add_argument("--record", type=Path, nargs="?", const=_default_output_path())
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--render-every", type=int, default=8)
    parser.add_argument(
        "--grasp-file",
        type=str,
        default=None,
        help="GraspGenX isaac_grasp YAML; drives the grasp pose instead of the "
        "hardcoded top-down cube pose.",
    )
    parser.add_argument(
        "--grasp-mode",
        choices=("topdown", "best", "full"),
        default="topdown",
        help="topdown: force vertical approach (validated regime); "
        "best/full: use GraspGenX's selected 6-DOF orientation.",
    )
    args = parser.parse_args()

    setup = ArmSetup.from_args(
        xml=openarm_demo_xml(),
        mode="right",
        frame_right="right_ee_control_point",
        frame_type_right="site",
        frame_left="left_ee_control_point",
        frame_type_left="site",
        keyframe="home",
    )
    kin = Kinematics(
        setup,
        IKParams(damping=0.25, posture_cost=0.01, max_iters=50, dt=0.05),
    )

    targets = prepare_targets(
        setup, grasp_file=args.grasp_file, grasp_mode=args.grasp_mode, kin=kin
    )
    print("Generating pick-and-place trajectory...")
    try:
        frames = generate_trajectory(setup, kin, targets)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    home_pose = kin.fk("right", frames[0].q16[:8])
    trajectory_summary(frames, kin, targets, home_pose)

    if args.generate_only:
        return 0

    if args.simulate_only:
        reset_sim_to_home(setup)
        tune_manipulation_physics(setup.model)
        from openarm_mp_labs.simulation import CubeAttachment, read_cube_center

        floor_z = read_cube_center(setup.model, setup.data)[2]
        lift = simulate_frames(
            setup,
            frames,
            args.hold_steps,
            CubeAttachment.create(setup.model, setup.frame_ids["right"], floor_z=floor_z),
            kin=kin,
        )
        print(f"Simulated cube lift: {lift * 1000:.1f} mm")
        return 0 if lift > 0.03 else 1

    if args.record is not None:
        lift = record_trajectory(
            setup,
            frames,
            hold_steps=args.hold_steps,
            output_path=args.record,
            fps=args.fps,
            render_every=args.render_every,
            kin=kin,
        )
        return 0 if lift > 0.03 else 1

    replay_trajectory(setup, frames, hold_steps=args.hold_steps, kin=kin)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
