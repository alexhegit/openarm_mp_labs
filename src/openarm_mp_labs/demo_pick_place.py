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


_REPO_ROOT = Path(__file__).resolve().parents[2]
_ASSETS_DIR = _REPO_ROOT / "assets"

# Bundled non-cube manipulands shipped with the repo so the demo runs
# standalone. name -> (object MJCF, default GraspGenX grasps YAML).
# The ginger asset is a 3D scan from https://github.com/alexhegit/Scan2Sim,
# converted to MuJoCo by Scan2Sim and vendored here (see assets/ginger/).
_BUNDLED_OBJECTS = {
    "ginger": (
        _ASSETS_DIR / "ginger" / "ginger.xml",
        _ASSETS_DIR / "grasps" / "ginger_grasps.yml",
    ),
}


def _default_output_path() -> Path:
    return _REPO_ROOT / "output" / "pick_place_demo.mp4"


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
    parser.add_argument(
        "--object",
        type=str,
        default=None,
        help="Manipuland instead of the orange cube: a bundled name "
        f"({', '.join(sorted(_BUNDLED_OBJECTS))}) or a path to any Scan2Sim "
        "object MJCF. A bundled name also supplies its default --grasp-file.",
    )
    args = parser.parse_args()

    grasp_file = args.grasp_file
    object_mjcf = args.object
    if object_mjcf in _BUNDLED_OBJECTS:
        bundled_xml, bundled_grasps = _BUNDLED_OBJECTS[object_mjcf]
        object_mjcf = str(bundled_xml)
        if grasp_file is None:
            grasp_file = str(bundled_grasps)

    if object_mjcf is not None:
        from openarm_mp_labs.scene_builder import build_scanned_object_scene

        scene_xml = build_scanned_object_scene(object_mjcf)
        print(f"Built scanned-object scene: {scene_xml}")
    else:
        scene_xml = openarm_demo_xml()

    setup = ArmSetup.from_args(
        xml=scene_xml,
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
        setup, grasp_file=grasp_file, grasp_mode=args.grasp_mode, kin=kin
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
