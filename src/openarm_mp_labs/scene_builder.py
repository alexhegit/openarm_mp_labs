# Copyright 2026 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Build a demo scene that swaps the orange cube for a scanned mesh object.

Given a Scan2Sim-style object MJCF (``<obj>.xml`` + ``meshes/``), this emits a
self-contained scene XML that attaches the OpenArm cell and places the scanned
object (visual + convex-hull collision + estimated inertial) at the pick
location, plus the black place frame. All asset paths are absolute so the
generated file can live anywhere.
"""

from __future__ import annotations

import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from openarm_mujoco.v2 import openarm_cell_xml

# Matches demo.xml's cube spawn / place frame so the existing tuning carries over.
_DEFAULT_POS = (0.45, 0.0, 1.05)
_BLACK_FRAME = """
    <body name="black_frame" pos="0.47 0.15 1.005">
      <geom type="box" pos="0 0 0.005"    size="0.07 0.07 0.005" rgba="0.1 0.1 0.1 1" contype="1" conaffinity="1"/>
      <geom type="box" pos="0.065 0 0.02"  size="0.005 0.07 0.02" rgba="0.1 0.1 0.1 1" contype="1" conaffinity="1"/>
      <geom type="box" pos="-0.065 0 0.02" size="0.005 0.07 0.02" rgba="0.1 0.1 0.1 1" contype="1" conaffinity="1"/>
      <geom type="box" pos="0 0.065 0.02"  size="0.06 0.005 0.02" rgba="0.1 0.1 0.1 1" contype="1" conaffinity="1"/>
      <geom type="box" pos="0 -0.065 0.02" size="0.06 0.005 0.02" rgba="0.1 0.1 0.1 1" contype="1" conaffinity="1"/>
    </body>
"""


def build_scanned_object_scene(
    object_mjcf: str | Path,
    body_name: str = "orange_cube",
    pos: tuple[float, float, float] = _DEFAULT_POS,
) -> str:
    """Return a path to a generated scene XML using the scanned object.

    The body is named ``orange_cube`` by default so the manipulation code that
    looks it up keeps working unchanged; pass ``body_name`` to override.
    """
    object_mjcf = Path(object_mjcf).resolve()
    obj_dir = object_mjcf.parent
    tree = ET.parse(object_mjcf)
    root = tree.getroot()

    # meshdir from the object's compiler (default "meshes").
    compiler = root.find("compiler")
    meshdir = obj_dir / (compiler.get("meshdir", ".") if compiler is not None else ".")

    asset = root.find("asset")
    if asset is None:
        raise ValueError(f"No <asset> in {object_mjcf}")
    tex_file = vis_file = col_file = None
    for mesh in asset.findall("mesh"):
        f = mesh.get("file", "")
        if f.endswith(".stl"):
            col_file = (meshdir / f).resolve()
        else:
            vis_file = (meshdir / f).resolve()
    tex = asset.find("texture[@name='obj_tex']")
    if tex is not None:
        tex_file = (obj_dir / tex.get("file")).resolve()

    body = root.find(".//body")
    inertial = body.find("inertial") if body is not None else None
    inertial_xml = ET.tostring(inertial, encoding="unicode").strip() if inertial is not None else ""

    tex_asset = (
        f'<texture name="obj_tex" type="2d" file="{tex_file}"/>\n'
        f'    <material name="obj_mat" texture="obj_tex" specular="0.1" '
        f'shininess="0.1" rgba="1.0 0.8 0.5 1.0"/>'
        if tex_file is not None
        else '<material name="obj_mat" rgba="1.0 0.8 0.5 1.0"/>'
    )
    col_asset = (
        f'<mesh name="obj_collision" file="{col_file}"/>' if col_file else ""
    )
    col_geom = (
        f'<geom name="obj_col" type="mesh" mesh="obj_collision" '
        f'friction="1 0.5 0.01" contype="1" conaffinity="1" condim="6" '
        f'solref="0.004 1" rgba="0.8 0.3 0.3 0"/>'
        if col_file
        else ""
    )
    px, py, pz = pos
    scene = f"""<mujoco model="openarm scanned-object demo">
  <visual>
    <global offwidth="960" offheight="600"/>
  </visual>
  <asset>
    <model name="cell" file="{openarm_cell_xml()}"/>
    {tex_asset}
    <mesh name="obj_visual" file="{vis_file}"/>
    {col_asset}
  </asset>
  <worldbody>
    <attach model="cell" prefix=""/>

    <body name="{body_name}" pos="{px} {py} {pz}">
      <freejoint/>
      {inertial_xml}
      <geom name="obj_vis" type="mesh" mesh="obj_visual" material="obj_mat" contype="0" conaffinity="0" group="0"/>
      {col_geom}
    </body>
{_BLACK_FRAME}
  </worldbody>
</mujoco>
"""
    out = Path(tempfile.gettempdir()) / f"openarm_scene_{body_name}.xml"
    out.write_text(scene)
    return str(out)
