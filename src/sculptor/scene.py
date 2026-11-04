"""Helpers for the Swift scene."""

from pathlib import Path
from typing import Any

import spatialgeometry as sg
import spatialmath as sm
import swift

from sculptor.cell import CHISEL_LENGTH, Cell
from sculptor.model import Voxels

# Kept before fix_mesh_paths replaces it, so calling the fix twice does no harm
_mesh_to_dict = sg.Mesh.to_dict


def _mesh_to_dict_posix(self: sg.Mesh) -> dict[str, Any]:
    shape = _mesh_to_dict(self)
    if shape.get("filename"):
        shape["filename"] = Path(shape["filename"]).as_posix()
    return shape


def fix_mesh_paths() -> None:
    """Send the paths of the meshes to Swift with forward slashes.

    On Windows the paths have backslashes, and the Swift page asks its file server for them
    as they are. The server only understands forward slashes, so no mesh loads, not even the
    robot. Call this once before adding anything to Swift. On macOS and Linux it changes nothing.
    """
    sg.Mesh.to_dict = _mesh_to_dict_posix


# One color per thing in the scene; marble and statue checked with palette_check.py
MARBLE: list[float] = [0.92, 0.92, 0.88, 1.0]   # #ebebe0
STATUE: list[float] = [0.63, 0.32, 0.18, 1.0]   # #a0522d
WOOD: list[float] = [0.40, 0.30, 0.20, 1.0]
STEEL: list[float] = [0.30, 0.30, 0.30, 1.0]
TIP_RED: list[float] = [0.85, 0.10, 0.10, 1.0]

# Sizes used only for drawing [m]. The chisel is thin so it does not hide the cubes
# around the one it touches, and the ball marks the point that removes them
TABLE_RADIUS: float = 0.22
CHISEL_RADIUS: float = 0.008
TIP_RADIUS: float = 0.012

# The two parts of the chisel, always in this order: the body and the ball at its tip
Chisel = tuple[sg.Cylinder, sg.Sphere]


def add_table(env: swift.Swift, cell: Cell) -> None:
    """A cylinder from the floor to the table top. It looks the same at any angle, so it never moves."""
    x, y, height = cell.table_top.t
    env.add(sg.Cylinder(radius=TABLE_RADIUS, length=height, pose=sm.SE3(x, y, height / 2), color=WOOD))


def place_block(cubes: list[sg.Cuboid], voxels: Voxels, cell: Cell, phi: float) -> None:
    """Put every cube where it is with the table turned by phi."""
    T_block = cell.block_pose(phi)
    for cube, p in zip(cubes, voxels.centers):
        cube.T = T_block * sm.SE3(p)


def add_block(env: swift.Swift, voxels: Voxels, cell: Cell, phi: float = 0.0) -> tuple[list[sg.Cuboid], list[int]]:
    """Add one cube per voxel. The ids are kept because removing by id is much faster."""
    cubes = [sg.Cuboid(scale=[voxels.size] * 3, color=STATUE if s else MARBLE)
             for s in voxels.is_statue]
    place_block(cubes, voxels, cell, phi)
    ids = [env.add(cube) for cube in cubes]
    return cubes, ids


def place_chisel(chisel: Chisel, T_tip: sm.SE3) -> None:
    """Body and tip of the chisel, from the pose of the tip."""
    body, tip = chisel
    body.T = T_tip * sm.SE3(0.0, 0.0, -CHISEL_LENGTH / 2)
    tip.T = T_tip


def add_chisel(env: swift.Swift, T_tip: sm.SE3) -> Chisel:
    """A thin cylinder with a red ball at the tip, so the contact point is easy to see."""
    chisel = (sg.Cylinder(radius=CHISEL_RADIUS, length=CHISEL_LENGTH, color=STEEL),
              sg.Sphere(radius=TIP_RADIUS, color=TIP_RED))
    place_chisel(chisel, T_tip)
    for part in chisel:
        env.add(part)
    return chisel
