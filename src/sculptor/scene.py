"""Helpers for the Swift scene."""

import base64
import io
import tarfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import spatialgeometry as sg
import spatialmath as sm
import swift
import trimesh
from IPython.display import HTML
from PIL import Image

from sculptor.cell import CHISEL_LENGTH, CHISEL_RADIUS, TIP_LENGTH, TIP_WIDTH, Cell
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
POLISHED_STEEL: list[float] = [0.55, 0.55, 0.58, 1.0]

# Only used for drawing [m]; the sizes of the chisel are in cell.py
TABLE_RADIUS: float = 0.22

# Side of a drawn cube over the side of its voxel when add_block shows the gaps
GAP_SCALE: float = 0.94

# Swift has no cones, so the tip of the chisel is a mesh made once with trimesh
TIP_MESH: Path = Path(__file__).resolve().parents[2] / "models" / "chisel_tip.stl"

# The body of the chisel and its pointed tip, in this order
Chisel = tuple[sg.Cylinder, sg.Mesh]


def add_table(env: swift.Swift, cell: Cell) -> None:
    """A cylinder from the floor to the table top. It looks the same at any angle, so it never moves."""
    x, y, height = cell.table_top.t
    env.add(sg.Cylinder(radius=TABLE_RADIUS, length=height, pose=sm.SE3(x, y, height / 2), color=WOOD))


def place_block(cubes: list[sg.Cuboid], voxels: Voxels, cell: Cell, phi: float) -> None:
    """Put every cube where it is with the table turned by phi."""
    T_block = cell.block_pose(phi)
    for cube, p in zip(cubes, voxels.centers):
        cube.T = T_block * sm.SE3(p)


def add_block(env: swift.Swift, voxels: Voxels, cell: Cell, phi: float = 0.0,
              gaps: bool = False) -> tuple[list[sg.Cuboid], list[int]]:
    """Add one cube per voxel. The ids are kept because removing by id is much faster.

    With gaps=True the cubes are drawn a bit smaller than the voxels, so the grid of the block
    shows. It is only for looking at the voxelization; contact and collisions use voxels.size.
    """
    side = voxels.size * (GAP_SCALE if gaps else 1.0)
    cubes = [sg.Cuboid(scale=[side] * 3, color=STATUE if s else MARBLE)
             for s in voxels.is_statue]
    place_block(cubes, voxels, cell, phi)
    ids = [env.add(cube) for cube in cubes]
    return cubes, ids


def _tip_mesh() -> Path:
    """The cone of the tip as an .stl file, with its base at z = 0 and its point at z = TIP_LENGTH."""
    if not TIP_MESH.exists():
        TIP_MESH.parent.mkdir(parents=True, exist_ok=True)
        trimesh.creation.cone(radius=TIP_WIDTH / 2, height=TIP_LENGTH, sections=24).export(TIP_MESH)
    return TIP_MESH


def place_chisel(chisel: Chisel, T_tip: sm.SE3) -> None:
    """Body and tip of the chisel, from the pose of the point."""
    body, tip = chisel
    body_length = CHISEL_LENGTH - TIP_LENGTH
    body.T = T_tip * sm.SE3(0.0, 0.0, -TIP_LENGTH - body_length / 2)
    tip.T = T_tip * sm.SE3(0.0, 0.0, -TIP_LENGTH)


def add_chisel(env: swift.Swift, T_tip: sm.SE3) -> Chisel:
    """A point chisel, a thin steel body with a polished cone at its end."""
    chisel = (sg.Cylinder(radius=CHISEL_RADIUS, length=CHISEL_LENGTH - TIP_LENGTH, color=STEEL),
              sg.Mesh(str(_tip_mesh()), color=POLISHED_STEEL))
    place_chisel(chisel, T_tip)
    for part in chisel:
        env.add(part)
    return chisel


# Recordings of the notebooks, apart from the numeric results. Names start with the notebook
# number, so one notebook can have several (01_cell_frames_turn.gif, ...)
RECORDINGS: Path = Path(__file__).resolve().parents[2] / "results" / "recordings"
# Swift can only ask the browser to download a recording, so it lands in the downloads folder
DOWNLOADS: Path = Path.home() / "Downloads"

# Same view in every recording, from the side so the robot and the table are both in sight
CAMERA_POSITION: list[float] = [0.45, -1.35, 0.85]
CAMERA_LOOK_AT: list[float] = [0.45, 0.0, 0.3]

# Part of the browser window that shows the robot and the table (left, top, right, bottom,
# as fractions of the window), and the size of the GIF, so it stays light
CROP: tuple[float, float, float, float] = (0.25, 0.18, 0.75, 0.78)
GIF_WIDTH: int = 560
KEEP_EVERY: int = 3


@contextmanager
def recording(env: swift.Swift, gif: Path, downloads: Path = DOWNLOADS) -> Iterator[None]:
    """Record what Swift draws inside the with block and save it as `gif`.

    Swift saves PNG frames, which the browser writes much faster than a GIF, and the GIF is
    built here from them. It plays at the real speed of the recording.
    """
    download = downloads / f"{gif.stem}.tar"
    # The browser renames a download that already exists, so an old one would hide the new one
    download.unlink(missing_ok=True)
    env.set_camera_pose(CAMERA_POSITION, CAMERA_LOOK_AT)
    # The framerate only names the frames; their real timing is measured below
    env.start_recording(gif.stem, 20, format="png")
    started = time.perf_counter()
    yield
    elapsed = time.perf_counter() - started
    env.stop_recording()
    _save_gif(download, gif, elapsed)


def _wait_for_download(download: Path, timeout: float) -> None:
    """Wait until the file exists and its size stops changing (the browser writes it at the end)."""
    deadline = time.time() + timeout
    last_size = -1
    while time.time() < deadline:
        if download.exists():
            size = download.stat().st_size
            if size > 0 and size == last_size:
                return
            last_size = size
        time.sleep(1.0)
    raise TimeoutError(f"{download} did not appear; is the browser saving downloads to {download.parent}?")


def _save_gif(download: Path, gif: Path, elapsed: float, timeout: float = 120.0) -> None:
    """Build the GIF from the downloaded frames, crop and shrink it, and delete the download."""
    _wait_for_download(download, timeout)
    with tarfile.open(download) as tar:
        names = sorted(m.name for m in tar.getmembers() if m.isfile())[::KEEP_EVERY]
        frames = [Image.open(io.BytesIO(tar.extractfile(name).read())).convert("RGB") for name in names]
    w, h = frames[0].size
    box = (int(w * CROP[0]), int(h * CROP[1]), int(w * CROP[2]), int(h * CROP[3]))
    size = (GIF_WIDTH, round(GIF_WIDTH * (box[3] - box[1]) / (box[2] - box[0])))
    frames = [frame.crop(box).resize(size) for frame in frames]
    # Spread the real duration over the frames we keep
    duration = round(1000 * elapsed / len(frames))
    gif.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(gif, save_all=True, append_images=frames[1:], duration=duration, loop=0, optimize=True)
    download.unlink()


def show_gif(gif: Path) -> HTML:
    """Show a GIF in the output. It is embedded, so the HTML export shows it without the file."""
    data = base64.b64encode(gif.read_bytes()).decode()
    return HTML(f'<img src="data:image/gif;base64,{data}" alt="{gif.stem}">')
