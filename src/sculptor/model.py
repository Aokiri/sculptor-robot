"""The statue as a set of cubes in the block frame

The block frame has its origin at the center of the bottom face of the block, with z up.
It is the frame the table carries (see cell.py), so a cube never changes its coordinates 
while the table turns.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

import trimesh

# Chosen in notebook 00 from the step times of Swift
CUBE_SIZE: float = 0.03

# The arm reaches the whole block up to 0.54 m and Swift draws it in time; 0.42 has fewer cubes,
# so every run is shorter (notebook 03)
STATUE_HEIGHT: float = 0.42


@dataclass(frozen=True)
class Voxels:
    """The cubes of the block that are loaded in Swift.
    
    These are the cubes of a box around the statue (one layer of margin on the sides and on top).
    `drop_interior` can remove the statue cubes that cannot be seen, the ones the chisel
    could never "damage".
    """
    
    centers: np.ndarray # (N, 3) in the block frame, metres
    is_statue: np.ndarray   # (N,) True for the cubes of the statue
    size: float # side of a cube, metres
    
    def __post_init__(self) -> None:
        # The checks of the contract, so a wrong voxelization fails as soon as it is built
        assert self.centers.ndim == 2 and self.centers.shape[1] == 3, "centers must be (N, 3)"
        assert self.is_statue.shape == (len(self.centers),), "one is_statue per cube"
        assert self.is_statue.dtype == bool, "is_statue must be bool"
        assert self.is_statue.any() and not self.is_statue.all(), "need statue and marble"
        assert np.isclose(self.centers[:, 2].min(), self.size / 2), "the block must sit on z = 0"
        # Frozen only stops reassigning the fields, so the arrays are made read-only too
        self.centers.flags.writeable = False
        self.is_statue.flags.writeable = False
        
        
def from_grid(statue: np.ndarray, size: float) -> Voxels:
    """Turn a filled grid of the statue into the cubes of the block.
    
    `statue[i, j, k]` is True when the cube is inside the statue; k = 0 is the layer on the table.
    The statue should be centered in x and y.
    """
    # One layer of marble around the statue, except below it (that is the table)
    statue = np.pad(statue, ((1, 1), (1, 1), (0, 1)))
    
    i, j, k = np.indices(statue.shape).reshape(3, -1)
    n_x, n_y, _ = statue.shape
    centers = np.column_stack([(i - (n_x - 1) / 2) * size,
                               (j - (n_y - 1) / 2) * size,
                               (k + 0.5) * size])
    return Voxels(centers, statue.ravel(), size)

# Folder of the 3D models of the statues
MODELS: Path = Path(__file__).resolve().parents[2] / "models"

# Moves the rays of `inside` a few micrometres sideways, so none runs exactly along an edge
# or through a vertex (it would cross two triangles at the same point and count twice)
RAY_SHIFT = np.array([3.1e-6, 1.7e-6])


def place(mesh: trimesh.Trimesh, height: float) -> trimesh.Trimesh:
    """A copy of the mesh, `height` metres tall, standing on z = 0 and centred on the z axis."""
    mesh = mesh.copy()
    mesh.apply_scale(height / mesh.extents[2])
    low, high = mesh.bounds
    mesh.apply_translation([-(low[0] + high[0]) / 2, -(low[1] + high[1]) / 2, -low[2]])
    return mesh


def grid_centers(mesh: trimesh.Trimesh, size: float) -> np.ndarray:
    """Centres of a grid of cubes around a placed mesh, as an (n, n, n_z, 3) array."""
    half = np.ceil(np.abs(mesh.bounds[:, :2]).max() / size)  # cubes from the axis to the side
    xy = np.arange(-half, half + 1) * size  # odd number of cubes, one column on the axis
    z = (np.arange(round(mesh.extents[2] / size)) + 0.5) * size
    return np.stack(np.meshgrid(xy, xy, z, indexing="ij"), axis=-1)



def inside(mesh: trimesh.Trimesh, centers: np.ndarray) -> np.ndarray:
    """True for the cubes whose centre is inside the mesh.

    A vertical ray goes up through each column of cubes; a centre is inside when the ray
    crosses the surface an odd number of times above it.
    """
    statue = np.zeros(centers.shape[:3], dtype=bool)
    for i, j in np.ndindex(statue.shape[:2]):
        x, y = centers[i, j, 0, :2] + RAY_SHIFT
        heights = crossings(mesh.triangles, x, y)
        statue[i, j] = (heights > centers[i, j, :, 2, None]).sum(axis=1) % 2 == 1
    return statue



def crossings(triangles: np.ndarray, x: float, y: float) -> np.ndarray:
    """Heights where the vertical line through (x, y) passes through the triangles."""
    flat = triangles * [1, 1, 0]  # the triangles seen from above
    point = np.tile([x, y, 0.0], (len(triangles), 1))
    with np.errstate(divide="ignore", invalid="ignore"):  # walls seen edge-on give NaN: never hit
        weights = trimesh.triangles.points_to_barycentric(flat, point)
        hit = (weights >= 0).all(axis=1)
        return (weights * triangles[:, :, 2]).sum(axis=1)[hit]

def voxelize(mesh_path: Path, height: float, size: float = CUBE_SIZE) -> Voxels:
    """Voxelize a 3D model of the statue.

    The model is scaled so it is `height` metres tall, standing upright (z up) and centered
    on the z axis. Returns `from_grid(statue, size)`, where `statue` is the filled grid of
    the model with layer 0 at its base.
    """
    mesh = place(trimesh.load(mesh_path, force="mesh"), height) # np.place§
    return from_grid(inside(mesh, grid_centers(mesh, size)), size)


def drop_interior(voxels: Voxels) -> Voxels:
    """Remove the statue cubes that cannot be seen from outside.
    
    A statue cube is inside when its six neighbours are statue too; the table under
    the statue counts as covered. Fewer cubes means less work for Swift at each step.
    """
    #raise NotImplementedError
        # The cubes back on a 3D grid of True (statue) and False
    index = np.round((voxels.centers - voxels.centers.min(axis=0)) / voxels.size).astype(int)
    statue = np.zeros(index.max(axis=0) + 1, dtype=bool)
    statue[tuple(index[voxels.is_statue].T)] = True

    # One more layer all around; the one below is the table, which covers like a statue cube
    s = np.pad(statue, 1)
    s[:, :, 0] = True
    # Statue on all six sides: left and right, front and back, below and above
    covered = (s[:-2, 1:-1, 1:-1] & s[2:, 1:-1, 1:-1]
               & s[1:-1, :-2, 1:-1] & s[1:-1, 2:, 1:-1]
               & s[1:-1, 1:-1, :-2] & s[1:-1, 1:-1, 2:])

    hidden = (statue & covered)[tuple(index.T)]
    return Voxels(voxels.centers[~hidden], voxels.is_statue[~hidden], voxels.size)


# Radius of a chess pawn along its height, both as fractions of the height
PAWN_Z = [0.00, 0.10, 0.16, 0.58, 0.62, 0.66, 0.70, 0.85, 1.00]
PAWN_R = [0.25, 0.25, 0.18, 0.10, 0.17, 0.17, 0.11, 0.16, 0.02]


def proxy_pawn(height: float, size: float = CUBE_SIZE) -> Voxels:
    """A pawn built from a radius profile, to work without a 3D model."""
    n_z = round(height / size)
    z = (np.arange(n_z) + 0.5) / n_z
    radius = np.interp(z, PAWN_Z, PAWN_R) * height

    # Odd number of cubes, so one column sits on the axis
    n_xy = 2 * int(np.ceil(radius.max() / size)) + 1
    x = (np.arange(n_xy) - (n_xy - 1) / 2) * size
    xx, yy = np.meshgrid(x, x, indexing="ij")
    statue = (xx**2 + yy**2)[:, :, None] <= radius[None, None, :] ** 2
    return from_grid(statue, size)

