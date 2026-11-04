"""The statue as a set of cubes in the block frame

The block frame has its origin at the center of the bottom face of the block, with z up.
It is the frame the table carries (see cell.py), so a cube never changes its coordinates 
while the table turns.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Chosen in notebook 00 from the step times of Swift
CUBE_SIZE: float = 0.03


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
        # The checks of the contract, so a wrong voxelization fails here and not in Swift
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


def voxelize(mesh_path: Path, height: float, size: float = CUBE_SIZE) -> Voxels:
    """Voxelize a 3D model of the statue.

    The model is scaled so it is `height` metres tall, standing upright (z up) and centered
    on the z axis. Returns `from_grid(statue, size)`, where `statue` is the filled grid of
    the model with layer 0 at its base.
    """
    raise NotImplementedError


def drop_interior(voxels: Voxels) -> Voxels:
    """Remove the statue cubes that cannot be seen from outside.
    
    A statue cube is inside when its six neighbours are statue too; the table under
    the statue counts as covered. Fewer cubes means less work for Swift at each step.
    """
    raise NotImplementedError


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

