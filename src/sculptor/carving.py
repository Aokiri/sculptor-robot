"""The carving simulation and the log it saves at each step.

The simulation runs without Swift. The experiments read the log, and Swift can replay it later.
"""

from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import roboticstoolbox as rtb

from sculptor.cell import N_SIDES, Cell
from sculptor.model import Voxels

# Time step of the simulation [s]
DT: float = 0.05


@dataclass(frozen=True)
class Settings:
    """Settings that can change from one run to another."""
    
    closed_loop: bool = True    # False: the arm follows the plan without looking at where the tip is
    posture_task: bool = True   # keep the arm close to its ready pose in the null space
    noise_std: float = 0.0      # std of the noise added to the joints at each step [rad]
    seed: int = 0               # seed of the noise, for reproducibility.
    

@dataclass(frozen=True)
class Log:
    """Every step of a run (T steps), and the step at which each cube was removed (N cubes)."""
    
    t: np.ndarray # (T,) time [s]
    q: np.ndarray # (T, 7) joints of the arm [rad]
    phi: np.ndarray # (T,) angle of the table [rad]
    target: np.ndarray # (T,) index of the cube the chisel is going to, -1 while the table turns
    tip_error: np.ndarray # (T,) distance from the tip to the target [m], NaN when target is -1
    removed_at: np.ndarray # (N,) step at which each cube was removed, -1 if it never was
    voxels: Voxels # the block that was carved
    settings: Settings # how the run was made
    
    def __post_init__(self) -> None:
        # The checks of the contract, so a wrong log fails as soon as it is built
        n_steps = len(self.t)
        assert self.q.shape == (n_steps, 7), "q must be (T, 7)"
        assert self.phi.shape == self.target.shape == self.tip_error.shape == (n_steps,), "one value per step"
        assert self.removed_at.shape == (len(self.voxels.centers),), "one removed_at per cube"
        assert np.all((self.removed_at >= -1) & (self.removed_at < n_steps)), "removed_at must be a step or -1"
        assert np.all((self.target >= -1) & (self.target < len(self.voxels.centers))), "target must be a cube or -1"
        assert np.all(np.isnan(self.tip_error) == (self.target == -1)), "tip_error is NaN only without a target"
        for array in (self.t, self.q, self.phi, self.target, self.tip_error, self.removed_at):
            array.flags.writeable = False
            

def run(voxels: Voxels, cell: Cell, settings: Settings) -> Log:
    """Carve the block and return every step. Written in the next notebooks."""
    raise NotImplementedError


def save_log(log: Log, path: Path) -> None:
    """Save a run in one .npz file, with its block and its settings."""
    np.savez_compressed(path, t=log.t, q=log.q, phi=log.phi, target=log.target,
                        tip_error=log.tip_error, removed_at=log.removed_at,
                        centers=log.voxels.centers, is_statue=log.voxels.is_statue,
                        size=log.voxels.size, **asdict(log.settings))
    

def load_log(path: Path) -> Log:
    """Load a run saved with save_log."""
    # The with block closes the file once the arrays are read into memory
    with np.load(path) as data:
        voxels = Voxels(data["centers"], data["is_statue"], data["size"].item())
        settings = Settings(**{f.name: data[f.name].item() for f in fields(Settings)})
        return Log(data["t"], data["q"], data["phi"], data["target"], data["tip_error"],
                   data["removed_at"], voxels, settings)
        
        
def side_of(centers: np.ndarray, n_sides: int) -> np.ndarray:
    """Side of the block of each cube, from 0 to n_sides - 1, counted from the x axis of the block."""
    # A cube belongs to the side its angle around the axis of the table is closest to
    theta = np.arctan2(centers[:, 1], centers[:, 0])
    return np.round(theta / (2 * np.pi / n_sides)).astype(int) % n_sides


def carving_order(voxels: Voxels, n_sides: int | None = N_SIDES) -> np.ndarray:
    """Indices of the marble cubes in the order they are removed, layer by layer from the top.

    With n_sides, the table shows one side after the other, and in each side the arm starts with the
    cubes closest to the robot. With None, the table turns every cube to the front (notebook 03) and
    each layer goes from the outside in.
    """
    c = voxels.centers
    if n_sides is None:
        order = np.lexsort((-np.hypot(c[:, 0], c[:, 1]), -c[:, 2]))
    else:
        side = side_of(c, n_sides)
        angle = side * 2 * np.pi / n_sides
        # How far each cube sticks out toward the robot once its side faces it
        out = c[:, 0] * np.cos(angle) + c[:, 1] * np.sin(angle)
        order = np.lexsort((-out, side, -c[:, 2]))
    return order[~voxels.is_statue[order]]


def table_angles(voxels: Voxels, order: np.ndarray, cell: Cell, n_sides: int | None = N_SIDES) -> np.ndarray:
    """Angle of the table for each cube of the order."""
    if n_sides is None:
        return np.array([cell.facing_angle(voxels.centers[i]) for i in order])
    return cell.side_angle(side_of(voxels.centers[order], n_sides), n_sides)


def solve_poses(robot: rtb.Robot, cell: Cell, voxels: Voxels, order: np.ndarray, phi: np.ndarray,
                seed: int = 0) -> np.ndarray:
    """Joints (len(order), 7) with the tip on each cube of the order, each IK starting from the previous one.

    seed fixes the random restarts of ikine_LM, so every run gives the same poses.
    """
    q, q0 = [], robot.qr
    for i, angle in zip(order, phi):
        target = cell.tip_target(cell.block_to_world(voxels.centers[i], angle)[0]) * cell.tip.inv()
        solution = robot.ikine_LM(target, q0=q0, seed=seed)
        if not solution.success:
            solution = robot.ikine_LM(target, q0=robot.qr, seed=seed)
        assert solution.success, f"no IK for cube {i}"
        q.append(solution.q)
        q0 = solution.q
    return np.array(q)