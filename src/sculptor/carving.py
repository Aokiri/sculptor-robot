"""The carving simulation and the log it saves at each step.

The simulation runs without Swift. The experiments read the log, and Swift can replay it later.
"""

from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import roboticstoolbox as rtb
import spatialmath as sm

from sculptor.cell import N_SIDES, Cell
from sculptor.collision import Hitboxes, cube_at, touched
from sculptor.controller import joint_move, servo
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

def is_cut(voxels: Voxels, order: np.ndarray, n_sides: int = N_SIDES) -> np.ndarray:
    """For each move of the order (len(order) - 1), whether the arm cuts its way to the next cube.

    It cuts when both cubes are on the same side and share a face or an edge; otherwise it leaves the
    block and comes back.
    """
    centers = voxels.centers[order]
    side = side_of(centers, n_sides)
    same_side = side[1:] == side[:-1]
    # A face neighbour is one cube away and an edge neighbour sqrt(2); a corner one (sqrt(3)) does not count
    distance = np.linalg.norm(centers[1:] - centers[:-1], axis=1)
    return same_side & (distance < 1.5 * voxels.size)


class Carver:
    """Takes the chisel over the marble of a carving order with the controller, and counts what goes wrong.

    For now the table turns at once from one side to the next.
    """

    def __init__(self, robot: rtb.Robot, cell: Cell, voxels: Voxels, order: np.ndarray, phi: np.ndarray,
                 q_ik: np.ndarray, hitboxes: Hitboxes, seed: int = 0) -> None:
        self._robot, self._cell, self._voxels = robot, cell, voxels
        self._order, self._phi, self._q_ik = order, phi, q_ik
        self._hitboxes, self._seed = hitboxes, seed

    def tip_goal(self, k: int, back: float = 0.0) -> sm.SE3:
        """Pose of the tip at the cube in position k of the order, moved back along the chisel by back."""
        p_W = self._cell.block_to_world(self._voxels.centers[self._order[k]], self._phi[k])[0]
        return self._cell.tip_target(p_W) * sm.SE3(0.0, 0.0, -back)

    def run(self, cut: np.ndarray, posture: str = "qr", retract: float | None = None) -> dict:
        """Goes over the whole order and returns the counts.

        cut says, for each move, whether the arm cuts its way to the next cube. Otherwise it repositions:
        with retract=None it jumps straight to the IK pose of the next cube; with a distance it backs out
        along the chisel by retract, moves in joint space to the same distance in front of the next cube,
        and comes in with the controller. The posture is pulled toward qr, or with posture="ik" toward the
        IK pose of the point the tip is going to.
        """
        self._present = np.ones(len(self._voxels.centers), dtype=bool)
        self._present[self._order[0]] = False
        self._counts = {"cuts": int(cut.sum()), "repositions": int(len(cut) - cut.sum()), "steps": 0,
                        "past a joint limit [rad]": 0.0, "largest axis error [deg]": 0.0,
                        "tip in statue": 0, "tip in marble": 0, "hitbox contacts": 0}
        q = self._q_ik[0]
        for k in range(1, len(self._order)):
            if cut[k - 1]:
                q = self._move(q, k, 0.0, posture)
            elif retract is None:
                q = self._q_ik[k]
            else:
                q = self._move(q, k - 1, retract, posture)
                path = joint_move(q, self._ik(k, retract), DT)
                self._inspect(path, k, retract, joint_space=True)
                q = self._move(path[-1], k, 0.0, posture)
            self._present[self._order[k]] = False
        self._counts["time [min]"] = self._counts.pop("steps") * DT / 60
        return self._counts

    def _ik(self, k: int, back: float) -> np.ndarray:
        """IK pose with the tip at tip_goal(k, back), found next to the IK pose of the cube."""
        if back == 0.0:
            return self._q_ik[k]
        target = self.tip_goal(k, back) * self._cell.tip.inv()
        return self._robot.ikine_LM(target, q0=self._q_ik[k], seed=self._seed).q

    def _move(self, q: np.ndarray, k: int, back: float, posture: str) -> np.ndarray:
        """The controller takes the tip to tip_goal(k, back); returns the joints at the end."""
        q_posture = self._ik(k, back) if posture == "ik" else self._robot.qr
        path = servo(self._robot, self._cell, q, self.tip_goal(k, back), q_posture, DT)
        self._inspect(path, k, back)
        return path[-1]

    def _inspect(self, path: np.ndarray, k: int, back: float, joint_space: bool = False) -> None:
        """Counts what goes wrong along a path toward tip_goal(k, back).

        In the moves of the controller the chisel axis is checked. In the moves in joint space, the ones
        that cross the empty space around the block, the hitboxes are checked instead.
        """
        counts = self._counts
        counts["steps"] += len(path)

        # How far a joint goes past its limits
        over = np.maximum(path - self._robot.qlim[1], self._robot.qlim[0] - path).max()
        counts["past a joint limit [rad]"] = max(counts["past a joint limit [rad]"], float(over))

        # The cubes still there, except the target, which is meant to have the tip inside it
        others = self._present.copy()
        others[self._order[k]] = False
        axis_target = self.tip_goal(k, back).R[:, 2]

        for q in path:
            T = self._robot.fkine(q, tool=self._cell.tip)
            if joint_space:
                if self._hitboxes_touch(q, k, others):
                    counts["hitbox contacts"] += 1
            else:
                angle = np.degrees(np.arccos(np.clip(T.R[:, 2] @ axis_target, -1.0, 1.0)))
                counts["largest axis error [deg]"] = max(counts["largest axis error [deg]"], float(angle))

            # Whether the point of the tip is inside one of the other cubes
            c = cube_at(self._cell.world_to_block(T.t, self._phi[k])[0], self._voxels.centers, self._voxels.size)
            if c >= 0 and others[c]:
                counts["tip in statue" if self._voxels.is_statue[c] else "tip in marble"] += 1

    def _hitboxes_touch(self, q: np.ndarray, k: int, cubes: np.ndarray) -> bool:
        """Whether a hitbox touches one of the cubes (a mask over all the cubes), with the table at phi[k]."""
        centers, radii = self._hitboxes.world(q)
        centers = self._cell.world_to_block(centers, self._phi[k])
        return bool(touched(centers, radii, self._voxels.centers[cubes], self._voxels.size).any())
