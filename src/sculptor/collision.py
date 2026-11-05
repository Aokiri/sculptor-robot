"""Hitboxes of the arm and the chisel, and how they meet the cubes of the block.

Every hitbox is a sphere and none of them is drawn in Swift.
A sphere touches a cube when the point of the cube closest to its center is nearer than its radius.
"""

import numpy as np
import roboticstoolbox as rtb
import spatialgeometry as sg
import spatialmath as sm

from sculptor.cell import CHISEL_LENGTH, TIP_WIDTH, Cell
from sculptor.model import Voxels

# Hitboxes are this much bigger than the shapes Swift draws [m],
# so no collision here means no collision on screen either
MARGIN: float = 0.003

# Links that get close to the block; the base and the first links stay far away from it.
NEAR_LINKS: tuple[str, ...] = ("panda_link5", "panda_link6", "panda_link7", "panda_hand")


def capsule_spheres(a: np.ndarray, b: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray]:
    """A row of spheres that covers the capsule from a to b.
    
    They are half a radius apart and slightly bigger than the capsule, so the row has no gaps.
    """
    # Enough spheres for neighbours to be at most r/2 apart, both ends included
    n = int(np.ceil(np.linalg.norm(b - a) / (radius / 2))) + 1
    # Points a + s (b - a) for s evenly spaced from 0 to 1
    centers = a + np.linspace(0.0, 1.0, n)[:, None] * (b - a)
    # Neighbours are d = r/2 apart. The surface point farthest from both centers is halfway
    # between them, r/4 along the axis and r away from it, so the spheres need the radius
    # sqrt(r^2 + (r/4)^2) = r sqrt(1 + 1/16) to leave no gaps.
    return centers, np.full(n, radius * np.sqrt(1 + 1 / 16))


class Hitboxes:
    """Spheres of the last links and the hand of the Panda, and of the chisel.
    
    The arm and the hand use the collision shapes that come with the Panda in the toolbox
    (spheres and cylinders; each cylinder becomes a capsule).
    The chisel is one capsule as wide as the base of its tip, from the fingers to the point.
    """
    
    def __init__(self, robot: rtb.Robot, cell: Cell) -> None:
        self._robot = robot
        self._cell = cell
        self._link_index = {link.name: i for i, link in enumerate(robot.links)}
        # The hand is a link of the gripper, fixed to link 8
        self._hand_offset = sm.SE3(robot.grippers[0].links[0].Ts)
        links = {link.name: link for link in list(robot.links) + list(robot.grippers[0].links)}
        self._links = [(name, *self._link_spheres(links[name])) for name in NEAR_LINKS]
        point = cell.tip.t[2]
        self._chisel = capsule_spheres(np.array([0.0, 0.0, point - CHISEL_LENGTH]), 
                                       np.array([0.0, 0.0, point]), TIP_WIDTH / 2 + MARGIN)
        
    @staticmethod
    def _link_spheres(link: rtb.Link) -> tuple[np.ndarray, np.ndarray]:
        """Spheres of the collision shapes of a link, in the frame of the link."""
        centers, radii = [], []
        for shape in link.collision:
            if isinstance(shape, sg.Sphere):
                centers.append(shape.T[:3, 3])
                radii.append(shape.radius)
            elif isinstance(shape, sg.Cylinder):
                # The axis of a cylinder is the z axis of its pose (third column of the rotation),
                # so its ends are the center plus and minus half the length along it.
                half = shape.T[:3, 2] * shape.length / 2
                c, r = capsule_spheres(shape.T[:3, 3] - half, shape.T[:3, 3] + half, shape.radius)
                centers.extend(c)
                radii.extend(r)
        return np.array(centers), np.array(radii) + MARGIN
    
    @property
    def count(self) -> dict[str, int]:
        """Number of spheres of each part."""
        return {name: len(r) for name, _, r in self._links} | {"chisel": len(self._chisel[1])}
    
    def world(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Centers (S, 3) and radii (S,) of all the spheres in the world, for joints q."""
        T_links = self._robot.fkine_all(q)
        T_hand = T_links[self._link_index["panda_link8"]] * self._hand_offset
        centers, radii = [], []
        # Each center is stored in the frame of its link
        # p^W = T p^L moves it to the world.
        for name, c, r in self._links:
            T = T_hand if name == "panda_hand" else T_links[self._link_index[name]]
            centers.append((T * c.T).T)
            radii.append(r)
        T_ee = self._robot.fkine(q)
        centers.append((T_ee * self._chisel[0].T).T)
        radii.append(self._chisel[1])
        return np.vstack(centers), np.concatenate(radii)
    
    def blocked(self, voxels: Voxels, order: np.ndarray, phi: np.ndarray, q: np.ndarray,
                extra: float = 0.0, arm_scale: float = 1.0) -> list[int]:
        """Cubes of the order whose pose touches a cube that is still there.
        
        phi and q are the table angle and the joints for each cube of the order.
        arm_scale scales the spheres of the arm and the hand, and extra is added to every sphere
        (notebook 03, section 5)
        """
        n_chisel = self.count["chisel"]
        present = np.ones(len(voxels.centers), dtype=bool)
        out = []
        for i, angle, q_i in zip(order, phi, q):
            # The target goes first since the point of the chisel is meant to be inside it.
            present[i] = False
            centers, radii = self.world(q_i)
            radii[:-n_chisel] *= arm_scale
            radii += extra
            if touched(self._cell.world_to_block(centers, angle), radii, voxels.centers[present], voxels.size).any():
                out.append(int(i))
        return out
    
    
def touched(centers: np.ndarray, radii: np.ndarray, cubes: np.ndarray, size: float) -> np.ndarray:
    """For each cube, whether any sphere touches it. Everything in the block frame."""
    # The point of a box closest to a point c is c clipped to the box, one coordinate at a time.
    # This works because the cubes are aligned with the axes of the block frame.
    # Centers (S, 1, 3) against cubes (N, 3) gives every sphere-cube pair at once, as (S, N, 3).
    closest = np.clip(centers[:, None, :], cubes - size / 2, cubes + size / 2)
    distance = np.linalg.norm(closest - centers[:, None, :], axis=2)
    return (distance < radii[:, None]).any(axis=0)


def cube_at(point: np.ndarray, cubes: np.ndarray, size: float) -> int:
    """Index of the cube that contains a point (block frame), or -1 if no cube does."""
    # A point is inside a cube aligned with the axes when it is within half a side
    # of the center along x, y and z.
    inside = np.all(np.abs(cubes - point) <= size / 2, axis=1)
    return int(np.argmax(inside)) if inside.any() else -1