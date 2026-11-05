"""Frames of the table, the block on it and the chisel tip."""

import numpy as np
import spatialmath as sm

from sculptor.model import CUBE_SIZE

# Center of the table top in the world (robot base) frame, away from the hand at qr
TABLE_TOP = sm.SE3(0.75, 0.0, 0.2)

# The chisel is held along the z axis of the end-effector, so its tip is a fixed offset
CHISEL_LENGTH: float = 0.15
# The point sticks out 0.105 m past the fingers; the rest of the chisel is held between them
TIP = sm.SE3(0.0, 0.0, 0.105)

# 20 degrees is the smallest tilt that reaches every cube (notebook 03); 30 leaves a margin.
APPROACH_TILT: float = np.deg2rad(30)

# The table turns between the sides of the block and the arm carves a whole side with the table still.
# 4 sides keep the blocked cubes of one turn per cube; 8 blocks more (notebook 04)
N_SIDES: int = 4

# Only the point of the tip removes cubes, so the point can be up to half a cube away from the
# center of its target before it enters another cube.
# The tip is a cone half a cube wide.
TIP_LENGTH: float = CUBE_SIZE
TIP_WIDTH: float = CUBE_SIZE / 2

# The body is thinner than the base of the tip, like a real point chisel.
CHISEL_RADIUS: float = CUBE_SIZE / 5


class Cell:
    """Pose of the table top in the world and of the chisel tip on the hand."""
    
    def __init__(self, table_top: sm.SE3 = TABLE_TOP, tip: sm.SE3 = TIP) -> None:
        self._table_top = table_top
        self._tip = tip
        
    @property
    def table_top(self) -> sm.SE3:
        return self._table_top
    
    @property
    def tip(self) -> sm.SE3:
        return self._tip
    
    def block_pose(self, phi: float) -> sm.SE3:
        """Pose of the block frame in the world, with the table turned by phi."""
        return self._table_top * sm.SE3.Rz(phi)
    
    def block_to_world(self, p_B: np.ndarray, phi: float) -> np.ndarray:
        """Points from the block frame to the world, as an (N, 3) array."""
        p_B = np.atleast_2d(p_B)
        return (self.block_pose(phi) * p_B.T).T
    
    def world_to_block(self, p_W: np.ndarray, phi: float) -> np.ndarray:
        """Points from the world to the block frame, as an (N, 3) array."""
        p_W = np.atleast_2d(p_W)
        return (self.block_pose(phi).inv() * p_W.T).T
    
    def facing_angle(self, p_B: np.ndarray) -> float:
        """Angle of the table that brings a point of the block in front of the robot.
        
        The point ends up on the line from the axis of the table to the base of the robot.
        """
        x, y = self._table_top.t[:2]
        toward_robot = np.arctan2(-y, -x)
        return float(toward_robot - np.arctan2(p_B[1], p_B[0]))
    
    def side_angle(self, side: np.ndarray, n_sides: int = N_SIDES) -> np.ndarray:
        """Angle of the table that brings a side of the block in front of the robot.
        
        Side k points along the angle 2 pi k / n_sides of the block frame, so this is facing_angle for that direction.
        """
        x, y = self._table_top.t[:2]
        return np.arctan2(-y, -x) - side * 2 * np.pi / n_sides
    
    def tip_target(self, p_W: np.ndarray, tilt: float = APPROACH_TILT) -> sm.SE3:
        """Pose of the chisel tip at a point of the world, pointing at the table and tilted down."""
        x, y = self._table_top.t[:2]
        return sm.SE3(p_W) * sm.SE3.Rz(np.arctan2(y, x)) * sm.SE3.Ry(np.pi / 2 + tilt)