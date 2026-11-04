"""Frames of the cell: the table, the block on it and the chisel tip."""

import numpy as np
import spatialmath as sm

# Center of the table top in the world (robot base) frame, away from the hand at qr
TABLE_TOP = sm.SE3(0.75, 0.0, 0.2)

# The chisel is held along the z axis of the end-effector, so its tip is a fixed offset
CHISEL_LENGTH: float = 0.15
TIP = sm.SE3(0.0, 0.0, 0.105) # This is 0.03 (chisel's cylinder) + 0.075 (0.15 / 2, from chisel length)


class Cell:
    """Where the table top is in the world and where the chisel tip is on the hand."""
    
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