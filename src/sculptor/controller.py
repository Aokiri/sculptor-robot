"""Resolved-rate control of the chisel tip (lecture 6), with a posture task in the null space."""

import numpy as np
import roboticstoolbox as rtb
import spatialmath as sm

from sculptor.cell import Cell

# Gain of the task and of the posture, the values of lecture 6 (gain 3 at 0.05 s, k0 = 2)
GAIN: float = 3.0
K0: float = 2.0

# A target counts as reached when the tip is this close to it [m]
TOLERANCE: float = 0.001


def rrmc(robot: rtb.Robot, cell: Cell, q: np.ndarray, target: sm.SE3, q_posture: np.ndarray,
         axis: bool = True, gain: float = GAIN, k0: float = K0) -> np.ndarray:
    """Joint velocities that move the chisel tip toward the target.
    
    The task is the position of the tip and, with axis=True, also the direction of the chisel.
    The posture task pulls the joints toward q_posture without moving the tip (null space).
    """
    T = robot.fkine(q, tool=cell.tip)
    J = robot.jacob0(q, tool=cell.tip)
    error = target.t - T.t
    if axis:
        # The rotation that turns the current axis of the chisel into the target one
        error = np.concatenate([error, np.cross(T.R[:, 2], target.R[:, 2])])
    else:
        J = J[:3]
    J_pinv = np.linalg.pinv(J)
    P = np.eye(robot.n) - J_pinv @ J
    q_star = gain * J_pinv @ error + P @ (k0 * (q_posture - q))
    return q_star

# Gain of the joint-space moves, the 0.1 of every 0.05 s step of the course notebook 3
K_JOINT: float = 2.0

# The arm backs out this far along the chisel axis before it moves elsewhere [m] (notebook 04)
RETRACT: float = 0.09

# Steps after which a move that has not reached its target is an error
MAX_STEPS: int = 600


def servo(robot: rtb.Robot, cell: Cell, q: np.ndarray, target: sm.SE3, q_posture: np.ndarray, dt: float,
          tol: float = TOLERANCE) -> np.ndarray:
    """Joints at every step (steps, 7) while the controller brings the tip to target."""
    path = []
    for _ in range(MAX_STEPS):
        q = q + rrmc(robot, cell, q, target, q_posture) * dt
        path.append(q)
        if np.linalg.norm(target.t - robot.fkine(q, tool=cell.tip).t) < tol:
            return np.array(path)
    raise RuntimeError("the tip did not reach its target")


def joint_move(q: np.ndarray, q_goal: np.ndarray, dt: float, gain: float = K_JOINT, tol: float = 1e-3) -> np.ndarray:
    """Joints at every step (steps, 7) of a proportional move in joint space, until q_goal is within tol."""
    path = []
    while np.linalg.norm(q_goal - q) > tol:
        q = q + gain * (q_goal - q) * dt
        path.append(q)
    return np.array(path)
