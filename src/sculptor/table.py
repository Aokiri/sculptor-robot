"""The turning table and the PID on its angle."""

import numpy as np

from sculptor.controller import TOLERANCE

# The table carries the whole marble block at the start: 0.33 x 0.33 x 0.45 m of marble (2700 kg/m^3),
# about 132 kg, so I = m (a^2 + b^2) / 12 = 2.4 kg m^2 for the block, rounded up for the table top
INERTIA: float = 2.6    # [kg m^2]

# A constant torque on the table that the controller does not know about. It stands for what
# the model leaves out (friction of the bearing, a block not centred on the axis); the value is assumed
DISTURBANCE: float = 2.0    # [N m]

# Gains of the PID, tuned by hand in notebook 05
KP: float = 40.0    # [N m / rad]
KI: float = 40.0    # [N m / (rad s)]
KD: float = 20.0    # [N m s / rad]

# The table counts as stopped when the farthest corner of the block (0.212 m from the axis)
# is within the tolerance of the tip, and the table barely moves
PHI_TOLERANCE: float = TOLERANCE / 0.212    # [rad]
OMEGA_TOLERANCE: float = 0.01               # [rad/s]


class Table:
    """The turning table: an inertia turned by the torque of a motor, plus the disturbance."""
    
    def __init__(self, phi: float = 0.0, inertia: float = INERTIA, disturbance: float = DISTURBANCE) -> None:
        self._phi = phi
        self._omega: float = 0.0
        self._inertia = inertia
        self._disturbance = disturbance
        
    @property
    def phi(self) -> float:
        return self._phi
    
    @property
    def omega(self) -> float:
        return self._omega    
    
    def step(self, torque: float, dt: float) -> None:
        """Move the table for dt seconds under a motor torque, with one Euler step."""
        acceleration = (torque + self._disturbance) / self._inertia
        self._omega += acceleration * dt
        self._phi += self._omega * dt
        
    def stopped_at(self, goal: float) -> bool:
        """Whether the table is stopped on the goal angle, within the tolerances."""
        return abs(goal - self._phi) < PHI_TOLERANCE and abs(self._omega) < OMEGA_TOLERANCE
        
        
class PID:
    """A PID on the angle of the table, computed every dt seconds."""
    
    def __init__(self, kp: float = KP, ki: float = KI, kd: float = KD, dt: float = 0.05) -> None:
        self._kp, self._ki, self._kd, self._dt = kp, ki, kd, dt
        self._integral = 0.0
        self._previous: float | None = None
        self._terms = (0.0, 0.0, 0.0)

    @property
    def terms(self) -> tuple[float, float, float]:
        """Torque of the P, I and D terms at the last call [N m], to see what each one does."""
        return self._terms

    def __call__(self, error: float) -> float:
        """Torque for the current error between the goal angle and the measured one."""
        self._integral += error * self._dt
        # On the first step there is no previous error, so the derivative starts at zero
        derivative = 0.0 if self._previous is None else (error - self._previous) / self._dt
        self._previous = error
        self._terms = (self._kp * error, self._ki * self._integral, self._kd * derivative)
        return sum(self._terms)