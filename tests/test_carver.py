import numpy as np
import pytest
import roboticstoolbox as rtb

import sculptor.controller as controller
from sculptor.carving import Carver, Settings, carving_order, is_cut, solve_poses, table_angles
from sculptor.cell import Cell, wrap_angle
from sculptor.collision import Hitboxes
from sculptor.controller import RETRACT, TOLERANCE, servo
from sculptor.model import STATUE_HEIGHT, proxy_pawn
from sculptor.table import PHI_TOLERANCE

DT = 0.05


@pytest.fixture(scope="module")
def short_run():
    """A recorded run over the first cubes of the proxy pawn, with one change of side and the PID table."""
    panda, cell = rtb.models.Panda(), Cell()
    voxels = proxy_pawn(STATUE_HEIGHT)
    order = carving_order(voxels)
    phi = table_angles(voxels, order, cell)
    # The first side of the top layer and a few cubes of the next one
    n = int(np.flatnonzero(phi != phi[0])[0]) + 3
    order, phi = order[:n], phi[:n]
    q_ik = solve_poses(panda, cell, voxels, order, phi)
    carver = Carver(panda, cell, voxels, order, phi, q_ik, Hitboxes(panda, cell))
    counts = carver.run(is_cut(voxels, order), posture="ik", retract=RETRACT, record=True, table_pid=True,
                        stop_at_limits=True)
    return carver, counts, carver.log(Settings())


def test_the_log_removes_every_cube_of_the_order_once(short_run):
    carver, _, log = short_run
    removed = np.flatnonzero(log.removed_at >= 0)
    assert sorted(removed) == sorted(carver.order)


def test_the_tip_is_on_its_cube_when_the_cube_is_removed(short_run):
    carver, _, log = short_run
    steps = log.removed_at[carver.order]
    assert np.all(log.tip_error[steps] < TOLERANCE + 1e-9)


def test_the_target_is_missing_only_while_the_table_turns(short_run):
    _, counts, log = short_run
    turning = log.target == -1
    assert np.isclose(turning.sum() * DT, counts["table turning [s]"])
    # The table only moves on those steps, except right after a turn, when the angle goes from where the
    # PID stopped the table to the planned angle of the side; that jump is within the tolerance
    moving = np.flatnonzero(np.diff(log.phi) != 0) + 1
    for step in moving:
        if not turning[step]:
            assert turning[step - 1]
            assert abs(wrap_angle(log.phi[step] - log.phi[step - 1])) < PHI_TOLERANCE


def test_the_counts_and_the_log_have_the_same_steps(short_run):
    _, counts, log = short_run
    assert np.isclose(len(log.t) - 1, counts["time [min]"] * 60 / DT)


def test_stop_at_limits_keeps_a_joint_on_its_limit(monkeypatch):
    panda, cell = rtb.models.Panda(), Cell()
    q = panda.qr.copy()
    q[6] = panda.qlim[1, 6] - 0.001
    # A controller that only spins joint 7 upward; joint 7 spins the chisel, so the tip stays on the target
    monkeypatch.setattr(controller, "rrmc", lambda *args, **kwargs: np.eye(7)[6])
    target = panda.fkine(q, tool=cell.tip)
    free = servo(panda, cell, q, target, q, DT)
    stopped = servo(panda, cell, q, target, q, DT, stop_at_limits=True)
    assert free[-1, 6] > panda.qlim[1, 6]
    assert stopped[-1, 6] == panda.qlim[1, 6]
