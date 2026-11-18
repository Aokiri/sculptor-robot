from pathlib import Path

import numpy as np
import pytest
import roboticstoolbox as rtb
import spatialmath as sm

import sculptor.controller as controller
from sculptor.carving import (ArmMove, Carver, CarvingPlan, Log, Settings, carving_order, is_cut,
                              load_log, save_log, solve_poses, table_angles)
from sculptor.cell import Cell, wrap_angle
from sculptor.collision import Hitboxes, cube_at
from sculptor.controller import RETRACT, TOLERANCE, servo
from sculptor.evaluation import summarize
from sculptor.model import STATUE_HEIGHT, Voxels, proxy_pawn
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


def test_tip_accuracy_at_arrival_and_first_entry_at_removal(short_run: tuple[Carver, dict, Log]) -> None:
    carver, _, log = short_run
    ends = log.attempt_end_step[carver.order]
    assert ends[0] == 0
    assert np.all(log.tip_error[ends] <= TOLERANCE)
    assert np.any(log.removed_at[carver.order] < ends)

    panda, cell = rtb.models.Panda(), Cell()
    first = np.full(len(log.voxels.centers), -1)
    for step, q in enumerate(log.q):
        point = cell.world_to_block(panda.fkine(q, tool=cell.tip).t, log.phi[step])[0]
        cube = cube_at(point, log.voxels.centers, log.voxels.size)
        if cube >= 0 and first[cube] == -1:
            first[cube] = step
    np.testing.assert_array_equal(log.removed_at, first)


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


@pytest.fixture(scope="module")
def short_plan(short_run: tuple[Carver, dict, Log]) -> CarvingPlan:
    carver, _, _ = short_run
    return carver.comparison_plan()


@pytest.mark.parametrize("posture", [False, True])
def test_open_and_closed_match_without_noise(short_plan: CarvingPlan, posture: bool) -> None:
    opened = short_plan.run(Settings(False, posture))
    closed = short_plan.run(Settings(True, posture))
    for name in ("q", "phi", "t", "target", "removed_at", "tip_error", "attempt_end_step"):
        np.testing.assert_array_equal(getattr(opened, name), getattr(closed, name))
    if posture:
        np.testing.assert_allclose(closed.q, short_plan.nominal.q, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(closed.removed_at, short_plan.nominal.removed_at)


@pytest.mark.parametrize("closed,posture", [(False, False), (False, True), (True, False), (True, True)])
def test_matched_noise_on_active_steps_only(short_plan: CarvingPlan, closed: bool, posture: bool) -> None:
    panda, cell = rtb.models.Panda(), Cell()
    setting = Settings(closed, posture, noise_std=0.005, seed=3)
    log = short_plan.run(setting)
    np.testing.assert_array_equal(log.q, short_plan.run(setting).q)
    assert np.all((log.q >= panda.qlim[0]) & (log.q <= panda.qlim[1]))
    turning = np.flatnonzero(log.target == -1)
    assert len(turning) > 0
    np.testing.assert_array_equal(log.q[turning], log.q[turning - 1])
    np.testing.assert_array_equal(log.attempt_end_step, short_plan.nominal.attempt_end_step)
    commands = short_plan.nominal_commands(posture)
    rng = np.random.default_rng(setting.seed)
    k0 = 0.0
    if posture:
        k0 = controller.K0
    for move in short_plan.moves:
        for step in range(move.start, move.stop):
            q = log.q[step - 1]
            velocity = commands[step]
            if closed and move.tip_goal is None:
                velocity = controller.joint_velocity(q, move.q_ref)
            elif closed:
                velocity = controller.rrmc(panda, cell, q, move.tip_goal, move.q_ref, k0=k0)
            noise = rng.normal(0, setting.noise_std, 7)
            expected = np.clip(q + DT * velocity + noise, panda.qlim[0], panda.qlim[1])
            np.testing.assert_allclose(log.q[step], expected, rtol=0, atol=1e-13)


def test_table_can_remove_statue_and_other_marble_while_a_target_stays_unreached(tmp_path: Path) -> None:
    panda = rtb.models.Panda()
    q = panda.qr
    tip = panda.fkine(q, tool=Cell().tip).t
    centers = np.array([[0.03, 0, 0.015], [0, -0.03, 0.015], [-0.03, 0, 0.015], [0.09, 0, 0.015]])
    voxels = Voxels(centers, np.array([False, True, False, False]), 0.03)
    cell = Cell(table_top=sm.SE3(tip - centers[0]))
    # The stationary tip sees two new cubes during the turn, one of them on two consecutive samples.
    phi = np.array([0, 0, np.pi / 2, np.pi / 2, np.pi])
    target = np.array([0, 3, -1, -1, -1])
    nominal = Log(np.arange(5) * DT, np.tile(q, (5, 1)), phi, target,
                  np.array([0, 0.06, np.nan, np.nan, np.nan]), np.full(4, -1),
                  voxels, Settings(), np.array([0, -1, -1, 1]))
    plan = CarvingPlan(nominal, [ArmMove(1, 2, q, None)], cell)
    log = plan.run(Settings())
    np.testing.assert_array_equal(log.removed_at, [0, 2, 4, -1])
    assert log.attempt_end_step[3] == 1
    path = tmp_path / "contacts.npz"
    save_log(log, path)
    loaded = load_log(path)
    np.testing.assert_array_equal(loaded.attempt_end_step, log.attempt_end_step)
    assert not loaded.attempt_end_step.flags.writeable
    row = summarize(loaded)
    assert row["attempted_moves"] == 1 and row["within_1mm"] == 0
    assert row["arrival_rmse_mm"] == pytest.approx(60.0)
    assert row["statue_removed"] == 1 and row["marble_remaining"] == 1


def test_open_loop_keeps_the_command_before_the_joint_stop(short_plan: CarvingPlan) -> None:
    panda = rtb.models.Panda()
    q = panda.qr.copy()
    q[-1] = panda.qlim[1, -1] - 0.001
    goal = q.copy()
    goal[-1] += 0.2
    base = short_plan.nominal
    target = np.full(3, base.target[0])
    log = Log(np.arange(3) * DT, np.tile(q, (3, 1)), np.full(3, base.phi[0]), target,
              np.zeros(3), np.full(len(base.voxels.centers), -1), base.voxels, Settings())
    plan = CarvingPlan(log, [ArmMove(1, 3, goal, None)], short_plan.cell)
    command = plan.nominal_commands(True)
    assert command[1, -1] == pytest.approx(0.4)
    assert command[2, -1] > 0
    measured = plan.run(Settings(closed_loop=False))
    assert measured.q[1, -1] == measured.q[2, -1] == panda.qlim[1, -1]


def test_unlimited_joint_transfer_keeps_its_original_stopping_rule() -> None:
    goal = np.ones(7)
    path = controller.joint_move(np.zeros(7), goal, DT, gain=0.1)
    assert len(path) > controller.MAX_STEPS
    assert np.linalg.norm(goal - path[-1]) <= 0.001
