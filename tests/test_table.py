import numpy as np
import pytest

from sculptor.cell import wrap_angle
from sculptor.table import KD, KI, KP, PID, Table

DT = 0.05


def turn(pid: PID, duration: float, goal: float = np.pi / 2, disturbance: float = 2.0) -> Table:
    """The table after turning toward goal for duration seconds."""
    table = Table(disturbance=disturbance)
    for _ in range(int(duration / DT)):
        table.step(pid(goal - table.phi), DT)
    return table


@pytest.mark.parametrize("angle, expected", [
    (1.5 * np.pi, -0.5 * np.pi),
    (-1.5 * np.pi, 0.5 * np.pi),
    (0.25 * np.pi, 0.25 * np.pi),
    (4 * np.pi + 0.1, 0.1),
])
def test_wrap_angle_takes_the_short_way(angle, expected):
    assert np.isclose(wrap_angle(angle), expected)


def test_one_step_of_the_table_follows_newton():
    table = Table(inertia=2.0, disturbance=1.0)
    table.step(3.0, DT)
    # (3 + 1) / 2 = 2 rad/s^2 for 0.05 s
    assert np.isclose(table.omega, 0.1)
    assert np.isclose(table.phi, 0.1 * DT)


def test_a_torque_that_cancels_the_disturbance_keeps_the_table_still():
    table = Table(phi=0.3, disturbance=2.0)
    for _ in range(100):
        table.step(-2.0, DT)
    assert table.phi == 0.3 and table.omega == 0.0


def test_the_first_derivative_is_zero_and_the_terms_add_up():
    pid = PID(kp=1.0, ki=2.0, kd=3.0, dt=DT)
    torque = pid(0.5)
    p, i, d = pid.terms
    assert (p, d) == (0.5, 0.0)
    assert np.isclose(i, 2.0 * 0.5 * DT)
    assert np.isclose(torque, p + i + d)


def test_the_pd_stops_where_the_p_term_holds_the_disturbance():
    table = turn(PID(KP, 0.0, KD, DT), duration=30.0)
    # Still, past the goal by d / kp
    assert abs(table.omega) < 1e-6
    assert np.isclose(table.phi - np.pi / 2, 2.0 / KP, atol=1e-6)


def test_the_pid_stops_on_the_goal_with_the_integral_holding_the_disturbance():
    pid = PID(KP, KI, KD, DT)
    table = turn(pid, duration=30.0)
    assert table.stopped_at(np.pi / 2)
    assert np.isclose(pid.terms[1], -2.0, atol=1e-3)
