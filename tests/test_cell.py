import numpy as np
import pytest
import spatialmath as sm

from sculptor.cell import Cell


def test_quarter_turn_has_known_world_coordinates():
    cell = Cell(table_top=sm.SE3(1.0, 2.0, 3.0))
    point = cell.block_to_world(np.array([1.0, 0.0, 0.5]), np.pi / 2)
    assert point.shape == (1, 3)
    np.testing.assert_allclose(point, [[1.0, 3.0, 3.5]], atol=1e-12)


@pytest.mark.parametrize("phi", [0.0, np.pi / 2, -np.pi, 2 * np.pi])
def test_frame_round_trip_with_a_rotated_table_base(phi):
    cell = Cell(table_top=sm.SE3(0.7, -0.2, 0.3) * sm.SE3.Rx(0.2) * sm.SE3.Rz(0.4))
    points = np.array([[0.1, -0.2, 0.3], [-0.2, 0.0, 0.5]])
    world = cell.block_to_world(points, phi)
    np.testing.assert_allclose(cell.world_to_block(world, phi), points, atol=1e-12)


@pytest.mark.parametrize("point", [[0.1, 0.0, 0.1], [0.0, 0.1, 0.2], [-0.1, -0.1, 0.3]])
def test_facing_angle_brings_the_point_towards_the_robot(point):
    cell = Cell(table_top=sm.SE3(0.7, 0.2, 0.3))
    point = np.array(point)
    world = cell.block_to_world(point, cell.facing_angle(point))[0]
    direction = -cell.table_top.t[:2]
    direction /= np.linalg.norm(direction)
    expected = cell.table_top.t[:2] + np.linalg.norm(point[:2]) * direction
    np.testing.assert_allclose(world[:2], expected, atol=1e-12)


def test_side_angles_face_all_four_sides_towards_the_robot():
    cell = Cell()
    directions = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                           [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    angles = cell.side_angle(np.arange(4), n_sides=4)
    for point, phi in zip(directions, angles):
        world = cell.block_to_world(point, phi)[0]
        np.testing.assert_allclose(world - cell.table_top.t, [-1.0, 0.0, 0.0], atol=1e-12)


@pytest.mark.parametrize("tilt", [0.0, np.pi / 6])
def test_tip_target_points_towards_table_and_down(tilt):
    cell = Cell(table_top=sm.SE3(0.6, 0.8, 0.2))
    point = np.array([0.5, 0.1, 0.4])
    pose = cell.tip_target(point, tilt)
    np.testing.assert_allclose(pose.t, point)
    np.testing.assert_allclose(pose.R[:, 2],
                               [0.6 * np.cos(tilt), 0.8 * np.cos(tilt), -np.sin(tilt)],
                               atol=1e-12)
