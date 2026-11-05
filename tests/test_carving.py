from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from sculptor.carving import Settings, carving_order, load_log, save_log, side_of, table_angles
from sculptor.cell import Cell
from sculptor.model import Voxels


@pytest.mark.parametrize("field, value, message", [
    ("q", np.zeros((6, 6)), "q must be"),
    ("phi", np.zeros(5), "one value per step"),
    ("removed_at", np.zeros(5, dtype=int), "one removed_at"),
    ("removed_at", np.full(6, 6), "removed_at must be"),
    ("removed_at", np.full(6, -2), "removed_at must be"),
    ("target", np.full(6, 6), "target must be"),
    ("target", np.full(6, -2), "target must be"),
    ("tip_error", np.zeros(6), "tip_error is NaN"),
    ("tip_error", np.full(6, np.nan), "tip_error is NaN"),
])
def test_log_checks_shapes_indices_and_missing_errors(sample_log, field, value, message):
    with pytest.raises(AssertionError, match=message):
        replace(sample_log, **{field: value})


def test_log_and_settings_are_read_only(sample_log):
    with pytest.raises(FrozenInstanceError):
        sample_log.settings.seed = 10
    for field in ("t", "q", "phi", "target", "tip_error", "removed_at"):
        with pytest.raises(ValueError):
            getattr(sample_log, field).flat[0] = 0


def test_log_save_load_round_trip(sample_log, tmp_path):
    settings = Settings(closed_loop=False, posture_task=False, noise_std=0.005, seed=7)
    original = replace(sample_log, settings=settings)
    path = tmp_path / "run.npz"
    save_log(original, path)
    loaded = load_log(path)

    assert loaded.settings == settings
    assert loaded.voxels.size == original.voxels.size
    np.testing.assert_array_equal(loaded.voxels.centers, original.voxels.centers)
    np.testing.assert_array_equal(loaded.voxels.is_statue, original.voxels.is_statue)
    for field in ("t", "q", "phi", "target", "tip_error", "removed_at"):
        np.testing.assert_array_equal(getattr(loaded, field), getattr(original, field))
        assert not getattr(loaded, field).flags.writeable


def test_side_numbers_start_at_the_positive_x_axis():
    centers = np.array([[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]])
    np.testing.assert_array_equal(side_of(centers, 4), [0, 1, 2, 3])


@pytest.mark.parametrize("n_sides, expected", [(4, [3, 2, 0, 1]), (None, [2, 3, 0, 1])])
def test_carving_order_is_top_first_and_excludes_statue(n_sides, expected):
    centers = np.array([[0.06, 0, 0.015], [0.03, 0, 0.015],
                        [-0.03, 0, 0.045], [0.03, 0, 0.045], [0, 0, 0.015]])
    voxels = Voxels(centers, np.array([False, False, False, False, True]), 0.03)
    order = carving_order(voxels, n_sides)
    np.testing.assert_array_equal(order, expected)
    cell = Cell()
    for index, angle in zip(order, table_angles(voxels, order, cell, n_sides)):
        world = cell.block_to_world(voxels.centers[index], angle)[0]
        assert world[0] < cell.table_top.t[0]
        assert world[1] == pytest.approx(cell.table_top.t[1], abs=1e-12)
