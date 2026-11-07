from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from sculptor.model import Voxels, from_grid, proxy_pawn


def test_from_grid_adds_marble_and_keeps_coordinates():
    grid = np.zeros((2, 3, 2), dtype=bool)
    grid[0, 2, 0] = True
    grid[1, 0, 1] = True
    original = grid.copy()
    voxels = from_grid(grid, size=0.1)

    assert voxels.centers.shape == (60, 3)
    assert np.count_nonzero(voxels.is_statue) == 2
    np.testing.assert_allclose(voxels.centers[voxels.is_statue],
                               [[-0.05, 0.1, 0.05], [0.05, -0.1, 0.15]])
    np.testing.assert_allclose(voxels.centers.min(axis=0), [-0.15, -0.2, 0.05])
    np.testing.assert_allclose(voxels.centers.max(axis=0), [0.15, 0.2, 0.25])
    assert not voxels.is_statue[voxels.centers[:, 2] > 0.2].any()
    np.testing.assert_array_equal(grid, original)


def test_voxels_are_read_only(sample_log):
    voxels = sample_log.voxels
    with pytest.raises(FrozenInstanceError):
        voxels.size = 1.0
    with pytest.raises(ValueError):
        voxels.centers[0, 0] = 1.0
    with pytest.raises(ValueError):
        voxels.is_statue[0] = False


@pytest.mark.parametrize("field, value, message", [
    ("centers", np.zeros((6, 2)), "centers must be"),
    ("is_statue", np.array([True, False]), "one is_statue"),
    ("is_statue", np.array([1, 1, 0, 0, 0, 0]), "must be bool"),
    ("is_statue", np.ones(6, dtype=bool), "need statue and marble"),
    ("is_statue", np.zeros(6, dtype=bool), "need statue and marble"),
])
def test_voxels_reject_invalid_arrays(sample_log, field, value, message):
    with pytest.raises(AssertionError, match=message):
        replace(sample_log.voxels, **{field: value})


def test_voxels_must_rest_on_the_table(sample_log):
    centers = sample_log.voxels.centers.copy()
    centers[:, 2] += 0.1
    with pytest.raises(AssertionError, match="sit on z = 0"):
        Voxels(centers, sample_log.voxels.is_statue, sample_log.voxels.size)


@pytest.mark.parametrize("height, size", [(0.12, 0.03), (0.42, 0.03), (0.42, 0.06)])
def test_proxy_pawn_is_symmetric_and_has_a_marble_margin(height, size):
    voxels = proxy_pawn(height, size)
    statue = voxels.centers[voxels.is_statue]
    for axis in (0, 1):
        reflected = statue.copy()
        reflected[:, axis] *= -1
        points = {tuple(point) for point in np.round(statue, 10)}
        assert {tuple(point) for point in np.round(reflected, 10)} == points
    assert statue[:, 2].min() == pytest.approx(size / 2)
    assert voxels.centers[:, 2].max() - statue[:, 2].max() == pytest.approx(size)


def test_scaling_pawn_and_cubes_preserves_the_grid():
    small = proxy_pawn(0.12, 0.03)
    large = proxy_pawn(0.24, 0.06)
    np.testing.assert_array_equal(small.is_statue, large.is_statue)
    np.testing.assert_allclose(2 * small.centers, large.centers)
