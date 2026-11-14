from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest
import trimesh


from sculptor.model import MODELS, STATUE_HEIGHT, Voxels, drop_interior, from_grid, proxy_pawn, voxelize



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


def test_voxelize_fills_a_box(tmp_path):
    # A box of 9 x 9 x 12 cm, in millimetres and away from the origin, like a downloaded model
    box = trimesh.creation.box(extents=[90, 90, 120])
    box.apply_translation([50, -20, 7])
    box.export(tmp_path / "box.stl")

    voxels = voxelize(tmp_path / "box.stl", height=0.12, size=0.03)

    # Scaled to 12 cm and centred on the axis: 3 x 3 cubes across and 4 layers
    statue = voxels.centers[voxels.is_statue]
    assert len(statue) == 3 * 3 * 4
    np.testing.assert_allclose(np.unique(statue[:, 0].round(9)), [-0.03, 0.0, 0.03])
    np.testing.assert_allclose(np.unique(statue[:, 1].round(9)), [-0.03, 0.0, 0.03])
    np.testing.assert_allclose(np.unique(statue[:, 2].round(9)), [0.015, 0.045, 0.075, 0.105])


def test_voxelized_pawn_is_close_to_the_proxy_pawn():
    pawn = voxelize(MODELS / "Chess_Pawn.stl", STATUE_HEIGHT)
    proxy = proxy_pawn(STATUE_HEIGHT)
    # Same block of cubes, and about the same number of statue cubes (the profiles differ a bit)
    np.testing.assert_allclose(pawn.centers, proxy.centers)
    n_pawn, n_proxy = np.count_nonzero(pawn.is_statue), np.count_nonzero(proxy.is_statue)
    assert abs(n_pawn - n_proxy) <= 0.1 * n_proxy


def test_drop_interior_removes_only_the_hidden_statue_cubes():
    # A solid 3 x 3 x 3 statue: only the middle column is hidden, its top layer is not
    voxels = from_grid(np.ones((3, 3, 3), dtype=bool), size=0.1)
    dropped = drop_interior(voxels)

    kept = {tuple(center) for center in np.round(dropped.centers, 9)}
    assert len(voxels.centers) - len(dropped.centers) == 2
    assert (0.0, 0.0, 0.05) not in kept  # on the table, under the middle cube
    assert (0.0, 0.0, 0.15) not in kept  # in the middle
    assert (0.0, 0.0, 0.25) in kept  # on top, it can be seen
    assert np.count_nonzero(~dropped.is_statue) == np.count_nonzero(~voxels.is_statue)


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
