import numpy as np
import pytest

from sculptor.collision import capsule_spheres, cube_at, touched


@pytest.mark.parametrize("point, expected", [
    ([0.0, 0.0, 0.0], 0), ([0.5, 0.0, 0.0], 0),
    ([2.0, 0.0, 0.0], 1), ([1.0, 0.0, 0.0], -1), ([0.0, 0.0, 1.0], -1),
])
def test_cube_at_inside_boundary_and_outside(point, expected):
    cubes = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    assert cube_at(np.array(point), cubes, size=1.0) == expected


def test_touched_handles_inside_nearby_and_distant_cubes():
    cubes = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    centers = np.array([[0.0, 0.0, 0.0], [2.6, 0.0, 0.0]])
    np.testing.assert_array_equal(touched(centers, np.array([0.1, 0.2]), cubes, 1.0),
                                  [True, True, False])


def test_touched_uses_strict_overlap_at_tangency():
    result = touched(np.array([[1.0, 0.0, 0.0]]), np.array([0.5]), np.zeros((1, 3)), 1.0)
    np.testing.assert_array_equal(result, [False])


def test_capsule_spheres_include_both_ends_without_gaps():
    start = np.array([0.0, 0.0, 0.0])
    end = np.array([0.3, 0.0, 0.0])
    centers, radii = capsule_spheres(start, end, radius=0.1)
    np.testing.assert_allclose(centers[0], start)
    np.testing.assert_allclose(centers[-1], end)
    assert np.all(np.linalg.norm(np.diff(centers, axis=0), axis=1) <= 0.05 + 1e-12)
    assert np.all(radii >= 0.1)
