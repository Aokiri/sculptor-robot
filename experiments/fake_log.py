"""Small hand-written data for checking the analysis before simulation is ready."""

import numpy as np

from sculptor.carving import Log, Settings
from sculptor.model import Voxels


def make_fake_log(settings: Settings | None = None) -> Log:
    """Six cubes, one damaged statue cube and one marble cube left over.

    The arrays are illustrative and are not a robot trajectory. Changing settings
    only changes the labels; no controller behaviour is simulated here.
    """
    if settings is None:
        settings = Settings()
    size = 0.03
    centers = size * np.array([
        [-0.5, 0.0, 0.5], [0.5, 0.0, 0.5],
        [-0.5, -1.0, 0.5], [0.5, -1.0, 0.5],
        [-0.5, 1.0, 0.5], [0.5, 1.0, 0.5],
    ])
    voxels = Voxels(centers, np.array([True, True, False, False, False, False]), size)
    t = np.arange(6) * 0.05
    q = np.tile([0.0, -np.pi / 4, 0.0, -3 * np.pi / 4, 0.0, np.pi / 2, np.pi / 4], (6, 1))
    q[:, 0] = np.linspace(0.0, 0.1, len(t))
    return Log(
        t=t,
        q=q,
        phi=np.array([0.0, 0.0, 0.0, 0.1, 0.2, 0.2]),
        target=np.array([2, 2, 3, -1, 4, 5]),
        tip_error=np.array([0.02, 0.01, 0.003, np.nan, 0.006, 0.004]),
        removed_at=np.array([4, -1, 0, 2, 4, -1]),
        voxels=voxels,
        settings=settings,
    )
