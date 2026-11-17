import numpy as np
import roboticstoolbox as rtb

import sculptor.scene as scene
from sculptor.cell import Cell


class FakeSwift:
    """Keeps what replay asks of Swift, without a browser."""

    def __init__(self) -> None:
        self.removed: list[int] = []
        self.steps = 0

    def remove(self, cube_id: int) -> None:
        self.removed.append(cube_id)

    def step(self, dt: float) -> None:
        self.steps += 1


def test_replay_removes_each_cube_once_and_draws_one_step_in_every(sample_log, monkeypatch):
    # Nothing to place without Swift
    monkeypatch.setattr(scene, "place_block", lambda *args, **kwargs: None)
    monkeypatch.setattr(scene, "place_chisel", lambda *args, **kwargs: None)
    env = FakeSwift()
    n_cubes = len(sample_log.voxels.centers)
    ids = [100 + i for i in range(n_cubes)]
    scene.replay(env, sample_log, [None] * n_cubes, ids, rtb.models.Panda(), None, Cell(), every=2)

    removed = np.flatnonzero(sample_log.removed_at >= 0)
    assert sorted(env.removed) == sorted(100 + i for i in removed)
    # Steps 0, 2, 4, ... are drawn, and the last one too
    n_steps = len(sample_log.t)
    assert env.steps == len(set(range(0, n_steps, 2)) | {n_steps - 1})
