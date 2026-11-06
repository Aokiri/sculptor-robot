# Tests and evaluation

Work for [issue #2](https://github.com/Aokiri/sculptor-robot/issues/2).

From the repository root, follow the main README, then run:

```bash
uv sync
uv run pytest
uv run python -m experiments.evaluate --demo
```

The demo saves fake logs, a CSV and two plots in `results/evaluation/`.
The example has two statue cubes and four marble cubes: one statue cube is
removed, three marble cubes are removed and one marble cube remains.
The same hand-written data are used for every setting. They check the
analysis, not the behaviour of a controller.

To analyse saved simulation logs:

```bash
uv run python -m experiments.evaluate results/carving
```

One log gives a diagnostic plot. Multiple logs must contain all four
open/closed-loop and posture off/on conditions, with matching seeds at each
noise level. The block and constant time step must also match. Use separate
folders for different geometries, cube sizes, gains or initial states.
Input logs are read unchanged. Outputs go to `results/evaluation/`; use
`--output` to keep results from different analyses in separate folders.

## Measurements

| Field | Meaning |
| --- | --- |
| `statue_removed` | Statue cubes removed by mistake, including removals at step zero |
| `statue_damage_fraction` | Removed statue cubes / original statue cubes in the log |
| `marble_remaining` | Marble cubes still present |
| `marble_removed_fraction` | Removed marble cubes / original marble cubes |
| `tip_rmse_m` | RMSE over samples with a target, in metres |
| `tip_max_error_m` | Largest error over samples with a target |
| `log_span_s` | Last timestamp minus first timestamp |

If interior statue cubes were dropped, the damage fraction refers to the
reduced voxel set. Missing target errors remain NaN, not zero. The RMSE is
a sample statistic, not an arrival error; the time span does not prove completion.

The comparison shows damage together with remaining marble: removing nothing
must not look like a good result. Dots are runs, squares are means and bars are
sample standard deviations across seeds, not confidence intervals.

## Real runs

`experiment_settings()` prepares 60 settings: four control/posture conditions,
noise standard deviations 0, 0.001 and 0.005 rad, and seeds 0 to 4. Keep the initial state,
carving order, time step, limits and stopping rule fixed, and reset between runs.
Matching seeds only imply matching disturbances if random samples are consumed
consistently.

The simulator is still a stub. PID gains, a table reference and a termination
reason are not yet in the contract. PID response, success rates and unreachable
cubes will be evaluated when those data are available.

The tests cover the implemented geometry and log functions, the hand-counted
metrics, matched comparisons, plots and the demo. GitHub Actions runs them
with the locked project environment. The analysis uses NumPy and Matplotlib;
pytest is the test tool requested in the issue.
