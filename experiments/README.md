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

## Notebook 06 · Joint noise on the STL pawn

Open the [executed notebook](../notebooks/06_control_comparison.ipynb) or
download its [HTML copy](../html/06_control_comparison.html). The figures
use two tracked tables. [runs.csv](../results/06_control_comparison/runs.csv)
has one row per trial and [drift.csv](../results/06_control_comparison/drift.csv)
has the four arrival-error series for 0.001 rad, seed 0.

From the repository root, regenerate the local trajectories and both tables with

```bash
uv run python -m experiments.control_comparison --jobs 4
```

The default is one worker. Use `--output PATH` for a separate output directory.
The script rebuilds the 42 cm pawn from `models/Chess_Pawn.stl` at 3 cm
resolution. It checks the noise-free reference against the saved 05
trajectory, compares open and closed loop at zero noise, then verifies 12
pilot runs at 0.001 rad with seeds 0 through 2. It runs the remaining
settings once, for 60 trials in total. All four conditions share an IK seed,
initial state, target sequence, 0.05 s clock, table PID and movement durations.
The seven joint disturbances are drawn only on active arm steps.
A failed trial has its reason in `runs.csv`.

The full logs are under `results/06_control_comparison/logs/`. Each
`removed_at` value gives the first sampled tip contact with that cube.
`attempt_end_step` records the end of each planned cut or approach, even
for a missed target. Both arrays use -1 for an event that never occurred.
The setup cube has endpoint 0. It is excluded from `attempted_moves`,
`within_1mm` and `arrival_rmse_mm`. The saved 05 log can still be loaded
and replayed. It has no endpoint field.

To execute the notebook in a fresh kernel and export the figures to HTML, run

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/06_control_comparison.ipynb
uv run jupyter nbconvert --to html --template classic --no-prompt --output-dir html notebooks/06_control_comparison.ipynb
```
