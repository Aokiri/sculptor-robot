"""Counts and plots from a saved carving run."""

from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np

from sculptor.carving import Log
from sculptor.controller import TOLERANCE


def check_log(log: Log) -> None:
    """Check the numerical values needed by the analysis."""
    if log.t.ndim != 1 or len(log.t) == 0:
        raise ValueError("The log must contain at least one time sample")
    if not np.all(np.isfinite(log.t)) or not np.all(np.diff(log.t) > 0):
        raise ValueError("Time must be finite and strictly increasing")
    if not np.all(np.isfinite(log.q)) or not np.all(np.isfinite(log.phi)):
        raise ValueError("Joint positions and table angles must be finite")
    if not np.issubdtype(log.removed_at.dtype, np.integer):
        raise ValueError("removed_at must contain integer step indices")
    if not np.issubdtype(log.target.dtype, np.integer):
        raise ValueError("target must contain integer cube indices")
    error = log.tip_error[log.target >= 0]
    if not np.all(np.isfinite(error)) or np.any(error < 0):
        raise ValueError("The tip error must be finite and non-negative at a target")


def arrival_steps(log: Log) -> np.ndarray:
    """Ordered endpoints of actual moves. The setup cube at step zero has no approach."""
    if log.attempt_end_step is None:
        raise ValueError("This log has no recorded attempt endpoints")
    movement_ends = log.attempt_end_step[log.attempt_end_step > 0]
    return np.sort(movement_ends)


def summarize(log: Log) -> dict:
    """Summarize one run. Each fraction uses the original number of cubes of that material."""
    check_log(log)
    removed = log.removed_at >= 0
    statue = log.voxels.is_statue
    n_statue = int(np.count_nonzero(statue))
    n_marble = int(np.count_nonzero(~statue))
    statue_removed = int(np.count_nonzero(removed & statue))
    marble_removed = int(np.count_nonzero(removed & ~statue))
    error = log.tip_error[log.target >= 0]

    row = {
        **asdict(log.settings),
        "statue_cubes": n_statue,
        "statue_removed": statue_removed,
        "statue_damage_fraction": statue_removed / n_statue,
        "marble_cubes": n_marble,
        "marble_removed": marble_removed,
        "marble_remaining": n_marble - marble_removed,
        "marble_removed_fraction": marble_removed / n_marble,
        "log_span_s": float(log.t[-1] - log.t[0]),
        "tip_rmse_m": np.nan,
        "tip_max_error_m": np.nan,
    }
    if len(error):
        mean_squared_error = np.mean(error**2)
        row["tip_rmse_m"] = float(np.sqrt(mean_squared_error))
        row["tip_max_error_m"] = float(np.max(error))

    if log.attempt_end_step is not None:
        ends = arrival_steps(log)
        arrival = log.tip_error[ends]
        row["attempted_moves"] = len(ends)
        row["within_1mm"] = int(np.count_nonzero(arrival <= TOLERANCE))
        row["arrival_rmse_mm"] = np.nan
        if len(arrival):
            mean_squared_arrival_error = np.mean(arrival**2)
            row["arrival_rmse_mm"] = float(1000 * np.sqrt(mean_squared_arrival_error))
        if np.any(log.attempt_end_step == 0):
            row["setup_cube"] = int(log.target[0])
            row["setup_error_mm"] = float(1000 * log.tip_error[0])
    return row


def removal_history(log: Log) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative statue and marble removals, including removals at step zero."""
    check_log(log)
    counts = []
    for mask in (log.voxels.is_statue, ~log.voxels.is_statue):
        steps = log.removed_at[mask]
        # A step of -1 marks a cube that was never removed.
        removal_steps = steps[steps >= 0]
        events = np.bincount(removal_steps, minlength=len(log.t))
        counts.append(np.cumsum(events))
    return counts[0], counts[1]


def plot_run(log: Log, title: str = "Carving log"):
    """Removal counts, tip error, table angle and joints against logged time."""
    statue_removed, marble_removed = removal_history(log)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)

    axes[0, 0].step(log.t, marble_removed, where="post", label="Marble removed")
    axes[0, 0].step(log.t, statue_removed, where="post", label="Statue removed")
    axes[0, 0].set_ylabel("Cubes")
    axes[0, 0].legend()

    # NaN leaves a gap while the table turns because the tip has no target.
    axes[0, 1].plot(log.t, 1000 * log.tip_error, ".-")
    axes[0, 1].set_ylabel("Tip error [mm]")

    axes[1, 0].plot(log.t, log.phi)
    axes[1, 0].set_ylabel("Table angle [rad]")
    axes[1, 1].plot(log.t, log.q)
    axes[1, 1].set_ylabel("Joint position [rad]")
    joint_labels = []
    for i in range(log.q.shape[1]):
        joint_labels.append(f"q{i + 1}")
    axes[1, 1].legend(joint_labels, ncol=4)

    for ax in axes.flat:
        ax.set_xlabel("Time [s]")
        ax.grid(color="0.9")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def _noise_levels(groups: dict) -> list[float]:
    """Return the distinct noise levels in increasing order."""
    levels = set()
    for key in groups:
        levels.add(key[0])
    return sorted(levels)


def comparison_groups(logs: list[Log]) -> dict:
    """Group matched runs without pooling different blocks or seed sets."""
    if not logs:
        raise ValueError("No logs to compare")
    first = logs[0]
    groups = {}
    for log in logs:
        check_log(log)
        if (log.voxels.size != first.voxels.size
                or not np.array_equal(log.voxels.centers, first.voxels.centers)
                or not np.array_equal(log.voxels.is_statue, first.voxels.is_statue)):
            raise ValueError("Compare the same block and statue in every run")
        if len(log.t) < 2 or not np.allclose(np.diff(log.t), np.diff(first.t)[0]):
            raise ValueError("Compare logs with the same constant time step")
        settings = log.settings
        key = (settings.noise_std, settings.closed_loop, settings.posture_task)
        group = groups.setdefault(key, {})
        if settings.seed in group:
            raise ValueError("Duplicate seed in the same experiment condition")
        group[settings.seed] = summarize(log)

    for noise in _noise_levels(groups):
        seed_sets = []
        for closed_loop in (False, True):
            for posture_task in (False, True):
                key = (noise, closed_loop, posture_task)
                if key not in groups:
                    raise ValueError("Each noise level needs all four control/posture conditions")
                seed_sets.append(set(groups[key]))
        for seeds in seed_sets:
            if seeds != seed_sets[0]:
                raise ValueError("Use the same seeds in all four conditions")
    return groups


def plot_comparison(logs: list[Log], title: str = "Carving comparison"):
    """Show every run, the mean and sample standard deviation across seeds."""
    groups = comparison_groups(logs)
    noise_values = _noise_levels(groups)
    conditions = [(False, False), (False, True), (True, False), (True, True)]
    labels = ["Open\nposture off", "Open\nposture on",
              "Closed\nposture off", "Closed\nposture on"]
    fig, axes = plt.subplots(2, len(noise_values), squeeze=False,
                             sharey="row", figsize=(max(7, 5 * len(noise_values)), 7))

    for column, noise in enumerate(noise_values):
        for row, metric in enumerate(("statue_removed", "marble_remaining")):
            ax = axes[row, column]
            for x, (closed_loop, posture_task) in enumerate(conditions):
                group = groups[(noise, closed_loop, posture_task)]
                values = []
                for seed in sorted(group):
                    values.append(group[seed][metric])

                spread = 0.0
                if len(values) > 1:
                    spread = np.std(values, ddof=1)
                ax.plot(np.full(len(values), x), values, "o", alpha=0.5, color="C0")
                ax.errorbar(x, np.mean(values), yerr=spread, fmt="ks", capsize=5)
            ax.set_xticks(range(4), labels)
            ax.set_ylim(bottom=0)
            ax.grid(axis="y", color="0.9")
        n_runs = len(groups[(noise, False, False)])
        axes[0, column].set_title(f"Noise std = {noise:g} rad; {n_runs} runs per condition")

    axes[0, 0].set_ylabel("Statue cubes removed by mistake")
    axes[1, 0].set_ylabel("Marble cubes still present")
    fig.suptitle(f"{title}\nDots: runs; squares: mean; bars: sample standard deviation")
    fig.tight_layout()
    return fig
