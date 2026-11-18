"""Run the notebook 06 pawn comparison from the repository root."""

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from itertools import repeat
from pathlib import Path

import numpy as np
import roboticstoolbox as rtb

from experiments.evaluate import experiment_settings
from sculptor.carving import DT, CarvingPlan, Log, Settings, load_log, make_carver, save_log
from sculptor.cell import Cell
from sculptor.collision import cube_at
from sculptor.controller import K0, joint_velocity, rrmc
from sculptor.evaluation import arrival_steps, summarize
from sculptor.model import MODELS, STATUE_HEIGHT, voxelize

RESULTS = Path(__file__).resolve().parents[1] / "results"
FIELDS = ("t", "phi", "target", "attempt_end_step", "q", "tip_error", "removed_at")


def condition(setting: Settings) -> str:
    loop, posture = "open", "off"
    if setting.closed_loop:
        loop = "closed"
    if setting.posture_task:
        posture = "on"
    return f"{loop}_{posture}"


def log_path(folder: Path, setting: Settings) -> Path:
    return folder / f"{condition(setting)}_{setting.noise_std:g}_seed{setting.seed}.npz"


def compare_logs(first: Log, second: Log) -> None:
    """Check that motion, contact and attempt outcomes are identical for two noise-free runs."""
    for name in FIELDS:
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name), err_msg=name)


def check_reference(plan: CarvingPlan) -> None:
    """Compare the regenerated posture-on nominal to the shipped 05 trajectory."""
    old = load_log(RESULTS / "05_table_control" / "carving_pawn.npz")
    current = plan.nominal
    if old.voxels.size != current.voxels.size:
        raise ValueError("Voxel size differs from the saved 05 run")
    np.testing.assert_array_equal(current.voxels.centers, old.voxels.centers)
    np.testing.assert_array_equal(current.voxels.is_statue, old.voxels.is_statue)
    for name in ("t", "target"):
        np.testing.assert_array_equal(getattr(current, name), getattr(old, name), err_msg=name)
    for name in ("q", "phi", "tip_error"):
        np.testing.assert_allclose(getattr(current, name), getattr(old, name), rtol=0, atol=1e-7, err_msg=name)
    for label, log in (("Saved 05", old), ("Contact nominal", current)):
        row = summarize(log)
        print(f"{label}: {len(log.t)} samples, {row['statue_removed']} statue removed, "
              f"{row['marble_remaining']} marble remaining", flush=True)


def check_pilot(plan: CarvingPlan, log: Log, panda: rtb.Robot) -> None:
    """Check each reported contact and the matched noise at the start and end of every arm move."""
    voxels = log.voxels
    for cube in np.flatnonzero(log.removed_at >= 0):
        step = log.removed_at[cube]
        tip = panda.fkine(log.q[step], tool=plan.cell.tip).t
        point = plan.cell.world_to_block(tip, log.phi[step])[0]
        if cube_at(point, voxels.centers, voxels.size) != cube:
            raise ValueError(f"Tip outside removed cube {cube} at step {step}")
    settings = log.settings
    commands = plan.nominal_commands(settings.posture_task)
    rng = np.random.default_rng(settings.seed)
    k0 = 0.0
    if settings.posture_task:
        k0 = K0
    limits = panda.qlim
    for move in plan.moves:
        noise = rng.normal(0, settings.noise_std, (move.stop - move.start, panda.n))
        for step in (move.start, move.stop - 1):
            q = log.q[step - 1]
            velocity = commands[step]
            if settings.closed_loop and move.tip_goal is None:
                velocity = joint_velocity(q, move.q_ref)
            elif settings.closed_loop:
                velocity = rrmc(panda, plan.cell, q, move.tip_goal, move.q_ref, k0=k0)
            expected = np.clip(q + velocity * DT + noise[step - move.start], limits[0], limits[1])
            np.testing.assert_allclose(log.q[step], expected, atol=1e-12, rtol=0)


def run_trial(plan: CarvingPlan, settings: Settings, folder: Path, pilot: bool) -> dict:
    """Keep one outcome per setting, including the reason if simulation or validation fails."""
    row = asdict(settings)
    try:
        log = plan.run(settings)
        panda = rtb.models.Panda()
        if np.any(log.q < panda.qlim[0]) or np.any(log.q > panda.qlim[1]):
            raise ValueError("A sampled joint exceeded its limits")
        turning = np.flatnonzero(log.target == -1)
        np.testing.assert_array_equal(log.q[turning], log.q[turning - 1])
        for name in ("t", "target", "phi", "attempt_end_step"):
            np.testing.assert_array_equal(getattr(log, name), getattr(plan.nominal, name))
        if pilot:
            check_pilot(plan, log, panda)
        row.update(summarize(log))
        path = log_path(folder, settings)
        temporary = path.with_suffix(".partial.npz")
        save_log(log, temporary)
        loaded = load_log(temporary)
        compare_logs(loaded, log)
        if loaded.settings != settings:
            raise ValueError("Saved settings differ from the requested condition")
        temporary.replace(path)
        row.update(status="completed", reason="")
    except Exception as error:
        row.update(status="failed", reason=f"{type(error).__name__}: {error}")
    return row


def write_runs(path: Path, grid: list[Settings], outcomes: dict[Settings, dict]) -> None:
    rows, fields = [], []
    for settings in grid:
        if settings in outcomes:
            row = outcomes[settings]
            rows.append(row)
            for key in row:
                if key not in fields:
                    fields.append(key)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_drift(output: Path, folder: Path) -> None:
    """One row per movement endpoint for the four prespecified seed-0 runs at 0.001 rad."""
    columns = {}
    for settings in experiment_settings(noise_values=(0.001,), seeds=(0,)):
        log = load_log(log_path(folder, settings))
        ends = arrival_steps(log)
        if not columns:
            columns["move"] = np.arange(1, len(ends) + 1)
            columns["target_cube"] = log.target[ends]
        np.testing.assert_array_equal(log.target[ends], columns["target_cube"])
        columns[condition(settings) + "_mm"] = np.round(1000 * log.tip_error[ends], 6)
    with (output / "drift.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(columns)
        writer.writerows(zip(*columns.values()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RESULTS / "06_control_comparison")
    parser.add_argument("--jobs", type=int, default=1, help="Number of independent simulations in parallel")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    folder = args.output / "logs"
    folder.mkdir(parents=True, exist_ok=True)
    print("Preparing the STL pawn and the nominal 05 schedule", flush=True)
    voxels = voxelize(MODELS / "Chess_Pawn.stl", STATUE_HEIGHT)
    plan = make_carver(rtb.models.Panda(), Cell(), voxels, seed=0).comparison_plan()
    check_reference(plan)
    for posture in (False, True):
        plan.nominal_commands(posture)

    grid = experiment_settings()
    zero = experiment_settings(noise_values=(0.0,), seeds=(0,))
    pilot = experiment_settings(noise_values=(0.001,), seeds=range(3))
    remaining = []
    for setting in grid:
        if setting not in zero and setting not in pilot:
            remaining.append(setting)
    outcomes = {}
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for stage, settings in (("zero", zero), ("pilot", pilot), ("grid", remaining)):
            rows = pool.map(run_trial, repeat(plan), settings, repeat(folder), repeat(stage == "pilot"))
            for setting, row in zip(settings, rows):
                outcomes[setting] = row
                write_runs(args.output / "runs.csv", grid, outcomes)
                print(f"{len(outcomes):2d}/60 {stage} {log_path(folder, setting).stem} {row['status']}", flush=True)
            for setting in settings:
                if outcomes[setting]["status"] == "failed":
                    raise RuntimeError(f"See runs.csv for the failed {stage} trials")
            if stage == "zero":
                for posture in (False, True):
                    opened = load_log(log_path(folder, Settings(False, posture)))
                    closed = load_log(log_path(folder, Settings(True, posture)))
                    compare_logs(opened, closed)
                    if posture:
                        compare_logs(closed, plan.nominal)
    for setting in experiment_settings(noise_values=(0.0,)):
        reference = Settings(setting.closed_loop, setting.posture_task)
        compare_logs(load_log(log_path(folder, setting)), load_log(log_path(folder, reference)))
    write_drift(args.output, folder)
    print("Verified all 60 trials, including the 12 pilot runs.", flush=True)


if __name__ == "__main__":
    main()
