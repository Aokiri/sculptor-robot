"""Run with --demo now, or pass a folder of saved .npz logs later."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

from experiments.fake_log import make_fake_log
from sculptor.carving import Settings, load_log, save_log
from sculptor.evaluation import comparison_groups, plot_comparison, plot_run, summarize


def experiment_settings(noise_values=(0.0, 0.001, 0.005), seeds=range(5)) -> list[Settings]:
    """Build the four control and posture combinations for every noise level and seed."""
    settings = []
    seeds = tuple(seeds)
    for noise in noise_values:
        for closed_loop in (False, True):
            for posture_task in (False, True):
                for seed in seeds:
                    setting = Settings(closed_loop=closed_loop, posture_task=posture_task,
                                       noise_std=noise, seed=seed)
                    settings.append(setting)
    return settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="?", type=Path, help="Folder containing saved .npz logs")
    parser.add_argument("--demo", action="store_true", help="Generate clearly labelled fake logs")
    parser.add_argument("--output", type=Path, help="Folder for the CSV and figures")
    args = parser.parse_args()
    both_inputs = args.demo and args.logs is not None
    no_input = not args.demo and args.logs is None
    if both_inputs or no_input:
        parser.error("Choose either --demo or a folder of logs")

    output = args.output or Path("results/evaluation")
    logs = []
    names = []
    if args.demo:
        data_kind = "fake"
        title = "FAKE LOGS - pipeline check only"
        settings = experiment_settings(noise_values=(0.001,), seeds=range(3))
        for i, setting in enumerate(settings):
            logs.append(make_fake_log(setting))
            names.append(f"fake_{i:02d}.npz")
    else:
        data_kind = "loaded"
        title = "Loaded carving logs"
        paths = sorted(args.logs.glob("*.npz"))
        if not paths:
            parser.error(f"No .npz logs found in {args.logs}")
        for path in paths:
            logs.append(load_log(path))
            names.append(str(path))

    rows = []
    for name, log in zip(names, logs):
        row = {"source": name, "data_kind": data_kind}
        row.update(summarize(log))
        rows.append(row)
    if len(logs) > 1:
        comparison_groups(logs)

    output.mkdir(parents=True, exist_ok=True)
    if args.demo:
        log_folder = output / "logs"
        log_folder.mkdir(exist_ok=True)
        for name, log in zip(names, logs):
            save_log(log, log_folder / name)

    # Older logs can share a folder with logs that record movement endpoints.
    columns = []
    for row in rows:
        for name in row:
            if name not in columns:
                columns.append(name)
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    fig = plot_run(logs[0], title=f"{title}\n{names[0]}")
    fig.savefig(output / "first_run.png", dpi=150)
    plt.close(fig)
    if len(logs) > 1:
        fig = plot_comparison(logs, title=title)
        fig.savefig(output / "comparison.png", dpi=150)
        plt.close(fig)
    print(f"{len(logs)} {data_kind} logs; tables and plots in {output}")


if __name__ == "__main__":
    main()
