"""Run with --demo now, or pass a folder of saved .npz logs later."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

from experiments.fake_log import make_fake_log
from sculptor.carving import Settings, load_log, save_log
from sculptor.evaluation import comparison_groups, plot_comparison, plot_run, summarize


def experiment_settings(noise_values=(0.0, 0.001, 0.005), seeds=range(5)) -> list[Settings]:
    """The same seeds for open/closed loop and posture off/on."""
    settings = []
    seeds = tuple(seeds)
    for noise in noise_values:
        for closed_loop in (False, True):
            for posture_task in (False, True):
                for seed in seeds:
                    settings.append(Settings(closed_loop=closed_loop, posture_task=posture_task,
                                             noise_std=noise, seed=seed))
    return settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="?", type=Path, help="Folder containing saved .npz logs")
    parser.add_argument("--demo", action="store_true", help="Generate clearly labelled fake logs")
    parser.add_argument("--output", type=Path, help="Folder for the CSV and figures")
    args = parser.parse_args()
    if args.demo == (args.logs is not None):
        parser.error("Choose either --demo or a folder of logs")

    data_kind = "fake" if args.demo else "loaded"
    output = args.output or Path("results/evaluation")
    title = "FAKE LOGS - pipeline check only" if args.demo else "Loaded carving logs"

    if args.demo:
        settings = experiment_settings(noise_values=(0.001,), seeds=range(3))
        logs = [make_fake_log(setting) for setting in settings]
        names = [f"fake_{i:02d}.npz" for i in range(len(logs))]
    else:
        paths = sorted(args.logs.glob("*.npz"))
        if not paths:
            parser.error(f"No .npz logs found in {args.logs}")
        logs = [load_log(path) for path in paths]
        names = [str(path) for path in paths]

    rows = [{"source": name, "data_kind": data_kind, **summarize(log)}
            for name, log in zip(names, logs)]
    if len(logs) > 1:
        comparison_groups(logs)

    output.mkdir(parents=True, exist_ok=True)
    if args.demo:
        log_folder = output / "logs"
        log_folder.mkdir(exist_ok=True)
        for name, log in zip(names, logs):
            save_log(log, log_folder / name)

    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
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
