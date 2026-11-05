import csv
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import pytest

from experiments.evaluate import experiment_settings, main
from experiments.fake_log import make_fake_log
from sculptor.carving import load_log, save_log
from sculptor.evaluation import comparison_groups, plot_comparison, plot_run, removal_history, summarize


def test_metrics_match_the_hand_counted_example(sample_log):
    row = summarize(sample_log)
    assert row["statue_cubes"] == 2
    assert row["statue_removed"] == 1
    assert row["statue_damage_fraction"] == 0.5
    assert row["marble_cubes"] == 4
    assert row["marble_removed"] == 3
    assert row["marble_remaining"] == 1
    assert row["marble_removed_fraction"] == 0.75
    assert row["log_span_s"] == pytest.approx(0.25)
    assert row["tip_rmse_m"] == pytest.approx(np.sqrt(0.000561 / 5))
    assert row["tip_max_error_m"] == 0.02


def test_history_counts_step_zero_and_simultaneous_removals(sample_log):
    statue, marble = removal_history(sample_log)
    np.testing.assert_array_equal(statue, [0, 0, 0, 0, 1, 1])
    np.testing.assert_array_equal(marble, [1, 1, 2, 2, 3, 3])


@pytest.mark.parametrize("step, damage, remaining", [(-1, 0.0, 4), (0, 1.0, 0), (5, 1.0, 0)])
def test_no_removals_and_complete_removal(sample_log, step, damage, remaining):
    log = replace(sample_log, removed_at=np.full(6, step, dtype=int))
    row = summarize(log)
    assert row["statue_damage_fraction"] == damage
    assert row["marble_remaining"] == remaining
    statue, marble = removal_history(log)
    assert statue[-1] == row["statue_removed"]
    assert marble[-1] == row["marble_removed"]


def test_no_targets_has_missing_error_not_zero_error(sample_log):
    log = replace(sample_log, target=np.full(6, -1), tip_error=np.full(6, np.nan))
    row = summarize(log)
    assert np.isnan(row["tip_rmse_m"])
    assert np.isnan(row["tip_max_error_m"])


@pytest.mark.parametrize("time", [np.zeros(6), np.arange(6)[::-1],
                                  np.array([0, 1, 2, 3, 4, np.inf])])
def test_metrics_reject_invalid_times(sample_log, time):
    with pytest.raises(ValueError, match="Time must be"):
        summarize(replace(sample_log, t=time))


@pytest.mark.parametrize("field", ["removed_at", "target"])
def test_metrics_reject_fractional_index_types(sample_log, field):
    log = replace(sample_log, **{field: getattr(sample_log, field).astype(float)})
    with pytest.raises(ValueError, match="integer"):
        summarize(log)


@pytest.mark.parametrize("value", [-0.1, np.inf])
def test_metrics_reject_invalid_distance(sample_log, value):
    error = sample_log.tip_error.copy()
    error[0] = value
    with pytest.raises(ValueError, match="non-negative"):
        summarize(replace(sample_log, tip_error=error))


def test_empty_log_is_not_a_successful_run(sample_log):
    log = replace(sample_log, t=np.zeros(0), q=np.zeros((0, 7)), phi=np.zeros(0),
                  target=np.zeros(0, dtype=int), tip_error=np.zeros(0), removed_at=np.full(6, -1))
    with pytest.raises(ValueError, match="at least one"):
        summarize(log)


def test_experiment_settings_cover_all_conditions_with_matching_seeds():
    settings = experiment_settings(noise_values=(0.0, 0.001), seeds=(2, 5))
    assert len(settings) == 16
    groups = comparison_groups([make_fake_log(setting) for setting in settings])
    assert len(groups) == 8
    assert all(set(group) == {2, 5} for group in groups.values())


@pytest.mark.parametrize("problem", ["missing_condition", "missing_seed", "duplicate", "block", "dt"])
def test_comparison_rejects_unmatched_runs(problem):
    logs = [make_fake_log(setting) for setting in experiment_settings((0.001,), (0, 1))]
    if problem == "missing_condition":
        logs = logs[2:]
    elif problem == "missing_seed":
        logs = logs[1:]
    elif problem == "duplicate":
        logs.append(logs[0])
    elif problem == "block":
        centers = logs[-1].voxels.centers.copy()
        centers[:, 0] += 0.01
        logs[-1] = replace(logs[-1], voxels=replace(logs[-1].voxels, centers=centers))
    elif problem == "dt":
        logs[-1] = replace(logs[-1], t=logs[-1].t * 2)
    with pytest.raises(ValueError):
        comparison_groups(logs)


def test_plots_can_be_saved_without_swift(sample_log, tmp_path):
    logs = [make_fake_log(setting) for setting in experiment_settings((0.0, 0.001), (0, 1))]
    figures = [plot_run(sample_log, "Fake log"), plot_comparison(logs, "Fake logs")]
    for i, fig in enumerate(figures):
        path = tmp_path / f"plot_{i}.png"
        fig.savefig(path)
        assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert len(fig.axes) == 4
        plt.close(fig)


def test_demo_saves_logs_csv_and_figures(monkeypatch, tmp_path):
    monkeypatch.setattr("sys.argv", ["evaluate", "--demo", "--output", str(tmp_path)])
    main()
    paths = sorted((tmp_path / "logs").glob("*.npz"))
    assert len(paths) == 12
    assert len(comparison_groups([load_log(path) for path in paths])) == 4
    with (tmp_path / "summary.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 12
    assert all(row["data_kind"] == "fake" and row["statue_removed"] == "1" for row in rows)
    assert (tmp_path / "first_run.png").is_file()
    assert (tmp_path / "comparison.png").is_file()


def test_analysis_reads_saved_logs_without_changing_them(monkeypatch, tmp_path):
    folder = tmp_path / "input"
    output = tmp_path / "output"
    folder.mkdir()
    originals = {}
    for i, settings in enumerate(experiment_settings((0.001,), (0,))):
        path = folder / f"run_{i}.npz"
        save_log(make_fake_log(settings), path)
        originals[path] = path.read_bytes()
    monkeypatch.setattr("sys.argv", ["evaluate", str(folder), "--output", str(output)])
    main()
    assert all(path.read_bytes() == data for path, data in originals.items())
    with (output / "summary.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 4
    assert all(row["data_kind"] == "loaded" for row in rows)
    assert (output / "comparison.png").is_file()
