"""
The logger: every row reaches log.csv and W&B at its step, under the keys of the three modules, and a finished online
run lays out its user's W&B workspace.
"""

from __future__ import annotations

import csv

import numpy as np
import pytest
import wandb

import logger as logger_module
from evaluation import Rollouts
from flow import MeanFlow
from logger import (
    EPOCH,
    EVAL_SUCCESS,
    GRAD_NORM,
    LATE_REWARD,
    LATE_SUCCESS,
    LATENCY,
    LEARNING_RATE,
    MEAN_REWARD,
    MEANFLOW,
    SUCCESS,
    THROUGHPUT,
    TRAIN_LOSS,
    Logger,
    lay_out_spec,
)


def read_csv(path) -> list[dict[str, str]]:
    """
    Reads the rows of a log.csv.
    """
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def logger(tmp_path, mode: str = "disabled") -> Logger:
    """
    Returns a started logger in tmp_path.
    """
    log = Logger(
        tmp_path,
        project="pusht_jax",
        mode=mode,
        group="test",
        name="base/seed_0",
        total_steps=200,
        print_interval=100,
    )
    log.init_logging_writer({"seed": 0})
    return log


def test_csv_receives_every_scalar_at_its_step(tmp_path):
    """
    With W&B disabled, every training scalar lands in log.csv at its step, MeanFlow's metrics under meanflow/, and
    the header widens as rows bring new keys.
    """
    log = logger(tmp_path)
    training = dict(
        grad_norm=2.0, learning_rate=3e-4, throughput=50.0, epoch=0, metrics={}
    )
    log.log_training(100, loss=1.5, **training)
    names = MeanFlow.metric_names
    metrics = {name: float(i) for i, name in enumerate(names)}
    log.log_training(200, loss=0.5, **{**training, "metrics": metrics})
    log.stop_logging_writer()
    log.stop_logging_writer()

    rows = read_csv(tmp_path / "log.csv")
    assert [row["step"] for row in rows] == ["100", "200"]
    assert float(rows[0][TRAIN_LOSS]) == 1.5 and float(rows[1][TRAIN_LOSS]) == 0.5
    assert float(rows[0][GRAD_NORM]) == 2.0 and float(rows[0][LEARNING_RATE]) == 3e-4
    assert float(rows[0][THROUGHPUT]) == 50.0 and float(rows[0][EPOCH]) == 0
    assert all(rows[0][MEANFLOW.format(name)] == "" for name in names)
    assert [float(rows[1][MEANFLOW.format(name)]) for name in names] == [0, 1, 2]
    assert (tmp_path / "config.json").exists()
    assert not (tmp_path / "wandb").exists()
    with pytest.raises(FileExistsError):
        logger(tmp_path)


def test_wandb_receives_each_row_in_one_call(tmp_path, monkeypatch):
    """
    Every row goes to W&B in one call at its step, without a step key (W&B's own step is the x-axis); values log.csv
    cannot hold go to W&B only, and log.csv keeps the step.
    """
    log = logger(tmp_path)
    # The logger logs through its own run, so the patch goes on log.run, after wandb.init made it.
    logged = []
    monkeypatch.setattr(
        log.run, "log", lambda row, step=None: logged.append((row, step))
    )
    counts, edges = np.histogram([0.2, 0.7], bins=4, range=(0.0, 1.0))
    log.log({"a": 1.0, "b": wandb.Histogram(np_histogram=(counts, edges))}, 5)
    log.log({"a": 2.0}, 6)
    log.stop_logging_writer()
    assert [(sorted(r), step) for r, step in logged] == [(["a", "b"], 5), (["a"], 6)]
    assert isinstance(logged[0][0]["b"], wandb.Histogram)
    rows = read_csv(tmp_path / "log.csv")
    assert [row["a"] for row in rows] == ["1.0", "2.0"] and "b" not in rows[0]
    assert [row["step"] for row in rows] == ["5", "6"]


def test_offline_run_keeps_copies_of_its_files(tmp_path, monkeypatch):
    """
    An offline W&B run keeps copies, not symlinks, of the config, the code state and the Hydra files, so that the
    run stays complete when its directory moves before wandb sync.
    """
    for key in ("WANDB_DIR", "WANDB_CACHE_DIR", "WANDB_CONFIG_DIR", "WANDB_DATA_DIR"):
        monkeypatch.setenv(key, str(tmp_path / key.lower()))
    monkeypatch.setenv("WANDB_SILENT", "true")
    run = tmp_path / "run"
    (run / ".hydra").mkdir(parents=True)
    (run / ".hydra" / "config.yaml").write_text("seed: 0\n")
    log = logger(run, mode="offline")
    log.log({"a": 1.0}, 1)
    log.stop_logging_writer()
    (files,) = (run / "wandb").glob("offline-run-*/files")
    for name in ("config.json", ".hydra/config.yaml"):
        assert (files / name).is_file() and not (files / name).is_symlink()


def test_offline_loggers_keep_concurrent_runs(tmp_path, monkeypatch):
    """
    Two offline loggers in one process (two seeds) keep two W&B runs alive at once, each in its own directory, and
    each logger finishes its own run.
    """
    for key in ("WANDB_DIR", "WANDB_CACHE_DIR", "WANDB_CONFIG_DIR", "WANDB_DATA_DIR"):
        monkeypatch.setenv(key, str(tmp_path / key.lower()))
    monkeypatch.setenv("WANDB_SILENT", "true")
    logs = [logger(tmp_path / f"seed_{s}", mode="offline") for s in range(2)]
    assert logs[0].run.id != logs[1].run.id
    assert all(wandb.run is not log.run for log in logs)
    for step in (1, 2):
        for s, log in enumerate(logs):
            log.log({"a": float(s)}, step)
    for log in logs:
        log.stop_logging_writer()
        # Finished: logging to the run again fails.
        with pytest.raises(wandb.Error):
            log.run.log({"a": 0.0})
    for s, log in enumerate(logs):
        (sync,) = (tmp_path / f"seed_{s}" / "wandb").glob(f"offline-run-*-{log.run.id}")
        assert (sync / f"run-{log.run.id}.wandb").is_file()
        rows = read_csv(tmp_path / f"seed_{s}" / "log.csv")
        assert [row["a"] for row in rows] == [f"{s}.0"] * 2


def test_evaluation_rows(tmp_path):
    """
    Periodic evaluations under every setting's keys and the learning curve's, their late pool and evaluation.py's
    scores and latency all reach log.csv.
    """
    rollouts = Rollouts(
        seeds=np.array([0, 1, 2]),
        exec_horizon=4,
        success=np.array([True, False, True]),
        max_reward=np.array([1.0, 0.5, 0.9]),
    )
    log = logger(tmp_path)
    for step in (10, 20):
        for weights, default in (("ema0.999", True), ("raw", False)):
            log.log_evaluation(
                step,
                rollouts,
                weights=weights,
                exec_horizon=4,
                default=default,
                late=step == 20,
            )
    log.log_late(20)
    log.log_scores(
        20, {4: rollouts, 8: rollouts}, nfe=1, latency=np.array([1e-3, 2e-3])
    )
    log.stop_logging_writer()
    rows = read_csv(tmp_path / "log.csv")
    assert [row["step"] for row in rows] == ["10", "10", "20", "20", "20", "20"]
    assert float(rows[0]["eval/success_rate_E4_ema0.999"]) == pytest.approx(2 / 3)
    assert float(rows[0][SUCCESS]) == pytest.approx(2 / 3)
    assert float(rows[0][MEAN_REWARD]) == pytest.approx(0.8)
    assert float(rows[1]["eval/mean_reward_E4_raw"]) == pytest.approx(0.8)
    assert rows[1][SUCCESS] == ""
    late = rows[4]
    assert float(late[LATE_SUCCESS]) == pytest.approx(2 / 3)
    assert float(late[LATE_REWARD]) == pytest.approx(0.8)
    assert float(late["late/success_rate_E4_raw"]) == pytest.approx(2 / 3)
    scores = rows[5]
    assert float(scores[EVAL_SUCCESS]) == pytest.approx(2 / 3)
    assert float(scores["evaluate/mean_reward_E8"]) == pytest.approx(0.8)
    assert float(scores[LATENCY]) == pytest.approx(1.5)
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "config.json",
        "git",
        "log.csv",
    ]


def workspace_spec() -> dict:
    """
    Returns a W&B workspace spec as the app saves it: automatic sections in the order their keys arrived, the
    hidden panels' section and a panel setting made in the UI.
    """

    def section(name: str, is_open: bool = True, auto: bool = True) -> dict:
        """One section of the spec; an automatic one keeps its key prefix in defaultName."""
        fields = dict(
            __id__=f"id-{name}",
            name=name,
            isOpen=is_open,
            sorted=0,
            pinned=False,
            isPanelsAuto=auto,
            panels=[],
        )
        if auto:
            fields["defaultName"] = name
        return fields

    return {
        "section": {
            "panelBankConfig": {
                "state": 1,
                "settings": {"showEmptySections": False, "sortAlphabetically": False},
                "panelConfigOverrides": {TRAIN_LOSS: {"config": {"yLogScale": True}}},
                "sections": [
                    section("evaluate", is_open=False),
                    section("Charts"),
                    section("eval"),
                    section("late"),
                    section("debug", is_open=False),
                    section("train"),
                    section("System", is_open=False),
                    section("Hidden Panels", is_open=False, auto=False),
                ],
            },
            "runSets": [{"id": "runs"}],
        }
    }


def test_workspace_layout_orders_names_and_folds_sections():
    """
    The layout puts optimization, evaluation and evaluation.py first, open and renamed, keeps other sections as they
    are and folds late, Charts and System at the bottom above Hidden Panels. Every section keeps its id, key prefix
    and automatic panels, the rest of the spec stays, and laying out again changes nothing.
    """
    spec = workspace_spec()
    laid_out = lay_out_spec(spec, [])
    sections = laid_out["section"]["panelBankConfig"]["sections"]
    assert [(s["name"], s.get("defaultName"), s["isOpen"]) for s in sections] == [
        ("optimization", "train", True),
        ("evaluation", "eval", True),
        ("evaluation.py", "evaluate", True),
        ("debug", "debug", False),
        ("late", "late", False),
        ("Charts", "Charts", False),
        ("System", "System", False),
        ("Hidden Panels", None, False),
    ]
    assert [s["isPanelsAuto"] for s in sections] == [True] * 7 + [False]
    assert sorted(s["__id__"] for s in sections) == sorted(
        s["__id__"] for s in spec["section"]["panelBankConfig"]["sections"]
    )
    assert laid_out["section"]["runSets"] == spec["section"]["runSets"]
    assert spec == workspace_spec()
    assert lay_out_spec(laid_out, []) == laid_out


def test_workspace_layout_titles_the_axes_of_curves_and_hides_the_step():
    """
    Every curve gets the optimizer step and its metric as axis titles (a per-setting curve its metric's) and keeps
    the settings made in the UI; keys logged once per run get none, and the step panel goes to Hidden Panels.
    """
    curves = {
        TRAIN_LOSS: "Training loss",
        LEARNING_RATE: "Learning rate",
        GRAD_NORM: "Gradient norm before clipping",
        THROUGHPUT: "Optimizer steps per second",
        EPOCH: "Epoch",
        SUCCESS: "Success rate",
        MEAN_REWARD: "Mean max reward",
        "eval/success_rate_E4_ema0.999": "Success rate",
        "eval/mean_reward_E8_raw": "Mean max reward",
        MEANFLOW.format("loss_ref"): "Squared error, every pair",
        MEANFLOW.format("loss_u"): "Squared error, s > t pairs",
        MEANFLOW.format("loss_v"): "Squared error, s = t pairs",
    }
    once = [
        LATE_SUCCESS,
        "late/success_rate_E4_ema0.999",
        EVAL_SUCCESS,
        LATENCY,
        "step",
    ]
    bank = lay_out_spec(workspace_spec(), [*curves, *once])["section"][
        "panelBankConfig"
    ]
    configs = {key: o["config"] for key, o in bank["panelConfigOverrides"].items()}
    assert {key: c["yAxisTitle"] for key, c in configs.items()} == curves
    assert all(c["xAxisTitle"] == "Optimizer step" for c in configs.values())
    assert configs[TRAIN_LOSS]["yLogScale"]
    assert bank["panelPlacementOverrides"] == {
        "step": {"sectionId": "id-Hidden Panels", "orderKey": "a0"}
    }


def test_finished_online_run_lays_out_its_users_workspace(
    tmp_path, monkeypatch, capsys
):
    """
    A successful online run lays out its user's workspace with the keys it logged once it finishes, once per process
    and set of keys; a failure is printed and the run still finishes, and a failed run lays out nothing.
    """
    calls = []

    def lay_out_workspace(entity, project, keys):
        """Records the call; the first fails like a lost connection."""
        calls.append((entity, project, sorted(keys)))
        if len(calls) == 1:
            raise ConnectionError("network down")
        return "url"

    monkeypatch.setattr(logger_module, "lay_out_workspace", lay_out_workspace)
    monkeypatch.setattr(logger_module, "LAID_OUT", {})

    def run(directory: str, keys: list[str], exit_code: int = 0) -> None:
        """One run logging the keys; W&B is disabled, the logger told it is online."""
        log = logger(tmp_path / directory)
        log.mode, log.entity = "online", "entity"
        log.log(dict.fromkeys(keys, 1.0), 1)
        log.stop_logging_writer(exit_code=exit_code)

    run("failed", ["a"])
    assert "network down" in capsys.readouterr().out
    run("first", ["a"])
    run("same_keys", ["a"])
    run("exit_code", ["a", "b"], exit_code=1)
    run("new_key", ["b"])
    assert calls == [("entity", "pusht_jax", ["a", "step"])] * 2 + [
        ("entity", "pusht_jax", ["b", "step"])
    ]
