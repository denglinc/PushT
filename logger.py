r"""
Logging for Push-T runs: log.csv and a W&B run (the hw1 starter's logger), in three modules.

Optimization (training rows are means over the log.train_interval steps before them):
    train/loss         the objective's value          train/lr               the learning rate
    train/grad_norm    global norm, before clipping   train/throughput       optimizer steps/s of training alone
    eval/success_rate  the default setting's          eval/mean_reward       its mean of per-episode max reward
    train/epoch        passes over the training windows
MeanFlow (MeanFlow runs only; plain squared errors in the objective's reduction, see flow.MeanFlow):
    meanflow/loss_ref  every row                      meanflow/loss_u        the rows with $s > t$
    meanflow/loss_v    the rows with $s = t$: the instantaneous velocity $\bar u^\theta_{t,t}(x \mid y)$
Push-T evaluation (weights raw, the online weights, or ema<decay>; E the actions executed per query):
    eval/success_rate_E<e>_<weights>, eval/mean_reward_E<e>_<weights>     every evaluated setting
    late/success_rate, late/mean_reward  the default setting pooled over the late evaluations, the run's headline
       numbers; late/<metric>_E<e>_<weights> for every setting
    evaluate/<metric> (the first E), evaluate/<metric>_E<e> and evaluate/nfe: the scores of evaluation.py, with
       evaluate/latency_ms when it times queries (latency=true)
The default setting is the first of eval.weights at eval.exec_horizon; evaluations run every eval.interval steps.
FID and Inception Score, the image models' evaluation, have no Push-T counterpart: eval/success_rate and
eval/mean_reward take their place.

Every row goes to log.csv in the run directory (its scalars and the step) and, in one wandb.log call at its step, to
the W&B run: online by default (uploads while running), offline (wandb sync uploads later), or disabled (log.csv
only). Next to them a run directory gets config.json (the config the logger was started with), git/ (the commit,
status and diff of the code) and wandb/ with W&B on.

W&B draws a panel per key, in a section per key prefix. A finished online run lays out the workspace of its user
(lay_out_spec): optimization (train/), evaluation (eval/), meanflow and evaluation.py (evaluate/) first, late folded
at the bottom above System, the axes of the curves titled, and no step panel.

    uv run wandb sync $(find runs/<name> -type d -name 'offline-run-*')   # uploads offline runs, evaluations too
    uv run logger.py                                                       # lays out your workspace without a run
"""

from __future__ import annotations

import argparse
import copy
import csv
import datetime
import json
import subprocess
import time
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import wandb
from wandb_gql import gql

from flow import MeanFlow

if TYPE_CHECKING:
    from evaluation import Rollouts

ROOT = Path(__file__).parent

# 1 Optimization.
TRAIN_LOSS = "train/loss"
LEARNING_RATE = "train/lr"
GRAD_NORM = "train/grad_norm"
THROUGHPUT = "train/throughput"
SUCCESS = "eval/success_rate"
MEAN_REWARD = "eval/mean_reward"
EPOCH = "train/epoch"
# 2 MeanFlow: one key per name of MeanFlow.metric_names.
MEANFLOW = "meanflow/{}"
# 3 Push-T evaluation, beside the per-setting keys <section>/<metric>_E<e>_<weights>.
LATE_SUCCESS = "late/success_rate"
LATE_REWARD = "late/mean_reward"
EVAL_SUCCESS = "evaluate/success_rate"
EVAL_REWARD = "evaluate/mean_reward"
EVAL_NFE = "evaluate/nfe"
LATENCY = "evaluate/latency_ms"
# The W&B run summary keeps these aggregates of the learning curve.
SUMMARIES = {SUCCESS: "max,last", MEAN_REWARD: "max,last"}

# The sections of the W&B workspace (lay_out_spec), top to bottom: the automatic sections of these key prefixes, open
# and renamed; any other sections, as they are; the bottom ones, folded; Hidden Panels.
TOP_SECTIONS = {
    "train": "optimization",
    "eval": "evaluation",
    "meanflow": "meanflow",
    "evaluate": "evaluation.py",
}
BOTTOM_SECTIONS = ("late", "Charts", "System")
HIDDEN_PANELS = "Hidden Panels"
# Axis titles of the curves, by key; eval/<metric>_E<e>_<weights> takes its metric's. Keys logged once per run
# (late/, evaluate/) draw no curve and keep the app's axes.
X_TITLE = "Optimizer step"
Y_TITLES = {
    TRAIN_LOSS: "Training loss",
    LEARNING_RATE: "Learning rate",
    GRAD_NORM: "Gradient norm before clipping",
    THROUGHPUT: "Optimizer steps per second",
    EPOCH: "Epoch",
    SUCCESS: "Success rate",
    MEAN_REWARD: "Mean max reward",
    MEANFLOW.format("loss_ref"): "Squared error, every pair",
    MEANFLOW.format("loss_u"): "Squared error, s > t pairs",
    MEANFLOW.format("loss_v"): "Squared error, s = t pairs",
}
# log.csv's step column. W&B rows leave it out: W&B's own step is the x-axis, and the key would only plot a diagonal
# (older runs logged it, so the workspace hides that panel).
STEP = "step"
# Seconds each try of a W&B API request waits; the public API retries a request for 20 s at most.
WANDB_TIMEOUT = 10
# (entity, project) -> the keys this process laid out the workspace for.
LAID_OUT: dict[tuple[str, str], set[str]] = {}


class Logger:
    """
    Logger for a Push-T run, the hw1 starter's: each row goes to log.csv and, in one wandb.log call, to the W&B run;
    a summary goes to the console.
    Note: log.csv keeps the scalars, and its header widens as rows bring new keys; the starter's images, histograms
    and videos would go to W&B only. Rows of one step may arrive in several calls (training, evaluation).
    """

    CSV_DISALLOWED_TYPES = (wandb.Image, wandb.Video, wandb.Histogram)

    def __init__(
        self,
        log_dir: Path,
        *,
        project: str,
        mode: Literal["online", "offline", "disabled"],
        group: str,
        name: str,
        job_type: Literal["train", "evaluate"] = "train",
        total_steps: int = 0,
        print_interval: int = 0,
    ) -> None:
        """
        Args:
            - log_dir: the run directory
            - project: W&B project; the entity comes from WANDB_ENTITY or the login's default
            - mode: online (uploads while running) | offline (wandb sync uploads later) | disabled (log.csv only)
            - group: W&B group, the runs of one experiment
            - name: W&B run name
            - job_type: train | evaluate
            - total_steps: the run's optimizer steps, for the ETA
            - print_interval: steps between console summaries; 0 = none
        """
        self.log_dir = log_dir
        self.csv_path = log_dir / "log.csv"
        self.project = project
        self.mode = mode
        self.group = group
        self.name = name
        self.job_type = job_type
        self.total_steps = total_steps
        self.print_interval = print_interval
        self.running = False
        # The W&B run's entity, online, for the workspace laid out when the run finishes.
        self.entity: str | None = None
        self.start_time = time.perf_counter()
        # The CSV's columns and every scalar row so far, rewritten when a row brings a new column.
        self.header: list[str] = []
        self.rows: list[dict[str, Any]] = []
        # The last value and step of every scalar, for the console.
        self.latest: dict[str, tuple[int, float]] = {}
        # (weights, E) -> summaries of the late evaluations.
        self.late: dict[tuple[str, int], list[dict[str, float]]] = {}
        self.default_setting: tuple[str, int] | None = None

    def init_logging_writer(self, config: dict[str, Any]) -> None:
        """
        Starts the W&B run with the config, and stores the config, the code state and the Hydra files of the run.
        Args:
            - config: the resolved config, with anything else worth keeping with the run
        """
        self.log_dir.mkdir(parents=True, exist_ok=True)
        # Claims log.csv, so that a second logger in the directory fails (FileExistsError) instead of mixing rows.
        self.csv_path.open("x").close()
        # A -m sweep runs its jobs in one process: each job finishes the previous job's run. Without symlinks the
        # files saved with an offline run are copies, so the run stays complete when its directory moves.
        wandb.init(
            project=self.project,
            config=config,
            group=self.group,
            name=self.name,
            job_type=self.job_type,
            dir=self.log_dir,
            mode=self.mode,
            reinit="finish_previous",
            settings=wandb.Settings(symlink=False),
        )
        self.running = True
        if self.mode == "online":
            self.entity = wandb.run.entity
        for tag, summary in SUMMARIES.items():
            wandb.define_metric(tag, summary=summary)
        config_path = self.log_dir / "config.json"
        config_path.write_text(json.dumps(config, indent=1, default=str))
        code_state = self._store_code_state()
        if self.mode != "disabled":
            hydra_files = sorted((self.log_dir / ".hydra").glob("*.yaml"))
            for path in [config_path, *code_state, *hydra_files]:
                wandb.save(str(path), base_path=self.log_dir, policy="now")

    def log(self, row: dict[str, Any], step: int) -> None:
        """
        Logs one row at an optimizer step: its scalars and the step to log.csv, the row to W&B in one call at the
        step.
        Args:
            - row: key -> scalar (or a wandb.Image, wandb.Histogram or wandb.Video, which skip log.csv)
            - step: the optimizer step
        """
        filtered_row = {
            k: v for k, v in row.items() if not isinstance(v, self.CSV_DISALLOWED_TYPES)
        }
        filtered_row[STEP] = step
        # Train and eval rows introduce different metrics, so widen the CSV schema.
        new_keys = [key for key in filtered_row if key not in self.header]
        self.header.extend(new_keys)
        mode = "w" if new_keys else "a"
        with self.csv_path.open(mode, newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.header, extrasaction="ignore")
            if new_keys:
                writer.writeheader()
                writer.writerows(self.rows)
            writer.writerow(filtered_row)
        wandb.log(row, step=step)
        self.rows.append(filtered_row)
        for key, value in filtered_row.items():
            self.latest[key] = (step, float(value))

    def log_training(
        self,
        step: int,
        *,
        loss: float,
        grad_norm: float,
        learning_rate: float,
        throughput: float,
        epoch: int,
        metrics: dict[str, float],
    ) -> None:
        """
        Logs the optimization module's train/ keys and MeanFlow's, and prints the console summary.
        Args:
            - step: the optimizer step
            - loss: mean of the objective over the logging window
            - grad_norm: mean gradient norm before clipping over the window
            - learning_rate: current learning rate
            - throughput: optimizer steps per second of training alone
            - epoch: passes over the training windows completed
            - metrics: the objective's diagnostics over the window, by name (MeanFlow's; none for flow matching)
        """
        self.log(
            {
                TRAIN_LOSS: loss,
                LEARNING_RATE: learning_rate,
                GRAD_NORM: grad_norm,
                THROUGHPUT: throughput,
                EPOCH: epoch,
                **{MEANFLOW.format(name): value for name, value in metrics.items()},
            },
            step,
        )
        if self.print_interval and (
            step % self.print_interval == 0 or step == self.total_steps
        ):
            self._print_summary(step)

    def log_evaluation(
        self,
        step: int,
        rollouts: Rollouts,
        *,
        weights: str,
        exec_horizon: int,
        default: bool,
        late: bool,
    ) -> None:
        """
        Logs a periodic evaluation as eval/<metric>_E<e>_<weights>; the default setting also as eval/success_rate
        and eval/mean_reward.
        Args:
            - step: the optimizer step
            - rollouts: the evaluation
            - weights: raw | ema<decay>
            - exec_horizon: E
            - default: whether this is the setting of the learning curve
            - late: whether the step is a late evaluation, pooled by log_late
        """
        summary = rollouts.summary()
        row = {
            f"eval/{key}_E{exec_horizon}_{weights}": value
            for key, value in summary.items()
        }
        if late:
            self.late.setdefault((weights, exec_horizon), []).append(summary)
        if default:
            self.default_setting = (weights, exec_horizon)
            row[SUCCESS] = summary["success_rate"]
            row[MEAN_REWARD] = summary["mean_reward"]
        self.log(row, step)

    def log_late(self, step: int) -> None:
        """
        Logs every setting's late evaluations pooled, late/<metric>_E<e>_<weights>, and the default setting's success
        and reward as late/success_rate and late/mean_reward: the run's headline numbers.
        Args:
            - step: the last optimizer step
        """
        row = {}
        for (weights, exec_horizon), summaries in self.late.items():
            pooled = {
                k: float(np.mean([s[k] for s in summaries])) for k in summaries[0]
            }
            for key, value in pooled.items():
                row[f"late/{key}_E{exec_horizon}_{weights}"] = value
            if (weights, exec_horizon) == self.default_setting:
                row[LATE_SUCCESS] = pooled["success_rate"]
                row[LATE_REWARD] = pooled["mean_reward"]
            print(
                f"Late {weights} E{exec_horizon} ({len(summaries)} evaluations): success "
                f"{pooled['success_rate']:.3f}, mean max reward {pooled['mean_reward']:.3f}"
            )
        self.log(row, step)

    def log_scores(
        self,
        step: int,
        results: dict[int, Rollouts],
        *,
        nfe: int,
        latency: np.ndarray | None = None,
    ) -> None:
        """
        Logs the scores of evaluation.py: evaluate/<metric>_E<e> for every E, the first E's as evaluate/success_rate
        and evaluate/mean_reward, and the median latency as evaluate/latency_ms when measured.
        Args:
            - step: the checkpoint's optimizer step
            - results: E -> the evaluation at that execution horizon
            - nfe: network evaluations per chunk of the sampler
            - latency: seconds per batch-1 query, shape (num_queries,)
        """
        row: dict[str, Any] = {}
        for exec_horizon, rollouts in results.items():
            summary = rollouts.summary()
            for key, value in summary.items():
                row[f"evaluate/{key}_E{exec_horizon}"] = value
            print(
                f"{self.name} E{exec_horizon}: success {summary['success_rate']:.3f}, "
                f"mean max reward {summary['mean_reward']:.3f}"
            )
        first = next(iter(results.values())).summary()
        row[EVAL_SUCCESS] = first["success_rate"]
        row[EVAL_REWARD] = first["mean_reward"]
        row[EVAL_NFE] = nfe
        if latency is not None:
            row[LATENCY] = float(np.median(latency) * 1e3)
            print(f"{self.name}: latency {row[LATENCY]:.2f} ms, NFE {nfe}")
        self.log(row, step)

    def save_model(self, path: Path, step: int) -> None:
        """
        Logs the checkpoint as a model artifact of the W&B run (not when disabled).
        """
        if self.mode == "disabled":
            return
        artifact = wandb.Artifact(
            f"policy-{wandb.run.id}", type="model", metadata={"step": step}
        )
        # Immutable: the artifact refers to the file instead of copying it first.
        artifact.add_file(str(path), policy="immutable")
        wandb.log_artifact(artifact)

    def stop_logging_writer(self, exit_code: int = 0) -> None:
        """
        Finishes the W&B run (failed with a nonzero exit code); later calls do nothing. A successful online run then
        lays out its user's W&B workspace.
        """
        if not self.running:
            return
        wandb.finish(exit_code=exit_code)
        self.running = False
        if self.mode == "online" and exit_code == 0:
            self._lay_out_workspace()

    def _lay_out_workspace(self) -> None:
        """
        Lays out the user's W&B workspace with the keys this run logged, once per process and set of keys.
        Note: the layout is a convenience, so a failure is printed and the run still ends.
        """
        laid_out = LAID_OUT.setdefault((self.entity, self.project), set())
        if set(self.header) <= laid_out:
            return
        try:
            url = lay_out_workspace(self.entity, self.project, self.header)
        # Network, API and spec errors alike: the run is finished and saved either way.
        except Exception as error:  # noqa: BLE001
            print(f"The W&B workspace was not laid out: {error!r}")
            return
        laid_out.update(self.header)
        if url is None:
            print(
                f"No W&B workspace of yours in {self.entity}/{self.project} yet: open the project once, and the "
                "next online run lays it out."
            )

    def _print_summary(self, step: int, width: int = 80, pad: int = 35) -> None:
        """
        Prints the latest values, the elapsed time and the estimated time left.
        Note: at adaptive power 1, MeanFlow's objective prints as about 1; its plain losses follow it.
        """
        lines = [
            "#" * width,
            f"\033[1m{f' Optimizer step {step}/{self.total_steps} '.center(width)}\033[0m",
            "",
            f"{'Run:':>{pad}} {self.group}/{self.name}",
        ]
        for label, tag, spec in (
            ("Training loss:", TRAIN_LOSS, ".4f"),
            *(
                (f"MeanFlow {name}:", MEANFLOW.format(name), ".4f")
                for name in MeanFlow.metric_names
            ),
            ("Gradient norm:", GRAD_NORM, ".3f"),
            ("Learning rate:", LEARNING_RATE, ".2e"),
            ("Steps per second:", THROUGHPUT, ".0f"),
            ("Epoch:", EPOCH, ".0f"),
            ("Success rate:", SUCCESS, ".3f"),
            ("Mean max reward:", MEAN_REWARD, ".3f"),
        ):
            if tag in self.latest:
                at, value = self.latest[tag]
                when = "" if at == step else f" (step {at})"
                lines.append(f"{label:>{pad}} {value:{spec}}{when}")
        elapsed = time.perf_counter() - self.start_time
        eta = elapsed / step * max(0, self.total_steps - step)
        lines += [
            "-" * width,
            f"{'Time elapsed:':>{pad}} {datetime.timedelta(seconds=int(elapsed))}",
            f"{'ETA:':>{pad}} {datetime.timedelta(seconds=int(eta))}",
        ]
        print("\n".join(lines), flush=True)

    def _store_code_state(self) -> list[Path]:
        """
        Stores the commit, status and uncommitted diff of the code in git/<repository>.diff.
        Returns:
            - the written files, none outside a git checkout
        """

        def git(*args: str) -> str:
            """
            Runs a git command in the code's directory and returns its output.
            """
            return subprocess.run(
                ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
            ).stdout.strip()

        top_level = git("rev-parse", "--show-toplevel")
        if not top_level:
            print(f"{ROOT} is not a git checkout; the code state is not stored.")
            return []
        path = self.log_dir / "git" / f"{Path(top_level).name}.diff"
        path.parent.mkdir(exist_ok=True)
        path.write_text(
            f"--- git commit ---\n{git('rev-parse', 'HEAD')}\n\n\n"
            f"--- git status ---\n{git('status')}\n\n\n"
            f"--- git diff ---\n{git('diff', 'HEAD')}\n"
        )
        return [path]


def y_title(key: str) -> str | None:
    """
    Returns the y-axis title of a key's curve: its own, its metric's for eval/<metric>_E<e>_<weights>, else None.
    """
    if key in Y_TITLES:
        return Y_TITLES[key]
    metric = key.rpartition("_E")[0]
    return Y_TITLES.get(metric) if metric.startswith("eval/") else None


def lay_out_spec(spec: dict[str, Any], keys: Iterable[str]) -> dict[str, Any]:
    """
    Lays out a W&B workspace: orders, names and folds its sections, titles the axes of the keys' curves and hides
    the step panel; every section keeps its automatic panels and everything else stays as it is.
    Note: the app keeps an automatic section's key prefix in defaultName and its title in name, and the settings
    and section of a key's automatic panel in panelConfigOverrides and panelPlacementOverrides. A laid out spec
    comes back unchanged.
    Args:
        - spec: the workspace's spec, as the app saves it
        - keys: logged keys, the curves among them get axis titles
    Returns:
        - the laid out spec, a copy
    """
    spec = copy.deepcopy(spec)
    bank = spec["section"]["panelBankConfig"]
    sections = bank["sections"]
    auto = {s.get("defaultName"): s for s in sections if s.get("isPanelsAuto")}
    top = [auto[prefix] for prefix in TOP_SECTIONS if prefix in auto]
    bottom = [auto[prefix] for prefix in BOTTOM_SECTIONS if prefix in auto]
    hidden = [
        s
        for s in sections
        if s.get("name") == HIDDEN_PANELS and not s.get("isPanelsAuto")
    ]
    placed = {id(s) for s in top + bottom + hidden}
    rest = [s for s in sections if id(s) not in placed]
    for section in top:
        section.update(name=TOP_SECTIONS[section["defaultName"]], isOpen=True)
    for section in bottom:
        section["isOpen"] = False
    bank["sections"] = top + rest + bottom + hidden

    # The app may store either map as null.
    overrides = bank.get("panelConfigOverrides") or {}
    for key in keys:
        title = y_title(key)
        if title is not None:
            config = overrides.setdefault(key, {}).setdefault("config", {})
            config.update(xAxisTitle=X_TITLE, yAxisTitle=title)
    if overrides:
        bank["panelConfigOverrides"] = overrides
    if hidden:
        placements = bank.get("panelPlacementOverrides") or {}
        placements[STEP] = {"sectionId": hidden[0]["__id__"], "orderKey": "a0"}
        bank["panelPlacementOverrides"] = placements
    return spec


def lay_out_workspace(entity: str, project: str, keys: Iterable[str]) -> str | None:
    """
    Lays out (lay_out_spec) the logged-in user's W&B workspace of a project, and saves it when that changes it.
    Note: the app creates a user's workspace, nw-nwuser<username>-w, on their first visit of the project.
    Args:
        - entity: W&B entity
        - project: W&B project
        - keys: logged keys, the curves among them get axis titles
    Returns:
        - the workspace's URL, None while the user has none
    """
    api = wandb.Api(timeout=WANDB_TIMEOUT)
    username = api.viewer.username
    found = api.client.execute(
        gql(
            """
            query Workspace($entity: String, $project: String, $name: String) {
                project(name: $project, entityName: $entity) {
                    allViews(viewType: "project-view", viewName: $name) {
                        edges { node { id name displayName spec } }
                    }
                }
            }
            """
        ),
        variable_values=dict(
            entity=entity, project=project, name=f"nw-nwuser{username}-w"
        ),
    )
    if found["project"] is None:
        raise ValueError(f"No W&B project {entity}/{project}.")
    views = found["project"]["allViews"]["edges"]
    if not views:
        return None
    view = views[0]["node"]
    spec = json.loads(view["spec"])
    laid_out = lay_out_spec(spec, keys)
    if laid_out != spec:
        api.client.execute(
            gql(
                """
                mutation Workspace(
                    $id: ID, $entity: String, $project: String, $name: String, $displayName: String, $spec: String
                ) {
                    upsertView(input: {
                        id: $id, entityName: $entity, projectName: $project, name: $name,
                        displayName: $displayName, type: "project-view", spec: $spec
                    }) { view { id } }
                }
                """
            ),
            variable_values=dict(
                id=view["id"],
                entity=entity,
                project=project,
                name=view["name"],
                displayName=view["displayName"],
                spec=json.dumps(laid_out),
            ),
        )
    app_url = api.client.app_url.rstrip("/")
    return f"{app_url}/{entity}/{project}/workspace?nw=nwuser{username}"


def main() -> None:
    """
    Lays out your W&B workspace of the project of the command line, from the logged keys of all its runs.
    Note: the history's keys, not the summary's, where a defined summary splits a key (eval/success_rate.max).
    """
    parser = argparse.ArgumentParser(
        description="Lay out your W&B workspace of the Push-T project."
    )
    parser.add_argument("--entity", help="default: the login's entity")
    parser.add_argument("--project", default="pusht")
    args = parser.parse_args()
    api = wandb.Api(timeout=WANDB_TIMEOUT)
    entity = args.entity or api.default_entity
    # Not lazy: the lightweight listing leaves out the history's keys.
    keys = {
        key
        for run in api.runs(f"{entity}/{args.project}", lazy=False)
        for key in run.history_keys["keys"]
    }
    url = lay_out_workspace(entity, args.project, keys)
    print(
        url
        or f"No W&B workspace of yours in {entity}/{args.project} yet: open the project once."
    )


if __name__ == "__main__":
    main()
