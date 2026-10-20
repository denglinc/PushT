"""
Evaluation of Push-T policies: checkpoints, lock-step rollouts and query timing, and scoring of finished runs.

    uv run evaluation.py runs=runs/default                                # held-out success of every run below
    uv run evaluation.py runs=runs/default sampler.solver=heun sampler.nfe=4 split=selection
    uv run evaluation.py -m runs=runs/default 'sampler.nfe=1,2,4,8,16' latency=true   # with evaluate/latency_ms
    uv run evaluation.py runs=runs/<meanflow runs> sampler=mean           # MeanFlow's one step

Scoring needs no retraining: every finished run below runs= is scored with the sampler, env seeds and execution
horizons of conf/evaluation.yaml, and logged to <run>/evaluations/<time>[/<job>]/ (log.csv, config.json) and a W&B
run of its own (job_type evaluate).

Shapes: N is the number of episodes, K the observation history and H the chunk length.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import gym_pusht  # noqa: F401
import gymnasium as gym
import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from data import Frame, Normalizer, absolute_actions, relative_observation
from flow import Sampler
from logger import Logger
from model import FlowMatchingPolicy

ENV_ID = "gym_pusht/PushT-v0"
# The env's TimeLimit; every episode ends by success or after this many steps.
MAX_EPISODE_STEPS = 300
# Per-episode noise streams, seeded by (stream seed, env seed): the initial noise of every query, and the Brownian
# draws of stochastic samplers in a stream of their own, so ODE and SDE samplers share the initial noise.
EVAL_NOISE_SEED = 20261016
BROWNIAN_SEED = 20261017
# Env seeds: selection decisions see only the first range, final numbers come from the second.
Split = Literal["selection", "heldout"]
SPLIT_FIRST_SEED = {"selection": 0, "heldout": 10_000}
SPLIT_SIZE = 10_000


def env_seeds(split: Split, episodes: int) -> np.ndarray:
    """
    Returns the env seeds of an evaluation split.
    Args:
        - split: selection (seeds 0..) | heldout (seeds 10000..)
        - episodes: how many seeds, at most 10000
    Returns:
        - seeds, shape (episodes,)
    """
    if not 0 < episodes <= SPLIT_SIZE:
        raise ValueError(f"episodes must lie in [1, {SPLIT_SIZE}], got {episodes}.")
    return SPLIT_FIRST_SEED[split] + np.arange(episodes)


def build_policy(config: Any, model_kwargs: dict[str, Any]) -> FlowMatchingPolicy:
    """
    Builds the policy of a run config: the model's _target_ class with the run's flow matching objective.
    Args:
        - config: the run config (DictConfig or plain dict)
        - model_kwargs: obs_dim, goal_dim, action_dim, obs_horizon and pred_horizon
    Returns:
        - the policy, on the CPU
    """
    return instantiate(
        config["model"], **model_kwargs, flow=instantiate(config["flow"])
    )


def save_checkpoint(
    path: Path,
    model: FlowMatchingPolicy,
    model_kwargs: dict[str, Any],
    normalizer: Normalizer,
    config: dict[str, Any],
    ema_models: dict[float, FlowMatchingPolicy],
    step: int,
) -> None:
    """
    Saves the raw and averaged weights with everything needed to rebuild the policy (see load_policy).
    Args:
        - path: the checkpoint file
        - model: the trained policy
        - model_kwargs: the data-dependent arguments of the policy
        - normalizer: the training statistics
        - config: the resolved run config
        - ema_models: decay -> moving average of the weights
        - step: the optimizer step of the weights
    """
    torch.save(
        {
            "step": step,
            "model_kwargs": model_kwargs,
            "state_dict": model.state_dict(),
            "ema_state_dicts": {d: m.state_dict() for d, m in ema_models.items()},
            "normalizer": asdict(normalizer),
            "config": config,
        },
        path,
    )


def load_policy(
    path: Path, device: torch.device, ema_decay: float | None = None
) -> tuple[FlowMatchingPolicy, Normalizer, dict[str, Any]]:
    """
    Rebuilds a saved policy, with its raw weights or the moving average of the given decay.
    Args:
        - path: the checkpoint file
        - device: where to put the policy
        - ema_decay: None for the raw weights
    Returns:
        - model: the policy in eval mode
        - normalizer: the training statistics
        - config: the resolved run config
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    # MeanFlow's corner_share (since removed) only shaped training draws; runs saved with it still rebuild.
    checkpoint["config"]["flow"].pop("corner_share", None)
    model = build_policy(checkpoint["config"], checkpoint["model_kwargs"]).to(device)
    if ema_decay is None:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint["ema_state_dicts"][ema_decay])
    model.eval()
    return model, Normalizer(**checkpoint["normalizer"]), checkpoint["config"]


@torch.no_grad()
def query_policy(
    model: FlowMatchingPolicy,
    normalizer: Normalizer,
    device: torch.device,
    history: np.ndarray,
    set_points: np.ndarray | None,
    rngs: list[np.random.Generator],
    brownian_rngs: list[np.random.Generator],
    *,
    sampler: Sampler,
    frame: Frame,
) -> np.ndarray:
    """
    Queries the policy for world action targets.
    Args:
        - model: the policy
        - normalizer: the training statistics
        - device: the policy's device
        - history: world state histories, shape (bs, K, 5)
        - set_points: the targets that produced them, shape (bs, K, 2), or None without action history
        - rngs: one initial-noise generator per episode
        - brownian_rngs: one Brownian generator per episode (used by stochastic samplers)
        - sampler: the solver, NFE, time grid and guidance
        - frame: block | agent
    Returns:
        - world action targets before clipping, shape (bs, H, 2)
    """
    # Raw environment observations enter relative, normalized learning coordinates here.
    obs, goal = relative_observation(history, frame, set_points)
    shape = (model.pred_horizon, model.action_dim)
    noise = np.stack([rng.standard_normal(shape) for rng in rngs])
    brownian = None
    if sampler.num_brownian:
        brownian = np.stack(
            [
                rng.standard_normal((sampler.num_brownian, *shape))
                for rng in brownian_rngs
            ]
        )

    def to_device(x: np.ndarray) -> torch.Tensor:
        """
        Moves a float32 copy of an array to the policy's device.
        """
        return torch.from_numpy(x).float().to(device)

    chunk = model.sample_actions(
        to_device(normalizer.normalize_obs(obs)),
        to_device(normalizer.normalize_goal(goal)),
        to_device(noise),
        sampler=sampler,
        brownian=None if brownian is None else to_device(brownian),
    )
    # Policy output crosses from PyTorch/GPU back to NumPy/CPU, then to world targets of the query state.
    chunk = normalizer.denormalize_action(chunk.cpu().numpy().astype(np.float64))
    return absolute_actions(chunk, history[:, -1], frame)


@dataclass
class Rollouts:
    """
    Per-episode outcomes of one evaluation, indexed like the env seeds.
    """

    # (N,) env seeds.
    seeds: np.ndarray
    exec_horizon: int
    # (N,) any step with coverage > 95%.
    success: np.ndarray
    # (N,) maximum reward over the episode.
    max_reward: np.ndarray

    def summary(self) -> dict[str, float]:
        """
        Averages the outcomes over episodes.
        Returns:
            - success_rate, and mean_reward: the mean of the per-episode maximum reward
        """
        return {
            "success_rate": float(self.success.mean()),
            "mean_reward": float(self.max_reward.mean()),
        }


def evaluate_policy(
    model: FlowMatchingPolicy,
    normalizer: Normalizer,
    device: torch.device,
    *,
    seeds: np.ndarray,
    obs_horizon: int,
    exec_horizon: int,
    sampler: Sampler,
    frame: Frame,
    action_history: bool,
) -> Rollouts:
    r"""
    Runs one Push-T episode per env seed, all in lock-step, and returns per-episode outcomes.
    Note: every episode starts at $t = 0$ and queries the policy at $t = 0, E, 2E, \dots$, so stepping all envs together
    and batching their queries reproduces the one-episode-at-a-time protocol exactly. Each episode draws its noise
    from its own generators, so its chunks do not depend on which other episodes are still running.
    Args:
        - model: the policy
        - normalizer: the training statistics
        - device: the policy's device
        - seeds: env seeds, shape (N,)
        - obs_horizon: K
        - exec_horizon: E, actions executed open-loop per query
        - sampler: the solver, NFE, time grid and guidance
        - frame: block | agent
        - action_history: whether the policy sees the set-points of its history
    Returns:
        - the rollouts
    """
    if not 1 <= exec_horizon <= model.pred_horizon:
        raise ValueError(f"exec_horizon must lie in [1, {model.pred_horizon}].")
    # Evaluation owns coordinate conversion and env execution; policy owns generation.
    model.eval()
    envs = [gym.make(ENV_ID, obs_type="state", disable_env_checker=True) for _ in seeds]
    action_low, action_high = envs[0].action_space.low, envs[0].action_space.high
    num_envs = len(envs)
    obs = np.stack([env.reset(seed=int(seed))[0] for env, seed in zip(envs, seeds)])
    # Before the first step the history repeats the reset state, as the dataset pads episode starts.
    history = np.repeat(obs[:, None], obs_horizon, axis=1)
    # No set-point exists before the first query: the agent holds its reset position, as in the dataset.
    set_points = history[..., 0:2].copy()
    rngs = [np.random.default_rng([EVAL_NOISE_SEED, int(seed)]) for seed in seeds]
    brownian_rngs = [
        np.random.default_rng([BROWNIAN_SEED, int(seed)]) for seed in seeds
    ]

    alive = np.ones(num_envs, dtype=bool)
    success = np.zeros(num_envs, dtype=bool)
    max_reward = np.zeros(num_envs)
    plan = np.zeros((num_envs, model.pred_horizon, 2))

    for t in range(MAX_EPISODE_STEPS):
        # Execute E actions of each chunk open-loop; query again only after E env steps.
        if t % exec_horizon == 0:
            index = np.flatnonzero(alive)
            plan[index] = query_policy(
                model,
                normalizer,
                device,
                history[index],
                set_points[index] if action_history else None,
                [rngs[i] for i in index],
                [brownian_rngs[i] for i in index],
                sampler=sampler,
                frame=frame,
            )
        for i in np.flatnonzero(alive):
            # Physical action limits belong to execution, not to the flow.
            action = np.clip(plan[i, t % exec_horizon], action_low, action_high)
            # This advances rollout time only; it is unrelated to training global_step.
            obs_i, reward, terminated, truncated, info = envs[i].step(
                action.astype(np.float32)
            )
            history[i, :-1] = history[i, 1:]
            history[i, -1] = obs_i
            # The clipped target just executed is the set-point that produced obs_i.
            set_points[i, :-1] = set_points[i, 1:]
            set_points[i, -1] = action
            max_reward[i] = max(max_reward[i], float(reward))
            success[i] |= bool(info["is_success"])
            alive[i] = not (terminated or truncated)
        if not alive.any():
            break

    for env in envs:
        env.close()
    return Rollouts(
        seeds=np.asarray(seeds),
        exec_horizon=exec_horizon,
        success=success,
        max_reward=max_reward,
    )


def time_queries(
    model: FlowMatchingPolicy,
    normalizer: Normalizer,
    device: torch.device,
    *,
    obs_horizon: int,
    sampler: Sampler,
    frame: Frame,
    action_history: bool,
    num_queries: int = 200,
    num_warmup: int = 20,
) -> np.ndarray:
    """
    Times single-episode policy queries (batch 1), each one complete and synchronized.
    Note: a query is what a deployed policy pays per replanning: relative conversion and normalization of the state
    history, the sampler's network evaluations and the conversion back to world targets.
    Args:
        - model, normalizer, device: the policy and its statistics
        - obs_horizon: K
        - sampler: the solver, NFE, time grid and guidance
        - frame: block | agent
        - action_history: whether the policy sees set-points
        - num_queries: timed queries
        - num_warmup: untimed queries first
    Returns:
        - seconds per query, shape (num_queries,)
    """
    model.eval()
    env = gym.make(ENV_ID, obs_type="state", disable_env_checker=True)
    obs, _ = env.reset(seed=0)
    history = np.repeat(obs[None, None], obs_horizon, axis=1)
    set_points = history[..., 0:2].copy() if action_history else None
    rngs, brownian_rngs = [np.random.default_rng(0)], [np.random.default_rng(1)]
    seconds = []
    for i in range(num_warmup + num_queries):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        query_policy(
            model,
            normalizer,
            device,
            history,
            set_points,
            rngs,
            brownian_rngs,
            sampler=sampler,
            frame=frame,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        if i >= num_warmup:
            seconds.append(time.perf_counter() - start)
    env.close()
    return np.array(seconds)


def gpu_processes(min_memory_mib: int = 1024) -> list[int]:
    """
    Lists the live processes that compute on the GPU, other than this one.
    Note: processes holding less than min_memory_mib, such as a remote desktop's video encoder, do not count.
    Args:
        - min_memory_mib: GPU memory below which a process is ignored
    Returns:
        - PIDs, none without nvidia-smi
    """
    if shutil.which("nvidia-smi") is None:
        return []
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.splitlines()
    pids = set()
    for line in query:
        pid, _, memory = line.partition(",")
        if (
            pid.strip().isdigit()
            and memory.strip().isdigit()
            and int(memory) >= min_memory_mib
        ):
            pids.add(int(pid))
    pids.discard(os.getpid())
    # nvidia-smi can list a process that has just exited.
    return sorted(p for p in pids if Path(f"/proc/{p}").exists())


def finished_checkpoints(root: Path) -> list[Path]:
    """
    Finds the checkpoints of the finished runs below a directory.
    Args:
        - root: a run directory or any directory above runs
    Returns:
        - checkpoint files whose step is the run's last (train.steps); others are reported and left out
    """
    finished = []
    for path in sorted(root.rglob("checkpoint.pt")):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        step, steps = checkpoint.get("step"), checkpoint["config"]["train"]["steps"]
        if step == steps:
            finished.append(path)
        else:
            print(f"Skipping {path.parent}: step {step} of {steps}.")
    return finished


def score_run(
    checkpoint: Path,
    config: dict[str, Any],
    log_dir: Path,
    tag: str,
    device: torch.device,
) -> None:
    """
    Scores one finished run and logs the scores to a run of their own (W&B job_type evaluate).
    Note: a run the sampler does not fit (guidance without condition dropout, Heun for a data prediction, the mean
    sampler for flow matching) is reported and skipped.
    Args:
        - checkpoint: the run's checkpoint file
        - config: the resolved evaluation config
        - log_dir: where this evaluation of the run is logged
        - tag: the evaluation's overrides, naming its W&B run
        - device: where to run the policy
    """
    run = checkpoint.parent
    sampler: Sampler = instantiate(config["sampler"])
    ema_decay = None if config["weights"] == "raw" else float(config["weights"][3:])
    model, normalizer, run_config = load_policy(checkpoint, device, ema_decay)
    if sampler.cfg_scale != 1.0 and not run_config["model"]["cond_dropout"] > 0:
        print(
            f"Skipping {run}: trained without condition dropout, so it has no guidance."
        )
        return
    try:
        sampler.check(model.flow)
    except ValueError as error:
        print(f"Skipping {run}: {error}")
        return
    data = run_config["data"]
    # The logger claims its directory before the scoring, so a clash fails early.
    logger = Logger(
        log_dir,
        project=config["wandb_project"],
        mode=config["wandb_mode"],
        group=run_config["name"],
        name=f"{run_config['arm']}/seed_{run_config['seed']}/{tag}",
        job_type="evaluate",
    )
    logger.init_logging_writer(
        {
            "evaluation": config,
            "run": run_config,
            "checkpoint": str(checkpoint),
            "sampler_times": sampler.times(model.flow.prediction),
            "nfe": sampler.nfe,
        }
    )
    try:
        policy_kwargs = dict(
            obs_horizon=data["obs_horizon"],
            sampler=sampler,
            frame=data["frame"],
            action_history=data["action_history"],
        )
        seeds = env_seeds(config["split"], config["episodes"])
        results = {
            exec_horizon: evaluate_policy(
                model,
                normalizer,
                device,
                seeds=seeds,
                exec_horizon=exec_horizon,
                **policy_kwargs,
            )
            for exec_horizon in config["exec_horizons"]
            or [run_config["eval"]["exec_horizon"]]
        }
        latency = None
        if config["latency"]:
            latency = time_queries(model, normalizer, device, **policy_kwargs)
        logger.log_scores(
            run_config["train"]["steps"],
            results,
            nfe=sampler.nfe,
            latency=latency,
        )
    except BaseException:
        logger.stop_logging_writer(exit_code=1)
        raise
    logger.stop_logging_writer()


@hydra.main(version_base="1.3", config_path="conf", config_name="evaluation")
def main(cfg: DictConfig) -> None:
    """
    Scores every finished run below runs= with the configured sampler, env seeds and execution horizons.
    """
    config = OmegaConf.to_container(cfg, resolve=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config["latency"] and device.type == "cuda" and (busy := gpu_processes()):
        raise RuntimeError(f"Latency needs an idle GPU; in use by {busy}.")
    torch.backends.cuda.matmul.allow_tf32 = True
    hydra_config = HydraConfig.get()
    root = Path(os.path.abspath(config["runs"]))
    # Hydra's directory, <runs>/evaluations/<time>[/<job>], repeats below every run scored.
    suffix = Path(hydra_config.runtime.output_dir).relative_to(root)
    tag = hydra_config.job.override_dirname or "base"
    checkpoints = finished_checkpoints(root)
    if not checkpoints:
        raise FileNotFoundError(f"No finished runs below {root}.")
    for checkpoint in checkpoints:
        score_run(checkpoint, config, checkpoint.parent / suffix, tag, device)


if __name__ == "__main__":
    main()
