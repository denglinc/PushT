"""
Evaluation of Push-T policies: checkpoints, lock-step rollouts and query timing, and scoring of finished runs.

    uv run evaluation.py runs=runs/default                                # held-out success of every run below
    uv run evaluation.py runs=runs/default sampler.solver=heun sampler.nfe=4 split=selection
    uv run evaluation.py -m runs=runs/default 'sampler.nfe=1,2,4,8,16' latency=true   # with evaluate/latency_ms
    uv run evaluation.py runs=runs/<meanflow runs> sampler=mean           # MeanFlow's one step

Scoring needs no retraining: every finished run below runs= is scored with the sampler, env seeds and execution
horizons of conf/evaluation.yaml, and logged to <run>/evaluations/<time>[/<job>]/ (log.csv, config.json) and a W&B
run of its own (job_type evaluate).

A policy is queried through one compiled function per run (make_query): the graphdef and the sampler are fixed in it,
the weights are an argument, and it maps every seed's batch at once (jax.vmap over a leading seeds axis), so raw and
averaged weights, every execution horizon and training's seeds share one compilation. Queries always cover all
episodes, finished ones included, so the shapes never change; the initial noise and the Brownian draws come from the
same per-episode numpy streams as PushT's (PyTorch), drawn for live episodes only, so equal weights see equal noise.

Shapes: S is the number of seeds, N the number of episodes, K the observation history and H the chunk length.
"""

from __future__ import annotations

import os
import pickle
import shutil
import subprocess
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import gym_pusht  # noqa: F401
import gymnasium as gym
import hydra
import jax
import numpy as np
from flax import nnx
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from data import Frame, Normalizer, absolute_actions, relative_observation
from flow import Sampler
from logger import Logger
from model import FlowMatchingPolicy

ROOT = Path(__file__).parent
CHECKPOINT = "checkpoint.pkl"
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

# query(params, obs, goal, noise, brownian) -> normalized chunks, every argument with a leading seeds axis S.
Query = Callable[..., jax.Array]


def configure_jax() -> None:
    """
    Sets the runtime of both entry points, before the backend starts (nothing here creates arrays at import time).
    Note:
        - matrix products in TF32, float32 storage and accumulation with tensor-core products (PushT's allow_tf32);
        - no preallocation of most of the GPU's memory, so JAX shares the GPU with other jobs;
        - deterministic GPU kernels (--xla_gpu_deterministic_ops, 11% to 18% of the training throughput): XLA's
          autotuning otherwise picks kernels per compilation, so a seed rerun in another process differs; XLA_FLAGS
          naming the flag (=false for speed) takes precedence;
        - a persistent compilation cache in .jax_cache/ (or JAX_COMPILATION_CACHE_DIR), so every program compiles
          once per machine.
    """
    flags = os.environ.get("XLA_FLAGS", "")
    if "xla_gpu_deterministic_ops" not in flags:
        os.environ["XLA_FLAGS"] = f"{flags} --xla_gpu_deterministic_ops=true".strip()
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    jax.config.update("jax_default_matmul_precision", "tensorfloat32")
    cache = os.environ.get("JAX_COMPILATION_CACHE_DIR") or str(ROOT / ".jax_cache")
    jax.config.update("jax_compilation_cache_dir", cache)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)


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


def build_policy(
    config: Any, model_kwargs: dict[str, Any], *, flow: Any, rngs: nnx.Rngs
) -> FlowMatchingPolicy:
    """
    Builds the policy of a run config: the model's _target_ class with the run's flow matching objective.
    Args:
        - config: the run config (DictConfig or plain dict)
        - model_kwargs: obs_dim, goal_dim, action_dim, obs_horizon and pred_horizon
        - flow: the run's objective, instantiate(config["flow"]), built once outside any trace
        - rngs: the initialization keys
    Returns:
        - the policy
    """
    return instantiate(config["model"], **model_kwargs, flow=flow, rngs=rngs)


def abstract_policy(config: Any, model_kwargs: dict[str, Any]) -> FlowMatchingPolicy:
    """
    Builds the policy of a run config without its weights (shapes only): its graphdef, roles and objective.
    """
    flow = instantiate(config["flow"])
    return nnx.eval_shape(
        lambda: build_policy(config, model_kwargs, flow=flow, rngs=nnx.Rngs(params=0))
    )


def make_query(graphdef: nnx.GraphDef, sampler: Sampler) -> Query:
    """
    Compiles the policy query of a run: normalized observations and noise to normalized chunks, for every seed.
    Args:
        - graphdef: the policy's graph definition (nnx.split(policy, nnx.Param)[0])
        - sampler: the solver, NFE, time grid and guidance
    Returns:
        - query(params, obs (S, N, K, F), goal (S, N, 4), noise (S, N, H, d), brownian (S, N, B, H, d) or None)
          -> chunks (S, N, H, d), with params the seeds' weights stacked on a leading axis
    """

    def query(
        params: dict,
        obs: jax.Array,
        goal: jax.Array,
        noise: jax.Array,
        brownian: jax.Array | None,
    ) -> jax.Array:
        """
        Samples one seed's chunks.
        """
        policy = nnx.merge(graphdef, params)
        return policy.sample_actions(
            obs, goal, noise, sampler=sampler, brownian=brownian
        )

    return jax.jit(jax.vmap(query))


def save_checkpoint(
    path: Path,
    weights: dict[str, dict],
    model_kwargs: dict[str, Any],
    normalizer: Normalizer,
    config: dict[str, Any],
    step: int,
) -> None:
    """
    Saves one seed's raw and averaged weights with everything needed to rebuild the policy (see load_policy).
    Note: a pickle of plain dicts of numpy arrays, written to a temporary file first so a crash never leaves half a
    checkpoint.
    Args:
        - path: the checkpoint file
        - weights: raw | ema<decay> -> the weights, a plain dict of arrays
        - model_kwargs: the data-dependent arguments of the policy
        - normalizer: the training statistics
        - config: the resolved run config
        - step: the optimizer step of the weights
    """
    checkpoint = {
        "step": step,
        "model_kwargs": model_kwargs,
        "weights": jax.device_get(weights),
        "normalizer": asdict(normalizer),
        "config": config,
    }
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as f:
        pickle.dump(checkpoint, f)
    temporary.replace(path)


def read_checkpoint(path: Path) -> dict[str, Any]:
    """
    Reads a checkpoint file (see save_checkpoint).
    """
    with path.open("rb") as f:
        return pickle.load(f)


def load_policy(
    path: Path, weights: str
) -> tuple[FlowMatchingPolicy, dict, Normalizer, dict[str, Any]]:
    """
    Rebuilds a saved policy with one of its weights.
    Args:
        - path: the checkpoint file
        - weights: raw | ema<decay>
    Returns:
        - policy: the abstract policy (its graphdef and objective; nnx.split(policy, nnx.Param)[0])
        - params: the weights on the device, a plain dict of arrays
        - normalizer: the training statistics
        - config: the resolved run config
    """
    checkpoint = read_checkpoint(path)
    policy = abstract_policy(checkpoint["config"], checkpoint["model_kwargs"])
    template = nnx.to_pure_dict(nnx.state(policy, nnx.Param))
    params = checkpoint["weights"][weights]
    if jax.tree.structure(params) != jax.tree.structure(template) or any(
        a.shape != b.shape or a.dtype != b.dtype
        for a, b in zip(jax.tree.leaves(params), jax.tree.leaves(template))
    ):
        raise ValueError(f"{path}: the {weights} weights do not fit the run's policy.")
    return (
        policy,
        jax.device_put(params),
        Normalizer(**checkpoint["normalizer"]),
        checkpoint["config"],
    )


def query_policy(
    query: Query,
    params: dict,
    normalizer: Normalizer,
    history: np.ndarray,
    set_points: np.ndarray | None,
    alive: np.ndarray,
    rngs: list[list[np.random.Generator]],
    brownian_rngs: list[list[np.random.Generator]],
    *,
    sampler: Sampler,
    frame: Frame,
    chunk_shape: tuple[int, int],
) -> np.ndarray:
    """
    Queries the policy of every seed for world action targets.
    Args:
        - query: the run's compiled query (make_query)
        - params: the seeds' weights, stacked on a leading axis
        - normalizer: the training statistics
        - history: world state histories, shape (S, N, K, 5)
        - set_points: the targets that produced them, shape (S, N, K, 2), or None without action history
        - alive: whether each episode still runs, shape (S, N); noise is drawn for these only
        - rngs: one initial-noise generator per seed and episode
        - brownian_rngs: one Brownian generator per seed and episode (used by stochastic samplers)
        - sampler: the solver, NFE, time grid and guidance
        - frame: block | agent
        - chunk_shape: (H, d)
    Returns:
        - world action targets before clipping, shape (S, N, H, 2); finished episodes' rows are meaningless
    """
    num_seeds, num_episodes = alive.shape
    flat = (num_seeds * num_episodes,)
    # Raw environment observations enter relative, normalized learning coordinates here.
    obs, goal = relative_observation(
        history.reshape(*flat, *history.shape[2:]),
        frame,
        None
        if set_points is None
        else set_points.reshape(*flat, *set_points.shape[2:]),
    )
    noise = np.zeros((num_seeds, num_episodes, *chunk_shape))
    brownian = None
    if sampler.num_brownian:
        brownian = np.zeros(
            (num_seeds, num_episodes, sampler.num_brownian, *chunk_shape)
        )
    for s, i in zip(*np.nonzero(alive)):
        noise[s, i] = rngs[s][i].standard_normal(chunk_shape)
        if brownian is not None:
            brownian[s, i] = brownian_rngs[s][i].standard_normal(
                (sampler.num_brownian, *chunk_shape)
            )

    def per_seed(x: np.ndarray) -> np.ndarray:
        """
        Restores the leading (S, N) axes of a flattened batch, in float32.
        """
        return x.reshape(num_seeds, num_episodes, *x.shape[1:]).astype(np.float32)

    chunk = query(
        params,
        per_seed(normalizer.normalize_obs(obs)),
        per_seed(normalizer.normalize_goal(goal)),
        noise.astype(np.float32),
        None if brownian is None else brownian.astype(np.float32),
    )
    # Policy output crosses from the device back to NumPy, then to world targets of the query state.
    chunk = np.asarray(chunk).astype(np.float64)
    # A non-finite value at any sampling step stays non-finite, so the result covers the whole trajectory.
    finite = np.isfinite(chunk).all(axis=(2, 3)) | ~alive
    if not finite.all():
        bad = np.flatnonzero(~finite.all(axis=1)).tolist()
        raise FloatingPointError(
            f"Sampling produced non-finite values (seeds at {bad})."
        )
    chunk = normalizer.denormalize_action(chunk.reshape(*flat, *chunk_shape))
    targets = absolute_actions(chunk, history[:, :, -1].reshape(*flat, 5), frame)
    return targets.reshape(num_seeds, num_episodes, *targets.shape[1:])


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
    query: Query,
    params: dict,
    normalizer: Normalizer,
    *,
    seeds: np.ndarray,
    obs_horizon: int,
    exec_horizon: int,
    sampler: Sampler,
    frame: Frame,
    action_history: bool,
    chunk_shape: tuple[int, int],
) -> list[Rollouts]:
    r"""
    Runs one Push-T episode per env seed for every trained seed, all in lock-step, and returns per-episode outcomes.
    Note: every episode starts at $t = 0$ and queries the policy at $t = 0, E, 2E, \dots$, so stepping all envs together
    and batching their queries reproduces the one-episode-at-a-time protocol exactly. Each episode draws its noise
    from its own generators, so its chunks do not depend on which other episodes are still running.
    Args:
        - query: the run's compiled query (make_query)
        - params: the trained seeds' weights, stacked on a leading axis S
        - normalizer: the training statistics
        - seeds: env seeds, shape (N,)
        - obs_horizon: K
        - exec_horizon: E, actions executed open-loop per query
        - sampler: the solver, NFE, time grid and guidance
        - frame: block | agent
        - action_history: whether the policy sees the set-points of its history
        - chunk_shape: (H, d)
    Returns:
        - the rollouts of every trained seed
    """
    if not 1 <= exec_horizon <= chunk_shape[0]:
        raise ValueError(f"exec_horizon must lie in [1, {chunk_shape[0]}].")
    num_runs = jax.tree.leaves(params)[0].shape[0]
    # Evaluation owns coordinate conversion and env execution; policy owns generation.
    envs = [
        [gym.make(ENV_ID, obs_type="state", disable_env_checker=True) for _ in seeds]
        for _ in range(num_runs)
    ]
    action_low, action_high = envs[0][0].action_space.low, envs[0][0].action_space.high
    obs = np.stack(
        [
            [env.reset(seed=int(seed))[0] for env, seed in zip(row, seeds)]
            for row in envs
        ]
    )
    # Before the first step the history repeats the reset state, as the dataset pads episode starts.
    history = np.repeat(obs[:, :, None], obs_horizon, axis=2)
    # No set-point exists before the first query: the agent holds its reset position, as in the dataset.
    set_points = history[..., 0:2].copy()
    rngs = [
        [np.random.default_rng([EVAL_NOISE_SEED, int(seed)]) for seed in seeds]
        for _ in range(num_runs)
    ]
    brownian_rngs = [
        [np.random.default_rng([BROWNIAN_SEED, int(seed)]) for seed in seeds]
        for _ in range(num_runs)
    ]

    alive = np.ones((num_runs, len(seeds)), dtype=bool)
    success = np.zeros((num_runs, len(seeds)), dtype=bool)
    max_reward = np.zeros((num_runs, len(seeds)))
    plan = np.zeros((num_runs, len(seeds), *chunk_shape))

    for t in range(MAX_EPISODE_STEPS):
        # Execute E actions of each chunk open-loop; query again only after E env steps.
        if t % exec_horizon == 0:
            plan = query_policy(
                query,
                params,
                normalizer,
                history,
                set_points if action_history else None,
                alive,
                rngs,
                brownian_rngs,
                sampler=sampler,
                frame=frame,
                chunk_shape=chunk_shape,
            )
        for s, i in zip(*np.nonzero(alive)):
            # Physical action limits belong to execution, not to the flow.
            action = np.clip(plan[s, i, t % exec_horizon], action_low, action_high)
            # This advances rollout time only; it is unrelated to the training step.
            obs_i, reward, terminated, truncated, info = envs[s][i].step(
                action.astype(np.float32)
            )
            history[s, i, :-1] = history[s, i, 1:]
            history[s, i, -1] = obs_i
            # The clipped target just executed is the set-point that produced obs_i.
            set_points[s, i, :-1] = set_points[s, i, 1:]
            set_points[s, i, -1] = action
            max_reward[s, i] = max(max_reward[s, i], float(reward))
            success[s, i] |= bool(info["is_success"])
            alive[s, i] = not (terminated or truncated)
        if not alive.any():
            break

    for row in envs:
        for env in row:
            env.close()
    return [
        Rollouts(
            seeds=np.asarray(seeds),
            exec_horizon=exec_horizon,
            success=success[s],
            max_reward=max_reward[s],
        )
        for s in range(num_runs)
    ]


def time_queries(
    query: Query,
    params: dict,
    normalizer: Normalizer,
    *,
    obs_horizon: int,
    sampler: Sampler,
    frame: Frame,
    action_history: bool,
    chunk_shape: tuple[int, int],
    num_queries: int = 200,
    num_warmup: int = 20,
) -> np.ndarray:
    """
    Times single-episode policy queries (batch 1), each one complete: its result is back on the host.
    Note: a query is what a deployed policy pays per replanning: relative conversion and normalization of the state
    history, the sampler's network evaluations and the conversion back to world targets.
    Args:
        - query, params, normalizer: the policy (params of one seed, with a leading axis of 1) and its statistics
        - obs_horizon: K
        - sampler: the solver, NFE, time grid and guidance
        - frame: block | agent
        - action_history: whether the policy sees set-points
        - chunk_shape: (H, d)
        - num_queries: timed queries
        - num_warmup: untimed queries first (the first one compiles)
    Returns:
        - seconds per query, shape (num_queries,)
    """
    env = gym.make(ENV_ID, obs_type="state", disable_env_checker=True)
    obs, _ = env.reset(seed=0)
    history = np.repeat(obs[None, None, None], obs_horizon, axis=2)
    set_points = history[..., 0:2].copy() if action_history else None
    alive = np.ones((1, 1), dtype=bool)
    rngs, brownian_rngs = [[np.random.default_rng(0)]], [[np.random.default_rng(1)]]
    seconds = []
    for i in range(num_warmup + num_queries):
        start = time.perf_counter()
        # Returns numpy, so the device work is done when it returns.
        query_policy(
            query,
            params,
            normalizer,
            history,
            set_points,
            alive,
            rngs,
            brownian_rngs,
            sampler=sampler,
            frame=frame,
            chunk_shape=chunk_shape,
        )
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
    Note: reads each checkpoint whole (a pickle), weights included.
    Args:
        - root: a run directory or any directory above runs
    Returns:
        - checkpoint files whose step is the run's last (train.steps); others are reported and left out
    """
    finished = []
    for path in sorted(root.rglob(CHECKPOINT)):
        checkpoint = read_checkpoint(path)
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
    """
    run = checkpoint.parent
    sampler: Sampler = instantiate(config["sampler"])
    policy, params, normalizer, run_config = load_policy(checkpoint, config["weights"])
    if sampler.cfg_scale != 1.0 and not run_config["model"]["cond_dropout"] > 0:
        print(
            f"Skipping {run}: trained without condition dropout, so it has no guidance."
        )
        return
    try:
        sampler.check(policy.flow)
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
    try:
        logger.init_logging_writer(
            {
                "evaluation": config,
                "run": run_config,
                "checkpoint": str(checkpoint),
                "sampler_times": sampler.times(policy.flow.prediction),
                "nfe": sampler.nfe,
            }
        )
        # One run: a seeds axis of 1.
        params = jax.tree.map(lambda x: x[None], params)
        query = make_query(nnx.split(policy, nnx.Param)[0], sampler)
        policy_kwargs = dict(
            obs_horizon=data["obs_horizon"],
            sampler=sampler,
            frame=data["frame"],
            action_history=data["action_history"],
            chunk_shape=(policy.pred_horizon, policy.action_dim),
        )
        seeds = env_seeds(config["split"], config["episodes"])
        results = {
            exec_horizon: evaluate_policy(
                query,
                params,
                normalizer,
                seeds=seeds,
                exec_horizon=exec_horizon,
                **policy_kwargs,
            )[0]
            for exec_horizon in config["exec_horizons"]
            or [run_config["eval"]["exec_horizon"]]
        }
        latency = None
        if config["latency"]:
            latency = time_queries(query, params, normalizer, **policy_kwargs)
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
    configure_jax()
    config = OmegaConf.to_container(cfg, resolve=True)
    if (
        config["latency"]
        and jax.default_backend() == "gpu"
        and (busy := gpu_processes())
    ):
        raise RuntimeError(f"Latency needs an idle GPU; in use by {busy}.")
    hydra_config = HydraConfig.get()
    root = Path(os.path.abspath(config["runs"]))
    # Hydra's directory, <runs>/evaluations/<time>[/<job>], repeats below every run scored.
    suffix = Path(hydra_config.runtime.output_dir).relative_to(root)
    tag = hydra_config.job.override_dirname or "base"
    checkpoints = finished_checkpoints(root)
    if not checkpoints:
        raise FileNotFoundError(f"No finished runs below {root}.")
    for checkpoint in checkpoints:
        score_run(checkpoint, config, checkpoint.parent / suffix, tag)


if __name__ == "__main__":
    main()
