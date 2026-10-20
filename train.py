"""
Train and evaluate flow matching policies for Push-T; conf/config.yaml holds every option.

    uv run train.py                                   # the default configuration
    uv run train.py flow/path=gvp seed=1              # any option from the command line
    uv run train.py seed=600 num_seeds=6              # six seeds trained together in one compiled program
    uv run train.py -m experiment=horizons            # a preset's grid, one job after another

Hydra creates each job's directory: runs/<name>/<time>_seed_<seed>, or runs/<name>/<time>/<job>_seed_<seed> for the
jobs of a -m sweep, with .hydra/ (the config and the overrides). Each of its num_seeds seeds is a run of its own in
seed_<s>/: log.csv, config.json, git/, wandb/, checkpoint.pkl and a copy of .hydra/, and later the evaluations/ of
evaluation.py. The W&B run is <arm>/seed_<s> in the group <name>, where the arm is the overrides other than the seeds
and the logging keys.

Training is pure: the state (weights, optimizer state, moving averages) is a pytree, and every draw (initialization,
batch order, flow times, noise, condition dropout) is a function of (seed, optimizer step). One compiled program runs
log.train_interval optimizer steps (a lax.scan) for every seed at once (jax.vmap); the host reads its means back
once per window, to log, evaluate and save.
"""

from __future__ import annotations

import math
import shutil
import time
from pathlib import Path
from typing import Any, NamedTuple

import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from data import PushtChunkDataset, Windows, download_pusht, load_pusht_zarr
from evaluation import (
    CHECKPOINT,
    abstract_policy,
    build_policy,
    configure_jax,
    env_seeds,
    evaluate_policy,
    make_query,
    save_checkpoint,
)
from logger import Logger

ROOT = Path(__file__).parent
# torch 2.9's Muon defaults (torch.optim.Muon).
MUON_MOMENTUM = 0.95
MUON_NS_COEFFICIENTS = (3.4445, -4.7750, 2.0315)
MUON_NS_STEPS = 5
MUON_EPS = 1e-7


def check_config(config: DictConfig) -> None:
    """
    Raises on settings that do not fit together; needs no data.
    Args:
        - config: the composed config
    """
    data, train, evaluation, log = config.data, config.train, config.eval, config.log
    if data.obs_horizon < 2:
        raise ValueError(
            "obs_horizon must be at least 2: frame t relative to itself carries nothing."
        )
    if not 1 <= evaluation.exec_horizon <= data.pred_horizon:
        raise ValueError("eval.exec_horizon must lie in [1, data.pred_horizon].")
    if not all(e >= 1 for e in evaluation.late_exec_horizons):
        raise ValueError("eval.late_exec_horizons must be positive.")
    known = {"raw", *(f"ema{d}" for d in train.ema_decays)}
    if not evaluation.weights or not set(evaluation.weights) <= known:
        raise ValueError(f"eval.weights must be chosen from {sorted(known)}.")
    if not config.num_seeds >= 1:
        raise ValueError(f"num_seeds must be at least 1, got {config.num_seeds}.")
    if train.optimizer not in ("adamw", "muon"):
        raise ValueError(f"Unknown optimizer {train.optimizer!r}.")
    if train.schedule not in ("constant", "cosine"):
        raise ValueError(f"Unknown schedule {train.schedule!r}.")
    if train.steps % log.train_interval or evaluation.interval % log.train_interval:
        raise ValueError(
            "train.steps and eval.interval must be multiples of log.train_interval: one compiled window runs "
            "log.train_interval optimizer steps."
        )
    if config.wandb_mode not in ("online", "offline", "disabled"):
        raise ValueError(f"Unknown wandb_mode {config.wandb_mode!r}.")
    sampler = instantiate(config.sampler)
    if sampler.cfg_scale != 1.0 and not config.model.cond_dropout > 0:
        raise ValueError(
            "Guidance (sampler.cfg_scale != 1) needs a policy trained with model.cond_dropout > 0."
        )
    sampler.check(instantiate(config.flow))


def lr_factor(step: Any, train: DictConfig) -> Any:
    """
    Returns the learning rate of optimizer step `step` (1-based) relative to the configured one: linear warm-up over
    train.warmup_steps, then constant or cosine to 0 at the last step.
    Args:
        - step: the step, an int on the host or an integer array inside the compiled window
        - train: the config's train block
    Returns:
        - the factor, a float or a float32 array
    """
    progress = (step - train.warmup_steps) / max(1, train.steps - train.warmup_steps)
    after = (
        1.0 if train.schedule == "constant" else 0.5 * (1 + jnp.cos(math.pi * progress))
    )
    if train.warmup_steps == 0:
        return after
    return jnp.where(step <= train.warmup_steps, step / train.warmup_steps, after)


class MuonState(NamedTuple):
    """
    Muon's momentum buffers, shaped like the weights.
    """

    momentum: optax.Updates


def scale_by_torch_muon(
    momentum: float = MUON_MOMENTUM,
    ns_coefficients: tuple[float, float, float] = MUON_NS_COEFFICIENTS,
    ns_steps: int = MUON_NS_STEPS,
    eps: float = MUON_EPS,
) -> optax.GradientTransformation:
    """
    torch 2.9's Muon direction for weight matrices, with its update scaled to AdamW's RMS (match_rms_adamw).
    Note: the momentum is torch's lerp, without bias correction; Nesterov's lookahead g.lerp(buf, m); Newton-Schulz in
    bfloat16 with float32 accumulation, on the matrix in torch's (out, in) orientation; the result scaled by
    0.2 sqrt(max(out, in)). Chained with optax.add_decayed_weights and optax.scale_by_learning_rate it is
    torch.optim.Muon(nesterov=True, adjust_lr_fn="match_rms_adamw"), whose decay uses the unscaled lr.
    Args:
        - momentum: the momentum m
        - ns_coefficients: (a, b, c) of the quintic Newton-Schulz iteration
        - ns_steps: its iterations
        - eps: the floor of the Frobenius norm the matrix is divided by
    Returns:
        - the transformation
    """
    a, b, c = ns_coefficients

    def to_bfloat16(x: jax.Array) -> jax.Array:
        """
        Rounds float32 values to bfloat16, where torch's bfloat16 tensors round.
        Note: an explicit reduce_precision, which XLA keeps under jit; a cast to bfloat16 and back would let it keep
        the float32 value instead (excess precision), which drifts from torch's update by about 1%.
        """
        return jax.lax.reduce_precision(x, exponent_bits=8, mantissa_bits=7)

    def product(x: jax.Array, y: jax.Array) -> jax.Array:
        """
        Multiplies bfloat16-valued matrices, accumulating in float32.
        """
        return jnp.matmul(
            x.astype(jnp.bfloat16),
            y.astype(jnp.bfloat16),
            preferred_element_type=jnp.float32,
        )

    def orthogonalize(update: jax.Array) -> jax.Array:
        """
        Applies torch's _zeropower_via_newtonschulz to a (in, out) kernel's update.
        """
        x = to_bfloat16(update.T.astype(jnp.float32))
        tall = x.shape[0] > x.shape[1]
        if tall:
            x = x.T
        norm = to_bfloat16(jnp.sqrt(jnp.sum(jnp.square(x))))
        x = to_bfloat16(x / jnp.maximum(norm, to_bfloat16(jnp.float32(eps))))
        for _ in range(ns_steps):
            gram = to_bfloat16(product(x, x.T))
            # torch.addmm rounds b G + c G G once, and a X + (b G + c G G) X once.
            gram_update = to_bfloat16(b * gram + c * product(gram, gram))
            x = to_bfloat16(a * x + product(gram_update, x))
        if tall:
            x = x.T
        return x.T.astype(update.dtype)

    def init(params: optax.Params) -> MuonState:
        """
        Starts the momentum at zero.
        """
        return MuonState(jax.tree.map(jnp.zeros_like, params))

    def update(
        updates: optax.Updates, state: MuonState, params: optax.Params | None = None
    ) -> tuple[optax.Updates, MuonState]:
        """
        Moves the momentum and returns the orthogonalized, scaled direction.
        """
        # torch's lerp: start + w (end - start) for w < 0.5, end - (end - start)(1 - w) otherwise.
        buffer = jax.tree.map(
            lambda buf, g: buf + (1 - momentum) * (g - buf), state.momentum, updates
        )
        lookahead = jax.tree.map(
            lambda g, buf: buf - (buf - g) * (1 - momentum), updates, buffer
        )
        direction = jax.tree.map(
            lambda u: orthogonalize(u) * (0.2 * math.sqrt(max(u.shape))), lookahead
        )
        return direction, MuonState(buffer)

    return optax.GradientTransformation(init, update)


def build_optimizer(train: DictConfig, roles: dict) -> optax.GradientTransformation:
    """
    Builds the optimizer of train.optimizer over the policy's parameter roles.
    Note: AdamW's weight decay is decoupled and acts on the weight matrices only, never on the vectors; at
    weight_decay 0 it is Adam. With muon, torch's Muon moves the hidden matrices at train.muon_lr and AdamW the
    boundary matrices and the vectors. The schedule's factor scales both learning rates; the gradient clip is global.
    Args:
        - train: the config's train block
        - roles: the policy's parameter roles (FlowMatchingPolicy.parameter_roles)
    Returns:
        - the optimizer
    """

    def schedule(base: float) -> optax.Schedule:
        """
        Returns the learning rate base * lr_factor of the update count (0-based, so step count + 1).
        """
        return lambda count: base * lr_factor(count + 1, train)

    adamw = optax.adamw(
        learning_rate=schedule(train.lr),
        b1=train.betas[0],
        b2=train.betas[1],
        eps=train.eps,
        weight_decay=train.weight_decay,
        mask=jax.tree.map(lambda role: role != "vector", roles),
    )
    optimizer = adamw
    if train.optimizer == "muon":
        muon = optax.chain(
            scale_by_torch_muon(),
            optax.add_decayed_weights(train.weight_decay),
            optax.scale_by_learning_rate(schedule(train.muon_lr)),
        )
        groups = jax.tree.map(
            lambda role: "muon" if role == "hidden" else "adamw", roles
        )
        optimizer = optax.partition({"muon": muon, "adamw": adamw}, groups)
    if train.grad_clip > 0:
        optimizer = optax.chain(optax.clip_by_global_norm(train.grad_clip), optimizer)
    return optimizer


def is_finite(state: TrainState) -> jax.Array:
    """
    Returns whether every weight and moving average of one seed's state is finite, a boolean scalar.
    """
    leaves = jax.tree.leaves((state.params, state.ema))
    return jnp.all(jnp.stack([jnp.isfinite(x).all() for x in leaves]))


class TrainState(NamedTuple):
    """
    One seed's training state, a pytree (with a leading seeds axis when seeds train together).
    """

    # The policy's weights, a plain dict of arrays (nnx.merge(graphdef, params) is the policy).
    params: dict
    opt_state: optax.OptState
    # ema<decay> -> moving average of the weights.
    ema: dict[str, dict]


def init_state(
    policy_builder: Any,
    optimizer: optax.GradientTransformation,
    keys: jax.Array,
    ema_decays: list[float],
) -> TrainState:
    """
    Initializes every seed's state.
    Args:
        - policy_builder: rngs -> a fresh policy
        - optimizer: the optimizer
        - keys: one initialization key per seed, shape (S,)
        - ema_decays: the decays of the moving averages
    Returns:
        - the states, with a leading seeds axis
    """

    def init(key: jax.Array) -> TrainState:
        """
        Initializes one seed: its policy, optimizer state and averages (copies, never the weights' own buffers).
        """
        policy = policy_builder(nnx.Rngs(params=key))
        params = nnx.to_pure_dict(nnx.split(policy, nnx.Param)[1])
        ema = {f"ema{d}": jax.tree.map(jnp.copy, params) for d in ema_decays}
        return TrainState(params, optimizer.init(params), ema)

    return jax.jit(jax.vmap(init))(keys)


def make_train_window(
    graphdef: nnx.GraphDef,
    optimizer: optax.GradientTransformation,
    *,
    ema_decays: list[float],
    batch_size: int,
    num_windows: int,
    length: int,
) -> Any:
    """
    Builds the compiled training window: `length` optimizer steps of every seed in one program.
    Note: batches follow torch's epochs: epoch e is permutation(fold_in(data_key, e)), sliced into
    num_windows // batch_size batches (the remainder is never used); step k draws from fold_in(train_key, k).
    Args:
        - graphdef: the policy's graph definition
        - optimizer: the optimizer
        - ema_decays: the decays of the moving averages
        - batch_size: bs
        - num_windows: N, the training windows
        - length: optimizer steps per window
    Returns:
        - window(states, windows, keys, first_step) -> (states, means), jitted with the states donated; keys are
          (S, 2) (train, data) keys and first_step the 1-based step of the window's first update; means maps loss,
          grad_norm and the objective's metrics to their means over the window, shape (S,), and finite to whether
          every weight and average is finite after the window
    """
    steps_per_epoch = num_windows // batch_size

    def loss_fn(params: dict, batch: Windows, key: jax.Array) -> tuple[jax.Array, dict]:
        """
        The objective of one batch.
        """
        return nnx.merge(graphdef, params).compute_loss(*batch, key=key)

    def step(
        state: TrainState,
        k: jax.Array,
        windows: Windows,
        train_key: jax.Array,
        data_key: jax.Array,
    ) -> tuple[TrainState, dict[str, jax.Array]]:
        """
        Gathers batch k, computes the loss and its gradient, updates and averages.
        """
        epoch, batch = jnp.divmod(k - 1, steps_per_epoch)
        permutation = jax.random.permutation(
            jax.random.fold_in(data_key, epoch), num_windows
        )
        index = jax.lax.dynamic_slice_in_dim(
            permutation, batch * batch_size, batch_size
        )
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.params, windows.take(index), jax.random.fold_in(train_key, k)
        )
        updates, opt_state = optimizer.update(grads, state.opt_state, state.params)
        params = optax.apply_updates(state.params, updates)
        # torch's lerp for a weight below 1/2: average + (1 - decay) (weights - average).
        ema = {
            f"ema{d}": jax.tree.map(
                lambda e, p, d=d: e + (1 - d) * (p - e), state.ema[f"ema{d}"], params
            )
            for d in ema_decays
        }
        # The norm before clipping.
        return TrainState(params, opt_state, ema), {
            "loss": loss,
            "grad_norm": optax.tree.norm(grads),
            **metrics,
        }

    def window(
        state: TrainState, windows: Windows, keys: jax.Array, first_step: jax.Array
    ) -> tuple[TrainState, dict[str, jax.Array]]:
        """
        Runs one seed's `length` steps and averages their diagnostics (a NaN stays a NaN).
        """
        train_key, data_key = keys
        state, metrics = jax.lax.scan(
            lambda s, k: step(s, k, windows, train_key, data_key),
            state,
            first_step + jnp.arange(length),
        )
        # Each loss precedes its update, so the last update's weights are checked themselves.
        return state, {**jax.tree.map(jnp.mean, metrics), "finite": is_finite(state)}

    # The dataset and the keys are arguments, not constants, so every seed and run shares one program.
    return jax.jit(jax.vmap(window, in_axes=(0, None, 0, None)), donate_argnums=0)


def run_training(config: DictConfig, loggers: list[Logger], arm: str) -> None:
    """
    Trains the job's seeds together, evaluates them periodically and logs each seed's dashboard.
    Note: a seed whose training loss turns non-finite is stopped (its run fails) and the others go on; the job then
    fails at the end.
    Args:
        - config: the composed config
        - loggers: one logger per seed, not yet started; their directories also receive the checkpoints
        - arm: the arm's name
    """
    check_config(config)
    data, train, evaluation, log = config.data, config.train, config.eval, config.log
    seeds = [config.seed + i for i in range(config.num_seeds)]
    print(
        f"Training seeds {seeds} on {jax.default_backend()}, into {loggers[0].log_dir.parent}"
    )
    resolved = {**OmegaConf.to_container(config, resolve=True), "arm": arm}

    zarr_path = download_pusht(Path(config.data_dir))
    states, actions, episode_ends = load_pusht_zarr(zarr_path)
    # Push-T windows are obs (N, K, 6 or 8), goal (N, 4) and action chunks (N, H, 2), already on the device.
    dataset = PushtChunkDataset(
        states,
        actions,
        episode_ends,
        frame=data.frame,
        obs_horizon=data.obs_horizon,
        pred_horizon=data.pred_horizon,
        action_history=data.action_history,
        padding=data.padding,
    )
    normalizer, windows = dataset.normalizer, dataset.windows
    model_kwargs = dict(
        obs_dim=windows.obs.shape[2],
        goal_dim=windows.goal.shape[1],
        action_dim=windows.actions.shape[2],
        obs_horizon=data.obs_horizon,
        pred_horizon=data.pred_horizon,
    )

    # The policy's structure, roles and objective, without weights; the objective is built once.
    abstract = abstract_policy(resolved, model_kwargs)
    graphdef = nnx.split(abstract, nnx.Param)[0]
    optimizer = build_optimizer(train, abstract.parameter_roles())
    keys = jax.vmap(lambda s: jax.random.split(jax.random.key(s), 3))(
        jnp.asarray(seeds)
    )
    init_keys, step_keys = keys[:, 0], keys[:, 1:]
    state = init_state(
        lambda rngs: build_policy(
            resolved, model_kwargs, flow=abstract.flow, rngs=rngs
        ),
        optimizer,
        init_keys,
        list(train.ema_decays),
    )
    sampler = instantiate(config.sampler)
    query = make_query(graphdef, sampler)

    info = {
        "num_params": sum(
            math.prod(x.shape) for x in jax.tree.leaves(nnx.state(abstract, nnx.Param))
        ),
        "num_windows": len(dataset),
        "num_padded_targets": dataset.num_padded,
        "sampler_times": sampler.times(abstract.flow.prediction),
        "nfe": sampler.nfe,
        "seed_block": [config.seed, config.num_seeds],
    }
    run_configs = [{**resolved, "seed": seed} for seed in seeds]
    for logger, run_config in zip(loggers, run_configs):
        # Every seed's directory holds the job's Hydra files, like a run of its own.
        hydra_dir = logger.log_dir.parent / ".hydra"
        if hydra_dir.is_dir():
            shutil.copytree(hydra_dir, logger.log_dir / ".hydra")
        logger.init_logging_writer({**run_config, "info": info})

    window = make_train_window(
        graphdef,
        optimizer,
        ema_decays=list(train.ema_decays),
        batch_size=train.batch_size,
        num_windows=len(dataset),
        length=log.train_interval,
    )
    compile_start = time.perf_counter()
    window = window.lower(state, windows, step_keys, jnp.int32(1)).compile()
    print(
        f"Compiled {log.train_interval} steps x {len(seeds)} seeds in "
        f"{time.perf_counter() - compile_start:.1f} s"
    )

    eval_steps = list(range(evaluation.interval, train.steps + 1, evaluation.interval))
    if not eval_steps or eval_steps[-1] != train.steps:
        eval_steps.append(train.steps)
    late_steps = eval_steps[-evaluation.num_late :]
    eval_seeds = env_seeds("selection", evaluation.episodes)
    weights_default = evaluation.weights[0]
    steps_per_epoch = len(dataset) // train.batch_size
    chunk_shape = (data.pred_horizon, windows.actions.shape[2])
    # The seeds still training, by index.
    alive = list(range(len(seeds)))

    def stop(i: int, step: int) -> None:
        """
        Stops a seed that diverged: its run fails, the other seeds go on, and the console summary passes on.
        Args:
            - i: the seed's index
            - step: the optimizer step
        """
        print(
            f"Non-finite training loss or weights at step {step} in seed {seeds[i]}; its run stops."
        )
        loggers[i].stop_logging_writer(exit_code=1)
        alive.remove(i)
        if alive and loggers[i].print_interval:
            loggers[alive[0]].print_interval = loggers[i].print_interval
            loggers[i].print_interval = 0

    def run_eval(step: int) -> None:
        """
        Evaluates the default setting (and at the late steps every setting) of every live seed, logs it and saves
        each seed's checkpoint.
        Args:
            - step: the optimizer step
        """
        default = (weights_default, evaluation.exec_horizon)
        settings = [default]
        if step in late_steps:
            # A grid over H shares one list of late E; those longer than this run's H are skipped.
            late = [e for e in evaluation.late_exec_horizons if e <= data.pred_horizon]
            settings += [
                (w, e)
                for w in evaluation.weights
                for e in (evaluation.exec_horizon, *late)
            ]
        live = np.asarray(alive)
        for weights, exec_horizon in dict.fromkeys(settings):
            params = state.params if weights == "raw" else state.ema[weights]
            rollouts = evaluate_policy(
                query,
                jax.tree.map(lambda x: x[live], params),
                normalizer,
                seeds=eval_seeds,
                obs_horizon=data.obs_horizon,
                exec_horizon=exec_horizon,
                sampler=sampler,
                frame=data.frame,
                action_history=data.action_history,
                chunk_shape=chunk_shape,
            )
            for i, run_rollouts in zip(alive, rollouts):
                loggers[i].log_evaluation(
                    step,
                    run_rollouts,
                    weights=weights,
                    exec_horizon=exec_horizon,
                    default=(weights, exec_horizon) == default,
                    late=step in late_steps,
                )
        weights = jax.device_get({"raw": state.params, **state.ema})
        for i in alive:
            save_checkpoint(
                loggers[i].log_dir / CHECKPOINT,
                jax.tree.map(lambda x, i=i: x[i], weights),
                model_kwargs,
                normalizer,
                run_configs[i],
                step,
            )

    for first in range(1, train.steps + 1, log.train_interval):
        last = first + log.train_interval - 1
        window_start = time.perf_counter()
        state, means = window(state, windows, step_keys, jnp.int32(first))
        # The window's means stay on the device until here, so training waits for the host once per window.
        means = {
            name: np.asarray(value) for name, value in jax.device_get(means).items()
        }
        throughput = log.train_interval / max(time.perf_counter() - window_start, 1e-9)
        for i in list(alive):
            if not (math.isfinite(means["loss"][i]) and means["finite"][i]):
                stop(i, last)
        if not alive:
            break
        if last in eval_steps:
            run_eval(last)
        factor = float(lr_factor(last, train))
        for i in alive:
            loggers[i].log_training(
                last,
                loss=float(means["loss"][i]),
                grad_norm=float(means["grad_norm"][i]),
                # The rate as the optimizer applies it, in float32.
                learning_rate=float(np.float32(train.lr * factor)),
                throughput=throughput,
                epoch=(last - 1) // steps_per_epoch,
                metrics={
                    name: float(means[name][i]) for name in abstract.flow.metric_names
                },
            )
        if len(seeds) > 1 and log.print_interval and last % log.print_interval == 0:
            print(
                f"{len(alive)} seeds: {len(alive) * throughput:.0f} optimizer steps/s in all"
            )

    for i in alive:
        loggers[i].log_late(train.steps)
        loggers[i].save_model(loggers[i].log_dir / CHECKPOINT, train.steps)
    if len(alive) < len(seeds):
        # The seeds that trained to the end finish as successful runs; only the diverged ones failed.
        for i in alive:
            loggers[i].stop_logging_writer()
        failed = [seeds[i] for i in range(len(seeds)) if i not in alive]
        raise FloatingPointError(
            f"Non-finite training loss or weights in seeds {failed}."
        )


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(config: DictConfig) -> None:
    """
    Trains the job of the command line's config in the directory Hydra created for it.
    """
    configure_jax()
    hydra_config = HydraConfig.get()
    arm = hydra_config.job.override_dirname or "base"
    job_dir = Path(hydra_config.runtime.output_dir)
    loggers = [
        Logger(
            job_dir / f"seed_{seed}",
            project=config.wandb_project,
            mode=config.wandb_mode,
            group=config.name,
            name=f"{arm}/seed_{seed}",
            total_steps=config.train.steps,
            # The first seed prints the console summary (the first live one, should it diverge).
            print_interval=config.log.print_interval if i == 0 else 0,
        )
        for i, seed in enumerate(config.seed + j for j in range(config.num_seeds))
    ]
    try:
        run_training(config, loggers, arm)
    except BaseException:
        # Every W&B run still open is marked failed; under -m the next job still runs.
        for logger in loggers:
            logger.stop_logging_writer(exit_code=1)
        raise
    for logger in loggers:
        logger.stop_logging_writer()


if __name__ == "__main__":
    main()
