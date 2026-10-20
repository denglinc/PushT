"""
Train and evaluate a flow matching policy for Push-T; conf/config.yaml holds every option.

    uv run train.py                                   # the default configuration
    uv run train.py flow/path=gvp seed=1              # any option from the command line
    uv run train.py -m experiment=horizons            # a preset's grid, one job after another

Hydra creates each run's directory: runs/<name>/<time>_seed_<seed>, or runs/<name>/<time>/<job>_seed_<seed> for
the jobs of a -m sweep. It holds .hydra/ (the config and the overrides), log.csv, config.json, git/, wandb/ and
checkpoint.pt, and later the evaluations/ of evaluation.py. The W&B run is <arm>/seed_<seed> in the group
<name>, where the arm is the overrides other than the seed and the logging keys.
"""

from __future__ import annotations

import copy
import math
import time
from collections.abc import Callable
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.nn.attention import SDPBackend, sdpa_kernel

from data import PushtChunkDataset, download_pusht, load_pusht_zarr
from evaluation import build_policy, env_seeds, evaluate_policy, save_checkpoint
from flow import MeanFlow
from logger import Logger
from model import FlowMatchingPolicy

ROOT = Path(__file__).parent
# Eager steps on a side stream before the training step is captured as a CUDA graph.
GRAPH_WARMUP_STEPS = 3


def check_config(config: DictConfig) -> None:
    """
    Raises on settings that do not fit together; needs neither data nor a GPU.
    Args:
        - config: the composed config
    """
    data, train, evaluation = config.data, config.train, config.eval
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
    if train.optimizer not in ("adamw", "muon"):
        raise ValueError(f"Unknown optimizer {train.optimizer!r}.")
    if train.schedule not in ("constant", "cosine"):
        raise ValueError(f"Unknown schedule {train.schedule!r}.")
    if train.optimizer == "muon" and (
        train.schedule != "constant" or train.warmup_steps > 0
    ):
        raise ValueError(
            "Muon trains at a constant learning rate: torch's Muon reads its lr as a number, which the captured "
            "step would freeze; use train.schedule=constant train.warmup_steps=0, or train.optimizer=adamw."
        )
    if train.precision not in ("tf32", "bf16"):
        raise ValueError(f"Unknown precision {train.precision!r}.")
    if config.wandb_mode not in ("online", "offline", "disabled"):
        raise ValueError(f"Unknown wandb_mode {config.wandb_mode!r}.")
    sampler = instantiate(config.sampler)
    if sampler.cfg_scale != 1.0 and not config.model.cond_dropout > 0:
        raise ValueError(
            "Guidance (sampler.cfg_scale != 1) needs a policy trained with model.cond_dropout > 0."
        )
    flow = instantiate(config.flow)
    if isinstance(flow, MeanFlow) and train.precision == "bf16":
        raise ValueError(
            "MeanFlow differentiates the network forward in float32, so bf16 autocast cannot speed it up; use "
            "train.precision=tf32."
        )
    sampler.check(flow)


def build_optimizers(
    model: FlowMatchingPolicy, train: DictConfig, device: torch.device, use_graph: bool
) -> tuple[torch.optim.AdamW, list[torch.optim.Optimizer]]:
    """
    Builds the optimizers of train.optimizer over the policy's parameter groups.
    Note: AdamW's weight decay is decoupled and acts on the weight matrices only, never on the vectors; at
    weight_decay 0 it is Adam. With muon, torch's Muon moves the hidden matrices at train.muon_lr (scaled to AdamW's
    update RMS) and AdamW the boundary matrices and the vectors. A captured graph bakes in a float lr, so with the
    graph AdamW's lr is one device tensor, shared by its groups, that the schedule changes in place.
    Args:
        - model: the policy
        - train: the config's train block
        - device: the training device
        - use_graph: whether the step is captured as a CUDA graph
    Returns:
        - adamw: the AdamW optimizer, which the schedule drives
        - optimizers: every optimizer the step calls
    """
    groups = model.parameter_groups()
    use_muon = train.optimizer == "muon"
    adamw = torch.optim.AdamW(
        [
            dict(
                params=groups.boundary if use_muon else groups.hidden + groups.boundary,
                weight_decay=train.weight_decay,
            ),
            dict(params=groups.vectors, weight_decay=0.0),
        ],
        lr=torch.tensor(train.lr, device=device) if use_graph else train.lr,
        betas=tuple(train.betas),
        eps=train.eps,
        fused=device.type == "cuda",
        capturable=use_graph,
    )
    if not use_muon:
        return adamw, [adamw]
    muon = torch.optim.Muon(
        groups.hidden,
        lr=train.muon_lr,
        weight_decay=train.weight_decay,
        adjust_lr_fn="match_rms_adamw",
    )
    return adamw, [muon, adamw]


def set_seed(seed: int) -> None:
    """
    Seeds numpy, torch and every CUDA device.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loss(
    model: FlowMatchingPolicy, use_compile: bool
) -> Callable[..., tuple[torch.Tensor, dict[str, torch.Tensor]]]:
    """
    Returns the training loss, model.compute_loss, compiled by TorchInductor when use_compile is set.
    Note: compiled, attention runs on the MATH SDPA backend, faster than the fused kernels at a chunk's 16 tokens.
    fallback_random keeps torch's own draws (flow times, noise, condition dropout), and every kernel keeps Inductor's
    first config, so a seed repeats bit for bit whatever the compile cache holds.
    Args:
        - model: the policy
        - use_compile: whether to compile
    Returns:
        - the loss, called as model.compute_loss(obs, goal, action_chunk, weight)
    """
    if not use_compile:
        return model.compute_loss

    def loss(*batch: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Computes model.compute_loss with attention on the MATH backend."""
        with sdpa_kernel(SDPBackend.MATH):
            return model.compute_loss(*batch)

    return torch.compile(
        loss,
        fullgraph=True,
        options={"fallback_random": True, "triton.autotune_pointwise": False},
    )


def run_training(config: DictConfig, logger: Logger, arm: str) -> None:
    """
    Trains one policy, evaluates it periodically and logs the dashboard.
    Args:
        - config: the composed config
        - logger: the run's logger; its directory also receives the checkpoint
        - arm: the arm's name
    """
    check_config(config)
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training {logger.log_dir} on {device}")
    data, train, evaluation, log = config.data, config.train, config.eval, config.log
    resolved = {**OmegaConf.to_container(config, resolve=True), "arm": arm}

    zarr_path = download_pusht(Path(config.data_dir))
    states, actions, episode_ends = load_pusht_zarr(zarr_path)
    window_kwargs = dict(
        frame=data.frame,
        obs_horizon=data.obs_horizon,
        pred_horizon=data.pred_horizon,
        action_history=data.action_history,
        padding=data.padding,
        device=device,
    )
    # Push-T windows are obs (N, K, 6 or 8), goal (N, 4) and action chunks (N, H, 2), already on the device.
    dataset = PushtChunkDataset(states, actions, episode_ends, **window_kwargs)
    normalizer = dataset.normalizer

    model_kwargs = dict(
        obs_dim=dataset.obs.shape[2],
        goal_dim=dataset.goal.shape[1],
        action_dim=dataset.actions.shape[2],
        obs_horizon=data.obs_horizon,
        pred_horizon=data.pred_horizon,
    )
    model = build_policy(config, model_kwargs).to(device)
    # Averages never train; the captured step moves them towards the weights after every update.
    ema_models = {
        decay: copy.deepcopy(model).requires_grad_(False) for decay in train.ema_decays
    }
    sampler = instantiate(config.sampler)
    use_graph = train.cuda_graph and device.type == "cuda"
    use_compile = train.compile and device.type == "cuda"
    # TF32 matmuls: float32 storage and accumulation with tensor-core products.
    torch.backends.cuda.matmul.allow_tf32 = True

    info = {
        "num_params": sum(p.numel() for p in model.parameters()),
        "num_windows": len(dataset),
        "num_padded_targets": dataset.num_padded,
        "sampler_times": sampler.times(model.flow.prediction),
        "nfe": sampler.nfe,
        "compiled": use_compile,
    }
    logger.init_logging_writer({**resolved, "info": info})

    adamw, optimizers = build_optimizers(model, train, device, use_graph)
    scheduled = train.schedule != "constant" or train.warmup_steps > 0

    def lr_at(step: int) -> float:
        """Learning rate of optimizer step `step` (1-based)."""
        if step <= train.warmup_steps:
            return train.lr * step / train.warmup_steps
        if train.schedule == "constant":
            return train.lr
        progress = (step - train.warmup_steps) / max(
            1, train.steps - train.warmup_steps
        )
        return 0.5 * train.lr * (1 + math.cos(math.pi * progress))

    eval_steps = list(range(evaluation.interval, train.steps + 1, evaluation.interval))
    if not eval_steps or eval_steps[-1] != train.steps:
        eval_steps.append(train.steps)
    late_steps = eval_steps[-evaluation.num_late :]
    eval_seeds = env_seeds("selection", evaluation.episodes)
    checkpoint_path = logger.log_dir / "checkpoint.pt"
    weights_default = evaluation.weights[0]

    def run_eval(step: int) -> None:
        """
        Evaluates the default setting (and at the late steps every setting), logs it and saves the checkpoint.
        Args:
            - step: the optimizer step
        """
        # Log evaluation against optimizer updates; rollout env steps stay internal.
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
        for weights, exec_horizon in dict.fromkeys(settings):
            is_default = (weights, exec_horizon) == default
            policy = model if weights == "raw" else ema_models[float(weights[3:])]
            rollouts = evaluate_policy(
                policy,
                normalizer,
                device,
                seeds=eval_seeds,
                obs_horizon=data.obs_horizon,
                exec_horizon=exec_horizon,
                sampler=sampler,
                frame=data.frame,
                action_history=data.action_history,
            )
            logger.log_evaluation(
                step,
                rollouts,
                weights=weights,
                exec_horizon=exec_horizon,
                default=is_default,
                late=step in late_steps,
            )
        save_checkpoint(
            checkpoint_path, model, model_kwargs, normalizer, resolved, ema_models, step
        )
        # Evaluation enters eval mode; restore train mode (condition dropout) before the next update.
        model.train()

    # An epoch is one pass over a fresh permutation; global_step counts completed optimizer updates.
    steps_per_epoch = len(dataset) // train.batch_size
    # Static buffers: the step reads its batch through batch_index and adds its loss, gradient norm and the
    # objective's metrics to the window sums.
    batch_index = torch.zeros(train.batch_size, dtype=torch.long, device=device)
    window = {
        name: torch.zeros((), device=device)
        for name in ("loss", "grad_norm", *model.flow.metric_names)
    }
    params = list(model.parameters())
    ema_params = {decay: list(m.parameters()) for decay, m in ema_models.items()}
    if use_compile:
        # Each run compiles afresh: a -m sweep's jobs share one process, and with fullgraph the ninth distinct config
        # would exceed Dynamo's recompile limit.
        torch.compiler.reset()
    loss_fn = build_loss(model, use_compile)

    def train_step() -> None:
        """
        Gathers the batch, computes the loss, backpropagates, updates and averages: the unit captured as one
        CUDA graph.
        """
        obs, goal, action_chunk, weight = dataset.batch(batch_index)
        with torch.autocast(
            device.type,
            dtype=torch.bfloat16,
            enabled=train.precision == "bf16",
            cache_enabled=False,
        ):
            # compute_loss draws its own flow times and noise; predictions come back in float32.
            loss, metrics = loss_fn(obs, goal, action_chunk, weight)
        loss.backward()
        # The norm before clipping. No host sync (no error on non-finite norms), so it can live inside the graph.
        grads = [p.grad for p in params if p.grad is not None]
        if train.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                params, train.grad_clip, foreach=True
            )
        else:
            grad_norm = torch.nn.utils.get_total_norm(grads, foreach=True)
        for optimizer in optimizers:
            optimizer.step()
        for name, value in {
            "loss": loss.detach(),
            "grad_norm": grad_norm,
            **metrics,
        }.items():
            window[name].add_(value)
        for decay, ema in ema_params.items():
            torch._foreach_lerp_(ema, params, 1.0 - decay)

    graph: torch.cuda.CUDAGraph | None = None
    side_stream = torch.cuda.Stream() if use_graph else None
    window_steps = 0
    window_start = time.perf_counter()
    # train/throughput leaves out the steps up to the capture (the first compiles), as it leaves out evaluation.
    untimed_steps = GRAPH_WARMUP_STEPS + 1 if use_graph else int(use_compile)
    timed_steps = 0

    model.train()
    for global_step in range(1, train.steps + 1):
        epoch, batch_in_epoch = divmod(global_step - 1, steps_per_epoch)
        if batch_in_epoch == 0:
            permutation = torch.randperm(len(dataset), device=device)
        batch_index.copy_(
            permutation[
                batch_in_epoch * train.batch_size : (batch_in_epoch + 1)
                * train.batch_size
            ]
        )

        if scheduled:
            lr = lr_at(global_step)
            for group in adamw.param_groups:
                if use_graph:
                    group["lr"].fill_(lr)
                else:
                    group["lr"] = lr

        if graph is not None:
            graph.replay()
        elif use_graph and global_step > GRAPH_WARMUP_STEPS:
            # Grads set to None are allocated from the graph's pool; replays then overwrite them.
            model.zero_grad(set_to_none=True)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                train_step()
            graph.replay()
        elif use_graph:
            # Warm-up steps run eagerly on a side stream, as capture requires.
            side_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side_stream):
                model.zero_grad(set_to_none=True)
                train_step()
            torch.cuda.current_stream().wait_stream(side_stream)
        else:
            model.zero_grad(set_to_none=True)
            train_step()
        window_steps += 1
        if global_step > untimed_steps:
            timed_steps += 1
        elif global_step == untimed_steps:
            torch.cuda.synchronize()
            window_start = time.perf_counter()

        if global_step % log.train_interval == 0:
            # The sums stay on the device until here, so training never waits for a host sync.
            means = {
                name: total.item() / window_steps for name, total in window.items()
            }
            if not math.isfinite(means["loss"]):
                raise FloatingPointError(
                    f"Non-finite training loss at step {global_step}."
                )
            train_row = dict(
                loss=means.pop("loss"),
                grad_norm=means.pop("grad_norm"),
                learning_rate=float(adamw.param_groups[0]["lr"]),
                throughput=timed_steps / max(time.perf_counter() - window_start, 1e-9),
                epoch=epoch,
                metrics=means,
            )
            for total in window.values():
                total.zero_()
            window_steps = 0
            timed_steps = 0
        eval_start = time.perf_counter()
        if global_step in eval_steps:
            run_eval(global_step)
        if global_step % log.train_interval == 0:
            logger.log_training(global_step, **train_row)
            window_start = time.perf_counter()
        else:
            # Keep train/throughput a pure training measurement.
            window_start += time.perf_counter() - eval_start

    logger.log_late(train.steps)
    logger.save_model(checkpoint_path, train.steps)


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(config: DictConfig) -> None:
    """
    Trains the run of the command line's config in the directory Hydra created for it.
    """
    hydra_config = HydraConfig.get()
    arm = hydra_config.job.override_dirname or "base"
    logger = Logger(
        Path(hydra_config.runtime.output_dir),
        project=config.wandb_project,
        mode=config.wandb_mode,
        group=config.name,
        name=f"{arm}/seed_{config.seed}",
        total_steps=config.train.steps,
        print_interval=config.log.print_interval,
    )
    try:
        run_training(config, logger, arm)
    except BaseException:
        # The W&B run is marked failed; under -m the next job still runs.
        logger.stop_logging_writer(exit_code=1)
        raise
    logger.stop_logging_writer()


if __name__ == "__main__":
    main()
