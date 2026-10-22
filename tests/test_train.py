"""
The policies of model.py, the optimizers and the compiled training window of train.py, checkpoints and evaluation,
config checks, run directories and naming.
"""

from __future__ import annotations

import csv
import json
import math
import pickle
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx
from hydra import compose, initialize
from hydra._internal.core_plugins.basic_sweeper import BasicSweeper
from hydra.core.override_parser.overrides_parser import OverridesParser
from hydra.errors import ConfigCompositionException, InstantiationException
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

import train
from data import ZARR_RELATIVE_PATH, Normalizer, Windows
from evaluation import (
    CHECKPOINT,
    abstract_policy,
    build_policy,
    evaluate_policy,
    finished_checkpoints,
    load_policy,
    make_query,
    read_checkpoint,
    save_checkpoint,
)
from flow import MeanFlow, Sampler
from logger import LATE_SUCCESS, LEARNING_RATE, SUCCESS, TRAIN_LOSS, Logger
from model import FlowMatchingPolicy

PRESETS = sorted(p.stem for p in (train.ROOT / "conf" / "experiment").glob("*.yaml"))
# The seeds each preset of PushT (PyTorch) at afcbf6d swept, one run per seed: seed: range(a, b).
TORCH_SEEDS = {
    "conditioning": range(100, 104),
    "flow_matching": range(710, 713),
    "history": range(200, 204),
    "horizons": range(300, 304),
    "meanflow": range(600, 606),
    "optimizer": range(700, 703),
    "training_loop": range(400, 403),
}
DATASET = train.ROOT / "data" / ZARR_RELATIVE_PATH

MODEL_KWARGS = {
    "obs_dim": 6,
    "goal_dim": 4,
    "action_dim": 2,
    "obs_horizon": 2,
    "pred_horizon": 16,
}

needs_data = pytest.mark.skipif(
    not DATASET.exists(), reason=f"needs the Push-T dataset in {DATASET}"
)


def config(
    *overrides: str, name: str = "config", hydra_config: bool = False
) -> DictConfig:
    """
    Composes a config with overrides, with Hydra's own node when hydra_config is set.
    """
    with initialize(version_base="1.3", config_path="../conf"):
        return compose(
            name, overrides=list(overrides), return_hydra_config=hydra_config
        )


def small(*overrides: str) -> DictConfig:
    """
    Composes the config of a small policy (the DiT or, with model=mlp, the MLP).
    """
    size = (
        ("model.hidden_dims=[32,32]",)
        if "model=mlp" in overrides
        else ("model.depth=2", "model.dim=32", "model.heads=2")
    )
    return config(*overrides, *size)


def small_meanflow(*overrides: str) -> DictConfig:
    """
    Composes the config of a small MeanFlow policy.
    """
    return small("flow=meanflow", *overrides)


def new_policy(c: DictConfig) -> FlowMatchingPolicy:
    """
    Builds the policy of a config from the initialization key of seed 0.
    """
    return build_policy(
        c, MODEL_KWARGS, flow=instantiate(c.flow), rngs=nnx.Rngs(params=0)
    )


def weights(policy: FlowMatchingPolicy) -> dict:
    """
    Returns a policy's parameters, a plain dict of arrays.
    """
    return nnx.to_pure_dict(nnx.state(policy, nnx.Param))


def flat(tree: Any) -> dict[str, Any]:
    """
    Flattens a plain dict of parameters (or of their roles) to dotted names, such as blocks.0.attn.qkv.kernel.
    """
    return {
        jax.tree_util.keystr(path, simple=True, separator="."): leaf
        for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]
    }


def random_like(tree: Any, key: jax.Array) -> Any:
    """
    Draws N(0, 1) arrays shaped like the leaves of a tree.
    """
    leaves, treedef = jax.tree.flatten(tree)
    keys = jax.random.split(key, len(leaves))
    return jax.tree.unflatten(
        treedef,
        [jax.random.normal(k, x.shape, x.dtype) for k, x in zip(keys, leaves)],
    )


def assert_equal_trees(a: Any, b: Any) -> None:
    """
    Asserts that two trees have the same structure and bitwise equal leaves.
    """
    assert jax.tree.structure(a) == jax.tree.structure(b)
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        assert np.array_equal(np.asarray(x), np.asarray(y))


def draws(n: int, seed: int = 1, dtype: Any = jnp.float32) -> tuple[jax.Array, ...]:
    """
    Draws observations (n, 2, 6), goals (n, 4) and chunks or noise (n, 16, 2) from N(0, 1), and flow times from
    U(0, 1), shape (n, 1).
    """
    k = jax.random.split(jax.random.key(seed), 4)
    return (
        jax.random.normal(k[0], (n, 2, 6), dtype),
        jax.random.normal(k[1], (n, 4), dtype),
        jax.random.normal(k[2], (n, 16, 2), dtype),
        jax.random.uniform(k[3], (n, 1), dtype),
    )


def unit_normalizer() -> Normalizer:
    """
    Returns statistics that leave the features as they are.
    """
    ones = np.ones
    return Normalizer(
        ones((2, 6)), ones((2, 6)), ones(4), ones(4), ones((16, 2)), ones((16, 2))
    )


def perturbed(
    policy: FlowMatchingPolicy, skip: tuple[str, ...] = ()
) -> FlowMatchingPolicy:
    """
    Returns the policy with every parameter moved by 0.05 N(0, 1), zero-initialized layers included, so that its
    output and derivatives are not trivially zero; parameters whose name starts with one of skip stay.
    """
    graphdef, state = nnx.split(policy, nnx.Param)
    leaves, treedef = jax.tree_util.tree_flatten_with_path(nnx.to_pure_dict(state))
    keys = jax.random.split(jax.random.key(1), len(leaves))
    moved = [
        x
        if jax.tree_util.keystr(path, simple=True, separator=".").startswith(skip)
        else x + 0.05 * jax.random.normal(key, x.shape, x.dtype)
        for (path, x), key in zip(leaves, keys)
    ]
    return nnx.merge(graphdef, jax.tree.unflatten(treedef, moved))


def meanflow_policy(*overrides: str) -> FlowMatchingPolicy:
    """
    Builds a small MeanFlow policy with every parameter perturbed.
    """
    return perturbed(new_policy(small_meanflow(*overrides)))


def in_float64(policy: FlowMatchingPolicy) -> FlowMatchingPolicy:
    """
    Returns the policy with its parameters cast to float64 (inside jax.enable_x64).
    """
    graphdef, state = nnx.split(policy, nnx.Param)
    params = jax.tree.map(lambda x: x.astype(jnp.float64), nnx.to_pure_dict(state))
    return nnx.merge(graphdef, params)


def read_csv(path: Path) -> list[dict[str, str]]:
    """
    Reads the rows of a log.csv.
    """
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_arms_leave_out_the_seed_and_logging():
    """
    A run's arm, Hydra's override_dirname, keeps the overrides that change the experiment, sorted; the seed block
    (seed, num_seeds) is not one of them.
    """
    assert config(hydra_config=True).hydra.job.override_dirname == ""
    arm = config(
        "seed=3",
        "num_seeds=2",
        "name=x",
        "experiment=smoke",
        "wandb_mode=disabled",
        "log.print_interval=10",
        "flow/path=gvp",
        "data.pred_horizon=8",
        hydra_config=True,
    ).hydra.job.override_dirname
    assert arm == "data.pred_horizon=8,flow/path=gvp"
    arm = config(
        "flow=meanflow", "flow/pairs=imf", "sampler=mean", hydra_config=True
    ).hydra.job.override_dirname
    assert arm == "flow/pairs=imf,flow=meanflow,sampler=mean"


def test_runs_log_online_into_hydra_directories(monkeypatch):
    """
    Without WANDB_MODE runs log to W&B online (project pusht_jax); Hydra puts each job, named by its first seed,
    each sweep job and each evaluation in its own directory.
    """
    monkeypatch.delenv("WANDB_MODE", raising=False)
    c = config(hydra_config=True)
    assert c.wandb_mode == "online" and c.wandb_project == "pusht_jax"
    assert c.hydra.output_subdir == ".hydra"
    assert c.hydra.run.dir.startswith("runs/default/")
    assert c.hydra.run.dir.endswith("_seed_0")
    block = config("seed=600", "num_seeds=6", hydra_config=True)
    assert block.hydra.run.dir.endswith("_seed_600")
    sweep = OmegaConf.to_container(c.hydra.sweep, resolve=False)
    assert sweep["dir"].startswith("runs/${name}/") and "${seed}" in sweep["subdir"]
    assert config("runs=runs/x", name="evaluation").wandb_mode == "online"
    monkeypatch.setenv("WANDB_MODE", "offline")
    assert config().wandb_mode == "offline"
    e = config("runs=runs/x", name="evaluation", hydra_config=True)
    assert e.hydra.run.dir.startswith("runs/x/evaluations/")


def sweep_jobs(preset: str) -> list[list[str]]:
    """
    Returns the overrides of every job of a preset's sweep, as Hydra's basic sweeper splits them.
    """
    params = config(f"experiment={preset}", hydra_config=True).hydra.sweeper.params
    if not params:
        return [[]]
    overrides = OverridesParser.create().parse_overrides(
        [f"{key}={value}" for key, value in params.items()]
    )
    return BasicSweeper.split_arguments(overrides, None)[0]


@pytest.mark.parametrize("preset", PRESETS)
def test_presets_pass_config_checks(preset):
    """
    Every job of every preset's sweep fits together.
    """
    for job in sweep_jobs(preset):
        train.check_config(config(f"experiment={preset}", *job))


@pytest.mark.parametrize("preset", PRESETS)
def test_presets_train_torch_seeds_as_blocks(preset):
    """
    No preset sweeps the seed any more: every job of a preset trains the seeds torch swept, one run each, as one
    block (a single seed and num_seeds); without -m a preset still trains one seed.
    """
    params = config(f"experiment={preset}", hydra_config=True).hydra.sweeper.params
    params = params or {}
    if preset not in TORCH_SEEDS:
        assert "seed" not in params and "num_seeds" not in params
        return
    seed, num_seeds = OverridesParser.create().parse_overrides(
        [f"{key}={params[key]}" for key in ("seed", "num_seeds")]
    )
    assert not seed.is_sweep_override() and not num_seeds.is_sweep_override()
    first, count = seed.value(), num_seeds.value()
    assert isinstance(first, int) and isinstance(count, int)
    assert range(first, first + count) == TORCH_SEEDS[preset]
    for job in sweep_jobs(preset):
        c = config(f"experiment={preset}", *job)
        assert (c.seed, c.num_seeds) == (first, count)
    assert config(f"experiment={preset}").num_seeds == 1


def test_finished_checkpoints_are_those_at_the_last_step(tmp_path):
    """
    Evaluation scores only checkpoints whose step is the run's last.
    """
    for run, step in (("done", 10), ("interrupted", 5)):
        (tmp_path / run).mkdir()
        with (tmp_path / run / CHECKPOINT).open("wb") as f:
            pickle.dump({"step": step, "config": {"train": {"steps": 10}}}, f)
    assert finished_checkpoints(tmp_path) == [Path(tmp_path / "done" / CHECKPOINT)]


def test_unsupported_combinations_fail_before_training():
    """
    check_config refuses settings that do not fit together, now also runs that a whole number of compiled windows
    cannot cover and an empty seed block. Muon follows the warm-up and cosine schedules torch refused it (its
    captured step froze Muon's lr), and the precision and CUDA graph keys are gone.
    """
    with pytest.raises(ValueError):
        train.check_config(config("eval.exec_horizon=32"))
    with pytest.raises(ValueError):
        train.check_config(config("sampler.cfg_scale=1.5"))
    with pytest.raises(ValueError):
        train.check_config(config("flow.recipe=data_mse", "sampler.solver=heun"))
    with pytest.raises(ValueError):
        train.check_config(config("eval.weights=[ema0.9]"))
    with pytest.raises(ValueError):
        train.check_config(config("wandb_mode=dryrun"))
    with pytest.raises(ValueError):
        train.check_config(config("sampler=mean"))
    with pytest.raises(ValueError):
        train.check_config(config("train.optimizer=adam"))
    with pytest.raises(ValueError, match="multiples of log.train_interval"):
        train.check_config(config("train.steps=1050"))
    with pytest.raises(ValueError, match="multiples of log.train_interval"):
        train.check_config(config("eval.interval=150"))
    with pytest.raises(ValueError, match="multiples of log.train_interval"):
        train.check_config(config("log.train_interval=300"))
    train.check_config(
        config("train.steps=1050", "eval.interval=150", "log.train_interval=50")
    )
    with pytest.raises(ValueError, match="num_seeds"):
        train.check_config(config("num_seeds=0"))
    train.check_config(config("seed=600", "num_seeds=6"))
    assert "precision" not in config().train and "cuda_graph" not in config().train
    with pytest.raises(ConfigCompositionException):
        config("flow=meanflow", "train.precision=bf16")
    train.check_config(config("train.warmup_steps=10"))
    train.check_config(config("train.optimizer=adamw", "train.warmup_steps=10"))
    assert config().train.optimizer == "muon"
    assert config("experiment=meanflow").train.optimizer == "adamw"
    train.check_config(config("train.optimizer=muon", "train.schedule=cosine"))
    with pytest.raises(InstantiationException):
        train.check_config(config("flow=meanflow", "flow.flow_matching_share=1.5"))
    train.check_config(config("model=mlp", "flow/path=gvp", "flow/time=logit_normal"))
    # A logit-normal time draws on the whole of [0, 1], so it takes no narrower support and no velocity-space loss.
    with pytest.raises(InstantiationException):
        train.check_config(
            config("flow/time=logit_normal", "flow.time.support=[0.0,0.99]")
        )
    with pytest.raises(InstantiationException):
        train.check_config(
            config("flow/time=logit_normal", "flow.recipe=data_output_velocity_loss")
        )
    control = config(
        "experiment=meanflow",
        "flow=default",
        "flow.recipe=velocity_mse",
        "sampler=default",
        "train.warmup_steps=0",
    )
    train.check_config(control)
    assert (
        control.flow._target_ == "flow.FlowMatching"
        and control.sampler.solver == "euler"
    )
    assert "flow_matching_share" not in control.flow and control.train.warmup_steps == 0
    meanflow = config("experiment=meanflow")
    assert (
        meanflow.flow._target_ == "flow.MeanFlow" and meanflow.sampler.solver == "mean"
    )
    assert meanflow.train.warmup_steps == 5000
    for pairs in ("mf", "imf"):
        train.check_config(
            config("flow=meanflow", f"flow/pairs={pairs}", "sampler=mean")
        )
    c = config("flow=meanflow", "flow/pairs=mf", "flow.time.mean=0.3")
    assert c.flow.time.mean == 0.3 and c.flow.second_time.mean == 4.0


def random_windows(n: int = 64) -> Windows:
    """
    Draws n training windows of N(0, 1) features with every target step valid.
    """
    obs, goal, chunk, _ = draws(n, seed=2)
    return Windows(obs, goal, chunk, jnp.ones((n, 16)))


def training(
    c: DictConfig, windows: Windows, *, length: int, seeds: tuple[int, ...] = (0,)
) -> tuple[FlowMatchingPolicy, train.TrainState, Any, jax.Array]:
    """
    Sets up train.py's training of a config on given windows, as run_training does.
    Args:
        - c: the config
        - windows: the training windows
        - length: optimizer steps per compiled window
        - seeds: the seeds trained together
    Returns:
        - the abstract policy (its graphdef, roles and objective), the seeds' states, the compiled window and every
          seed's (train, data) keys
    """
    abstract = abstract_policy(c, MODEL_KWARGS)
    graphdef = nnx.split(abstract, nnx.Param)[0]
    optimizer = train.build_optimizer(c.train, abstract.parameter_roles())
    keys = jax.vmap(lambda s: jax.random.split(jax.random.key(s), 3))(
        jnp.asarray(seeds)
    )
    state = train.init_state(
        lambda rngs: build_policy(c, MODEL_KWARGS, flow=abstract.flow, rngs=rngs),
        optimizer,
        keys[:, 0],
        list(c.train.ema_decays),
    )
    window = train.make_train_window(
        graphdef,
        optimizer,
        ema_decays=list(c.train.ema_decays),
        batch_size=c.train.batch_size,
        num_windows=len(windows.obs),
        length=length,
    )
    return abstract, state, window, keys[:, 1:]


@pytest.mark.parametrize(
    "overrides",
    [
        (),
        ("flow/path=gvp", "flow/time=logit_normal", "flow.recipe=data_mse"),
        ("flow.recipe=scaled_score_mse", "flow.reduction=example_mean"),
        ("model.conditioning=cross", "model.cond_dropout=0.1"),
        ("model=mlp", "model.cond_dropout=0.1", "train.grad_clip=1.0"),
        ("train.optimizer=adamw", "train.weight_decay=0.1"),
        ("train.optimizer=muon", "train.weight_decay=0.1"),
        (
            "train.optimizer=muon",
            "model=mlp",
            "model.cond_dropout=0.1",
            "train.grad_clip=1.0",
        ),
        ("flow=meanflow", "sampler=mean"),
        (
            "flow=meanflow",
            "flow.tangent=predicted",
            "flow/pairs=imf",
            "flow.adaptive_power=0",
        ),
        (
            "flow=meanflow",
            "model.conditioning=cross",
            "flow/pairs=mf",
        ),
        (
            "flow=meanflow",
            "model=mlp",
            "model.cond_dropout=0.1",
            "flow.reduction=example_mean",
        ),
    ],
)
def test_compiled_window_trains(overrides):
    """
    Policy, condition dropout, loss, gradient, its norm and clipping, train.py's optimizers and averaging, as one
    compiled window of scanned steps (make_train_window on init_state and build_optimizer): the loss and the
    objective's metrics are finite, the loss falls, every weight matrix moves (Muon's included), and the average
    moves.
    Note: MeanFlow's own loss moves with its target, so for MeanFlow it is the plain flow matching loss (loss_ref at
    s = t) on fixed draws that must fall. torch's bf16 arm is gone with train.precision (JAX has no autocast). The
    64 windows are one batch, as torch's fixed batch was, here permuted at every step; the policies are the small
    ones (width 32), which train as well.
    """
    c = small(*overrides, "train.lr=1e-3", "train.muon_lr=1e-3", "train.batch_size=64")
    train.check_config(c)
    windows = random_windows(64)
    abstract, state, window, keys = training(c, windows, length=50)
    graphdef = nnx.split(abstract, nnx.Param)[0]
    fixed_time = jax.random.uniform(jax.random.key(3), (64, 1))
    fixed_noise = jax.random.normal(jax.random.key(4), (64, 16, 2))

    def fixed_loss(params: dict) -> float:
        """
        Returns the plain flow matching loss at fixed flow times (s = t) and noise, without condition dropout.
        """
        policy = nnx.merge(graphdef, jax.tree.map(lambda x: x[0], params))
        loss, metrics = policy.compute_loss(
            *windows,
            flow_time=fixed_time,
            end_time=fixed_time if policy.mean_flow else None,
            noise=fixed_noise,
        )
        return float(metrics["loss_ref"] if policy.mean_flow else loss)

    before = fixed_loss(state.params)
    # The window donates its state, so the initial weights are kept on the host.
    initial = flat(jax.device_get(state.params))
    losses, norms, metrics = [], [], []
    for i in range(4):
        state, means = window(state, windows, keys, jnp.int32(1 + 50 * i))
        means = jax.device_get(means)
        losses.append(float(means["loss"][0]))
        norms.append(float(means["grad_norm"][0]))
        metrics += [float(means[name][0]) for name in abstract.flow.metric_names]
    assert all(math.isfinite(v) for v in losses + metrics)
    if abstract.mean_flow:
        assert fixed_loss(state.params) < before
    else:
        assert losses[-1] < losses[0]
    assert all(math.isfinite(v) and v > 0 for v in norms)
    roles = flat(abstract.parameter_roles())
    params = flat(state.params)
    ema = flat(state.ema[f"ema{c.train.ema_decays[0]}"])
    assert not all(np.array_equal(ema[name], params[name]) for name in roles)
    for name, role in roles.items():
        if role != "vector":
            assert not np.array_equal(initial[name], params[name])
            assert not np.array_equal(initial[name], ema[name])


def test_two_windows_continue_one_window():
    """
    Two windows of 50 steps reach the state of one window of 100 from the same seed, and their means average to
    its means: every draw (batch order across epochs, flow times, noise, dropout) and the schedule follow the
    global step, wherever the windows split.
    """
    c = small(
        "model.cond_dropout=0.1",
        "train.schedule=cosine",
        "train.warmup_steps=20",
        "train.steps=100",
        "train.batch_size=16",
    )
    windows = random_windows(64)
    _, state, halves, keys = training(c, windows, length=50)
    _, whole_state, whole, _ = training(c, windows, length=100)
    state, first = halves(state, windows, keys, jnp.int32(1))
    state, second = halves(state, windows, keys, jnp.int32(51))
    whole_state, means = whole(whole_state, windows, keys, jnp.int32(1))
    for a, b in zip(jax.tree.leaves(state), jax.tree.leaves(whole_state)):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-6)
    assert (
        first.pop("finite").all()
        and second.pop("finite").all()
        and means.pop("finite").all()
    )
    for name, value in means.items():
        np.testing.assert_allclose((first[name] + second[name]) / 2, value, rtol=1e-6)


def test_seeds_train_side_by_side():
    """
    A seed draws the same keys whatever num_seeds is: it starts bitwise the same (weights, optimizer state and
    averages) alone or beside another seed, and trains the same way up to rounding (1e-4 of each array's scale).
    Note: results are bitwise only at the same num_seeds, since the vmapped matrix products of two seeds may round
    differently. This runs AdamW: Muon's bfloat16 Newton-Schulz amplifies such rounding in the matrices that start
    at zero (a few percent of their size after 20 steps).
    """
    c = small("model.cond_dropout=0.1", "train.batch_size=16", "train.optimizer=adamw")
    windows = random_windows(64)
    _, alone, window, keys = training(c, windows, length=20, seeds=(3,))
    _, pair, pair_window, pair_keys = training(c, windows, length=20, seeds=(3, 4))
    assert_equal_trees(jax.tree.map(lambda x: x[:1], pair), alone)
    first, second = jax.tree.leaves(pair.params)[0]
    assert not np.array_equal(first, second)
    alone, means = window(alone, windows, keys, jnp.int32(1))
    pair, pair_means = pair_window(pair, windows, pair_keys, jnp.int32(1))
    for a, b in zip(jax.tree.leaves(alone), jax.tree.leaves(pair)):
        a, b = np.asarray(a[0]), np.asarray(b[0])
        assert np.abs(a - b).max() <= 1e-4 * np.abs(a).max()
    for name, value in means.items():
        np.testing.assert_allclose(value[0], pair_means[name][0], rtol=1e-6)


@pytest.mark.parametrize(
    "overrides",
    [
        (),
        ("model.conditioning=concat",),
        ("model.conditioning=cross", "model.cond_dropout=0.1"),
        ("model=mlp",),
        ("flow=meanflow",),
        ("flow=meanflow", "model=mlp"),
    ],
)
def test_parameter_roles_split_the_parameters_by_role(overrides):
    """
    hidden, boundary and vector label every parameter: the kernels of the linear layers, those of the input
    embeddings and the output head among them, and everything else (biases, positions, null condition, time
    frequencies; vectors although some are 2-D).
    """
    policy = new_policy(small(*overrides))
    roles = flat(policy.parameter_roles())
    params = flat(weights(policy))
    assert list(roles) == list(params)
    names = {id(m): ".".join(map(str, path)) for path, m in nnx.iter_modules(policy)}
    linear = {
        f"{names[id(m)]}.kernel"
        for _, m in nnx.iter_modules(policy)
        if isinstance(m, nnx.Linear)
    }
    boundary = {f"{names[id(layer)]}.kernel" for layer in policy.boundary_layers()}
    assert {name for name, role in roles.items() if role != "vector"} == linear
    assert {name for name, role in roles.items() if role == "boundary"} == boundary
    assert set(roles.values()) == {"hidden", "boundary", "vector"}
    assert all(params[name].ndim == 2 for name in linear)
    if "model=mlp" in overrides:
        assert boundary == {"net.layers.0.kernel", "net.layers.4.kernel"}
        assert roles["null"] == "vector"
    else:
        assert roles["head.kernel"] == "boundary"
        assert roles["blocks.0.attn.qkv.kernel"] == "hidden"
        assert roles["blocks.0.ada_ln.layers.1.kernel"] == "hidden"
        for vector in ("time_embedder.weights", "action_pos", "null_tokens"):
            assert roles[vector] == "vector" and params[vector].ndim == 2


@pytest.mark.parametrize("optimizer", ["adamw", "muon"])
def test_weight_decay_acts_on_the_weight_matrices_only(optimizer):
    """
    With zero gradients a step only decays: every weight matrix shrinks by 1 - lr * weight_decay (Muon's at its own
    lr), every vector stays bitwise as it was.
    """
    c = small(
        f"train.optimizer={optimizer}", "train.weight_decay=0.1", "train.muon_lr=0.02"
    )
    policy = new_policy(c)
    params = weights(policy)
    tx = train.build_optimizer(c.train, policy.parameter_roles())
    zeros = jax.tree.map(jnp.zeros_like, params)
    updates, _ = tx.update(zeros, tx.init(params), params)
    after = flat(optax.apply_updates(params, updates))
    hidden_lr = c.train.muon_lr if optimizer == "muon" else c.train.lr
    shrink = {
        "hidden": 1 - hidden_lr * c.train.weight_decay,
        "boundary": 1 - c.train.lr * c.train.weight_decay,
    }
    for (name, role), old in zip(
        flat(policy.parameter_roles()).items(), flat(params).values()
    ):
        old, new = np.asarray(old), np.asarray(after[name])
        if role in shrink:
            np.testing.assert_allclose(new, old * shrink[role], rtol=1e-6, atol=0)
        else:
            assert np.array_equal(new, old)


def test_adamw_at_zero_weight_decay_is_adam():
    """
    At weight_decay 0 train.py's AdamW, its decay masked to the matrices, steps every parameter bit for bit as
    optax's Adam over all of them.
    """
    c = small("train.optimizer=adamw", "train.weight_decay=0.0")
    policy = new_policy(c)
    params = reference = weights(policy)
    tx = train.build_optimizer(c.train, policy.parameter_roles())
    adam = optax.adam(
        c.train.lr, b1=c.train.betas[0], b2=c.train.betas[1], eps=c.train.eps
    )
    state, adam_state = tx.init(params), adam.init(reference)
    for key in jax.random.split(jax.random.key(1), 5):
        grads = random_like(params, key)
        updates, state = tx.update(grads, state, params)
        params = optax.apply_updates(params, updates)
        updates, adam_state = adam.update(grads, adam_state, reference)
        reference = optax.apply_updates(reference, updates)
    assert_equal_trees(params, reference)


def test_gradient_clip_wraps_the_muon_partition():
    """
    The gradient clip is global and wraps Muon's partition: clipping at train.grad_clip and stepping equals stepping
    on the gradients clipped by their norm over every parameter, the hidden matrices included, whereas clipping each
    partition by its own norm moves the hidden matrices differently.
    """
    c = small("train.optimizer=muon", "train.grad_clip=1.0")
    policy = new_policy(c)
    roles = policy.parameter_roles()
    tx = train.build_optimizer(c.train, roles)
    unclipped = train.build_optimizer(
        small("train.optimizer=muon", "train.grad_clip=0.0").train, roles
    )
    hidden = jax.tree.map(lambda role: role == "hidden", roles)

    def clipped(grads: Any, norm: jax.Array) -> Any:
        """
        Scales gradients of a given norm down to train.grad_clip, as optax.clip_by_global_norm does.
        """
        if norm < c.train.grad_clip:
            return grads
        return jax.tree.map(lambda g: (g / norm) * c.train.grad_clip, grads)

    def clipped_by_partition(grads: dict) -> dict:
        """
        Clips the hidden matrices' gradients and the others' each by their own norm.
        """
        norms = {
            part: optax.tree.norm(
                [
                    g
                    for g, h in zip(jax.tree.leaves(grads), jax.tree.leaves(hidden))
                    if h == part
                ]
            )
            for part in (True, False)
        }
        return jax.tree.map(lambda g, h: clipped(g, norms[h]), grads, hidden)

    params = global_params = partition_params = weights(policy)
    state, global_state = tx.init(params), unclipped.init(params)
    partition_state = unclipped.init(params)
    # Gradients of norm 10, 0.5 and 3: the second step's stay below the clip.
    for norm, key in zip((10.0, 0.5, 3.0), jax.random.split(jax.random.key(1), 3)):
        grads = random_like(params, key)
        grads = optax.tree.scale(norm / optax.tree.norm(grads), grads)
        updates, state = tx.update(grads, state, params)
        params = optax.apply_updates(params, updates)
        updates, global_state = unclipped.update(
            clipped(grads, optax.tree.norm(grads)), global_state, global_params
        )
        global_params = optax.apply_updates(global_params, updates)
        updates, partition_state = unclipped.update(
            clipped_by_partition(grads), partition_state, partition_params
        )
        partition_params = optax.apply_updates(partition_params, updates)
    start = flat(weights(policy))
    for (name, role), a, b, p in zip(
        flat(roles).items(),
        flat(params).values(),
        flat(global_params).values(),
        flat(partition_params).values(),
    ):
        a, b, p = np.asarray(a), np.asarray(b), np.asarray(p)
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-9)
        if role == "hidden":
            moved = np.abs(a - np.asarray(start[name])).max()
            assert np.abs(a - p).max() > 1e-2 * moved, name


def torch_lr_at(step: int, train_config: DictConfig) -> float:
    """
    Returns the learning rate of optimizer step `step` (1-based) as PushT's (PyTorch) train.py sets it.
    """
    if step <= train_config.warmup_steps:
        return train_config.lr * step / train_config.warmup_steps
    if train_config.schedule == "constant":
        return train_config.lr
    progress = (step - train_config.warmup_steps) / max(
        1, train_config.steps - train_config.warmup_steps
    )
    return 0.5 * train_config.lr * (1 + math.cos(math.pi * progress))


@pytest.mark.parametrize("optimizer", ["adamw", "muon"])
@pytest.mark.parametrize(
    "schedule",
    [
        ("train.schedule=constant",),
        ("train.warmup_steps=5",),
        ("train.schedule=cosine", "train.warmup_steps=5"),
    ],
    ids=["constant", "warmup", "cosine"],
)
def test_learning_rates_follow_the_schedule(optimizer, schedule):
    """
    At update count k the optimizer steps at lr * lr_factor(k + 1), the factor of step k + 1, which is torch's lr_at;
    with muon the same factor scales muon_lr, so Muon follows the warm-up and cosine schedules that torch refused.
    Note: read off zero gradients, under which a step only decays the matrices, by its learning rate times
    weight_decay.
    """
    c = small(
        f"train.optimizer={optimizer}",
        "train.weight_decay=0.1",
        "train.muon_lr=0.02",
        "train.steps=20",
        "log.train_interval=10",
        "eval.interval=10",
        *schedule,
    )
    train.check_config(c)
    policy = new_policy(c)
    roles = flat(policy.parameter_roles())
    params = weights(policy)
    tx = train.build_optimizer(c.train, policy.parameter_roles())
    update = jax.jit(tx.update)
    state, zeros = tx.init(params), jax.tree.map(jnp.zeros_like, params)
    for count in range(c.train.steps):
        factor = float(train.lr_factor(count + 1, c.train))
        # The factor is float32, as inside the window: torch's to float32's rounding of cos.
        assert c.train.lr * factor == pytest.approx(
            torch_lr_at(count + 1, c.train), rel=1e-6, abs=1e-6 * c.train.lr
        )
        updates, state = update(zeros, state, params)
        for (name, role), p, u in zip(
            roles.items(), flat(params).values(), flat(updates).values()
        ):
            lr = (
                c.train.muon_lr
                if role == "hidden" and optimizer == "muon"
                else c.train.lr
            )
            decay = 0.0 if role == "vector" else c.train.weight_decay
            np.testing.assert_allclose(
                np.asarray(u),
                -lr * factor * decay * np.asarray(p, np.float64),
                rtol=1e-5,
                atol=1e-12,
            )


def torch_muon(
    weights: list[np.ndarray],
    grads: list[list[np.ndarray]],
    *,
    lr: float,
    weight_decay: float,
    momentum: float = 0.95,
    ns_coefficients: tuple[float, float, float] = (3.4445, -4.7750, 2.0315),
    ns_steps: int = 5,
    eps: float = 1e-7,
) -> list[np.ndarray]:
    """
    Steps weight matrices as torch 2.9's Muon does (torch.optim.Muon, nesterov, adjust_lr_fn="match_rms_adamw"),
    in float64 numpy: the momentum's lerp, Nesterov's lookahead, the quintic Newton-Schulz iteration on the matrix
    divided by its Frobenius norm (at least eps), transposed while it is taller than wide, then a decay at lr and
    the update at lr 0.2 sqrt(max(out, in)).
    Note: torch runs Newton-Schulz in bfloat16, this reference in float64; outside the tests it agreed with
    torch.optim.Muon itself to bfloat16's precision.
    Args:
        - weights: the matrices in torch's (out, in) orientation
        - grads: every step's gradients of the matrices
        - lr, weight_decay, momentum, ns_coefficients, ns_steps, eps: torch.optim.Muon's
    Returns:
        - the matrices after the steps, float64
    """
    a, b, c = ns_coefficients
    weights = [w.astype(np.float64) for w in weights]
    buffers = [np.zeros_like(w) for w in weights]
    for step_grads in grads:
        for i, grad in enumerate(step_grads):
            grad = grad.astype(np.float64)
            buffers[i] = buffers[i] + (1 - momentum) * (grad - buffers[i])
            x = grad + momentum * (buffers[i] - grad)
            tall = x.shape[0] > x.shape[1]
            if tall:
                x = x.T
            x = x / max(np.linalg.norm(x), eps)
            for _ in range(ns_steps):
                gram = x @ x.T
                x = a * x + (b * gram + c * gram @ gram) @ x
            if tall:
                x = x.T
            adjusted_lr = lr * 0.2 * math.sqrt(max(weights[i].shape))
            weights[i] = weights[i] * (1 - lr * weight_decay) - adjusted_lr * x
    return weights


def muon_problem() -> tuple[dict[str, np.ndarray], list[dict[str, np.ndarray]]]:
    """
    Draws the weight matrices (torch's (out, in) orientation: tall, wide and square) of the Muon comparison, and
    the gradients of five steps.
    """
    rng = np.random.default_rng(0)
    shapes = {"tall": (64, 32), "wide": (32, 64), "square": (48, 48)}
    matrices = {k: rng.standard_normal(s).astype(np.float32) for k, s in shapes.items()}
    grads = [
        {k: rng.standard_normal(s).astype(np.float32) for k, s in shapes.items()}
        for _ in range(5)
    ]
    return matrices, grads


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
def test_muon_matches_torch(compiled):
    """
    build_optimizer's Muon (scale_by_torch_muon, the decay and the learning rate) steps tall, wide and square
    matrices over five steps as torch's Muon does, to the precision of its bfloat16 Newton-Schulz, eagerly and
    compiled as in the training window.
    Note: the reference is torch's update in float64 numpy (torch_muon), so the tolerance is a few bfloat16
    roundings of the update (2^-8 each). Compiled, XLA may keep the products' excess precision instead of rounding
    them to bfloat16 as torch does, a difference of the same size. The weights are of order 1, so the decay (at the
    unadjusted lr) moves them about as far as the orthogonalized update, and a decay at the adjusted lr would fail.
    """
    matrices, grads = muon_problem()
    c = small("train.optimizer=muon", "train.muon_lr=0.02", "train.weight_decay=0.1")
    roles = {name: "hidden" for name in matrices} | {"bias": "vector"}
    tx = train.build_optimizer(c.train, roles)
    update = jax.jit(tx.update) if compiled else tx.update
    # JAX kernels are (in, out): torch's matrices transposed.
    params = {name: jnp.asarray(w.T) for name, w in matrices.items()}
    params["bias"] = jnp.zeros(4)
    state = tx.init(params)
    for step_grads in grads:
        step_grads = {name: jnp.asarray(g.T) for name, g in step_grads.items()}
        step_grads["bias"] = jnp.ones(4)
        updates, state = update(step_grads, state, params)
        params = optax.apply_updates(params, updates)
    names = list(matrices)
    expected = torch_muon(
        [matrices[name] for name in names],
        [[g[name] for name in names] for g in grads],
        lr=c.train.muon_lr,
        weight_decay=c.train.weight_decay,
    )
    for name, reference in zip(names, expected):
        moved = np.asarray(params[name]).T - matrices[name]
        reference = reference - matrices[name]
        assert np.linalg.norm(moved - reference) <= 0.03 * np.linalg.norm(reference)
        assert np.abs(moved - reference).max() <= 0.05 * np.abs(reference).max()


def test_parameter_counts_match_torch():
    """
    The default DiT has torch's 7,768,194 parameters, and with MeanFlow's gap embedding 7,834,114.
    """
    for overrides, count in (((), 7_768_194), (("flow=meanflow",), 7_834_114)):
        abstract = abstract_policy(config(*overrides), MODEL_KWARGS)
        params = jax.tree.leaves(nnx.state(abstract, nnx.Param))
        assert sum(math.prod(x.shape) for x in params) == count


def assert_spread(x: np.ndarray, std: float, kurtosis: float) -> None:
    """
    Asserts that values have mean 0 and a standard deviation std, within 5 standard errors of the estimates for
    their distribution's kurtosis (1.8 for a uniform, 3 for a normal distribution).
    """
    n = x.size
    assert abs(x.mean()) <= 5 * std / math.sqrt(n)
    assert abs(x.std() / std - 1) <= 5 * math.sqrt((kurtosis - 1) / (4 * n))


def test_initialization_follows_torch():
    """
    Every parameter starts as torch's: linear kernels and biases uniform in +-1/sqrt(fan_in); the adaLN
    modulations, the head, the cross-attention outputs and the gap layer at exactly zero; positions and null tokens
    from N(0, 0.02^2), time and gap frequencies from N(0, 1); the MLP's null vector and its first kernel's gap row
    at zero.
    """
    zero = (
        "ada_ln.layers.1.",
        "final_ada_ln.",
        "head.",
        "cross_attn.out.",
        "gap_embedder.layers.1.",
    )
    positions = ("action_pos", "cond_pos", "null_tokens")
    frequencies = ("time_embedder.weights", "gap_embedder.layers.0.weights")
    dit = flat(weights(new_policy(config("flow=meanflow", "model.conditioning=cross"))))
    mlp = flat(weights(new_policy(config("model=mlp", "flow=meanflow"))))
    mlp["net.layers.0.kernel"], gap_row = np.split(
        np.asarray(mlp["net.layers.0.kernel"]), [-1]
    )
    assert np.all(gap_row == 0) and np.all(np.asarray(mlp.pop("null")) == 0)
    for params in (dit, mlp):
        for name, x in params.items():
            x = np.asarray(x)
            assert x.dtype == np.float32
            if any(part in f".{name}" for part in (f".{z}" for z in zero)):
                assert np.all(x == 0), name
            elif name in positions:
                assert_spread(x, 0.02, 3.0)
            elif name in frequencies:
                assert_spread(x, 1.0, 3.0)
            else:
                layer = name.rsplit(".", 1)[0]
                fan_in = params[f"{layer}.kernel"].shape[0] + (layer == "net.layers.0")
                bound = 1 / math.sqrt(fan_in)
                assert np.abs(x).max() <= bound, name
                assert_spread(x, bound / math.sqrt(3), 1.8)


@pytest.mark.parametrize(
    "overrides",
    [
        ("model.conditioning=adaln",),
        ("model.conditioning=concat",),
        ("model.conditioning=cross",),
        ("model=mlp",),
    ],
    ids=["adaln", "concat", "cross", "mlp"],
)
def test_jvp_of_the_policy_matches_finite_differences(overrides):
    """
    MeanFlow's forward-mode derivative of the real network along (v, 1) in (x, t), s fixed, matches central
    differences in float64; feeding the gap computed outside (fixed while t moves) misses d/dh and fails.
    Note: float64 needs jax.enable_x64 and the parameters cast to float64; the passes are compiled, which is
    faster than float64's op-by-op dispatch.
    """
    policy = meanflow_policy(*overrides)
    with jax.enable_x64(True):
        policy = in_float64(policy)
        obs, goal, x, t = draws(8, dtype=jnp.float64)
        v = jax.random.normal(jax.random.key(5), (8, 16, 2), jnp.float64)
        t = 0.2 + 0.5 * t
        s = t + 0.25
        condition = policy.encode_condition(obs, goal)

        @jax.jit
        def predict(xx, tt, ss):
            """
            The policy's network for the fixed condition.
            """
            return policy._predict(condition, xx, tt, ss)

        @jax.jit
        def derivatives(x, t, s, v):
            """
            The average velocity and its derivative, with the gap formed inside and fixed outside the network.
            """
            gap = s - t
            average, derivative = MeanFlow.average_velocity(predict, x, t, s, v)
            _, closed = MeanFlow.average_velocity(
                lambda xx, tt, ss: predict(xx, tt, tt + gap), x, t, s, v
            )
            return average, derivative, closed

        average, derivative, closed = derivatives(x, t, s, v)
        assert average.dtype == derivative.dtype == jnp.float64
        h = 1e-6
        fd = (predict(x + h * v, t + h, s) - predict(x - h * v, t - h, s)) / (2 * h)
        assert jnp.allclose(average, predict(x, t, s), rtol=1e-10, atol=1e-12)
        assert jnp.abs(derivative - fd).max() <= 1e-6 * jnp.abs(fd).max()
        assert jnp.abs(closed - fd).max() > 1e-2 * jnp.abs(fd).max()


@pytest.mark.parametrize("overrides", [(), ("model=mlp",)], ids=["dit", "mlp"])
def test_end_time_none_is_the_zero_gap(overrides):
    """
    Without an end time a MeanFlow network still sees the gap, at h = 0: exactly its prediction at s = t.
    """
    policy = meanflow_policy(*overrides)
    obs, goal, x, t = draws(8)
    condition = policy.encode_condition(obs, goal)
    assert jnp.array_equal(
        policy._predict(condition, x, t), policy._predict(condition, x, t, t)
    )
    assert not jnp.array_equal(
        policy._predict(condition, x, t), policy._predict(condition, x, t, t + 0.3)
    )


def test_meanflow_policy_starts_at_the_instantaneous_velocity():
    """
    With only the gap layers at their initialization, the average velocity does not depend on s, and the one-step
    mean sampler is the one-step Euler sampler.
    """
    for overrides in ((), ("model=mlp",)):
        policy = new_policy(small_meanflow(*overrides))
        if overrides:
            assert jnp.all(policy.net.layers[0].kernel[-1] == 0)
        policy = perturbed(policy, skip=("gap_embedder.layers.1.",))
        if overrides:
            first = policy.net.layers[0].kernel
            first[...] = first[...].at[-1].set(0.0)
        obs, goal, noise, t = draws(8)
        condition = policy.encode_condition(obs, goal)
        assert jnp.array_equal(
            policy._predict(condition, noise, t),
            policy._predict(condition, noise, t, t + (1 - t) * 0.7),
        )
        one_step = {}
        for solver in ("mean", "euler"):
            sampler = instantiate(
                config(f"sampler.solver={solver}", "sampler.nfe=1").sampler
            )
            one_step[solver] = policy.sample_actions(obs, goal, noise, sampler=sampler)
        assert jnp.allclose(one_step["mean"], one_step["euler"], atol=1e-6)


def test_meanflow_adds_only_the_gap_parameters():
    """
    A MeanFlow DiT is a flow matching DiT plus the gap embedding, with the same initialization of every shared
    weight for the same key: the gap embedding is built last.
    """
    fm = flat(weights(new_policy(config())))
    mf = flat(weights(new_policy(config("flow=meanflow"))))
    assert set(mf) - set(fm) == {
        "gap_embedder.layers.0.weights",
        "gap_embedder.layers.1.kernel",
        "gap_embedder.layers.1.bias",
    }
    assert set(fm) <= set(mf) and all(jnp.array_equal(fm[k], mf[k]) for k in fm)


def test_meanflow_checkpoint_round_trip(tmp_path):
    """
    A MeanFlow checkpoint, with no second time sampler, rebuilds to the same network with each of its weights, and
    refuses weights that do not fit the saved config.
    """
    c = small_meanflow("flow/pairs=imf")
    resolved = OmegaConf.to_container(c, resolve=True)
    assert resolved["flow"]["second_time"] is None
    policy = meanflow_policy("flow/pairs=imf")
    raw = weights(new_policy(c))
    normalizer = unit_normalizer()
    path = tmp_path / CHECKPOINT
    saved = {"raw": raw, "ema0.999": weights(policy)}
    save_checkpoint(path, saved, MODEL_KWARGS, normalizer, resolved, 10)
    loaded, params, loaded_normalizer, loaded_config = load_policy(path, "ema0.999")
    assert isinstance(loaded.flow, MeanFlow) and loaded.flow.second_time is None
    assert loaded_config == resolved and read_checkpoint(path)["step"] == 10
    for name, value in vars(normalizer).items():
        assert np.array_equal(getattr(loaded_normalizer, name), value)
    assert_equal_trees(load_policy(path, "raw")[1], raw)
    obs, goal, noise, _ = draws(4)
    sampler = Sampler(**{**instantiate(c.sampler).__dict__, "solver": "mean", "nfe": 2})
    graphdef = nnx.split(loaded, nnx.Param)[0]
    assert jnp.array_equal(
        nnx.merge(graphdef, params).sample_actions(obs, goal, noise, sampler=sampler),
        policy.sample_actions(obs, goal, noise, sampler=sampler),
    )
    other = tmp_path / "other.pkl"
    mlp = {"raw": weights(new_policy(small_meanflow("model=mlp")))}
    save_checkpoint(other, mlp, MODEL_KWARGS, normalizer, resolved, 10)
    with pytest.raises(ValueError, match="do not fit"):
        load_policy(other, "raw")


@pytest.mark.parametrize("overrides", [(), ("model=mlp",)], ids=["dit", "mlp"])
def test_the_network_sees_the_gap(overrides, monkeypatch):
    """
    The gap embedding (DiT) or the last input column of the first layer (MLP) receives h = s - t, and 0 without
    an end time.
    Note: NNX has no forward hooks, so a recording wrapper takes the place of the layer class's __call__ and records
    the inputs of that one layer.
    """
    policy = meanflow_policy(*overrides)
    seen = []
    layer = policy.net.layers[0] if overrides else policy.gap_embedder
    column = (lambda x: x[:, -1:]) if overrides else (lambda x: x)
    call = type(layer).__call__

    def recording(self, x, *args, **kwargs):
        """
        Records the input of the watched layer, then calls the layer.
        """
        if self is layer:
            seen.append(column(x))
        return call(self, x, *args, **kwargs)

    monkeypatch.setattr(type(layer), "__call__", recording)
    obs, goal, x, t = draws(8)
    t = 0.2 + 0.5 * t
    condition = policy.encode_condition(obs, goal)
    policy._predict(condition, x, t, t + 0.25)
    policy._predict(condition, x, t)
    assert len(seen) == 2
    assert jnp.allclose(seen[0], jnp.full_like(t, 0.25))
    assert jnp.array_equal(seen[1], jnp.zeros_like(t))


@pytest.mark.parametrize("overrides", [(), ("model=mlp",)], ids=["dit", "mlp"])
def test_flow_matching_policies_take_no_end_time(overrides):
    """
    A flow matching policy refuses an end time, also through the mean sampler past the config checks.
    """
    policy = new_policy(small(*overrides))
    obs, goal, x, t = draws(4)
    with pytest.raises(ValueError, match="no end_time"):
        policy._predict(policy.encode_condition(obs, goal), x, t, t + 0.1)
    sampler = instantiate(config("sampler=mean", "sampler.nfe=2").sampler)
    with pytest.raises(ValueError, match="no end_time"):
        policy.sample_actions(obs, goal, x, sampler=sampler)


def test_guided_mean_sampler():
    """
    Guidance mixes the conditional and the null average velocity of one batched pass, s included.
    """
    policy = meanflow_policy()
    obs, goal, noise, _ = draws(4)
    sampler = instantiate(
        config("sampler=mean", "sampler.nfe=1", "sampler.cfg_scale=1.5").sampler
    )
    start, end = jnp.zeros((4, 1)), jnp.ones((4, 1))
    conditional = policy._predict(policy.encode_condition(obs, goal), noise, start, end)
    null = policy._predict(policy.null_condition(4), noise, start, end)
    expected = noise + (-0.5 * null + 1.5 * conditional)
    assert jnp.allclose(
        policy.sample_actions(obs, goal, noise, sampler=sampler), expected, atol=1e-5
    )


def test_evaluation_scores_every_episode():
    """
    A lock-step evaluation returns each episode's success and maximum reward, and their averages, for every
    trained seed: a leading seeds axis of 2 gives two Rollouts, equal for equal weights since every episode draws
    its noise from its own streams.
    """
    c = small("model=mlp")
    policy = new_policy(c)
    sampler = instantiate(c.sampler)
    query = make_query(nnx.split(policy, nnx.Param)[0], sampler)
    kwargs = dict(
        seeds=np.array([0, 1]),
        obs_horizon=2,
        exec_horizon=8,
        sampler=sampler,
        frame=c.data.frame,
        action_history=c.data.action_history,
        chunk_shape=(16, 2),
    )
    params = weights(policy)
    (rollouts,) = evaluate_policy(
        query, jax.tree.map(lambda x: x[None], params), unit_normalizer(), **kwargs
    )
    assert rollouts.seeds.tolist() == [0, 1] and rollouts.exec_horizon == 8
    assert rollouts.success.dtype == bool and rollouts.success.shape == (2,)
    assert rollouts.max_reward.shape == (2,) and np.all(rollouts.max_reward >= 0)
    summary = rollouts.summary()
    assert set(summary) == {"success_rate", "mean_reward"}
    assert summary["mean_reward"] == pytest.approx(rollouts.max_reward.mean())
    pair = evaluate_policy(
        query,
        jax.tree.map(lambda x: jnp.stack((x, x)), params),
        unit_normalizer(),
        **kwargs,
    )
    assert len(pair) == 2
    for run in pair:
        assert run.seeds.tolist() == [0, 1]
        assert np.array_equal(run.success, pair[0].success)
        assert np.array_equal(run.max_reward, pair[0].max_reward)
    assert np.array_equal(pair[0].success, rollouts.success)
    np.testing.assert_allclose(pair[0].max_reward, rollouts.max_reward, atol=1e-6)


def test_query_is_traced_once(monkeypatch):
    """
    One query serves the raw and the averaged weights at every execution horizon: it is traced (so compiled) once
    for a number of episodes, as at training's late evaluations.
    Note: the query rebuilds the policy once per trace, so counting nnx.merge counts the traces.
    """
    merge, traces = nnx.merge, []

    def counting_merge(*args, **kwargs):
        """
        Counts the policy's rebuilds.
        """
        traces.append(None)
        return merge(*args, **kwargs)

    c = small("model=mlp")
    policy = new_policy(c)
    sampler = instantiate(c.sampler)
    query = make_query(nnx.split(policy, nnx.Param)[0], sampler)
    monkeypatch.setattr(nnx, "merge", counting_merge)
    raw = jax.tree.map(lambda x: x[None], weights(policy))
    averaged = jax.tree.map(lambda x: x + 1e-3, raw)
    for params in (raw, averaged):
        for exec_horizon in (8, 4):
            evaluate_policy(
                query,
                params,
                unit_normalizer(),
                seeds=np.array([0, 1]),
                obs_horizon=2,
                exec_horizon=exec_horizon,
                sampler=sampler,
                frame=c.data.frame,
                action_history=c.data.action_history,
                chunk_shape=(16, 2),
            )
    assert len(traces) == 1


def tiny(*overrides: str) -> DictConfig:
    """
    Composes a short two-seed job on the Push-T dataset: a small MLP, 200 steps in windows of 100, an evaluation of
    two episodes after each window, W&B disabled.
    """
    return config(
        "model=mlp",
        "model.hidden_dims=[32,32]",
        "seed=7",
        "num_seeds=2",
        "train.steps=200",
        "log.train_interval=100",
        "log.print_interval=100",
        "eval.interval=100",
        "eval.episodes=2",
        f"data_dir={train.ROOT / 'data'}",
        "wandb_mode=disabled",
        *overrides,
    )


def job_loggers(job_dir: Path, c: DictConfig) -> list[Logger]:
    """
    Creates one logger per seed of a job, as train.main does.
    """
    return [
        Logger(
            job_dir / f"seed_{seed}",
            project=c.wandb_project,
            mode=c.wandb_mode,
            group=c.name,
            name=f"base/seed_{seed}",
            total_steps=c.train.steps,
            print_interval=c.log.print_interval if i == 0 else 0,
        )
        for i, seed in enumerate(c.seed + j for j in range(c.num_seeds))
    ]


@needs_data
def test_run_training_keeps_a_run_per_seed(tmp_path):
    """
    A two-seed job trains both seeds in one program and keeps each seed's run apart: its own directory with the
    job's Hydra files, a config.json of its seed and the seed block, log.csv rows of training and evaluation after
    each window and the late pool at the end, and a finished checkpoint of its own weights.
    """
    c = tiny()
    job = tmp_path / "job"
    (job / ".hydra").mkdir(parents=True)
    (job / ".hydra" / "config.yaml").write_text("seed: 7\n")
    loggers = job_loggers(job, c)
    try:
        train.run_training(c, loggers, "base")
    finally:
        for logger in loggers:
            logger.stop_logging_writer()
    runs = [job / f"seed_{seed}" for seed in (7, 8)]
    for seed, run in zip((7, 8), runs):
        run_config = json.loads((run / "config.json").read_text())
        assert run_config["seed"] == seed and run_config["arm"] == "base"
        assert run_config["num_seeds"] == 2
        assert run_config["info"]["seed_block"] == [7, 2]
        assert (run / ".hydra" / "config.yaml").read_text() == "seed: 7\n"
        rows = read_csv(run / "log.csv")
        assert [row["step"] for row in rows] == ["100", "100", "200", "200", "200"]
        # An evaluation's row comes before the training row of its step; the late pool closes the run.
        kinds = [
            "eval" if row[SUCCESS] else "train" if row[TRAIN_LOSS] else "late"
            for row in rows
        ]
        assert kinds == ["eval", "train", "eval", "train", "late"]
        assert rows[-1][LATE_SUCCESS] != ""
        assert all(math.isfinite(float(rows[i][TRAIN_LOSS])) for i in (1, 3))
        assert float(rows[1][LEARNING_RATE]) == pytest.approx(c.train.lr)
        checkpoint = read_checkpoint(run / CHECKPOINT)
        assert checkpoint["step"] == 200 and checkpoint["config"]["seed"] == seed
        assert set(checkpoint["weights"]) == {"raw", "ema0.999"}
    assert finished_checkpoints(job) == [run / CHECKPOINT for run in runs]
    first, second = (
        read_checkpoint(run / CHECKPOINT)["weights"]["raw"] for run in runs
    )
    kernels = [w["net"]["layers"][0]["kernel"] for w in (first, second)]
    assert not np.array_equal(*kernels)


@needs_data
@pytest.mark.parametrize("diverging", [0, 1], ids=["first", "second"])
@pytest.mark.parametrize("when", ["loss", "last_update"])
def test_a_diverging_seed_stops_alone(tmp_path, monkeypatch, diverging, when):
    """
    A seed whose training loss or weights turn non-finite stops alone: its run fails (exit code 1) without rows or
    a checkpoint, the other seed trains, evaluates and saves finite weights to the last step and its run finishes
    (exit code 0), the console summary passes to it, and the job then raises FloatingPointError.
    Note: the window is wrapped to turn one seed's weights to NaN, before every window (its loss turns non-finite)
    or after it (the window's losses stay finite, so only the weights tell; the evaluation must not see them).
    """
    window = train.make_train_window

    def poison(state, index):
        """
        Turns one seed's weights to NaN.
        """
        return state._replace(
            params=jax.tree.map(lambda x: x.at[index].set(jnp.nan), state.params)
        )

    def poisoned(*args, **kwargs):
        """
        Builds train.py's window with one seed's weights turned to NaN.
        """
        run_window = window(*args, **kwargs)

        def run(state, windows, keys, first_step):
            """
            Poisons the seed before or after the window.
            """
            if when == "loss":
                return run_window(poison(state, diverging), windows, keys, first_step)
            state, means = run_window(state, windows, keys, first_step)
            # As if the window's last update had diverged: the weights and their check, not the losses.
            state = poison(state, diverging)
            return state, {**means, "finite": jax.vmap(train.is_finite)(state)}

        return jax.jit(run)

    exits = {}
    stop = Logger.stop_logging_writer

    def recording_stop(self, exit_code=0):
        """
        Records the exit code each open run finishes with.
        """
        if self.running:
            exits[self.log_dir.name] = exit_code
        stop(self, exit_code)

    monkeypatch.setattr(train, "make_train_window", poisoned)
    monkeypatch.setattr(Logger, "stop_logging_writer", recording_stop)
    c = tiny()
    job = tmp_path / "job"
    loggers = job_loggers(job, c)
    bad, good = (7, 8) if diverging == 0 else (8, 7)
    try:
        with pytest.raises(FloatingPointError, match=rf"seeds \[{bad}\]"):
            train.run_training(c, loggers, "base")
        assert not any(logger.running for logger in loggers)
    finally:
        for logger in loggers:
            logger.stop_logging_writer()
    assert exits == {f"seed_{bad}": 1, f"seed_{good}": 0}
    # The first seed prints the console summary; when it diverges the other takes over.
    assert loggers[1 - diverging].print_interval == c.log.print_interval
    assert loggers[diverging].print_interval == 0
    assert read_csv(job / f"seed_{bad}" / "log.csv") == []
    assert not (job / f"seed_{bad}" / CHECKPOINT).exists()
    rows = read_csv(job / f"seed_{good}" / "log.csv")
    assert [row["step"] for row in rows] == ["100", "100", "200", "200", "200"]
    assert finished_checkpoints(job) == [job / f"seed_{good}" / CHECKPOINT]
    saved = read_checkpoint(job / f"seed_{good}" / CHECKPOINT)["weights"]
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(saved))
