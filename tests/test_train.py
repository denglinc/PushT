"""
The training step of train.py under CUDA-graph capture, config checks, run directories and naming.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize
from hydra._internal.core_plugins.basic_sweeper import BasicSweeper
from hydra.core.override_parser.overrides_parser import OverridesParser
from hydra.errors import InstantiationException
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

import train
from data import Normalizer
from evaluation import (
    build_policy,
    evaluate_policy,
    finished_checkpoints,
    load_policy,
    save_checkpoint,
)
from flow import MeanFlow, Sampler

PRESETS = sorted(p.stem for p in (train.ROOT / "conf" / "experiment").glob("*.yaml"))

MODEL_KWARGS = {
    "obs_dim": 6,
    "goal_dim": 4,
    "action_dim": 2,
    "obs_horizon": 2,
    "pred_horizon": 16,
}


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


def test_arms_leave_out_the_seed_and_logging():
    """
    A run's arm, Hydra's override_dirname, keeps the overrides that change the experiment, sorted.
    """
    assert config(hydra_config=True).hydra.job.override_dirname == ""
    arm = config(
        "seed=3",
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
    Without WANDB_MODE runs log to W&B online; Hydra puts each run, sweep job and evaluation in its own directory.
    """
    monkeypatch.delenv("WANDB_MODE", raising=False)
    c = config(hydra_config=True)
    assert c.wandb_mode == "online"
    assert c.hydra.output_subdir == ".hydra"
    assert c.hydra.run.dir.startswith("runs/default/")
    assert c.hydra.run.dir.endswith("_seed_0")
    sweep = OmegaConf.to_container(c.hydra.sweep, resolve=False)
    assert sweep["dir"].startswith("runs/${name}/") and "${seed}" in sweep["subdir"]
    assert config("runs=runs/x", name="evaluation").wandb_mode == "online"
    monkeypatch.setenv("WANDB_MODE", "offline")
    assert config().wandb_mode == "offline"
    e = config("runs=runs/x", name="evaluation", hydra_config=True)
    assert e.hydra.run.dir.startswith("runs/x/evaluations/")


@pytest.mark.parametrize("preset", PRESETS)
def test_presets_pass_config_checks(preset):
    """
    Every job of every preset's sweep fits together.
    """
    params = config(f"experiment={preset}", hydra_config=True).hydra.sweeper.params
    jobs = [[]]
    if params:
        overrides = OverridesParser.create().parse_overrides(
            [f"{key}={value}" for key, value in params.items()]
        )
        jobs = BasicSweeper.split_arguments(overrides, None)[0]
    for job in jobs:
        train.check_config(config(f"experiment={preset}", *job))


def test_finished_checkpoints_are_those_at_the_last_step(tmp_path):
    """
    Evaluation scores only checkpoints whose step is the run's last.
    """
    for run, step in (("done", 10), ("interrupted", 5)):
        (tmp_path / run).mkdir()
        torch.save(
            {"step": step, "config": {"train": {"steps": 10}}},
            tmp_path / run / "checkpoint.pt",
        )
    assert finished_checkpoints(tmp_path) == [Path(tmp_path / "done/checkpoint.pt")]


def test_unsupported_combinations_fail_before_training():
    """
    check_config refuses settings that do not fit together.
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
        train.check_config(config("flow=meanflow", "train.precision=bf16"))
    with pytest.raises(ValueError):
        train.check_config(config("train.optimizer=adam"))
    with pytest.raises(ValueError):
        train.check_config(config("train.warmup_steps=10"))
    train.check_config(config("train.optimizer=adamw", "train.warmup_steps=10"))
    default = config()
    assert default.train.optimizer == "muon" and default.train.compile
    assert config("experiment=meanflow").train.optimizer == "adamw"
    with pytest.raises(ValueError):
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


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize(
    "overrides",
    [
        (),
        ("flow/path=gvp", "flow/time=logit_normal", "flow.recipe=data_mse"),
        ("flow.recipe=scaled_score_mse", "flow.reduction=example_mean"),
        ("model.conditioning=cross", "model.cond_dropout=0.1", "train.precision=bf16"),
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
        ("train.compile=true",),
        ("train.optimizer=adamw", "train.compile=true"),
        ("flow=meanflow", "sampler=mean", "train.compile=true"),
    ],
)
def test_captured_step_trains(overrides):
    """
    Policy, condition dropout, loss (eager, or compiled as train.py builds it), backward, gradient norm and clipping,
    train.py's optimizers and averaging, replayed from one CUDA graph: the loss and the objective's metrics are
    finite, the loss falls, every weight matrix moves (Muon's included), and the average moves.
    Note: MeanFlow's own loss moves with its target, so for MeanFlow it is the plain flow matching loss (loss_ref at
    s = t) on fixed draws that must fall.
    """
    small = (
        ("model.hidden_dims=[64,64]",)
        if "model=mlp" in overrides
        else ("model.depth=2", "model.dim=64")
    )
    c = config(
        "train.compile=false",
        *overrides,
        *small,
        "train.lr=1e-3",
        "train.muon_lr=1e-3",
    )
    train.check_config(c)
    device = torch.device("cuda")
    torch.manual_seed(0)
    model = build_policy(c, MODEL_KWARGS).to(device).train()
    ema = [p.detach().clone() for p in model.parameters()]
    params = list(model.parameters())
    _, optimizers = train.build_optimizers(model, c.train, device, use_graph=True)
    obs, goal = torch.randn(64, 2, 6, device=device), torch.randn(64, 4, device=device)
    z, weight = torch.randn(64, 16, 2, device=device), torch.ones(64, 16, device=device)
    loss_sum = torch.zeros((), device=device)
    norm_sum = torch.zeros((), device=device)
    metric_sums = {
        name: torch.zeros((), device=device) for name in model.flow.metric_names
    }
    g = torch.Generator(device=device).manual_seed(1)
    fixed_time = torch.rand(64, 1, device=device, generator=g)
    fixed_noise = torch.randn(64, 16, 2, device=device, generator=g)

    def fixed_loss() -> float:
        """
        Returns the plain flow matching loss at fixed flow times (s = t) and noise, in eval mode.
        """
        model.eval()
        with torch.no_grad():
            loss, metrics = model.compute_loss(
                obs,
                goal,
                z,
                weight,
                flow_time=fixed_time,
                end_time=fixed_time if model.mean_flow else None,
                noise=fixed_noise,
            )
        model.train()
        return (metrics["loss_ref"] if model.mean_flow else loss).item()

    before = fixed_loss()
    if c.train.compile:
        torch.compiler.reset()
    loss_fn = train.build_loss(model, c.train.compile)

    def step():
        """
        Runs one training step of train.py's shape.
        """
        with torch.autocast(
            "cuda",
            dtype=torch.bfloat16,
            enabled=c.train.precision == "bf16",
            cache_enabled=False,
        ):
            loss, metrics = loss_fn(obs, goal, z, weight)
        loss.backward()
        grads = [p.grad for p in params if p.grad is not None]
        if c.train.grad_clip > 0:
            norm = torch.nn.utils.clip_grad_norm_(
                params, c.train.grad_clip, foreach=True
            )
        else:
            norm = torch.nn.utils.get_total_norm(grads, foreach=True)
        for optimizer in optimizers:
            optimizer.step()
        loss_sum.add_(loss.detach())
        norm_sum.add_(norm)
        for name, value in metrics.items():
            metric_sums[name].add_(value)
        torch._foreach_lerp_(ema, params, 0.01)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            model.zero_grad(set_to_none=True)
            step()
    torch.cuda.current_stream().wait_stream(stream)
    model.zero_grad(set_to_none=True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    groups = model.parameter_groups()
    matrices_before = [p.detach().clone() for p in groups.hidden + groups.boundary]
    losses = []
    for _ in range(4):
        for _ in range(50):
            graph.replay()
        losses.append(loss_sum.item() / 50)
        loss_sum.zero_()
    assert all(math.isfinite(v) for v in losses)
    assert all(math.isfinite(total.item()) for total in metric_sums.values())
    if model.mean_flow:
        assert fixed_loss() < before
    else:
        assert losses[-1] < losses[0]
    assert math.isfinite(norm_sum.item()) and norm_sum.item() > 0
    assert not torch.equal(ema[0], params[0].detach())
    assert all(
        not torch.equal(before, p.detach())
        for before, p in zip(matrices_before, groups.hidden + groups.boundary)
    )


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
def test_parameter_groups_split_the_parameters_by_role(overrides):
    """
    hidden, boundary and vectors partition the parameters: the weight matrices of the linear layers, those of the
    input embeddings and the output head among them, and everything else (biases, positions, null condition, time
    frequencies).
    """
    model = build_policy(small(*overrides), MODEL_KWARGS)
    groups = model.parameter_groups()
    ids = [[id(p) for p in group] for group in groups]
    assert sorted(i for group in ids for i in group) == sorted(
        id(p) for p in model.parameters()
    )
    linear = {id(m.weight) for m in model.modules() if isinstance(m, torch.nn.Linear)}
    assert set(ids[0]) | set(ids[1]) == linear
    assert set(ids[1]) == {id(layer.weight) for layer in model.boundary_layers()}
    assert all(p.ndim == 2 for p in groups.hidden + groups.boundary)
    if "model=mlp" in overrides:
        assert len(groups.boundary) == 2 and id(model.null) in ids[2]
    else:
        assert id(model.head.weight) in ids[1]
        assert id(model.blocks[0].attn.qkv.weight) in ids[0]
        assert id(model.blocks[0].ada_ln[1].weight) in ids[0]
        for vector in (
            model.time_embedder.weights,
            model.action_pos,
            model.null_tokens,
        ):
            assert id(vector) in ids[2]


@pytest.mark.parametrize("optimizer", ["adamw", "muon"])
def test_weight_decay_acts_on_the_weight_matrices_only(optimizer):
    """
    With zero gradients a step only decays: every weight matrix shrinks by 1 - lr * weight_decay (Muon's at its own
    lr), every vector stays bitwise as it was.
    """
    c = small(
        f"train.optimizer={optimizer}", "train.weight_decay=0.1", "train.muon_lr=0.02"
    )
    model = build_policy(c, MODEL_KWARGS)
    groups = model.parameter_groups()
    before = [p.detach().clone() for p in model.parameters()]
    for p in model.parameters():
        p.grad = torch.zeros_like(p)
    _, optimizers = train.build_optimizers(
        model, c.train, torch.device("cpu"), use_graph=False
    )
    for opt in optimizers:
        opt.step()
    hidden_lr = c.train.muon_lr if optimizer == "muon" else c.train.lr
    shrink = {id(p): 1 - hidden_lr * c.train.weight_decay for p in groups.hidden}
    shrink |= {id(p): 1 - c.train.lr * c.train.weight_decay for p in groups.boundary}
    for old, p in zip(before, model.parameters()):
        if id(p) in shrink:
            torch.testing.assert_close(
                p.detach(), old * shrink[id(p)], rtol=1e-6, atol=0
            )
        else:
            assert torch.equal(p.detach(), old)


def test_adamw_at_zero_weight_decay_is_adam():
    """
    At weight_decay 0 train.py's AdamW, split into matrices and vectors, steps every parameter bit for bit as torch's
    Adam over all of them.
    """
    c = small("train.optimizer=adamw", "train.weight_decay=0.0")
    torch.manual_seed(0)
    model, reference = build_policy(c, MODEL_KWARGS), build_policy(c, MODEL_KWARGS)
    reference.load_state_dict(model.state_dict())
    _, optimizers = train.build_optimizers(
        model, c.train, torch.device("cpu"), use_graph=False
    )
    adam = torch.optim.Adam(
        reference.parameters(),
        lr=c.train.lr,
        betas=tuple(c.train.betas),
        eps=c.train.eps,
    )
    g = torch.Generator().manual_seed(1)
    for _ in range(5):
        for p, q in zip(model.parameters(), reference.parameters()):
            p.grad = torch.randn(p.shape, generator=g)
            q.grad = p.grad.clone()
        for optimizer in optimizers:
            optimizer.step()
        adam.step()
    assert all(
        torch.equal(p, q) for p, q in zip(model.parameters(), reference.parameters())
    )


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_adamw_groups_share_one_learning_rate_tensor():
    """
    Under the graph AdamW's groups share one lr tensor, so the schedule's fill_ reaches every group; without it
    the lr is a number per group.
    """
    c = small("train.optimizer=adamw", "train.weight_decay=0.1")
    model = build_policy(c, MODEL_KWARGS).cuda()
    adamw, _ = train.build_optimizers(
        model, c.train, torch.device("cuda"), use_graph=True
    )
    first, second = (group["lr"] for group in adamw.param_groups)
    assert isinstance(first, torch.Tensor) and first is second
    adamw, _ = train.build_optimizers(
        model, c.train, torch.device("cuda"), use_graph=False
    )
    assert all(group["lr"] == c.train.lr for group in adamw.param_groups)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize(
    "overrides",
    [
        ("model.cond_dropout=0.1",),
        ("model.cond_dropout=0.0",),
        ("model.cond_dropout=0.0", "flow/time=logit_normal"),
    ],
    ids=["dropout", "no_dropout", "logit_normal"],
)
def test_compiled_loss_draws_like_eager(overrides):
    """
    The compiled loss (build_loss) draws the flow times, noise and condition dropout the eager one draws: from the
    same generator state it gives the same loss, to rounding, and leaves the same state. Without dropout the dropout
    uniforms are still drawn, so the compiled loss must keep that draw.
    Note: every weight is perturbed first, so that the zero-initialized output layers do not hide the network (and
    with it the dropout mask) from the loss.
    """
    torch.manual_seed(0)
    model = build_policy(small(*overrides), MODEL_KWARGS).cuda().train()
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.05 * torch.randn_like(p))
    obs, goal = torch.randn(64, 2, 6, device="cuda"), torch.randn(64, 4, device="cuda")
    z, weight = torch.randn(64, 16, 2, device="cuda"), torch.ones(64, 16, device="cuda")
    torch.compiler.reset()
    compiled = train.build_loss(model, use_compile=True)
    torch.cuda.manual_seed(1)
    eager_loss, _ = model.compute_loss(obs, goal, z, weight)
    eager_state = torch.cuda.get_rng_state()
    torch.cuda.manual_seed(1)
    compiled_loss, _ = compiled(obs, goal, z, weight)
    assert torch.equal(torch.cuda.get_rng_state(), eager_state)
    torch.testing.assert_close(compiled_loss, eager_loss, rtol=1e-5, atol=0)


def test_condition_dropout_draws_at_every_probability():
    """
    Training draws the dropout uniforms whatever cond_dropout is, so runs that differ only in it share a seed's flow
    times, noise and batch order; at cond_dropout = 0 every example keeps its condition.
    """
    obs, goal = torch.randn(8, 2, 6), torch.randn(8, 4)
    z, weight = torch.randn(8, 16, 2), torch.ones(8, 16)
    states = []
    for p in (0.0, 0.1):
        torch.manual_seed(0)
        model = build_policy(small(f"model.cond_dropout={p}"), MODEL_KWARGS).train()
        torch.manual_seed(1)
        model.compute_loss(obs, goal, z, weight)
        states.append(torch.get_rng_state())
    assert torch.equal(states[0], states[1])
    torch.manual_seed(0)
    model = build_policy(small("model.cond_dropout=0.0"), MODEL_KWARGS).train()
    assert torch.equal(model.condition(obs, goal), model.encode_condition(obs, goal))


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


def unit_normalizer() -> Normalizer:
    """
    Returns statistics that leave the features as they are.
    """
    ones = np.ones
    return Normalizer(
        ones((2, 6)), ones((2, 6)), ones(4), ones(4), ones((16, 2)), ones((16, 2))
    )


def meanflow_policy(*overrides: str, double: bool = False):
    """
    Builds a small MeanFlow policy with every parameter perturbed, zero-initialized layers included, so that its
    output and derivatives are not trivially zero.
    """
    torch.manual_seed(0)
    model = build_policy(small_meanflow(*overrides), MODEL_KWARGS).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.05 * torch.randn_like(p))
    return model.double() if double else model


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
    """
    model = meanflow_policy(*overrides, double=True)
    g = torch.Generator().manual_seed(1)
    obs = torch.randn(8, 2, 6, generator=g, dtype=torch.float64)
    goal = torch.randn(8, 4, generator=g, dtype=torch.float64)
    x = torch.randn(8, 16, 2, generator=g, dtype=torch.float64)
    v = torch.randn(8, 16, 2, generator=g, dtype=torch.float64)
    t = 0.2 + 0.5 * torch.rand(8, 1, generator=g, dtype=torch.float64)
    s = t + 0.25
    condition = model.encode_condition(obs, goal)

    def predict(xx, tt, ss=None):
        """
        The policy's network for the fixed condition.
        """
        return model._predict(condition, xx, tt, ss)

    with torch.no_grad():
        average, derivative = MeanFlow.average_velocity(predict, x, t, s, v)
        h = 1e-6
        fd = (predict(x + h * v, t + h, s) - predict(x - h * v, t - h, s)) / (2 * h)
        # The pass runs on the math attention kernel, which matches the default one up to rounding.
        assert torch.allclose(average, predict(x, t, s), rtol=1e-10, atol=1e-12)
        assert (derivative - fd).abs().max() <= 1e-6 * fd.abs().max()
        gap = s - t
        _, closed = MeanFlow.average_velocity(
            lambda xx, tt, ss: predict(xx, tt, tt + gap), x, t, s, v
        )
        assert (closed - fd).abs().max() > 1e-2 * fd.abs().max()


@pytest.mark.parametrize("overrides", [(), ("model=mlp",)], ids=["dit", "mlp"])
def test_end_time_none_is_the_zero_gap(overrides):
    """
    Without an end time a MeanFlow network still sees the gap, at h = 0: exactly its prediction at s = t.
    """
    model = meanflow_policy(*overrides)
    obs, goal, x = torch.randn(8, 2, 6), torch.randn(8, 4), torch.randn(8, 16, 2)
    t = torch.rand(8, 1)
    condition = model.encode_condition(obs, goal)
    assert torch.equal(
        model._predict(condition, x, t), model._predict(condition, x, t, t)
    )
    assert not torch.equal(
        model._predict(condition, x, t), model._predict(condition, x, t, t + 0.3)
    )


def test_meanflow_policy_starts_at_the_instantaneous_velocity():
    """
    With only the gap layers at their initialization, the average velocity does not depend on s, and the one-step
    mean sampler is the one-step Euler sampler.
    """
    for overrides in ((), ("model=mlp",)):
        torch.manual_seed(0)
        model = build_policy(small_meanflow(*overrides), MODEL_KWARGS).eval()
        with torch.no_grad():
            if overrides:
                assert torch.all(model.net[0].weight[:, -1] == 0)
            for name, p in model.named_parameters():
                if name.startswith("gap_embedder.1"):
                    continue
                change = 0.05 * torch.randn_like(p)
                if name == "net.0.weight":
                    change[:, -1] = 0
                p.add_(change)
        obs, goal, noise = (
            torch.randn(8, 2, 6),
            torch.randn(8, 4),
            torch.randn(8, 16, 2),
        )
        condition = model.encode_condition(obs, goal)
        t = torch.rand(8, 1)
        assert torch.equal(
            model._predict(condition, noise, t),
            model._predict(condition, noise, t, t + (1 - t) * 0.7),
        )
        one_step = {}
        for solver in ("mean", "euler"):
            sampler = instantiate(
                config(f"sampler.solver={solver}", "sampler.nfe=1").sampler
            )
            one_step[solver] = model.sample_actions(obs, goal, noise, sampler=sampler)
        assert torch.allclose(one_step["mean"], one_step["euler"], atol=1e-6)


def test_meanflow_adds_only_the_gap_parameters():
    """
    A MeanFlow DiT is a flow matching DiT plus the gap embedding, with the same initialization of every shared
    weight for the same seed; its checkpoint rebuilds from the saved config.
    """
    torch.manual_seed(0)
    fm = build_policy(config(), MODEL_KWARGS).state_dict()
    torch.manual_seed(0)
    mf = build_policy(config("flow=meanflow"), MODEL_KWARGS).state_dict()
    assert set(mf) - set(fm) == {
        "gap_embedder.0.weights",
        "gap_embedder.1.weight",
        "gap_embedder.1.bias",
    }
    assert set(fm) <= set(mf) and all(torch.equal(fm[k], mf[k]) for k in fm)


def test_meanflow_checkpoint_round_trip(tmp_path):
    """
    A MeanFlow checkpoint, with no second time sampler, rebuilds to the same network.
    """
    c = small_meanflow("flow/pairs=imf")
    resolved = OmegaConf.to_container(c, resolve=True)
    assert resolved["flow"]["second_time"] is None
    model = meanflow_policy("flow/pairs=imf")
    normalizer = unit_normalizer()
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, MODEL_KWARGS, normalizer, resolved, {0.999: model}, 10)
    loaded, _, _ = load_policy(path, torch.device("cpu"), 0.999)
    assert isinstance(loaded.flow, MeanFlow) and loaded.flow.second_time is None
    obs, goal, noise = torch.randn(4, 2, 6), torch.randn(4, 4), torch.randn(4, 16, 2)
    sampler = Sampler(**{**instantiate(c.sampler).__dict__, "solver": "mean", "nfe": 2})
    assert torch.equal(
        loaded.sample_actions(obs, goal, noise, sampler=sampler),
        model.sample_actions(obs, goal, noise, sampler=sampler),
    )


@pytest.mark.parametrize("overrides", [(), ("model=mlp",)], ids=["dit", "mlp"])
def test_the_network_sees_the_gap(overrides):
    """
    The gap embedding (DiT) or the last input column (MLP) receives h = s - t, and 0 without an end time.
    """
    model = meanflow_policy(*overrides)
    seen = []
    layer = model.net[0] if overrides else model.gap_embedder
    column = (lambda x: x[:, -1:]) if overrides else (lambda x: x)
    layer.register_forward_pre_hook(lambda module, args: seen.append(column(args[0])))
    obs, goal, x = torch.randn(8, 2, 6), torch.randn(8, 4), torch.randn(8, 16, 2)
    t = 0.2 + 0.5 * torch.rand(8, 1)
    condition = model.encode_condition(obs, goal)
    model._predict(condition, x, t, t + 0.25)
    model._predict(condition, x, t)
    assert torch.allclose(seen[0], torch.full_like(t, 0.25))
    assert torch.equal(seen[1], torch.zeros_like(t))


@pytest.mark.parametrize("overrides", [(), ("model=mlp",)], ids=["dit", "mlp"])
def test_flow_matching_policies_take_no_end_time(overrides):
    """
    A flow matching policy refuses an end time, also through the mean sampler past the config checks.
    """
    model = build_policy(small(*overrides), MODEL_KWARGS).eval()
    obs, goal, x = torch.randn(4, 2, 6), torch.randn(4, 4), torch.randn(4, 16, 2)
    t = torch.rand(4, 1)
    with pytest.raises(ValueError, match="no end_time"):
        model._predict(model.encode_condition(obs, goal), x, t, t + 0.1)
    sampler = instantiate(config("sampler=mean", "sampler.nfe=2").sampler)
    with pytest.raises(ValueError, match="no end_time"):
        model.sample_actions(obs, goal, x, sampler=sampler)


def test_guided_mean_sampler():
    """
    Guidance mixes the conditional and the null average velocity of one batched pass, s included.
    """
    model = meanflow_policy()
    obs, goal, noise = torch.randn(4, 2, 6), torch.randn(4, 4), torch.randn(4, 16, 2)
    sampler = instantiate(
        config("sampler=mean", "sampler.nfe=1", "sampler.cfg_scale=1.5").sampler
    )
    start, end = torch.zeros(4, 1), torch.ones(4, 1)
    conditional = model._predict(model.encode_condition(obs, goal), noise, start, end)
    null = model._predict(model.null_condition(4), noise, start, end)
    expected = noise + (-0.5 * null + 1.5 * conditional)
    assert torch.allclose(
        model.sample_actions(obs, goal, noise, sampler=sampler), expected, atol=1e-5
    )


def test_evaluation_scores_every_episode():
    """
    A lock-step evaluation returns each episode's success and maximum reward, and their averages.
    """
    c = small("model=mlp")
    model = build_policy(c, MODEL_KWARGS)
    rollouts = evaluate_policy(
        model,
        unit_normalizer(),
        torch.device("cpu"),
        seeds=np.array([0, 1]),
        obs_horizon=2,
        exec_horizon=8,
        sampler=instantiate(c.sampler),
        frame=c.data.frame,
        action_history=c.data.action_history,
    )
    assert rollouts.seeds.tolist() == [0, 1] and rollouts.exec_horizon == 8
    assert rollouts.success.dtype == bool and rollouts.success.shape == (2,)
    assert rollouts.max_reward.shape == (2,) and np.all(rollouts.max_reward >= 0)
    summary = rollouts.summary()
    assert set(summary) == {"success_rate", "mean_reward"}
    assert summary["mean_reward"] == pytest.approx(rollouts.max_reward.mean())
