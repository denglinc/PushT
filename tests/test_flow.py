"""flow.py on a Gaussian toy where every quantity is analytic.

Data z ~ N(m, s^2) per coordinate and eps ~ N(0, 1): x_t ~ N(alpha m, V) with V = alpha^2 s^2 + beta^2, and
    E[z | x] = m + alpha s^2 (x - alpha m) / V,   E[eps | x] = beta (x - alpha m) / V,
    v = d_alpha E[z | x] + d_beta E[eps | x],     q = -E[eps | x],     score = -(x - alpha m) / V.
The probability-flow map is affine, so from t = 0 every sample has an exact endpoint x_1 = m + s x_0, and the exact
MeanFlow average velocity over [t, s] is a quadrature of v along it.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from hydra import compose, initialize
from hydra.utils import instantiate

from flow import (
    RECIPES,
    DataPrediction,
    FlowMatching,
    GVPPath,
    LearnedFlow,
    LinearPath,
    LogitNormalTime,
    MeanFlow,
    ProbabilityPath,
    Sampler,
    ScaledScorePrediction,
    UniformTime,
    VelocityPrediction,
)

PATHS = [LinearPath(), GVPPath()]
PREDICTIONS = [VelocityPrediction(), DataPrediction(), ScaledScorePrediction()]
# Per-coordinate data mean and std of a (H, d) = (2, 2) chunk, so broadcasting mistakes show.
M = torch.tensor([[-1.0, 0.5], [2.0, 0.0]], dtype=torch.float64)
S = torch.tensor([[0.1, 0.5], [2.0, 1.0]], dtype=torch.float64)


def sampler(**changes) -> Sampler:
    """
    Returns the default sampler with some settings changed.
    """
    settings = {
        "solver": "euler",
        "nfe": 10,
        "grid": "uniform",
        "rho": 1.5,
        "start_time": None,
        "sde_eta": 0.1,
        "sde_end": 0.999,
        "cfg_scale": 1.0,
    }
    return Sampler(**{**settings, **changes})


def posterior(path: ProbabilityPath, x: torch.Tensor, t: torch.Tensor):
    """E[z|x], E[eps|x], v and score of the toy at x (B, 2, 2), t broadcastable, in float64."""
    c = path(t.double())
    var = c.alpha**2 * S**2 + c.beta**2
    centered = x.double() - c.alpha * M
    data = M + c.alpha * S**2 * centered / var
    noise = c.beta * centered / var
    return data, noise, c.d_alpha * data + c.d_beta * noise, -centered / var


def exact_predict(path: ProbabilityPath, prediction):
    """The Bayes-optimal network for the toy: predict(x, t (B, 1)) -> its prediction (float32)."""

    def predict(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Returns the Bayes-optimal prediction at x (bs, 2, 2) and t (bs, 1).
        """
        data, noise, velocity, _ = posterior(path, x, t[:, :, None])
        if isinstance(prediction, VelocityPrediction):
            return velocity.float()
        return (data if isinstance(prediction, DataPrediction) else -noise).float()

    return predict


@pytest.mark.parametrize("path", PATHS, ids=["linear", "gvp"])
def test_path_coefficients(path):
    """
    Endpoints are exact in float32, D matches its definition and the derivatives match finite differences.
    """
    # Exact endpoints in float32 (cos(pi / 2) is not 0 in float32, so GVP writes beta as a sine).
    ends = torch.tensor([0.0, 1.0])
    assert path.alpha(ends).tolist() == [0.0, 1.0]
    assert path.beta(ends).tolist() == [1.0, 0.0]
    t = torch.linspace(0.01, 0.99, 99, dtype=torch.float64)
    c = path(t)
    assert torch.allclose(
        c.beta * c.d_alpha - c.alpha * c.d_beta, torch.full_like(t, c.D)
    )
    h = 1e-6
    for f, df in ((path.alpha, path.d_alpha), (path.beta, path.d_beta)):
        assert torch.allclose((f(t + h) - f(t - h)) / (2 * h), df(t), atol=1e-8)


@pytest.mark.parametrize("path", PATHS, ids=["linear", "gvp"])
def test_adapters_on_the_gaussian_posterior(path):
    """
    Every adapter maps the analytic posterior quantities onto each other, and guidance commutes with them.
    """
    torch.manual_seed(0)
    x = torch.randn(64, 2, 2, dtype=torch.float64) * 2
    t = torch.rand(64, 1, 1, dtype=torch.float64) * 0.9 + 0.05
    c = path(t)
    data, noise, velocity, score = posterior(path, x, t)
    assert torch.allclose(c.data_to_velocity(data, x), velocity)
    assert torch.allclose(c.scaled_score_to_velocity(-noise, x), velocity)
    assert torch.allclose(c.velocity_to_scaled_score(velocity, x), -noise)
    assert torch.allclose(c.velocity_to_score(velocity, x), score)
    # Guidance mixes network outputs; the adapters are affine with unit weight on the output, so mixing commutes.
    for w in (0.0, 1.5, 3.0):
        a, b = data, data + noise
        mixed = c.data_to_velocity((1 - w) * a + w * b, x)
        assert torch.allclose(
            mixed, (1 - w) * c.data_to_velocity(a, x) + w * c.data_to_velocity(b, x)
        )


def test_conditional_velocity_is_the_path_derivative():
    """
    u = d_alpha z + d_beta eps is the time derivative of x_t.
    """
    z, eps = torch.randn(2, 8, 2, 2, dtype=torch.float64)
    for path in PATHS:
        for t in (0.1, 0.5, 0.9):
            tt = torch.tensor(t, dtype=torch.float64)
            h = torch.tensor(1e-6, dtype=torch.float64)
            fd = (
                path(tt + h).interpolate(z, eps) - path(tt - h).interpolate(z, eps)
            ) / (2 * h)
            assert torch.allclose(fd, path(tt).velocity(z, eps), atol=1e-7)


def test_uniform_time_draws():
    """
    Uniform flow times are float32 draws on the support: t_min itself when u = 0, and never t = 1 on [0, 1].
    """
    t = UniformTime(support=(0.0, 1.0)).sample(1_000_000, torch.device("cpu"))
    assert t.dtype == torch.float32 and t.shape == (1_000_000, 1)
    assert 0.0 <= t.min() and t.max() < 1.0
    # On the CPU, torch.rand returns multiples of 2^-24.
    assert torch.all((t.double() * 2**24) % 1 == 0)
    assert abs(t.mean().item() - 0.5) < 2e-3
    short = UniformTime(support=(0.0, 0.99))
    assert short.sample(1_000_000, torch.device("cpu")).max() <= 0.99
    # The smallest and largest values of torch.rand.
    edges = torch.tensor([[0.0], [1.0 - 2**-24]])
    assert short.icdf(edges)[0].item() == 0.0 and short.icdf(edges)[1].item() <= 0.99
    assert UniformTime(support=(0.2, 0.3)).icdf(edges)[0].item() == np.float32(0.2)


def test_time_sampler_validation():
    """
    Invalid supports and parameters are refused, and a logit-normal draws on the whole of [0, 1] only.
    """
    with pytest.raises(ValueError):
        UniformTime(support=(0.5, 0.5))
    with pytest.raises(ValueError):
        UniformTime(support=(-0.1, 1.0))
    with pytest.raises(ValueError):
        LogitNormalTime(mean=0.0, std=0.0, support=(0.0, 1.0))
    for support in [(0.0, 0.99), (0.001, 1.0)]:
        with pytest.raises(ValueError):
            LogitNormalTime(mean=0.0, std=1.0, support=support)
    # Configs and checkpoints store the support as a list.
    LogitNormalTime(mean=4.0, std=1.6, support=[0.0, 1.0])
    LogitNormalTime(mean=0.0, std=3.3, support=(0.0, 1.0))


def logit_normal_cdf(mean: float, std: float):
    """The CDF Phi((logit t - mean) / std) of a logit-normal law, in float64."""
    return lambda t: torch.special.ndtr((torch.logit(t) - mean) / std)


@pytest.mark.parametrize(
    "time, cdf",
    [
        (UniformTime(support=(0.0, 1.0)), lambda t: t),
        (UniformTime(support=(0.001, 0.999)), lambda t: (t - 0.001) / 0.998),
        (UniformTime(support=(0.2, 0.3)), lambda t: (t - 0.2) / 0.1),
        (
            LogitNormalTime(mean=0.0, std=1.0, support=(0.0, 1.0)),
            logit_normal_cdf(0.0, 1.0),
        ),
        (
            LogitNormalTime(mean=0.4, std=1.0, support=(0.0, 1.0)),
            logit_normal_cdf(0.4, 1.0),
        ),
        (
            LogitNormalTime(mean=0.6, std=1.6, support=(0.0, 1.0)),
            logit_normal_cdf(0.6, 1.6),
        ),
        (
            LogitNormalTime(mean=4.0, std=1.6, support=(0.0, 1.0)),
            logit_normal_cdf(4.0, 1.6),
        ),
    ],
    ids=["u01", "u001_0999", "u02_03", "ln0_1", "ln04_1", "ln06_16", "ln4_16"],
)
def test_draws_follow_the_law(time, cdf):
    """Kolmogorov-Smirnov against the law's CDF on its support."""
    n = 200_000
    torch.manual_seed(0)
    t = torch.sort(time.sample(n, torch.device("cpu")).flatten().double()).values
    low, high = time.support
    assert low <= t[0] and t[-1] <= high
    law = cdf(t)
    empirical = torch.arange(1, n + 1, dtype=torch.float64) / n
    assert torch.maximum(
        empirical - law, law - (empirical - 1 / n)
    ).max() < 1.95 / math.sqrt(n)


TIME_SAMPLERS = [
    UniformTime(support=(0.0, 1.0)),
    UniformTime(support=(0.0, 0.99)),
    LogitNormalTime(mean=0.4, std=1.0, support=(0.0, 1.0)),
    LogitNormalTime(mean=4.0, std=1.6, support=(0.0, 1.0)),
]


def test_time_draws_use_one_uniform_per_example():
    """
    Every sampler draws t = icdf(u) from one torch.rand per example, so it leaves the generator where torch.rand
    does and a sweep over time laws keeps a seed's other draws.
    """
    for time in TIME_SAMPLERS:
        torch.manual_seed(3)
        t = time.sample(100, torch.device("cpu"))
        state = torch.get_rng_state()
        torch.manual_seed(3)
        u = torch.rand(100, 1)
        assert torch.equal(t, time.icdf(u)) and torch.equal(
            state, torch.get_rng_state()
        )


def test_time_extremes_are_finite():
    """
    The extreme uniforms map into [0, 1] without NaN; a logit-normal sends u = 0 to t = 0 exactly.
    """
    edges = torch.tensor([[0.0], [1.0 - 2**-24]])
    for time in TIME_SAMPLERS:
        t = time.icdf(edges)
        assert torch.isfinite(t).all() and 0.0 <= t.min() and t.max() <= 1.0
    assert TIME_SAMPLERS[2].icdf(torch.zeros(1, 1)).item() == 0.0


def linear_net(
    x: torch.Tensor, t: torch.Tensor, s: torch.Tensor | None = None
) -> torch.Tensor:
    """A fixed affine network of (x, t[, s]) for the endpoint tests; s omitted is t."""
    s = t if s is None else s
    return 0.5 * x + (0.3 * t - 0.2 * s + 0.1)[:, :, None]


@pytest.mark.parametrize("path", PATHS, ids=["linear", "gvp"])
def test_losses_are_finite_at_the_ends(path):
    """
    A draw can land on t = 0 (u = 0) or round to t = 1: every loss that may train there is finite. The velocity-space
    loss is finite at the end of its shortened support.
    """
    z, weight = torch.randn(4, 2, 2), torch.ones(4, 2)
    ends = torch.tensor([[0.0], [1.0], [0.0], [1.0]])
    for recipe in ["velocity_mse", "data_mse", "scaled_score_mse"]:
        flow = FlowMatching(
            path=path,
            time=UniformTime(support=(0.0, 1.0)),
            recipe=recipe,
            reduction="sum",
        )
        assert math.isfinite(flow.loss(linear_net, z, weight, flow_time=ends)[0].item())
    for tangent in ["conditional", "predicted"]:
        flow = meanflow(path=path, tangent=tangent)
        loss, metrics = flow.loss(
            linear_net, z, weight, flow_time=ends, end_time=torch.ones(4, 1)
        )
        assert math.isfinite(loss.item()) and all(
            math.isfinite(v.item()) for v in metrics.values()
        )
    short = UniformTime(support=(0.0, 0.99))
    flow = FlowMatching(
        path=path, time=short, recipe="data_output_velocity_loss", reduction="sum"
    )
    t_max = short.icdf(torch.full((4, 1), 1.0 - 2**-24))
    assert math.isfinite(flow.loss(linear_net, z, weight, flow_time=t_max)[0].item())


def replicate_draws(flow: FlowMatching, z: torch.Tensor, seed: int):
    """The flow times and noise that flow.loss draws after torch.manual_seed(seed), in its order."""
    torch.manual_seed(seed)
    t = flow.time.sample(z.shape[0], z.device)
    return t, torch.randn_like(z)


@pytest.mark.parametrize("recipe", sorted(RECIPES))
@pytest.mark.parametrize("path", PATHS, ids=["linear", "gvp"])
def test_loss_targets(recipe, path):
    """
    An oracle that predicts its own target has zero loss, and the network sees x_t and t of the right shape.
    """
    support = (0.001, 0.999) if recipe == "data_output_velocity_loss" else (0.0, 1.0)
    flow = FlowMatching(
        path=path, time=UniformTime(support=support), recipe=recipe, reduction="sum"
    )
    z = torch.randn(16, 2, 2)
    weight = torch.ones(16, 2)
    t, eps = replicate_draws(flow, z, seed=1)
    c = path(t[:, :, None])
    # The oracle that knows (z, eps) predicts its own target exactly: zero loss, whatever the space.
    oracle = flow.prediction.target(z, eps, c)
    seen = {}

    def predict(x, tt):
        """
        Records the network's inputs and returns the oracle's prediction.
        """
        seen["x"], seen["t"] = x, tt
        return oracle

    torch.manual_seed(1)
    loss, metrics = flow.loss(predict, z, weight)
    assert loss.item() < 1e-6 and metrics == {}
    assert torch.equal(seen["t"], t) and seen["t"].shape == (16, 1)
    assert torch.allclose(seen["x"], c.interpolate(z, eps))


def test_loss_reductions_by_hand():
    """
    Both reductions match a hand computation on a masked batch.
    """
    flow = FlowMatching(
        path=LinearPath(),
        time=UniformTime(support=(0.0, 1.0)),
        recipe="data_mse",
        reduction="sum",
    )
    z = torch.zeros(2, 3, 2)
    error = torch.tensor(
        [[[1.0, 1.0], [2.0, 0.0], [0.0, 3.0]], [[0.0, 2.0], [1.0, 0.0], [1.0, 1.0]]]
    )
    weight = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    # data_mse regresses z, so a prediction z + error leaves exactly this error.
    assert flow.loss(lambda x, t: z + error, z, weight)[0].item() == pytest.approx(5.0)
    flow.reduction = "example_mean"
    assert flow.loss(lambda x, t: z + error, z, weight)[0].item() == pytest.approx(1.75)


def test_velocity_loss_needs_a_support_away_from_the_singularity():
    """
    A velocity-space loss on a data prediction needs a support that excludes t = 1.
    """
    refused = [
        UniformTime(support=(0.0, 1.0)),
        LogitNormalTime(mean=0.0, std=1.0, support=(0.0, 1.0)),
        # Below 1 in float64, but float32 draws on it round to t = 1.
        UniformTime(support=(0.75, 1.0 - 2**-26)),
    ]
    for time in refused:
        with pytest.raises(ValueError):
            FlowMatching(
                path=LinearPath(),
                time=time,
                recipe="data_output_velocity_loss",
                reduction="sum",
            )
    FlowMatching(
        path=LinearPath(),
        time=UniformTime(support=(0.0, 0.99)),
        recipe="data_output_velocity_loss",
        reduction="sum",
    )


def test_time_grids():
    """
    The resolved grids have the expected points for every solver and grid.
    """
    euler = sampler().times(VelocityPrediction())
    assert euler[0] == 0.0 and euler[-1] == 1.0 and len(euler) == 11
    heun = sampler(solver="heun").times(VelocityPrediction())
    assert heun == torch.tensor([0, 0.2, 0.4, 0.6, 0.8, 1.0]).tolist()
    dense = sampler(grid="data_dense").times(VelocityPrediction())
    steps = torch.tensor(dense).diff()
    assert torch.all(steps[1:] < steps[:-1]) and dense[-1] == 1.0
    assert sampler().times(ScaledScorePrediction())[0] == pytest.approx(0.05)
    sde = sampler(solver="sde").times(VelocityPrediction())
    assert len(sde) == 11 and sde[-2] == pytest.approx(0.999) and sde[-1] == 1.0
    assert sampler(solver="sde").num_brownian == 9


def test_unsupported_samplers_raise():
    """
    Unsupported sampler settings and combinations raise before sampling.
    """
    with pytest.raises(ValueError):
        sampler(nfe=0)
    with pytest.raises(ValueError):
        sampler(solver="heun", nfe=9)
    with pytest.raises(ValueError):
        sampler(solver="sde", nfe=1)
    with pytest.raises(ValueError):
        sampler(solver="sde", grid="data_dense")
    with pytest.raises(ValueError):
        sampler(grid="data_dense", rho=0.5)
    # Heun's corrector evaluates t = 1, where a data prediction divides by beta = 0.
    with pytest.raises(ValueError):
        sampler(solver="heun").times(DataPrediction())
    with pytest.raises(ValueError):
        sampler(start_time=0.0).times(ScaledScorePrediction())


def solve(path, prediction, settings, x0, brownian=None):
    """
    Samples the toy's exact network with a sampler; returns the endpoints and the network calls.
    """
    flow = LearnedFlow(exact_predict(path, prediction), path, prediction)
    return settings.sample(flow, x0, brownian), flow.calls


@pytest.mark.parametrize("path", PATHS, ids=["linear", "gvp"])
@pytest.mark.parametrize("prediction", PREDICTIONS, ids=["velocity", "data", "score"])
@pytest.mark.parametrize(
    "settings,tolerance",
    [
        (sampler(nfe=1000), 1e-2),
        (sampler(nfe=1000, grid="data_dense"), 1e-2),
        (sampler(solver="heun", nfe=200), 2e-3),
    ],
    ids=["euler", "euler-dense", "heun"],
)
def test_ode_reaches_the_exact_endpoint(path, prediction, settings, tolerance):
    """
    Every solver, grid, path and prediction reaches the exact flow map's endpoint of every sample.
    """
    if isinstance(prediction, DataPrediction) and settings.solver == "heun":
        pytest.skip(
            "Heun evaluates t = 1, which a data prediction cannot (covered above)"
        )
    torch.manual_seed(0)
    x0 = torch.randn(256, 2, 2)
    x1, calls = solve(path, prediction, settings, x0)
    start = settings.times(prediction)[0]
    c = path.at(start)
    var = c.alpha**2 * S**2 + c.beta**2
    exact = M + S * (x0.double() - c.alpha * M) / var.sqrt()
    assert calls == settings.nfe
    assert torch.all((x1 - exact).abs() <= tolerance * (S * x0.abs() + M.abs()) + 1e-4)


def test_convergence_orders():
    """
    Euler converges at order 1 and Heun at order 2.
    """
    torch.manual_seed(0)
    x0 = torch.randn(256, 2, 2)
    exact = M + S * x0.double()

    def error(settings):
        """
        Returns the largest endpoint error of a sampler.
        """
        x1, _ = solve(LinearPath(), VelocityPrediction(), settings, x0)
        return (x1 - exact).abs().max().item()

    assert 0.4 < error(sampler(nfe=400)) / error(sampler(nfe=200)) < 0.6
    assert (
        0.15
        < error(sampler(solver="heun", nfe=160)) / error(sampler(solver="heun", nfe=80))
        < 0.35
    )


@pytest.mark.parametrize("path", PATHS, ids=["linear", "gvp"])
@pytest.mark.parametrize("recipe", ["velocity_mse", "data_mse", "scaled_score_mse"])
def test_sde_sampler_accepts_every_path_and_prediction(path, recipe):
    """
    The SDE sampler accepts a policy trained on any path with any prediction.
    """
    flow = FlowMatching(
        path=path,
        time=UniformTime(support=(0.0, 1.0)),
        recipe=recipe,
        reduction="sum",
    )
    sampler(solver="sde").check(flow)


@pytest.mark.parametrize("path", PATHS, ids=["linear", "gvp"])
@pytest.mark.parametrize("prediction", PREDICTIONS, ids=["velocity", "data", "score"])
@pytest.mark.parametrize("eta", [0.1, 1.0])
def test_sde_keeps_the_marginals(path, prediction, eta):
    """
    The SDE sampler ends with the data's mean and standard deviation, for every path and prediction, when it starts
    from the exact marginal at its first time (a scaled score starts at t > 0).
    """
    torch.manual_seed(0)
    settings = sampler(solver="sde", nfe=1000, sde_eta=eta)
    c = path.at(settings.times(prediction)[0])
    x0 = (
        c.alpha * M + (c.alpha**2 * S**2 + c.beta**2).sqrt() * torch.randn(20_000, 2, 2)
    ).float()
    brownian = torch.randn(20_000, settings.num_brownian, 2, 2)
    x1, calls = solve(path, prediction, settings, x0, brownian)
    assert calls == 1000
    assert torch.all((x1.double().mean(0) - M).abs() < 0.03 * S + 2e-3)
    assert torch.all((x1.double().std(0) - S).abs() < 0.03 * S + 2e-3)


@pytest.mark.parametrize("path", PATHS, ids=["linear", "gvp"])
@pytest.mark.parametrize("prediction", PREDICTIONS, ids=["velocity", "data", "score"])
def test_sde_drift_is_velocity_plus_half_g_squared_score(path, prediction):
    """
    The SDE's drift equals v + g^2 / 2 times the score, with g^2 = 2 eta alpha beta, for every path and prediction.
    """
    from flow import LangevinFlow

    flow = LearnedFlow(exact_predict(path, prediction), path, prediction)
    sde = LangevinFlow(flow, eta=0.7)
    x = torch.randn(32, 2, 2)
    for t in (0.1, 0.5, 0.95):
        v = flow.drift(x, t)
        c = path.at(t)
        expected = v + 0.5 * sde.diffusion(t) ** 2 * c.velocity_to_score(v, x)
        assert torch.allclose(sde.drift(x, t), expected, atol=1e-5)
        assert sde.diffusion(t) ** 2 == pytest.approx(2 * 0.7 * c.alpha * c.beta)


def test_sde_on_the_linear_path_keeps_its_closed_form():
    """
    On the linear path the general drift and diffusion are bit for bit v - eta t (x - t v) and
    sqrt(2 eta t (1 - t)), so linear-path SDE samples do not change.
    """
    from flow import LangevinFlow

    torch.manual_seed(0)
    flow = LearnedFlow(
        lambda x, t: torch.sin(3 * x) + t[:, :, None],
        LinearPath(),
        VelocityPrediction(),
    )
    sde = LangevinFlow(flow, eta=0.3)
    x = torch.randn(64, 16, 2)
    for t in sampler(solver="sde", nfe=10).times(VelocityPrediction())[:-1]:
        v = flow.drift(x, t)
        assert torch.equal(sde.drift(x, t), v - 0.3 * t * (x - t * v))
        assert sde.diffusion(t) == math.sqrt(2 * 0.3 * t * (1 - t))


def test_data_prediction_last_step_lands_on_it():
    """
    With a data prediction, the last Euler step lands on the last prediction.
    """
    path, prediction = LinearPath(), DataPrediction()
    settings = sampler(nfe=10)
    x0 = torch.randn(64, 2, 2)
    final_prediction = {}

    def predict(x, t):
        """
        Returns the exact prediction and keeps the last one.
        """
        y = exact_predict(path, prediction)(x, t)
        final_prediction["y"] = y
        return y

    x1 = settings.sample(LearnedFlow(predict, path, prediction), x0)
    assert torch.allclose(x1, final_prediction["y"], atol=1e-6)


def test_sampler_errors_on_bad_inputs():
    """
    Non-finite samples and mismatched Brownian draws raise.
    """
    path, prediction = LinearPath(), VelocityPrediction()
    nan_flow = LearnedFlow(lambda x, t: torch.full_like(x, math.nan), path, prediction)
    with pytest.raises(FloatingPointError):
        sampler().sample(nan_flow, torch.randn(4, 2, 2))
    flow = LearnedFlow(exact_predict(path, prediction), path, prediction)
    with pytest.raises(ValueError):
        sampler(solver="sde").sample(flow, torch.randn(4, 2, 2))  # no Brownian draws
    with pytest.raises(ValueError):
        sampler().sample(flow, torch.randn(4, 2, 2), torch.randn(4, 9, 2, 2))


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("recipe", sorted(RECIPES))
@pytest.mark.parametrize("time", ["uniform", "logit_normal"])
def test_loss_is_graph_capturable(recipe, time):
    """
    The loss of every recipe and time sampler is captured in a CUDA graph without host syncs.
    """
    if time == "logit_normal" and recipe == "data_output_velocity_loss":
        pytest.skip("A logit-normal time draws on [0, 1], which this recipe refuses.")
    device = torch.device("cuda")
    support = (0.001, 0.999) if recipe == "data_output_velocity_loss" else (0.0, 1.0)
    sampler_time = (
        UniformTime(support=support)
        if time == "uniform"
        else LogitNormalTime(mean=0.0, std=1.0, support=support)
    )
    flow = FlowMatching(
        path=GVPPath(), time=sampler_time, recipe=recipe, reduction="sum"
    )
    net = torch.nn.Linear(5, 2).to(device)
    z = torch.randn(32, 4, 2, device=device)
    weight = torch.ones(32, 4, device=device)
    out = torch.zeros((), device=device)

    def step():
        """
        Computes the loss of a linear network into a static buffer.
        """
        out.copy_(
            flow.loss(
                lambda x, t: net(torch.cat((x, t[:, None].expand(-1, 4, 3)), -1)),
                z,
                weight,
            )[0]
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        step()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    torch.cuda.set_sync_debug_mode("error")
    try:
        with torch.cuda.graph(graph):
            step()
    finally:
        torch.cuda.set_sync_debug_mode("default")
    losses = []
    for _ in range(3):
        graph.replay()
        losses.append(out.item())
    assert all(math.isfinite(v) for v in losses) and len(set(losses)) == 3


def meanflow(**changes) -> MeanFlow:
    """
    Returns a MeanFlow objective on the linear path with some settings changed.
    """
    settings = {
        "path": LinearPath(),
        "time": UniformTime(support=(0.0, 1.0)),
        "second_time": None,
        "order": "sort",
        "flow_matching_share": 0.25,
        "tangent": "conditional",
        "adaptive_power": 0.0,
        "adaptive_eps": 1e-3,
        "reduction": "sum",
    }
    return MeanFlow(**{**settings, **changes})


def exact_average(path: ProbabilityPath):
    """The toy's exact average velocity, predict(x, t (bs, 1)[, s (bs, 1)]) -> u_bar (float32), by quadrature."""
    nodes, weights = (torch.from_numpy(a) for a in np.polynomial.legendre.leggauss(48))

    def average(x: torch.Tensor, t: torch.Tensor, s: torch.Tensor | None = None):
        """
        Integrates v along the exact flow map from t to s, which is smooth at s = t.
        """
        x, t = x.double(), t.double()
        s = t if s is None else s.double()
        start = path(t[:, :, None])
        start_var = start.alpha**2 * S**2 + start.beta**2
        total = 0.0
        for node, weight in zip(nodes, weights):
            tau = t + (node + 1) / 2 * (s - t)
            c = path(tau[:, :, None])
            var = c.alpha**2 * S**2 + c.beta**2
            x_tau = c.alpha * M + (var / start_var).sqrt() * (x - start.alpha * M)
            total = total + weight / 2 * posterior(path, x_tau, tau[:, :, None])[2]
        return total.float()

    return average


@pytest.mark.parametrize("path", PATHS, ids=["linear", "gvp"])
def test_meanflow_identity_on_the_exact_average_velocity(path):
    """
    The exact u_bar satisfies u_bar = v + (s - t) d/dt u_bar, with the derivative taken by MeanFlow's forward pass;
    the identity with the wrong sign fails wherever s != t.
    """
    torch.manual_seed(0)
    x = torch.randn(64, 2, 2) * 2
    t = torch.rand(64, 1) * 0.9
    s = t + (1 - t) * torch.rand(64, 1)
    t[:4], s[:4] = s[:4], s[:4]
    t[4:8], s[4:8] = 0.0, 1.0
    predict = exact_average(path)
    v = predict(x, t)
    average, derivative = MeanFlow.average_velocity(predict, x, t, s, v)
    gap = (s - t)[:, :, None]
    assert torch.allclose(average, v + gap * derivative, atol=2e-5)
    wrong = (average - (v - gap * derivative)).abs().amax(dim=(1, 2))
    assert torch.all(wrong[4:] > 1e-2)


@pytest.mark.parametrize("path", PATHS, ids=["linear", "gvp"])
@pytest.mark.parametrize("grid", ["uniform", "data_dense"])
@pytest.mark.parametrize("nfe", [1, 2, 4, 7])
def test_mean_sampler_reaches_the_exact_endpoint(path, grid, nfe):
    """
    With the exact average velocity, the mean sampler lands on the exact endpoint in any number of steps.
    """
    torch.manual_seed(0)
    x0 = torch.randn(256, 2, 2)
    settings = sampler(solver="mean", nfe=nfe, grid=grid)
    flow = LearnedFlow(exact_average(path), path, VelocityPrediction())
    x1 = settings.sample(flow, x0)
    assert flow.calls == nfe
    assert torch.allclose(x1.double(), M + S * x0.double(), atol=1e-4)


@pytest.mark.parametrize("path", PATHS, ids=["linear", "gvp"])
def test_meanflow_loss_of_the_exact_average_velocity(path):
    """
    With the exact u_bar and its own velocity as the tangent, the target is u_bar + (u - v): the loss equals flow
    matching's of the exact velocity, at any (t, s).
    """
    torch.manual_seed(0)
    z = M.float() + S.float() * torch.randn(64, 2, 2)
    weight = torch.ones(64, 2)
    t = torch.rand(64, 1) * 0.9
    s = t + (1 - t) * torch.rand(64, 1)
    eps = torch.randn_like(z)
    flow = meanflow(path=path, tangent="predicted")
    loss, _ = flow.loss(
        exact_average(path), z, weight, flow_time=t, end_time=s, noise=eps
    )
    matching = FlowMatching(
        path=path, time=flow.time, recipe="velocity_mse", reduction="sum"
    )
    expected, _ = matching.loss(
        exact_predict(path, VelocityPrediction()), z, weight, flow_time=t, noise=eps
    )
    assert loss.item() == pytest.approx(expected.item(), rel=1e-4)


def affine(a: float, b: float, c: float, k: float = 0.0, m: float = 0.0):
    """
    A network u_bar = a x + m mean_steps(x) + b t + c s + k (s omitted is t), whose derivative along (tangent, 1)
    is a tangent + m mean_steps(tangent) + b; the mean carries padded steps' tangents into valid ones.
    """

    def predict(x: torch.Tensor, t: torch.Tensor, s: torch.Tensor | None = None):
        """
        Returns the affine prediction.
        """
        s = t if s is None else s
        mixed = a * x + m * x.mean(dim=1, keepdim=True)
        return mixed + (b * t + c * s + k)[:, :, None]

    return predict


@pytest.mark.parametrize("tangent", ["conditional", "predicted"])
@pytest.mark.parametrize("reduction", ["sum", "example_mean"])
def test_meanflow_target_by_hand(tangent, reduction):
    """
    On an affine network the per-chunk loss L is the hand-computed one, with the conditional velocity standing in
    for the predicted tangent on padded steps: the objective is the mean of (L + eps)^-p L, loss_ref the mean of L.
    """
    torch.manual_seed(0)
    z, eps = torch.randn(2, 16, 3, 2)
    weight = torch.ones(16, 3)
    weight[:5, 2] = 0
    t = torch.rand(16, 1) * 0.8
    s = t + (1 - t) * torch.rand(16, 1)
    a, b, c, k, m = 0.7, -0.4, 1.3, 0.2, 0.9
    path = GVPPath()
    coefficients = path(t[:, :, None])
    x = coefficients.interpolate(z, eps)
    u = coefficients.velocity(z, eps)
    net = affine(a, b, c, k, m)
    move = u
    if tangent == "predicted":
        move = torch.where(weight[:, :, None] > 0, net(x, t), u)
    derivative = a * move + m * move.mean(dim=1, keepdim=True) + b
    target = u + (s - t)[:, :, None] * derivative
    error = net(x, t, s) - target
    squared = error.pow(2).sum(dim=2) * weight
    per_chunk = squared.sum(dim=1)
    if reduction == "example_mean":
        per_chunk = per_chunk / (weight.sum(dim=1) * 2)
    for power in (0.0, 1.0):
        flow = meanflow(
            path=path, tangent=tangent, reduction=reduction, adaptive_power=power
        )
        loss, metrics = flow.loss(net, z, weight, flow_time=t, end_time=s, noise=eps)
        objective = ((per_chunk + 1e-3) ** -power * per_chunk).mean()
        assert loss.item() == pytest.approx(objective.item(), rel=1e-5)
        assert metrics["loss_ref"].item() == pytest.approx(
            per_chunk.mean().item(), rel=1e-5
        )
    if tangent == "predicted":
        # Without the fallback the padded steps' own predictions would move the valid steps' target.
        unmasked = u + (s - t)[:, :, None] * (
            a * net(x, t) + m * net(x, t).mean(dim=1, keepdim=True) + b
        )
        assert not torch.allclose(net(x, t, s) - unmasked, error)


def test_adaptive_weight_scales_each_gradient():
    """
    The gradient is the mean of (L + eps)^-p times each chunk's gradient, py-meanflow's L / sg(L + eps)^p, while
    the dU/dt term carries no gradient.
    """
    torch.manual_seed(0)
    z, eps = torch.randn(2, 8, 3, 2)
    weight = torch.ones(8, 3)
    t = torch.rand(8, 1) * 0.8
    s = t + (1 - t) * torch.rand(8, 1)
    for power in (0.0, 0.5, 1.0):
        gradients = []
        for adaptive in (True, False):
            theta = torch.tensor([0.7, -0.4, 1.3], requires_grad=True)

            def predict(x, tt, ss=None, theta=theta):
                """
                An affine network with trainable coefficients.
                """
                return affine(*theta)(x, tt, ss)

            flow = meanflow(adaptive_power=power if adaptive else 0.0)
            loss, _ = flow.loss(predict, z, weight, flow_time=t, end_time=s, noise=eps)
            if not adaptive:
                # py-meanflow's form, per chunk: the dU/dt term is the constant a u + b.
                c = flow.path(t[:, :, None])
                x, u = c.interpolate(z, eps), c.velocity(z, eps)
                target = u + (s - t)[:, :, None] * (theta[0] * u + theta[1]).detach()
                chunk = (predict(x, t, s) - target).pow(2).sum(dim=(1, 2))
                loss = (chunk / (chunk.detach() + 1e-3) ** power).mean()
            (gradient,) = torch.autograd.grad(loss, theta)
            gradients.append(gradient)
        assert torch.allclose(*gradients, rtol=1e-5)


def reference_pairs(kind: str, n: int, rng: np.random.Generator) -> tuple:
    """
    Ports of the papers' time-pair samplers, mapped to noise at t = 0: (t, s) = (1 - t_paper, 1 - r_paper).
    """

    def logit_normal(mean, std):
        """
        Draws sigmoid(N(mean, std^2)).
        """
        return 1 / (1 + np.exp(-(rng.standard_normal(n) * std + mean)))

    if kind == "v1":
        # py-meanflow v1: r = t with probability 1 - ratio, then r = min(t, r).
        tp, rp = logit_normal(-0.6, 1.6), logit_normal(-4.0, 1.6)
        rp = np.where(rng.random(n) < 0.25, tp, rp)
        rp = np.minimum(tp, rp)
    elif kind == "v0":
        # py-meanflow v0 (the paper's): sorted, then r = t with probability 1 - ratio.
        tp, rp = logit_normal(-2.0, 2.0), logit_normal(-2.0, 2.0)
        tp, rp = np.maximum(tp, rp), np.minimum(tp, rp)
        rp = np.where(rng.random(n) < 0.25, tp, rp)
    else:
        # Improved MeanFlow's sample_tr: sorted, then the first half of the rows get r = t.
        tp, rp = logit_normal(-0.4, 1.0), logit_normal(-0.4, 1.0)
        tp, rp = np.maximum(tp, rp), np.minimum(tp, rp)
        rp = np.where(np.arange(n) < n // 2, tp, rp)
    return 1 - tp, 1 - rp


def shipped_pairs(pairs: str, share: float) -> MeanFlow:
    """
    Returns the MeanFlow objective of a shipped pairs file (conf/flow/pairs) with a flow matching share.
    """
    with initialize(version_base="1.3", config_path="../conf"):
        c = compose(
            "config",
            overrides=[
                "flow=meanflow",
                f"flow/pairs={pairs}",
                f"flow.flow_matching_share={share}",
            ],
        )
    return instantiate(c.flow)


@pytest.mark.parametrize(
    "kind,flow",
    [
        ("v1", shipped_pairs("mf", 0.25)),
        (
            "v0",
            meanflow(time=LogitNormalTime(mean=2.0, std=2.0, support=(0.0, 1.0))),
        ),
        ("imf", shipped_pairs("imf", 0.5)),
    ],
    ids=["meanflow-v1", "meanflow-v0", "improved-meanflow"],
)
def test_time_pairs_match_the_papers(kind, flow):
    """
    Two-sample Kolmogorov-Smirnov of t, s and s - t against the papers' samplers mapped to this convention.
    """
    n = 200_000
    torch.manual_seed(0)
    t, s = (
        v.double().flatten().numpy() for v in flow.sample_times(n, torch.device("cpu"))
    )
    t_ref, s_ref = reference_pairs(kind, n, np.random.default_rng(0))
    assert np.all(s >= t)
    for ours, theirs in ((t, t_ref), (s, s_ref), (s - t, s_ref - t_ref)):
        grid = np.sort(np.concatenate((ours, theirs)))
        distance = (
            np.abs(
                np.searchsorted(np.sort(ours), grid, side="right")
                - np.searchsorted(np.sort(theirs), grid, side="right")
            ).max()
            / n
        )
        assert distance < 1.63 * math.sqrt(2 / n)


def test_time_pairs_by_index():
    """
    The first round(share n) rows are flow matching rows (s = t) and the rest pairs.
    """
    t, s = meanflow(flow_matching_share=0.9).sample_times(10, torch.device("cpu"))
    assert torch.equal(t[:9], s[:9]) and s[9] > t[9]
    t, s = meanflow(flow_matching_share=0.25).sample_times(64, torch.device("cpu"))
    assert torch.equal(t[:16], s[:16]) and torch.all(s[16:] > t[16:])
    assert t.shape == s.shape == (64, 1) and t.dtype == torch.float32


def test_meanflow_at_equal_times_is_flow_matching():
    """
    At s = t, for any network, either tangent and with the adaptive weight, the plain loss (loss_ref, and loss_v of
    the rows with s = t) is flow matching's, loss_u has no rows, and fixed times and noise draw nothing.
    """
    torch.manual_seed(0)
    z, eps = torch.randn(2, 16, 3, 2)
    weight = torch.ones(16, 3)
    weight[:4, 1:] = 0
    t = torch.rand(16, 1)
    net = torch.nn.Linear(4, 2)

    def predict(x, tt, ss=None):
        """
        An arbitrary network of x, t and s.
        """
        ss = tt if ss is None else ss
        times = torch.cat((tt, ss), dim=1)[:, None].expand(-1, x.shape[1], -1)
        return net(torch.cat((x, times), dim=-1))

    for reduction in ("sum", "example_mean"):
        matching = FlowMatching(
            path=GVPPath(),
            time=UniformTime(support=(0.0, 1.0)),
            recipe="velocity_mse",
            reduction=reduction,
        )
        expected, _ = matching.loss(predict, z, weight, flow_time=t, noise=eps)
        for tangent in ("conditional", "predicted"):
            flow = meanflow(
                path=GVPPath(), tangent=tangent, reduction=reduction, adaptive_power=1.0
            )
            state = torch.random.get_rng_state()
            _, metrics = flow.loss(
                predict, z, weight, flow_time=t, end_time=t, noise=eps
            )
            assert torch.equal(torch.random.get_rng_state(), state)
            for name in ("loss_ref", "loss_v"):
                assert metrics[name].item() == pytest.approx(expected.item(), rel=1e-6)
            assert math.isnan(metrics["loss_u"].item())
            with torch.no_grad():
                _, quiet = flow.loss(
                    predict, z, weight, flow_time=t, end_time=t, noise=eps
                )
            assert quiet["loss_ref"].item() == pytest.approx(expected.item(), rel=1e-6)


def test_meanflow_metrics_split_the_rows():
    """
    loss_u and loss_v average the plain per-chunk loss over the rows with s > t and with s = t, including the clip
    rows outside the share that have b <= a; the objective is its adaptively weighted mean, and its parameter
    gradients are bit for bit those of the former form, the plain value with w times the gradient.
    """
    torch.manual_seed(0)
    flow = meanflow(
        time=LogitNormalTime(mean=0.6, std=1.6, support=(0.0, 1.0)),
        second_time=LogitNormalTime(mean=4.0, std=1.6, support=(0.0, 1.0)),
        order="clip",
        adaptive_power=1.0,
    )
    z, eps = torch.randn(2, 64, 3, 2)
    weight = torch.ones(64, 3)
    weight[:6, 2] = 0
    t, s = flow.sample_times(64, torch.device("cpu"))
    same = (s == t)[:, 0]
    assert same[:16].all() and same[16:].any() and not same.all()
    theta = torch.tensor([0.7, -0.4, 1.3, 0.2, 0.9], requires_grad=True)

    def predict(x, tt, ss=None):
        """
        The affine network with trainable coefficients.
        """
        return affine(*theta)(x, tt, ss)

    loss, metrics = flow.loss(predict, z, weight, flow_time=t, end_time=s, noise=eps)
    assert set(metrics) == set(flow.metric_names)
    assert all(not m.requires_grad for m in metrics.values())
    # The per-chunk loss by hand, with the conditional tangent.
    c = flow.path(t[:, :, None])
    x, u = c.interpolate(z, eps), c.velocity(z, eps)
    a, b, _, _, m = theta.detach()
    target = u + (s - t)[:, :, None] * (a * u + m * u.mean(dim=1, keepdim=True) + b)
    chunk = ((predict(x, t, s).detach() - target).pow(2).sum(dim=2) * weight).sum(1)
    expected = {
        "loss_ref": chunk.mean(),
        "loss_u": chunk[~same].mean(),
        "loss_v": chunk[same].mean(),
    }
    for name, value in expected.items():
        assert metrics[name].item() == pytest.approx(value.item(), rel=1e-5)
    assert loss.item() == pytest.approx(
        (chunk / (chunk + 1e-3)).mean().item(), rel=1e-5
    )
    (gradient,) = torch.autograd.grad(loss, theta)
    # The former form: the value stays the plain loss, the gradient is w times its gradient.
    target = u + (s - t)[:, :, None] * (
        theta[0] * u + theta[4] * u.mean(dim=1, keepdim=True) + theta[1]
    )
    per_chunk = flow.per_example(predict(x, t, s) - target.detach(), weight)
    w = (per_chunk.detach() + 1e-3) ** -1.0
    former = (per_chunk.detach() + w * (per_chunk - per_chunk.detach())).mean()
    (former_gradient,) = torch.autograd.grad(former, theta)
    assert torch.equal(gradient, former_gradient)
    assert former.item() == pytest.approx(metrics["loss_ref"].item(), rel=1e-6)


def test_meanflow_validation():
    """
    Invalid MeanFlow settings, the mean sampler on flow matching, end times for flow matching and one fixed time
    of MeanFlow's pairs without the other are refused.
    """
    for changes in (
        {"order": "clip"},
        {"order": "max"},
        {"tangent": "own"},
        {"flow_matching_share": 1.5},
        {"flow_matching_share": -0.1},
        {"tangent": "predicted", "flow_matching_share": 0.0},
        {"adaptive_power": -1.0},
        {"adaptive_eps": 0.0},
    ):
        with pytest.raises(ValueError):
            meanflow(**changes)
    matching = FlowMatching(
        path=LinearPath(),
        time=UniformTime(support=(0.0, 1.0)),
        recipe="velocity_mse",
        reduction="sum",
    )
    with pytest.raises(ValueError):
        sampler(solver="mean").check(matching)
    sampler(solver="mean").check(meanflow())
    sampler(solver="euler").check(meanflow())
    z = torch.zeros(2, 3, 2)
    with pytest.raises(ValueError):
        matching.loss(
            lambda x, t: x,
            z,
            torch.ones(2, 3),
            flow_time=torch.zeros(2, 1),
            end_time=torch.ones(2, 1),
        )
    for times in ({"end_time": torch.ones(2, 1)}, {"flow_time": torch.zeros(2, 1)}):
        with pytest.raises(ValueError):
            meanflow().loss(lambda x, t, s=None: x, z, torch.ones(2, 3), **times)


class AttentionNet(torch.nn.Module):
    """
    A one-layer attention network of (x, t, s), to capture MeanFlow's forward-mode pass through SDPA.
    """

    def __init__(self) -> None:
        """
        An embedding of (x, t, s - t), one two-head attention layer and a linear readout.
        """
        super().__init__()
        self.embed = torch.nn.Linear(4, 16)
        self.qkv = torch.nn.Linear(16, 48)
        self.out = torch.nn.Linear(16, 2)

    def forward(self, x, t, s=None):
        """
        Attends over the chunk's steps.
        """
        s = t if s is None else s
        times = torch.cat((t, s - t), dim=1)[:, None].expand(-1, x.shape[1], -1)
        h = self.embed(torch.cat((x, times), dim=-1))
        q, k, v = (
            self.qkv(h).reshape(x.shape[0], x.shape[1], 3, 2, 8).permute(2, 0, 3, 1, 4)
        )
        h = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        return self.out(h.transpose(1, 2).reshape(x.shape[0], x.shape[1], 16))


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("tangent", ["conditional", "predicted"])
def test_meanflow_loss_is_graph_capturable(tangent):
    """
    MeanFlow's loss, its metrics and its backward pass through attention are captured in a CUDA graph without host
    syncs.
    """
    device = torch.device("cuda")
    flow = meanflow(
        time=LogitNormalTime(mean=0.6, std=1.6, support=(0.0, 1.0)),
        second_time=LogitNormalTime(mean=4.0, std=1.6, support=(0.0, 1.0)),
        order="clip",
        tangent=tangent,
        adaptive_power=0.75,
    )
    net = AttentionNet().to(device)
    z = torch.randn(32, 4, 2, device=device)
    weight = torch.ones(32, 4, device=device)
    weight[:8, 2:] = 0
    out = torch.zeros((), device=device)
    sums = {name: torch.zeros((), device=device) for name in flow.metric_names}

    def step():
        """
        Computes the loss and its gradients into static buffers, and adds the metrics to their sums.
        """
        for p in net.parameters():
            p.grad = None
        loss, metrics = flow.loss(net, z, weight)
        loss.backward()
        out.copy_(loss.detach())
        for name, value in metrics.items():
            sums[name].add_(value)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        step()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    torch.cuda.set_sync_debug_mode("error")
    try:
        with torch.cuda.graph(graph):
            step()
    finally:
        torch.cuda.set_sync_debug_mode("default")
    losses = []
    for _ in range(3):
        graph.replay()
        losses.append(out.item())
    assert all(math.isfinite(v) for v in losses) and len(set(losses)) == 3
    assert all(torch.isfinite(p.grad).all() for p in net.parameters())
    assert all(
        math.isfinite(total.item()) and total.item() > 0 for total in sums.values()
    )
