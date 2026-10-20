"""flow.py on a Gaussian toy where every quantity is analytic.

Data z ~ N(m, s^2) per coordinate and eps ~ N(0, 1): x_t ~ N(alpha m, V) with V = alpha^2 s^2 + beta^2, and
    E[z | x] = m + alpha s^2 (x - alpha m) / V,   E[eps | x] = beta (x - alpha m) / V,
    v = d_alpha E[z | x] + d_beta E[eps | x],     q = -E[eps | x],     score = -(x - alpha m) / V.
The probability-flow map is affine, so from t = 0 every sample has an exact endpoint x_1 = m + s x_0, and the exact
MeanFlow average velocity over [t, s] is a quadrature of v along it.

The oracles compute in float64 under jax.enable_x64, on arrays cast to float64 explicitly; x64 changes the default
dtypes, so the code under test is always fed float32 arrays and runs in float32, as in training.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hydra import compose, initialize
from hydra.utils import instantiate

from flow import (
    QUANTILE_BITS,
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
M = np.array([[-1.0, 0.5], [2.0, 0.0]])
S = np.array([[0.1, 0.5], [2.0, 1.0]])


def randn(rng: np.random.Generator, *shape: int) -> jax.Array:
    """
    Returns standard normal draws of a numpy generator as a float32 array.
    """
    return jnp.asarray(rng.standard_normal(shape, dtype=np.float32))


def rand(rng: np.random.Generator, *shape: int) -> jax.Array:
    """
    Returns U(0, 1) draws of a numpy generator as a float32 array.
    """
    return jnp.asarray(rng.random(shape, dtype=np.float32))


def f64(x) -> jax.Array:
    """
    Casts to float64; only under jax.enable_x64, which the oracles run in.
    """
    x = jnp.asarray(x, jnp.float64)
    assert x.dtype == jnp.float64, "float64 oracles run under jax.enable_x64(True)"
    return x


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


def posterior(path: ProbabilityPath, x: jax.Array, t: jax.Array):
    """E[z|x], E[eps|x], v and score of the toy at x (B, 2, 2), t broadcastable, in float64."""
    c = path(f64(t))
    var = c.alpha**2 * S**2 + c.beta**2
    centered = f64(x) - c.alpha * M
    data = M + c.alpha * S**2 * centered / var
    noise = c.beta * centered / var
    return data, noise, c.d_alpha * data + c.d_beta * noise, -centered / var


def exact_predict(path: ProbabilityPath, prediction):
    """The Bayes-optimal network for the toy: predict(x, t (B, 1)) -> its prediction (float32)."""

    def predict(x: jax.Array, t: jax.Array) -> jax.Array:
        """
        Returns the Bayes-optimal prediction at x (bs, 2, 2) and t (bs, 1).
        """
        data, noise, velocity, _ = posterior(path, x, t[:, :, None])
        if isinstance(prediction, VelocityPrediction):
            return velocity.astype(jnp.float32)
        return (data if isinstance(prediction, DataPrediction) else -noise).astype(
            jnp.float32
        )

    return predict


@pytest.mark.parametrize("path", PATHS, ids=["linear", "gvp"])
def test_path_coefficients(path):
    """
    Endpoints are exact in float32, D matches its definition and the derivatives match finite differences; the
    host evaluation at a grid time (path.at, numpy float64) has the same exact endpoints and agrees with the traced
    one.
    """
    # Exact endpoints in float32 (cos(pi / 2) is not 0 in float32, so GVP writes beta as a sine).
    ends = jnp.array([0.0, 1.0], jnp.float32)
    assert path.alpha(ends).tolist() == [0.0, 1.0]
    assert path.beta(ends).tolist() == [1.0, 0.0]
    assert path.at(0.0)[:2] == (0.0, 1.0) and path.at(1.0)[:2] == (1.0, 0.0)
    with jax.enable_x64(True):
        t = f64(jnp.linspace(0.01, 0.99, 99, dtype=jnp.float64))
        c = path(t)
        assert np.allclose(
            c.beta * c.d_alpha - c.alpha * c.d_beta, np.full(t.shape, c.D)
        )
        h = 1e-6
        for f, df in ((path.alpha, path.d_alpha), (path.beta, path.d_beta)):
            assert np.allclose((f(t + h) - f(t - h)) / (2 * h), df(t), atol=1e-8)
        for v in (0.0, 0.3, 0.999, 1.0):
            traced = [float(x) for x in path(f64(v))]
            assert list(path.at(v)) == pytest.approx(traced, rel=1e-12, abs=1e-15)


@pytest.mark.parametrize("path", PATHS, ids=["linear", "gvp"])
def test_adapters_on_the_gaussian_posterior(path):
    """
    Every adapter maps the analytic posterior quantities onto each other, and guidance commutes with them.
    """
    rng = np.random.default_rng(0)
    with jax.enable_x64(True):
        x = f64(rng.standard_normal((64, 2, 2)) * 2)
        t = f64(rng.random((64, 1, 1)) * 0.9 + 0.05)
        c = path(t)
        data, noise, velocity, score = posterior(path, x, t)
        assert np.allclose(c.data_to_velocity(data, x), velocity)
        assert np.allclose(c.scaled_score_to_velocity(-noise, x), velocity)
        assert np.allclose(c.velocity_to_scaled_score(velocity, x), -noise)
        assert np.allclose(c.velocity_to_score(velocity, x), score)
        # Guidance mixes network outputs; the adapters are affine with unit weight on the output, so mixing commutes.
        for w in (0.0, 1.5, 3.0):
            a, b = data, data + noise
            mixed = c.data_to_velocity((1 - w) * a + w * b, x)
            assert np.allclose(
                mixed, (1 - w) * c.data_to_velocity(a, x) + w * c.data_to_velocity(b, x)
            )


def test_conditional_velocity_is_the_path_derivative():
    """
    u = d_alpha z + d_beta eps is the time derivative of x_t.
    """
    rng = np.random.default_rng(0)
    with jax.enable_x64(True):
        z, eps = f64(rng.standard_normal((2, 8, 2, 2)))
        for path in PATHS:
            for t in (0.1, 0.5, 0.9):
                tt = f64(t)
                h = f64(1e-6)
                fd = (
                    path(tt + h).interpolate(z, eps) - path(tt - h).interpolate(z, eps)
                ) / (2 * h)
                assert np.allclose(fd, path(tt).velocity(z, eps), atol=1e-7)


def test_uniform_time_lattice():
    """
    Uniform flow times lie strictly inside (0, 1) on the odd multiples of 2^-24.
    """
    t = UniformTime(support=(0.0, 1.0)).sample(jax.random.key(0), 1_000_000)
    assert t.dtype == jnp.float32 and t.shape == (1_000_000, 1)
    assert 0.0 < t.min() and t.max() < 1.0
    # Quantiles (2k + 1) 2^-24: every draw is an odd multiple of 2^-24.
    assert np.all((np.asarray(t, np.float64) * 2**24) % 2 == 1)
    assert abs(t.mean().item() - 0.5) < 2e-3


def test_time_sampler_validation():
    """
    Invalid supports and distributions that reach a float32 endpoint are refused.
    """
    with pytest.raises(ValueError):
        UniformTime(support=(0.5, 0.5))
    with pytest.raises(ValueError):
        UniformTime(support=(-0.1, 1.0))
    with pytest.raises(ValueError):
        LogitNormalTime(mean=0.0, std=0.0, support=(0.0, 1.0))
    # The extreme quantile is 5.29 standard deviations out; past logit 17.33 a draw rounds to 1 in float32.
    with pytest.raises(ValueError):
        LogitNormalTime(mean=0.0, std=3.3, support=(0.0, 1.0))
    LogitNormalTime(mean=0.0, std=3.25, support=(0.0, 1.0))
    LogitNormalTime(mean=0.0, std=3.3, support=(0.0, 0.99))


@pytest.mark.parametrize(
    "time",
    [
        UniformTime(support=(0.001, 0.999)),
        UniformTime(support=(0.2, 0.3)),
        LogitNormalTime(mean=0.0, std=1.0, support=(0.0, 1.0)),
        LogitNormalTime(mean=0.0, std=1.0, support=(0.3, 0.9)),
        LogitNormalTime(mean=1.0, std=0.5, support=(0.001, 0.6)),
    ],
)
def test_truncated_draws_follow_the_conditional_cdf(time):
    """Kolmogorov-Smirnov against F_S(t) = (F(t) - F(a)) / (F(b) - F(a)): sampling from F, redrawing outside S.
    Draws and the extreme quantiles lie in S up to float32 rounding: a draw may exceed b by one float32 step."""
    n = 200_000
    t = np.sort(np.asarray(time.sample(jax.random.key(0), n), np.float64).ravel())
    low, high = time.support
    # The extreme quantiles bound every draw; at (0.2, 0.3) the last is float32(0.3) > 0.3.
    extremes = jnp.array([0, 2**QUANTILE_BITS - 1])
    first, last = np.asarray(time._draw(extremes), np.float64).tolist()
    for a, b in ((first, last), (t[0], t[-1])):
        assert low <= a and b <= np.nextafter(np.float32(high), np.float32(np.inf))
    cdf = np.array([time.cdf(v) for v in t.tolist()])
    conditional = (cdf - time.mass[0]) / (time.mass[1] - time.mass[0])
    empirical = np.arange(1, n + 1) / n
    assert np.abs(conditional - empirical).max() < 1.95 / math.sqrt(n)


def replicate_draws(flow: FlowMatching, z: jax.Array, key: jax.Array):
    """The flow times and noise that flow.loss draws from key, in its order."""
    time_key, noise_key = jax.random.split(key)
    t = flow.time.sample(time_key, z.shape[0])
    return t, jax.random.normal(noise_key, z.shape, z.dtype)


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
    z = randn(np.random.default_rng(0), 16, 2, 2)
    weight = jnp.ones((16, 2))
    key = jax.random.key(1)
    t, eps = replicate_draws(flow, z, key)
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

    loss, metrics = flow.loss(predict, z, weight, key=key)
    assert loss.item() < 1e-6 and metrics == {}
    assert np.array_equal(seen["t"], t) and seen["t"].shape == (16, 1)
    assert np.allclose(seen["x"], c.interpolate(z, eps))


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
    z = jnp.zeros((2, 3, 2))
    error = jnp.array(
        [[[1.0, 1.0], [2.0, 0.0], [0.0, 3.0]], [[0.0, 2.0], [1.0, 0.0], [1.0, 1.0]]]
    )
    weight = jnp.array([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    key = jax.random.key(0)
    # data_mse regresses z, so a prediction z + error leaves exactly this error.
    loss, _ = flow.loss(lambda x, t: z + error, z, weight, key=key)
    assert loss.item() == pytest.approx(5.0)
    flow.reduction = "example_mean"
    loss, _ = flow.loss(lambda x, t: z + error, z, weight, key=key)
    assert loss.item() == pytest.approx(1.75)


def test_velocity_loss_needs_a_support_away_from_the_singularity():
    """
    A velocity-space loss on a data prediction needs a support that excludes t = 1.
    """
    with pytest.raises(ValueError):
        FlowMatching(
            path=LinearPath(),
            time=UniformTime(support=(0.0, 1.0)),
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
    assert heun == np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0], np.float32).tolist()
    dense = sampler(grid="data_dense").times(VelocityPrediction())
    steps = np.diff(np.array(dense, np.float32))
    assert np.all(steps[1:] < steps[:-1]) and dense[-1] == 1.0
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
    x1 = settings.sample(flow, x0, brownian)
    assert x1.dtype == jnp.float32
    return x1, flow.calls


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
    x0 = randn(np.random.default_rng(0), 256, 2, 2)
    with jax.enable_x64(True):
        x1, calls = solve(path, prediction, settings, x0)
    start = settings.times(prediction)[0]
    c = path.at(start)
    var = c.alpha**2 * S**2 + c.beta**2
    x0 = np.asarray(x0, np.float64)
    exact = M + S * (x0 - c.alpha * M) / np.sqrt(var)
    assert calls == settings.nfe
    assert np.all(np.abs(x1 - exact) <= tolerance * (S * np.abs(x0) + np.abs(M)) + 1e-4)


def test_convergence_orders():
    """
    Euler converges at order 1 and Heun at order 2.
    """
    x0 = randn(np.random.default_rng(0), 256, 2, 2)
    exact = M + S * np.asarray(x0, np.float64)

    def error(settings):
        """
        Returns the largest endpoint error of a sampler.
        """
        with jax.enable_x64(True):
            x1, _ = solve(LinearPath(), VelocityPrediction(), settings, x0)
        return np.abs(x1 - exact).max()

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
    rng = np.random.default_rng(0)
    settings = sampler(solver="sde", nfe=1000, sde_eta=eta)
    c = path.at(settings.times(prediction)[0])
    x0 = jnp.asarray(
        (
            c.alpha * M
            + np.sqrt(c.alpha**2 * S**2 + c.beta**2)
            * rng.standard_normal((20_000, 2, 2))
        ).astype(np.float32)
    )
    brownian = randn(rng, 20_000, settings.num_brownian, 2, 2)
    with jax.enable_x64(True):
        x1, calls = solve(path, prediction, settings, x0, brownian)
    assert calls == 1000
    x1 = np.asarray(x1, np.float64)
    assert np.all(np.abs(x1.mean(0) - M) < 0.03 * S + 2e-3)
    assert np.all(np.abs(x1.std(0, ddof=1) - S) < 0.03 * S + 2e-3)


@pytest.mark.parametrize("path", PATHS, ids=["linear", "gvp"])
@pytest.mark.parametrize("prediction", PREDICTIONS, ids=["velocity", "data", "score"])
def test_sde_drift_is_velocity_plus_half_g_squared_score(path, prediction):
    """
    The SDE's drift equals v + g^2 / 2 times the score, with g^2 = 2 eta alpha beta, for every path and prediction.
    """
    from flow import LangevinFlow

    flow = LearnedFlow(exact_predict(path, prediction), path, prediction)
    sde = LangevinFlow(flow, eta=0.7)
    x = randn(np.random.default_rng(0), 32, 2, 2)
    with jax.enable_x64(True):
        for t in (0.1, 0.5, 0.95):
            v = flow.drift(x, t)
            c = path.at(t)
            expected = v + 0.5 * sde.diffusion(t) ** 2 * c.velocity_to_score(v, x)
            assert np.allclose(sde.drift(x, t), expected, atol=1e-5)
            assert sde.diffusion(t) ** 2 == pytest.approx(2 * 0.7 * c.alpha * c.beta)


def test_sde_on_the_linear_path_keeps_its_closed_form():
    """
    On the linear path the general drift and diffusion are bit for bit v - eta t (x - t v) and
    sqrt(2 eta t (1 - t)), so linear-path SDE samples do not change.
    """
    from flow import LangevinFlow

    flow = LearnedFlow(
        lambda x, t: jnp.sin(3 * x) + t[:, :, None],
        LinearPath(),
        VelocityPrediction(),
    )
    sde = LangevinFlow(flow, eta=0.3)
    x = randn(np.random.default_rng(0), 64, 16, 2)
    for t in sampler(solver="sde", nfe=10).times(VelocityPrediction())[:-1]:
        v = flow.drift(x, t)
        assert np.array_equal(sde.drift(x, t), v - 0.3 * t * (x - t * v))
        assert sde.diffusion(t) == math.sqrt(2 * 0.3 * t * (1 - t))


def test_data_prediction_last_step_lands_on_it():
    """
    With a data prediction, the last Euler step lands on the last prediction.
    """
    path, prediction = LinearPath(), DataPrediction()
    settings = sampler(nfe=10)
    x0 = randn(np.random.default_rng(0), 64, 2, 2)
    final_prediction = {}

    def predict(x, t):
        """
        Returns the exact prediction and keeps the last one.
        """
        y = exact_predict(path, prediction)(x, t)
        final_prediction["y"] = y
        return y

    with jax.enable_x64(True):
        x1 = settings.sample(LearnedFlow(predict, path, prediction), x0)
    assert np.allclose(x1, final_prediction["y"], atol=1e-6)


def test_sampler_errors_on_bad_inputs():
    """
    Mismatched Brownian draws raise. Sampling runs inside jit and cannot look at values, so the finiteness check
    (torch's FloatingPointError) moved to the caller, evaluation.query_policy, which checks the result only: a NaN
    at any step, here the first or a middle one, eagerly or compiled, must reach every entry of the result.
    """
    path, prediction = LinearPath(), VelocityPrediction()
    x0 = randn(np.random.default_rng(0), 4, 2, 2)
    for nan_time in (0.0, 0.5):
        nan_flow = LearnedFlow(
            lambda x, t, nan_time=nan_time: jnp.where(
                t[:, :, None] == nan_time, jnp.nan, -x
            ),
            path,
            prediction,
        )
        assert np.isnan(sampler().sample(nan_flow, x0)).all()
        compiled = jax.jit(lambda x, flow=nan_flow: sampler().sample(flow, x))
        assert np.isnan(compiled(x0)).all()
    flow = LearnedFlow(exact_predict(path, prediction), path, prediction)
    with pytest.raises(ValueError):
        sampler(solver="sde").sample(flow, x0)  # no Brownian draws
    with pytest.raises(ValueError):
        sampler().sample(flow, x0, randn(np.random.default_rng(1), 4, 9, 2, 2))


def linear(key: jax.Array, d_in: int, d_out: int) -> dict[str, jax.Array]:
    """
    Returns the weights of a linear layer, initialized as torch.nn.Linear: U(+-1 / sqrt(d_in)).
    """
    weight_key, bias_key = jax.random.split(key)
    bound = 1 / math.sqrt(d_in)
    return {
        "w": jax.random.uniform(weight_key, (d_in, d_out), minval=-bound, maxval=bound),
        "b": jax.random.uniform(bias_key, (d_out,), minval=-bound, maxval=bound),
    }


def dense(layer: dict[str, jax.Array], x: jax.Array) -> jax.Array:
    """
    Applies a linear layer, x w + b.
    """
    return x @ layer["w"] + layer["b"]


def tree_allclose(a, b, **tolerance) -> bool:
    """
    Whether two pytrees have the same structure and every pair of leaves is allclose.
    """
    return jax.tree.structure(a) == jax.tree.structure(b) and all(
        np.allclose(x, y, **tolerance)
        for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b))
    )


def tree_equal(a, b) -> bool:
    """
    Whether two pytrees have the same structure and bitwise equal leaves, NaN equal to NaN.
    """
    return jax.tree.structure(a) == jax.tree.structure(b) and all(
        np.array_equal(x, y, equal_nan=True)
        for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b))
    )


@pytest.mark.parametrize("recipe", sorted(RECIPES))
@pytest.mark.parametrize("time", ["uniform", "logit_normal"])
def test_loss_jits(recipe, time):
    """
    The loss of every recipe and time sampler compiles once, the key an argument (torch: captured in a CUDA graph
    without host syncs): the compiled loss equals the eager one, a key reproduces its loss and different keys give
    different finite losses.
    """
    support = (0.001, 0.999) if recipe == "data_output_velocity_loss" else (0.0, 1.0)
    sampler_time = (
        UniformTime(support=support)
        if time == "uniform"
        else LogitNormalTime(mean=0.0, std=1.0, support=support)
    )
    flow = FlowMatching(
        path=GVPPath(), time=sampler_time, recipe=recipe, reduction="sum"
    )
    net = linear(jax.random.key(0), 5, 2)
    z = randn(np.random.default_rng(0), 32, 4, 2)
    weight = jnp.ones((32, 4))
    traces = []

    def loss(net, z, weight, key):
        """
        Computes the loss of a linear network.
        """
        traces.append(key)

        def predict(x, t):
            """
            A linear network of (x, t).
            """
            times = jnp.broadcast_to(t[:, :, None], (*x.shape[:2], 3))
            return dense(net, jnp.concatenate((x, times), axis=-1))

        return flow.loss(predict, z, weight, key=key)[0]

    compiled = jax.jit(loss)
    keys = jax.random.split(jax.random.key(1), 3)
    losses = [compiled(net, z, weight, key).item() for key in keys]
    assert len(traces) == 1
    assert compiled(net, z, weight, keys[0]).item() == losses[0]
    assert np.allclose(losses, [loss(net, z, weight, key).item() for key in keys])
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
    nodes, weights = np.polynomial.legendre.leggauss(48)

    def average(x: jax.Array, t: jax.Array, s: jax.Array | None = None):
        """
        Integrates v along the exact flow map from t to s, which is smooth at s = t.
        """
        x, t = f64(x), f64(t)
        s = t if s is None else f64(s)
        start = path(t[:, :, None])
        start_var = start.alpha**2 * S**2 + start.beta**2
        total = 0.0
        for node, weight in zip(nodes, weights):
            tau = t + (node + 1) / 2 * (s - t)
            c = path(tau[:, :, None])
            var = c.alpha**2 * S**2 + c.beta**2
            x_tau = c.alpha * M + jnp.sqrt(var / start_var) * (x - start.alpha * M)
            total = total + weight / 2 * posterior(path, x_tau, tau[:, :, None])[2]
        return total.astype(jnp.float32)

    return average


@pytest.mark.parametrize("path", PATHS, ids=["linear", "gvp"])
def test_meanflow_identity_on_the_exact_average_velocity(path):
    """
    The exact u_bar satisfies u_bar = v + (s - t) d/dt u_bar, with the derivative taken by MeanFlow's forward pass;
    the identity with the wrong sign fails wherever s != t.
    """
    rng = np.random.default_rng(0)
    x = randn(rng, 64, 2, 2) * 2
    t = rng.random((64, 1), dtype=np.float32) * 0.9
    s = t + (1 - t) * rng.random((64, 1), dtype=np.float32)
    t[:4], s[:4] = s[:4], s[:4]
    t[4:8], s[4:8] = 0.0, 1.0
    t, s = jnp.asarray(t), jnp.asarray(s)
    predict = exact_average(path)
    with jax.enable_x64(True):
        v = predict(x, t)
        average, derivative = MeanFlow.average_velocity(predict, x, t, s, v)
    assert average.dtype == derivative.dtype == jnp.float32
    gap = (s - t)[:, :, None]
    assert np.allclose(average, v + gap * derivative, atol=2e-5)
    wrong = np.abs(average - (v - gap * derivative)).max(axis=(1, 2))
    assert np.all(wrong[4:] > 1e-2)


@pytest.mark.parametrize("path", PATHS, ids=["linear", "gvp"])
@pytest.mark.parametrize("grid", ["uniform", "data_dense"])
@pytest.mark.parametrize("nfe", [1, 2, 4, 7])
def test_mean_sampler_reaches_the_exact_endpoint(path, grid, nfe):
    """
    With the exact average velocity, the mean sampler lands on the exact endpoint in any number of steps.
    """
    x0 = randn(np.random.default_rng(0), 256, 2, 2)
    settings = sampler(solver="mean", nfe=nfe, grid=grid)
    flow = LearnedFlow(exact_average(path), path, VelocityPrediction())
    with jax.enable_x64(True):
        x1 = settings.sample(flow, x0)
    assert flow.calls == nfe
    assert np.allclose(x1, M + S * np.asarray(x0, np.float64), atol=1e-4)


@pytest.mark.parametrize("path", PATHS, ids=["linear", "gvp"])
def test_meanflow_loss_of_the_exact_average_velocity(path):
    """
    With the exact u_bar and its own velocity as the tangent, the target is u_bar + (u - v): the loss equals flow
    matching's of the exact velocity, at any (t, s).
    """
    rng = np.random.default_rng(0)
    z = jnp.asarray(M + S * rng.standard_normal((64, 2, 2)), jnp.float32)
    weight = jnp.ones((64, 2))
    t = rand(rng, 64, 1) * 0.9
    s = t + (1 - t) * rand(rng, 64, 1)
    eps = randn(rng, 64, 2, 2)
    flow = meanflow(path=path, tangent="predicted")
    matching = FlowMatching(
        path=path, time=flow.time, recipe="velocity_mse", reduction="sum"
    )
    with jax.enable_x64(True):
        loss, _ = flow.loss(
            exact_average(path), z, weight, flow_time=t, end_time=s, noise=eps
        )
        expected, _ = matching.loss(
            exact_predict(path, VelocityPrediction()), z, weight, flow_time=t, noise=eps
        )
    assert loss.dtype == jnp.float32
    assert loss.item() == pytest.approx(expected.item(), rel=1e-4)


def affine(a: float, b: float, c: float, k: float = 0.0, m: float = 0.0):
    """
    A network u_bar = a x + m mean_steps(x) + b t + c s + k (s omitted is t), whose derivative along (tangent, 1)
    is a tangent + m mean_steps(tangent) + b; the mean carries padded steps' tangents into valid ones.
    """

    def predict(x: jax.Array, t: jax.Array, s: jax.Array | None = None):
        """
        Returns the affine prediction.
        """
        s = t if s is None else s
        mixed = a * x + m * x.mean(axis=1, keepdims=True)
        return mixed + (b * t + c * s + k)[:, :, None]

    return predict


@pytest.mark.parametrize("tangent", ["conditional", "predicted"])
@pytest.mark.parametrize("reduction", ["sum", "example_mean"])
def test_meanflow_target_by_hand(tangent, reduction):
    """
    On an affine network the per-chunk loss L is the hand-computed one, with the conditional velocity standing in
    for the predicted tangent on padded steps: the objective is the mean of (L + eps)^-p L, loss_ref the mean of L.
    """
    rng = np.random.default_rng(0)
    z, eps = randn(rng, 2, 16, 3, 2)
    weight = np.ones((16, 3), np.float32)
    weight[:5, 2] = 0
    weight = jnp.asarray(weight)
    t = rand(rng, 16, 1) * 0.8
    s = t + (1 - t) * rand(rng, 16, 1)
    a, b, c, k, m = 0.7, -0.4, 1.3, 0.2, 0.9
    path = GVPPath()
    coefficients = path(t[:, :, None])
    x = coefficients.interpolate(z, eps)
    u = coefficients.velocity(z, eps)
    net = affine(a, b, c, k, m)
    move = u
    if tangent == "predicted":
        move = jnp.where(weight[:, :, None] > 0, net(x, t), u)
    derivative = a * move + m * move.mean(axis=1, keepdims=True) + b
    target = u + (s - t)[:, :, None] * derivative
    error = net(x, t, s) - target
    squared = jnp.square(error).sum(axis=2) * weight
    per_chunk = squared.sum(axis=1)
    if reduction == "example_mean":
        per_chunk = per_chunk / (weight.sum(axis=1) * 2)
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
            a * net(x, t) + m * net(x, t).mean(axis=1, keepdims=True) + b
        )
        assert not np.allclose(net(x, t, s) - unmasked, error)


def test_adaptive_weight_scales_each_gradient():
    """
    The gradient is the mean of (L + eps)^-p times each chunk's gradient, py-meanflow's L / sg(L + eps)^p, while
    the dU/dt term carries no gradient.
    """
    rng = np.random.default_rng(0)
    z, eps = randn(rng, 2, 8, 3, 2)
    weight = jnp.ones((8, 3))
    t = rand(rng, 8, 1) * 0.8
    s = t + (1 - t) * rand(rng, 8, 1)
    theta = jnp.array([0.7, -0.4, 1.3])
    for power in (0.0, 0.5, 1.0):
        gradients = []
        for adaptive in (True, False):

            def objective(theta, adaptive=adaptive, power=power):
                """
                The loss of an affine network with trainable coefficients.
                """
                predict = affine(*theta)
                flow = meanflow(adaptive_power=power if adaptive else 0.0)
                loss, _ = flow.loss(
                    predict, z, weight, flow_time=t, end_time=s, noise=eps
                )
                if not adaptive:
                    # py-meanflow's form, per chunk: the dU/dt term is the constant a u + b.
                    c = flow.path(t[:, :, None])
                    x, u = c.interpolate(z, eps), c.velocity(z, eps)
                    target = u + (s - t)[:, :, None] * jax.lax.stop_gradient(
                        theta[0] * u + theta[1]
                    )
                    chunk = jnp.square(predict(x, t, s) - target).sum(axis=(1, 2))
                    loss = (
                        chunk / (jax.lax.stop_gradient(chunk) + 1e-3) ** power
                    ).mean()
                return loss

            gradients.append(jax.grad(objective)(theta))
        assert np.allclose(*gradients, rtol=1e-5)


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
    t, s = (
        np.asarray(v, np.float64).ravel()
        for v in flow.sample_times(jax.random.key(0), n)
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
    key = jax.random.key(0)
    t, s = meanflow(flow_matching_share=0.9).sample_times(key, 10)
    assert np.array_equal(t[:9], s[:9]) and s[9, 0] > t[9, 0]
    t, s = meanflow(flow_matching_share=0.25).sample_times(key, 64)
    assert np.array_equal(t[:16], s[:16]) and np.all(s[16:] > t[16:])
    assert t.shape == s.shape == (64, 1) and t.dtype == jnp.float32


def test_meanflow_at_equal_times_is_flow_matching():
    """
    At s = t, for any network, either tangent and with the adaptive weight, the plain loss (loss_ref, and loss_v of
    the rows with s = t) is flow matching's, loss_u has no rows, and fixed times and noise draw nothing: the loss
    needs no key and a key changes nothing. torch's no_grad run is the eager call here, its autograd run the call
    under jax.value_and_grad.
    """
    rng = np.random.default_rng(0)
    z, eps = randn(rng, 2, 16, 3, 2)
    weight = np.ones((16, 3), np.float32)
    weight[:4, 1:] = 0
    weight = jnp.asarray(weight)
    t = rand(rng, 16, 1)
    net = linear(jax.random.key(0), 4, 2)

    def network(net):
        """
        An arbitrary network of x, t and s with weights net.
        """

        def predict(x, tt, ss=None):
            """
            A linear network of (x, t, s).
            """
            ss = tt if ss is None else ss
            times = jnp.concatenate((tt, ss), axis=1)[:, None]
            times = jnp.broadcast_to(times, (*x.shape[:2], 2))
            return dense(net, jnp.concatenate((x, times), axis=-1))

        return predict

    for reduction in ("sum", "example_mean"):
        matching = FlowMatching(
            path=GVPPath(),
            time=UniformTime(support=(0.0, 1.0)),
            recipe="velocity_mse",
            reduction=reduction,
        )
        expected, _ = matching.loss(network(net), z, weight, flow_time=t, noise=eps)
        for tangent in ("conditional", "predicted"):
            flow = meanflow(
                path=GVPPath(), tangent=tangent, reduction=reduction, adaptive_power=1.0
            )

            def objective(net, key=None, flow=flow):
                """
                The MeanFlow loss at s = t with fixed noise.
                """
                return flow.loss(
                    network(net),
                    z,
                    weight,
                    key=key,
                    flow_time=t,
                    end_time=t,
                    noise=eps,
                )

            (_, metrics), _ = jax.value_and_grad(objective, has_aux=True)(net)
            for name in ("loss_ref", "loss_v"):
                assert metrics[name].item() == pytest.approx(expected.item(), rel=1e-6)
            assert math.isnan(metrics["loss_u"].item())
            quiet = objective(net)
            assert quiet[1]["loss_ref"].item() == pytest.approx(
                expected.item(), rel=1e-6
            )
            assert tree_equal(objective(net, jax.random.key(5)), quiet)


def test_meanflow_metrics_split_the_rows():
    """
    loss_u and loss_v average the plain per-chunk loss over the rows with s > t and with s = t, including the clip
    rows outside the share that have b <= a; the objective is its adaptively weighted mean, and its parameter
    gradients are bit for bit those of the former form, the plain value with w times the gradient.
    """
    flow = meanflow(
        time=LogitNormalTime(mean=0.6, std=1.6, support=(0.0, 1.0)),
        second_time=LogitNormalTime(mean=4.0, std=1.6, support=(0.0, 1.0)),
        order="clip",
        adaptive_power=1.0,
    )
    rng = np.random.default_rng(0)
    z, eps = randn(rng, 2, 64, 3, 2)
    weight = np.ones((64, 3), np.float32)
    weight[:6, 2] = 0
    weight = jnp.asarray(weight)
    t, s = flow.sample_times(jax.random.key(0), 64)
    same = np.asarray(s == t)[:, 0]
    assert same[:16].all() and same[16:].any() and not same.all()
    theta = jnp.array([0.7, -0.4, 1.3, 0.2, 0.9])

    def objective(theta):
        """
        The loss of the affine network with trainable coefficients.
        """
        return flow.loss(affine(*theta), z, weight, flow_time=t, end_time=s, noise=eps)

    (loss, metrics), gradient = jax.value_and_grad(objective, has_aux=True)(theta)
    assert set(metrics) == set(flow.metric_names)
    # The metrics carry no gradient (torch: not requires_grad).
    metric_gradient = jax.grad(lambda theta: sum(objective(theta)[1].values()))(theta)
    assert np.all(metric_gradient == 0)
    # The per-chunk loss by hand, with the conditional tangent.
    c = flow.path(t[:, :, None])
    x, u = c.interpolate(z, eps), c.velocity(z, eps)
    a, b, _, _, m = theta
    target = u + (s - t)[:, :, None] * (a * u + m * u.mean(axis=1, keepdims=True) + b)
    chunk = (jnp.square(affine(*theta)(x, t, s) - target).sum(axis=2) * weight).sum(1)
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

    def former(theta):
        """
        The former form: the value stays the plain loss, the gradient is w times its gradient.
        """
        target = u + (s - t)[:, :, None] * (
            theta[0] * u + theta[4] * u.mean(axis=1, keepdims=True) + theta[1]
        )
        per_chunk = flow.per_example(
            affine(*theta)(x, t, s) - jax.lax.stop_gradient(target), weight
        )
        plain = jax.lax.stop_gradient(per_chunk)
        w = (plain + 1e-3) ** -1.0
        return (plain + w * (per_chunk - plain)).mean()

    value, former_gradient = jax.value_and_grad(former)(theta)
    assert np.array_equal(gradient, former_gradient)
    assert value.item() == pytest.approx(metrics["loss_ref"].item(), rel=1e-6)


def test_meanflow_validation():
    """
    Invalid MeanFlow settings, the mean sampler on flow matching, end times for flow matching, one fixed time of
    MeanFlow's pairs without the other, and a loss that would draw times or noise without a key are refused.
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
    z = jnp.zeros((2, 3, 2))
    weight = jnp.ones((2, 3))
    with pytest.raises(ValueError):
        matching.loss(
            lambda x, t: x,
            z,
            weight,
            flow_time=jnp.zeros((2, 1)),
            end_time=jnp.ones((2, 1)),
        )
    for times in ({"end_time": jnp.ones((2, 1))}, {"flow_time": jnp.zeros((2, 1))}):
        with pytest.raises(ValueError):
            meanflow().loss(lambda x, t, s=None: x, z, weight, **times)
    # Without a key, only fixed times and fixed noise are allowed.
    for fixed in ({}, {"flow_time": jnp.zeros((2, 1))}, {"noise": z}):
        with pytest.raises(ValueError):
            matching.loss(lambda x, t: x, z, weight, **fixed)
    for fixed in ({}, {"noise": z}):
        with pytest.raises(ValueError):
            meanflow().loss(lambda x, t, s=None: x, z, weight, **fixed)
    with pytest.raises(ValueError):
        meanflow().loss(
            lambda x, t, s=None: x,
            z,
            weight,
            flow_time=jnp.zeros((2, 1)),
            end_time=jnp.ones((2, 1)),
        )


class AttentionNet(NamedTuple):
    """
    A one-layer attention network of (x, t, s), to compile MeanFlow's forward-mode pass and its gradient through
    einsum attention; a pytree of its weights, so the network is the argument it is differentiated by.
    """

    embed: dict[str, jax.Array]
    qkv: dict[str, jax.Array]
    out: dict[str, jax.Array]

    @classmethod
    def init(cls, key: jax.Array) -> AttentionNet:
        """
        An embedding of (x, t, s - t), one two-head attention layer and a linear readout.
        """
        embed, qkv, out = jax.random.split(key, 3)
        return cls(linear(embed, 4, 16), linear(qkv, 16, 48), linear(out, 16, 2))

    def __call__(self, x, t, s=None):
        """
        Attends over the chunk's steps.
        """
        s = t if s is None else s
        bs, steps = x.shape[:2]
        times = jnp.broadcast_to(
            jnp.concatenate((t, s - t), axis=1)[:, None], (bs, steps, 2)
        )
        h = dense(self.embed, jnp.concatenate((x, times), axis=-1))
        q, k, v = jnp.moveaxis(dense(self.qkv, h).reshape(bs, steps, 3, 2, 8), 2, 0)
        logits = jnp.einsum("bnhd,bmhd->bhnm", q, k) / math.sqrt(8)
        h = jnp.einsum("bhnm,bmhd->bnhd", jax.nn.softmax(logits, axis=-1), v)
        return dense(self.out, h.reshape(bs, steps, 16))


@pytest.mark.parametrize("tangent", ["conditional", "predicted"])
def test_meanflow_loss_jits(tangent):
    """
    MeanFlow's loss, its metrics and its gradients through attention compile once, the key an argument (torch:
    captured in a CUDA graph without host syncs): they equal the eager ones, a key reproduces them, different keys
    give different losses, and the gradients and metrics are finite.
    """
    flow = meanflow(
        time=LogitNormalTime(mean=0.6, std=1.6, support=(0.0, 1.0)),
        second_time=LogitNormalTime(mean=4.0, std=1.6, support=(0.0, 1.0)),
        order="clip",
        tangent=tangent,
        adaptive_power=0.75,
    )
    net = AttentionNet.init(jax.random.key(0))
    z = randn(np.random.default_rng(0), 32, 4, 2)
    weight = np.ones((32, 4), np.float32)
    weight[:8, 2:] = 0
    weight = jnp.asarray(weight)
    traces = []

    def loss(net, z, weight, key):
        """
        Computes the loss of the attention network and its metrics.
        """
        traces.append(key)
        return flow.loss(net, z, weight, key=key)

    step = jax.value_and_grad(loss, has_aux=True)
    compiled = jax.jit(step)
    keys = jax.random.split(jax.random.key(1), 3)
    results = [compiled(net, z, weight, key) for key in keys]
    assert len(traces) == 1
    assert tree_equal(compiled(net, z, weight, keys[0]), results[0])
    assert tree_allclose(step(net, z, weight, keys[0]), results[0], rtol=1e-5)
    losses = [loss.item() for (loss, _), _ in results]
    assert all(math.isfinite(v) for v in losses) and len(set(losses)) == 3
    for (_, metrics), gradients in results:
        assert all(np.isfinite(g).all() for g in jax.tree.leaves(gradients))
        assert all(math.isfinite(v.item()) and v.item() > 0 for v in metrics.values())
