r"""
Flow matching for action chunks: probability paths, flow-time samplers, predictions, the losses and ODE/SDE samplers.

Convention: noise at $t = 0$, data at $t = 1$, and sampling integrates upward in $t$.
$$
\begin{aligned}
x &= \alpha_t z + \beta_t \epsilon
&&\blacktriangleright\, z\ \text{a clean (normalized) chunk,}\ \epsilon \sim \mathcal N(0, I)
\ \text{independent of}\ z \\
u_t^{\text{target}}(x \mid z) &= \dot\alpha_t z + \dot\beta_t \epsilon
&&\blacktriangleright\,\text{the conditional velocity,}\ \tfrac{\mathrm d}{\mathrm dt}\ \text{of}\ x \\
\kappa_t &= \dot\alpha_t \beta_t - \alpha_t \dot\beta_t > 0
&&\blacktriangleright\,\text{on both paths here} \\
\beta_t \nabla \log p_t(x) &
&&\blacktriangleright\,\text{the scaled score, whose conditional regression target is}\ -\epsilon
\end{aligned}
$$
The network sees $(x, t)$ and predicts a velocity, the clean chunk or the scaled score (a Prediction); samplers map
its output to a velocity and integrate. Nothing is clamped: denominators that vanish at an endpoint are kept away
from it by rules checked when a training support or a time grid is built. Integration, adapters, losses and time
draws run in float32; a sampler's grid coefficients are float64 floats computed on the host.

Everything here is a pure function of its inputs: randomness enters as explicit keys (jax.random), so the losses and
samplers run inside jit, scan and vmap unchanged.

MeanFlow trains the average velocity over $[t, s]$ instead,
$\bar u_{t,s}(X_t) = \frac{1}{s - t}\int_t^s u_\tau^{\text{target}}(X_\tau)\,\mathrm d\tau$, which is the velocity at
$s = t$ and jumps $X_s = X_t + (s - t)\,\bar u_{t,s}(X_t)$ in one network evaluation, from noise to a chunk in the
extreme. Nothing here knows about Push-T: the network enters as predict(x, t[, s]), s omitted meaning $s = t$.

Shapes: bs is the batch size, H the chunk length and d the action dimension.
"""

from __future__ import annotations

import abc
import itertools
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import ndtri

# predict(x_t (bs, H, d), t (bs, 1)[, s (bs, 1)]) -> output (bs, H, d); s, for MeanFlow, omitted means s = t.
Predict = Callable[..., jax.Array]
TimeOrder = Literal["sort", "clip"]
Tangent = Literal["conditional", "predicted"]
Reduction = Literal["sum", "example_mean"]

HALF_PI = 0.5 * math.pi
# A velocity-space loss must not train at its prediction's singular time. A float32 draw can round onto a support end
# that lies within 2^-23 of that time, so such an end counts as reaching it.
SINGULAR_MARGIN = 2.0**-23
# A scaled-score prediction cannot be evaluated at t = 0 (its velocity divides by alpha); sampling starts here
# from N(0, I), an approximation of p_t at this time, not its exact marginal. Near 0 the first step multiplies the
# network's error by dt D / alpha (about 100 for 10 steps from 0.001), so the start stays clear of it: on Push-T
# 0.05 and 0.01 gave 0.87 held-out success where 0.001 gave 0.42.
SCALED_SCORE_START = 0.05


def _sin(t: jax.Array | np.ndarray) -> jax.Array | np.ndarray:
    r"""
    Evaluates $\sin t$: traced in the graph for an array of flow times, in float64 on the host for a grid time.
    """
    return jnp.sin(t) if isinstance(t, jax.Array) else np.sin(t)


class Coefficients(NamedTuple):
    """
    A probability path at one flow time, with the conversions between parameterizations at that time.
    Note: fields are arrays that broadcast against the chunk, or floats at a sampler's grid time.
    """

    alpha: jax.Array | float
    beta: jax.Array | float
    d_alpha: jax.Array | float
    d_beta: jax.Array | float
    D: jax.Array | float

    def interpolate(self, z: jax.Array, eps: jax.Array) -> jax.Array:
        r"""
        Samples the conditional path $x = \alpha_t z + \beta_t \epsilon$.
        Args:
            - z: clean chunks, shape (bs, H, d)
            - eps: Gaussian noise, shape (bs, H, d)
        Returns:
            - $x$: noisy chunks, shape (bs, H, d)
        """
        return self.alpha * z + self.beta * eps

    def velocity(self, z: jax.Array, eps: jax.Array) -> jax.Array:
        r"""
        Evaluates the conditional velocity $u_t^{\text{target}}(x \mid z) = \dot\alpha_t z + \dot\beta_t \epsilon$.
        Args:
            - z: clean chunks, shape (bs, H, d)
            - eps: Gaussian noise, shape (bs, H, d)
        Returns:
            - $u_t^{\text{target}}(x \mid z)$: conditional velocity, shape (bs, H, d)
        """
        return self.d_alpha * z + self.d_beta * eps

    def data_to_velocity(self, data: jax.Array, x: jax.Array) -> jax.Array:
        r"""
        Converts a clean-chunk prediction to a velocity: $u_t^\theta = (\dot\beta_t x + \kappa_t D_t^\theta)/\beta_t$.
        Note: undefined where $\beta_t = 0$ ($t = 1$).
        Args:
            - data: predicted clean chunks $D_t^\theta(x)$, shape (bs, H, d)
            - x: noisy chunks $x$, shape (bs, H, d)
        Returns:
            - $u_t^\theta(x)$: velocity, shape (bs, H, d)
        """
        return (self.d_beta * x + self.D * data) / self.beta

    def scaled_score_to_velocity(
        self, scaled_score: jax.Array, x: jax.Array
    ) -> jax.Array:
        r"""
        Converts a scaled-score prediction to a velocity:
        $u_t^\theta = (\dot\alpha_t x + \kappa_t \beta_t s_t^\theta)/\alpha_t$.
        Note: undefined where $\alpha_t = 0$ ($t = 0$).
        Args:
            - scaled_score: predicted $\beta_t s_t^\theta(x)$, shape (bs, H, d)
            - x: noisy chunks $x$, shape (bs, H, d)
        Returns:
            - $u_t^\theta(x)$: velocity, shape (bs, H, d)
        """
        return (self.d_alpha * x + self.D * scaled_score) / self.alpha

    def velocity_to_scaled_score(self, velocity: jax.Array, x: jax.Array) -> jax.Array:
        r"""
        Converts a velocity to the scaled score:
        $\beta_t \nabla \log p_t = (\alpha_t u_t^{\text{target}} - \dot\alpha_t x)/\kappa_t$.
        Args:
            - velocity: $u_t^{\text{target}}(x)$, shape (bs, H, d)
            - x: noisy chunks $x$, shape (bs, H, d)
        Returns:
            - $\beta_t \nabla \log p_t(x)$: scaled score, shape (bs, H, d)
        """
        return (self.alpha * velocity - self.d_alpha * x) / self.D

    def velocity_to_score(self, velocity: jax.Array, x: jax.Array) -> jax.Array:
        r"""
        Converts a velocity to the score:
        $\nabla \log p_t = (\alpha_t u_t^{\text{target}} - \dot\alpha_t x)/(\beta_t \kappa_t)$.
        Note: undefined where $\beta_t = 0$ ($t = 1$).
        Args:
            - velocity: $u_t^{\text{target}}(x)$, shape (bs, H, d)
            - x: noisy chunks $x$, shape (bs, H, d)
        Returns:
            - score: shape (bs, H, d)
        """
        return self.velocity_to_scaled_score(velocity, x) / self.beta


class ProbabilityPath(abc.ABC):
    r"""
    A Gaussian probability path $x = \alpha_t z + \beta_t \epsilon$ from noise ($t = 0$) to data ($t = 1$).
    """

    def __init__(self) -> None:
        r"""
        Checks the endpoints exactly in float32: $\alpha_0 = 0$, $\alpha_1 = 1$, $\beta_0 = 1$, $\beta_1 = 0$.
        """
        # Exact endpoints in float32: the samplers rely on alpha(0) = 0 and beta(1) = 0, not approximately 0. The check
        # runs eagerly even when a path is built inside a trace.
        with jax.ensure_compile_time_eval():
            ends = jnp.array([0.0, 1.0], jnp.float32)
            alpha, beta = (
                np.asarray(f(ends)).tolist() for f in (self.alpha, self.beta)
            )
        if alpha != [0.0, 1.0] or beta != [1.0, 0.0]:
            raise ValueError(
                f"{type(self).__name__} must run from noise (t = 0) to data (t = 1) exactly."
            )

    @abc.abstractmethod
    def alpha(self, t: jax.Array) -> jax.Array:
        r"""
        Evaluates $\alpha_t$. Should satisfy $\alpha_0 = 0$ and $\alpha_1 = 1$.
        Args:
            - t: flow times, any shape
        Returns:
            - $\alpha_t$, the same shape
        """

    @abc.abstractmethod
    def beta(self, t: jax.Array) -> jax.Array:
        r"""
        Evaluates $\beta_t$. Should satisfy $\beta_0 = 1$ and $\beta_1 = 0$.
        Args:
            - t: flow times, any shape
        Returns:
            - $\beta_t$, the same shape
        """

    @abc.abstractmethod
    def d_alpha(self, t: jax.Array) -> jax.Array | float:
        r"""
        Evaluates $\dot\alpha_t = \frac{\mathrm d}{\mathrm dt}\alpha_t$.
        Args:
            - t: flow times, any shape
        Returns:
            - $\dot\alpha_t$, the same shape, or a float when it is constant
        """

    @abc.abstractmethod
    def d_beta(self, t: jax.Array) -> jax.Array | float:
        r"""
        Evaluates $\dot\beta_t = \frac{\mathrm d}{\mathrm dt}\beta_t$.
        Args:
            - t: flow times, any shape
        Returns:
            - $\dot\beta_t$, the same shape, or a float when it is constant
        """

    def D(self, t: jax.Array) -> jax.Array | float:
        r"""
        Evaluates $\kappa_t = \dot\alpha_t \beta_t - \alpha_t \dot\beta_t$; paths with a closed form override this.
        Args:
            - t: flow times, any shape
        Returns:
            - $\kappa_t$, the same shape, or a float when it is constant
        """
        return self.beta(t) * self.d_alpha(t) - self.alpha(t) * self.d_beta(t)

    def __call__(self, t: jax.Array) -> Coefficients:
        """
        Evaluates the path at flow times $t$.
        Args:
            - t: flow times, shape broadcastable against the chunk, e.g. (bs, 1, 1)
        Returns:
            - the coefficients at $t$
        """
        return Coefficients(
            self.alpha(t), self.beta(t), self.d_alpha(t), self.d_beta(t), self.D(t)
        )

    def at(self, t: float) -> Coefficients:
        """
        Evaluates the path at one grid time, in float64 on the host.
        Args:
            - t: a flow time
        Returns:
            - the coefficients at $t$, as floats
        """
        c = self(np.float64(t))
        return Coefficients(*(float(v) for v in c))


class LinearPath(ProbabilityPath):
    r"""
    $\alpha_t = t$, $\beta_t = 1 - t$: straight lines from noise to data, with constant conditional velocity
    $z - \epsilon$.
    """

    def alpha(self, t: jax.Array) -> jax.Array:
        r"""
        Evaluates $\alpha_t = t$.
        """
        return t

    def beta(self, t: jax.Array) -> jax.Array:
        r"""
        Evaluates $\beta_t = 1 - t$.
        """
        return 1 - t

    def d_alpha(self, t: jax.Array) -> float:
        r"""
        Evaluates $\dot\alpha_t = 1$.
        """
        return 1.0

    def d_beta(self, t: jax.Array) -> float:
        r"""
        Evaluates $\dot\beta_t = -1$.
        """
        return -1.0

    def D(self, t: jax.Array) -> float:
        r"""
        Returns $\kappa_t = 1$.
        """
        return 1.0


class GVPPath(ProbabilityPath):
    r"""
    $\alpha_t = \sin\frac{\pi t}{2}$, $\beta_t = \cos\frac{\pi t}{2}$: variance preserving,
    $\alpha_t^2 + \beta_t^2 = 1$, $\kappa_t = \pi/2$.
    """

    def alpha(self, t: jax.Array) -> jax.Array:
        r"""
        Evaluates $\alpha_t = \sin\frac{\pi t}{2}$.
        """
        return _sin(HALF_PI * t)

    def beta(self, t: jax.Array) -> jax.Array:
        r"""
        Evaluates $\beta_t = \cos\frac{\pi t}{2}$, written as $\sin\frac{\pi (1 - t)}{2}$.
        """
        # sin(pi (1 - t) / 2) rather than cos(pi t / 2), whose float32 value at t = 1 is -4.4e-8, not 0.
        return _sin(HALF_PI * (1 - t))

    def d_alpha(self, t: jax.Array) -> jax.Array:
        r"""
        Evaluates $\dot\alpha_t = \frac{\pi}{2} \cos\frac{\pi t}{2} = \frac{\pi}{2} \beta_t$.
        """
        return HALF_PI * self.beta(t)

    def d_beta(self, t: jax.Array) -> jax.Array:
        r"""
        Evaluates $\dot\beta_t = -\frac{\pi}{2} \sin\frac{\pi t}{2} = -\frac{\pi}{2} \alpha_t$.
        """
        return -HALF_PI * self.alpha(t)

    def D(self, t: jax.Array) -> float:
        r"""
        Returns $\kappa_t = \pi/2$.
        """
        return HALF_PI


class TimeSampler(abc.ABC):
    r"""
    A training distribution of flow times, given by its inverse CDF $F^{-1}$ on a support $[t_{\min}, t_{\max}]$
    within $[0, 1]$.
    Note: a draw is $t = F^{-1}(u)$ with one uniform $u \in [0, 1)$ per example. Every distribution therefore draws
    the same $u$ from the same key, so runs that differ only in it have their times coupled through $u$ (their noise
    and batch order come from other keys). Nothing is clamped and nothing depends on array values, so it runs inside
    jit. $u$ can be exactly 0, so a draw can equal $t_{\min}$: no regression target divides by a path coefficient, and
    a velocity-space loss is kept off its singular time by the support rule of FlowMatching.
    """

    def __init__(self, *, support: tuple[float, float]) -> None:
        r"""
        Args:
            - support: $(t_{\min}, t_{\max})$ with $0 \le t_{\min} < t_{\max} \le 1$
        """
        low, high = (float(s) for s in support)
        if not 0.0 <= low < high <= 1.0:
            raise ValueError(
                f"The time support must satisfy 0 <= a < b <= 1, got {support}."
            )
        self.support = (low, high)

    @abc.abstractmethod
    def icdf(self, u: jax.Array) -> jax.Array:
        r"""
        Evaluates the inverse CDF.
        Args:
            - u: uniform draws in $[0, 1)$, 0 included, float32, any shape
        Returns:
            - flow times in $[t_{\min}, t_{\max}]$, float32, the same shape
        """

    def sample(self, key: jax.Array, n: int) -> jax.Array:
        """
        Draws one flow time per example.
        Args:
            - key: a PRNG key
            - n: number of draws
        Returns:
            - $t$: flow times in float32, shape (n, 1)
        """
        return self.icdf(jax.random.uniform(key, (n, 1), jnp.float32))


class UniformTime(TimeSampler):
    r"""
    $t \sim \mathrm{Unif}(t_{\min}, t_{\max})$.
    """

    def icdf(self, u: jax.Array) -> jax.Array:
        r"""
        Evaluates $F^{-1}(u) = t_{\min} + (t_{\max} - t_{\min})\,u$.
        """
        low, high = self.support
        return low + (high - low) * u


class LogitNormalTime(TimeSampler):
    r"""
    $\mathrm{logit}\,t \sim \mathcal N(\mu, \varsigma^2)$ on the whole of $[0, 1]$; a positive mean puts more times near
    the data.
    """

    def __init__(
        self, *, mean: float, std: float, support: tuple[float, float]
    ) -> None:
        r"""
        Args:
            - mean: mean of $\mathrm{logit}\,t$
            - std: standard deviation of $\mathrm{logit}\,t$, positive
            - support: $(0, 1)$ only; a uniform time takes a narrower support
        """
        if not std > 0:
            raise ValueError(f"std must be positive, got {std}.")
        if tuple(float(s) for s in support) != (0.0, 1.0):
            raise ValueError(
                f"LogitNormalTime draws on the whole of [0, 1], got the support {support}; restrict a uniform time "
                "instead."
            )
        self.mean, self.std = float(mean), float(std)
        super().__init__(support=support)

    def icdf(self, u: jax.Array) -> jax.Array:
        r"""
        Evaluates $F^{-1}(u) = \mathrm{sigmoid}\big(\mu + \varsigma\,\Phi^{-1}(u)\big)$; $u = 0$ gives $t = 0$.
        """
        return jax.nn.sigmoid(self.mean + self.std * ndtri(u))


class Prediction(abc.ABC):
    """
    What the network outputs at $(x, t)$: its regression target and its map to the velocity.
    """

    # The flow time at which to_velocity divides by zero (None: nowhere); samplers never evaluate it there.
    singular_time: float | None = None

    @abc.abstractmethod
    def target(self, z: jax.Array, eps: jax.Array, c: Coefficients) -> jax.Array:
        """
        Returns the conditional regression target, in the prediction's own space.
        Args:
            - z: clean chunks, shape (bs, H, d)
            - eps: the noise in $x$, shape (bs, H, d)
            - c: the path at the chunks' flow times
        Returns:
            - target, shape (bs, H, d)
        """

    @abc.abstractmethod
    def to_velocity(self, y: jax.Array, x: jax.Array, c: Coefficients) -> jax.Array:
        r"""
        Maps the network's output to the velocity of the marginal flow.
        Args:
            - y: network output, shape (bs, H, d)
            - x: noisy chunks $x$, shape (bs, H, d)
            - c: the path at the chunks' flow times
        Returns:
            - $u_t^\theta(x)$: velocity, shape (bs, H, d)
        """


class VelocityPrediction(Prediction):
    r"""
    The network predicts the velocity; its target is the conditional velocity $u_t^{\text{target}}(x \mid z)$.
    """

    def target(self, z: jax.Array, eps: jax.Array, c: Coefficients) -> jax.Array:
        r"""
        Returns the conditional velocity $u_t^{\text{target}}(x \mid z) = \dot\alpha_t z + \dot\beta_t \epsilon$.
        """
        return c.velocity(z, eps)

    def to_velocity(self, y: jax.Array, x: jax.Array, c: Coefficients) -> jax.Array:
        """
        Returns the prediction itself.
        """
        return y


class DataPrediction(Prediction):
    """
    The network predicts the clean chunk $z$; undefined as a velocity at $t = 1$.
    """

    singular_time = 1.0

    def target(self, z: jax.Array, eps: jax.Array, c: Coefficients) -> jax.Array:
        """
        Returns the clean chunk $z$.
        """
        return z

    def to_velocity(self, y: jax.Array, x: jax.Array, c: Coefficients) -> jax.Array:
        r"""
        Returns $u_t^\theta = (\dot\beta_t x + \kappa_t D_t^\theta)/\beta_t$.
        """
        return c.data_to_velocity(y, x)


class ScaledScorePrediction(Prediction):
    r"""
    The network predicts the scaled score $\beta_t s_t^\theta(x) = -\epsilon_t^\theta(x)$, whose target is $-\epsilon$;
    undefined as a velocity at $t = 0$.
    """

    singular_time = 0.0

    def target(self, z: jax.Array, eps: jax.Array, c: Coefficients) -> jax.Array:
        r"""
        Returns $-\epsilon$.
        """
        return -eps

    def to_velocity(self, y: jax.Array, x: jax.Array, c: Coefficients) -> jax.Array:
        r"""
        Returns $u_t^\theta = (\dot\alpha_t x + \kappa_t \beta_t s_t^\theta)/\alpha_t$.
        """
        return c.scaled_score_to_velocity(y, x)


@dataclass(frozen=True)
class Recipe:
    """
    A training recipe: what the network predicts, and whether the loss compares it after mapping it to a velocity.
    """

    prediction: Prediction
    velocity_loss: bool


RECIPES = {
    "velocity_mse": Recipe(VelocityPrediction(), velocity_loss=False),
    # Predicting z changes both the output and the effective weighting: in velocity terms the loss is weighted by
    # (beta / D)^2.
    "data_mse": Recipe(DataPrediction(), velocity_loss=False),
    # A data output scored in velocity space, which separates the two effects of data_mse.
    "data_output_velocity_loss": Recipe(DataPrediction(), velocity_loss=True),
    # The scaled score, regressed onto -eps.
    "scaled_score_mse": Recipe(ScaledScorePrediction(), velocity_loss=False),
}


class FlowMatching:
    """
    The training objective: a probability path, a flow-time distribution, a recipe and a loss reduction.
    Note: reductions of the squared error over a chunk's valid (step, coordinate) entries are
        - sum: summed per chunk, then averaged over chunks, so every valid step weighs the same;
        - example_mean: averaged per chunk, so a chunk cut short by the end of its episode weighs as much as a
          full one (its steps more).
    """

    # The scalar diagnostics loss() returns with the objective, by name (logged under the objective's section).
    metric_names: tuple[str, ...] = ()

    def __init__(
        self,
        *,
        path: ProbabilityPath,
        time: TimeSampler,
        recipe: str,
        reduction: Reduction,
    ) -> None:
        """
        Args:
            - path: the probability path
            - time: the training distribution of flow times
            - recipe: a key of RECIPES
            - reduction: sum | example_mean
        """
        if recipe not in RECIPES:
            raise ValueError(
                f"Unknown recipe {recipe!r}; choose from {sorted(RECIPES)}."
            )
        if reduction not in ("sum", "example_mean"):
            raise ValueError(f"Unknown reduction {reduction!r}.")
        self.path, self.time, self.reduction = path, time, reduction
        self.recipe = RECIPES[recipe]
        self.prediction = self.recipe.prediction
        singular = self.prediction.singular_time
        if self.recipe.velocity_loss and singular is not None:
            low, high = time.support
            if low - SINGULAR_MARGIN <= singular <= high + SINGULAR_MARGIN:
                raise ValueError(
                    f"Recipe {recipe!r} divides by zero at t = {singular}; draw the times uniformly on a support "
                    "that excludes it (e.g. flow/time=uniform 'flow.time.support=[0.0,0.99]')."
                )

    def per_example(self, error: jax.Array, weight: jax.Array) -> jax.Array:
        """
        Reduces each chunk's squared error over its valid entries, as the reduction says.
        Args:
            - error: prediction minus target, shape (bs, H, d)
            - weight: 1 on valid steps and 0 on padding, shape (bs, H)
        Returns:
            - the loss of every chunk, shape (bs,)
        """
        squared_error = jnp.square(error).sum(axis=2) * weight
        if self.reduction == "sum":
            return squared_error.sum(axis=1)
        return squared_error.sum(axis=1) / (weight.sum(axis=1) * error.shape[2])

    @staticmethod
    def draw_keys(
        key: jax.Array | None, flow_time: jax.Array | None, noise: jax.Array | None
    ) -> tuple[jax.Array | None, jax.Array | None]:
        """
        Splits a loss's key into the keys of its flow-time and noise draws.
        Args:
            - key: a PRNG key, or None when the times and the noise are both fixed
            - flow_time: fixed flow times, or None to draw them
            - noise: fixed noise, or None to draw it
        Returns:
            - time_key, noise_key: None without a key
        """
        if key is None:
            if flow_time is None or noise is None:
                raise ValueError("Drawing flow times or noise needs a key.")
            return None, None
        time_key, noise_key = jax.random.split(key)
        return time_key, noise_key

    def loss(
        self,
        predict: Predict,
        action_chunk: jax.Array,
        weight: jax.Array,
        *,
        key: jax.Array | None = None,
        flow_time: jax.Array | None = None,
        end_time: jax.Array | None = None,
        noise: jax.Array | None = None,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        """
        Computes the flow matching loss of a batch of clean chunks.
        Note: this runs inside the compiled training window, so no host syncs and no control flow on array values.
        Args:
            - predict: the network, predict(x_t (bs, H, d), t (bs, 1)) -> output (bs, H, d)
            - action_chunk: clean normalized chunks $z$, shape (bs, H, d)
            - weight: 1 on valid steps and 0 on padding, shape (bs, H)
            - key: the PRNG key of the flow times and noise drawn (None when both are fixed)
            - flow_time: fixed flow times instead of draws, shape (bs, 1)
            - end_time: MeanFlow's end times; flow matching takes none
            - noise: fixed Gaussian noise instead of draws, shape (bs, H, d)
        Returns:
            - loss: scalar, the objective
            - metrics: scalar diagnostics without gradient, by name (metric_names); none for flow matching
        """
        if end_time is not None:
            raise ValueError(
                "Flow matching trains the instantaneous velocity only (no end_time)."
            )
        time_key, noise_key = self.draw_keys(key, flow_time, noise)
        if flow_time is None:
            flow_time = self.time.sample(time_key, action_chunk.shape[0])
        if noise is None:
            # Pair each normalized expert chunk with independent Gaussian action noise.
            noise = jax.random.normal(noise_key, action_chunk.shape, action_chunk.dtype)
        # Broadcasting (bs, 1, 1) applies one flow time to every action in a chunk.
        c = self.path(flow_time[:, :, None])
        interpolated_chunk = c.interpolate(action_chunk, noise)
        # The network sees (condition, x_t, t), not noise and expert as separate inputs.
        prediction = predict(interpolated_chunk, flow_time)
        if self.recipe.velocity_loss:
            error = self.prediction.to_velocity(
                prediction, interpolated_chunk, c
            ) - c.velocity(action_chunk, noise)
        else:
            error = prediction - self.prediction.target(action_chunk, noise, c)
        return self.per_example(error, weight).mean(), {}


class MeanFlow(FlowMatching):
    r"""
    MeanFlow: the network predicts the average velocity over $[t, s]$, $s \ge t$, on the probability path,
    $$
    \bar u_{t,s}(X_t) = \frac{1}{s - t}\int_t^s u_\tau^{\text{target}}(X_\tau)\,\mathrm d\tau, \qquad
    X_s = X_t + (s - t)\,\bar u_{t,s}(X_t),
    $$
    the velocity $u_t^{\text{target}}(X_t)$ itself at $s = t$. Differentiating $(s - t)\,\bar u_{t,s}$ along the flow in
    $t$, with $s$ fixed, gives the identity
    $$
    \bar u_{t,s} = u_t^{\text{target}} + (s - t)\,\frac{\mathrm d}{\mathrm dt}\bar u_{t,s}, \qquad
    \frac{\mathrm d}{\mathrm dt}\bar u_{t,s} = (\partial_x \bar u_{t,s})\,u_t^{\text{target}} + \partial_t \bar u_{t,s},
    $$
    and the regression target
    $\mathrm{sg}\big(u_t^{\text{target}}(x \mid z) + (s - t)\,\frac{\mathrm d}{\mathrm dt}\bar u^\theta_{t,s}\big)$ with
    the conditional velocity $u_t^{\text{target}}(x \mid z)$ in place of $u_t^{\text{target}}(x)$;
    $\frac{\mathrm d}{\mathrm dt}\bar u^\theta_{t,s}$ is a Jacobian-vector product in $(x, t)$ with the tangent
    $(\mathrm{tangent}, 1)$, and $\mathrm{sg}$ is a stop-gradient.
    Note:
        - tangent: conditional uses $u_t^{\text{target}}(x \mid z)$ (MeanFlow, Geng et al. 2025); predicted uses the
          network's own $\bar u^\theta_{t,t}(x)$, without gradient (the boundary-condition variant of improved MeanFlow,
          Geng et al. 2025), except on padded steps, which have no loss and keep $u_t^{\text{target}}(x \mid z)$. Both
          give the same target in expectation.
        - time pairs: $a \sim$ time and $b \sim$ second_time (time when None), ordered by sort ($t = \min(a, b)$,
          $s = \max(a, b)$) or clip ($t = a$, $s = \max(a, b)$); then the first $\mathrm{round}(\text{share}\cdot n)$
          rows of a batch get $s = t$, flow matching rows. Under sort the flow matching rows' $t$ follows $\min(a, b)$;
          under clip the realized share of $s = t$ is $\text{share} + (1 - \text{share})\,P(b \le a)$.
        - the papers put noise at $t = 1$: their $(t, r)$ is $(1 - t, 1 - s)$ here, their logit-normal
          $\mathrm{LN}(\mu, \varsigma)$ is $\mathrm{LN}(-\mu, \varsigma)$, and their $u$ is $-\bar u_{t,s}$. MeanFlow's
          CIFAR-10 pairs (py-meanflow v1) are clip with $\mathrm{LN}(0.6, 1.6)$ and $\mathrm{LN}(4.0, 1.6)$; its
          ImageNet and improved MeanFlow's are sort with $\mathrm{LN}(0.4, 1.0)$ twice.
        - adaptive weight: the objective is the mean of $\lambda\,\ell$, with $\ell$ a chunk's reduced loss and
          $\lambda = (\ell + \delta)^{-p}$ ($\mathrm{sg}$), so each chunk's gradient is scaled by $\lambda$. At $p = 1$
          the gradient is that of $\log(\ell + \delta)$, so the reduction then barely matters and chunks cut short by
          padding weigh like full ones; $\delta$ is relative to $\ell$ (about 7 per chunk under sum, 0.2 under
          example_mean). The objective's value is then about 1 whatever the fit.
        - metrics, all of the plain $\ell$: loss_ref its mean over every row (MeanFlow's reference loss), loss_u over
          the rows with $s > t$ and loss_v over those with $s = t$, where the target is $u_t^{\text{target}}(x \mid z)$
          and the prediction $\bar u^\theta_{t,t}(x)$: the squared error of the instantaneous velocity. Improved
          MeanFlow's loss_u and loss_v cover every row, its loss_v from an auxiliary velocity head that is not carried
          over here. A batch without rows of one kind gives NaN.
    """

    metric_names = ("loss_ref", "loss_u", "loss_v")

    def __init__(
        self,
        *,
        path: ProbabilityPath,
        time: TimeSampler,
        second_time: TimeSampler | None,
        order: TimeOrder,
        flow_matching_share: float,
        tangent: Tangent,
        adaptive_power: float,
        adaptive_eps: float,
        reduction: Reduction,
    ) -> None:
        r"""
        Args:
            - path: the probability path
            - time: distribution of the first time of a pair
            - second_time: distribution of the second (None: time again)
            - order: sort | clip (needs second_time)
            - flow_matching_share: share of rows with $s = t$, in $[0, 1]$
            - tangent: conditional | predicted (needs flow_matching_share $> 0$)
            - adaptive_power: $p \ge 0$ of the adaptive weight; 0 is the plain squared error
            - adaptive_eps: $\delta > 0$ of the adaptive weight
            - reduction: sum | example_mean
        """
        super().__init__(
            path=path, time=time, recipe="velocity_mse", reduction=reduction
        )
        if order not in ("sort", "clip"):
            raise ValueError(f"Unknown order {order!r}.")
        if order == "clip" and second_time is None:
            raise ValueError(
                "clip draws s from second_time; with time alone half the pairs would collapse to s = t."
            )
        if tangent not in ("conditional", "predicted"):
            raise ValueError(f"Unknown tangent {tangent!r}.")
        if not 0.0 <= flow_matching_share <= 1.0:
            raise ValueError(
                f"flow_matching_share must lie in [0, 1], got {flow_matching_share}."
            )
        if tangent == "predicted" and not flow_matching_share > 0:
            raise ValueError(
                "The predicted tangent u_bar(x, t, t) is learned on the flow matching rows; set "
                "flow_matching_share > 0."
            )
        if not (adaptive_power >= 0 and adaptive_eps > 0):
            raise ValueError(
                f"The adaptive weight needs p >= 0 and eps > 0, got {adaptive_power} and {adaptive_eps}."
            )
        self.second_time, self.order, self.tangent = second_time, order, tangent
        self.flow_matching_share = flow_matching_share
        self.adaptive_power, self.adaptive_eps = adaptive_power, adaptive_eps

    def sample_times(self, key: jax.Array, n: int) -> tuple[jax.Array, jax.Array]:
        r"""
        Draws one time pair per example (see the class note).
        Note: the number of flow matching rows, $\mathrm{round}(\text{share}\cdot n)$, is fixed and the draws involve no
        control flow on array values, so this runs inside jit; batches are shuffled, so the rows by index are a fixed
        share of random examples.
        Args:
            - key: a PRNG key, split for the two draws
            - n: number of pairs
        Returns:
            - $t$: start times, shape (n, 1)
            - $s$: end times, $s \ge t$, shape (n, 1)
        """
        first_key, second_key = jax.random.split(key)
        a = self.time.sample(first_key, n)
        second = self.time if self.second_time is None else self.second_time
        b = second.sample(second_key, n)
        s = jnp.maximum(a, b)
        t = jnp.minimum(a, b) if self.order == "sort" else a
        row = jnp.arange(n)[:, None]
        return t, jnp.where(row < round(self.flow_matching_share * n), t, s)

    @staticmethod
    def average_velocity(
        predict: Predict,
        x: jax.Array,
        t: jax.Array,
        s: jax.Array,
        tangent: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        r"""
        Evaluates $\bar u^\theta_{t,s}(x)$ and its derivative along the flow,
        $(\partial_x \bar u^\theta_{t,s})\,\mathrm{tangent} + \partial_t \bar u^\theta_{t,s}$ with $s$ fixed, in one
        forward-mode pass (jax.jvp).
        Args:
            - predict: the network, predict(x, t, s) -> u_bar
            - x: noisy chunks $x$, shape (bs, H, d)
            - t: start times, shape (bs, 1)
            - s: end times, shape (bs, 1)
            - tangent: the velocity that moves $x$ along the flow, shape (bs, H, d)
        Returns:
            - $\bar u^\theta_{t,s}(x)$: shape (bs, H, d)
            - $\frac{\mathrm d}{\mathrm dt}\bar u^\theta_{t,s}(x)$: shape (bs, H, d)
        """
        return jax.jvp(
            lambda x, t: predict(x, t, s), (x, t), (tangent, jnp.ones_like(t))
        )

    def loss(
        self,
        predict: Predict,
        action_chunk: jax.Array,
        weight: jax.Array,
        *,
        key: jax.Array | None = None,
        flow_time: jax.Array | None = None,
        end_time: jax.Array | None = None,
        noise: jax.Array | None = None,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        r"""
        Computes the MeanFlow loss of a batch of clean chunks.
        Note: this runs inside the compiled training window, so no host syncs and no control flow on array values.
        Args:
            - predict: the network, predict(x_t (bs, H, d), t (bs, 1)[, s (bs, 1)]) -> u_bar (bs, H, d)
            - action_chunk: clean normalized chunks $z$, shape (bs, H, d)
            - weight: 1 on valid steps and 0 on padding, shape (bs, H)
            - key: the PRNG key of the time pairs and noise drawn (None when both are fixed)
            - flow_time: fixed start times instead of draws (with end_time), shape (bs, 1)
            - end_time: fixed end times $s \ge$ flow_time (with flow_time), shape (bs, 1)
            - noise: fixed Gaussian noise instead of draws, shape (bs, H, d)
        Returns:
            - loss: scalar, the adaptively weighted objective
            - metrics: loss_ref, loss_u and loss_v (see the class note), scalars without gradient
        """
        if (flow_time is None) != (end_time is None):
            raise ValueError("Fix both times of the pairs or neither.")
        time_key, noise_key = self.draw_keys(key, flow_time, noise)
        if flow_time is None:
            flow_time, end_time = self.sample_times(time_key, action_chunk.shape[0])
        if noise is None:
            noise = jax.random.normal(noise_key, action_chunk.shape, action_chunk.dtype)
        c = self.path(flow_time[:, :, None])
        interpolated_chunk = c.interpolate(action_chunk, noise)
        velocity = c.velocity(action_chunk, noise)
        move = velocity
        if self.tangent == "predicted":
            predicted = jax.lax.stop_gradient(predict(interpolated_chunk, flow_time))
            move = jnp.where(weight[:, :, None] > 0, predicted, velocity)
        average, derivative = self.average_velocity(
            predict, interpolated_chunk, flow_time, end_time, move
        )
        target = velocity + (end_time - flow_time)[:, :, None] * derivative
        loss = self.per_example(average - jax.lax.stop_gradient(target), weight)
        plain = jax.lax.stop_gradient(loss)
        adaptive_weight = (plain + self.adaptive_eps) ** -self.adaptive_power
        same = (end_time == flow_time)[:, 0].astype(loss.dtype)
        metrics = {
            "loss_ref": plain.mean(),
            "loss_u": (plain * (1 - same)).sum() / (1 - same).sum(),
            "loss_v": (plain * same).sum() / same.sum(),
        }
        return (adaptive_weight * loss).mean(), metrics


class ODE(abc.ABC):
    r"""
    An ordinary differential equation $\frac{\mathrm dx}{\mathrm dt} = \mathrm{drift}(x, t)$.
    """

    @abc.abstractmethod
    def drift(self, x: jax.Array, t: float) -> jax.Array:
        """
        Returns the drift of the ODE.
        Args:
            - x: state at time $t$, shape (bs, H, d)
            - t: time
        Returns:
            - drift, shape (bs, H, d)
        """


class SDE(ODE):
    r"""
    A stochastic differential equation $\mathrm dx = \mathrm{drift}(x, t)\,\mathrm dt + \sigma_t\,\mathrm dW_t$.
    """

    @abc.abstractmethod
    def diffusion(self, t: float) -> float:
        r"""
        Returns the diffusion coefficient $\sigma_t$ of the SDE.
        Args:
            - t: time
        Returns:
            - $\sigma_t$
        """


class LearnedFlow(ODE):
    r"""
    $\frac{\mathrm dx}{\mathrm dt} = u_t^\theta(x)$: the network's prediction at $(x, t)$, mapped to a velocity. Counts
    network evaluations.
    Note: for MeanFlow, average gives the network's average velocity over a step and $u_t^\theta(x)$ is
    $\bar u^\theta_{t,t}(x)$. Under jit the count is taken while tracing, which the samplers' Python loops over their
    grids make exact.
    """

    def __init__(
        self, predict: Predict, path: ProbabilityPath, prediction: Prediction
    ) -> None:
        """
        Args:
            - predict: the network, predict(x_t (bs, H, d), t (bs, 1)[, s (bs, 1)]) -> output (bs, H, d)
            - path: the probability path the network was trained on
            - prediction: what the network outputs
        """
        self.predict, self.path, self.prediction = predict, path, prediction
        self.calls = 0

    def drift(self, x: jax.Array, t: float) -> jax.Array:
        """
        Evaluates the network at $(x, t)$ and maps its prediction to a velocity; counts the call.
        Args and Returns: see ODE.drift.
        """
        self.calls += 1
        y = self.predict(x, jnp.full((x.shape[0], 1), t, x.dtype))
        return self.prediction.to_velocity(y, x, self.path.at(t))

    def average(self, x: jax.Array, t: float, s: float) -> jax.Array:
        r"""
        Evaluates a MeanFlow network's average velocity over $[t, s]$ at $x$; counts the call.
        Args:
            - x: state at time $t$, shape (bs, H, d)
            - t: time
            - s: end time, $s > t$
        Returns:
            - $\bar u^\theta_{t,s}(x)$, shape (bs, H, d)
        """
        self.calls += 1
        times = [jnp.full((x.shape[0], 1), time, x.dtype) for time in (t, s)]
        return self.predict(x, *times)


class LangevinFlow(SDE):
    r"""
    $\mathrm dx = \big[u_t^\theta(x) + \frac{\sigma_t^2}{2} s_t^\theta(x)\big]\,\mathrm dt + \sigma_t\,\mathrm dW_t$
    with $\sigma_t^2 = 2\eta\,\alpha_t \beta_t$, which keeps the marginals of the flow.
    Note: the velocity of any path and prediction gives the score,
    $s_t^\theta = (\alpha_t u_t^\theta - \dot\alpha_t x)/(\beta_t \kappa_t)$ with the scaled score
    $\beta_t s_t^\theta = (\alpha_t u_t^\theta - \dot\alpha_t x)/\kappa_t$, and $\sigma_t^2/2$ cancels its $1/\beta_t$:
    the drift $u_t^\theta - \eta\,\alpha_t (\dot\alpha_t x - \alpha_t u_t^\theta)/\kappa_t$ is bounded wherever
    $u_t^\theta$ is, up to $t = 1$ for velocity and scaled-score predictions. A data prediction's $u_t^\theta$ divides
    by $\beta_t$, so, as for its ODE, the sampler never evaluates it at $t = 1$ (the SDE stops at sde_end $< 1$).
    On the linear path ($\alpha_t = t$, $\dot\alpha_t = \kappa_t = 1$) it is $u_t^\theta - \eta t\,(x - t\,u_t^\theta)$,
    with $\sigma_t^2 = 2\eta t (1 - t)$.
    """

    def __init__(self, flow: LearnedFlow, eta: float) -> None:
        r"""
        Args:
            - flow: the learned flow
            - eta: noise scale; $\eta = 0$ gives back the ODE
        """
        self.flow, self.eta = flow, eta

    def drift(self, x: jax.Array, t: float) -> jax.Array:
        r"""
        Returns $u_t^\theta - \eta\,\alpha_t (\dot\alpha_t x - \alpha_t u_t^\theta)/\kappa_t$.
        """
        c = self.flow.path.at(t)
        v = self.flow.drift(x, t)
        return v - self.eta * c.alpha * (c.d_alpha * x - c.alpha * v) / c.D

    def diffusion(self, t: float) -> float:
        r"""
        Returns $\sigma_t = \sqrt{2\eta\,\alpha_t \beta_t}$.
        """
        c = self.flow.path.at(t)
        return math.sqrt(2 * self.eta * c.alpha * c.beta)


class Simulator(abc.ABC):
    """
    A one-step rule for integrating an ODE or SDE along a time grid.
    """

    # Network evaluations per step.
    nfe_per_step: int

    @abc.abstractmethod
    def step(
        self, x: jax.Array, t: float, t_next: float, noise: jax.Array | None
    ) -> jax.Array:
        r"""
        Takes one step from $t$ to $t_{\text{next}}$.
        Args:
            - x: state at time $t$, shape (bs, H, d)
            - t: time
            - t_next: next time
            - noise: $\mathcal N(0, I)$ draws for a stochastic step, shape (bs, H, d), else None
        Returns:
            - state at time $t_{\text{next}}$, shape (bs, H, d)
        """

    def simulate(
        self, x: jax.Array, times: list[float], noise: jax.Array | None = None
    ) -> jax.Array:
        r"""
        Simulates along a time grid.
        Args:
            - x: state at times[0], shape (bs, H, d)
            - times: the grid, increasing
            - noise: $\mathcal N(0, I)$ draws for stochastic steps, shape (bs, len(times) - 1, H, d)
        Returns:
            - state at times[-1], shape (bs, H, d)
        """
        for i in range(len(times) - 1):
            x = self.step(
                x, times[i], times[i + 1], None if noise is None else noise[:, i]
            )
        return x


class Euler(Simulator):
    r"""
    Euler: $x + (t_{\text{next}} - t)\,\mathrm{drift}(x, t)$, one network evaluation at the left end of each step.
    """

    nfe_per_step = 1

    def __init__(self, ode: ODE) -> None:
        """
        Args:
            - ode: the ODE to integrate
        """
        self.ode = ode

    def step(
        self, x: jax.Array, t: float, t_next: float, noise: jax.Array | None
    ) -> jax.Array:
        r"""
        Takes $x + (t_{\text{next}} - t)\,\mathrm{drift}(x, t)$. Args and Returns: see Simulator.step.
        """
        return x + (t_next - t) * self.ode.drift(x, t)


class Heun(Simulator):
    """
    Euler predictor and trapezoid corrector; the corrector evaluates the right end of every step.
    """

    nfe_per_step = 2

    def __init__(self, ode: ODE) -> None:
        """
        Args:
            - ode: the ODE to integrate
        """
        self.ode = ode

    def step(
        self, x: jax.Array, t: float, t_next: float, noise: jax.Array | None
    ) -> jax.Array:
        r"""
        Takes $x + \frac{h}{2}(\mathrm{drift}(x, t) + \mathrm{drift}(x + h\,\mathrm{drift}(x, t), t_{\text{next}}))$.
        Args and Returns: see Simulator.step.
        """
        dt = t_next - t
        v = self.ode.drift(x, t)
        return x + 0.5 * dt * (v + self.ode.drift(x + dt * v, t_next))


class MeanFlowStep(Simulator):
    r"""
    MeanFlow's step: $x + (t_{\text{next}} - t)\,\bar u^\theta_{t,t_{\text{next}}}(x)$, one evaluation of the average
    velocity over the step, exact on any grid for an exact $\bar u_{t,s}$.
    """

    nfe_per_step = 1

    def __init__(self, flow: LearnedFlow) -> None:
        """
        Args:
            - flow: the learned flow of a MeanFlow network
        """
        self.flow = flow

    def step(
        self, x: jax.Array, t: float, t_next: float, noise: jax.Array | None
    ) -> jax.Array:
        r"""
        Takes $x + (t_{\text{next}} - t)\,\bar u^\theta_{t,t_{\text{next}}}(x)$. Args and Returns: see Simulator.step.
        """
        return x + (t_next - t) * self.flow.average(x, t, t_next)


class EulerMaruyama(Simulator):
    r"""
    Euler-Maruyama: $x + h\,\mathrm{drift}(x, t) + \sigma_t \sqrt{h}\,\mathrm{noise}$.
    """

    nfe_per_step = 1

    def __init__(self, sde: SDE) -> None:
        """
        Args:
            - sde: the SDE to integrate
        """
        self.sde = sde

    def step(
        self, x: jax.Array, t: float, t_next: float, noise: jax.Array | None
    ) -> jax.Array:
        r"""
        Takes $x + h\,\mathrm{drift}(x, t) + \sigma_t \sqrt{h}\,\mathrm{noise}$. Args and Returns: see Simulator.step.
        """
        dt = t_next - t
        return (
            x
            + dt * self.sde.drift(x, t)
            + self.sde.diffusion(t) * math.sqrt(dt) * noise
        )


@dataclass(frozen=True)
class Sampler:
    """
    How a chunk is generated from noise: the solver, its network evaluations (NFE), the time grid and guidance.
    Note: NFE counts every network evaluation, every solver stage and the SDE's final step included.
    """

    # euler (1 NFE per step) | heun (2) | sde (Euler-Maruyama on [start, sde_end], then one Euler step to 1) |
    # mean (MeanFlow's average velocity over each step, 1; MeanFlow only).
    solver: Literal["euler", "heun", "sde", "mean"]
    nfe: int
    # uniform t_i = i / n | data_dense t_i = 1 - (1 - i / n)^rho (smaller steps near the data), from start_time.
    grid: Literal["uniform", "data_dense"]
    rho: float
    # None takes the prediction's rule: 0, or SCALED_SCORE_START for a scaled score.
    start_time: float | None
    sde_eta: float
    sde_end: float
    # Classifier-free guidance weight w, (1 - w) v(null) + w v(condition); 1 is unguided.
    cfg_scale: float

    def __post_init__(self) -> None:
        """
        Raises on settings that do not fit together.
        """
        if self.solver not in ("euler", "heun", "sde", "mean"):
            raise ValueError(f"Unknown solver {self.solver!r}.")
        if self.grid not in ("uniform", "data_dense"):
            raise ValueError(f"Unknown grid {self.grid!r}.")
        if not (isinstance(self.nfe, int) and self.nfe >= 1):
            raise ValueError(f"nfe must be a positive integer, got {self.nfe!r}.")
        if self.solver == "heun" and self.nfe % 2:
            raise ValueError(
                "Heun takes 2 network evaluations per step, so nfe must be even."
            )
        if self.solver == "sde":
            if self.nfe < 2 or self.grid != "uniform":
                raise ValueError("The SDE sampler needs nfe >= 2 on the uniform grid.")
            if not (0.0 < self.sde_end < 1.0 and self.sde_eta >= 0.0):
                raise ValueError(
                    "The SDE sampler needs 0 < sde_end < 1 and sde_eta >= 0."
                )
        if self.grid == "data_dense" and not self.rho >= 1.0:
            raise ValueError(f"data_dense needs rho >= 1, got {self.rho}.")

    @property
    def num_brownian(self) -> int:
        r"""
        Returns the number of $\mathcal N(0, I)$ draws per chunk entry that one query needs (its stochastic steps).
        """
        return self.nfe - 1 if self.solver == "sde" else 0

    def times(self, prediction: Prediction) -> list[float]:
        r"""
        Resolves the time grid, checked against the prediction's singular time.
        Args:
            - prediction: what the network outputs
        Returns:
            - the grid $t_0 < \dots < t_n = 1$, as float32 values
        """
        start = self.start_time
        if start is None:
            start = SCALED_SCORE_START if prediction.singular_time == 0.0 else 0.0
        if self.solver == "sde":
            steps = self.nfe - 1
            grid = [
                start + (self.sde_end - start) * i / steps for i in range(steps + 1)
            ]
            grid.append(1.0)
        else:
            steps = self.nfe // (2 if self.solver == "heun" else 1)
            rho = self.rho if self.grid == "data_dense" else 1.0
            grid = [1 - (1 - start) * (1 - i / steps) ** rho for i in range(steps + 1)]
        times = np.asarray(grid, np.float64).astype(np.float32).tolist()
        if not all(a < b for a, b in itertools.pairwise(times)) or not 0.0 <= times[0]:
            raise ValueError(f"The time grid is not increasing in float32: {times}.")
        # Euler and mean steps evaluate the left end of each step, Heun both ends.
        evaluated = times if self.solver == "heun" else times[:-1]
        if prediction.singular_time in evaluated:
            raise ValueError(
                f"A {type(prediction).__name__} cannot be evaluated at t = {prediction.singular_time}; "
                f"the {self.solver} sampler on this grid would."
            )
        return times

    def check(self, flow: FlowMatching) -> None:
        """
        Raises if this sampler cannot run on a policy trained with this objective.
        Args:
            - flow: the objective the policy was trained with
        """
        self.times(flow.prediction)
        if self.solver == "mean" and not isinstance(flow, MeanFlow):
            raise ValueError(
                "The mean sampler needs a policy trained with MeanFlow (flow=meanflow)."
            )

    def sample(
        self,
        flow: LearnedFlow,
        noise: jax.Array,
        brownian: jax.Array | None = None,
    ) -> jax.Array:
        r"""
        Integrates from noise at the grid's start to a chunk at $t = 1$.
        Note: runs inside jit, so it cannot look at the values; the caller checks the result is finite (a non-finite
        value at any step stays non-finite, so the result covers the whole trajectory).
        Args:
            - flow: the learned flow
            - noise: initial $\mathcal N(0, I)$ chunks, shape (bs, H, d)
            - brownian: $\mathcal N(0, I)$ draws for the stochastic steps, shape (bs, num_brownian, H, d)
        Returns:
            - chunks at $t = 1$, shape (bs, H, d)
        """
        times = self.times(flow.prediction)
        if (brownian is None) != (self.num_brownian == 0):
            raise ValueError(
                f"The {self.solver} sampler takes {self.num_brownian} Brownian draws per entry."
            )
        flow.calls = 0
        x = noise
        if self.solver == "sde":
            x = EulerMaruyama(LangevinFlow(flow, self.sde_eta)).simulate(
                x, times[:-1], brownian
            )
            x = Euler(flow).simulate(x, times[-2:])
        else:
            simulator = {"euler": Euler, "heun": Heun, "mean": MeanFlowStep}
            x = simulator[self.solver](flow).simulate(x, times)
        if flow.calls != self.nfe:
            raise RuntimeError(
                f"The sampler made {flow.calls} network evaluations, {self.nfe} requested."
            )
        return x
