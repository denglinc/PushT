"""
Flow matching action chunking policies for Push-T, as Flax NNX modules.

FlowMatchingPolicy holds the contract (compute_loss for training, sample_actions for evaluation) and the flow
matching objective; a subclass supplies the network that maps (condition, noisy chunk, flow time) to a prediction:
    - DiTPolicy: a diffusion transformer with one token per chunk step. The flow time conditions every block through
      adaLN-Zero; the goal and history tokens enter by adaln (mean-pooled into the conditioning vector), concat
      (prepended tokens) or cross (a cross-attention sub-layer per block).
    - MLPPolicy: the flattened observation, chunk and flow time through a ReLU MLP.
With the MeanFlow objective the network predicts the average velocity over $[t, s]$ and also sees the gap
$h = s - t$: the DiT adds a gap embedding to the time embedding, the MLP takes $h$ as one more input. Both start at
zero, so the average velocity starts as the instantaneous one for every $s$.

A policy's state is its Params and nothing else (no stored keys, no dropout layers, no arrays outside Params), so
nnx.split(policy, nnx.Param) gives the whole state and nnx.merge(graphdef, params) rebuilds the policy from a plain
dict of arrays inside jit, vmap and scan. Static attributes are hashable (tuples, not lists). Randomness enters as
keys: condition dropout draws from the key compute_loss is given. Layers are initialized as torch's defaults, so the
hyperparameters tuned in PushT (PyTorch) carry over.

Shapes: bs is the batch size, K the observation history, H the chunk length, d the action dimension and D the width.
"""

from __future__ import annotations

import abc
import math
from collections.abc import Sequence
from typing import Literal

import jax
import jax.numpy as jnp
from flax import nnx

from flow import FlowMatching, LearnedFlow, MeanFlow, Predict, Sampler

Conditioning = Literal["adaln", "concat", "cross"]
# A parameter's role, for the optimizers:
#   - hidden: the weight matrices between hidden layers, Muon's domain;
#   - boundary: the weight matrices of the layers that read the inputs or write the prediction;
#   - vector: biases, positions, the null condition and the time frequencies.
# Weight decay acts on the matrices (hidden and boundary), never on the vectors.
Role = Literal["hidden", "boundary", "vector"]


def linear(din: int, dout: int, *, rngs: nnx.Rngs, zero: bool = False) -> nnx.Linear:
    r"""
    Builds a linear layer initialized as torch's nn.Linear: kernel and bias drawn from
    $\mathcal U(-1/\sqrt{d_{in}}, 1/\sqrt{d_{in}})$, with $d_{in}$ the input features.
    Args:
        - din: input features
        - dout: output features
        - rngs: the initialization keys
        - zero: start the kernel and the bias at zero instead
    Returns:
        - the layer, with kernel shape (din, dout)
    """
    if zero:
        zeros = nnx.initializers.zeros
        return nnx.Linear(din, dout, kernel_init=zeros, bias_init=zeros, rngs=rngs)
    bound = 1.0 / math.sqrt(din)

    def bias_init(key: jax.Array, shape: Sequence[int], dtype=jnp.float32) -> jax.Array:
        r"""
        Draws the bias from $\mathcal U(-1/\sqrt{d_{in}}, 1/\sqrt{d_{in}})$.
        """
        return jax.random.uniform(key, shape, dtype, -bound, bound)

    # Variance scaling 1/3 over fan_in, uniform: the limit sqrt(3 / 3 / din) = 1/sqrt(din), torch's kaiming bound.
    kernel_init = nnx.initializers.variance_scaling(1 / 3, "fan_in", "uniform")
    return nnx.Linear(
        din, dout, kernel_init=kernel_init, bias_init=bias_init, rngs=rngs
    )


def rms_norm(x: jax.Array) -> jax.Array:
    """
    RMS normalization over the last axis without a learned scale, with torch's default epsilon (the dtype's).
    Args:
        - x: tokens, shape (..., D)
    Returns:
        - normalized tokens, the same shape
    """
    eps = jnp.finfo(x.dtype).eps
    return x * jax.lax.rsqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + eps)


def attention(q: jax.Array, k: jax.Array, v: jax.Array) -> jax.Array:
    r"""
    Scaled dot-product attention, $\mathrm{softmax}(q k^\top / \sqrt{d_h})\,v$ with $d_h$ the head dimension, the
    softmax in float32 or wider (float64 for a float64 network).
    Note: plain XLA ops, so forward-mode differentiation (MeanFlow's jax.jvp) passes through it.
    Args:
        - q: queries, shape (bs, N, heads, head_dim)
        - k: keys, shape (bs, M, heads, head_dim)
        - v: values, shape (bs, M, heads, head_dim)
    Returns:
        - attended values, shape (bs, N, heads, head_dim)
    """
    logits = jnp.einsum("bnhd,bmhd->bhnm", q, k) / math.sqrt(q.shape[-1])
    dtype = jnp.promote_types(logits.dtype, jnp.float32)
    weights = jax.nn.softmax(logits.astype(dtype), axis=-1).astype(v.dtype)
    return jnp.einsum("bhnm,bmhd->bnhd", weights, v)


def modulate(x: jax.Array, scale: jax.Array, bias: jax.Array) -> jax.Array:
    r"""
    Applies an adaLN modulation $(1 + \gamma) \odot x + \beta$.
    Args:
        - x: tokens, shape (bs, N, D)
        - scale: shape (bs, 1, D)
        - bias: shape (bs, 1, D)
    Returns:
        - modulated tokens, shape (bs, N, D)
    """
    return x * (1 + scale) + bias


class FlowMatchingPolicy(nnx.Module):
    """
    Predicts relative action chunks with a flow matching objective; subclasses supply the network.
    Note: compute_loss's key also draws condition dropout: a share cond_dropout of each batch sees the learned null
    condition instead of its own, which trains the unconditional prediction that guidance needs.
    """

    def __init__(
        self,
        obs_dim: int,
        goal_dim: int,
        action_dim: int,
        obs_horizon: int,
        pred_horizon: int,
        *,
        flow: FlowMatching,
        cond_dropout: float,
    ) -> None:
        """
        Args:
            - obs_dim: features per history frame
            - goal_dim: features of the goal token
            - action_dim: d
            - obs_horizon: K
            - pred_horizon: H
            - flow: the flow matching objective, which also fixes what the network predicts
            - cond_dropout: share of training examples that see the null condition, in $[0, 1)$
        """
        if not 0.0 <= cond_dropout < 1.0:
            raise ValueError(f"cond_dropout must lie in [0, 1), got {cond_dropout}.")
        self.obs_dim = obs_dim
        self.goal_dim = goal_dim
        self.action_dim = action_dim
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.flow = flow
        self.cond_dropout = cond_dropout

    @abc.abstractmethod
    def encode_condition(self, obs: jax.Array, goal: jax.Array) -> jax.Array:
        """
        Encodes the observation into the network's condition.
        Args:
            - obs: normalized history frames, shape (bs, K, obs_dim)
            - goal: normalized goal, shape (bs, goal_dim)
        Returns:
            - condition, shape (bs, ...)
        """

    @abc.abstractmethod
    def null_condition(self, batch_size: int) -> jax.Array:
        """
        Returns the learned condition of 'no observation'.
        Args:
            - batch_size: bs
        Returns:
            - null condition, shaped like encode_condition's output
        """

    @abc.abstractmethod
    def boundary_layers(self) -> list[nnx.Linear]:
        """
        Returns the linear layers that read the inputs or write the prediction; every other linear layer is hidden.
        """

    def parameter_roles(self) -> dict:
        """
        Labels the parameters by role (see Role).
        Returns:
            - a plain dict of roles with the structure of nnx.to_pure_dict(nnx.state(self, nnx.Param))
        """
        # Variables are unhashable; their identities tell the layers apart.
        boundary = {id(layer.kernel) for layer in self.boundary_layers()}
        matrices = {
            id(m.kernel) for _, m in nnx.iter_modules(self) if isinstance(m, nnx.Linear)
        }

        def role(variable: nnx.Variable) -> Role:
            """
            Returns the role of one parameter.
            """
            if id(variable) in boundary:
                return "boundary"
            return "hidden" if id(variable) in matrices else "vector"

        return nnx.to_pure_dict(nnx.state(self, nnx.Param), extract_fn=role)

    @property
    def mean_flow(self) -> bool:
        """
        Whether the policy is trained with MeanFlow, so that it predicts average velocities.
        """
        return isinstance(self.flow, MeanFlow)

    @abc.abstractmethod
    def _predict(
        self,
        condition: jax.Array,
        action_chunk: jax.Array,
        flow_time: jax.Array,
        end_time: jax.Array | None = None,
    ) -> jax.Array:
        r"""
        Predicts what the recipe asks for (velocity, clean chunk or scaled score) at the current flow state.
        Note: a MeanFlow network sees the gap $h = s - t$, formed here so that a derivative in $t$ also moves $h$.
        Args:
            - condition: the encoded observation, shape (bs, ...)
            - action_chunk: noisy chunks $x$, shape (bs, H, d)
            - flow_time: $t$, shape (bs, 1)
            - end_time: $s \ge t$ of a MeanFlow average velocity, shape (bs, 1); None is $s = t$
        Returns:
            - prediction, shape (bs, H, d)
        """

    def gap(self, flow_time: jax.Array, end_time: jax.Array | None) -> jax.Array | None:
        """
        Returns the gap $h = s - t$ a MeanFlow network sees, 0 when end_time is None; None for flow matching.
        Args:
            - flow_time: $t$, shape (bs, 1)
            - end_time: $s$, shape (bs, 1), or None
        Returns:
            - $h$, shape (bs, 1), or None
        """
        if not self.mean_flow:
            if end_time is not None:
                raise ValueError(
                    "A flow matching policy predicts the instantaneous velocity only (no end_time)."
                )
            return None
        return (flow_time if end_time is None else end_time) - flow_time

    def condition(
        self, obs: jax.Array, goal: jax.Array, *, dropout_key: jax.Array | None = None
    ) -> jax.Array:
        """
        Encodes the observation, with condition dropout when given a key.
        Args:
            - obs: shape (bs, K, obs_dim)
            - goal: shape (bs, goal_dim)
            - dropout_key: draws which examples see the null condition; None keeps every condition
        Returns:
            - condition, shape (bs, ...)
        """
        condition = self.encode_condition(obs, goal)
        if dropout_key is not None and self.cond_dropout > 0:
            keep = jax.random.uniform(dropout_key, (obs.shape[0],)) >= self.cond_dropout
            keep = keep.reshape(-1, *[1] * (condition.ndim - 1))
            condition = jnp.where(keep, condition, self.null_condition(obs.shape[0]))
        return condition

    # Training calls compute_loss; evaluation calls sample_actions.
    def compute_loss(
        self,
        obs: jax.Array,
        goal: jax.Array,
        action_chunk: jax.Array,
        weight: jax.Array,
        *,
        key: jax.Array | None = None,
        flow_time: jax.Array | None = None,
        end_time: jax.Array | None = None,
        noise: jax.Array | None = None,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        """
        Computes the training loss of a batch.
        Args:
            - obs: shape (bs, K, obs_dim)
            - goal: shape (bs, goal_dim)
            - action_chunk: normalized expert chunks, shape (bs, H, d)
            - weight: 0 on padded targets that the loss skips, shape (bs, H)
            - key: draws condition dropout, the flow times and the noise; None needs the times and noise fixed and
              drops no condition
            - flow_time: fixed flow times, shape (bs, 1); drawn when None
            - end_time: fixed MeanFlow end times (with flow_time), shape (bs, 1)
            - noise: fixed Gaussian noise, shape (bs, H, d); drawn when None
        Returns:
            - loss: scalar, the objective
            - metrics: the objective's scalar diagnostics without gradient (FlowMatching.metric_names)
        """
        expected_shape = (obs.shape[0], self.pred_horizon, self.action_dim)
        if action_chunk.shape != expected_shape:
            raise ValueError(
                f"Expected action chunks with shape {expected_shape}, got "
                f"{tuple(action_chunk.shape)}."
            )
        dropout_key, flow_key = (None, None) if key is None else jax.random.split(key)
        # One condition per batch: the loss's every pass (a predicted tangent, the JVP) sees the same dropout.
        condition = self.condition(obs, goal, dropout_key=dropout_key)
        return self.flow.loss(
            lambda x, t, s=None: self._predict(condition, x, t, s),
            action_chunk,
            weight,
            key=flow_key,
            flow_time=flow_time,
            end_time=end_time,
            noise=noise,
        )

    def predictor(self, condition: jax.Array, cfg_scale: float = 1.0) -> Predict:
        r"""
        Returns the network for a fixed condition, guided when cfg_scale $w \ne 1$:
        $(1 - w)\,o_t^\theta(x \mid \varnothing) + w\,o_t^\theta(x \mid y)$.
        Note: the guided prediction batches the conditional and the null pass, one network evaluation.
        Args:
            - condition: shape (bs, ...)
            - cfg_scale: guidance weight $w$
        Returns:
            - predict(x (bs, H, d), t (bs, 1)[, s (bs, 1)]) -> prediction (bs, H, d)
        """
        if cfg_scale == 1.0:
            return lambda x, t, s=None: self._predict(condition, x, t, s)
        both = jnp.concatenate((condition, self.null_condition(condition.shape[0])))

        def guided(x: jax.Array, t: jax.Array, s: jax.Array | None = None) -> jax.Array:
            """
            Mixes the conditional and the null prediction of one batched pass.
            """
            s = None if s is None else jnp.concatenate((s, s))
            prediction = self._predict(
                both, jnp.concatenate((x, x)), jnp.concatenate((t, t)), s
            )
            conditional, null = jnp.split(prediction, 2)
            return (1 - cfg_scale) * null + cfg_scale * conditional

        return guided

    def sample_actions(
        self,
        obs: jax.Array,
        goal: jax.Array,
        noise: jax.Array,
        *,
        sampler: Sampler,
        brownian: jax.Array | None = None,
    ) -> jax.Array:
        r"""
        Generates action chunks by integrating the learned flow from noise.
        Args:
            - obs: shape (bs, K, obs_dim)
            - goal: shape (bs, goal_dim)
            - noise: the Gaussian start, drawn by the caller so each episode can own its noise stream,
              shape (bs, H, d)
            - sampler: the solver, NFE, time grid and guidance
            - brownian: $\mathcal N(0, I)$ draws for a stochastic sampler, shape (bs, sampler.num_brownian, H, d)
        Returns:
            - normalized relative chunks, shape (bs, H, d)
        """
        # The condition does not change along the flow, so it is encoded once per query.
        condition = self.encode_condition(obs, goal)
        predict = self.predictor(condition, sampler.cfg_scale)
        flow = LearnedFlow(predict, self.flow.path, self.flow.prediction)
        return sampler.sample(flow, noise, brownian)


class FourierEncoder(nnx.Module):
    r"""
    Fourier features $\sqrt 2\,[\sin(2\pi \omega t), \cos(2\pi \omega t)]$ of the flow time with learned frequencies
    $\omega \sim \mathcal N(0, 1)$.
    """

    def __init__(self, dim: int, *, rngs: nnx.Rngs) -> None:
        """
        Args:
            - dim: embedding width, even
            - rngs: the initialization keys
        """
        self.weights = nnx.Param(jax.random.normal(rngs.params(), (1, dim // 2)))

    def __call__(self, t: jax.Array) -> jax.Array:
        """
        Args:
            - t: flow times, shape (bs, 1)
        Returns:
            - embeddings, shape (bs, dim)
        """
        freqs = t.reshape(-1, 1) * self.weights[...] * 2 * math.pi
        return jnp.concatenate((jnp.sin(freqs), jnp.cos(freqs)), axis=-1) * math.sqrt(2)


class Attention(nnx.Module):
    """
    Multi-head self-attention with a fused qkv projection.
    """

    def __init__(self, dim: int, heads: int, *, rngs: nnx.Rngs) -> None:
        """
        Args:
            - dim: token width D
            - heads: attention heads, dividing D
            - rngs: the initialization keys
        """
        self.heads = heads
        self.qkv = linear(dim, 3 * dim, rngs=rngs)
        self.out = linear(dim, dim, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Args:
            - x: tokens, shape (bs, N, D)
        Returns:
            - attended tokens, shape (bs, N, D)
        """
        # (bs, N, 3D) -> q, k, v of shape (bs, N, heads, D / heads).
        batch, n, dim = x.shape
        qkv = self.qkv(x).reshape(batch, n, 3, self.heads, dim // self.heads)
        x = attention(qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2])
        return self.out(x.reshape(batch, n, dim))


class CrossAttention(nnx.Module):
    """
    Multi-head attention of tokens to context tokens; the output projection starts at zero.
    """

    def __init__(self, dim: int, heads: int, *, rngs: nnx.Rngs) -> None:
        """
        Args:
            - dim: token width D
            - heads: attention heads, dividing D
            - rngs: the initialization keys
        """
        self.heads = heads
        self.q = linear(dim, dim, rngs=rngs)
        self.kv = linear(dim, 2 * dim, rngs=rngs)
        self.out = linear(dim, dim, rngs=rngs, zero=True)

    def __call__(self, x: jax.Array, context: jax.Array) -> jax.Array:
        """
        Args:
            - x: queries, shape (bs, N, D)
            - context: keys and values, shape (bs, M, D)
        Returns:
            - attended tokens, shape (bs, N, D)
        """
        batch, n, dim = x.shape
        head_dim = dim // self.heads
        q = self.q(x).reshape(batch, n, self.heads, head_dim)
        kv = self.kv(context).reshape(batch, context.shape[1], 2, self.heads, head_dim)
        x = attention(q, kv[:, :, 0], kv[:, :, 1])
        return self.out(x.reshape(batch, n, dim))


def mlp(dim: int, *, rngs: nnx.Rngs) -> nnx.Sequential:
    r"""
    Builds the token-wise MLP $D \to 4D \to D$ with the tanh approximation of GELU (jax.nn.gelu's default).
    """
    return nnx.Sequential(
        linear(dim, 4 * dim, rngs=rngs), jax.nn.gelu, linear(4 * dim, dim, rngs=rngs)
    )


class DiTBlock(nnx.Module):
    """
    DiT block with adaLN-Zero: the conditioning vector gives a scale, bias and gate for attention and MLP.
    Note: the modulation is RMSNorm -> Linear with zero initialization, so every block starts as the identity map.
    With cross=True a cross-attention sub-layer to the condition tokens sits between attention and MLP; its output
    projection also starts at zero.
    """

    def __init__(self, dim: int, heads: int, cross: bool, *, rngs: nnx.Rngs) -> None:
        """
        Args:
            - dim: token width D
            - heads: attention heads, dividing D
            - cross: whether the block cross-attends to the condition tokens
            - rngs: the initialization keys
        """
        self.attn = Attention(dim, heads, rngs=rngs)
        self.mlp = mlp(dim, rngs=rngs)
        self.ada_ln = nnx.Sequential(
            rms_norm, linear(dim, 6 * dim, rngs=rngs, zero=True)
        )
        self.cross = cross
        self.cross_attn = CrossAttention(dim, heads, rngs=rngs) if cross else None

    def __call__(
        self, x: jax.Array, c: jax.Array, context: jax.Array | None = None
    ) -> jax.Array:
        """
        Args:
            - x: tokens, shape (bs, N, D)
            - c: conditioning vector, shape (bs, D)
            - context: condition tokens for cross-attention, shape (bs, K + 1, D)
        Returns:
            - tokens, shape (bs, N, D)
        """
        attn_scale, attn_bias, attn_gate, mlp_scale, mlp_bias, mlp_gate = jnp.split(
            self.ada_ln(c)[:, None], 6, axis=-1
        )
        x = x + attn_gate * self.attn(modulate(rms_norm(x), attn_scale, attn_bias))
        if self.cross:
            x = x + self.cross_attn(rms_norm(x), rms_norm(context))
        x = x + mlp_gate * self.mlp(modulate(rms_norm(x), mlp_scale, mlp_bias))
        return x


class DiTPolicy(FlowMatchingPolicy):
    """
    Flow matching policy with a DiT over the chunk's steps; the condition is K + 1 tokens (goal, then history).
    """

    def __init__(
        self,
        obs_dim: int,
        goal_dim: int,
        action_dim: int,
        obs_horizon: int,
        pred_horizon: int,
        *,
        flow: FlowMatching,
        cond_dropout: float,
        conditioning: Conditioning,
        dim: int,
        depth: int,
        heads: int,
        rngs: nnx.Rngs,
    ) -> None:
        """
        Args:
            - obs_dim, goal_dim, action_dim, obs_horizon, pred_horizon, flow, cond_dropout: see FlowMatchingPolicy
            - conditioning: adaln | concat | cross
            - dim: width D
            - depth: number of blocks
            - heads: attention heads
            - rngs: the initialization keys
        """
        super().__init__(
            obs_dim,
            goal_dim,
            action_dim,
            obs_horizon,
            pred_horizon,
            flow=flow,
            cond_dropout=cond_dropout,
        )
        if conditioning not in ("adaln", "concat", "cross"):
            raise ValueError(f"Unknown conditioning {conditioning!r}.")
        self.conditioning = conditioning

        def normal(*shape: int) -> nnx.Param:
            """
            Returns a parameter drawn from N(0, 0.02^2), as positions and null tokens start.
            """
            return nnx.Param(jax.random.normal(rngs.params(), shape) * 0.02)

        self.time_embedder = FourierEncoder(dim, rngs=rngs)
        # One token per chunk step, with a learned position per step.
        self.action_embedder = linear(action_dim, dim, rngs=rngs)
        self.action_pos = normal(pred_horizon, dim)
        # One token for the goal and one per history frame, with a learned position each, then a token-wise MLP.
        self.goal_embedder = linear(goal_dim, dim, rngs=rngs)
        self.obs_embedder = linear(obs_dim, dim, rngs=rngs)
        self.cond_pos = normal(obs_horizon + 1, dim)
        self.cond_mlp = mlp(dim, rngs=rngs)

        self.blocks = nnx.List(
            [
                DiTBlock(dim, heads, conditioning == "cross", rngs=rngs)
                for _ in range(depth)
            ]
        )
        self.final_ada_ln = nnx.Sequential(
            rms_norm, linear(dim, 2 * dim, rngs=rngs, zero=True)
        )
        self.head = linear(dim, action_dim, rngs=rngs, zero=True)
        # Guidance replaces all condition tokens by these learned null tokens.
        self.null_tokens = normal(obs_horizon + 1, dim)
        # MeanFlow's gap embedding, built last so that the other weights draw the same initialization as flow
        # matching's; its zero output layer starts every s at the instantaneous velocity.
        self.gap_embedder = nnx.data(None)
        if self.mean_flow:
            self.gap_embedder = nnx.Sequential(
                FourierEncoder(dim, rngs=rngs), linear(dim, dim, rngs=rngs, zero=True)
            )

    def boundary_layers(self) -> list[nnx.Linear]:
        """
        Returns the embeddings of the actions, the goal and the history frames, and the output head.
        """
        return [self.action_embedder, self.goal_embedder, self.obs_embedder, self.head]

    def encode_condition(self, obs: jax.Array, goal: jax.Array) -> jax.Array:
        """
        Encodes the goal and the history frames as condition tokens.
        Args:
            - obs: shape (bs, K, obs_dim)
            - goal: shape (bs, goal_dim)
        Returns:
            - condition tokens, shape (bs, K + 1, D)
        """
        tokens = jnp.concatenate(
            (self.goal_embedder(goal)[:, None], self.obs_embedder(obs)), axis=1
        )
        # Positions tell the token-wise MLP which token it sees, so pooling keeps every token apart.
        return self.cond_mlp(tokens + self.cond_pos[...])

    def null_condition(self, batch_size: int) -> jax.Array:
        """
        Returns the learned null tokens, shape (bs, K + 1, D).
        """
        tokens = self.null_tokens[...]
        return jnp.broadcast_to(tokens, (batch_size, *tokens.shape))

    def _predict(
        self,
        condition: jax.Array,
        action_chunk: jax.Array,
        flow_time: jax.Array,
        end_time: jax.Array | None = None,
    ) -> jax.Array:
        """
        Predicts with the DiT: time (with MeanFlow's gap) and (for adaln) the pooled condition modulate the blocks.
        Args and Returns: see FlowMatchingPolicy._predict.
        """
        c = self.time_embedder(flow_time)
        gap = self.gap(flow_time, end_time)
        if gap is not None:
            c = c + self.gap_embedder(gap)
        # (bs, H, d) chunk -> (bs, H, D) tokens.
        x = self.action_embedder(action_chunk) + self.action_pos[...]
        context = None
        if self.conditioning == "adaln":
            c = c + condition.mean(axis=1)
        elif self.conditioning == "concat":
            x = jnp.concatenate((condition, x), axis=1)
        else:
            context = condition
        for block in self.blocks:
            x = block(x, c, context)
        # Only the H action tokens are read out (concat also carries K + 1 condition tokens in front).
        x = x[:, -self.pred_horizon :]
        scale, bias = jnp.split(self.final_ada_ln(c)[:, None], 2, axis=-1)
        return self.head(modulate(rms_norm(x), scale, bias))


class MLPPolicy(FlowMatchingPolicy):
    """
    Flow matching policy with a ReLU MLP on [flattened history and goal, flattened chunk, flow time], and MeanFlow's
    gap $h = s - t$.
    Note: with MeanFlow the first layer is one input wider, so its initialization differs from flow matching's.
    """

    def __init__(
        self,
        obs_dim: int,
        goal_dim: int,
        action_dim: int,
        obs_horizon: int,
        pred_horizon: int,
        *,
        flow: FlowMatching,
        cond_dropout: float,
        hidden_dims: Sequence[int],
        rngs: nnx.Rngs,
    ) -> None:
        """
        Args:
            - obs_dim, goal_dim, action_dim, obs_horizon, pred_horizon, flow, cond_dropout: see FlowMatchingPolicy
            - hidden_dims: widths of the hidden layers
            - rngs: the initialization keys
        """
        super().__init__(
            obs_dim,
            goal_dim,
            action_dim,
            obs_horizon,
            pred_horizon,
            flow=flow,
            cond_dropout=cond_dropout,
        )
        self.hidden_dims = tuple(hidden_dims)
        cond_dim = obs_horizon * obs_dim + goal_dim
        layers = []
        in_dim = cond_dim + pred_horizon * action_dim + 1 + self.mean_flow
        for hidden_dim in self.hidden_dims:
            layers += [linear(in_dim, hidden_dim, rngs=rngs), jax.nn.relu]
            in_dim = hidden_dim
        layers.append(linear(in_dim, pred_horizon * action_dim, rngs=rngs))
        self.net = nnx.Sequential(*layers)
        if self.mean_flow:
            # The gap's input weights start at zero, so every s starts at the instantaneous velocity.
            first = self.net.layers[0].kernel
            first[...] = first[...].at[-1].set(0.0)
        self.null = nnx.Param(jnp.zeros(cond_dim))

    def boundary_layers(self) -> list[nnx.Linear]:
        """
        Returns the first and the last layer.
        """
        return [self.net.layers[0], self.net.layers[-1]]

    def encode_condition(self, obs: jax.Array, goal: jax.Array) -> jax.Array:
        """
        Flattens the history and the goal into one vector.
        Args:
            - obs: shape (bs, K, obs_dim)
            - goal: shape (bs, goal_dim)
        Returns:
            - condition, shape (bs, K * obs_dim + goal_dim)
        """
        return jnp.concatenate((obs.reshape(obs.shape[0], -1), goal), axis=1)

    def null_condition(self, batch_size: int) -> jax.Array:
        """
        Returns the learned null vector, shape (bs, K * obs_dim + goal_dim).
        """
        null = self.null[...]
        return jnp.broadcast_to(null, (batch_size, *null.shape))

    def _predict(
        self,
        condition: jax.Array,
        action_chunk: jax.Array,
        flow_time: jax.Array,
        end_time: jax.Array | None = None,
    ) -> jax.Array:
        """
        Predicts with the MLP on [condition, flattened chunk, $t$] and, with MeanFlow, $h = s - t$.
        Args and Returns: see FlowMatchingPolicy._predict.
        """
        # Keep bs intact across the (bs, H * d) <-> (bs, H, d) contract.
        inputs = [condition, action_chunk.reshape(action_chunk.shape[0], -1), flow_time]
        gap = self.gap(flow_time, end_time)
        if gap is not None:
            inputs.append(gap)
        x = jnp.concatenate(inputs, axis=1)
        return self.net(x).reshape(action_chunk.shape)
