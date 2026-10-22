"""
Flow matching action chunking policies for Push-T.

FlowMatchingPolicy holds the contract (compute_loss for training, sample_actions for evaluation) and the flow
matching objective; a subclass supplies the network that maps (condition, noisy chunk, flow time) to a prediction:
    - DiTPolicy: a diffusion transformer with one token per chunk step. The flow time conditions every block through
      adaLN-Zero; the goal and history tokens enter by adaln (mean-pooled into the conditioning vector), concat
      (prepended tokens) or cross (a cross-attention sub-layer per block).
    - MLPPolicy: the flattened observation, chunk and flow time through a ReLU MLP.
With the MeanFlow objective the network predicts the average velocity over $[t, s]$ and also sees the gap
$h = s - t$: the DiT adds a gap embedding to the time embedding, the MLP takes $h$ as one more input. Both start at
zero, so the average velocity starts as the instantaneous one for every $s$.

Shapes: bs is the batch size, K the observation history, H the chunk length, d the action dimension and D the width.
"""

from __future__ import annotations

import abc
import math
from typing import Literal, NamedTuple

import torch
from torch import nn

from flow import FlowMatching, LearnedFlow, MeanFlow, Predict, Sampler

Conditioning = Literal["adaln", "concat", "cross"]


class ParameterGroups(NamedTuple):
    """
    A policy's parameters by role, for the optimizers; together they are parameters(), each in its order:
        - hidden: the weight matrices between hidden layers, Muon's domain;
        - boundary: the weight matrices of the layers that read the inputs or write the prediction;
        - vectors: biases, positions, the null condition and the time frequencies.
    Note: weight decay acts on the matrices (hidden and boundary), never on the vectors.
    """

    hidden: list[nn.Parameter]
    boundary: list[nn.Parameter]
    vectors: list[nn.Parameter]


def at_least_float32(x: torch.Tensor) -> torch.Tensor:
    """
    Returns network outputs in float32, or float64 when the network runs in float64; a float32 tensor itself.
    """
    return x.to(torch.promote_types(x.dtype, torch.float32))


class FlowMatchingPolicy(nn.Module, metaclass=abc.ABCMeta):
    r"""
    Predicts relative action chunks with a flow matching objective; subclasses supply the network.
    At noisy chunks $x$, flow time $t$ and guiding variable $y$ (the goal and the history frames) its output is
    $o_t^\theta(x \mid y)$.
    Note: in training mode, like nn.Dropout, each example sees the learned null condition $\varnothing$ instead of its
    own with probability cond_dropout ($p_{\text{uncond}}$). That trains the unguided (null) prediction
    $o_t^\theta(x \mid \varnothing)$ that classifier-free guidance needs.
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
            - cond_dropout: probability that a training example sees the null condition, in $[0, 1)$
        """
        super().__init__()
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
    def encode_condition(self, obs: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """
        Encodes the observation into the network's condition.
        Args:
            - obs: normalized history frames, shape (bs, K, obs_dim)
            - goal: normalized goal, shape (bs, goal_dim)
        Returns:
            - condition, shape (bs, ...)
        """

    @abc.abstractmethod
    def null_condition(self, batch_size: int) -> torch.Tensor:
        """
        Returns the learned condition of 'no observation'.
        Args:
            - batch_size: bs
        Returns:
            - null condition, shaped like encode_condition's output
        """

    @abc.abstractmethod
    def boundary_layers(self) -> list[nn.Linear]:
        """
        Returns the linear layers that read the inputs or write the prediction; every other linear layer is hidden.
        """

    def parameter_groups(self) -> ParameterGroups:
        """
        Splits the parameters by role (see ParameterGroups).
        """
        boundary = {id(layer.weight) for layer in self.boundary_layers()}
        matrices = {id(m.weight) for m in self.modules() if isinstance(m, nn.Linear)}
        groups = ParameterGroups([], [], [])
        for p in self.parameters():
            if id(p) in boundary:
                groups.boundary.append(p)
            elif id(p) in matrices:
                groups.hidden.append(p)
            else:
                groups.vectors.append(p)
        return groups

    @property
    def mean_flow(self) -> bool:
        """
        Whether the policy is trained with MeanFlow, so that it predicts average velocities.
        """
        return isinstance(self.flow, MeanFlow)

    @abc.abstractmethod
    def _predict(
        self,
        condition: torch.Tensor,
        action_chunk: torch.Tensor,
        flow_time: torch.Tensor,
        end_time: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r"""
        Predicts what the recipe asks for (velocity, clean chunk or scaled score) at the current flow state.
        Note: a MeanFlow network sees the gap $h = s - t$, formed here so that a derivative in $t$ also moves $h$.
        Args:
            - condition: the encoded observation, shape (bs, ...)
            - action_chunk: noisy chunks $x$, shape (bs, H, d)
            - flow_time: $t$, shape (bs, 1)
            - end_time: $s \ge t$ of a MeanFlow average velocity, shape (bs, 1); None is $s = t$
        Returns:
            - prediction in float32 (float64 for a float64 network), shape (bs, H, d)
        """

    def gap(
        self, flow_time: torch.Tensor, end_time: torch.Tensor | None
    ) -> torch.Tensor | None:
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

    def condition(self, obs: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """
        Encodes the observation, with condition dropout in training mode.
        Args:
            - obs: shape (bs, K, obs_dim)
            - goal: shape (bs, goal_dim)
        Returns:
            - condition, shape (bs, ...)
        """
        condition = self.encode_condition(obs, goal)
        if self.training:
            # Drawn at every cond_dropout, 0 included (every example then keeps its condition), so runs that differ
            # only in it share a seed's flow times, noise and batch order. The draw feeds torch.where, so the compiled
            # loss keeps it, and torch.where, not masked assignment, lets the dropout live inside a CUDA graph.
            keep = torch.rand(obs.shape[0], device=obs.device) >= self.cond_dropout
            keep = keep.reshape(-1, *[1] * (condition.dim() - 1))
            condition = torch.where(keep, condition, self.null_condition(obs.shape[0]))
        return condition

    # Training calls compute_loss; evaluation calls sample_actions.
    def compute_loss(
        self,
        obs: torch.Tensor,
        goal: torch.Tensor,
        action_chunk: torch.Tensor,
        weight: torch.Tensor,
        *,
        flow_time: torch.Tensor | None = None,
        end_time: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Computes the training loss of a batch.
        Args:
            - obs: shape (bs, K, obs_dim)
            - goal: shape (bs, goal_dim)
            - action_chunk: normalized expert chunks, shape (bs, H, d)
            - weight: 0 on padded targets that the loss skips, shape (bs, H)
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
        condition = self.condition(obs, goal)
        return self.flow.loss(
            lambda x, t, s=None: self._predict(condition, x, t, s),
            action_chunk,
            weight,
            flow_time=flow_time,
            end_time=end_time,
            noise=noise,
        )

    def predictor(self, condition: torch.Tensor, cfg_scale: float = 1.0) -> Predict:
        r"""
        Returns the network for a fixed condition, with classifier-free guidance when cfg_scale $w \ne 1$:
        $(1 - w)\,o_t^\theta(x \mid \varnothing) + w\,o_t^\theta(x \mid y)$.
        Note: the mix batches the guided pass, on $y$, and the unguided (null) pass, on $\varnothing$, into one network
        evaluation. Under MeanFlow's mean solver the outputs are average velocities, so the mix matches guidance only
        to first order in $s - t$.
        Args:
            - condition: the encoded guiding variable $y$, shape (bs, ...)
            - cfg_scale: guidance weight $w$
        Returns:
            - predict(x (bs, H, d), t (bs, 1)[, s (bs, 1)]) -> prediction (bs, H, d)
        """
        if cfg_scale == 1.0:
            return lambda x, t, s=None: self._predict(condition, x, t, s)
        both = torch.cat((condition, self.null_condition(condition.shape[0])))

        def guided(
            x: torch.Tensor, t: torch.Tensor, s: torch.Tensor | None = None
        ) -> torch.Tensor:
            """
            Mixes the guided and the unguided (null) prediction of one batched pass.
            """
            s = None if s is None else torch.cat((s, s))
            prediction = self._predict(both, torch.cat((x, x)), torch.cat((t, t)), s)
            conditional, null = prediction.chunk(2)
            return (1 - cfg_scale) * null + cfg_scale * conditional

        return guided

    @torch.no_grad()
    def sample_actions(
        self,
        obs: torch.Tensor,
        goal: torch.Tensor,
        noise: torch.Tensor,
        *,
        sampler: Sampler,
        brownian: torch.Tensor | None = None,
    ) -> torch.Tensor:
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


class FourierEncoder(nn.Module):
    r"""
    Fourier features $\sqrt 2\,[\sin(2\pi \omega t), \cos(2\pi \omega t)]$ of the flow time with learned frequencies
    $\omega \sim \mathcal N(0, 1)$.
    """

    def __init__(self, dim: int) -> None:
        """
        Args:
            - dim: embedding width, even
        """
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, dim // 2))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            - t: flow times, shape (bs, 1)
        Returns:
            - embeddings, shape (bs, dim)
        """
        freqs = t.reshape(-1, 1) * self.weights * 2 * math.pi
        return torch.cat((torch.sin(freqs), torch.cos(freqs)), dim=-1) * math.sqrt(2)


def modulate(x: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
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


class Attention(nn.Module):
    """
    Multi-head self-attention with a fused qkv projection and sdpa kernels.
    """

    def __init__(self, dim: int, heads: int) -> None:
        """
        Args:
            - dim: token width D
            - heads: attention heads, dividing D
        """
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            - x: tokens, shape (bs, N, D)
        Returns:
            - attended tokens, shape (bs, N, D)
        """
        # Heads fold into (bs, h, N, D/h).
        batch, n, dim = x.shape
        q, k, v = (
            self.qkv(x)
            .reshape(batch, n, 3, self.heads, dim // self.heads)
            .permute(2, 0, 3, 1, 4)
        )
        x = nn.functional.scaled_dot_product_attention(q, k, v)
        return self.out(x.transpose(1, 2).reshape(batch, n, dim))


class CrossAttention(nn.Module):
    """
    Multi-head attention of tokens to context tokens.
    """

    def __init__(self, dim: int, heads: int) -> None:
        """
        Args:
            - dim: token width D
            - heads: attention heads, dividing D
        """
        super().__init__()
        self.heads = heads
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, 2 * dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            - x: queries, shape (bs, N, D)
            - context: keys and values, shape (bs, M, D)
        Returns:
            - attended tokens, shape (bs, N, D)
        """
        batch, n, dim = x.shape
        head_dim = dim // self.heads
        q = self.q(x).reshape(batch, n, self.heads, head_dim).transpose(1, 2)
        k, v = (
            self.kv(context)
            .reshape(batch, context.shape[1], 2, self.heads, head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        x = nn.functional.scaled_dot_product_attention(q, k, v)
        return self.out(x.transpose(1, 2).reshape(batch, n, dim))


class DiTBlock(nn.Module):
    """
    DiT block with adaLN-Zero: the conditioning vector gives a scale, bias and gate for attention and MLP.
    Note: the modulation is RMSNorm -> Linear with zero initialization, so every block starts as the identity map.
    With cross=True a cross-attention sub-layer to the condition tokens sits between attention and MLP; its output
    projection also starts at zero.
    """

    def __init__(self, dim: int, heads: int, cross: bool) -> None:
        """
        Args:
            - dim: token width D
            - heads: attention heads, dividing D
            - cross: whether the block cross-attends to the condition tokens
        """
        super().__init__()
        self.norm1 = nn.RMSNorm(dim, elementwise_affine=False)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.RMSNorm(dim, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * dim, dim),
        )
        self.ada_ln = nn.Sequential(
            nn.RMSNorm(dim, elementwise_affine=False), nn.Linear(dim, 6 * dim)
        )
        nn.init.zeros_(self.ada_ln[1].weight)
        nn.init.zeros_(self.ada_ln[1].bias)
        self.cross = cross
        if cross:
            self.norm_query = nn.RMSNorm(dim, elementwise_affine=False)
            self.norm_context = nn.RMSNorm(dim, elementwise_affine=False)
            self.cross_attn = CrossAttention(dim, heads)
            nn.init.zeros_(self.cross_attn.out.weight)
            nn.init.zeros_(self.cross_attn.out.bias)

    def forward(
        self, x: torch.Tensor, c: torch.Tensor, context: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            - x: tokens, shape (bs, N, D)
            - c: conditioning vector, shape (bs, D)
            - context: condition tokens for cross-attention, shape (bs, K + 1, D)
        Returns:
            - tokens, shape (bs, N, D)
        """
        attn_scale, attn_bias, attn_gate, mlp_scale, mlp_bias, mlp_gate = (
            self.ada_ln(c).unsqueeze(1).chunk(6, dim=-1)
        )
        x = x + attn_gate * self.attn(modulate(self.norm1(x), attn_scale, attn_bias))
        if self.cross:
            x = x + self.cross_attn(self.norm_query(x), self.norm_context(context))
        x = x + mlp_gate * self.mlp(modulate(self.norm2(x), mlp_scale, mlp_bias))
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
    ) -> None:
        """
        Args:
            - obs_dim, goal_dim, action_dim, obs_horizon, pred_horizon, flow, cond_dropout: see FlowMatchingPolicy
            - conditioning: adaln | concat | cross
            - dim: width D
            - depth: number of blocks
            - heads: attention heads
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

        self.time_embedder = FourierEncoder(dim)
        # One token per chunk step, with a learned position per step.
        self.action_embedder = nn.Linear(action_dim, dim)
        self.action_pos = nn.Parameter(torch.randn(pred_horizon, dim) * 0.02)
        # One token for the goal and one per history frame, with a learned position each, then a token-wise MLP.
        self.goal_embedder = nn.Linear(goal_dim, dim)
        self.obs_embedder = nn.Linear(obs_dim, dim)
        self.cond_pos = nn.Parameter(torch.randn(obs_horizon + 1, dim) * 0.02)
        self.cond_mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * dim, dim),
        )

        self.blocks = nn.ModuleList(
            [DiTBlock(dim, heads, cross=conditioning == "cross") for _ in range(depth)]
        )
        self.final_norm = nn.RMSNorm(dim, elementwise_affine=False)
        self.final_ada_ln = nn.Sequential(
            nn.RMSNorm(dim, elementwise_affine=False), nn.Linear(dim, 2 * dim)
        )
        self.head = nn.Linear(dim, action_dim)
        for layer in (self.final_ada_ln[1], self.head):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        # The learned null label: condition dropout and the unguided (null) pass of classifier-free guidance replace
        # all condition tokens by these tokens.
        self.null_tokens = nn.Parameter(torch.randn(obs_horizon + 1, dim) * 0.02)
        # MeanFlow's gap embedding, built last so that the other weights draw the same initialization as flow
        # matching's; its zero output layer starts every s at the instantaneous velocity.
        self.gap_embedder = None
        if self.mean_flow:
            self.gap_embedder = nn.Sequential(FourierEncoder(dim), nn.Linear(dim, dim))
            nn.init.zeros_(self.gap_embedder[1].weight)
            nn.init.zeros_(self.gap_embedder[1].bias)

    def boundary_layers(self) -> list[nn.Linear]:
        """
        Returns the embeddings of the actions, the goal and the history frames, and the output head.
        """
        return [self.action_embedder, self.goal_embedder, self.obs_embedder, self.head]

    def encode_condition(self, obs: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """
        Encodes the goal and the history frames as condition tokens.
        Args:
            - obs: shape (bs, K, obs_dim)
            - goal: shape (bs, goal_dim)
        Returns:
            - condition tokens, shape (bs, K + 1, D)
        """
        tokens = torch.cat(
            (self.goal_embedder(goal).unsqueeze(1), self.obs_embedder(obs)), dim=1
        )
        # Positions tell the token-wise MLP which token it sees, so pooling keeps every token apart.
        return self.cond_mlp(tokens + self.cond_pos)

    def null_condition(self, batch_size: int) -> torch.Tensor:
        """
        Returns the learned null tokens, shape (bs, K + 1, D).
        """
        return self.null_tokens.expand(batch_size, -1, -1)

    def _predict(
        self,
        condition: torch.Tensor,
        action_chunk: torch.Tensor,
        flow_time: torch.Tensor,
        end_time: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Predicts with the DiT: time (with MeanFlow's gap) and (for adaln) the pooled condition modulate the blocks.
        Args and Returns: see FlowMatchingPolicy._predict.
        """
        c = self.time_embedder(flow_time)
        gap = self.gap(flow_time, end_time)
        if gap is not None:
            c = c + self.gap_embedder(gap)
        # (bs, H, d) chunk -> (bs, H, D) tokens.
        x = self.action_embedder(action_chunk) + self.action_pos
        context = None
        if self.conditioning == "adaln":
            c = c + condition.mean(dim=1)
        elif self.conditioning == "concat":
            x = torch.cat((condition, x), dim=1)
        else:
            context = condition
        for block in self.blocks:
            x = block(x, c, context)
        # Only the H action tokens are read out (concat also carries K + 1 condition tokens in front).
        x = x[:, -self.pred_horizon :]
        scale, bias = self.final_ada_ln(c).unsqueeze(1).chunk(2, dim=-1)
        return at_least_float32(self.head(modulate(self.final_norm(x), scale, bias)))


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
        hidden_dims: list[int],
    ) -> None:
        """
        Args:
            - obs_dim, goal_dim, action_dim, obs_horizon, pred_horizon, flow, cond_dropout: see FlowMatchingPolicy
            - hidden_dims: widths of the hidden layers
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
        cond_dim = obs_horizon * obs_dim + goal_dim
        layers: list[nn.Module] = []
        in_dim = cond_dim + pred_horizon * action_dim + 1 + self.mean_flow
        for hidden_dim in hidden_dims:
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, pred_horizon * action_dim))
        self.net = nn.Sequential(*layers)
        if self.mean_flow:
            # The gap's input weights start at zero, so every s starts at the instantaneous velocity.
            nn.init.zeros_(self.net[0].weight[:, -1])
        self.null = nn.Parameter(torch.zeros(cond_dim))

    def boundary_layers(self) -> list[nn.Linear]:
        """
        Returns the first and the last layer.
        """
        return [self.net[0], self.net[-1]]

    def encode_condition(self, obs: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """
        Flattens the history and the goal into one vector.
        Args:
            - obs: shape (bs, K, obs_dim)
            - goal: shape (bs, goal_dim)
        Returns:
            - condition, shape (bs, K * obs_dim + goal_dim)
        """
        return torch.cat((obs.flatten(1), goal), dim=1)

    def null_condition(self, batch_size: int) -> torch.Tensor:
        """
        Returns the learned null vector, shape (bs, K * obs_dim + goal_dim).
        """
        return self.null.expand(batch_size, -1)

    def _predict(
        self,
        condition: torch.Tensor,
        action_chunk: torch.Tensor,
        flow_time: torch.Tensor,
        end_time: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Predicts with the MLP on [condition, flattened chunk, $t$] and, with MeanFlow, $h = s - t$.
        Args and Returns: see FlowMatchingPolicy._predict.
        """
        # Keep bs intact across the (bs, H * d) <-> (bs, H, d) contract.
        inputs = [condition, action_chunk.flatten(1), flow_time]
        gap = self.gap(flow_time, end_time)
        if gap is not None:
            inputs.append(gap)
        x = torch.cat(inputs, dim=1)
        return at_least_float32(self.net(x).reshape(action_chunk.shape))
