# Model

The policy's network has three inputs: a noisy chunk $x$, the flow time $t$ and the **guiding variable** $y$ (the goal
and the history frames). With parameters $\theta$, it returns one output shaped like the chunk, $o_t^\theta(x \mid y)$.
Following Holderrieth and Erives' notes, we say **guided** for conditioning on $y$ and *conditional* for conditioning on
a clean chunk $z$ ([glossary](notation.md#glossary)). In code names, `condition` means $y$, except in MeanFlow's
`flow.tangent=conditional`.

The policy is trained to sample the guided data distribution $p_{\mathrm{data}}(\cdot \mid y)$. The exact guided vector
field, simulated exactly from noise and without classifier-free guidance (below), would sample it. Each observation
occurs essentially once in the demonstrations, so the network learns that distribution only by generalizing across
nearby observations.

The objective decides what the output is. Under flow matching, its recipe makes $o_t^\theta$ a vector field, a
denoiser or a scaled score. The objective converts any of them into the guided vector field $u_t^\theta(x \mid y)$, at
every $t$ but the output's singular end: $t = 1$ for a denoiser and $t = 0$ for a scaled score
([four parameterizations](flow/index.md#four-parameterizations)). That vector field is the policy, and the samplers
simulate its ODE or SDE ({doc}`flow/sampling`). Under MeanFlow the output is the average velocity
$\bar u^\theta_{t,s}(x \mid y)$ over a jump from $t$ to a later time $s$. The `mean` solver jumps with it directly. Its
value at $s = t$, $\bar u^\theta_{t,t}(x \mid y)$, is the vector field that the other solvers simulate
({doc}`flow/meanflow`). {py:class}`~model.FlowMatchingPolicy` holds the contract, and {py:class}`~model.DiTPolicy`
(the default) and {py:class}`~model.MLPPolicy` supply the network.

```bash
uv run train.py model.conditioning=concat   # the DiT with the goal and history tokens prepended
uv run train.py model=mlp                   # the MLP baseline
```

{py:func}`evaluation.build_policy` instantiates the configured network with the data-dependent sizes and the run's
objective, the same way training and checkpoint loading do:

```python
import torch
from hydra import compose, initialize
from hydra.utils import instantiate

from evaluation import build_policy

# A script or REPL in the repository root: config_path is relative to the calling file.
with initialize(version_base="1.3", config_path="conf"):
    config = compose("config", overrides=["model.conditioning=concat"])
model_kwargs = dict(obs_dim=6, goal_dim=4, action_dim=2, obs_horizon=2, pred_horizon=16)
policy = build_policy(config, model_kwargs)

obs, goal, chunk = torch.randn(8, 2, 6), torch.randn(8, 4), torch.randn(8, 16, 2)
loss, metrics = policy.compute_loss(obs, goal, chunk, torch.ones(8, 16))
policy.eval()
actions = policy.sample_actions(
    obs, goal, torch.randn(8, 16, 2), sampler=instantiate(config.sampler)
)
```

## The policy contract

{py:class}`~model.FlowMatchingPolicy` is an `nn.Module` that holds the objective as `self.flow` (a
{py:class}`~flow.FlowMatching` or a {py:class}`~flow.MeanFlow`) and two entry points: training calls
{py:meth}`~model.FlowMatchingPolicy.compute_loss`, evaluation calls {py:meth}`~model.FlowMatchingPolicy.sample_actions`.

**Training.** {py:func}`~train.build_loss` returns `compute_loss`, compiled or not, as the loss of the captured
training step ({doc}`training`). `compute_loss` checks the chunk's shape and encodes $y$ with
{py:meth}`~model.FlowMatchingPolicy.condition`, which applies condition dropout (below) in training mode. It then calls
{py:meth}`flow.loss <flow.FlowMatching.loss>` with the clean chunk $z$, the loss weight and a closure
`predict(x, t[, s])` over the encoding of $y$. The objective draws the flow times and the noise, builds the noisy chunk,
calls the closure and returns the loss with its metrics. {doc}`flow/training` writes these steps as an algorithm
([one training step](flow/training.md#one-training-step)).

**Evaluation.** {py:func}`~evaluation.query_policy` normalizes the observation and the goal into $y$ and calls
`sample_actions` with a start noise drawn by the episode's own generator. `sample_actions` encodes $y$ once per query,
since $y$ does not change along the trajectory. It takes the network at that $y$ from
{py:meth}`~model.FlowMatchingPolicy.predictor` and wraps it in a {py:class}`~flow.LearnedFlow`. Its
{py:meth}`~flow.LearnedFlow.drift` converts the output to $u_t^\theta(x \mid y)$ with the objective's path and
{py:class}`~flow.Prediction`. Its {py:meth}`~flow.LearnedFlow.average` instead returns a MeanFlow network's raw average
velocity, unconverted, to the `mean` solver. {py:meth}`Sampler.sample <flow.Sampler.sample>` then runs the solver from
the start noise to $t = 1$ and returns normalized relative chunks, which the caller maps back to world targets
({doc}`data`). {doc}`flow/sampling` follows the chain inside the sampler
([implementation](flow/sampling.md#implementation)). Classifier-free guidance changes only what `predictor` returns
(below).

**What a subclass supplies.** Four methods:

- {py:meth}`~model.FlowMatchingPolicy.encode_condition`: the network's encoding of the observation $y$;
- {py:meth}`~model.FlowMatchingPolicy.null_condition`: the learned encoding of the null label $\varnothing$ of
  classifier-free guidance (below), shaped like the encoding of $y$;
- {py:meth}`~model.FlowMatchingPolicy.boundary_layers`: the linear layers that read the inputs or write the
  prediction;
- {py:meth}`~model.FlowMatchingPolicy._predict`: the network, from the encoded $y$, $x$, $t$ (and MeanFlow's end time
  $s$) to the output $o_t^\theta(x \mid y)$.

{doc}`extending` walks through adding one. `_predict` returns its output through
{py:func}`~model.at_least_float32`, so the objective and the samplers receive float32 even when the network runs
under bf16 autocast.

**The objective fixes the output.** The network does not know what it predicts. The objective's
{py:class}`~flow.Recipe` (`flow.recipe`) picks the {py:class}`~flow.Prediction`, and whether the loss compares the
output in its own space or after converting it to a vector field. For the ODE and SDE solvers, the Prediction converts
the output to $u_t^\theta(x \mid y)$. With a `MeanFlow` objective the same network predicts the average velocity
$\bar u^\theta_{t,s}(x \mid y)$ ({py:attr}`~model.FlowMatchingPolicy.mean_flow`). Changing the flow matching recipe
never changes the network; MeanFlow only adds the gap input (below).

## The DiT

{py:class}`~model.DiTPolicy` is the notes' diffusion transformer, laid out over the steps of the chunk: where an image
DiT has one token per patch, it has one token per chunk step. In this section $d$ is the width of the transformer, $L$
its number of blocks and $n_{\text{heads}}$ its number of attention heads, as in the notes. The network first embeds
its three inputs to width $d$: the flow time as a time embedding $\tilde t$, the goal and the $K$ history frames as
$K + 1$ tokens $\tilde y$, and the $H$ steps of the noisy chunk as $H$ action tokens $\tilde x_0$.

$$
\begin{aligned}
\tilde t &= \mathrm{FourierEmb}(t) \in \mathbb R^{d}
&& \blacktriangleright\ \text{time embedding} \\
\tilde y &= \mathrm{ObsEmb}(y) \in \mathbb R^{(K+1) \times d}
&& \blacktriangleright\ \text{goal and history tokens} \\
\tilde x_0 &= \mathrm{ActionEmb}(x) \in \mathbb R^{H \times d}
&& \blacktriangleright\ \text{action tokens}
\end{aligned}
$$

-  **Time.** {py:class}`~model.FourierEncoder` embeds the flow time with frequencies $\omega \in \mathbb R^{d/2}$, drawn
  from $\mathcal N(0, 1)$ at initialization and then trained, as
  $\tilde t = \sqrt 2\,[\sin(2\pi \omega t), \cos(2\pi \omega t)]$. Unlike the notes' $\mathrm{TimeEmb}$, which fixes
  its frequencies on a geometric grid and scales by $\sqrt{2/d}$ to norm 1, it learns its frequencies and scales by
  $\sqrt 2$ to root mean square 1 over the coordinates.
- **Goal and history.** {py:meth}`~model.DiTPolicy.encode_condition` makes these tokens once per batch or query,
  before `_predict`. $\tilde y^{(0)}$ is the goal and $\tilde y^{(1)}, \dots, \tilde y^{(K)}$ are the history frames.
  Each token is a linear embedding per kind plus a learned position per token, and a token-wise MLP then runs over all
  $K + 1$ tokens. They are the notes' embedded guiding variable.
- **Actions.** Each of the $H$ noisy actions of $x$ becomes one token: a `Linear(action_dim, dim)` shared by all
  steps, plus a learned position per chunk step. This is the notes' patch embedding, with a patch of one step.

Every block reads the flow time through one **conditioning vector** $c \in \mathbb R^d$. In the default mode it also
carries the observation:

$$
c = \tilde t + \underbrace{\frac{1}{K+1}\sum_{j=0}^{K} \tilde y^{(j)}}_{\texttt{adaln}\ \text{only}} .
$$

The conditioning modes below compare this with the two other ways of feeding $\tilde y$, and MeanFlow adds a third
term to $c$ (MeanFlow's gap, below).

Each {py:class}`~model.DiTBlock` reads $c$ through adaLN-Zero, the notes' AdaNorm with gates. Its modulation layer is a
non-affine RMSNorm followed by a linear layer. It turns $c$ into a scale $\gamma_k$, a shift $\beta_k$ and a gate $g_k$
for each sub-layer $k$: $k = \text{self}$ for the self-attention and $k = \text{MLP}$ for the MLP. Block $i$ starts from
$\tilde x = \tilde x_i$, applies the updates below in order and returns $\tilde x_{i+1} = \tilde x$:

$$
(\gamma_\text{self}, \beta_\text{self}, g_\text{self}, \gamma_\text{MLP}, \beta_\text{MLP}, g_\text{MLP})
&= \mathrm{Linear}\big(\mathrm{RMSNorm}(c)\big), \\
\mathrm{AdaNorm}_k(\tilde x) &= (1 + \gamma_k) \odot \mathrm{RMSNorm}(\tilde x) + \beta_k
\qquad (k = \text{self}, \text{MLP}), \\
\tilde x &\leftarrow \tilde x + g_\text{self} \odot
\mathrm{MultiHeadAttention}\big(\mathrm{AdaNorm}_\text{self}(\tilde x),\,\mathrm{AdaNorm}_\text{self}(\tilde x)\big), \\
\tilde x &\leftarrow \tilde x + g_\text{MLP} \odot \mathrm{MLP}\big(\mathrm{AdaNorm}_\text{MLP}(\tilde x)\big).
$$

As in the notes, $\mathrm{MultiHeadAttention}$ splits the width into $n_{\text{heads}}$ heads of width
$d/n_{\text{heads}}$, and the MLP acts on each token separately. After the $L$ blocks, a final AdaNorm with its own
scale and shift and a linear head map each action token back to an action, the notes' depatchification:

$$
(\gamma_\text{out}, \beta_\text{out}) = \mathrm{Linear}\big(\mathrm{RMSNorm}(c)\big), \qquad
o_t^\theta(x \mid y) = \mathrm{Linear}\big((1 + \gamma_\text{out}) \odot \mathrm{RMSNorm}(\tilde x_L)
+ \beta_\text{out}\big),
$$

on the $H$ action tokens of $\tilde x_L$.

The *Zero* of adaLN-Zero is the initialization: every modulation layer, the final one included, and the head start
with zero weights and biases. At initialization $\gamma_k = \beta_k = g_k = 0$, so each AdaNorm is a plain RMSNorm,
each gated branch adds nothing and every block is the identity. With the zero head, the whole network starts at
output 0. Intuitively, each block starts as a no-op, and its residual branches join in only as fast as their gates
open (in `cross`, the ungated cross-attention joins as its zero output projection grows).

**Against the notes' DiT block.** The structure is the same: pre-norm residual sub-layers, each read through an
AdaNorm and scaled by a gate, then a linear readout per token. Four details differ. The norm is an RMSNorm without
learned affine parameters. The scales, shifts and gates of a block come from one linear layer on
$\mathrm{RMSNorm}(c)$, not from an MLP of $\tilde t$. The cross-attention sub-layer of the notes' generic block exists
only in the `cross` mode. A final AdaNorm precedes the readout.

**Conditioning modes** (`model.conditioning`). How does the observation reach the action tokens? The flow time enters
every block through $c$ in all three modes; they differ only in how the tokens $\tilde y$ enter. The notes' rule of
thumb is to add a single embedding vector into every block and to read a sequence of embeddings by attention. `adaln`
pools the $K + 1$ tokens into one vector, while `concat` and `cross` keep the sequence:

| mode | how $\tilde y$ enters | extra tokens and parameters | what $y$ can express | cost per network evaluation, beyond `adaln` |
| --- | --- | --- | --- | --- |
| `adaln` (default) | mean-pooled into $c$ | none | one scale, shift and gate per channel and block, the same for all $H$ action tokens | none |
| `concat` | as $K + 1$ in-context tokens in front of the action tokens | $K + 1$ tokens in every block; no parameters | attention between every action token and every observation token, both ways; the observation tokens are refined block by block | $K + 1$ more tokens through every attention and MLP |
| `cross` | as keys and values of a cross-attention sub-layer between self-attention and MLP | no tokens; $4d^2 + 4d$ parameters per block, at $d = 256$ 22% more per block and 20% more for the default 6-block network | attention from every action token to the same $\tilde y$ in every block | one more attention sub-layer per block, over $K + 1$ keys |

Peebles and Xie's DiT paper names the three designs adaLN-Zero, in-context conditioning and cross-attention. Three
details complete the table.

- `adaln` is the notes' class-conditioned DiT, with the pooled observation in place of a class embedding. Pooling need
  not mix the tokens: the per-token positions and the token-wise MLP can route each token into its own subspace of
  the $d$ channels. This is a design rationale, not tested. Pooling has no weights, while a linear layer over the
  flattened tokens would grow with K.
- In `concat`, the blocks act on all K + 1 + H tokens, and only the H action tokens are read out. The observation
  tokens pass through attention, MLP and the AdaNorms driven by $c$, like the action tokens.
- `cross` sits where the notes' block has its cross-attention, with the action tokens as queries. Unlike the notes'
  sub-layer, which is gated by $g_\text{cross}$ and reads $\mathrm{AdaNorm}(\tilde x)$, it has no gate and uses a
  non-affine RMSNorm on both inputs. Its output projection starts at zero, so the block still starts as the identity.

**Size.** `model.dim`, `model.depth` and `model.heads` set the width $d = 256$, the depth $L = 6$ blocks and the
number of attention heads $n_{\text{heads}} = 4$. The transformer of Diffusion Policy uses width 256 and 8 layers. Only
the position tables and the null tokens grow with K and H. Width, depth and heads were not studied.

**Theory against practice.** None of these choices changes what training aims at. For an expressive enough network,
every mode of the DiT and the MLP below have the same minimizer: the posterior mean of the regression target given
$(x, t, y)$, at every flow time that training draws ([regression learns a posterior
mean](flow/training.md#regression-learns-a-posterior-mean)). In practice a network of fixed size fits some ways of
reading $y$ more easily than others, and the modes differ in cost per query. The findings measure both.

**What we found**

- **DiT >> MLP.** At the same K, H and E the DiT is far ahead, and it fits the long chunks on which the MLP falls
  apart (see below).
- **adaln > concat > cross, raw weights (the second gap weak).** adaln also converges fastest.
- **adaln = concat, averaged (EMA) weights.** The adaln lead holds for the raw weights only, while the policy that
  training evaluates and `evaluation.py` scores by default is the average ({doc}`training`).
- **concat = cross > adaln on mean max reward, raw weights.** concat and cross have a thinner tail of complete
  failures, while adaln fails on a few specific initial states for a reason not known (mean pooling of the tokens
  $\tilde y$ is the obvious suspect, not tested).
- **query cost: adaln < concat < cross.** adaln is the cheapest per query and cross the most expensive
  ({doc}`evaluation`).

These ran with Adam, the conditional flow matching loss $\mathcal L_{\text{CFM}}$ (`velocity_mse`) and `hold`
padding, at K = 2, H = 16 and E = 8. adaln stays the default for its raw-weight lead and its cost; concat is a
reasonable alternative.

Reproduce: `uv run train.py -m experiment=conditioning` (it pins the study's settings: Adam, `velocity_mse`, `hold`
padding, E = 8, and the raw weights as the headline with both averages scored alongside)

## The MLP

{py:class}`~model.MLPPolicy` is the baseline the DiT replaced: the notes' MLP on the concatenated inputs, which the
notes call sufficient for low-dimensional data. {py:meth}`~model.MLPPolicy.encode_condition` flattens the K frames and
the goal into one vector. {py:meth}`~model.MLPPolicy._predict` concatenates that vector with the flattened noisy chunk
$x$ and the raw flow time $t$, plus the gap $h$ under MeanFlow (below). A ReLU MLP (`model.hidden_dims`) then maps the
result to the whole chunk at once. Its encoding of $\varnothing$ is one learned vector, and its boundary layers are the
first and the last. Its input grows with K and H; the hidden widths were not studied.

**The DiT against the MLP.** The axis is how a network uses the structure of the chunk. The contract is the same: both
read $(x, t, y)$, return an output shaped like the chunk and learn an encoding of $\varnothing$. Three things differ.
The MLP sees the chunk as one vector of $2H$ numbers, while the DiT gives each step a token and shares its weights
across steps. The MLP reads $t$ once, at its input, while every DiT block reads it through $c$. The MLP's first and
last layers grow with $H$, while in the DiT only the position table of the action tokens does. In theory, a wide
enough MLP has the same minimizer as the DiT (above). In practice, a chunk has $2H$ coordinates, so the MLP leaves the
low-dimensional regime as $H$ grows, which fits the finding below.

**What we found**

- **H = 8 > H = 16 >> H = 64.** The MLP degrades quickly as chunks get longer and fails at H = 64 with every padding,
  the reason for the DiT.

These ran with Adam and $\mathcal L_{\text{CFM}}$, in the agent frame with `hold` padding, at E = 8.

Try: `uv run train.py model=mlp`

## Parameter groups

{py:meth}`~model.FlowMatchingPolicy.parameter_groups` splits the parameters $\theta$ by role into a
{py:class}`~model.ParameterGroups`, for the optimizers:

- {py:attr}`~model.ParameterGroups.hidden`: the weight matrices of the linear layers between hidden layers, Muon's
  domain. In the DiT these are the self- and cross-attention, MLP and adaLN modulation layers (the final one
  included), the token-wise MLP over the goal and history tokens (`cond_mlp`) and, with MeanFlow, the gap
  embedding's linear layer.
- {py:attr}`~model.ParameterGroups.boundary`: the weight matrices of the layers that read the inputs or write the
  prediction, as {py:meth}`~model.FlowMatchingPolicy.boundary_layers` names them: in the DiT the action, goal and
  history embeddings and the head; in the MLP its first and last layer.
- {py:attr}`~model.ParameterGroups.vectors`: everything that is not the weight of a linear layer: biases, positions,
  the learned encoding of $\varnothing$, the time frequencies $\omega$.

Every `nn.Linear` weight that `boundary_layers` does not name is hidden, so a new linear layer is hidden unless it
is named there, and the three groups together are exactly the policy's parameters (a test checks the partition).
{py:func}`~train.build_optimizers` moves the hidden matrices with Muon when `train.optimizer=muon`, and the boundary
matrices and the vectors with AdamW; weight decay acts on the matrices only, never on the vectors. {doc}`training`
covers the optimizers and how they compared.

## Classifier-free guidance

Classifier-free guidance mixes the network's guided output with its unguided one, the output for the null label
$\varnothing$, "no observation". {doc}`flow/sampling` defines the resulting classifier-free guided vector field
$\tilde u_t^\theta(x \mid y)$ with its guidance scale $w$, and explains how it departs from the policy's target
([guidance](flow/sampling.md#guidance)). Here we describe how one network supplies both outputs.

**The null encoding.** Each network learns an encoding of $\varnothing$ in the place of its encoding of $y$, which
{py:meth}`~model.FlowMatchingPolicy.null_condition` returns. In the DiT these are K + 1 learned null tokens, one per
token position, which replace the output of the token-wise MLP. In the MLP it is one learned vector in the space of
the flattened observation.

**Condition dropout.** In training mode, {py:meth}`~model.FlowMatchingPolicy.condition` replaces the encoded $y$ of each
example by the null encoding, independently with probability $p_{\mathrm{uncond}}$ = `model.cond_dropout`. This is the
step $y \leftarrow \varnothing$ of the training algorithm ([one training step](flow/training.md#one-training-step)), and
it trains the unguided output. The swap is a `torch.where` on a random mask rather than a masked assignment, so it lives
inside the captured CUDA graph. Like `nn.Dropout` it acts only in training mode.

**The mix.** At sampling, {py:meth}`~model.FlowMatchingPolicy.predictor` mixes the outputs for the encoded $y$ and
for $\varnothing$ with $w$ = {py:attr}`~flow.Sampler.cfg_scale`:

$$
(1 - w)\, o_t^\theta(x \mid \varnothing) + w\, o_t^\theta(x \mid y).
$$

At $w = 1$ it returns the network for $y$ unmixed. Otherwise it stacks $y$ with $\varnothing$, and $x$, $t$ and
MeanFlow's $s$ with themselves, then evaluates both halves in one batched pass, which counts as one network evaluation.
For the ODE and SDE solvers, the objective's {py:class}`~flow.Prediction` converts the mixed output into
$\tilde u_t^\theta(x \mid y)$. Under MeanFlow's `mean` solver the mixed average velocities are used unconverted.
{doc}`flow/sampling` shows why mixing outputs mixes vector fields, and why under the `mean` solver it agrees with
guidance only to first order in $s - t$ ([guidance](flow/sampling.md#guidance)). {py:func}`~train.check_config` refuses
$w \ne 1$ unless `model.cond_dropout` is above 0, since the null encoding would be untrained.

**What we found**

- **guidance (w from 1.1 to 1.5) = no guidance.** Once the scale is chosen on separate episodes, it moves success by
  less than the noise.
- **condition dropout = no dropout, at w = 1.** Training with dropout changes nothing measurable in
  $u_t^\theta(x \mid y)$.

Guidance also doubles the network batch of every query. The default is therefore no dropout and $w = 1$; both stay in
the code for experiments. The study ran with Adam, $\mathcal L_{\text{CFM}}$, `hold` padding and E = 8.

Reproduce: `uv run train.py -m experiment=conditioning`, then score its dropout runs at several scales without
retraining, at the raw weights as in the study (`evaluation.py` scores the averaged weights by default):

```bash
uv run evaluation.py -m runs=runs/conditioning split=selection episodes=200 weights=raw \
    'sampler.cfg_scale=1.1,1.25,1.5'
```

## MeanFlow's gap

With `flow=meanflow` the objective is a {py:class}`~flow.MeanFlow`, {py:attr}`~model.FlowMatchingPolicy.mean_flow` is
true, and the network's output is the average velocity $\bar u^\theta_{t,s}(x \mid y)$ of {doc}`flow/meanflow`, over a
jump from $t$ to an end time $s \ge t$. The network then takes one more input, the **gap** $h = s - t$, the step size of
that jump. {py:meth}`~model.FlowMatchingPolicy.gap` forms it inside {py:meth}`~model.FlowMatchingPolicy._predict`
and returns $h = 0$ when no end time is given. A flow matching policy refuses an end time.

- In the DiT, a gap embedding joins the conditioning vector. It is a linear layer on $\mathrm{FourierEmb}_h(h)$,
  which has the form of the time embedding but its own learned frequencies $\omega_h$:

  $$
  c = \tilde t + \mathrm{Linear}\big(\mathrm{FourierEmb}_h(h)\big)
  + \underbrace{\frac{1}{K+1}\sum_{j=0}^{K} \tilde y^{(j)}}_{\texttt{adaln}\ \text{only}} .
  $$

  It is built last, so every other weight draws the same initialization as a flow matching DiT of the same seed.
- In the MLP, $h$ is one more input. The first layer is then one input wider, so its initialization differs from flow
  matching's.

The gap's path into the network starts at zero: the gap embedding's linear layer in the DiT, the first layer's weights
on $h$ in the MLP. At initialization the network therefore ignores $h$, and
$\bar u^\theta_{t,s}(x \mid y) = \bar u^\theta_{t,t}(x \mid y)$ for every $s$. In other words, a MeanFlow network starts
as a flow matching network does, a function of $(x, t, y)$ alone.

The gap is formed inside `_predict` because MeanFlow's target differentiates $\bar u^\theta_{t,s}$ in $t$ at fixed $s$,
and that derivative must move $h = s - t$ along with $t$ ([computing the
derivative](flow/meanflow.md#computing-the-derivative)). A test compares the derivative with finite differences for
every network.
