# Data

The data layer turns the Diffusion Policy demonstrations into training windows. The policy is a guided generative
model: given what it observes, $y$, it is trained to sample an action chunk from $p_{\mathrm{data}}(\cdot \mid y)$.
Every training objective of {doc}`flow/training` therefore draws the pair $(z, y) \sim p_{\mathrm{data}}(z, y)$
jointly, and one window is one such pair. The data point $z \in \mathbb R^{H \times 2}$
is the chunk of the next H targets, relative to the current state and normalized. The guiding variable $y$ is the last
K states and the goal, also relative and normalized ({doc}`notation`). The windows are kept on the training device.
{py:func}`train.run_training` builds them like this:

```python
from pathlib import Path

import torch

from data import PushtChunkDataset, download_pusht, load_pusht_zarr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
states, actions, episode_ends = load_pusht_zarr(download_pusht(Path("data")))
dataset = PushtChunkDataset(
    states,
    actions,
    episode_ends,
    frame="block",
    obs_horizon=2,
    pred_horizon=16,
    action_history=False,
    padding="mask",
    device=device,
)
obs, goal, chunk, weight = dataset.batch(
    torch.randperm(len(dataset), device=device)[:128]
)
```

From the command line, the same choices are the `data.*` keys:

```bash
uv run train.py data.frame=agent                          # the agent's frame instead of the block's
uv run train.py data.obs_horizon=4 data.action_history=true
uv run train.py data.pred_horizon=64 data.padding=hold
```

## From the dataset to a batch

{py:func}`~data.download_pusht` fetches the Diffusion Policy archive into `data_dir` once and returns the path of
its Zarr store. {py:func}`~data.load_pusht_zarr` reads three flat arrays:

- the states, one row per step: [agent xy, block xy, block angle];
- the actions: the PD set-point (the agent's target position) applied after each state, so the action of step t is
  the command issued after observing state t;
- the episode ends, one past each episode's last step.

The states and the actions are read in float64 (the episode ends as int64), the precision of the env's own
observations, so the relative conversion gives the same numbers in training and at evaluation.

The constructor of {py:class}`~data.PushtChunkDataset` does all the work once:

1. {py:func}`~data.build_windows` turns the episode ends into index arrays: the current step t of every window,
   its K history steps and H future steps, and which future steps lie inside the episode.
2. With action history, {py:func}`~data.previous_set_points` looks up the set-point that produced each history
   frame.
3. {py:func}`~data.relative_observation` expresses the K history states (and set-points) and the goal in the frame
   of the current state; {py:func}`~data.relative_actions` expresses the H future targets as displacements from the
   current agent position.
4. The loss weight is 1 on every target, or 0 on padded targets with `data.padding=mask`.
5. {py:meth}`Normalizer.from_data <data.Normalizer.from_data>` fits the statistics, and the normalized windows are
   copied to the device in float32.

Nothing is computed per batch after that. {py:meth}`~data.PushtChunkDataset.batch` gathers windows by index, and
{py:func}`~train.run_training` draws the indices: an epoch is a fresh `torch.randperm` on the device, cut into
batches (windows past the last full batch of a permutation sit out that epoch). There is no DataLoader and
no host copy, so the gather is part of the captured training step ({doc}`training`).

A batch holds four tensors, which {py:meth}`~model.FlowMatchingPolicy.compute_loss` takes in this order:

- `obs`: K history frames of 6 features each (8 with action history), normalized;
- `goal`: one goal token of 4 features, normalized; with `obs`, the guiding variable $y$;
- `actions`: the normalized relative chunk, H targets of 2 coordinates, the data point $z \in \mathbb R^{H \times 2}$
  of the flow pages;
- `weight`: one loss weight per chunk step.

The dataset also keeps its {py:class}`~data.Normalizer` (`dataset.normalizer`) and the number of padded targets,
which {py:func}`~train.run_training` stores with the run's config (`info.num_padded_targets`).

## The relative representation

Why express everything relative to the current state? In world coordinates the same push looks different at every
place on the table, and the policy would have to learn each place separately. Every quantity is therefore expressed in
the axes of a reference frame attached to the current state: positions relative to the frame's origin, actions and
set-points relative to the agent (see **Actions**). This is the planar version of UMI's relative pose representation,
with the current pose as the base. In SE(2) a frame is an origin $o$ and a heading $\phi$, with the rotation

$$
R(\phi) = \begin{pmatrix} \cos\phi & -\sin\phi \\ \sin\phi & \cos\phi \end{pmatrix}.
$$

{py:func}`~data.to_frame` maps a world point $r$ into the frame and {py:func}`~data.from_frame` maps it back:

$$
r' = R(\phi)^\top (r - o), \qquad r = o + R(\phi)\, r'.
$$

The code subtracts first and rotates second, so the frame's own origin maps to exactly 0, not to a rounding error.
An angle enters as the cosine and sine of its difference to the heading, for the block angle $\phi^{\mathrm b}$ the
pair $(\cos(\phi^{\mathrm b} - \phi), \sin(\phi^{\mathrm b} - \phi))$. The pair is continuous; the raw block angle
wraps from $2\pi$ to 0 in many demonstrations, and the pair has no such jump.

{py:func}`~data.reference_frame` picks the frame of a state from `data.frame`:

- `block` (the default): the block's current pose, its position as $o$ and its angle as $\phi$.
- `agent`: the agent's current position with the world's axes, $\phi = 0$. The agent is a disk without heading, so
  this is the literal UMI analogue with the agent as end effector.

The paragraph **Which frame?** below compares the two.

Write $s_\tau$ for the world state at step $\tau$ (agent position $r^{\mathrm a}_\tau$, block position
$r^{\mathrm b}_\tau$, block angle $\phi^{\mathrm b}_\tau$), $a_\tau$ for the action, and $(o_t, \phi_t)$ for the frame
of the current state $s_t$ with $R_t = R(\phi_t)$. On this page $t$ and $\tau$ index env steps; only
{ref}`Normalization <data-normalization>` uses $t$ as the flow time. [Letters with two
meanings](notation.md#letters-with-two-meanings) lists the letters that serve twice.

**Observation.** {py:func}`~data.relative_observation` turns the history $s_{t-K+1}, \dots, s_t$ into K frames, all
in the frame of $s_t$:

$$
f_\tau = \big[\, R_t^\top (r^{\mathrm a}_\tau - o_t),\; R_t^\top (r^{\mathrm b}_\tau - o_t),\;
\cos(\phi^{\mathrm b}_\tau - \phi_t),\; \sin(\phi^{\mathrm b}_\tau - \phi_t) \,\big].
$$

In the last frame, $f_t$, the reference object's own entries are constant: in the block frame the block reads
(0, 0, 1, 0), in the agent frame the agent reads (0, 0). They are kept so that every frame has the same layout; the
normalizer maps them to 0. With action history each frame also carries the set-point that produced it, as a
displacement from the current agent position, $R_t^\top (a_{\tau-1} - r^{\mathrm a}_t)$: the same representation as
the predicted actions.

**Actions.** {py:func}`~data.relative_actions` expresses the H future targets as displacements from the current agent
position, in the frame's axes:

$$
a'_{t+k} = R_t^\top (a_{t+k} - r^{\mathrm a}_t), \qquad k = 0, \dots, H-1.
$$

The origin is the agent, not the frame's origin $o_t$. An action is the set-point of the agent's PD controller, so the
policy says where the agent goes from where it is. Measured from the agent, the first target has a standard deviation
of about 20 px per coordinate, against about 70 px measured from the block. Since
$a'_{t+k} = R_t^\top(a_{t+k} - o_t) - R_t^\top(r^{\mathrm a}_t - o_t)$, the chunk is still written in the frame: the
target's frame coordinates minus the agent's. In the agent frame the two origins coincide; in the block frame they
differ.

After {ref}`normalization <data-normalization>`, $(a'_t, \dots, a'_{t+H-1})$ is the chunk $z$. At evaluation
{py:func}`~data.absolute_actions` inverts this with the pose of the query state,
$a_{t+k} = r^{\mathrm a}_t + R_t\, a'_{t+k}$, and the rollout clips each target to the env's action space before it
executes it.

**Goal.** The goal is a fixed world pose $(r^{\mathrm g}, \phi^{\mathrm g})$. A relative observation alone would
lose it, and with it the walls, so it enters as one more token, its pose in the same frame:

$$
\big[\, R_t^\top (r^{\mathrm g} - o_t),\; \cos(\phi^{\mathrm g} - \phi_t),\; \sin(\phi^{\mathrm g} - \phi_t) \,\big].
$$

Does a representation relative to the current state forget where the scene is? It does not, because the goal is
fixed in the world.

:::{admonition} Proposition: no information is lost
:class: important

Since the goal pose $(r^{\mathrm g}, \phi^{\mathrm g})$ is a known constant, the world history
$s_{t-K+1}, \dots, s_t$ (block angles modulo $2\pi$) can be recovered from the K frames $f_\tau$ and the goal token.
:::

In other words, the relative observation is one-to-one: it moves the world into the current frame but forgets nothing,
not even where the walls are. The recovery takes three steps. (i) The goal token's angle pair is
$(\cos(\phi^{\mathrm g} - \phi_t), \sin(\phi^{\mathrm g} - \phi_t))$, which gives $\phi_t$ modulo $2\pi$. (ii) Its
position is $r'^{\mathrm g} = R_t^\top(r^{\mathrm g} - o_t)$, so the inverse frame map gives
$o_t = r^{\mathrm g} - R_t\, r'^{\mathrm g}$. (iii) With $(o_t, \phi_t)$ known, every position maps back by
$r = o_t + R_t r'$, and every angle pair gives $\phi^{\mathrm b}_\tau = \phi_t + \operatorname{atan2}(\sin, \cos)$.

The map is not onto the space of its numbers, though. The angle pairs lie on the unit circle, the reference object's
entries of $f_t$ are constant, and the goal token's four numbers carry three degrees of freedom, or two in the agent
frame, where its angle pair is constant.

The goal token is what makes the recovery possible. Without it, steps (i) and (ii) have nothing to read, and the frames
alone do not fix the world pose of the current frame. In the block frame, every rigid motion of agent and block together
gives the same frames. In the agent frame $\phi_t = 0$ is known, but every translation gives the same frames.

Normalized, the K frames $f_\tau$ and the goal token are the guiding variable $y$ that the network reads ({doc}`model`).

The same functions run at query time on the env's observations, in {py:func}`evaluation.query_policy`
({doc}`evaluation`).

**Which frame?** The two frames differ along one axis: which rigid motions of the scene leave the history frames
$f_\tau$ unchanged. The goal token, fixed in the world, then carries what those motions change.

| | `block` (default) | `agent` |
| --- | --- | --- |
| origin $o_t$, heading $\phi_t$ | the block's position and angle | the agent's position, $\phi_t = 0$ |
| motions of agent and block together that leave $f_\tau$ unchanged | every rotation and translation | translations only |
| constant entries of $f_t$ | the block's, $(0, 0, 1, 0)$ | the agent's, $(0, 0)$ |
| goal token | the goal's pose seen from the block: the pose error left to correct | the goal's position seen from the agent; its angle pair is constant |
| actions | displacements from the agent, in the block's axes | displacements from the agent, in the world's axes |

In theory the two frames carry the same information: by the proposition above, either one determines the world history.
They differ in what the network must learn. In the block frame, two scenes that differ by a rigid motion have the same
history frames. What the network learns about pushing the block at one place and orientation therefore carries over
to every other, and only the goal token tells the scenes apart. In the agent frame only translations are shared, and
the network must learn each orientation of the scene separately. In absolute world coordinates, the baseline of the
first finding below, no motion leaves the features unchanged. How much this sharing matters with 206 demonstrations is
a question of practice, which the findings below answer.

**What we found**

- **relative > absolute world coordinates.** Mean max reward is clearly higher and rises faster; success is tied,
  both near the floor of the MLP they were measured with.
- **block frame > agent frame.** Mean max reward is higher, on held-out episodes too; success is tied.

Both ran with the MLP, Adam and the conditional flow matching loss $\mathcal L_{\text{CFM}}$ (`velocity_mse`) at
H = E = 8. The absolute row was the original nano-PushT pipeline (K = 1, boundary windows dropped, its own loader), a
baseline rather than a clean ablation of the representation; world coordinates are not an option in the code. The
plan had named the agent frame for a tie on success; with success tied near its floor, the decision went to mean max
reward after seeing the data.

Try: `uv run train.py data.frame=agent`

## Windows and padding

{py:func}`~data.build_windows` starts one window at every step of every episode, for any K and H. The history of a
window is $t-K+1, \dots, t$ and its future $t, \dots, t+H-1$; both are clamped to the episode.

- **Before the start**, the history repeats the episode's first state. That is what the env shows right after a reset
  (the agent at rest), and {py:func}`~evaluation.evaluate_policy` fills the history of a new episode the same way.
- **Past the end**, the future repeats the episode's last action. `data.padding` decides what the loss does with
  those steps:
  - `hold`: the repeated action is a target like any other;
  - `mask` (the default): the window stays, and the padded steps get weight 0;
  - `drop`: windows that run past the end are discarded.

The three modes differ along one axis: what the loss asks of a step past the episode's end.

| `data.padding` | windows kept | padded steps in the loss and the normalizer | under the `sum` reduction, step k learns the posterior mean over |
| --- | --- | --- | --- |
| `hold` | all | targets, weight 1 | all windows, with the last action held past the end |
| `mask` (default) | all | weight 0 | the windows in which step k lies inside the episode ([Reduction](flow/training.md#reduction)) |
| `drop` | those whose future ends inside the episode | none | the windows kept, which never use an episode's last H − 1 steps as the current state |

Step 0 of a window always lies inside its episode, so every window keeps at least one weighted target. The weight
reaches the loss through {py:meth}`~flow.FlowMatching.per_example`, which multiplies each step's squared error by it
before the chunk's error is reduced ({doc}`flow/training`). With `mask`, the same weight also keeps the padded
targets out of the normalizer's statistics.

**What we found**

- **H = 16, MLP: hold > drop, hold = mask.** Repeating the last action beats discarding the boundary windows, and
  whether the padded targets are trained on (hold against mask) makes no detectable difference.
- **H = 64, DiT, averaged (EMA) weights: mask > hold.** Hold's failures are mostly near misses; a plausible reading
  (not tested) is that repeating the last action teaches the policy to settle once the T is nearly in place.
- **H = 64, DiT, raw weights: mask > hold (weak).** The direction agrees with the averaged weights.

At H = 64 the MLP fails with every padding, so the long-horizon check was repeated with the DiT ({doc}`model`), with
Adam and $\mathcal L_{\text{CFM}}$. `mask` is the default since.

Try: `uv run train.py data.pred_horizon=64 data.padding=hold`

(data-normalization)=
## Normalization

Why normalize the chunk? The [probability path](flow/index.md#the-gaussian-probability-path) mixes it with the noise one
coordinate at a time, $x = \alpha_t z + \beta_t\epsilon$ with $\epsilon \sim \mathcal N(0, I)$, where the noise
schedulers $\alpha_t$ and $\beta_t$ set the share of signal and noise at the flow time $t$. How noisy coordinate $i$ is
at time $t$ is therefore set by its signal-to-noise ratio $\alpha_t^2\,\mathrm{Var}(z_i)/\beta_t^2$. In pixels, the
standard deviation of a relative target grows with its step, from about 20 at the first step to about 90 at the
sixteenth. Left in pixels, the far steps would be nearly clean at a time $t$ where the near ones are still buried, and
the schedulers would set a different noise level for every coordinate. Z-scoring gives every coordinate of $z$ mean 0
and variance 1, the scale of $\epsilon$. Every coordinate then has the path's own signal-to-noise ratio
$\alpha_t^2/\beta_t^2$, whose logarithm [Paths](flow/index.md#paths) tabulates. The variances of the noisy chunk stated
there assume the same z-scoring, and they hold per coordinate only: the normalizer scales each index on its own, so the
steps of a chunk stay correlated. The guiding variable $y$ is normalized for the usual reason, to give the network
inputs of unit scale.

{py:class}`~data.Normalizer` holds z-scores with one mean and one standard deviation per index: per history lag,
chunk step and feature for the observation and the chunk, per feature for the goal. For an index $i$, with $n$
running over the windows, $q_{n,i}$ the feature and weights $m_{n,i}$ (1 for the observation and the goal, the
loss weight for the chunk):

$$
\mu_i = \frac{\sum_n m_{n,i}\, q_{n,i}}{\sum_n m_{n,i}}, \qquad
\sigma_i^2 = \frac{\sum_n m_{n,i}\, (q_{n,i} - \mu_i)^2}{\sum_n m_{n,i}}, \qquad
q_i \mapsto \frac{q_i - \mu_i}{\sigma_i}.
$$

Here $\sigma_i$ is a standard deviation, not the diffusion coefficient $\sigma_t$ of the flow pages.

Masked targets have weight 0 and do not count. A feature whose standard deviation is below a small threshold
($10^{-4}$) is constant: the reference object's entries of the current frame and, in the agent frame, where
$\phi_t = 0$, the goal's angle pair. It gets $\sigma_i = 1$ and its constant as mean, so it normalizes to exactly 0.

Every step starts a window and the history padding repeats edge frames, so the statistics of a given lag or chunk
step are the same for every K and H (with `hold` or `mask`; `drop` removes windows by H). Per-index statistics
therefore do not move when K or H does: a longer chunk does not rescale the steps it shares with a shorter one, while
statistics pooled over a long chunk would shrink its normalized first step. This is a design choice; it was not
compared against pooled statistics.

The normalizer is fitted once, on the training windows. {py:func}`~evaluation.save_checkpoint` stores it with the
weights and {py:func}`~evaluation.load_policy` rebuilds it, so a policy is always queried with the statistics it was
trained on: {py:func}`~evaluation.query_policy` normalizes the observation and the goal into $y$, and maps the
sampled chunk $X_1$ back with {py:meth}`~data.Normalizer.denormalize_action`.

## History

`data.obs_horizon` is K, the number of states the policy sees, $t-K+1, \dots, t$. Two states already carry the
velocities of the agent and the block; one carries no motion at all, and {py:func}`~train.check_config` refuses
K < 2.

`data.action_history=true` adds to every history frame the set-point that produced it.
{py:func}`~data.previous_set_points` pairs frame $t-k$ with $a_{t-k-1}$, the target that was active when the frame
was observed, so frame $t$ carries $a_{t-1}$ and the K frames carry the K most recent actions. Before an episode's
first action the agent holds its initial position, so the padded set-points are that position. This holds in the
data and in {py:func}`~evaluation.evaluate_policy` right after a reset. At evaluation the history stores the clipped
targets that were actually executed.

**What we found**

- **K = 2 = K = 4 > K = 8 > K = 16.** Shorter is better; a likely reason (not tested) is that more frames are more
  inputs to fit with 206 demonstrations, though the gap shrinks with frequent replanning, so it is not only fitting.
- **states only > with action history, at every K.** Success drops clearly. Commands become markedly
  smoother (lower jerk, seams between chunks barely larger than ordinary steps), and mean max reward rises slightly at
  short histories. The pattern fits the copycat problem of behaviour cloning with action history (a hypothesis, not
  tested directly).

These ran with Adam, $\mathcal L_{\text{CFM}}$ and `hold` padding, scored at E = 8; the interaction of history with the
execution horizon, and the metrics, are on {doc}`evaluation`. The demonstrations themselves never reach the success
threshold, so success measures how often a policy goes beyond its data, and mean max reward, the more sensitive
metric, breaks ties.

Reproduce: `uv run train.py -m experiment=history` (it pins the study's settings: Adam, `velocity_mse`, `hold`
padding, E = 8 and the raw weights as the headline; it also scores every E at the late evaluations)
