# Overview

PushT is a research harness for flow matching policies that generate chunks of actions on the state-based Push-T
task. Every component (the representation, the network, the probability path, the flow-time distribution, the network's
output, the sampler and the training loop) is a Hydra option, and almost every default comes out of a measured
comparison.

```bash
uv sync                                   # Python 3.11+ and uv
uv run train.py experiment=smoke          # a one-minute end-to-end check
uv run train.py                           # train the default policy
uv run evaluation.py runs=runs/default    # score the finished runs on held-out episodes
```

The guide follows the data through the code: {doc}`data`, then the {doc}`model`, then {doc}`flow/index` (the
{doc}`training objective <flow/training>`, {doc}`sampling <flow/sampling>` and {doc}`MeanFlow <flow/meanflow>`), the
{doc}`training loop <training>` and {doc}`evaluation`. {doc}`configuration` and {doc}`logging` cover running
experiments and reading their output; {doc}`extending` and {doc}`contributing` are for changing the code.

## The method in four steps

What does a policy compute? It reads an observation and returns the next few actions. A flow matching policy does this
by sampling: it draws a chunk from the distribution of the expert's chunks for that observation. We first name the
objects. The first three steps then follow the FM Guide's blueprint, a path, a regression and a simulation, and the
fourth closes the control loop.

The **guiding variable** $y$ is the observation: the goal and the last $K = 2$ states. A **chunk**
$z \in \mathbb R^{H \times 2}$ holds the agent's next $H = 16$ target positions. Both are expressed in the frame of
the current state and normalized ({doc}`data`). The demonstrations give pairs $(z, y) \sim p_{\mathrm{data}}(z, y)$,
and the policy should sample $p_{\mathrm{data}}(\cdot \mid y)$. The **flow time** $t \in [0, 1]$ runs from noise at
$t = 0$ to data at $t = 1$.

1. **A path from noise to the expert chunks.** Draw noise $\epsilon \sim \mathcal N(0, I)$ of a chunk's shape and mix
   it with a chunk, $x = \alpha_t z + \beta_t \epsilon$. The noise schedulers $\alpha_t$ and $\beta_t$ make $x$ pure
   noise at $t = 0$ and the chunk at $t = 1$; the default is the linear path $\alpha_t = t$, $\beta_t = 1 - t$. For
   chunks drawn from $p_{\mathrm{data}}(\cdot \mid y)$, the law of $x$ therefore runs from $\mathcal N(0, I)$ at
   $t = 0$ to $p_{\mathrm{data}}(\cdot \mid y)$ at $t = 1$. These laws form the **probability path** $p_t$
   ([The Gaussian probability path](flow/index.md#the-gaussian-probability-path)).
2.  **Regress a network onto a conditional target.** The network sees $(x, t, y)$ and is trained by least squares onto a
   target computed from $(z, \epsilon)$. Each training step draws the flow time from a **flow-time distribution**,
   uniform by default. An example of a target is the velocity of $x$ along its path,
   $\dot\alpha_t z + \dot\beta_t \epsilon$, where the dots are time derivatives. At the minimizer the network outputs
   the posterior mean of the target given $(x, t, y)$: the target averaged over every pair $(z, \epsilon)$ that could
   have produced $x$. This holds for an expressive enough network, at every flow time that training draws. For the
   velocity, that average is the marginal vector field, whose ODE carries $\mathcal N(0, I)$ to
   $p_{\mathrm{data}}(\cdot \mid y)$. Training never simulates the ODE ([Regression learns a posterior
   mean](flow/training.md#regression-learns-a-posterior-mean)).
3.  **Simulate from noise to a chunk.** Convert the network's output to a vector field $u_t^\theta(x \mid y)$. Draw
   $X_{t_0} \sim \mathcal N(0, I)$ at a start time $t_0$ and integrate
   $\frac{\mathrm d}{\mathrm dt}X_t = u_t^\theta(X_t \mid y)$ to $t = 1$; $X_1$ is the chunk. By default this takes 10
   Euler steps. An SDE with the same marginals, which adds noise along the way, is the other option
   ({doc}`flow/sampling`).
4. **Execute $E$ steps, then replan.** The environment executes the first $E = 4$ targets of the chunk, and the policy
   is queried again with the new observation ({doc}`evaluation`).

Note that the policy returns a sample, not the mean chunk. The mean $\mathbb E[z \mid y]$ is what a policy regressed
with a squared error outputs, and when the demonstrations for one observation take two routes, it can lie between them.
Each noise draw $X_{t_0}$ instead leads to one chunk ([The marginal vector
field](flow/index.md#the-marginal-vector-field)).

In theory, step 2 learns the vector field exactly and step 3 integrates it exactly; then neither the path, nor the
network's output, nor the solver changes the samples. In practice the network has an error and the solver takes a few
steps, so these choices matter, and the flow pages compare them. The default network outputs the **scaled score**, an
estimate of minus the expected noise in $x$, rather than the velocity. Its conversion to a vector field divides by
$\alpha_t$, which is 0 at $t = 0$, so sampling starts at $t_0 = 0.05$ ([Where sampling
starts](flow/sampling.md#where-sampling-starts)). Symbols, shapes and the time conventions are on {doc}`notation`.

## Core components

The six modules at the repository root are the library. Each has an API page generated from its docstrings.

**data** ({py:mod}`data`) builds the pairs $(z, y)$ of the method from the raw Zarr steps of the demonstrations.
{py:func}`~data.download_pusht` and {py:func}`~data.load_pusht_zarr` read the flat arrays of states, actions and
episode ends. {py:class}`~data.PushtChunkDataset` starts one window at every step (except the last H - 1 steps of an
episode under `data.padding=drop`), expresses the history, the goal and the action chunk in the frame of the current
state ({py:func}`~data.relative_observation`, {py:func}`~data.relative_actions`), fits a {py:class}`~data.Normalizer`
and keeps the normalized tensors on the device. Evaluation reuses {py:func}`~data.relative_observation` and the
normalizer on the env's states, and maps each chunk back to world targets with {py:func}`~data.absolute_actions`.

**model** ({py:mod}`model`) holds the policy, the network of steps 2 and 3. {py:class}`~model.FlowMatchingPolicy` is
the contract: training calls {py:meth}`~model.FlowMatchingPolicy.compute_loss`, evaluation calls
{py:meth}`~model.FlowMatchingPolicy.sample_actions`, and the policy carries the flow objective it was built with. A
subclass supplies the network. It reads the noisy chunk $x$, the flow time $t$ and the guiding variable $y$, and it
outputs what the objective asks for: a velocity, a clean chunk or the scaled score. {py:class}`~model.DiTPolicy` is a
diffusion transformer with one token per chunk step, and {py:class}`~model.MLPPolicy` a multilayer perceptron.

**flow** ({py:mod}`flow`) is the math of steps 1 to 3, for chunks of any shape. Its probability paths
({py:class}`~flow.LinearPath`, {py:class}`~flow.GVPPath`) implement step 1. Its flow-time distributions
({py:class}`~flow.UniformTime`, {py:class}`~flow.LogitNormalTime`), what the network outputs
({py:class}`~flow.Prediction`) and the objectives {py:class}`~flow.FlowMatching` and {py:class}`~flow.MeanFlow`
implement step 2. The {py:class}`~flow.Sampler` implements step 3, simulating the ODE or SDE from noise to a chunk.
The module knows nothing about Push-T: the network enters as a function `predict(x, t[, s])`, where `s` is the end
time of a MeanFlow step ({doc}`flow/meanflow`).

**train** ({py:mod}`train`) runs step 2. {py:func}`~train.run_training` checks the config, builds the dataset, the
policy, its weight averages and the optimizers, captures the whole optimizer step as one CUDA graph, evaluates the
policy periodically and saves the checkpoint.

**evaluation** ({py:mod}`evaluation`) runs step 4 in closed loop. {py:func}`~evaluation.evaluate_policy` plays one
episode per env seed, all in lock-step; {py:func}`~evaluation.save_checkpoint` and {py:func}`~evaluation.load_policy`
store and rebuild a policy; its command line scores finished runs under any sampler without retraining.

**logger** ({py:mod}`logger`) writes every row of scalars to the run's `log.csv` and to a W&B run, and lays out the
W&B workspace.

## Two paths through the code

Training builds windows once and then repeats one captured step:

```text
conf/*.yaml -> Hydra -> train.main -> run_training
  check_config                          fail before any data or GPU work
  download_pusht, load_pusht_zarr       states (T, 5), actions (T, 2), episode ends, world units
  PushtChunkDataset                     relative, normalized windows on the device; its Normalizer
  evaluation.build_policy               the model's _target_ with the flow objective
  build_optimizers, build_loss          Muon + AdamW, or AdamW alone; the loss compiled by TorchInductor
  every optimizer step (one CUDA graph):
      dataset.batch(index) -> policy.compute_loss -> flow.loss(predict, chunk, weight)
      -> backward -> optimizers -> weight averages (EMA)
  every log.train_interval steps:       Logger.log_training
  every eval.interval steps:            evaluate_policy -> Logger.log_evaluation; save_checkpoint
```

{py:func}`~train.main` is the Hydra entry point: it names the run and creates its {py:class}`~logger.Logger`, then
calls {py:func}`~train.run_training`. {py:func}`~train.check_config` refuses settings that cannot run together.
{py:class}`~data.PushtChunkDataset` holds every window as tensors on the device, so a batch is an index gather
({py:meth}`~data.PushtChunkDataset.batch`): history frames, the goal, a chunk of H normalized relative targets and a
loss weight per step. {py:func}`~evaluation.build_policy` instantiates the model's `_target_` with the flow objective
of the config. Inside the step, {py:meth}`~model.FlowMatchingPolicy.compute_loss` encodes $y$ and hands
{py:meth}`~flow.FlowMatching.loss` a closure over the network. The objective of step 2 draws the flow times $t$ and
the noise $\epsilon$, builds $x$ and compares the network's output with its target ({doc}`flow/training`).
{py:func}`~train.build_optimizers` and {py:func}`~train.build_loss` set up what the step calls. The loss sums stay on
the device until a logging step. See {doc}`training`.

A query, during evaluation or deployment, goes the other way:

```text
the last K env states (bs, K, 5), world units
  -> relative_observation               frames and goal in the current state's frame
  -> Normalizer.normalize_obs, normalize_goal
  -> policy.sample_actions              encode y once; LearnedFlow(predict, path, prediction)
       -> Sampler.sample                simulate from noise to t = 1: a normalized chunk (bs, H, 2)
  -> Normalizer.denormalize_action
  -> absolute_actions                   world targets (bs, H, 2) from the query state's pose
  -> clip to the action space, execute E of them, query again
```

{py:func}`~evaluation.query_policy` is one query. {py:func}`~data.relative_observation` expresses the history and the
goal in the frame of the current state, and the training statistics normalize them
({py:meth}`~data.Normalizer.normalize_obs`, {py:meth}`~data.Normalizer.normalize_goal`).
{py:meth}`~model.FlowMatchingPolicy.sample_actions` encodes $y$ once and wraps the network for that $y$ in a
{py:class}`~flow.LearnedFlow`, which converts its output to the vector field $u_t^\theta(x \mid y)$. With a guidance
scale $w \ne 1$ (`sampler.cfg_scale`), the wrapped network first mixes its outputs with and without $y$, and the same
conversion gives the classifier-free guided field $\tilde u_t^\theta(x \mid y)$ ([Guidance](flow/sampling.md#guidance)).
{py:meth}`~flow.Sampler.sample` then simulates the ODE or SDE of step 3 to $t = 1$, from $X_{t_0} \sim \mathcal N(0, I)$
at the start time $t_0$, which by default (`sampler.start_time` unset) is 0.05 for the scaled-score output and 0 for the
other outputs. Each episode draws $X_{t_0}$ from a random stream of its own, seeded by its env seed, so its chunks do
not depend on which other episodes share the batch. {py:meth}`~data.Normalizer.denormalize_action` and
{py:func}`~data.absolute_actions` turn the chunk back into world targets with the pose of the query state.
{py:func}`~evaluation.evaluate_policy` clips them to the action space, executes the first E and queries again. See
{doc}`evaluation`.

## Conventions

- **Notation.** The math follows Holderrieth and Erives' notes on flow matching and diffusion models.
  {doc}`notation` cites them, lists every symbol with its code name and config key, and translates the symbols to the
  other references ([Dictionary to the references](notation.md#dictionary-to-the-references)).
- **Flow time.** Noise sits at $t = 0$ and data at $t = 1$, and samplers simulate upward in $t$. MeanFlow's papers
  and the diffusion literature use other clocks ([Time conventions](notation.md#time-conventions)).
- **Shapes.** The API pages write a chunk as `(bs, H, d)`, with `d` = 2 the action dimension
  ([Code names and config keys](notation.md#code-names-and-config-keys)). A history frame has 6 features (8 with the
  action history), the goal 4.
- **Coordinates.** Everything the network sees and predicts is relative to the current state and normalized. World
  units exist only in the raw dataset and at the env boundary of a query.

## Reading the findings

Pages that cover a measured option end each topic with **What we found**, one bullet per comparison. A bullet opens
with a verdict:

| verdict | meaning |
| --- | --- |
| `A > B` | A ahead by more than the seed-to-seed noise |
| `A >> B` | a large gap |
| `A = B` | no difference detected, which is not proof of equality |
| `A > B (weak)` | A ahead, within the noise |

- **Metrics.** Success (coverage of the goal above 95% at any step of an episode) is the primary metric. Mean max
  reward (the per-episode maximum reward, averaged over episodes) is secondary and breaks ties. It is the more
  sensitive of the two because the demonstrations never reach the success threshold: success measures how often a
  policy goes beyond its data.
- **Episodes.** Decisions use the selection episodes (env seeds 0 and up), which the periodic evaluations of training
  play. Final comparisons use held-out episodes (env seeds 10000 and up), which no decision saw. A run's headline is
  its success at the default setting pooled over its late evaluations.
- **Weights.** Raw weights are the optimizer's; averaged weights are their moving average (EMA), the policy that is
  evaluated by default (the checkpoint holds both). A finding says which it rests on where they disagree.
- **Settings of the time.** Each comparison ran on the defaults of its time (early ones on an MLP, Adam or the
  velocity loss, `velocity_mse`); a finding says so where it changes the reading.
- **Numbers.** These pages give directions, not numbers. The success rates and intervals are in the experiment
  report; the README lists the headline results.

Where a finding names a phase, it is the phase of the study as the experiment report numbers them. Phases 1 to 4 ran
in nano-PushT, a reduced copy of this code, and their conclusions carry over:

| phase | where | what it compared |
| --- | --- | --- |
| 1 | nano-PushT | relative and absolute coordinates, the reference frame, padding (MLP) |
| 2 | nano-PushT | the DiT against the MLP, conditioning modes, classifier-free guidance, padding at H = 64 |
| 3 | nano-PushT | observation and action history, prediction and execution horizons, query cost |
| 4 | nano-PushT | the training loop: precision, weight averages, learning rate and schedule, batch size, Adam's details, length |
| 5 | PushT | AdamW's weight decay, Muon |
| 6 | PushT | probability paths, network outputs and losses, flow-time distributions, reductions, samplers |
| 7 | PushT | Muon with the noise predictor (`scaled_score_mse`), today's default |
| MeanFlow study | PushT | MeanFlow's recipe, and MeanFlow against flow matching |

The evaluation protocol is in {doc}`evaluation`; the defaults with their evidence, one row per component, are in
{doc}`configuration`.
