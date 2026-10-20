# Evaluation

Evaluation plays Push-T episodes in closed loop with a policy and scores them. `train.py` runs it periodically on the
selection episodes; `evaluation.py` scores finished runs again, on held-out episodes by default, under any sampler
and execution horizon, without retraining:

```bash
uv run evaluation.py runs=runs/default                                   # held-out success of every finished run below
uv run evaluation.py runs=runs/default 'exec_horizons=[2,4,8]'            # three execution horizons, same checkpoints
uv run evaluation.py -m runs=runs/default 'sampler.nfe=1,2,4,10' latency=true   # an NFE sweep with query latency
```

## Closed-loop rollouts

A query returns a chunk of H targets. The rollout executes the first E of them, the **execution horizon**
(`eval.exec_horizon`, [below](#execution-and-prediction-horizons)), before it queries again.
{py:func}`~evaluation.evaluate_policy` plays one episode per env seed, all in lock-step. Every episode starts at step 0
and queries the policy at steps $0, E, 2E, \dots$, so stepping all environments together and batching the queries of the
live episodes reproduces the one-episode-at-a-time protocol (up to floating-point differences between batch sizes).
Between two queries an episode executes E actions of its chunk open-loop.

Before the first step the state history repeats the reset state K times, as the dataset pads episode starts
({doc}`data`). With action history, the set-points start at the agent's reset position, which the agent holds until
its first action, as in the dataset. After each step the newest state, and the target just executed, enter the
history.

{py:func}`~evaluation.query_policy` does the conversions around the policy, for a batch of episodes:

1. {py:func}`~data.relative_observation` expresses the K world states (and set-points) and the goal in the frame of
   the current state;
2. the run's {py:class}`~data.Normalizer` normalizes them ({py:meth}`~data.Normalizer.normalize_obs`,
   {py:meth}`~data.Normalizer.normalize_goal`);
3. {py:meth}`~model.FlowMatchingPolicy.sample_actions` simulates the learned ODE or SDE from each episode's noise
   $X_{t_0}$ to a chunk $X_1$ of H normalized relative actions ({doc}`flow/sampling`; symbols as in {doc}`notation`);
4. the chunk moves back to the CPU, and {py:meth}`~data.Normalizer.denormalize_action` undoes the normalization;
5. {py:func}`~data.absolute_actions` turns the relative actions into world targets with the pose of the query state.

The rollout clips an action to the environment's action space only when it executes it: physical limits belong to
execution, and no intermediate state $X_t$ of the sampler is ever clipped. An episode ends on success or after 300
steps, the environment's time limit. Evaluation owns coordinates and environment execution; the policy owns
generation and sees only normalized relative tensors.

## Metrics

Evaluation scores each episode by two metrics, success and max reward. How do they relate? Both come from the
**coverage** $c_\tau \in [0, 1]$, the fraction of the goal's area that the block covers after env step $\tau$. The
environment's reward is $r_\tau = \mathrm{clip}(c_\tau/0.95, 0, 1)$, and an episode succeeds at the first step with
$c_\tau > 0.95$, which also ends it. Write $c_{\max}$ for the episode's highest coverage.
{py:class}`~evaluation.Rollouts` holds two outcomes per episode, indexed like the env seeds:

- **success**: $S = \mathbb 1[c_{\max} > 0.95]$, the environment's `is_success` at some step. The success rate is
  the primary metric.
- **max reward**: $R = \max_\tau r_\tau$, the episode's highest reward. Its mean over episodes, the mean max reward, is
  the secondary metric and breaks ties.

:::{admonition} Proposition: mean max reward is success with partial credit
:class: important

Over a set of episodes,

$$
\text{mean max reward} = \text{success rate} + (1 - \text{success rate})\;\mathbb E\big[c_{\max}/0.95 \,\big|\, S = 0\big],
$$

where the conditional mean runs over the failed episodes.
:::

In other words, mean max reward is the success rate plus partial credit for every failure, in proportion to how close
it came. The identity takes three steps. (i) A successful episode reaches $c_\tau > 0.95$ at some step, where the
reward is clipped to 1, so $R = 1$. (ii) A failed episode has $c_\tau \le 0.95$ at every step, so
$R = c_{\max}/0.95 \le 1$. (iii) Averaging $R$ over the episodes splits the mean by $S$. Two policies with the same
success rate therefore still differ in mean max reward when their failures differ. Mean max reward responds to every
gain in a failed episode's highest coverage, and success only to gains that cross 0.95. Mean max reward is therefore
the more sensitive metric and breaks ties, while success stays primary because it is the task.

{py:meth}`~evaluation.Rollouts.summary` averages the two outcomes into `success_rate` and `mean_reward` (the mean max
reward, not a mean of per-step rewards), logged as `eval/success_rate` and `eval/mean_reward` during training
({doc}`logging`).

Mean max reward matters here because the demonstrations never reach the success threshold. A policy that imitated them
perfectly would almost never succeed, so success measures how often a policy goes beyond its data, while mean max reward
also grades the failures. A success depends on the policy's small corrective motions near the goal, which may be
sensitive to details that were not varied (the coverage threshold, the simulator's damping).

## Seeds and splits

{py:func}`~evaluation.env_seeds` returns the env seeds of a split. `selection` starts at 0 and serves the periodic
evaluations of training and every decision of the study; `heldout` starts at 10000 and is kept for final numbers and
confirmations. `evaluation.py` scores the held-out split by default (`split=heldout`). The run's `seed` is
separate: it sets the initialization of $\theta$, the batch order, the flow times $t$ and noise $\epsilon$ of the
training loss and its condition-dropout draws, and evaluation never draws from it.

Each episode draws its sampling noise from two generators of its own, seeded by its env seed: one for the initial
noise $X_{t_0}$ of every query, one for the Brownian draws of the SDE sampler. An episode's chunks therefore do not
depend on which other episodes are still running, and two samplers or two execution horizons draw the same noise
stream on the same episode: their comparisons are paired.

**Periodic evaluation.** {py:func}`~train.run_training` evaluates every `eval.interval` steps and at the last step,
on the first `eval.episodes` selection seeds (200 by default), with the run's sampler. It scores the default setting,
the first entry of `eval.weights` at `eval.exec_horizon`, and saves a checkpoint after each evaluation
({py:func}`~evaluation.save_checkpoint`). The last `eval.num_late` evaluations are the late ones: they also score
every weight copy of `eval.weights` at `eval.exec_horizon` and at every `eval.late_exec_horizons` up to H. When
training ends, the run's headline `late/success_rate` is the default setting's success pooled over the late
evaluations; pooling damps checkpoint-to-checkpoint swings. Every other setting gets its own `late/` key
({doc}`logging`).

**Weights.** `eval.weights` defaults to the averaged copy $\bar\theta$, `ema0.999` ({doc}`training`); `raw` scores the
optimizer's parameters $\theta$. `evaluation.py` picks one copy with `weights=`.

## Execution and prediction horizons

H (`data.pred_horizon`, 16) is a training parameter: the network predicts the next H actions. E
(`eval.exec_horizon`, 4) is an inference parameter: the rollout executes the first E of them before it queries
again. One trained model can therefore be scored at every E ≤ H on the same checkpoints, and comparisons between E
are paired: during training through `eval.late_exec_horizons`, afterwards through `evaluation.py exec_horizons=`.
{py:func}`~train.check_config` and {py:func}`~evaluation.evaluate_policy` refuse an E outside $[1, H]$.

**What E trades.** E trades feedback against the consistency of the plan and the number of queries. Each query samples a
new chunk for its $y$, from fresh noise. In theory E would not matter: if the policy were exact and $y$ held everything
the demonstrator acted on, a new query after E steps would continue the same plan in distribution. In practice $y$ holds
the last K states, which do not show the demonstrator's intent, for example the side from which it was about to push the
T. Two queries can then return plans from different modes. A small E reacts quickly to what the scene does, but it can
switch modes from one query to the next. A large E keeps one sampled plan for E steps and makes only
$\lceil \ell/E \rceil$ queries in an episode of $\ell$ steps, but it ignores the states that those steps produce.

**What H changes.** H fixes the data point: a chunk is $z \in \mathbb R^{H \times 2}$, a plan that reaches H steps
ahead, and E can be at most H. In theory H does not matter once E ≤ H. With `hold` or `mask` padding, the windows,
their first E targets and the statistics of those targets are the same for every H ({doc}`data`). The first E steps
of a chunk drawn from $p_{\mathrm{data}}(\cdot \mid y)$ therefore have the same law whatever H is, and a policy that
sampled $p_{\mathrm{data}}(\cdot \mid y)$ exactly would execute the same actions. In practice the network is trained
on the whole chunk, so H changes the learning problem: a longer chunk is a distribution of higher dimension to fit
from 206 demonstrations. Changing H also means training another network, so comparisons between values of H compare
different networks, while comparisons between values of E score one network. Which H fits best is a question of
practice, which the findings below answer.

The study also measured the smoothness of the executed (clipped) world targets $a_\tau$ at env step $\tau$ (the
actions of {doc}`data`, not the coefficient $a_t$), outside the code:

- **jerk**: the study's name for the mean size of the second difference $a_{\tau+1} - 2a_\tau + a_{\tau-1}$ of the
  executed targets. Strictly, it is a discrete acceleration of the commanded position. It measures how abruptly the
  commanded step $a_{\tau+1} - a_\tau$ changes from one env step to the next, the overall roughness of the command;
- **seam step**: the mean step $|a_\tau - a_{\tau-1}|$ at the replanning steps $\tau = E, 2E, \dots$, where a new chunk
  takes over;
- **inside step**: the same mean over the steps inside a chunk. The ratio of seam to inside step isolates the
  discontinuity between chunks from ordinary motion; at E = 1 every step is a seam.

**What we found** (phase 3, nano-PushT: Adam, $\mathcal L_{\text{CFM}}$ (`velocity_mse`), raw weights; the history
runs with `hold` padding at H = 16, the horizon runs with `mask`)

- **E = 4 > E = 8** in every configuration without action history, with larger gains at longer histories: long
  histories need frequent replanning. At K = 2 the gain is small, and E = 8, which halves the queries, is a
  reasonable alternative.
- **E = 4 >> E = 16.** Executing 16 steps per query, the whole chunk at H = 16, loses a lot everywhere.
- **E = 4 > E = 1 and E = 2** at K = 2 and K = 4 for every H, and up to K = 8 without action history. At K = 16,
  and with action history, E = 1-2 do as well or better.
- **H = 16 = H = 32 > H = 8 = H = 64 at K = 2.** Both too short and too long a plan lose.
- **H = 8 = H = 16 = H = 32 = H = 64 at K = 4.** A longer history makes the prediction horizon matter less.
- **smoothness: E = 8 > E = 4 > E = 1.** Replanning every step re-samples the plan from fresh noise and can switch
  between modes, the reason action chunking exists; jerk falls as E grows.
- **seam step > inside step.** Executing chunks trades the roughness of replanning for a jump every E steps.
- **smoothness: action history >> states only.** Lower jerk, and seams barely larger than ordinary steps; but the
  action history costs success ({doc}`data`).

Reproduce: `uv run train.py -m experiment=horizons` (H = 8 to 64 at K = 2, scored at E = 1, 2, 4, 8, 16, 32 up to H
at the late evaluations) and
`uv run train.py -m experiment=history` (K with and without action history, every E up to 16; it pins `hold`
padding and E = 8, as the history runs had); add `data.obs_horizon=4` to the first for the K = 4 row. Both pin their
studies' Adam, `velocity_mse` and raw-weight headline; smoothness is not computed by the code.

## Scoring finished runs without retraining

{py:func}`~evaluation.main` reads `conf/evaluation.yaml` and finds every finished run below `runs=` with
{py:func}`~evaluation.finished_checkpoints`: a `checkpoint.pt` whose step is the run's last (`train.steps`).
Interrupted runs are reported and left out. For each run, {py:func}`~evaluation.score_run`:

1. builds the configured sampler, which replaces the run's own;
2. rebuilds the policy with {py:func}`~evaluation.load_policy` from the file {py:func}`~evaluation.save_checkpoint`
   wrote: the run's resolved config and data-dependent arguments ({py:func}`~evaluation.build_policy`), the raw weights
   or an averaged copy (`weights=`), and the training {py:class}`~data.Normalizer`;
3. skips the run, with a message, if the sampler cannot serve it: classifier-free guidance without condition
   dropout, or anything {py:meth}`~flow.Sampler.check` refuses (Heun for a data prediction, the `mean` sampler for a
   policy not trained with {py:class}`~flow.MeanFlow`);
4. plays the first `episodes` env seeds of `split` at every E of `exec_horizons` (null: the run's
   `eval.exec_horizon`), and times queries if `latency=true` ([below](#latency));
5. logs the scores, `evaluate/success_rate`, `evaluate/mean_reward`, their per-E keys and `evaluate/nfe`, to
   `<run>/evaluations/<time>[/<job>]/` (`log.csv`, `config.json`) and to a W&B run of job type `evaluate` in the run's
   group, named after the run and the evaluation's overrides ({doc}`logging`).

A Hydra multirun (`-m`) sweeps evaluation options; each job writes its own subdirectory below every run it scores.
The configuration:

```{literalinclude} ../../conf/evaluation.yaml
:language: yaml
```

## Latency

`latency=true` adds {py:func}`~evaluation.time_queries`: batch-1 queries of one episode, each timed end to end
(relative conversion and normalization, the sampler's network evaluations, the copy back to the CPU and the
conversion to world targets). Untimed warm-up queries come first, and the GPU is synchronized before and after every
timed query. The median per query is logged as `evaluate/latency_ms`; it is what a deployed policy pays per
replanning, under the configured sampler. {py:func}`~evaluation.main` refuses to time while another process computes
on the GPU ({py:func}`~evaluation.gpu_processes`, which ignores processes that hold little GPU memory, such as a
remote desktop's video encoder).

**What we found** (phase 3, the DiT with `adaln` conditioning and the Euler method at 10 NFE unless noted; the NFE
from the MeanFlow study)

- **latency: K = 2 = K = 16, and H = 8 = H = 64.** At batch 1 a query is bound by kernel launches (network
  evaluations times blocks), not by arithmetic, so the history and chunk lengths barely change it.
- **speed: 1 NFE > 2 NFE > 10 NFE.** Latency grows with the network evaluations per query.
- **speed: adaln > concat > cross.** Classifier-free guidance, which doubles the network batch, also slows a query
  ({doc}`model`).
- **speed per episode: E = 4 >> E = 1.** An episode of $\ell$ steps makes $\lceil \ell/E \rceil$ queries at a nearly
  constant latency, so its inference time scales as $1/E$; timing whole episodes one at a time confirmed the
  product.

Try: `uv run evaluation.py -m runs=runs/default 'sampler.nfe=1,2,4,10' latency=true`
