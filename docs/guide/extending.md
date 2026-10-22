# Extending

Every layer is a small abstract class with a Hydra config next to it: a new probability path, flow-time distribution
or network is a subclass plus a YAML file, a prediction a subclass plus a `RECIPES` entry, and a solver a subclass
wired into the sampler. The code's own checks refuse what it cannot run. Each section below names the class, the
config, the checks you get for free, the test to copy and the page to update.

```bash
uv run pytest tests/test_flow.py -m "not cuda"     # the Gaussian toy checks every path, prediction and solver
uv run train.py experiment=smoke flow/path=gvp     # a short end-to-end run of any component, by its config name
```

## A probability path

Subclass {py:class}`~flow.ProbabilityPath` and give the noise schedulers $\alpha_t$, $\beta_t$ and their time
derivatives $\dot\alpha_t$, $\dot\beta_t$ as functions of a tensor of flow times; they define the conditional
probability path $p_t(x\mid z) = \mathcal N(x;\alpha_t z, \beta_t^2 I)$ ([the Gaussian probability
path](flow/index.md#the-gaussian-probability-path)). Override {py:meth}`~flow.ProbabilityPath.D` when
$\kappa_t = \dot\alpha_t\beta_t - \alpha_t\dot\beta_t$ (`D` in the code) has a closed form; the base class computes it
otherwise.

:::{admonition} Definition: what a path must satisfy
:class: important

A new path needs

1. noise schedulers in the sense of [the Gaussian probability path](flow/index.md#the-gaussian-probability-path):
   $\alpha_t$ and $\beta_t$ continuously differentiable, $\alpha_t$ increasing and $\beta_t$ decreasing (not checked
   at run time);
2. exact endpoints $\alpha_0 = \beta_1 = 0$ and $\alpha_1 = \beta_0 = 1$ (checked by the constructor, in float32);
3. $\kappa_t > 0$ on $[0, 1]$ (not checked at run time).
:::

In other words, the last condition asks that the path trade noise for signal at every time. Since
$\frac{\mathrm d}{\mathrm dt}\frac{\alpha_t}{\beta_t} = \frac{\kappa_t}{\beta_t^2}$ for $t < 1$, it says that the ratio
$\alpha_t/\beta_t$ rises at a positive rate at every $t < 1$, so the signal-to-noise ratio $\alpha_t^2/\beta_t^2$
increases strictly. A positive rate is more than increasing strictly. At the ends it reads $\dot\alpha_0 > 0$ and
$\dot\beta_1 < 0$ ([what $\kappa_t$ measures](flow/index.md#four-parameterizations)). For example, $\alpha_t = t^2$,
$\beta_t = 1 - t^2$ passes the endpoint check, and its ratio $t^2/(1 - t^2)$ increases strictly. But $\kappa_t = 2t$
vanishes at $t = 0$, where the SDE sampler evaluates its drift for a velocity network and gets $0/0$.

```python
class MyPath(ProbabilityPath):
    """
    alpha = ..., beta = ...: one line on what the path is.
    """

    def alpha(self, t: torch.Tensor) -> torch.Tensor: ...
    def beta(self, t: torch.Tensor) -> torch.Tensor: ...
    def d_alpha(self, t: torch.Tensor) -> torch.Tensor: ...
    def d_beta(self, t: torch.Tensor) -> torch.Tensor: ...

    def D(self, t: torch.Tensor) -> float:  # optional: a closed form, here a constant
        ...
```

```yaml
# conf/flow/path/my.yaml
# alpha = ..., beta = ...
_target_: flow.MyPath
```

- **Checks you get.** The constructor evaluates $\alpha_t$ and $\beta_t$ at $t = 0$ and $t = 1$ in float32 and
  refuses a path that does not run exactly from noise ($\alpha_0 = 0$, $\beta_0 = 1$) to data ($\alpha_1 = 1$,
  $\beta_1 = 0$): the samplers rely on exact zeros. {py:class}`~flow.GVPPath` writes $\beta_t$ as a sine for this
  reason, since $\cos(\pi/2)$ is not 0 in float32. A subclass with its own `__init__` must call
  `super().__init__()` last.
- **Your responsibilities.** The functions run on float32 tensors inside the captured step, and on 0-dimensional float64
  tensors in {py:meth}`~flow.ProbabilityPath.at`, where the samplers evaluate the path at grid times. The conditions of
  the box that the constructor does not check are yours. The maps from the vector field to the (scaled) score, and with
  them the SDE sampler's drift, divide by $\kappa_t$, so they blow up where it vanishes. The maps from a denoiser or a
  scaled score to the vector field multiply the output by $\kappa_t$, so where it vanishes the output no longer reaches
  the vector field. $\kappa_t$ must therefore not vanish on $[0, 1]$; it is 1 on the linear path and $\pi/2$ on GVP.
- **Test.** Add the path to `PATHS` (and its id) in `tests/test_flow.py`. `test_path_coefficients`,
  `test_conditional_velocity_is_the_path_derivative` and `test_ode_reaches_the_exact_endpoint` then check its
  endpoints, $\kappa_t$, its derivatives against finite differences and every sampler on the Gaussian toy.
  `test_path_coefficients` compares $\kappa_t$ with a constant (`torch.full_like(t, c.D)`); for a path whose
  $\kappa_t$ varies with $t$, compare with the tensor `c.D` itself.
- **Docs.** {doc}`flow/index` (paths and their findings: the path is a second-order choice) and the path list on
  {doc}`configuration`.

## A flow-time distribution

Subclass {py:class}`~flow.TimeSampler`, the distribution of $t$ in the loss (uniform in the notes'
$\mathcal L_{\text{CFM}}$), and give its inverse CDF. {py:meth}`~flow.TimeSampler.sample` applies
{py:meth}`~flow.TimeSampler.icdf` to one `torch.rand` draw per example, inside the captured step, so every
distribution uses the same random numbers and the draw needs neither clamping nor redrawing. A support
$[t_{\min}, t_{\max}]$ narrower than $[0, 1]$ (`flow.time.support`) needs an inverse CDF of the law restricted to it:
scaling the draw into the support is right for the uniform law only. [Drawing on a
support](flow/training.md#drawing-on-a-support) derives it.

```python
class MyTime(TimeSampler):
    def __init__(self, *, width: float, support: tuple[float, float]) -> None:
        self.width = float(width)
        super().__init__(support=support)

    # On float32 uniforms u in [0, 1), 0 included, inside the CUDA graph.
    def icdf(self, u: torch.Tensor) -> torch.Tensor: ...
```

```yaml
# conf/flow/time/my.yaml
# F^-1(u) = ...; one flow time per chunk
_target_: flow.MyTime
width: 1.0
support: [0.0, 1.0]
```

- **Checks you get.** The constructor refuses a support outside $0 \le t_{\min} < t_{\max} \le 1$. Nothing is
  clamped: {py:class}`~flow.FlowMatching` keeps a velocity-space loss away from the prediction's singular time by
  refusing a support that contains it or ends within $2^{-23}$ of it.
- **Your responsibilities.** `icdf` takes and returns float32 tensors, maps $u = 0$ to a finite time (a draw can be
  exactly 0), stays in the support and must not branch on tensor values. Refuse parameters you cannot draw exactly in
  the constructor, as {py:class}`~flow.LogitNormalTime` refuses a support other than $[0, 1]$.
- **Test.** Add a case with its exact CDF to the parameters of `test_draws_follow_the_law` (a Kolmogorov-Smirnov
  test) and to `TIME_SAMPLERS` (one uniform per example, finite at the extreme uniforms), and a refused case to
  `test_time_sampler_validation`; on a GPU, extend `test_loss_is_graph_capturable`.
- **Docs.** {doc}`flow/training` (flow-time distributions: moving the training weight toward the noise hurt).

## A prediction and its recipe

A {py:class}`~flow.Prediction` says what the network's raw output $o_t^\theta(x \mid y)$ is: the guided vector field
$u_t^\theta(x\mid y)$, the denoiser $D_t^\theta(x\mid y)$ or the scaled score
$\beta_t s_t^\theta(x\mid y) = -\epsilon_t^\theta(x\mid y)$, a noise predictor up to sign.
{py:meth}`~flow.Prediction.target` is its conditional regression target $o_t^{\text{target}}$, a function of the chunk
$z$ and the noise $\epsilon$. {py:meth}`~flow.Prediction.to_velocity` maps an output to the vector field
$u_t^\theta(x\mid y)$ the samplers integrate, and {py:attr}`~flow.Prediction.singular_time` is the flow time where that
map divides by zero. A {py:class}`~flow.Recipe` pairs a prediction with the loss space; register it in `RECIPES` in
`flow.py`, whose keys are the values of `flow.recipe`.

What makes a new prediction correct? Two conditions: one ties the target to the path, the other lets the conversion
pass through the averaging that regression does.

:::{admonition} Proposition: the prediction contract
:class: important

Suppose that at every $t$ where `to_velocity` is defined

1.  the target recovers the conditional vector field: for all $z$ and $\epsilon$, with
   $x = \alpha_t z + \beta_t\epsilon$, `to_velocity(target(z, eps), x)`
   $= \dot\alpha_t z + \dot\beta_t\epsilon = u_t^{\text{target}}(x \mid z)$;
2. `to_velocity` is affine in the output: `to_velocity(o, x)` $= A_t x + B_t\, o$, with $A_t$ and $B_t$ depending
   only on $t$.

Then, for an expressive enough network and at every $t$ that the flow-time distribution draws, the minimizer of the
loss converts to the guided marginal vector field $u_t^{\text{target}}(x \mid y)$.
:::

In other words, condition 1 makes the target right sample by sample, and condition 2 makes the conversion of the
averaged target equal the average of the converted targets.

The best network outputs the posterior mean of its target, $o^* = \mathbb E[o_t^{\text{target}} \mid x, y]$
([Regression learns a posterior mean](flow/training.md#regression-learns-a-posterior-mean); a velocity-space loss has
the same minimizer, see [Recipes](flow/training.md#recipes)). By condition 2, `to_velocity(o*, x)` is
$A_t x + B_t\, o^*$, and

$$
A_t x + B_t\, o^*
\overset{(i)}{=} \mathbb E\big[A_t x + B_t\, o_t^{\text{target}} \,\big|\, x, y\big]
\overset{(ii)}{=} \mathbb E\big[u_t^{\text{target}}(x \mid z) \,\big|\, x, y\big]
\overset{(iii)}{=} u_t^{\text{target}}(x \mid y),
$$

where (i) moves the known $x$ and the coefficients $A_t$, $B_t$ inside the posterior mean, (ii) uses both conditions,
and (iii) is [the marginal vector field](flow/index.md#the-marginal-vector-field) as a posterior mean, at the
observation $y$. In (ii), condition 2 makes $A_t x + B_t\, o_t^{\text{target}}$ equal to
`to_velocity(target(z, eps), x)`, and condition 1 equates that with $u_t^{\text{target}}(x \mid z)$. A nonlinear
`to_velocity` would break this chain silently: it would convert the averaged target, not average the converted ones.
Condition 2 also lets classifier-free guidance, which mixes outputs, mix vector fields
([Guidance](flow/sampling.md#guidance)). The three shipped predictions meet both conditions with the coefficients of
[the table of parameterizations](flow/index.md#four-parameterizations). The code checks neither condition for a new
one; the endpoint test on the Gaussian toy (below) is where a violation shows.

The example is a noise predictor that outputs $\epsilon_t^\theta$ itself. Its target is $\epsilon$, so its loss is
$\mathcal L_{\text{DDPM}}$, as for `scaled_score_mse`, whose output is $-\epsilon_t^\theta$. Its conversion is the
scaled-score column of [the table](flow/index.md#four-parameterizations) applied to $-\epsilon_t^\theta$, so it divides
by $\alpha_0 = 0$ at $t = 0$. It meets the contract because `scaled_score_mse` does and negating the output is linear:
the two recipes differ only in the sign of the output.

```python
class NoisePrediction(Prediction):
    """
    A noise predictor: the network predicts eps; undefined as a velocity at t = 0.
    """

    singular_time = 0.0

    def target(self, z, eps, c: Coefficients) -> torch.Tensor:
        return eps

    def to_velocity(self, y, x, c: Coefficients) -> torch.Tensor:
        # y is the network output, here eps (not the guiding variable); the scaled score is -y.
        return c.scaled_score_to_velocity(-y, x)


RECIPES = {
    ...,
    "noise_mse": Recipe(NoisePrediction(), velocity_loss=False),
}
```

- **Checks you get.** Nothing else needs to know about the singular time. {py:meth}`~flow.Sampler.times` refuses a grid
  that would evaluate the prediction there (Heun's method on a denoiser, a scaled score from $t = 0$), and a
  prediction singular at $t = 0$ starts sampling at `SCALED_SCORE_START` by default. {py:class}`~flow.FlowMatching`
  refuses `velocity_loss=True` with a support that contains the singular time.
- **Your responsibilities.** `target` (or `to_velocity`, for a velocity-space loss) runs inside the captured step, and
  `to_velocity` runs at every sampler step. The {py:class}`~flow.Coefficients` they receive ($\alpha_t$, $\beta_t$,
  $\dot\alpha_t$, $\dot\beta_t$ and $\kappa_t$ at the flow time) are tensors that broadcast against the chunk in the
  loss, and floats at a sampler's grid time.
- **Test.** `test_loss_targets` and `test_loss_is_graph_capturable` iterate over `RECIPES` and pick the recipe up by
  themselves. Add the prediction to `PREDICTIONS` and to `exact_predict` (the Bayes-optimal network of the toy, which
  otherwise answers the scaled score $-\mathbb E[\epsilon \mid x]$) in `tests/test_flow.py`, so that
  `test_ode_reaches_the_exact_endpoint` and `test_sde_keeps_the_marginals` cover it; a new conversion on
  {py:class}`~flow.Coefficients` belongs in `test_adapters_on_the_gaussian_posterior`.
- **Docs.** {doc}`flow/index` (parameterizations), {doc}`flow/training` (recipes: on the CondOT path, where the loss
  puts its weight mattered more than what the network outputs) and {doc}`flow/sampling` (start times).

## A solver

A {py:class}`~flow.Simulator` is a simulation method (the Euler method, Heun's method, the Euler-Maruyama method,
MeanFlow's step): {py:meth}`~flow.Simulator.step` takes one step from $t_i$ to $t_{i+1}$, of step size
$h_i = t_{i+1} - t_i$, {py:attr}`~flow.Simulator.nfe_per_step` declares its network evaluations per step, and
{py:meth}`~flow.Simulator.simulate` loops over the grid.

```python
class Midpoint(Simulator):
    """
    Explicit midpoint: two network evaluations per step, at t and at the middle of the step.
    """

    nfe_per_step = 2

    def __init__(self, ode: ODE) -> None:
        self.ode = ode

    def step(self, x, t, t_next, noise):
        dt = t_next - t
        return x + dt * self.ode.drift(
            x + 0.5 * dt * self.ode.drift(x, t), t + 0.5 * dt
        )
```

Then wire it into {py:class}`~flow.Sampler`: add the name to {py:attr}`~flow.Sampler.solver` and to the checks of its
`__post_init__` (the even-NFE rule of Heun's method is the model), the number of steps and the evaluated grid points
in {py:meth}`~flow.Sampler.times`, and the dispatch in {py:meth}`~flow.Sampler.sample`. List it in the `solver`
comment of `conf/sampler/default.yaml`.

- **Checks you get.** {py:meth}`~flow.Sampler.sample` counts the network evaluations of every query and raises when
  they differ from `sampler.nfe`, which catches a step count in {py:meth}`~flow.Sampler.times` that does not match the
  evaluations the solver makes per step; `nfe_per_step` itself is declarative, and nothing reads it. It also raises on
  non-finite samples. {py:meth}`~flow.Sampler.times` refuses a grid that is not increasing in float32.
- **Your responsibilities.** Tell {py:meth}`~flow.Sampler.times` which grid points the solver evaluates (the Euler
  method the left end of each step, Heun's method both ends), so that the singular-time check sees them. Sampling runs
  eagerly in evaluation, so host syncs are allowed here.
- **Test.** Add the solver to the parameters of `test_ode_reaches_the_exact_endpoint` and to
  `test_convergence_orders` (its order on the toy), its grid to `test_time_grids` and its refused settings to
  `test_unsupported_samplers_raise`.
- **Docs.** {doc}`flow/sampling` (solvers: Heun's method lost to the Euler method at equal NFE, so a new solver needs
  a comparison at equal NFE on the same policies and noise).

## A network

Subclass {py:class}`~model.FlowMatchingPolicy`. The base class owns the contract
({py:meth}`~model.FlowMatchingPolicy.compute_loss` for training, {py:meth}`~model.FlowMatchingPolicy.sample_actions`
for evaluation), condition dropout and classifier-free guidance; the subclass supplies the network, which maps $x$,
$t$ and the guiding variable $y$ to the recipe's output.

```python
class MyPolicy(FlowMatchingPolicy):
    def __init__(
        self,
        obs_dim,
        goal_dim,
        action_dim,
        obs_horizon,
        pred_horizon,
        *,
        flow,
        cond_dropout,
        width,
    ):
        super().__init__(
            obs_dim,
            goal_dim,
            action_dim,
            obs_horizon,
            pred_horizon,
            flow=flow,
            cond_dropout=cond_dropout,
        )
        # The layers; with self.mean_flow, MeanFlow's gap input last, zero-initialized.
        ...

    # (bs, K, obs_dim) and (bs, goal_dim), the guiding variable y -> its encoding, (bs, ...).
    def encode_condition(self, obs, goal): ...

    # The learned encoding of the null label, "no observation", shaped like the encoding of y.
    def null_condition(self, batch_size): ...

    # The nn.Linear layers that read the inputs or write the prediction.
    def boundary_layers(self): ...

    def _predict(self, condition, action_chunk, flow_time, end_time=None):
        # None for flow matching; MeanFlow's gap s - t, formed here.
        gap = self.gap(flow_time, end_time)
        ...
        # (bs, H, d), in float32.
        return at_least_float32(out)
```

```yaml
# conf/model/my.yaml
# One line on the network (model.MyPolicy).
_target_: model.MyPolicy
width: 256
cond_dropout: 0.0        # check_config reads it: guidance needs a policy trained with dropout
```

- **Checks you get.** {py:meth}`~model.FlowMatchingPolicy.compute_loss` checks the chunk's shape, and
  {py:meth}`~model.FlowMatchingPolicy.gap` refuses an end time for a flow matching policy.
  {py:meth}`~model.FlowMatchingPolicy.parameter_groups` sorts the parameters for the optimizers: the weights of the
  layers `boundary_layers` names are boundary matrices, every other `nn.Linear` weight is a hidden matrix (Muon's), and
  every remaining parameter is a vector, which never decays. A matrix that is not an `nn.Linear` weight (an embedding
  table, a convolution kernel) therefore counts as a vector.
- **Your responsibilities.** Form the gap $s - t$ inside `_predict` with {py:meth}`~model.FlowMatchingPolicy.gap`:
  MeanFlow differentiates $\bar u^\theta_{t,s}(x\mid y)$ in $t$ with $s$ fixed, and a gap computed outside misses the
  derivative through it ([Computing the derivative](flow/meanflow.md#computing-the-derivative)). Start MeanFlow's gap
  input at zero, so the average velocity $\bar u^\theta_{t,s}$ starts as the instantaneous one, $\bar u^\theta_{t,t}$;
  the DiT builds its gap embedding last, so the shared weights draw the same initialization as flow matching's. The DiT
  also zero-initializes its output head (adaLN-Zero). The forward pass runs inside the captured step, under
  `torch.compile(fullgraph=True)` and, for MeanFlow, inside `torch.func.jvp` on the math attention kernel.
- **Test.** Add a parameter set with a small size to `test_captured_step_trains` (its size overrides are the DiT's and
  the MLP's), including one with `train.compile=true`, and the network to
  `test_parameter_groups_split_the_parameters_by_role`, `test_jvp_of_the_policy_matches_finite_differences`,
  `test_end_time_none_is_the_zero_gap`, `test_the_network_sees_the_gap` and
  `test_meanflow_policy_starts_at_the_instantaneous_velocity`.
- **Docs.** {doc}`model` (the contract, conditioning and the DiT against the MLP).

## An experiment preset

A preset is a YAML file in `conf/experiment/` that sets the fixed settings of a comparison and its grid. Its comment
says what it compares and how to launch it, including any second launch: a sweep is a product, so coupled settings
(a schedule and its warm-up) are a launch of their own.

```yaml
# @package _global_
# What the grid compares, on what, and how to launch it:
#   uv run train.py -m experiment=my_study
name: my_study           # runs/my_study/... and the W&B group
train:
  optimizer: adamw       # pin what the results are measured with, in case the default changes later
hydra:
  sweeper:
    params:
      flow/path: linear,gvp
      seed: range(900,903)
```

- **Checks you get.** `test_presets_pass_config_checks` finds every file in `conf/experiment/`, expands its
  `hydra.sweeper.params` grid and runs {py:func}`~train.check_config` on every job. The other commands in a preset's
  comment are not checked.
- **Your responsibilities.** Pin every setting the comparison depends on when its default may move: the `conditioning`,
  `history`, `horizons`, `optimizer`, `flow_matching` and `meanflow` presets pin what they were measured with, while
  `training_loop` trains AdamW at weight decay 0.01 instead of its study's Adam ({doc}`configuration`).
- **Docs.** The `Reproduce:` line under the finding the preset reproduces, and the preset list on {doc}`configuration`.

## A config check

A combination that cannot run should fail before training. Put the refusal where its inputs live:

| what it checks | where | when it runs |
| --- | --- | --- |
| settings across config sections (optimizer and schedule, precision and objective, guidance and dropout) | {py:func}`~train.check_config` | first in {py:func}`~train.run_training`, before data and GPU work |
| one sampler's settings | `Sampler.__post_init__` | whenever a {py:class}`~flow.Sampler` is built, in training and in `evaluation.py` |
| a sampler against a trained objective | {py:meth}`~flow.Sampler.check`, {py:meth}`~flow.Sampler.times` | in {py:func}`~train.check_config`, and in `evaluation.py`, which skips a run the sampler does not fit |
| one component's own settings | its constructor | at instantiation; Hydra wraps the error in `InstantiationException` |

{py:func}`~train.check_config` instantiates the flow objective and the sampler, so their constructors' checks also run
before any data. The policy is built only after the data, so a refusal that needs no data belongs in
{py:func}`~train.check_config` rather than in the network's constructor. Say how to fix the setting in the message.
Add a `pytest.raises(ValueError)` case to `test_unsupported_combinations_fail_before_training` (or to
`test_unsupported_samplers_raise` and `test_meanflow_validation` for flow-level checks).

## Code inside the captured step

Probability paths ({py:meth}`~flow.ProbabilityPath.alpha` and the rest on tensors), {py:meth}`~flow.TimeSampler.icdf`,
recipes and the network's forward and backward passes all run inside the captured training step. The rules are on
{doc}`training`: no host syncs and no control flow on tensor values, `torch.where` instead of masked assignment,
captured tensors kept alive, no Python number that must change between steps, no graph break under the compiled loss.
`test_loss_is_graph_capturable` shows how to capture a loss under `torch.cuda.set_sync_debug_mode("error")`, which
turns any host sync into an error.
