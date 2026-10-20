# Notation

The guide writes its math in the notation of one set of lecture notes and keeps the code's names apart. This page
lists every symbol once, maps it to its code name and config key, defines each term in a sentence and translates the
notation to two other references. Three texts are cited throughout:

- Peter Holderrieth and Ezra Erives, *An Introduction to Flow Matching and Diffusion Models*, MIT 6.S184 lecture
  notes, on the course website <https://diffusion.csail.mit.edu/>. The pages call them **the notes** and use their
  notation. They build flow and diffusion models, flow matching, score matching, guidance and the network
  architectures of image generators.
- Yaron Lipman, Marton Havasi, Peter Holderrieth, Neta Shaul, Matt Le, Brian Karrer, Ricky T. Q. Chen, David
  Lopez-Paz, Heli Ben-Hamu and Itai Gat, *Flow Matching Guide and Code*, 2024,
  [arXiv:2412.06264](https://arxiv.org/abs/2412.06264). The pages call it **the FM Guide**. Its organizing idea, that
  every network output is a posterior mean of a known function of the noise and the data, shapes the flow pages.
- Chieh-Hsin Lai, Yang Song, Dongjun Kim, Yuki Mitsufuji and Stefano Ermon, *The Principles of Diffusion Models*,
  2025, [arXiv:2510.21890](https://arxiv.org/abs/2510.21890). The pages call it **the Diffusion Guide**. Its
  chapter 4, on the score SDE, supplies the view of one probability path realized by many processes.

Math symbols follow the notes, and code identifiers keep their own names. The docstrings of `flow.py` and `model.py`
write their formulas in the same symbols, and [a table below](#code-names-and-config-keys) maps each symbol to the
code.

## Time conventions

Which end of time holds the noise? The references disagree, and reversing the clock flips the sign of every velocity.
We follow the notes and the FM Guide.

:::{admonition} Definition: the flow time and two other clocks
:class: important

The **flow time** $t \in [0, 1]$ runs from noise at $t = 0$ to data at $t = 1$, and samplers simulate upward in $t$.
The references use two other clocks:

| clock | noise at | data at | used by | relation to the flow time |
| --- | --- | --- | --- | --- |
| flow time $t$ | $t = 0$ | $t = 1$ | these pages and the code, the notes, the FM Guide | itself |
| diffusion time $\tau$ | $\tau = T$ | $\tau = 0$ | the Diffusion Guide (as $t$), score SDEs and DDPM | $\tau = 1 - t$ on our paths, with $T = 1$ |
| the MeanFlow papers' time | $t = 1$ | $t = 0$ | MeanFlow, improved MeanFlow and rectified flows such as SD3 | their pair $(t, r)$ is our $(1 - t, 1 - s)$ |
:::

In other words, a quantity that points toward the data in one clock points toward the noise in the other. The FM
Guide writes diffusion time as $r$, with noise at $r = \infty$, and maps the flow time to it by a decreasing
$r = k(t)$. Two consequences recur in the guide. The MeanFlow papers' average velocity $u$ is our $-\bar u_{t,s}$,
and their logit-normal $\mathrm{LN}(\mu, \varsigma)$ is our $\mathrm{LN}(-\mu, \varsigma)$
([The papers' convention](flow/meanflow.md#the-papers-convention)). The Diffusion Guide's forward SDE has our path as
its perturbation kernel, read in reversed time ([Four parameterizations](flow/index.md#four-parameterizations)).

## Symbols

Each table collects the symbols of one stage, in the order the flow pages introduce them. A meaning is one clause;
the linked section defines the symbol and states the domain of its identities. A symbol not listed here is local to
its page and defined there.

**Data and guidance.**

| symbol | meaning | defined on |
| --- | --- | --- |
| $t$ | the flow time, in $[0, 1]$ | [Conventions](flow/index.md#conventions) |
| $H$, $K$, $E$ | the steps of a chunk (16), the history frames (2), the steps executed per query (4) | [The method in four steps](overview.md#the-method-in-four-steps) |
| $z$, $p_{\mathrm{data}}$ | a clean, normalized chunk $z \in \mathbb R^{H \times 2}$, and the distribution of the expert chunks | [Conventions](flow/index.md#conventions) |
| $\epsilon$, $p_{\mathrm{init}}$ | the noise $\epsilon \sim p_{\mathrm{init}} = \mathcal N(0, I)$, of a chunk's shape and independent of $z$ | [Conventions](flow/index.md#conventions) |
| $y$ | the guiding variable: the normalized goal and the $K$ history frames | [Conventions](flow/index.md#conventions) |
| $p_{\mathrm{data}}(\cdot \mid y)$ | the guided data distribution, which the policy samples | [Conventions](flow/index.md#conventions) |
| $\varnothing$ | the null label, "no observation", of classifier-free guidance | [Guidance](flow/sampling.md#guidance) |

**Guided at a fixed $y$.** The policy samples $p_{\mathrm{data}}(\cdot \mid y)$, while the path and the regression
targets do not depend on $y$. The flow pages therefore fix $y$ and drop it from the marginal quantities:
$p_t(x)$, $u_t^{\text{target}}(x)$, $D_t(x)$ and $\nabla \log p_t(x)$ are the guided ones. The notes write the
unguided ones without $y$; here they carry the null label, as in $u_t^{\text{target}}(x \mid \varnothing)$. Where
guided and unguided quantities meet, in the loss and in guidance, $y$ is written back, as in
$u_t^{\text{target}}(x \mid y)$. The network always keeps its argument, as in $u_t^\theta(x \mid y)$
([Conventions](flow/index.md#conventions)).

**The probability path.**

| symbol | meaning | defined on |
| --- | --- | --- |
| $\alpha_t$, $\beta_t$ | the noise schedulers, with $\alpha_0 = \beta_1 = 0$ and $\alpha_1 = \beta_0 = 1$ | [The Gaussian probability path](flow/index.md#the-gaussian-probability-path) |
| $\dot\alpha_t$, $\dot\beta_t$ | their derivatives in $t$ | [The Gaussian probability path](flow/index.md#the-gaussian-probability-path) |
| $p_t(x \mid z)$ | the conditional probability path $\mathcal N(x;\, \alpha_t z,\, \beta_t^2 I)$, from $p_{\mathrm{init}}$ to the one chunk $z$ | [The Gaussian probability path](flow/index.md#the-gaussian-probability-path) |
| $x$ | a noisy chunk, $x = \alpha_t z + \beta_t \epsilon$ | [The Gaussian probability path](flow/index.md#the-gaussian-probability-path) |
| $p_t(x)$ | the marginal probability path, from $p_0 = p_{\mathrm{init}}$ to $p_1 = p_{\mathrm{data}}$ | [The Gaussian probability path](flow/index.md#the-gaussian-probability-path) |
| $\psi_t(\epsilon \mid z)$ | the conditional flow $\alpha_t z + \beta_t \epsilon$, which moves $\epsilon$ to $z$ | [The conditional vector field](flow/index.md#the-conditional-vector-field) |
| $\alpha_t^2/\beta_t^2$ | the signal-to-noise ratio (SNR); $2\log(\alpha_t/\beta_t)$ is the log-SNR | [Paths](flow/index.md#paths) |

**Vector fields and network outputs.**

| symbol | meaning | defined on |
| --- | --- | --- |
| $u_t^{\text{target}}(x \mid z)$ | the conditional vector field $\dot\alpha_t z + \dot\beta_t \epsilon$ | [The conditional vector field](flow/index.md#the-conditional-vector-field) |
| $u_t^{\text{target}}(x)$ | the marginal vector field: the posterior mean of $u_t^{\text{target}}(x \mid z)$ given $x$, guided at a fixed $y$ | [The marginal vector field](flow/index.md#the-marginal-vector-field) |
| $u_t^{\text{target}}(x \mid \varnothing)$ | the unguided marginal vector field | [Conventions](flow/index.md#conventions) |
| $u_t^{\text{target}}(x \mid y)$ | the guided marginal vector field with $y$ written out, where it meets the unguided one | [Regression learns a posterior mean](flow/training.md#regression-learns-a-posterior-mean) |
| $D_t(x)$ | the denoiser $\mathbb E[z \mid x]$, the expected clean chunk | [The marginal vector field](flow/index.md#the-marginal-vector-field) |
| $\mathbb E[\epsilon \mid x]$ | the expected noise; with $D_t$ it satisfies the path identity $x = \alpha_t D_t(x) + \beta_t\,\mathbb E[\epsilon \mid x]$ | [The marginal vector field](flow/index.md#the-marginal-vector-field) |
| $m$, $s$, $v_t$ | the mean and the standard deviation of every coordinate in the Gaussian example, $z \sim \mathcal N(m, s^2)$, and the variance $v_t = \alpha_t^2 s^2 + \beta_t^2$ of a coordinate of $x$ | [An example with Gaussian data](flow/index.md#an-example-with-gaussian-data) |
| $\nabla \log p_t(x \mid z)$, $\nabla \log p_t(x)$ | the conditional and the marginal score | [Scores](flow/index.md#scores) |
| $\beta_t \nabla \log p_t(x)$ | the scaled score, equal to $-\mathbb E[\epsilon \mid x]$ | [Scores](flow/index.md#scores) |
| $s_t^\theta(x \mid y)$, $\epsilon_t^\theta(x \mid y)$ | the notes' score network and noise predictor, with $\epsilon_t^\theta = -\beta_t s_t^\theta$ | [Scores](flow/index.md#scores) |
| $o_t^\theta(x \mid y)$ | the network's raw output: a velocity, a denoiser $D_t^\theta$ or a scaled score $\beta_t s_t^\theta$, as the recipe asks | {doc}`model` |
| $o_t^{\text{target}}$ | the output's conditional regression target: $u_t^{\text{target}}(x \mid z)$, $z$ or $-\epsilon$ | [One training step](flow/training.md#one-training-step) |
| $\kappa_t$ | $\dot\alpha_t\beta_t - \alpha_t\dot\beta_t$, required positive; the factor of every conversion to or from the vector field | [Four parameterizations](flow/index.md#four-parameterizations) |
| $A_t$, $B_t$ | the coefficients of a conversion $u = A_t\, x + B_t\, o$, independent of $y$; $B_t$ is its gain | [Four parameterizations](flow/index.md#four-parameterizations) |
| $a_t$, $b_t$ | the coefficients of the notes' conversion formula $u_t^{\text{target}}(x) = a_t \nabla \log p_t(x) + b_t x$ | [Four parameterizations](flow/index.md#four-parameterizations) |
| $u_t^\theta(x \mid y)$ | the network's vector field: $o_t^\theta$ converted, for every recipe; the sampler integrates it | [One training step](flow/training.md#one-training-step) |

**Losses.**

| symbol | meaning | defined on |
| --- | --- | --- |
| $\mathcal L_{\text{FM}}$, $\mathcal L_{\text{CFM}}$ | the regressions of a vector-field output onto $u_t^{\text{target}}(x \mid y)$ and onto $u_t^{\text{target}}(x \mid z)$; they differ by a constant independent of $\theta$ | [Regression learns a posterior mean](flow/training.md#regression-learns-a-posterior-mean) |
| $\mathcal L_{\text{CSM}}$, $\mathcal L_{\text{DDPM}}$ | denoising score matching $\lVert s_t^\theta - \nabla \log p_t(x \mid z)\rVert^2$, and DDPM's $\lVert \epsilon_t^\theta - \epsilon\rVert^2$, which is $\beta_t^2$ times it | [Recipes](flow/training.md#recipes) |
| $p_{\mathrm{time}}$ | the density of the flow-time distribution that training draws $t$ from | [One training step](flow/training.md#one-training-step) |
| $\mu$, $\varsigma$ | the mean and the standard deviation of $\mathrm{logit}\,t$ under the logit-normal law $\mathrm{LN}(\mu, \varsigma)$ | [Flow-time distributions](flow/training.md#flow-time-distributions) |
| $m_k$ | the loss weight of step $k$ of a chunk: 0 on padded steps under `data.padding=mask`, 1 otherwise | [Reduction](flow/training.md#reduction) |
| $e$, $\ell(e)$ | a chunk's error, in the output's own space or in velocity space, and its reduction to one number | [One training step](flow/training.md#one-training-step) |
| $\lambda_t$ | a recipe's time weight: the factor on the velocity error that its loss amounts to | [Recipes](flow/training.md#recipes) |
| $t_{\min}$, $t_{\max}$ | the ends of the time support $[t_{\min}, t_{\max}] \subseteq [0, 1]$ that training draws from | [Drawing on a support](flow/training.md#drawing-on-a-support) |

**Sampling.**

| symbol | meaning | defined on |
| --- | --- | --- |
| $X_t$ | the sampled trajectory, from $X_{t_0} \sim \mathcal N(0, I)$ to the chunk $X_1$; for $t_0 > 0$ the start only approximates $p_{t_0}$ | [The flow model](flow/sampling.md#the-flow-model) |
| $\psi_{t \to s}$, $\psi^\theta_{t \to s}$ | the flow maps of the exact and of the learned ODE, from time $t$ to $s \ge t$; the notes' flow is $\psi_t = \psi_{0 \to t}$ | [The flow model](flow/sampling.md#the-flow-model) |
| $t_0$ | the start time: by default 0, or 0.05 for a scaled-score output | [Where sampling starts](flow/sampling.md#where-sampling-starts) |
| $t_0 < t_1 < \dots < t_n = 1$, $h_i$ | the time grid and its step sizes $h_i = t_{i+1} - t_i$ | [Time grids](flow/sampling.md#time-grids) |
| $n$, NFE | the number of steps, and the network evaluations per chunk | [Time grids](flow/sampling.md#time-grids) |
| $\rho$ | the exponent of the `data_dense` grid | [Time grids](flow/sampling.md#time-grids) |
| $\sigma_t$, $W_t$ | the diffusion coefficient and the Brownian motion of the SDE | [The SDE](flow/sampling.md#the-sde) |
| $\eta$ | the SDE's noise scale, in $\sigma_t^2 = 2\eta\,\alpha_t\beta_t$ | [The SDE](flow/sampling.md#the-sde) |
| $\xi_i$ | the Brownian draws $\xi_i \sim \mathcal N(0, I)$ of step $i$, independent of $X_{t_0}$ | [The SDE](flow/sampling.md#the-sde) |

**Guidance.**

| symbol | meaning | defined on |
| --- | --- | --- |
| $w$ | the guidance scale; $w = 1$ gives the guided field $u_t^\theta(x \mid y)$ alone, without classifier-free guidance, and $w = 0$ the unguided $u_t^\theta(x \mid \varnothing)$ | [Guidance](flow/sampling.md#guidance) |
| $\tilde u_t(x \mid y)$, $\tilde u_t^\theta(x \mid y)$ | the classifier-free guided vector field $(1 - w)\,u_t^{\text{target}}(x \mid \varnothing) + w\,u_t^{\text{target}}(x \mid y)$, and the same mix of $u_t^\theta$ | [Guidance](flow/sampling.md#guidance) |
| $p_{\mathrm{uncond}}$ | the probability that training replaces $y$ by $\varnothing$ | [One training step](flow/training.md#one-training-step) |

**MeanFlow.**

| symbol | meaning | defined on |
| --- | --- | --- |
| $s$ | the end time of a step, $s \ge t$ | [Average velocity](flow/meanflow.md#average-velocity) |
| $h$ | the gap $s - t$ that the network sees | [Computing the derivative](flow/meanflow.md#computing-the-derivative) |
| $\bar u_{t,s}(x)$ | the average velocity $(\psi_{t \to s}(x) - x)/(s - t)$ for $s > t$; $\bar u_{t,t} = u_t^{\text{target}}$ as its limit $s \to t$ | [Average velocity](flow/meanflow.md#average-velocity) |
| $\bar u^\theta_{t,s}(x \mid y)$ | the network's average velocity | [Average velocity](flow/meanflow.md#average-velocity) |
| $\frac{\mathrm d}{\mathrm dt}\bar u_{t,s}$ | the derivative along the flow, with $s$ fixed | [The MeanFlow identity](flow/meanflow.md#the-meanflow-identity) |
| $\mathrm{sg}$ | the stop-gradient | [The target](flow/meanflow.md#the-target) |
| $t'$, $t''$ | the two draws from which a time pair $(t, s)$ is formed | [Options](flow/meanflow.md#options) |
| $\gamma_i$, $p$, $\delta$ | the adaptive weight $(\ell_i + \delta)^{-p}$ of chunk $i$, whose loss is $\ell_i$ | [Options](flow/meanflow.md#options) |

**The network.**

| symbol | meaning | defined on |
| --- | --- | --- |
| $\theta$ | the trainable parameters of the policy | {doc}`model` |
| $\bar\theta$ | the exponential moving average of $\theta$ | [Weight averages](training.md#weight-averages) |
| $\lambda$ | the decay of a moving average | [Weight averages](training.md#weight-averages) |
| $\tilde t$, $\tilde y$, $\tilde x_i$ | the time embedding, the tokens of $y$, and the action tokens after $i$ blocks | [The DiT](model.md#the-dit) |
| $c$ | the conditioning vector that every block reads | [The DiT](model.md#the-dit) |
| $\gamma_k$, $\beta_k$, $g_k$ | the scale, shift and gate of sub-layer $k \in \{\text{self}, \text{MLP}\}$ in adaptive normalization | [The DiT](model.md#the-dit) |
| $d$, $L$, $n_{\text{heads}}$ | the width, the depth in blocks and the attention heads of the DiT | [The DiT](model.md#the-dit) |

## Code names and config keys

The code keeps its own identifiers. This table maps each symbol to the names that hold it and to the config key that
sets it; a dash means there is none.

| symbol | code names | config key |
| --- | --- | --- |
| $t$, $p_{\mathrm{time}}$ | `t`, `flow_time`; {py:class}`~flow.UniformTime`, {py:class}`~flow.LogitNormalTime` draw it with {py:meth}`~flow.TimeSampler.sample` | `flow/time` |
| $\mu$, $\varsigma$ | `mean`, `std` of {py:class}`~flow.LogitNormalTime` | `flow.time.mean`, `flow.time.std` |
| $t_{\min}$, $t_{\max}$ | `support` of a {py:class}`~flow.TimeSampler` | `flow.time.support` |
| $H$, $K$, $E$ | `pred_horizon`, `obs_horizon`, `exec_horizon`; `H` and `K` in shapes | `data.pred_horizon`, `data.obs_horizon`, `eval.exec_horizon` |
| $z$ | `z`; `action_chunk` of {py:meth}`~model.FlowMatchingPolicy.compute_loss` and {py:meth}`~flow.FlowMatching.loss` | - |
| $\epsilon$ | `eps`, `noise` | - |
| $x$ | `x`, `interpolated_chunk`; `action_chunk` of {py:meth}`~model.FlowMatchingPolicy._predict` | - |
| $y$ | `obs` and `goal`, encoded into `condition` by {py:meth}`~model.FlowMatchingPolicy.encode_condition` | `data.obs_horizon`, `data.action_history`, `data.frame` |
| $\varnothing$ | {py:meth}`~model.FlowMatchingPolicy.null_condition` | - |
| $p_{\mathrm{uncond}}$ | `cond_dropout`, applied by {py:meth}`~model.FlowMatchingPolicy.condition` in training mode | `model.cond_dropout` |
| $\alpha_t$, $\beta_t$, $\dot\alpha_t$, $\dot\beta_t$, $\kappa_t$ | the fields `alpha`, `beta`, `d_alpha`, `d_beta` and `D` of {py:class}`~flow.Coefficients`; {py:meth}`~flow.ProbabilityPath.D` | `flow/path` |
| the linear path, GVP | {py:class}`~flow.LinearPath`, {py:class}`~flow.GVPPath` | `flow/path=linear`, `flow/path=gvp` |
| $x = \alpha_t z + \beta_t\epsilon$ | {py:meth}`~flow.Coefficients.interpolate` | - |
| $u_t^{\text{target}}(x \mid z)$ | {py:meth}`~flow.Coefficients.velocity` | - |
| $o_t^\theta(x \mid y)$ | the output of `predict`; the argument `y` of {py:meth}`~flow.Prediction.to_velocity` | `flow.recipe` |
| $o_t^{\text{target}}$ | {py:meth}`~flow.Prediction.target` | `flow.recipe` |
| a velocity, $D_t^\theta$, $\beta_t s_t^\theta$ as outputs | {py:class}`~flow.VelocityPrediction`, {py:class}`~flow.DataPrediction`, {py:class}`~flow.ScaledScorePrediction` | `flow.recipe` |
| $u = A_t x + B_t o$ | {py:meth}`~flow.Prediction.to_velocity`, through {py:meth}`~flow.Coefficients.data_to_velocity` and {py:meth}`~flow.Coefficients.scaled_score_to_velocity` | - |
| $\beta_t s_t^\theta$, $s_t^\theta$ from a vector field | {py:meth}`~flow.Coefficients.velocity_to_scaled_score`, {py:meth}`~flow.Coefficients.velocity_to_score` | - |
| $u_t^\theta(x \mid y)$ | {py:meth}`~flow.LearnedFlow.drift` | - |
| the four recipes | {py:class}`~flow.Recipe`, the keys of `RECIPES`; the flag `velocity_loss` marks a loss compared in velocity space | `flow.recipe` |
| $e$, $\ell(e)$ | `error`; {py:meth}`~flow.FlowMatching.per_example` reduces it | `flow.reduction` |
| $X_t$ | `x` of {py:meth}`~flow.Simulator.step` | - |
| $t_0$ | `start_time`; `SCALED_SCORE_START` for a scaled score | `sampler.start_time` |
| $t_0 < \dots < t_n$, $h_i$ | {py:meth}`~flow.Sampler.times`; `t_next - t`, `dt` | `sampler.grid`, `sampler.nfe`, `sampler.sde_end` |
| NFE | {py:attr}`~flow.Sampler.nfe` | `sampler.nfe` |
| $\rho$ | `rho` | `sampler.rho` |
| the solvers | {py:class}`~flow.Euler`, {py:class}`~flow.Heun`, {py:class}`~flow.EulerMaruyama`, {py:class}`~flow.MeanFlowStep` | `sampler.solver` |
| $\sigma_t$, $\eta$ | {py:meth}`~flow.LangevinFlow.diffusion`, `eta` | `sampler.sde_eta` |
| $\xi_i$ | `brownian` | - |
| $w$, $\tilde u_t^\theta$ | `cfg_scale`; {py:meth}`~model.FlowMatchingPolicy.predictor` mixes the outputs | `sampler.cfg_scale` |
| $s$, $h$ | `s`, `end_time`; {py:meth}`~model.FlowMatchingPolicy.gap` | - |
| $\bar u^\theta_{t,s}$ | {py:meth}`~flow.LearnedFlow.average`, {py:meth}`~flow.MeanFlow.average_velocity` | `flow=meanflow`, `sampler=mean` |
| $t'$, $t''$ | the draws of {py:meth}`~flow.MeanFlow.sample_times` | `flow.time`, `flow.second_time`, `flow.order`, `flow/pairs` |
| the tangent | `tangent` | `flow.tangent` |
| $p$, $\delta$ | `adaptive_power`, `adaptive_eps` | `flow.adaptive_power`, `flow.adaptive_eps` |
| $\theta$, $\bar\theta$, $\lambda$ | the policy's parameters; the averaged copies and their decays | `train.ema_decays`, `eval.weights` |
| $\tilde t$, $\tilde y$ | {py:class}`~model.FourierEncoder`; {py:meth}`~model.DiTPolicy.encode_condition` | `model.conditioning` |
| $\gamma_k$, $\beta_k$, $g_k$ | `scale`, `bias`, `gate`; {py:func}`~model.modulate`, {py:class}`~model.DiTBlock` | - |
| $d$, $L$, $n_{\text{heads}}$ | `dim` (`D` in model.py's docstrings), `depth`, `heads` | `model.dim`, `model.depth`, `model.heads` |

The API pages keep the code's shapes. A chunk is `(bs, H, d)`: `bs` the batch, `H` the steps and `d` = 2 the action
dimension, the agent's target position. A history frame has 6 features (8 with the action history), the goal 4.

## Glossary

A sentence or two per term, with a link to the section that owns it. The terms come in the order of the symbol
tables.

- **Conditional, guided, conditioning.** In the math, *conditional* means given one chunk $z$, as in
  $p_t(x \mid z)$, and *guided* means given the observation $y$, as in $p_{\mathrm{data}}(\cdot \mid y)$. The code
  names the encoded $y$ `condition`, and {doc}`model` says *conditioning* for how $y$ and $t$ enter the network
  ([Conventions](flow/index.md#conventions)).
- **Marginal.** Averaged over the chunk $z$, by its posterior given $x$ for a vector field or a score, at a fixed $y$
  ([The marginal vector field](flow/index.md#the-marginal-vector-field)).
- **Flow time.** The time $t$ of the probability path, named to tell it from the steps of an episode
  ([Time conventions](#time-conventions)).
- **Chunk.** The data point of the generative model: $H$ relative, normalized action targets,
  $z \in \mathbb R^{H \times 2}$ ({doc}`data`).
- **Guiding variable.** The observation $y$ that the policy samples for: the goal and the last $K$ states, in the
  current state's frame and normalized ([Conventions](flow/index.md#conventions)).
-  **Probability path.** Distributions $p_t$ that run from $p_{\mathrm{init}}$ at $t = 0$ to $p_{\mathrm{data}}$ at
  $t = 1$; its conditional form $p_t(x \mid z)$ ends at the one chunk $z$ ([The Gaussian probability
  path](flow/index.md#the-gaussian-probability-path)).
- **Noise schedulers.** $\alpha_t$ and $\beta_t$, the signal and noise coefficients of
  $x = \alpha_t z + \beta_t \epsilon$ ([The Gaussian probability path](flow/index.md#the-gaussian-probability-path)).
- **Linear path, CondOT path, GVP.** The shipped paths: the linear one, $\alpha_t = t$ and $\beta_t = 1 - t$, which the
  notes call CondOT, and the generalized variance-preserving one, $\alpha_t = \sin\frac{\pi t}{2}$ and
  $\beta_t = \cos\frac{\pi t}{2}$, the FM Guide's cosine scheduler ([Paths](flow/index.md#paths)).
- **Equivalence of schedulers.** With exact vector fields the two paths carry the same noise to the same chunk,
  because they differ only by a clock and a scale ([Paths](flow/index.md#paths)).
- **Conditional flow.** $\psi_t(\epsilon \mid z) = \alpha_t z + \beta_t \epsilon$, the curve along which one noise
  sample moves to its chunk ([The conditional vector field](flow/index.md#the-conditional-vector-field)).
-  **Vector field, velocity.** The math says vector field; code names, velocity-space losses and MeanFlow's average
  velocity say velocity, as the papers do ([The conditional vector field](flow/index.md#the-conditional-vector-field)).
-  **Marginalization trick.** Averaging the conditional vector fields over the posterior of $z$ given $x$ gives a field
  whose ODE carries $p_{\mathrm{init}}$ to $p_{\mathrm{data}}$ ([The marginal vector
  field](flow/index.md#the-marginal-vector-field)).
-  **Posterior mean.** $\mathbb E[\,\cdot \mid x]$ at fixed $t$ and $y$, the average over every pair $(z, \epsilon)$
  that could have produced $x$; each flow matching output estimates one ([The marginal vector
  field](flow/index.md#the-marginal-vector-field)).
- **Denoiser.** $D_t(x) = \mathbb E[z \mid x]$, the expected clean chunk given the noisy one
  ([The marginal vector field](flow/index.md#the-marginal-vector-field)).
- **Path identity.** $x = \alpha_t D_t(x) + \beta_t\,\mathbb E[\epsilon \mid x]$: a noisy chunk splits into its
  expected signal and its expected noise ([The marginal vector field](flow/index.md#the-marginal-vector-field)).
-  **Regression policy.** $D_0 = \mathbb E[z]$ for the observation $y$, the mean chunk that a policy trained with a
  squared error outputs, which can lie between two modes ([The marginal vector
  field](flow/index.md#the-marginal-vector-field)).
- **Noisy chunk and trajectory.** Training builds the noisy chunk $x$ from $z$ and an independent $\epsilon$. The
  exact marginal ODE, started from $X_0 \sim p_{\mathrm{init}}$, instead pairs $X_0$ with its endpoint
  deterministically. Both have the law $p_t$ at every $t$, but their joint laws over time differ. The sampled
  trajectory $X_t$ approximates the second ([The marginal vector field](flow/index.md#the-marginal-vector-field)).
- **Score, Tweedie's formula.** The score $\nabla \log p_t(x)$ points where the log density rises fastest, and
  Tweedie's formula writes it through a posterior mean, as $-\mathbb E[\epsilon \mid x]/\beta_t$ for $t < 1$
  ([Scores](flow/index.md#scores)).
- **Scaled score.** $\beta_t \nabla \log p_t(x) = -\mathbb E[\epsilon \mid x]$, the default network output, which
  `scaled_score_mse` regresses onto $-\epsilon$ ([Scores](flow/index.md#scores)).
- **Noise predictor.** The notes' $\epsilon_t^\theta$, an estimate of $\mathbb E[\epsilon \mid x]$; a scaled-score
  network is a noise predictor up to sign, and the findings use the two names for the same policy
  ([Scores](flow/index.md#scores)).
- **Parameterization, prediction.** The quantity the network outputs, a velocity, a denoiser or a scaled score, each of
  which converts to the vector field ([Four parameterizations](flow/index.md#four-parameterizations)).
-  **Conversion gain.** $B_t$ in $u = A_t x + B_t o$, the factor by which a conversion multiplies the output's error:
  $\kappa_t/\beta_t$ for a denoiser and $\kappa_t/\alpha_t$ for a scaled score ([Four
  parameterizations](flow/index.md#four-parameterizations)).
- **Singular time.** The flow time at which an output's conversion divides by zero, $t = 1$ for a denoiser and $t = 0$
  for a scaled score; the code never evaluates an output there ([Numerics](flow/index.md#numerics)).
- **Recipe.** An output paired with the space its loss compares in: `velocity_mse`, `data_mse`,
  `data_output_velocity_loss` or `scaled_score_mse` ([Recipes](flow/training.md#recipes)).
- **Velocity loss, data loss.** The findings' names for `velocity_mse` and `data_mse`
  ([One training step](flow/training.md#one-training-step)).
-  **Conditional flow matching loss.** $\mathcal L_{\text{CFM}}$ differs from the regression onto the guided
  $u_t^{\text{target}}(x \mid y)$ by a constant independent of $\theta$, for any law of $t$, and rows whose $y$ became
  $\varnothing$ regress onto $u_t^{\text{target}}(x \mid \varnothing)$ ([Regression learns a posterior
  mean](flow/training.md#regression-learns-a-posterior-mean)).
- **Loss floor.** The expected conditional variance of the target, which no network removes, so absolute losses
  compare only between runs that share it ([Comparing training losses](flow/training.md#comparing-training-losses)).
-  **Flow-time distribution.** The law $p_{\mathrm{time}}$ of $t$ in training; since the loss averages over it, drawing
  a time more often is the same as weighting it more ([Flow-time
  distributions](flow/training.md#flow-time-distributions)).
- **Time weight.** $\lambda_t$, the factor on the velocity error that a recipe's loss amounts to at the flow time $t$
  ([Recipes](flow/training.md#recipes)).
- **Time support.** The interval $[t_{\min}, t_{\max}]$ that training draws its flow times from
  ([Drawing on a support](flow/training.md#drawing-on-a-support)).
- **Reduction.** How a chunk's squared errors are combined: summed over its valid entries (`sum`) or averaged over
  them (`example_mean`) ([Reduction](flow/training.md#reduction)).
- **Flow model, diffusion model.** A flow model simulates the ODE of $u_t^\theta$, and a diffusion model an SDE with
  $\sigma_t > 0$ and the same marginals; for one trained network they differ only in the sampler
  ([The flow model](flow/sampling.md#the-flow-model)).
- **One path, many processes.** The Diffusion Guide's observation that the ODE and a whole family of SDEs realize the
  same marginals $p_t$ ([The flow model](flow/sampling.md#the-flow-model)).
- **Flow map.** $\psi_{t \to s}(x)$, the point that the ODE started at $x$ at time $t$ reaches at time $s$
  ([The flow model](flow/sampling.md#the-flow-model)).
-  **Euler, Heun and Euler–Maruyama methods.** The first-order ODE step, the second-order predictor-corrector step, and
  the Euler step of an SDE with Brownian increments $\sqrt{h_i}\,\xi_i$ ([ODE solvers](flow/sampling.md#ode-solvers),
  [The SDE](flow/sampling.md#the-sde)).
- **Time grid, step size, NFE.** The flow times $t_0 < \dots < t_n = 1$ that a solver visits, their gaps $h_i$, and the
  number of network evaluations per chunk ([Time grids](flow/sampling.md#time-grids)).
- **SDE extension trick.** For a start $X_{t_0} \sim p_{t_0}$, adding $\frac{\sigma_t^2}{2}\nabla \log p_t$ to the
  drift and the noise $\sigma_t\,\mathrm dW_t$ keeps every marginal $p_t$, for any $\sigma_t \ge 0$
  ([The SDE](flow/sampling.md#the-sde)).
- **Probability flow ODE, reverse-time SDE.** The members $\sigma_t = 0$ and $\sigma_t^2 = 2a_t$ of that family; the
  second is DDPM's limit and lies outside the code's family $\sigma_t^2 = 2\eta\,\alpha_t\beta_t$
  ([The SDE](flow/sampling.md#the-sde)).
- **Start time.** The flow time $t_0$ at which sampling starts from $\mathcal N(0, I)$, which only approximates
  $p_{t_0}$ when $t_0 > 0$ ([Where sampling starts](flow/sampling.md#where-sampling-starts)).
- **Classifier-free guidance, guidance scale, null label.** Sampling with the mix
  $(1 - w)\,u_t^\theta(x \mid \varnothing) + w\,u_t^\theta(x \mid y)$ for a guidance scale $w$, where one network
  learns the null label $\varnothing$ because training drops $y$ ([Guidance](flow/sampling.md#guidance)).
-  **MeanFlow, average velocity.** An objective that trains $\bar u_{t,s}$, the displacement of the flow over $[t, s]$
  divided by $s - t$, so that one step can jump from noise to a chunk ([Average
  velocity](flow/meanflow.md#average-velocity)).
-  **MeanFlow identity.** $\bar u_{t,s} = u_t^{\text{target}} + (s - t)\,\frac{\mathrm d}{\mathrm dt}\bar u_{t,s}$,
  which turns the average velocity into a regression target ([The MeanFlow
  identity](flow/meanflow.md#the-meanflow-identity)).
- **Tangent.** The estimate of the vector field that moves $x$ in MeanFlow's derivative: the conditional vector field,
  or the network's own $\bar u^\theta_{t,t}$ ([Options](flow/meanflow.md#options)).
- **Adaptive weight.** $\gamma_i$, a weight without gradient on chunk $i$'s loss that down-weights chunks with a large
  error ([Options](flow/meanflow.md#options)).
- **Weight average.** $\bar\theta$, the exponential moving average of the parameters that evaluation uses by default
  ([Weight averages](training.md#weight-averages)).

## Dictionary to the references

The same objects carry other letters in the references. The table translates the main ones; a dash means that the
text does not name the object.

| object | here (the notes) | the FM Guide | the Diffusion Guide |
| --- | --- | --- | --- |
| clock | flow time $t$, noise at 0 | $t$, noise at 0 | $t$, data at 0; we write $\tau = 1 - t$ |
| noise | $\epsilon \sim p_{\mathrm{init}} = \mathcal N(0, I)$ | $X_0 \sim p$, the source | $\epsilon$; the prior $p_{\mathrm{prior}}$ at $\tau = T$ |
| data | $z \sim p_{\mathrm{data}}$ | $X_1 \sim q$, the target | $\mathbf x_0 \sim p_{\mathrm{data}}$ |
| given $y$ | guided: $p_{\mathrm{data}}(\cdot \mid y)$, $u_t^\theta(x \mid y)$ | the conditional distribution $q(x_1 \mid y)$, but the guided path $p_{t \mid Y}(x \mid y)$ and the guided velocity $u_t(x \mid y)$ | - |
| noise schedulers | $\alpha_t$, $\beta_t$ | $\alpha_t$, $\sigma_t$, a scheduler | $\alpha_t$, $\sigma_t$, in its own time |
| conditional path | $p_t(x \mid z)$ | $p_{t \mid 1}(x \mid x_1)$ | the perturbation kernel $p_t(\mathbf x_t \mid \mathbf x_0)$ |
| noisy chunk | $x = \alpha_t z + \beta_t \epsilon$ | $X_t = \alpha_t X_1 + \sigma_t X_0$ | $\mathbf x_t = \alpha_t \mathbf x_0 + \sigma_t \epsilon$ |
| conditional vector field | $u_t^{\text{target}}(x \mid z)$ | $u_t(x \mid x_1)$, the conditional velocity | - |
| marginal vector field | $u_t^{\text{target}}(x)$ | $u_t(x)$, the marginal velocity | minus the drift of the PF-ODE, whose clock runs backward |
| denoiser | $D_t(x) = \mathbb E[z \mid x]$ | $x_{1 \mid t}(x)$, the $x_1$-prediction | $\mathbb E[\mathbf x_0 \mid \mathbf x_t]$, the $\mathbf x$-prediction |
| expected noise | $\mathbb E[\epsilon \mid x]$ | $x_{0 \mid t}(x)$, the $x_0$-prediction | the $\epsilon$-prediction |
| score network | $s_t^\theta$ | $s_t^\theta$ | $\mathbf s_{\phi}$ |
| conversion coefficients | $a_t$, $b_t$ | $(a_t, b_t)$ of its conversion table, output $= a_t x + b_t \cdot$ input: our $A_t$, $B_t$; from the score to the velocity, the notes' $b_t$, $a_t$ | $g^2 = 2a_t$ and $f = -b_t$ of the forward SDE, at $\tau = 1 - t$ |
| diffusion coefficient | $\sigma_t$ of the sampling SDE, free to choose | $\beta_t$ of the added Langevin term, in its section on diffusion models | $g(t)$ of the forward SDE, fixed by the path ($g^2 = 2a_t$); our $\sigma_t$ equals it only in the reverse-time SDE |
| signal-to-noise ratio | $\alpha_t^2/\beta_t^2$ | $\rho(t) = \alpha_t/\sigma_t$, not squared | $\mathrm{SNR}(t) = \alpha_t^2/\sigma_t^2$; $\lambda_t = \log(\alpha_t/\sigma_t)$, half the log-SNR |
| flow map | $\psi_{t \to s}$; the notes' flow $\psi_t$ | $\psi_{s \mid t} = \psi_s \circ \psi_t^{-1}$; the flow $\psi_t$ | $\Psi_{s \to t}$, in its own time |
| condition dropout | $p_{\mathrm{uncond}}$ (the notes' $\eta$) | $p_{\mathrm{uncond}}$ | - |

Six names need care when moving between the texts.

- **$\sigma_t$ and $\beta_t$ swap.** The FM Guide's $\sigma_t$ is our noise scheduler $\beta_t$. Its section on
  diffusion models writes $\beta_t$ for the diffusion coefficient of the added noise, our $\sigma_t$.
- **$a_t$ and $b_t$ swap.** The FM Guide's conversion table writes each conversion as $a_t x + b_t$ times the input,
  so its $a_t$, $b_t$ are our $A_t$, $B_t$. From the score to the velocity, its $a_t = \dot\alpha_t/\alpha_t$ is the
  notes' $b_t$, and its $b_t = \beta_t\kappa_t/\alpha_t$ is the notes' $a_t$.
- **"$x_0$-prediction" means opposite things.** In diffusion papers $x_0$ is the data, so an $x_0$-prediction is a
  denoiser (`data_mse`). In the FM Guide $X_0$ is the noise, so its $x_0$-prediction is $\mathbb E[\epsilon \mid x]$:
  the noise prediction, minus our scaled score.
- **v-prediction.** Diffusion papers on a variance-preserving path regress $v = \alpha_t \epsilon - \beta_t z$, in
  our letters. On GVP, $v = -\frac{2}{\pi}\,u_t^{\text{target}}(x \mid z)$: the sign reverses the clock and $2/\pi$
  changes its speed. `velocity_mse` on GVP is therefore v-prediction, with $(\pi/2)^2$ times its loss
  ([Recipes](flow/training.md#recipes)).
- **$\rho$ and $\lambda_t$.** The FM Guide's $\rho(t)$ is an unsquared signal-to-noise ratio, not the grid exponent
  $\rho$ (`sampler.rho`). The Diffusion Guide's $\lambda_t$ is half the log-SNR, not a recipe's time weight $\lambda_t$.
- **"Conditional".** The FM Guide calls $p_{t \mid 1}(x \mid x_1)$ conditional, and also $q(x_1 \mid y)$ and the
  score $\nabla \log p_{t \mid Y}(x \mid y)$. It calls the path $p_{t \mid Y}(x \mid y)$ and the field
  $u_t(x \mid y)$ guided. The notes and these pages say guided for everything given $y$.

## Letters with two meanings

A few letters still serve twice. Some are code identifiers, which the guide does not rename; the others are told apart
by their subscripts or by the page that uses them.

| letter | meanings | how to tell them apart |
| --- | --- | --- |
| `D` | in `flow.py`, $\kappa_t$ (the field of {py:class}`~flow.Coefficients`, named after the denominator of the denoiser formula); in `model.py`'s docstrings, the DiT width $d$ | the math writes $\kappa_t$ and $d$; $D_t$ is only the denoiser |
| `y` | in `flow.py`, the argument of {py:meth}`~flow.Prediction.to_velocity`, the network's output $o_t^\theta$ | the math writes $o_t^\theta$; $y$ is only the guiding variable |
| `action_chunk` | the clean chunk $z$ in `compute_loss` and the losses; the noisy chunk $x$ in `_predict` | the docstrings say which |
| `condition` | the encoded observation $y$ | the math says guided for $y$ and conditional for $z$ |
| `velocity_loss` | the flag of a {py:class}`~flow.Recipe` whose loss is compared in velocity space | the velocity loss of the findings is `velocity_mse` |
| `d` | in code shapes, the action dimension 2 | the math writes $\mathbb R^{H \times 2}$ and keeps $d$ for the DiT width |
| $s$ | MeanFlow's end time; the score network $s_t^\theta$; the standard deviation of the Gaussian example; the scale $s_r$ between the two paths | the score network carries $t$ and $\theta$, the scale carries $r$, and the example's $s$ comes with its mean $m$ |
| $h$ | the step size $h_i$ of a time grid; MeanFlow's gap $s - t$ | both are the length of a step; the grid's carries its index $i$ |
| $n$ | the number of steps of a time grid; the optimizer steps of a weight average on {doc}`training`; the index of a window in the normalizer's statistics on {doc}`data` | the window index appears only on {doc}`data` |
| $\beta$ | the noise scheduler $\beta_t$; the shift $\beta_k$ of adaptive normalization | the subscript: $t$ or $k \in \{\text{self}, \text{MLP}\}$ |
| $\gamma$ | MeanFlow's adaptive weight $\gamma_i$; the scale $\gamma_k$ of adaptive normalization | the subscript: a chunk $i$ or a sub-layer $k$ |
| $g$ | the gate $g_k$ of adaptive normalization; MeanFlow's $g_p$; the Diffusion Guide's diffusion coefficient $g(t)$ | the subscript or argument: a sub-layer $k$, a power $p$ or the time |
| $\lambda$ | a recipe's time weight $\lambda_t$; the decay $\lambda$ of a moving average | the time weight always carries $t$ |
| $\ell$ | a chunk's reduced loss $\ell(e)$ and MeanFlow's plain loss $\ell_i$; the length of an episode in env steps on {doc}`evaluation` | the episode length appears only on {doc}`evaluation` |
| $\delta$ | a network's error in a conversion; the offset of MeanFlow's adaptive weight; $\delta_z$, the distribution that always returns $z$ | the page defines it where it is used |
| $c$ | the conditioning vector of the DiT; the per-chunk weight $1/(2\sum_k m_k)$ that separates the two reductions; the coverage $c_\tau$ on {doc}`evaluation` | {doc}`model` uses the first and [Reduction](flow/training.md#reduction) the second; the coverage carries its env step $\tau$ |
| $m$ | the mean of the Gaussian example; the loss weight $m_k$ of step $k$ | the step weight carries its step $k$ |
| $\mu$ | the mean of $\mathrm{logit}\,t$ under the logit-normal law; the per-index mean $\mu_i$ of the normalizer on {doc}`data` | the normalizer's mean carries its index $i$ |
| $A$, $B$ | the coefficients $A_t$, $B_t$ of a conversion; the batch size $B$ on {doc}`flow/meanflow`; the shape $A \times B$ of a hidden weight matrix on {doc}`training` | the coefficients carry $t$ |
| $W$ | the Brownian motion $W_t$ of the SDE; a hidden weight matrix $W$ on {doc}`training` | the Brownian motion carries $t$ |
| $o$ | the network's raw output $o_t^\theta$; the frame origin $o_t$ on {doc}`data` | the frame origin appears only on {doc}`data` |
| $v$ | the variance $v_t$ of a coordinate in the Gaussian example; the target of v-prediction; MeanFlow's tangent; the drift of a generic SDE on {doc}`flow/sampling` | the page defines it where it is used |
| $r$ | GVP's time in the equivalence of the two paths; an intermediate time of the flow map; the clock of a time grid on {doc}`flow/sampling`; the argument of a CDF on {doc}`flow/training`; the FM Guide's diffusion time; the MeanFlow papers' end time; a position on {doc}`data` and the reward $r_\tau$ on {doc}`evaluation` | the page defines it where it is used; the papers' carries MF |
| $R$ | the rotation $R(\phi)$ and $R_t = R(\phi_t)$ on {doc}`data`; an episode's max reward $R$ on {doc}`evaluation` | each appears only on its page |
| $F$ | the CDF of a flow-time distribution on {doc}`flow/training`; the network as a function of the gap, $F^\theta(x, t, h \mid y)$, on {doc}`flow/meanflow` | the network carries $\theta$ |
| $\tau$ | an integration variable; the Diffusion Guide's reversed time $1 - t$; an env step on {doc}`data` and {doc}`evaluation` | env steps appear only on those two pages, which say so |

On {doc}`data` and {doc}`evaluation`, $t$ and $\tau$ index env steps. On {doc}`data`, $s_\tau$, $a_\tau$ and
$(o_t, \phi_t)$ are a state, an action and the frame of the current state; {doc}`evaluation` writes $a_\tau$ for the
action at env step $\tau$. Both pages define these letters where they use them.
