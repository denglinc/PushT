# Sampling

A sampler turns noise into an action chunk. It draws a chunk of Gaussian noise and simulates the trained policy's
vector field from a start time up to the data at $t = 1$. Sampling is an inference setting, so every saved policy can
be scored under any sampler it supports, without retraining:

```bash
uv run evaluation.py runs=runs/default sampler.nfe=4      # 4 Euler steps instead of 10
uv run evaluation.py runs=runs/default sampler.solver=sde sampler.sde_eta=0.03 split=selection episodes=500
```

This page asks how to approximate the chunk well with a small, fixed number of network evaluations, the **NFE**. It
first builds what a sampler approximates: the flow model and its flow map. Then come the four choices a sampler makes.
The solver and its time grid discretize the flow. An SDE adds noise with a diffusion coefficient $\sigma_t$. The start
time $t_0$ says where the simulation begins, and the guidance scale $w$ says how strongly it follows the observation.
The path, the posterior means and the conversions are those of {doc}`index`; {doc}`../notation` lists every symbol.

:::{admonition} Overview: four choices, three errors
:class: note

In theory every solver, grid, noise scale and start time yields chunks distributed as
$p_{\mathrm{data}}(\cdot \mid y)$. This needs the exact vector field, steps that shrink to zero and a start drawn from
the path itself, $X_{t_0} \sim p_{t_0}$. Only a guidance scale $w \ne 1$ changes the target. In practice three errors
decide. The network has a training error, which the conversion of its output to a vector field can amplify. A finite
number of steps has a discretization error, even for the exact field. And a start after $t = 0$ has a start mismatch,
since the sampler's Gaussian start only approximates the path there. The first three choices trade among these errors,
and the fourth leaves the target on purpose:

| choice | config | default | what it trades in practice |
| --- | --- | --- | --- |
| [solver, NFE](#ode-solvers) and [grid](#time-grids) | `sampler.solver`, `sampler.nfe`, `sampler.grid` | Euler, 10, uniform | discretization error against network evaluations |
| [noise](#the-sde) $\sigma_t$ | `sampler.solver=sde`, `sampler.sde_eta` | none (the ODE) | a pull back toward the path against a harder discretization |
| [start time](#where-sampling-starts) $t_0$ | `sampler.start_time` | 0, or 0.05 for the [scaled score](index.md#scores) | amplified network error against start mismatch |
| [guidance scale](#guidance) $w$ | `sampler.cfg_scale` | 1 | a closer fit to $y$ against leaving $p_{\mathrm{data}}(\cdot \mid y)$ |
:::

## How the findings were measured

The findings on this page score samplers on policies trained once, so they compare samplers, not trainings. Each
episode draws its sampling noise from two generators of its own, both seeded by its env seed: one for the initial
noise of every query, one for the Brownian draws of the SDE. The k-th query of an episode therefore starts from the
same noise under every sampler, ODE and SDE alike, and a sampler's Brownian draws never shift it. Scores of two
samplers on the same policy and episodes are paired, and the contrasts below are paired within each policy.
{doc}`../evaluation` describes the protocol ([seeds and splits](../evaluation.md#seeds-and-splits)).

The sampler findings come from phase 6 (PushT, AdamW, every path and prediction, a scaled score then started at
0.001) on the selection episodes. Only the scaled score's start was confirmed on held-out episodes with fresh
training seeds. The only other held-out evidence, the MeanFlow study's Adam policies (linear path, velocity), agrees
only in part: there 4 Euler steps lose clearly to 10, and the data-dense grid at 2 NFE gains within the noise.

## The flow model

What does a trained policy define, and what must a sampler compute? The network maps a noisy chunk $x$, a flow time
$t$ and the guiding variable $y$ to a raw output $o_t^\theta(x \mid y)$. Depending on the recipe, the output is a
vector field, a denoiser or a scaled score. Each conversion to a vector field has the affine form of
[the parameterization table](index.md#four-parameterizations),

$$
u_t^\theta(x \mid y) = A_t\,x + B_t\,o_t^\theta(x \mid y),
$$

with coefficients $A_t$ and $B_t$ that depend only on the path and on $t$. The **flow model** is the ODE of this
vector field, started from Gaussian noise at a **start time** $t_0$:

$$
\begin{aligned}
X_{t_0} &\sim \mathcal N(0, I) && \blacktriangleright\ \text{initialize with noise} \\
\tfrac{\mathrm d}{\mathrm dt} X_t &= u_t^\theta(X_t \mid y), \quad t_0 \le t \le 1 && \blacktriangleright\ \text{simulate the ODE} \\
X_1 &\in \mathbb R^{H \times 2} && \blacktriangleright\ \text{the action chunk}
\end{aligned}
$$

The notes start at $t_0 = 0$, where $\mathcal N(0, I) = p_{\mathrm{init}} = p_0$ exactly; a later start has
[a price](#where-sampling-starts). As on {doc}`index`, the path $p_t$, the field $u_t^{\text{target}}$ and the
posterior means are those of $p_{\mathrm{data}}(\cdot \mid y)$ for one fixed $y$.

**The flow map.** A sampler needs only where a trajectory ends, not the whole trajectory. For a vector field $u_t$ and
times $t \le s$, the **flow map** $\psi_{t \to s}(x)$ is the solution at time $s$ of
$\frac{\mathrm d}{\mathrm d\tau}X_\tau = u_\tau(X_\tau)$ started at $X_t = x$. It is characterized by

$$
\begin{aligned}
\psi_{t \to t}(x) &= x && \blacktriangleright\ \text{no time, no motion} \\
\partial_s\,\psi_{t \to s}(x) &= u_s\big(\psi_{t \to s}(x)\big) && \blacktriangleright\ \text{it moves with the field} \\
\psi_{r \to s} \circ \psi_{t \to r} &= \psi_{t \to s}, \quad t \le r \le s && \blacktriangleright\ \text{composition}
\end{aligned}
$$

The last line says that the state at time $s$ depends only on the state at any earlier time $r$, and deterministically:
a flow model is a deterministic Markov process. We write $\psi_{t \to s}$ for the flow map of the exact field
$u_t^{\text{target}}(\cdot \mid y)$ and $\psi^\theta_{t \to s}$ for that of $u_t^\theta(\cdot \mid y)$. By the
marginalization trick, the exact map carries the path along: $X_t \sim p_t$ gives $\psi_{t \to s}(X_t) \sim p_s$
([the marginal vector field](index.md#the-marginal-vector-field)).

The chunk of the flow model is $X_1 = \psi^\theta_{t_0 \to 1}(X_{t_0})$, a deterministic function of the noise draw. A
**sampler** approximates $\psi^\theta_{t_0 \to 1}$ with a fixed NFE. Under the ODE, the chunk
$\psi^\theta_{t_0 \to 1}(X_{t_0})$ depends only on the start time and the noise draw. Two samplers with the same $t_0$
and the same noise therefore aim at the same chunk. This is what makes the [paired
comparisons](#how-the-findings-were-measured) informative.

Note that this pairing of $X_0$ with $\psi_{0 \to 1}(X_0)$ is not the pairing of training. Training draws the noise
$\epsilon$ independently of the chunk $z$, while the ODE maps each noise draw to one chunk. The two constructions have
the same marginals $p_t$ but different joint laws (the two random variables called $X_t$ of
[the marginal vector field](index.md#the-marginal-vector-field)). The one-step argument below rests on this
difference.

:::{admonition} Observation: one path, many processes
:class: important

The probability path $p_t$ prescribes only the distribution at each time, and many processes realize it. The ODE of
the marginal vector field is one. The SDEs of [The SDE](#the-sde), one for every diffusion coefficient
$\sigma_t \ge 0$, are others. Even the conditional vector field that training regresses onto is one choice among many
fields that generate $p_t(\cdot \mid z)$. What a process must satisfy is the continuity equation of $p_t$, or, with
noise, its Fokker–Planck equation ([The SDE](#the-sde)).
:::

In other words, the same network simulated with an SDE is a **diffusion model**, and a flow model and a diffusion model
trained on the same path differ only in the sampler. With the exact fields both produce
$p_{\mathrm{data}}(\cdot \mid y)$. The ODE maps each noise draw to one chunk, while an SDE adds fresh noise on the way,
so one noise draw gives many chunks. Which of the two samples better is a question about errors, not about the target.
The Diffusion Guide states this as its observation on matching prescribed marginal densities.

**Where a conversion is undefined.** Two conversions divide by a scheduler that vanishes at an end of the path
([the singular ends](index.md#four-parameterizations)). The sampler never evaluates an output where its conversion is
undefined:

| network output $o_t^\theta$ | vector field $u_t^\theta$ | undefined at | what the sampler does |
| --- | --- | --- | --- |
| vector field | the output itself | nowhere | nothing |
| denoiser $D_t^\theta$ | $(\dot\beta_t x + \kappa_t D_t^\theta)/\beta_t$ | $t = 1$, where $\beta_1 = 0$ | never evaluates it at $t = 1$, so it refuses [Heun's method](#ode-solvers) |
| scaled score $\beta_t s_t^\theta$ | $(\dot\alpha_t x + \kappa_t \beta_t s_t^\theta)/\alpha_t$ | $t = 0$, where $\alpha_0 = 0$ | [starts](#where-sampling-starts) at $t_0 = 0.05$ |

## ODE solvers

The flow map has no closed form, so how does a sampler compute it? It replaces the ODE by a finite number of steps.
A **time grid** $t_0 < t_1 < \dots < t_n = 1$ splits $[t_0, 1]$ into $n$ steps of sizes $h_i = t_{i+1} - t_i$;
[Time grids](#time-grids) says where they go.

**The Euler method** freezes the vector field over a step at its value at the step's start:

$$
X_{t_{i+1}} = X_{t_i} + h_i\,u^\theta_{t_i}(X_{t_i} \mid y) .
$$

It is the first-order Taylor expansion of the trajectory. Since $\dot X_t = u_t^\theta(X_t \mid y)$, for a smooth field

$$
X_{t + h} = X_t + h\,\dot X_t + \tfrac{h^2}{2}\,\ddot X_t + O(h^3),
$$

so each step errs by about $\frac{h^2}{2}\ddot X_t$, and about $1/h$ steps together err by $O(h)$. The Euler method
therefore has **order 1**: its error vanishes in proportion to the step size. It is exact when $\ddot X_t = 0$, that
is, when the trajectory is a straight line run at constant speed. One step then suffices, as the FM Guide notes for
optimal-transport trajectories.

Note that a straight conditional path does not make a trajectory of the flow model that moves at constant velocity. On
the linear path each conditional path $t z + (1 - t)\epsilon$ is a straight line run at constant speed, but the marginal
trajectories accelerate. In [the example with Gaussian data](index.md#an-example-with-gaussian-data), with mean $m$ and
spread $s$ per coordinate, $X_t = t m + \sqrt{t^2 s^2 + (1 - t)^2}\,X_0$. Its acceleration vanishes only when $s = 0$, a
single chunk.

**Heun's method** takes the Euler step as a first guess and corrects it with the average of the field at both ends
of the step:

$$
\begin{aligned}
X'_{t_{i+1}} &= X_{t_i} + h_i\,u^\theta_{t_i}(X_{t_i} \mid y) && \blacktriangleright\ \text{predict with an Euler step} \\
X_{t_{i+1}} &= X_{t_i} + \tfrac{h_i}{2}\big[u^\theta_{t_i}(X_{t_i} \mid y) + u^\theta_{t_{i+1}}(X'_{t_{i+1}} \mid y)\big] && \blacktriangleright\ \text{correct with the trapezoid rule}
\end{aligned}
$$

The trapezoid rule cancels the $\frac{h^2}{2}\ddot X_t$ term, so each step errs by $O(h^3)$ and the whole grid by
$O(h^2)$: Heun's method has **order 2**. It costs 2 NFE per step, and its last corrector evaluates $t = 1$.

**MeanFlow's step.** A policy trained with MeanFlow outputs an average velocity $\bar u^\theta_{t,s}(x \mid y)$ over
$[t, s]$. The `mean` solver steps with it, at 1 NFE per step:
$X_{t_{i+1}} = X_{t_i} + h_i\,\bar u^\theta_{t_i, t_{i+1}}(X_{t_i} \mid y)$. The exact average velocity is defined so
that this step is the flow map, $x + (s - t)\,\bar u_{t,s}(x) = \psi_{t \to s}(x)$ ([average
velocity](meanflow.md#average-velocity)). With the exact average velocity, the `mean` step is therefore exact on any
grid. Evaluated at $s = t$, a MeanFlow network estimates the vector field, since its training rows with $s = t$ are flow
matching ([the target](meanflow.md#the-target)). A MeanFlow policy therefore also runs under the other solvers.

**What the tests check.** On the linear path, with a velocity prediction of the exact field of Gaussian data, the tests
check that the Euler method converges at order 1 and Heun's method at order 2 (`test_convergence_orders` in
`tests/test_flow.py`). On both paths the tests also check that each solver reaches the exact endpoint for every
prediction it accepts (`test_ode_reaches_the_exact_endpoint`; Heun is not run on data predictions, which it would
evaluate at $t = 1$).

**Comparing the solvers.** Two comparisons are fair, and they answer different questions. At equal NFE, which is equal
cost, Heun's method takes half as many steps as the Euler method. At equal steps, which is equal resolution, it costs
twice the NFE. In theory, for an exact and smooth vector field, order 2 wins both comparisons once the steps are small
enough, the one at equal steps sooner. In practice the field is learned, and two things could break the argument. Each
Heun step evaluates the network twice, the second time at the Euler guess $X'_{t_{i+1}}$, so each step takes in the
network's error twice. And Heun's last corrector evaluates $t = 1$, where the exact field of finitely many
demonstrations changes fastest. The study measured the outcome, not its cause:

**What we found** (phase 6, the policies without a scaled score; Heun on the velocity policies)

- **NFE 10 = 20 = 50 (Euler).** Ten steps are enough; more add nothing.
- **NFE 10 > NFE 4 for velocity policies.** The loss is small. Held-out episodes resolve it (the MeanFlow study's
  Adam policies); over all phase-6 policies on selection episodes it stays within the noise.
- **NFE 4 > NFE 2 > NFE 1.** Below four steps success falls fast.
- **Euler > Heun at equal NFE**, at every budget, 20 included.
- **Euler > Heun at equal steps** too, although Heun then spends twice the NFE: its corrector lowers success on these
  policies. Why was not measured.

Try: `uv run evaluation.py -m runs=runs/flow_matching split=selection episodes=500 'sampler.nfe=1,2,4,10,20'`, and
`sampler.solver=heun sampler.nfe=20` for Heun.

## Time grids

Where should a few steps go? The sampler places them with a grid exponent $\rho \ge 1$:

$$
t_i = 1 - (1 - t_0)\left(1 - \frac{i}{n}\right)^{\rho}, \qquad i = 0, \dots, n .
$$

The **uniform grid** has $\rho = 1$, equal steps $t_i = t_0 + (1 - t_0)\,i/n$. The **data-dense grid** has
$\rho > 1$. Its factor $(1 - i/n)^\rho$ flattens as $i \to n$, so the step sizes $h_i$ shrink toward the data. The
number of steps $n$ is the NFE for the Euler method and the `mean` solver, and half the NFE for Heun's method. At 2 NFE
of the Euler method from $t_0 = 0$, for example, the grid's middle point is 0.5 on the uniform grid, 0.646 at
$\rho = 1.5$, 0.75 at $\rho = 2$ and 0.875 at $\rho = 3$. The SDE builds [its own grid](#the-sde).

**A grid is a clock.** In theory the grid does not matter: as the steps shrink, every grid reaches the same $X_1$. There
is a sharper way to say this. Write the grid as $t_i = t_{r_i}$, where $r_i = i/n$ are equal steps of a new clock
$r \in [0, 1]$ and $r \mapsto t_r$ increases from $[0, 1]$ onto $[t_0, 1]$. Run on the clock $r$, the path has the
schedulers $\alpha_{t_r}$, $\beta_{t_r}$ and the vector field $\dot t_r\,u_{t_r}(x)$, with
$\dot t_r = \mathrm dt_r/\mathrm dr$. Its exact flow map from $r = 0$ to $r = 1$ is that of the original field from
$t_0$ to 1. An Euler step on the grid moves by $(t_{r_{i+1}} - t_{r_i})\,u_{t_{r_i}}$, and a uniform Euler step on the
re-clocked path moves by $\frac1n\,\dot t_{r_i}\,u_{t_{r_i}}$. The two increments agree to first order in $1/n$.
Wherever $\ddot t_r$ is bounded they differ at second order, the order of Euler's own error per step. To first order,
then, a grid is the FM Guide's post-training change of scheduler, a pure change of clock without rescaling, followed by
equal steps. With exact fields every scheduler leads to the same samples ([Paths](index.md#paths)), so the grid only
decides where the discretization error falls. The data-dense grid puts the short steps near the data. That is where the
trajectories of concentrated data accelerate. In the Gaussian example on the linear path,
$X_t = t m + \sqrt{t^2 s^2 + (1 - t)^2}\,X_0$ has the acceleration $\ddot X_t = s^2 X_0/(t^2 s^2 + (1 - t)^2)^{3/2}$,
which for a small spread $s$ is concentrated within about $s$ of $t = 1$.

**Grid against flow-time distribution.** Training's [flow-time distribution](training.md#flow-time-distributions)
also spreads effort along $t$, but it acts on a different error. The time distribution spends network capacity, which
acts on training error. The grid spends network evaluations, which act on discretization error. The two are
independent choices, except that a grid should evaluate the network only at times it was trained on.

**In the code.** {py:meth}`~flow.Sampler.times` builds the grid: `sampler.grid=uniform` or `data_dense`, with
`sampler.rho` for $\rho$. The grid is computed in float64 and rounded to float32, the type the network sees. Training
records it as `info.sampler_times` in the run's `config.json`, and `evaluation.py` as `sampler_times` in its own.

**What we found** (phase 6, the policies without a scaled score)

- **data_dense > uniform at 2 NFE on GVP** (ρ 1.5 and 2). It brings the GVP path to the linear path's level.
- **data_dense > uniform (weak) at 2 NFE on the linear path.** Held-out episodes (the MeanFlow study's Adam
  policies) show the same small, unresolved gain.
- **ρ = 3 = uniform at 2 NFE** on average. It overshoots on two of the three GVP policy types and helps the third.
- **data_dense = uniform at 4 NFE.** No grid comparison was made at 10 NFE or more.

Try: `uv run evaluation.py -m runs=runs/flow_matching split=selection episodes=500 sampler.nfe=2 sampler.grid=data_dense 'sampler.rho=1.5,2'`

## One Euler step on GVP

What does a single Euler step return, and why does it fail on GVP but not on the linear path? In theory the path does
not matter. With exact fields, the linear and the GVP flows carry every noise draw to the same chunk, since the two
paths differ only by a clock and a scale ([Paths](index.md#paths)). In practice they differ in how well a few Euler
steps follow that common map. We compute this first for the exact field and then for a network. Recall the two
posterior-mean identities of [the marginal vector field](index.md#the-marginal-vector-field), with the denoiser
$D_t(x) = \mathbb E[z \mid x]$:

$$
x = \alpha_t\,D_t(x) + \beta_t\,\mathbb E[\epsilon \mid x], \qquad
u_t^{\text{target}}(x) = \dot\alpha_t\,D_t(x) + \dot\beta_t\,\mathbb E[\epsilon \mid x] .
$$

An Euler step from $t$ with step size $h$ therefore extrapolates the schedulers linearly:

$$
X_t + h\,u_t^{\text{target}}(X_t)
\overset{(i)}{=} \alpha_t D_t + \beta_t\,\mathbb E[\epsilon \mid X_t] + h\big(\dot\alpha_t D_t + \dot\beta_t\,\mathbb E[\epsilon \mid X_t]\big)
\overset{(ii)}{=} (\alpha_t + h\dot\alpha_t)\,D_t + (\beta_t + h\dot\beta_t)\,\mathbb E[\epsilon \mid X_t],
$$

where (i) writes the state through the path identity and the field through its posterior-mean form, with
$D_t = D_t(X_t)$, and (ii) collects terms.

**Euler and DDIM.** The DDIM sampler was introduced for variance-preserving schedules but is defined on any Gaussian
path. It keeps the same two posterior means and places them on the path at the next time:

$$
X_{t+h} = \alpha_{t+h}\,D_t(X_t) + \beta_{t+h}\,\mathbb E[\epsilon \mid X_t] .
$$

Its step to $t = 1$ therefore always lands on the predicted chunk $D_t(X_t)$. The two steps agree for all posterior
means exactly when $\alpha_{t+h} = \alpha_t + h\dot\alpha_t$ and $\beta_{t+h} = \beta_t + h\dot\beta_t$, that is, when
the schedulers are affine in $t$. On the linear path the Euler method is DDIM, step for step. On GVP the schedulers
curve, with $\ddot\alpha_t = -\frac{\pi^2}{4}\alpha_t$ and $\ddot\beta_t = -\frac{\pi^2}{4}\beta_t$. To second order in
$h$, the Euler step minus the DDIM step is therefore
$-\frac{h^2}{2}\big(\ddot\alpha_t D_t + \ddot\beta_t\,\mathbb E[\epsilon \mid X_t]\big) = \frac{\pi^2}{8}h^2 X_t$, by
the path identity. With a network, the same holds with the two means its output implies through the path identity. For a
data prediction these are $D_t^\theta$ and $(x - \alpha_t D_t^\theta)/\beta_t$.

**One step from $t = 0$.** With $n = 1$ the Euler method takes a single step of size $h = 1$ from the noise draw
$X_0$. At $t = 0$ the posterior means are explicit ([the marginal vector field](index.md#the-marginal-vector-field)):
$\mathbb E[\epsilon \mid X_0] = X_0$, and $D_0(X_0) = \mathbb E[z]$ is the mean chunk for the observation. Therefore

$$
X_1 = (\alpha_0 + \dot\alpha_0)\,\mathbb E[z] + (\beta_0 + \dot\beta_0)\,X_0 = \dot\alpha_0\,\mathbb E[z] + \big(1 + \dot\beta_0\big)\,X_0 .
$$

- Linear path, $\dot\alpha_t = 1$ and $\dot\beta_t = -1$: the noise cancels, and $X_1 = \mathbb E[z]$ for every noise
  draw.
- GVP, $\dot\alpha_0 = \frac{\pi}{2}\beta_0 = \frac{\pi}{2}$ and $\dot\beta_0 = -\frac{\pi}{2}\alpha_0 = 0$:
  $X_1 = X_0 + \frac{\pi}{2}\mathbb E[z]$. Since $\dot\beta_0 = 0$, the step keeps all the noise, which DDIM would
  remove.

**Reading the linear result.** One linear step returns the same chunk $D_0 = \mathbb E[z]$ for every noise draw. This is
exactly the output of a behavior-cloning policy trained with a squared error; when the demonstrations for one
observation take two routes, it can lie between them. In the Gaussian example the step lands on $m$, while the flow
model's exact endpoint is $m + s X_0$: one step loses the whole spread $s X_0$. Here the straight-path picture stops.
The linear path makes each conditional path straight, and one Euler step would follow it exactly. The marginal
trajectories accelerate, because the posterior means change along them. One step freezes them at $t = 0$, where they know
nothing about $z$ beyond its mean. {doc}`meanflow` starts from this gap ([why one step needs
more](meanflow.md#why-one-step-needs-more-than-the-vector-field)).

**With a network.** A data prediction gives the same through its conversion at $t = 0$, with its output
$D_0^\theta(X_0)$ in place of $\mathbb E[z]$. A velocity prediction outputs its estimate of
$u_0^{\text{target}}(X_0) = \dot\alpha_0\,\mathbb E[z] + \dot\beta_0 X_0$ directly. A scaled score never takes this
step, since it starts at $t_0 > 0$.

**A shorter last step.** The same identity gives every Euler step that ends at $t = 1$. From time $t$ it returns
$(\alpha_t + (1 - t)\dot\alpha_t)\,D_t + (\beta_t + (1 - t)\dot\beta_t)\,\mathbb E[\epsilon \mid X_t]$. On the linear
path the two coefficients are 1 and 0 from every $t$. On GVP they approach 1 and 0 only as the last step shrinks:

| last step from $t$ | where it occurs | coefficient of $D_t$ | coefficient of $\mathbb E[\epsilon \mid X_t]$ |
| --- | --- | --- | --- |
| 0 | 1 NFE | 1.571 | 1 |
| 0.5 | 2 NFE, uniform grid | 1.262 | 0.152 |
| 0.646 | 2 NFE, data-dense grid, $\rho = 1.5$ | 1.143 | 0.055 |
| 0.75 | 2 NFE, data-dense grid, $\rho = 2$ | 1.074 | 0.020 |
| 0.999 | the SDE's last step ([below](#the-sde)) | 1.000 | 0.000 |

A shorter last step brings GVP's Euler step closer to the DDIM step that the linear path takes exactly. This fits the
data-dense grid's gain on GVP at 2 NFE.

**What we found** (phase 6)

- **linear >> GVP at 1 Euler step** for the velocity and data predictions, with either loss: GVP collapses, the
  linear path does not. At 2 NFE the data-dense grid brings GVP back to the linear path's level
  ([time grids](#time-grids)).

Try: `uv run evaluation.py -m runs=runs/flow_matching split=selection episodes=500 'sampler.nfe=1,2'`

## The SDE

The ODE carries an error made early all the way to the chunk. Can noise correct it? It turns out that noise can be
added to the flow model without changing its marginals, provided a score term is added to the drift.

:::{admonition} Theorem: the SDE extension trick
:class: important

Let $\sigma_t \ge 0$ be any **diffusion coefficient** and $W_t$ a Brownian motion. If $X_{t_0} \sim p_{t_0}$ and

$$
\mathrm dX_t = \Big[u_t^{\text{target}}(X_t) + \frac{\sigma_t^2}{2}\,\nabla\log p_t(X_t)\Big]\mathrm dt + \sigma_t\,\mathrm dW_t,
\qquad t_0 \le t \le 1,
$$

then $X_t \sim p_t$ for every $t \in [t_0, 1]$. In particular $X_1 \sim p_{\mathrm{data}}(\cdot \mid y)$.
:::

Intuitively, the noise $\sigma_t\,\mathrm dW_t$ spreads probability out, and the score term pushes it back uphill at
exactly the same rate. The noise adds $\frac{\sigma_t^2}{2}\Delta p_t$ to the change $\partial_t p_t$ of the density,
where $\Delta$ is the Laplacian. The score term removes it again, since
$\mathrm{div}(p_t\nabla\log p_t) = \mathrm{div}(\nabla p_t) = \Delta p_t$.

:::{dropdown} Proof, through the Fokker–Planck equation
An SDE $\mathrm dX_t = v_t(X_t)\,\mathrm dt + \sigma_t\,\mathrm dW_t$ moves $X_{t_0} \sim p_{t_0}$ along a path $p_t$
exactly when the pair satisfies the **Fokker–Planck equation**
$\partial_t p_t = -\mathrm{div}(p_t v_t) + \frac{\sigma_t^2}{2}\Delta p_t$. With $\sigma_t = 0$ it is the continuity
equation. Then

$$
\begin{aligned}
\partial_t p_t
&\overset{(i)}{=} -\mathrm{div}\big(p_t\,u_t^{\text{target}}\big)
\overset{(ii)}{=} -\mathrm{div}\big(p_t\,u_t^{\text{target}}\big) - \frac{\sigma_t^2}{2}\Delta p_t + \frac{\sigma_t^2}{2}\Delta p_t \\
&\overset{(iii)}{=} -\mathrm{div}\Big(p_t\Big[u_t^{\text{target}} + \frac{\sigma_t^2}{2}\nabla\log p_t\Big]\Big) + \frac{\sigma_t^2}{2}\Delta p_t,
\end{aligned}
$$

where (i) is the continuity equation of the marginal pair ([the marginal vector
field](index.md#the-marginal-vector-field)), (ii) adds and subtracts the same term, and (iii) writes
$\Delta p_t = \mathrm{div}(p_t\nabla\log p_t)$ and uses the linearity of the divergence. The SDE's pair satisfies the
Fokker–Planck equation, so it keeps the path. Holderrieth and Erives prove the Fokker–Planck equation itself.
:::

Note that the theorem holds for every $\sigma_t$, so $\sigma_t$ is a sampling choice, free to change after training. In
practice two errors choose it. The network is not the exact field, which is a training error. And the Euler–Maruyama
method below needs small steps when $\sigma_t$ is large, which is a simulation error. For a trained policy and a step
budget there is therefore a best $\sigma_t$, and it is found empirically. The precondition $X_{t_0} \sim p_{t_0}$ also
holds only at $t_0 = 0$, since the sampler starts from $\mathcal N(0, I)$ ([Where sampling
starts](#where-sampling-starts)).

**Why noise can help.** The two added terms, $\frac{\sigma_t^2}{2}\nabla\log p_t\,\mathrm dt + \sigma_t\,\mathrm dW_t$,
are Langevin dynamics for the current $p_t$. On their own they leave $p_t$ fixed. Under mild conditions they also pull
a distribution that has drifted off $p_t$ back toward it, as the notes' remark on Langevin dynamics explains. The ODE
has no such pull: an error made early is carried to $X_1$. That is the case for adding noise, and its cost is a harder
discretization.

**The score from the vector field.** The sampler has no score network. It uses the score that the trained vector field
implies, by the notes' conversion formula $u_t = a_t\nabla\log p_t + b_t x$ read backwards, with
$a_t = \beta_t\kappa_t/\alpha_t$ and $b_t = \dot\alpha_t/\alpha_t$ ([Four
parameterizations](index.md#four-parameterizations)). For $0 < t < 1$,

$$
s_t^\theta(x \mid y) = \frac{u_t^\theta(x \mid y) - b_t\,x}{a_t} = \frac{\alpha_t\,u_t^\theta(x \mid y) - \dot\alpha_t\,x}{\beta_t\,\kappa_t},
$$

and the last form is defined at $t = 0$ as well.

**Choosing $\sigma_t$.** The theorem allows any $\sigma_t$, but the score is hard to use near the data. It divides by
$\beta_t$, which vanishes at $t = 1$. The exact score $-\mathbb E[\epsilon \mid x]/\beta_t$ grows like $1/\beta_t$
there too, wherever the expected noise does not vanish. The term $\frac{\sigma_t^2}{2}s_t^\theta$ therefore stays
bounded only if $\sigma_t^2$ vanishes at least as fast as $\beta_t$. The code takes

$$
\sigma_t^2 = 2\eta\,\alpha_t\beta_t, \qquad \eta \ge 0,
$$

with a **noise scale** $\eta$. The factor $\beta_t$ cancels the score's $1/\beta_t$. The factor $\alpha_t$ is a design
choice, not a necessity: it switches the noise off at $t = 0$, where the state is pure noise already. The choice
$\sigma_t^2 = 2\eta\beta_t$ would also keep the drift bounded. The drift follows in two steps:

$$
u_t^\theta + \frac{\sigma_t^2}{2}\,s_t^\theta
\overset{(i)}{=} u_t^\theta + \eta\,\alpha_t\beta_t\,\frac{\alpha_t u_t^\theta - \dot\alpha_t x}{\beta_t\kappa_t}
\overset{(ii)}{=} u_t^\theta + \eta\,\alpha_t\,\frac{\alpha_t u_t^\theta - \dot\alpha_t x}{\kappa_t},
$$

where (i) inserts $\sigma_t^2$ and the converted score, and (ii) cancels $\beta_t$. The drift divides only by
$\kappa_t > 0$, so it is bounded wherever $u_t^\theta$ is, for every path and output. In other words, this is the
notes' Gaussian form of the trick, $(a_t + \frac{\sigma_t^2}{2})\,s_t^\theta + b_t x$. The SDE multiplies the score part
$a_t s_t^\theta$ of the flow's drift by $1 + \frac{\sigma_t^2}{2a_t} = 1 + \eta\alpha_t^2/\kappa_t$. On the linear path,
$\alpha_t = t$ and $\dot\alpha_t = \kappa_t = 1$, so the drift is $u_t^\theta - \eta t\,(x - t\,u_t^\theta)$ with
$\sigma_t^2 = 2\eta t(1 - t)$.

**Two named members.** Two members of the family have names. With $\sigma_t = 0$ it is the **probability-flow ODE**,
the flow model itself. With $\sigma_t^2 = 2a_t$ the score part doubles, and the SDE becomes

$$
\mathrm dX_t = \big[2a_t\,\nabla\log p_t(X_t) + b_t\,X_t\big]\mathrm dt + \sqrt{2a_t}\,\mathrm dW_t ,
$$

the **reverse-time SDE** of the diffusion model whose forward noising process has our path as its perturbation
kernel. On a variance-preserving path, DDPM's ancestral sampler tends to it as its steps shrink (the Diffusion Guide).
On the linear path $2a_t = 2(1 - t)/t$, which diverges at $t = 0$. The code's family never contains it: there
$\frac{\sigma_t^2}{2a_t} = \eta\alpha_t^2/\kappa_t$ varies with $t$, while the reverse-time SDE needs it equal to 1.

All these processes share the marginals, so sampling does not need a time reversal. The FM Guide stresses that the
probability-flow ODE is not a time reversal in the strict sense, yet it is the state of the art at a low NFE.

**The noise scale across paths.** $\eta$ means a different diffusion coefficient on each path. $\alpha_t\beta_t$ peaks
at $1/4$ on the linear path, where it is $t(1 - t)$, and at $1/2$ on GVP, where it is $\frac12\sin\pi t$. Compare
$\eta$ within a path.

**The Euler–Maruyama method** takes a step of size $h$ along the drift, plus Gaussian noise scaled by
$\sqrt{h}\,\sigma_t$. The SDE's grid, below, gives all these steps the same size $h$. With fresh draws
$\xi_i \sim \mathcal N(0, I)$, independent of the initial noise and of each other,

$$
X_{t_{i+1}} = X_{t_i}
+ h\,\Big[u_{t_i}^\theta(X_{t_i} \mid y) + \frac{\sigma_{t_i}^2}{2}\,s_{t_i}^\theta(X_{t_i} \mid y)\Big]
+ \sqrt{h}\,\sigma_{t_i}\,\xi_i .
$$

**The grid.** With a budget of NFE network evaluations, the SDE takes NFE − 1 Euler–Maruyama steps of equal size $h$
from $t_0$ to `sampler.sde_end` (0.999), then one deterministic Euler step to $t = 1$. That last step costs one of the
evaluations for a step of 0.001. At 10 NFE from $t_0 = 0$ the stochastic part therefore takes 9 steps of 0.111, where
the Euler method's uniform grid takes 10 steps of 0.1. $\eta = 0$ is the Euler method on this grid, not on the
uniform one.

The last step is a denoising step. On the linear path the Euler method is DDIM ([above](#one-euler-step-on-gvp)). Every
Euler step to $t = 1$ therefore lands on the predicted chunk,
$X_t + (1 - t)\,u_t^\theta(X_t \mid y) = D_t^\theta(X_t \mid y)$, the denoiser that the output implies. It removes
whatever noise the last stochastic step left. On GVP, from 0.999, it does the same to within $10^{-5}$.

**In the code.** The last evaluation is at `sde_end` < 1, so a data prediction runs under the SDE too. The runs trained
with `data_output_velocity_loss` saw only $t \le 0.99$ ([Endpoints](training.md#endpoints)), so their SDE stops at
`sampler.sde_end=0.99`. {py:meth}`~flow.LangevinFlow.drift` and {py:meth}`~flow.LangevinFlow.diffusion` implement the
drift and $\sigma_t$, with $\eta$ from `sampler.sde_eta`. The tests check the linear-path form bit for bit. They also
check that the SDE keeps the marginals of Gaussian data when it starts from the exact $p_{t_0}$
(`test_sde_keeps_the_marginals`).

**What we found** (phase 6, every path and prediction)

- **SDE η 0 = Euler** at 10 NFE, for the policies without a scaled score. There the SDE's grid alone changes nothing.
- **ODE = SDE η ≤ 0.03 > SDE η 0.1-0.3 > SDE η 1** at 10 NFE, for the policies whose ODE works. At η = 1 the
  velocity policies and the data outputs trained with $\mathcal L_{\text{CFM}}$ (`data_output_velocity_loss`) lose
  clearly; the policies trained with the denoiser regression (`data_mse`) lose nothing.
- **SDE = ODE at 20 and 50 NFE**, even with each policy type's best η picked on the same episodes. The SDE never
  beats the ODE of a working policy.
- **SDE (η ≤ 0.3) > Heun** at 10 NFE. Heun was scored on the velocity policies only.
- **SDE > ODE for a scaled score started at 0.001**, at large η. This fits the noise partly repairing the damage of
  its first step ([below](#where-sampling-starts)); starting at 0.05 repairs all of it.
- **ODE > SDE η 1 for the scaled score from 0.05.** No noise scale helps it: smaller η are below the ODE within the
  noise, except η 0.3 on GVP, which also loses clearly.

Reproduce: `uv run evaluation.py -m runs=runs/flow_matching split=selection episodes=500 sampler.solver=sde 'sampler.sde_eta=0,0.03,0.1,0.3,1.0'`
(the data-output runs add `sampler.sde_end=0.99`; add `sampler.start_time=0.001` for the scaled score's old start)

## Where sampling starts

Where should the simulation begin? By default the sampler starts every solver at $t_0 = 0$, or at $t_0 = 0.05$ for the
scaled score, whose conversion divides by $\alpha_t$. Sampling then starts from $X_{t_0} \sim \mathcal N(0, I)$. Why
not start the scaled score at 0 too?

**In theory the division is harmless.** The scaled score's vector field is
$u_t^\theta = (\dot\alpha_t x + \kappa_t\,o_t^\theta)/\alpha_t$, with the output $o_t^\theta = \beta_t s_t^\theta$. For
the exact output, $-\mathbb E[\epsilon \mid x]$, the path identity makes the numerator $\alpha_t$ times the field. So
$\alpha_t$ cancels, and the field tends to $\dot\alpha_0\,\mathbb E[z] + \dot\beta_0\,x$ as $t \to 0$ ([the singular
ends](index.md#four-parameterizations)). The singularity is removable at every $x$.

**In practice the division amplifies the network's error.** The network's output differs from the exact one by an
error $\delta$ that does not shrink with $\alpha_t$. The conversion multiplies it by $\kappa_t/\alpha_t$, so the first
Euler step from $t_0$ adds to the state the error

$$
h\,\frac{\kappa_{t_0}}{\alpha_{t_0}}\,\delta, \qquad \text{on the linear path } \frac{h}{t_0}\,\delta .
$$

Ten uniform steps from $t_0 = 0.001$ give $h / t_0 = 0.0999 / 0.001 \approx 100$; from $t_0 = 0.05$,
$0.095 / 0.05 = 1.9$. With fewer steps the first step is longer: from 0.05 the factor is 4.75 at 4 NFE and 19 at
1 NFE. On GVP, $\kappa_t/\alpha_t = \frac{\pi/2}{\sin(\pi t/2)} \approx 1/t$ for small $t$, so the amplification is
nearly the same (20.02 against 20 at $t_0 = 0.05$). The factor shrinks as $\alpha_t$ grows, so the first steps amplify
the network's errors most. Intuitively, near $t = 0$ the state is almost pure noise, and its information about $z$
sits in a part of size $\alpha_t$. The scaled score must be accurate to that size, and an error beyond it becomes a
large velocity error.

**The price of a late start.** Starting late trades one error for another. From $t_0$ the sampler should start at
$p_{t_0}$, the law of $\alpha_{t_0} z + \beta_{t_0}\epsilon$; it starts at $\mathcal N(0, I)$ instead. On the linear
path at $t_0 = 0.05$, that start has a noise part 5% too wide and lacks the signal $0.05\,z$. On GVP the noise part is
0.3% too wide and the missing signal is $0.078\,z$. A diffusion model whose forward noising process runs for a finite
time makes this error by construction, since that process reaches its Gaussian prior only approximately. The Diffusion
Guide lists that mismatch among the errors of sampling, beside training and discretization error. A flow-matching path
avoids it by starting at $t = 0$, which the scaled score cannot do.

The FM Guide offers another remedy: keep $t_0 = 0$ and replace the conversion at $t = 0$ by its limit
$\dot\alpha_0\,D_0(x) + \dot\beta_0\,x$. That limit needs $D_0 = \mathbb E[z]$, which a scaled-score network does not
output. At 0.05 the price of the late start was not measurable: velocity and data predictions, which could start at 0,
do as well from 0.05.

**In the code.** `sampler.start_time` sets $t_0$ for every solver. {py:meth}`~flow.Sampler.times` resolves a null
start: 0, or `flow.SCALED_SCORE_START` (0.05) when the prediction's {py:attr}`~flow.Prediction.singular_time` is 0.

**What we found** (phase 6; the start of the scaled score also on held-out episodes)

- **start 0.05 = 0.01 = 0 for velocity and data predictions.** Only the scaled score is sensitive to the start.
- **start 0.05 = 0.01 >> 0.001 for a scaled score**, on selection episodes and again on held-out episodes with
  fresh training seeds. From 0.001 the scaled score is the worst prediction; from 0.05 it is the best, which made it
  the default ({doc}`training`). Values between and around 0.01 and 0.05 were not tried.
- **scaled score from 0.05 > velocity at 4 NFE on the linear path.** Its lead over the velocity policy holds with
  fewer steps.
- **scaled score from 0.05 > velocity at 4 NFE on GVP (weak).** Each is compared with the velocity policy on its own
  path.
- **scaled score from 0.05 at 4 NFE > linear velocity at 10 NFE (weak).** 4 NFE was not scored on held-out episodes.
- **scaled score from 0.05: NFE 4 >> NFE 1.** At one step it fails on both paths.

Try: `uv run evaluation.py -m runs=runs/default 'sampler.start_time=0.05,0.01,0.001'`

## Guidance

How can a sampler follow the observation more closely than the plain policy does? In theory it need not: the flow
model already samples $p_{\mathrm{data}}(\cdot \mid y)$. In practice a trained model can follow $y$ too loosely, and
guidance reinforces it.

**The classifier split.** In this section we write $y$ explicitly. The **guided** field $u_t^{\text{target}}(x \mid y)$
is the marginal vector field of $p_{\mathrm{data}}(\cdot \mid y)$, not the conditional field
$u_t^{\text{target}}(x \mid z)$. For a Gaussian path, Bayes' rule splits it into an unguided field plus a classifier
term. Write $p_t(x \mid \varnothing)$ for the path of chunks pooled over all observations, and
$u_t^{\text{target}}(x \mid \varnothing)$ for its vector field. Write $p_t(y \mid x)$ for the density of the
observation given the noisy chunk, a classifier of noisy chunks. For $t \in (0, 1)$,

$$
u_t^{\text{target}}(x \mid y)
\overset{(i)}{=} a_t\,\nabla\log p_t(x \mid y) + b_t\,x
\overset{(ii)}{=} a_t\,\nabla\log p_t(x \mid \varnothing) + b_t\,x + a_t\,\nabla\log p_t(y \mid x)
\overset{(iii)}{=} u_t^{\text{target}}(x \mid \varnothing) + a_t\,\nabla\log p_t(y \mid x),
$$

where (i) is the conversion formula at fixed $y$, (ii) uses Bayes' rule
$p_t(x \mid y) = p_t(x \mid \varnothing)\,p_t(y \mid x)/p(y)$, whose last factor does not depend on $x$, and (iii) is
the conversion formula for $\varnothing$. Classifier guidance scales the classifier term by a **guidance scale** $w$.
The classifier need not be trained: the same split gives
$a_t\nabla\log p_t(y \mid x) = u_t^{\text{target}}(x \mid y) - u_t^{\text{target}}(x \mid \varnothing)$, and
substituting it leaves only the two vector fields.

:::{admonition} Definition: the classifier-free guided vector field
:class: important

For a guidance scale $w$, the **classifier-free guided vector field** is

$$
\tilde u_t(x \mid y) = (1 - w)\,u_t^{\text{target}}(x \mid \varnothing) + w\,u_t^{\text{target}}(x \mid y)
= u_t^{\text{target}}(x \mid \varnothing) + w\big(u_t^{\text{target}}(x \mid y) - u_t^{\text{target}}(x \mid \varnothing)\big).
$$

The sampler simulates its network version $\tilde u_t^\theta(x \mid y)$, with $u_t^\theta$ in place of
$u_t^{\text{target}}$. At $w = 1$ it is the guided field itself.
:::

Intuitively, the difference $u_t^{\text{target}}(x \mid y) - u_t^{\text{target}}(x \mid \varnothing)$ is the direction
the observation adds. $w = 0$ ignores the observation, $w = 1$ follows it as the data do, and $w > 1$ goes past the
guided field, further along that direction. For $w \ne 1$ this is a heuristic:
$\tilde u_t(x \mid y) \ne u_t^{\text{target}}(x \mid y)$, so $X_1$ no longer follows $p_{\mathrm{data}}(\cdot \mid y)$.
Which distribution it does follow is not known, as the FM Guide notes. The definition itself needs no Gaussian path; the
classifier split only motivates it.

**One network for both fields.** The unguided field comes from the same network, which sees the null label
$\varnothing$ in place of $y$. Training replaces $y$ by $\varnothing$ with probability $p_{\mathrm{uncond}}$
(`model.cond_dropout`), as in [one training step](training.md#one-training-step). A policy trained without it has no
unguided field. {doc}`../model` describes the null encodings.

**Mixing outputs.** The code mixes the network's outputs, not the vector fields. The two are the same:

:::{admonition} Proposition: mixing outputs mixes vector fields
:class: important

Let $u = A_t x + B_t o$ be the conversion of [the flow model](#the-flow-model), with $A_t$ and $B_t$ independent of
$y$. Let $o_\varnothing = o_t^\theta(x \mid \varnothing)$ and $o_y = o_t^\theta(x \mid y)$ be the outputs of the two
passes, at the same $x$ and $t$. Then

$$
(1 - w)\big(A_t x + B_t\,o_\varnothing\big) + w\big(A_t x + B_t\,o_y\big)
= A_t x + B_t\big((1 - w)\,o_\varnothing + w\,o_y\big).
$$
:::

In other words, since the weights $1 - w$ and $w$ sum to 1, the shared term $A_t x$ passes through the mix unchanged.
Converting the mixed output therefore gives $\tilde u_t^\theta(x \mid y)$. The score that the SDE uses is affine in
$u_t^\theta$ in the same way, so under the SDE guidance mixes the two implied scores as well. For $w \ne 1$ that mixture
is in general not the score of the guided ODE's marginals. The SDE then inherits the heuristic, not the extension
trick's guarantee.

**Under MeanFlow's step.** The `mean` solver applies no conversion. Its outputs are average velocities, and the guided
step $x + (s - t)\big[(1 - w)\,\bar u^\theta_{t,s}(x \mid \varnothing) + w\,\bar u^\theta_{t,s}(x \mid y)\big]$ mixes
two jumps. For exact average velocities it equals $(1 - w)\,\psi^{\varnothing}_{t \to s}(x) + w\,\psi^{y}_{t \to s}(x)$,
where $\psi^{\varnothing}$ and $\psi^{y}$ are the flow maps of the unguided and the guided field. The flow map of
$\tilde u_t$ is a different map. Each flow map expands as

$$
\psi_{t \to s}(x) = x + (s - t)\,u_t + \frac{(s - t)^2}{2}\big(\partial_t u_t + \partial_x u_t\,u_t\big) + O\big((s - t)^3\big),
$$

with the field and its derivatives at $x$. The two therefore agree to first order in $s - t$. At second order, the flow
map of $\tilde u_t$ minus the mix is

$$
\frac{(s - t)^2}{2}\,w(1 - w)\,\big(\partial_x u_t^{\varnothing} - \partial_x u_t^{y}\big)\big(u_t^{y} - u_t^{\varnothing}\big),
$$

where $u_t^{y}$ and $u_t^{\varnothing}$ are the guided and the unguided field at $x$, and $\partial_x$ takes their
Jacobians. In general this term is nonzero when $w$ is neither 0 nor 1. The Euler method, Heun's method and the SDE
evaluate the network at $s = t$, so for them mixing outputs is exact.

**The cost.** One guided evaluation stacks $y$ and $\varnothing$ into one batched pass and counts as one NFE. NFE
counts sequential network calls, the cost that sets the latency. A guided call doubles the batch, so at equal NFE it
costs about twice the compute of an unguided one.

**In the code.** `sampler.cfg_scale` is $w$, and {py:meth}`~model.FlowMatchingPolicy.predictor` returns the mixed
output. Guidance needs a policy trained with `model.cond_dropout` > 0: {py:func}`~train.check_config` refuses
$w \ne 1$ otherwise, and `evaluation.py` skips such runs.

**What we found** (phase 2, nano-PushT: Adam, $\mathcal L_{\text{CFM}}$ (`velocity_mse`), E = 8, `hold` padding,
raw weights)

- **w = 1 = w ∈ {1.1, 1.25, 1.5}.** Once the scale is chosen on separate episodes, guidance moves success by less
  than the noise, and condition dropout itself changes nothing at w = 1. The default trains without dropout and
  samples at w = 1.

Reproduce (the preset pins phase 2's settings: Adam, `velocity_mse`, E = 8, `hold` padding, raw weights first):

```bash
uv run train.py -m experiment=conditioning
uv run evaluation.py -m runs=runs/conditioning split=selection episodes=200 weights=raw \
    'sampler.cfg_scale=1.1,1.25,1.5'
```

:::{admonition} Summary
:class: note

The trained network defines a flow model, the ODE of $u_t^\theta(\cdot \mid y)$ from $X_{t_0} \sim \mathcal N(0, I)$,
and a sampler approximates its flow map $\psi^\theta_{t_0 \to 1}$ with a fixed NFE. The Euler method is a first-order
Taylor step, exact only on trajectories of constant velocity. On the linear path it is DDIM, and for the exact field one
step from $t = 0$ returns the mean chunk $\mathbb E[z]$. A grid is a change of clock: free in theory, it places the
discretization error in practice. For the exact field and a start from $p_{t_0}$, any $\sigma_t$ gives an SDE with the
flow's marginals. The code takes $\sigma_t^2 = 2\eta\,\alpha_t\beta_t$, and the errors decide $\eta$. The scaled score's
singularity at $t = 0$ is removable in theory and amplifies the network's error in practice, so it starts at 0.05.
Guidance with $w \ne 1$ is a heuristic that leaves $p_{\mathrm{data}}(\cdot \mid y)$ on purpose.
:::

## Implementation

{py:class}`~flow.Sampler` is a frozen dataclass that Hydra builds from `conf/sampler/default.yaml` (`sampler=mean`
selects MeanFlow's, `conf/sampler/mean.yaml`). It holds the solver, the NFE, the time grid, the start time, the SDE's
noise scale and the guidance scale; the {doc}`API page </api/flow>` lists the fields. The same config group sits in
`conf/config.yaml`, where it drives the periodic evaluations of training, and in `conf/evaluation.yaml`, where it
replaces each run's own sampler. It never touches the weights: changing it changes the evaluations and their logged
curves only.

One policy query runs this chain:

1. {py:func}`~evaluation.query_policy` draws one initial noise chunk $X_{t_0}$ per episode (H steps of the 2-D
   action, in the normalized relative space), plus the Brownian draws if the solver needs them, and calls
   {py:meth}`~model.FlowMatchingPolicy.sample_actions`.
2. `sample_actions` encodes the guiding variable $y$, the history frames and the goal, once with
   {py:meth}`~model.FlowMatchingPolicy.encode_condition`, since $y$ does not change along the trajectory. It wraps the
   network for that $y$ with {py:meth}`~model.FlowMatchingPolicy.predictor`, which mixes in the null pass when
   `sampler.cfg_scale` is not 1. It then builds a {py:class}`~flow.LearnedFlow` from the wrapped network, the run's
   path and the run's prediction.
3. {py:meth}`~flow.Sampler.sample` resolves the grid with {py:meth}`~flow.Sampler.times` and simulates from the noise
   $X_{t_0}$ to $X_1$ with a {py:class}`~flow.Simulator`.
4.  Every Euler, Heun or SDE step calls {py:meth}`~flow.LearnedFlow.drift`. It evaluates the network at $(x, t)$ and
   maps the output to the vector field $u_t^\theta(x \mid y)$, or to $\tilde u_t^\theta(x \mid y)$ when $w \ne 1$, with
   {py:meth}`~flow.Prediction.to_velocity` and the noise schedulers at that grid time. The `mean` solver calls
   {py:meth}`~flow.LearnedFlow.average` instead, which returns the average velocity unconverted.

The solvers are {py:class}`~flow.Simulator` subclasses: {py:class}`~flow.Euler`, {py:class}`~flow.Heun`,
{py:class}`~flow.EulerMaruyama` and {py:class}`~flow.MeanFlowStep`. {py:meth}`~flow.Simulator.simulate` calls their
{py:meth}`~flow.Simulator.step` along the grid, and {py:attr}`~flow.Simulator.nfe_per_step` gives each one's NFE per
step. Under `sampler.solver=sde`, {py:meth}`~flow.Sampler.sample` runs {py:class}`~flow.EulerMaruyama` on a
{py:class}`~flow.LangevinFlow` up to `sde_end` and then one {py:class}`~flow.Euler` step on the flow itself.
{py:attr}`~flow.Prediction.singular_time` records where an output's conversion is undefined.

The sampler refuses what cannot run, before any network call:

| setting | refused because | checked in |
| --- | --- | --- |
| `heun` with an odd NFE | Heun's method takes 2 evaluations per step | `Sampler.__post_init__` |
| `sde` with fewer than 2 NFE | it needs a stochastic step and the final Euler step | `Sampler.__post_init__` |
| `sde` on the `data_dense` grid | the SDE builds its own equal-step grid | `Sampler.__post_init__` |
| `sde_end` outside $(0, 1)$, or a negative `sde_eta` | the final step needs $0 <$ `sde_end` $< 1$, and $\sigma_t^2$ needs $\eta \ge 0$ | `Sampler.__post_init__` |
| `data_dense` with $\rho < 1$ | its steps would grow toward the data | `Sampler.__post_init__` |
| a grid not strictly increasing in float32 | its steps would not move forward | {py:meth}`~flow.Sampler.times` |
| an output evaluated at its singular time: Heun for a data prediction, `sampler.start_time=0` for a scaled score | the conversion divides by zero there | {py:meth}`~flow.Sampler.times` |
| `mean` for a policy not trained with {py:class}`~flow.MeanFlow` | the policy has no average velocity | {py:meth}`~flow.Sampler.check` |
| $w \ne 1$ for a policy trained without condition dropout | the policy has no unguided field | {py:func}`~train.check_config` |

{py:meth}`~flow.Sampler.check` checks the grid for the run's objective and the `mean` solver.
{py:func}`~train.check_config` calls it before training starts, and {py:func}`~evaluation.score_run` skips the runs it
refuses.

NFE counts every network evaluation, both stages of a Heun step and the SDE's deterministic last step included.
{py:class}`~flow.LearnedFlow` counts its calls, and {py:meth}`~flow.Sampler.sample` raises if the count differs from
{py:attr}`~flow.Sampler.nfe`. A non-finite value at any step stays non-finite, so `sample` checks the final chunk and
raises on a non-finite one.

**The noise streams.** {py:func}`~evaluation.evaluate_policy` gives every episode two noise generators of its own,
both seeded by its env seed. {py:func}`~evaluation.query_policy` draws each query's initial noise from the first. From
the second it draws {py:attr}`~flow.Sampler.num_brownian` Brownian draws $\xi_i$ of the chunk's shape, NFE − 1 under
the SDE and none otherwise.

Reproduce: train the phase-6 policies with `uv run train.py -m experiment=flow_matching` and the preset's data-output
command, then score them with the `evaluation.py` command in the comments of `conf/experiment/flow_matching.yaml`,
changing the sampler options (the data-output runs add `sampler.sde_end=0.99` under the SDE). The preset pins
AdamW; pass `sampler.start_time=0.001` to sample a scaled score from the study's start.
