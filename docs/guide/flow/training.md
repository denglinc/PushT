# Training objective

How can a network learn the marginal vector field when nobody can compute it? The field of {doc}`index` averages the
conditional vector field over every chunk that could have produced the noisy chunk, an integral over the whole dataset.
Training avoids the integral by regression. It pairs each clean chunk $z$ with fresh noise $\epsilon$, builds
$x = \alpha_t z + \beta_t \epsilon$, and regresses the network onto a target that $z$ and $\epsilon$ determine. No
ODE is simulated along the way: as the notes say, flow matching training is **simulation-free**.

The objective, {py:class}`~flow.FlowMatching` in `conf/flow/default.yaml`, has four parts. The path of {doc}`index`
sets how a chunk is noised. The flow-time distribution sets which times the loss looks at. The recipe sets what the
network outputs and how its error is weighted. The reduction sets how the errors within a chunk add up. The first
section shows why the regression works, and the second writes out one training step. Each of the last three parts
then has its own section. The parts are Hydra options:

```bash
uv run train.py flow.recipe=velocity_mse                    # regress the velocity instead of the scaled score
uv run train.py flow/time=logit_normal flow.time.mean=0.5   # more flow times near the data
uv run train.py flow.reduction=example_mean                 # average each chunk's error over its valid steps
```

## Regression learns a posterior mean

Why does a regression onto a target built from $(z, \epsilon)$ teach a network that never sees $z$ the marginal vector
field? Training sees many guiding variables at once, so on this page we write $y$ back into the notation. We write
$u_t^{\text{target}}(x \mid y)$ for the marginal vector field of {doc}`index` with the data distribution
$p_{\mathrm{data}}(\cdot \mid y)$, and $u_t^{\text{target}}(x \mid \varnothing)$ for the one of the null label. We want
the network's vector field $u_t^\theta(x \mid y)$ to equal $u_t^{\text{target}}(x \mid y)$ for every $y$. Until
[One training step](#one-training-step) defines it for every output, take the network to output this field directly.

The direct way is a least-squares regression onto the marginal field, the guided **flow matching loss**

$$
\mathcal L_{\text{FM}}(\theta) = \mathbb E_{t \sim \mathrm{Unif},\, (z, y) \sim p_{\mathrm{data}},\, x \sim p_t(\cdot \mid z)}
\big\lVert u_t^\theta(x \mid y) - u_t^{\text{target}}(x \mid y) \big\rVert^2 .
$$

It cannot be computed, because its target is the integral over the dataset. The conditional vector field of the one
chunk that did produce $x$, however, is known in closed form. The guided **conditional flow matching loss** regresses
onto it instead:

$$
\mathcal L_{\text{CFM}}(\theta) = \mathbb E_{t \sim \mathrm{Unif},\, (z, y) \sim p_{\mathrm{data}},\, \epsilon \sim \mathcal N(0, I)}
\big\lVert u_t^\theta(x \mid y) - u_t^{\text{target}}(x \mid z) \big\rVert^2,
\qquad x = \alpha_t z + \beta_t \epsilon .
$$

The two losses differ only by a constant. The reason does not depend on the vector field: it holds for any target
computed from the pair $(z, \epsilon)$. We write $o_t^\theta(x \mid y)$ for the network's raw output, whatever the
recipe makes it parameterize.

:::{admonition} Theorem: regression learns a posterior mean
:class: important

Draw $(z, y) \sim p_{\mathrm{data}}$, a flow time $t$ from any law on $[0, 1]$ and noise
$\epsilon \sim \mathcal N(0, I)$, all three independently, and set $x = \alpha_t z + \beta_t \epsilon$. Let
$o_t^{\text{target}}$ be a target computed from $(t, z, \epsilon)$, such as $u_t^{\text{target}}(x \mid z)$, $z$ or
$-\epsilon$, with $\mathbb E\lVert o_t^{\text{target}}\rVert^2 < \infty$. For every output $o_t^\theta(x \mid y)$ that
depends on $x$, $t$ and $y$ only,

$$
\mathbb E\,\big\lVert o_t^\theta(x \mid y) - o_t^{\text{target}} \big\rVert^2
= \mathbb E\,\big\lVert o_t^\theta(x \mid y) - \mathbb E[o_t^{\text{target}} \mid x, t, y] \big\rVert^2
+ \mathbb E\,\mathrm{tr}\,\mathrm{Cov}(o_t^{\text{target}} \mid x, t, y) .
$$

The last term does not depend on $\theta$, so a regression onto $o_t^{\text{target}}$ has the gradient of a regression
onto its posterior mean. Over all functions of $(x, t, y)$, that is for an expressive enough network, the minimizer is
$o_t^\theta(x \mid y) = \mathbb E[o_t^{\text{target}} \mid x, t, y]$ at every time the law of $t$ draws. For the three
targets this posterior mean is the guided vector field $u_t^{\text{target}}(x \mid y)$, the denoiser $D_t(x)$ and the
scaled score $-\mathbb E[\epsilon \mid x]$ of {doc}`index`, for the data distribution $p_{\mathrm{data}}(\cdot \mid y)$.
:::

Intuitively, the same noisy chunk $x$ arises from many pairs $(z, \epsilon)$. Each training example shows the network
one of them, with its target. The least-squares compromise is the average of these targets, weighted by how likely each
pair is to have produced $x$: the posterior mean. For the vector field, the first term on the right is
$\mathcal L_{\text{FM}}$ and the left side is $\mathcal L_{\text{CFM}}$. This is Holderrieth and Erives' theorem
$\mathcal L_{\text{FM}} = \mathcal L_{\text{CFM}} + C$, with $C = -\mathbb E\,\mathrm{tr}\,\mathrm{Cov} \le 0$.

The proof is a bias-variance split at a fixed $(x, t, y)$. Write $o^\star = \mathbb E[o_t^{\text{target}} \mid x, t, y]$
and $o = o_t^\theta(x \mid y)$; both are fixed once $(x, t, y)$ is. Then

$$
\begin{aligned}
\mathbb E\big[\lVert o - o_t^{\text{target}}\rVert^2 \,\big|\, x, t, y\big]
&\overset{(i)}{=} \lVert o - o^\star\rVert^2 + 2\,(o - o^\star)^\top\, \mathbb E[o^\star - o_t^{\text{target}} \mid x, t, y]
+ \mathbb E\big[\lVert o^\star - o_t^{\text{target}}\rVert^2 \,\big|\, x, t, y\big] \\
&\overset{(ii)}{=} \lVert o - o^\star\rVert^2 + \mathrm{tr}\,\mathrm{Cov}(o_t^{\text{target}} \mid x, t, y),
\end{aligned}
$$

where (i) writes $o - o_t^{\text{target}} = (o - o^\star) + (o^\star - o_t^{\text{target}})$ and expands the square,
and (ii) uses $\mathbb E[o_t^{\text{target}} \mid x, t, y] = o^\star$, which removes the cross term and leaves the
conditional variance. Averaging over $(x, t, y)$ gives the theorem. The FM Guide of Lipman et al. proves that the
gradients agree for every Bregman divergence, of which the squared error is one.

Two consequences are used on the rest of this page.

- **The minimizer is chosen time by time.** Over all functions, the output at one time does not enter the loss at
  another. Therefore, at every time the flow-time distribution draws, neither the distribution nor a positive weight
  on the times changes the minimizer; they change only how much each time counts
  ([Flow-time distributions](#flow-time-distributions), [Recipes](#recipes)). At a time it never draws, the loss says
  nothing about the network.
-  **The loss has a floor.** At the minimizer the loss equals
  $\mathbb E\,\mathrm{tr}\,\mathrm{Cov}(o_t^{\text{target}} \mid x, t, y)$, the part of the target that $x$, $t$ and $y$
  leave open. It is positive whenever several chunks fit one noisy chunk and one $y$, which is the case a generative
  policy exists for. As the notes put it, "the absolute value of the loss is meaningless" ([Comparing training
  losses](#comparing-training-losses)).

:::{admonition} Example: the floor of the conditional flow matching loss
:class: tip

At $t = 0$ the noisy chunk is the noise, $x = \epsilon$, and the target is
$u_0^{\text{target}}(x \mid z) = \dot\alpha_0 z + \dot\beta_0 \epsilon$. Given $x$ and $y$, only $z$ is random, so the
minimizer is $\dot\alpha_0\,\mathbb E[z \mid y] + \dot\beta_0 x$ and the floor is
$\dot\alpha_0^2\, \mathrm{tr}\,\mathrm{Cov}(z \mid y)$. On the linear path the minimizer is $\mathbb E[z \mid y] - x$
and the floor is $\mathrm{tr}\,\mathrm{Cov}(z \mid y)$. On GVP, $\dot\alpha_0 = \pi/2$ and $\dot\beta_0 = 0$, so the
floor is $(\pi/2)^2 \approx 2.47$ times larger for the same data.

On the [Gaussian data](index.md#an-example-with-gaussian-data), where each coordinate of $z$ has the variance $s^2$, the
floor has a closed form at every $t$. Given $x$, $z$ has the variance $s^2\beta_t^2/v_t$ per coordinate, with
$v_t = \alpha_t^2 s^2 + \beta_t^2$. For $t < 1$ the target
$u_t^{\text{target}}(x \mid z) = (\dot\beta_t x + \kappa_t z)/\beta_t$ varies with $z$ alone, so

$$
\mathrm{Var}\big(u_t^{\text{target}}(x \mid z) \,\big|\, x\big)
= \frac{\kappa_t^2}{\beta_t^2}\cdot\frac{s^2 \beta_t^2}{v_t} = \frac{\kappa_t^2\, s^2}{v_t}
\quad \text{per coordinate.}
$$

At $t = 1$ on the linear path this is $1$: the noisy chunk is $z$, and the target $z - \epsilon$ keeps all of the
noise.
:::

## One training step

How does one step of training estimate such a loss? The step draws everything the expectation averages over, for a
batch. It draws the flow time from a **flow-time distribution** with density $p_{\mathrm{time}}$ on $[0, 1]$, uniform
by default ([below](#flow-time-distributions)). It also drops the guiding variable: with probability
$p_{\mathrm{uncond}}$ it replaces $y$ by the null label $\varnothing$.

The output $o_t^\theta(x \mid y)$ has its **conditional target** $o_t^{\text{target}}$, the target of the theorem above,
computed from $(z, \epsilon)$: $u_t^{\text{target}}(x \mid z)$, $z$ or $-\epsilon$. The **network's vector field**
$u_t^\theta(x \mid y)$ is the output converted by the affine maps of [the four
parameterizations](index.md#four-parameterizations). The **error** $e$ compares the output with its target in one of two
spaces:

$$
\begin{aligned}
e &= o_t^\theta(x \mid y) - o_t^{\text{target}}
&& \blacktriangleright\ \text{in the output's own space} \\
e &= u_t^\theta(x \mid y) - u_t^{\text{target}}(x \mid z)
&& \blacktriangleright\ \text{in velocity space}
\end{aligned}
$$

We call $u_t^\theta(x \mid y) - u_t^{\text{target}}(x \mid z)$ the **velocity error**, whatever space the loss compares
in. A **reduction** turns the chunk's error into one number $\ell(e)$. Under the default `sum` reduction and with no
step of the chunk masked, $\ell(e) = \lVert e\rVert^2$. [Reduction](#reduction) gives the general form and
`example_mean`.

:::{admonition} Algorithm: one training step
:class: important

For each mini-batch of pairs $(z, y)$ from the dataset, and for each pair:

1. Sample a flow time $t \sim p_{\mathrm{time}}$.
2. Sample noise $\epsilon \sim \mathcal N(0, I)$ of the chunk's shape.
3. Set $x = \alpha_t z + \beta_t \epsilon$.
4. With probability $p_{\mathrm{uncond}}$, drop the guiding variable: $y \leftarrow \varnothing$.
5. Compute the output $o_t^\theta(x \mid y)$ and the error $e$.
6. Reduce the chunk's error to the number $\ell(e)$.

Then average $\ell(e)$ over the batch and update $\theta$ by a gradient step on that average.
:::

The step estimates the objective, written in the notes' form for classifier-free guidance training:

$$
\begin{aligned}
\mathcal L(\theta) &= \mathbb E_{\square}\,\big[\ell(e)\big], \\
\square &= (z, y) \sim p_{\mathrm{data}},\ t \sim p_{\mathrm{time}},\ \epsilon \sim \mathcal N(0, I), \\
&\phantom{{}={}} x = \alpha_t z + \beta_t \epsilon,\ y \leftarrow \varnothing \text{ with probability } p_{\mathrm{uncond}} .
\end{aligned}
$$

With $t$ uniform on $[0, 1]$, the sum reduction and no step masked, $\mathcal L(\theta)$ is a loss of the references. It
is the notes' $\mathcal L_{\text{CFM}}$ for `velocity_mse`, their $\mathcal L_{\text{DDPM}}$ for `scaled_score_mse` and
the FM Guide's $x_1$-prediction loss for `data_mse`, each in its guided form ([Recipes](#recipes) explains the last
name). When $p_{\mathrm{uncond}} > 0$, each also drops $y$ as the notes' $\mathcal L_{\text{CFM}}^{\text{CFG}}$ does.
`data_output_velocity_loss` cannot draw $t$ on all of $[0, 1]$ ([Endpoints](#endpoints)), so it is
$\mathcal L_{\text{CFM}}$ with $t$ uniform on its support. In general, $\mathcal L(\theta)$ is one of these losses with
$t$ drawn from $p_{\mathrm{time}}$, the chunk's error reduced by $\ell$ and the padded steps masked. We call
`velocity_mse` the **velocity loss** and `data_mse` the **data loss**, as the findings do.

Why drop $y$? The dataset never contains $y = \varnothing$, so training has to introduce it artificially, as the notes
explain. The theorem above holds with $y$ read as the value the network sees. Rows whose $y$ was dropped therefore
regress onto the unguided field $u_t^{\text{target}}(x \mid \varnothing)$, which classifier-free guidance mixes with the
guided one at sampling ([Guidance](sampling.md#guidance)). With $p_{\mathrm{uncond}} = 0$, the default, step 4 does
nothing.

**In the code.** {py:meth}`~model.FlowMatchingPolicy.compute_loss` runs the step for a batch. The policy's
{py:meth}`~model.FlowMatchingPolicy.condition` encodes $y$ and replaces the encoding by the learned null encoding of
$\varnothing$ with probability $p_{\mathrm{uncond}}$ = `model.cond_dropout`. `compute_loss` then calls
{py:meth}`~flow.FlowMatching.loss` with the network bound to the encoding, as `predict(x, t)`, together with the clean
normalized chunks and their loss weights. The loss draws one flow time per chunk with
{py:meth}`~flow.TimeSampler.sample`, shared by every action of the chunk, and fresh noise of the chunk's shape. It
builds $x$ with {py:meth}`~flow.Coefficients.interpolate`, calls `predict` once and casts the output to float32. The
error is the output minus {py:meth}`~flow.Prediction.target`, or, in velocity space,
{py:meth}`~flow.Prediction.to_velocity` minus {py:meth}`~flow.Coefficients.velocity`.
{py:meth}`~flow.FlowMatching.per_example` reduces each chunk's error, and the batch mean is the loss.

The loss runs inside the captured training step of {doc}`../training`, compiled by TorchInductor when `train.compile`
is on ({py:func}`~train.build_loss`). It therefore has no host syncs and no control flow on tensor values: the number
of draws is fixed, and every rule is checked when the objective is built. The keyword arguments `flow_time` and
`noise` fix the draws, which the tests use.

## Flow-time distributions

Which flow times should the loss look at? The notes draw $t$ uniformly, but the theorem holds for any law of $t$. It
turns out that the choice of law is a choice of weighting.

:::{admonition} Proposition: a time density is a time weighting
:class: important

Let $\mathcal L_t(\theta)$ be the expected loss at the flow time $t$, averaged over the pairs, the noise and the drop
of $y$. For a flow-time density $p_{\mathrm{time}}$ on $[0, 1]$,

$$
\mathbb E_{t \sim p_{\mathrm{time}}}\big[\mathcal L_t(\theta)\big]
= \int_0^1 p_{\mathrm{time}}(t)\, \mathcal L_t(\theta)\, \mathrm dt
= \mathbb E_{t \sim \mathrm{Unif}[0, 1]}\big[p_{\mathrm{time}}(t)\, \mathcal L_t(\theta)\big] .
$$

At every $t$ with $p_{\mathrm{time}}(t) > 0$ the minimizer is still the posterior mean. Where
$p_{\mathrm{time}}(t) = 0$, the loss says nothing about the network.
:::

In other words, drawing a time more often is the same as counting it more. In theory, then, every positive density
learns the same vector field. In practice the network is finite, and $p_{\mathrm{time}}$ decides where its error is
small. Drawing and weighting also differ in practice: they have the same expected loss but not the same gradient
noise. Drawing spends the batch on the times that count, while weighting draws times that it then nearly ignores.
The FM Guide reports, from Stable Diffusion 3, that drawing trains better at scale.

A recipe weights the times too, by a factor $\lambda_t$ on the velocity error ([Recipes](#recipes)). An objective
therefore counts the velocity error at time $t$ with the weight $p_{\mathrm{time}}(t)\,\lambda_t$. The default, uniform
times with `scaled_score_mse` on the linear path, counts it with $t^2$.

The sampler's time grid spreads effort along the same axis, against a different error ([Time
grids](sampling.md#time-grids)). The grid spends network evaluations where the discretization error is large. The
time density spends network capacity where the training error matters. The two are independent choices, but a grid
must stay where the network was trained ([Endpoints](#endpoints)).

Two distributions ship, as `conf/flow/time/uniform.yaml` (the default, the notes' choice) and
`conf/flow/time/logit_normal.yaml`. Each is given by its inverse CDF $F^{-1}$, applied to one uniform draw per example
([Drawing on a support](#drawing-on-a-support)):

- {py:class}`~flow.UniformTime`: $F^{-1}(u) = t_{\min} + (t_{\max} - t_{\min})\,u$ on the support
  $[t_{\min}, t_{\max}]$ (`flow.time.support`).
-  {py:class}`~flow.LogitNormalTime`: $\mathrm{logit}\,t \sim \mathcal N(\mu, \varsigma^2)$ on all of $[0, 1]$, with
  mean $\mu$ (`flow.time.mean`) and standard deviation $\varsigma$ (`flow.time.std`). With $\Phi$ the standard normal
  CDF, $F(t) = \Phi\big((\mathrm{logit}\,t - \mu)/\varsigma\big)$ and
  $F^{-1}(u) = \mathrm{sigmoid}\big(\mu + \varsigma\,\Phi^{-1}(u)\big)$. A positive mean $\mu$ puts more times near
  the data, a negative one near the noise. Its default, $(\mu, \varsigma) = (0, 1)$, is Stable Diffusion 3's choice.
  Stable Diffusion 3 runs time the other way, with the noise at $t = 1$. Since
  $\mathrm{logit}(1 - t) = -\mathrm{logit}\,t$, its logit-normal with mean $\mu_{\text{SD3}}$ is ours with
  $\mu = -\mu_{\text{SD3}}$.

π0 draws its flow times from a Beta distribution that puts more of them near the noise. It does not ship, but the
findings below name its direction.

**What we found** (linear path, velocity loss, AdamW)

- **logit-normal(+0.5) = uniform.** Moving the training times toward the data is not resolved either way.
- **uniform > logit-normal(−0.5).** Moving training weight toward the noise, the direction of π0's Beta
  distribution, hurts.
- **uniform = logit-normal(0), although logit-normal(0) puts much less mass near $t = 1$.** The runs do not show that
  the end near the data carries the success.

Reproduce: `uv run train.py -m experiment=flow_matching flow/path=linear flow.recipe=velocity_mse flow/time=logit_normal 'flow.time.mean=-0.5,0.0,0.5'`

### Drawing on a support

How does the code draw a flow time at a fixed cost, inside a captured training step? A
{py:class}`~flow.TimeSampler` is a distribution on a support $[t_{\min}, t_{\max}] \subseteq [0, 1]$
(`flow.time.support`), given by its inverse CDF $F^{-1}$. Each example gets one uniform $u \in [0, 1)$ from
`torch.rand`, and its time is $t = F^{-1}(u)$.

:::{admonition} Proposition: inverse-transform sampling
:class: important

Let $F$ be a continuous, increasing CDF and let $U$ be uniform on $[0, 1)$. Then $t = F^{-1}(U)$ has the CDF $F$:

$$
P(t \le r) = F(r).
$$
:::

The derivation takes two steps:

$$
P(t \le r) \overset{(i)}{=} P\big(U \le F(r)\big) \overset{(ii)}{=} F(r),
$$

where (i) applies the increasing $F$ to both sides of $F^{-1}(U) \le r$, and (ii) uses that $U$ is uniform. For the
uniform law on the support, $F^{-1}(u) = t_{\min} + (t_{\max} - t_{\min})\,u$ scales the draw into the support. For any
other $F$, restricting the law to a narrower support is a separate question, and three ways to do it compare on the
law of the draw and its cost:

| way to restrict to $[t_{\min}, t_{\max}]$ | law of the draw | cost |
| --- | --- | --- |
| redraw until $t$ lands in the support | $F$ conditioned on the support | a loop whose length depends on the draws |
| clamp $t$ to the support | wrong: atoms at $t_{\min}$ and $t_{\max}$ | fixed |
| scale the draw into the support | right for the uniform law only; another law for any other $F$ | fixed |

Only scaling a uniform has both the right law and a fixed cost, which a training step captured as a CUDA graph needs.
Clamping is not harmless. `data_output_velocity_loss` weights the denoising error by $(1 - t)^{-2}$ on the linear path
([Endpoints](#endpoints)): on $[0, 0.99]$ the uniform law gives it an average weight of 100, while clamping uniform
draws from $[0, 1]$ to $0.99$ adds an atom of mass $0.01$ at weight $10^4$ and about doubles it, to 199. So the uniform
time takes any support, and {py:class}`~flow.LogitNormalTime` draws on all of $[0, 1]$ and refuses a narrower one.
The one recipe that needs a narrower support, `data_output_velocity_loss`, therefore draws uniform times. Two more
properties of the draw matter.

- **It can land on an end.** `torch.rand` returns exactly 0 with probability $2^{-24}$ per draw on the CPU and
  $2^{-25}$ on CUDA, and never 1, so a draw can equal $t_{\min}$, and a logit-normal can give $t = 0$. A logit-normal
  wide enough to put $\mu + 5.3\,\varsigma$ above about 16.6 can also round to $t = 1$ in float32. This is harmless:
  no target in the output's own space divides by a path coefficient ([Endpoints](#endpoints)), and
  {py:class}`~flow.FlowMatching` keeps the velocity-space loss off its singular time by a rule on the support
  ([Numerics](index.md#numerics)). Nothing is clamped.
- **Every distribution uses the same random numbers.** Each sampler draws exactly one `torch.rand` per example, so
  runs that differ only in the time law share a seed's noise and batch order, and their times are coupled through $u$:
  at the same seed, a uniform run sees $t = u$ where a logit-normal run sees $F^{-1}(u)$. This is why the
  logit-normal applies $\Phi^{-1}$ to the uniform draw instead of drawing a separate normal. The draws run in
  float32.

## Recipes

What should the network output, and where should the loss compare it? A {py:class}`~flow.Recipe` (`flow.recipe`)
answers both. The output is one of the three parameterizations of {doc}`index` that a network can produce: the vector
field, the denoiser or the scaled score. The loss compares the output with its conditional target, either in the
output's own space or in velocity space, after converting the output to a vector field. The two choices form a grid:

| output | loss in its own space | loss in velocity space |
| --- | --- | --- |
| vector field | `velocity_mse` | `velocity_mse` (the same loss) |
| denoiser | `data_mse` | `data_output_velocity_loss` |
| scaled score (default) | `scaled_score_mse` | not shipped, singular at $t = 0$ ([Endpoints](#endpoints)) |

Remember that `velocity_mse` is the velocity loss and `data_mse` the data loss. The flag
{py:attr}`~flow.Recipe.velocity_loss` is a different thing: it marks a loss compared in velocity space, and only
`data_output_velocity_loss` sets it.

Why compare in velocity space at all? The sampler integrates the vector field, so an error in velocity space is what
moves the sample. Velocity space is therefore the common currency in which the recipes compare. It turns out that every
recipe is the squared velocity error with a weight on the flow times.

:::{admonition} Proposition: every recipe weights the velocity error in time
:class: important

Write the network's vector field as $u_t^\theta(x \mid y) = A_t\, x + B_t\, o_t^\theta(x \mid y)$, with the
coefficients of [the four parameterizations](index.md#four-parameterizations). For every $t \in (0, 1)$ and every
sample, a recipe compared in its own space has the squared error

$$
\big\lVert o_t^\theta(x \mid y) - o_t^{\text{target}} \big\rVert^2
= \lambda_t\, \big\lVert u_t^\theta(x \mid y) - u_t^{\text{target}}(x \mid z) \big\rVert^2,
\qquad \lambda_t = \frac{1}{B_t^2} .
$$

A recipe compared in velocity space has $\lambda_t = 1$.
:::

The proof needs only that each conditional target obeys the conversion of its output. The proposition [one pair of
unknowns, four names](index.md#four-parameterizations) gives
$u_t^{\text{target}}(x \mid z) = A_t\, x + B_t\, o_t^{\text{target}}$ for each of the three targets. Therefore

$$
u_t^\theta(x \mid y) - u_t^{\text{target}}(x \mid z)
\overset{(i)}{=} \big(A_t\, x + B_t\, o_t^\theta(x \mid y)\big) - \big(A_t\, x + B_t\, o_t^{\text{target}}\big)
\overset{(ii)}{=} B_t\, \big(o_t^\theta(x \mid y) - o_t^{\text{target}}\big),
$$

where (i) writes both fields through the same affine map, the network's by conversion and the target's by that
proposition, and (ii) cancels the $x$ terms. Squaring and dividing by $B_t^2 > 0$ gives the proposition. A draw lands exactly
on $t = 0$ or $t = 1$ with probability of order $2^{-24}$ ([Drawing on a support](#drawing-on-a-support)), so the
identity covers almost every training sample.

In other words, a loss in the output's own space forgives a velocity error wherever the conversion's gain $B_t$ is
large. We call $\lambda_t$ the recipe's **time weight**. It is not the Diffusion Guide's $\lambda_t$, which is half the
log-SNR.

:::{admonition} Example: the weights on the linear path
:class: tip

On the linear path, $x = t z + (1 - t)\epsilon$ and $u_t^{\text{target}}(x \mid z) = z - \epsilon$. Solving $x$ for
$\epsilon$ or for $z$ writes the target in two more forms:

$$
z - \epsilon = \frac{z - x}{1 - t} = \frac{x - \epsilon}{t} .
$$

Dropping the arguments $(x \mid y)$, a denoiser output $D_t^\theta$ has the vector field $(D_t^\theta - x)/(1 - t)$,
and a scaled-score output $-\epsilon_t^\theta$ has the vector field $(x - \epsilon_t^\theta)/t$. Subtracting the
target gives the velocity error of each output:

$$
\begin{aligned}
u_t^\theta - (z - \epsilon) &= \frac{D_t^\theta - z}{1 - t}
&& \blacktriangleright\ \text{denoiser output} \\
u_t^\theta - (z - \epsilon) &= -\frac{\epsilon_t^\theta - \epsilon}{t}
&& \blacktriangleright\ \text{scaled-score output}
\end{aligned}
$$

The data loss is therefore $(1 - t)^2$ times the squared velocity error, and the scaled-score loss $t^2$ times.
:::

The table lists the four recipes with their time weights, and the notes' denoising score matching loss
$\mathcal L_{\text{CSM}}$, which does not ship. Its fourth column names each loss in the references, and the paragraphs
below the table explain $x_1$-prediction, v-prediction and $\mathcal L_{\text{CSM}}$. GVP is the path
$\alpha_t = \sin\frac{\pi t}{2}$, $\beta_t = \cos\frac{\pi t}{2}$ with $\kappa_t = \pi/2$ ([Paths](index.md#paths)). Its
weights carry the constant $(2/\pi)^2 = 1/\kappa_t^2$, which only rescales them; their shapes are $\cos^2$ and $\sin^2$.
The names in the fourth column hold under the conditions of [One training step](#one-training-step).

| recipe | output $o_t^\theta(x \mid y)$ | target | as a conditional matching loss | $\lambda_t$ | linear | GVP |
| --- | --- | --- | --- | --- | --- | --- |
| `velocity_mse` | $u_t^\theta(x \mid y)$ | $u_t^{\text{target}}(x \mid z)$ | the notes' $\mathcal L_{\text{CFM}}$; on GVP, $(\pi/2)^2$ times v-prediction | $1$ | $1$ | $1$ |
| `data_mse` | $D_t^\theta(x \mid y)$ | $z$ | the FM Guide's $x_1$-prediction loss | $(\beta_t/\kappa_t)^2$ | $(1-t)^2$ | $(2/\pi)^2\cos^2\frac{\pi t}{2}$ |
| `data_output_velocity_loss` | $D_t^\theta(x \mid y)$ | $u_t^{\text{target}}(x \mid z)$, after converting the output | $\mathcal L_{\text{CFM}}$ through the denoiser | $1$ | $1$ | $1$ |
| `scaled_score_mse` (default) | $\beta_t s_t^\theta(x \mid y) = -\epsilon_t^\theta(x \mid y)$ | $-\epsilon$ | the notes' $\mathcal L_{\text{DDPM}}$ | $(\alpha_t/\kappa_t)^2$ | $t^2$ | $(2/\pi)^2\sin^2\frac{\pi t}{2}$ |
| not shipped | $s_t^\theta(x \mid y)$ | $-\epsilon/\beta_t$ | the notes' $\mathcal L_{\text{CSM}}$ | $\big(\alpha_t/(\beta_t\kappa_t)\big)^2$ | $t^2/(1-t)^2$ | $(2/\pi)^2\tan^2\frac{\pi t}{2}$ |

The references name the outputs differently. The FM Guide calls the denoiser and the noise the $x_1$- and the
$x_0$-prediction: its $X_1$ is our $z$, and its $X_0$ our $\epsilon$. Diffusion papers put the data at time 0, so for
them the $x_0$-prediction is the denoiser ([the dictionary](../notation.md#dictionary-to-the-references)). Their
v-prediction reads GVP as $x = \cos\varphi\, z + \sin\varphi\, \epsilon$ with the angle $\varphi = \pi(1 - t)/2$,
and regresses onto the velocity $v = \mathrm dx/\mathrm d\varphi$. Since $\mathrm d\varphi/\mathrm dt = -\pi/2$, the
conditional vector field is $u_t^{\text{target}}(x \mid z) = -(\pi/2)\,v$, and the velocity loss on GVP is
$(\pi/2)^2$ times the v-prediction loss.

**The same in theory, different in practice.** In theory the recipes learn the same vector field. The minimizer is
chosen time by time, and at each $t \in (0, 1)$ the conversion is an invertible affine map of the output. Mapped to a
vector field, every recipe's minimizer is therefore $u_t^{\text{target}}(x \mid y)$ at every time its flow-time
distribution draws: on all of $(0, 1)$ for the default support, and only up to $t_{\max} = 0.99$ for
`data_output_velocity_loss` as the study ran it ([Endpoints](#endpoints)). A positive $\lambda_t$ changes how
much a time counts, not what the network should output there. In practice the network is not exact, and two things
decide where its error lands. The weight $\lambda_t$ decides where the loss tolerates error. The gain $B_t$ of
[the conversion](index.md#four-parameterizations) decides how an output error $\delta$ becomes a velocity error at
sampling: $\delta$ for the vector field, $(\kappa_t/\beta_t)\,\delta$ for the denoiser and
$(\kappa_t/\alpha_t)\,\delta$ for the scaled score.

Intuitively, each output is easy to predict at one end of the flow. Near the data $x$ is almost $z$, so a denoiser
nearly copies its input. Its own error stays small even when the vector field it implies is poor, because the
conversion divides that error by $\beta_t$. Near the noise $x$ is almost $\epsilon$, and the same holds for the scaled
score with $\alpha_t$. Measured in its own space, each loss stops paying for errors at its easy end. The data loss
thus differs from the velocity loss in two ways at once, the output and the weighting.
`data_output_velocity_loss` keeps the denoiser output with $\lambda_t = 1$, which separates the two effects, up to
the support of [Endpoints](#endpoints).

The time weights act on the same axis as the flow-time distribution. On the linear path with uniform times, and for
any integrable function $f(t)$ of the flow time, such as the expected squared velocity error at $t$,

$$
\mathbb E_{t \sim \mathrm{Unif}}\big[(1 - t)^2\, f(t)\big] = \tfrac13\, \mathbb E_{t \sim \mathrm{Beta}(1, 3)}\big[f(t)\big],
\qquad
\mathbb E_{t \sim \mathrm{Unif}}\big[t^2\, f(t)\big] = \tfrac13\, \mathbb E_{t \sim \mathrm{Beta}(3, 1)}\big[f(t)\big],
$$

since $\mathrm{Beta}(1, 3)$ and $\mathrm{Beta}(3, 1)$ have the densities $3(1 - t)^2$ and $3t^2$. In velocity space the
data loss therefore counts the times as the velocity loss would under $\mathrm{Beta}(1, 3)$. That moves weight toward
the noise, the direction of logit-normal(−0.5) and of π0's Beta. `scaled_score_mse` corresponds to
$\mathrm{Beta}(3, 1)$, which moves weight toward the data.

**Diffusion losses.** A diffusion model trains a score network $s_t^\theta$ by denoising score matching, a regression
onto the conditional score $\nabla \log p_t(x \mid z) = -\epsilon/\beta_t$. The notes' $\mathcal L_{\text{CSM}}$ counts
every time equally there. Since
$\beta_t s_t^\theta + \epsilon = \beta_t\big(s_t^\theta - \nabla \log p_t(x \mid z)\big)$, `scaled_score_mse` is
denoising score matching with its score error weighted by $\beta_t^2$:

$$
\big\lVert \beta_t s_t^\theta(x \mid y) + \epsilon \big\rVert^2
= \big\lVert \epsilon_t^\theta(x \mid y) - \epsilon \big\rVert^2
= \beta_t^2\, \big\lVert s_t^\theta(x \mid y) - \nabla \log p_t(x \mid z) \big\rVert^2 .
$$

The middle form is the integrand of the notes' $\mathcal L_{\text{DDPM}}$, with the noise predictor
$\epsilon_t^\theta = -\beta_t s_t^\theta$. The notes explain why DDPM dropped the weight $1/\beta_t^2$ of
$\mathcal L_{\text{CSM}}$: denoising score matching is numerically unstable where $\beta_t \approx 0$. The table shows
the instability in velocity space. $\mathcal L_{\text{CSM}}$ weights the velocity error by $t^2/(1 - t)^2$ on the
linear path, which is not integrable up to the data.

The FM Guide shows that training a diffusion model is flow matching with a Gaussian path and a score
parameterization. `scaled_score_mse` is that training on our paths, up to the sign of the output.

**Diffusion Policy.** Diffusion Policy trains a noise predictor with DDPM. It draws one of $T = 100$ noise levels
uniformly under the cosine schedule, $x = \sqrt{\bar\alpha}\, z + \sqrt{1 - \bar\alpha}\, \epsilon$, where DDPM's
$\bar\alpha$ falls from 1 at the data to 0. Write the diffusion time as $\tau = 1 - t$, with the data at $\tau = 0$. As
the schedule's offset (0.008 in Diffusion Policy) goes to 0, $\sqrt{\bar\alpha} = \cos\frac{\pi\tau}{2}$ and
$\sqrt{1 - \bar\alpha} = \sin\frac{\pi\tau}{2}$. In flow time these are $\alpha_t = \sin\frac{\pi t}{2}$ and
$\beta_t = \cos\frac{\pi t}{2}$: the GVP path. As $T \to \infty$ the uniform level becomes $t \sim \mathrm{Unif}[0, 1]$.
So GVP with `scaled_score_mse` trains the same noise predictor with the same weighting, up to the reduction and a
small change of clock near the data. It reproduces neither the 100-level grid nor the DDPM sampler.

### Endpoints

Where on $[0, 1]$ can each recipe train? A loss in the output's own space never converts the output. Their targets,
$\dot\alpha_t z + \dot\beta_t\epsilon$, $z$ and $-\epsilon$, are finite at both ends, so `velocity_mse`, `data_mse` and
`scaled_score_mse` train on all of $[0, 1]$. Only a loss in velocity space converts during training, so only it can
divide by zero.

`data_output_velocity_loss` converts a denoiser, $u_t^\theta = (\dot\beta_t x + \kappa_t D_t^\theta)/\beta_t$, which
divides by $\beta_t = 0$ at $t = 1$. In theory the singularity cancels: for the exact denoiser the numerator vanishes
like $\beta_t$ where the data have a density ([the singular ends](index.md#four-parameterizations)). In practice the
network's denoising error $\delta = D_t^\theta - z$ becomes the velocity error $(\kappa_t/\beta_t)\,\delta$. The loss
therefore weights $\lVert\delta\rVert^2$ by $(\kappa_t/\beta_t)^2$, which is $(1 - t)^{-2}$ on the linear path and
is not integrable up to $t = 1$.

{py:class}`~flow.FlowMatching` refuses this recipe on a support that contains $t = 1$ ([Numerics](index.md#numerics)).
That rule alone would allow $[0, 0.9999]$. The study cut the support at $t_{\max} = 0.99$
(`flow.time.support=[0.0,0.99]`), which caps the weight at about $10^4$ on both paths. Above $t_{\max}$ the network is
never trained. The default 10-step Euler grid evaluates it at most at $t = 0.9$, and the study's SDE stopped at
`sampler.sde_end=0.99`.

The cut also qualifies the comparison with the velocity loss. `data_output_velocity_loss` separates output from
weighting only up to its support: it is never trained above $t = 0.99$, where `velocity_mse` spends 1% of its draws
at full weight. The data loss's weight there is at most $10^{-4}$ on the linear path.

A scaled score divides by $\alpha_t = 0$ at $t = 0$, but only when it is converted, at sampling. Its samplers
therefore start later, at $t_0 = 0.05$ by default (`SCALED_SCORE_START`, `sampler.start_time`).
[Where sampling starts](sampling.md#where-sampling-starts) explains why the start decides whether this recipe works.

**What we found** (on AdamW, before Muon became the default; the scaled score was then confirmed with Muon)

- **velocity_mse = data_output_velocity_loss >> data_mse >> scaled_score_mse on the linear path, with the scaled
  score sampled from $t = 0.001$.** This was the order at the old start; the scaled score's last place is a sampler
  effect.
- **data_output_velocity_loss >> data_mse, linear path.** Where the loss puts its weight matters more than what the
  network outputs: the data loss's $(1 - t)^2$ nearly ignores the late part of the flow.
- **velocity_mse > data_mse on GVP too, by less.** The drop splits about evenly between output and weighting (neither
  resolved), and the weighting costs less than on the linear path, consistent with GVP's weight
  $\cos^2\frac{\pi t}{2}$, which falls later and more gently.
- **scaled_score_mse sampled from $t = 0.05$ > velocity_mse, linear path.** It is ahead on the selection episodes
  and confirmed on held-out episodes with fresh training seeds, and starting velocity policies at 0.05 changes
  nothing, so the gain comes with the scaled score, now the default.
- **scaled_score_mse on GVP sampled from $t = 0.05$ > velocity_mse on the linear path (weak).** Paired by training
  seed, its lead is within the noise.
- **start $t = 0.01$ = start $t = 0.05$ for the scaled score.** Both are far ahead of the old start $t = 0.001$
  (see {doc}`sampling`).
- **GVP = linear for scaled_score_mse sampled from $t = 0.05$.** Diffusion Policy's objective in continuous time
  does as well as the linear path's.
- **Muon with scaled_score_mse > Muon with velocity_mse.** The two improvements add up, with no interaction (see
  {doc}`../training`).

Reproduce: `uv run train.py -m experiment=flow_matching` (path × recipe, on AdamW) and
`uv run train.py -m experiment=flow_matching 'flow/path=linear,gvp' flow.recipe=data_output_velocity_loss 'flow.time.support=[0.0,0.99]'`;
add `sampler.start_time=0.001` to log the scaled score's training curves from the old start, as the study did.

## Reduction

How do the $2H$ squared errors of one chunk become one number? Let $e_{kj}$ be the entry of the error $e$ at step $k$
and coordinate $j$ of the chunk. Let $m_k$ be the loss weight of step $k$: 0 on future steps past the episode's end
under `data.padding=mask`, 1 otherwise ({doc}`../data`). `flow.reduction` picks one of two reductions in
{py:meth}`~flow.FlowMatching.per_example`:

$$
\ell^{\mathrm{sum}}(e) = \sum_{k=0}^{H-1} m_k \sum_{j=1}^{2} e_{kj}^2,
\qquad
\ell^{\mathrm{example\_mean}}(e) = \frac{1}{2\sum_k m_k}\sum_{k=0}^{H-1} m_k \sum_{j=1}^{2} e_{kj}^2 .
$$

The loss is the batch mean of $\ell$. With every $m_k = 1$, $\ell^{\mathrm{sum}}(e) = \lVert e\rVert^2$, the squared
norm of the notes' losses. Every chunk has at least one valid step (its first step lies inside its episode, and the
dataset checks it), so the division is safe.

Under `sum` (the default) every valid step weighs the same. Under `example_mean` every chunk weighs the same, so a
chunk cut short by its episode's end weighs as much as a full one, and its steps more. In other words, the two
reductions differ by a weight per chunk, $c = 1/(2\sum_k m_k)$. Without masked padding it is the constant $1/(2H)$. A
constant rescales the loss and its gradient but not the gradient's direction, and the optimizers' normalized updates
nearly cancel it. With masked padding, the default, the weight depends on the chunk. Unlike a constant or a weight that
depends only on $t$, such a weight moves the minimizer. Under `sum` the minimizer at step $k$ is the posterior mean
over the chunks in which step $k$ is valid. Under `example_mean` it is a weighted posterior mean in which chunks cut
short count more. In practice every step of an episode starts a chunk, and only the chunks that start in its last
$H - 1$ steps are cut short. The two minimizers can therefore differ only at inputs $(x, t, y)$ that such a chunk could
have produced.

**What we found** (linear path, velocity loss, AdamW)

- **sum = example_mean.** The interval is wide; a small difference from the chunks cut short is expected, but this
  experiment does not establish one.

Reproduce: `uv run train.py -m experiment=flow_matching flow/path=linear flow.recipe=velocity_mse flow.reduction=example_mean`

## Comparing training losses

When do two training losses compare? By [the theorem that regression learns a posterior
mean](#regression-learns-a-posterior-mean), the loss is the network's error plus a floor,
$\mathbb E\,\mathrm{tr}\,\mathrm{Cov}(o_t^{\text{target}} \mid x, t, y)$. The floor is the part of the target that $x$,
$t$ and $y$ leave open, and no network removes it. Its size is unknown, so the absolute value of a loss says little. Two
losses compare only when their floors agree, and the floor depends on everything that defines the target and its
weighting:

- **The objective.** The path, the recipe, the flow-time distribution and the reduction all change the floor. For the
  velocity loss at $t = 0$, the floor is $(\pi/2)^2 \approx 2.47$ times larger on GVP than on the linear path
  ([the floor example](#regression-learns-a-posterior-mean)). Without masked padding, `example_mean` scales it by
  $1/(2H)$.
- **The data and what the network is told.** The chunk length $H$, the representation and observation history of
  {doc}`../data`, the padding and the share $p_{\mathrm{uncond}}$ of rows trained on $\varnothing$ change it too. Rows
  with $\varnothing$ regress onto the unguided posterior, which is wider.

Training losses therefore compare only between runs that share all of these, for example across optimizers, network
sizes or seeds. To compare objectives, compare late success ({doc}`../evaluation`).

:::{admonition} Summary
:class: note

Training regresses the network onto a target computed from $(z, \epsilon)$, and the regression learns the target's
posterior mean given $(x, t, y)$: for the conditional vector field, the guided vector field the sampler needs. The
minimizer is chosen time by time, so the flow-time distribution and the recipe only weight the times, and the velocity
error at $t$ counts with $p_{\mathrm{time}}(t)\,\lambda_t$. In theory no positive weight changes what is learned. In
practice the weight decides where a finite network is accurate, and the conversions amplify its error near the ends.
The loss has a floor, which the whole objective and the data determine, so loss values compare only within one setting.
:::
