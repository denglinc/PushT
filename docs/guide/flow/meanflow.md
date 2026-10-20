# MeanFlow

A flow matching policy learns the vector field at one flow time, and its sampler needs several small steps to follow
it. MeanFlow trains the network on the average velocity over an interval $[t, s]$ instead, so that one step from noise
lands on a chunk (MeanFlow and improved MeanFlow, Geng et al., 2025). {py:class}`~flow.MeanFlow` is the objective and
{py:class}`~flow.MeanFlowStep` its sampler.

```bash
uv run train.py experiment=meanflow                                  # flow=meanflow sampler=mean, AdamW with warm-up
uv run evaluation.py runs=runs/meanflow sampler=mean                 # one step from noise, held-out episodes
uv run evaluation.py runs=runs/meanflow sampler=mean sampler.nfe=2   # two steps
```

This page shows why one step of the vector field falls short, builds the average velocity from the flow map of
{doc}`sampling`, derives the identity that makes it learnable without simulation and turns that identity into a
regression target. The options, the sampler and the findings follow. The notation is that of {doc}`index` and
{doc}`the notation page <../notation>`. The MeanFlow papers run time the other way; [the papers'
convention](#the-papers-convention) translates their formulas.

## Why one step needs more than the vector field

Why can a flow matching policy not sample in one step? Its network learns the marginal vector field
$u_t^{\text{target}}(x)$, the velocity of the ODE that carries noise to chunks
([index](index.md#the-marginal-vector-field)), and a sampler integrates it. One step of the Euler method with step size
$h$,

$$
X_{t+h} = X_t + h\,u_t^{\text{target}}(X_t),
$$

replaces the motion over $[t, t + h]$ by the velocity at its start. It is a first-order Taylor step
([ODE solvers](sampling.md#ode-solvers)). It is exact when the velocity does not change along the step. In general,
its error vanishes only as $h \to 0$.

With one step from noise, that error is the whole answer. On the linear path, one Euler step from $t = 0$ lands on
$D_0 = \mathbb E[z]$, the mean chunk for the observation, whatever the noise
([sampling](sampling.md#one-euler-step-on-gvp)). This is the regression policy of
[index](index.md#the-marginal-vector-field): when the demonstrations take two routes, it can lie between them. Note
that this is not a training error. The exact vector field makes it too, because the velocity changes along the
trajectories; on [the Gaussian example](index.md#an-example-with-gaussian-data) it is constant only when the data are a
single chunk.

MeanFlow keeps the single step and changes what the network predicts. Instead of the velocity at the start of a step,
it learns the average velocity over the whole step, which moves $X_t$ exactly to $X_{t+h}$.

## Average velocity

What should a network predict so that one step is exact? The answer belongs to a whole trajectory, not to one point on
it. {doc}`sampling` defines the flow map $\psi_{t \to s}$ ([the flow model](sampling.md#the-flow-model)):
$\psi_{t \to s}(x)$ is the point where the marginal ODE, started at $x$ at time $t$, arrives at time $s$,

$$
\psi_{t \to s}(x) = x + \int_t^s u_\tau^{\text{target}}\big(\psi_{t \to \tau}(x)\big)\,\mathrm d\tau,
\qquad 0 \le t \le s \le 1 .
$$

The vector field describes the instantaneous motion, and the flow map records the motion accumulated over a finite
interval. The Diffusion Guide of Lai et al. draws this distinction and returns to the flow map for its fast samplers. A
step from $t$ to $s$ is exact when it applies the flow map. MeanFlow learns the flow map's displacement, divided by the
elapsed time.

:::{admonition} Definition: the average velocity
:class: important

For $0 \le t < s \le 1$, the **average velocity** over $[t, s]$ is the displacement of the flow map per unit time,

$$
\bar u_{t,s}(x) = \frac{\psi_{t \to s}(x) - x}{s - t}
= \frac{1}{s - t}\int_t^s u_\tau^{\text{target}}\big(\psi_{t \to \tau}(x)\big)\,\mathrm d\tau ,
$$

the mean of the velocity along the trajectory through $x$ at time $t$. At $s = t$ it is the limit,
$\bar u_{t,t}(x) = \lim_{s \to t}\bar u_{t,s}(x) = u_t^{\text{target}}(x)$.
:::

In other words, $\bar u_{t,s}(x)$ is the one constant velocity that, held from $t$ to $s$, reaches the same point as the
trajectory through $x$. The formula is $0/0$ at $s = t$. As $s \to t$, the mean of a continuous velocity over $[t, s]$
tends to its value at $t$, so the average velocity over an empty interval is the marginal vector field. Improved
MeanFlow calls this the **boundary condition**. The network does not satisfy it by construction. It learns it on the
rows with $s = t$, where [the target](#the-target) reduces to $u_t^{\text{target}}(x \mid z)$.

Multiplying the definition by $s - t$ turns it into an exact step:

$$
X_s = X_t + (s - t)\,\bar u_{t,s}(X_t), \qquad\text{in particular}\qquad X_1 = X_0 + \bar u_{0,1}(X_0) .
$$

One evaluation of $\bar u_{0,1}$ carries the noise to a chunk. The vector field, the flow map and the average velocity
describe the same motion of the marginal ODE. They differ in what they record of it and in what one step from $t$ to
$s$ built on each gives:

| | vector field $u_t^{\text{target}}(x)$ | flow map $\psi_{t \to s}(x)$ | average velocity $\bar u_{t,s}(x)$ |
| --- | --- | --- | --- |
| what it gives | the velocity at time $t$ | the position at time $s$ | the displacement over $[t, s]$ per unit time |
| at $s = t$ | itself | $x$ | $u_t^{\text{target}}(x)$ |
| one step from $t$ to $s$ | $X_t + (s - t)\,u_t^{\text{target}}(X_t)$, exact when the velocity is constant along the trajectory | $\psi_{t \to s}(X_t)$, exact | $X_t + (s - t)\,\bar u_{t,s}(X_t)$, exact |
| learned by | flow matching ({doc}`training`) | no objective in this code | MeanFlow |

Why predict $\bar u_{t,s}$ rather than $\psi_{t \to s}$? The average velocity stays on the scale of a velocity for
every interval, while the displacement shrinks to zero as $s \to t$. At $s = t$ it is the vector field itself, so one
network is both a one-step model and a flow matching model.

:::{admonition} Example: Gaussian data
:class: tip

Take the Gaussian data of [index](index.md#an-example-with-gaussian-data), with each coordinate
$z \sim \mathcal N(m, s^2)$. In this box only, $s$ is the data's standard deviation, not an end time. On every path the
marginal ODE carries a start $X_0$ to $X_1 = m + s X_0$, so the average velocity over the whole flow is

$$
\bar u_{0,1}(x) = \psi_{0 \to 1}(x) - x = m + (s - 1)\,x .
$$

One exact MeanFlow step sends $X_0 \sim \mathcal N(0, I)$ to $m + s X_0 \sim \mathcal N(m, s^2)$, the data
distribution. One Euler step uses $u_0^{\text{target}}(x) = \dot\alpha_0 m + \dot\beta_0 x$ instead, which is $m - x$
on the linear path and lands every draw on $m$. There the two velocities differ by $s x$, the spread that the Euler step
loses.
:::

## The MeanFlow identity

How can a network learn $\bar u_{t,s}$ without simulating the ODE? Regressing onto $\bar u_{t,s}$ would need it as a
target, that is, a simulation of the marginal ODE, whose vector field is itself unknown. Flow matching avoided
simulation by regressing onto a conditional target whose posterior mean is the unknown field
([training](training.md#regression-learns-a-posterior-mean)). MeanFlow does the same in two moves. An identity first
trades the integral for a derivative at the start of the interval. The conditional vector field then replaces the
marginal one inside it ([the target](#the-target)).

**Derivatives.** The average velocity is a function of a chunk $x$ and two times. Its **Jacobian**
$\partial_x \bar u_{t,s}(x)$ is its derivative in $x$ over the whole chunk, a linear map from chunks to chunks (a
$2H \times 2H$ matrix). Its partial derivative $\partial_t \bar u_{t,s}(x)$ is taken in the start time, at fixed $x$
and $s$. When $x$ moves with a velocity $v$ while the start time advances at unit rate, the chain rule gives the
**derivative along $(v, 1)$**:

$$
\frac{\mathrm d}{\mathrm dt}\,\bar u_{t,s}\big(x(t)\big) = \partial_x \bar u_{t,s}\,v + \partial_t \bar u_{t,s},
\qquad \frac{\mathrm d x(t)}{\mathrm dt} = v,\ s \text{ fixed} .
$$

:::{admonition} Theorem: the MeanFlow identity
:class: important

For $0 \le t < s \le 1$ and every chunk $x$, the average velocity satisfies

$$
\bar u_{t,s}(x) = u_t^{\text{target}}(x)
+ (s - t)\big[\partial_x \bar u_{t,s}(x)\,u_t^{\text{target}}(x) + \partial_t \bar u_{t,s}(x)\big] .
$$

Conversely, a function $f_{t,s}(x)$ that is differentiable for $t < s$, stays bounded as $t \to s$ and satisfies the
identity is the average velocity: $f_{t,s} = \bar u_{t,s}$.
:::

In other words, the average over $[t, s]$ is the velocity at the start of the interval plus a correction proportional
to its length. The correction is the derivative of $\bar u_{t,s}$ along the marginal vector field: how the average
changes when its start slides forward along the trajectory while its end stays fixed. Where the velocity is constant
along the trajectories, as for a single data chunk on the linear path, the correction vanishes and
$\bar u_{t,s} = u_t^{\text{target}}$. As $s \to t$ the correction vanishes too, and the identity becomes the boundary
condition.

To derive the identity, fix $s$ and follow one trajectory $X_t$ of the marginal ODE up to time $s$. Along it,

$$
\begin{aligned}
-u_t^{\text{target}}(X_t)
&\overset{(i)}{=} \frac{\mathrm d}{\mathrm dt}\big(X_s - X_t\big)
\overset{(ii)}{=} \frac{\mathrm d}{\mathrm dt}\Big((s - t)\,\bar u_{t,s}(X_t)\Big) \\
&\overset{(iii)}{=} -\bar u_{t,s}(X_t) + (s - t)\,\frac{\mathrm d}{\mathrm dt}\,\bar u_{t,s}(X_t) \\
&\overset{(iv)}{=} -\bar u_{t,s}(X_t)
+ (s - t)\big[\partial_x \bar u_{t,s}(X_t)\,u_t^{\text{target}}(X_t) + \partial_t \bar u_{t,s}(X_t)\big],
\end{aligned}
$$

where (i) holds because $X_t$ solves the ODE and its endpoint $X_s$ does not depend on $t$, (ii) uses the flow property,
(iii) is the product rule, and (iv) is the derivative along $(v, 1)$ with $v = u_t^{\text{target}}(X_t)$, the
trajectory's own velocity. The **flow property** is the composition rule of [the flow map](sampling.md#the-flow-model):
started anywhere on a trajectory, the ODE follows the same trajectory. Hence $\psi_{t \to s}(X_t) = X_s$ for every
$t \le s$, and the definition of $\bar u_{t,s}$ gives $X_s - X_t = (s - t)\,\bar u_{t,s}(X_t)$. Rearranging the chain
gives the identity at $X_t$. Every chunk $x$ at time $t$ lies on one trajectory, the one through $(t, x)$, so the
identity holds at every $x$.

The converse runs the same steps backward. Let $f_{t,s}$ satisfy the identity and follow a trajectory as before. Steps
(iii) and (iv) with $f$ in place of $\bar u$, together with the identity, give
$\frac{\mathrm d}{\mathrm dt}\big((s - t)\,f_{t,s}(X_t)\big) = -u_t^{\text{target}}(X_t)$. Integrating from $t$ to $s$
gives $(s - t)\,f_{t,s}(X_t) = X_s - X_t$. At the upper limit the term $(s - \tau)\,f_{\tau,s}(X_\tau)$ vanishes as
$\tau \to s$, because $f$ stays bounded. The result is the definition of $\bar u_{t,s}$. Therefore a bounded network
that satisfies the identity for every $t < s$ is the average velocity, and training needs no simulation. In theory this
also yields the boundary condition, as the limit $s \to t$. In practice the rows with $s = t$ teach it directly ([the
target](#the-target)).

## The target

What can training regress onto, when the identity contains the unknown marginal field twice and the unknown
$\bar u_{t,s}$ on its right side? MeanFlow replaces each unknown by something it can compute. In the first term it
puts the conditional vector field $u_t^{\text{target}}(x \mid z)$, as flow matching does. As the direction of the
derivative it puts a **tangent** $v$, one of two estimates of the marginal field. In place of $\bar u_{t,s}$ on the
right side it puts the network itself, under a **stop-gradient** $\mathrm{sg}$, which passes values but no gradients.

:::{admonition} Definition: the MeanFlow target
:class: important

Draw a pair of flow times $t \le s$ ([Options](#options)), a chunk $z$ with its guiding variable $y$, and the noise.
Then

$$
\begin{aligned}
x &= \alpha_t z + \beta_t \epsilon, \quad z \sim p_{\mathrm{data}},\ \epsilon \sim \mathcal N(0, I)
&& \blacktriangleright\ \text{noisy chunk} \\
v &= u_t^{\text{target}}(x \mid z) \quad\text{or}\quad v = \bar u^\theta_{t,t}(x \mid y)
&& \blacktriangleright\ \text{tangent: conditional or predicted} \\
Y &= \mathrm{sg}\Big(u_t^{\text{target}}(x \mid z)
+ (s - t)\big[\partial_x \bar u^\theta_{t,s}(x \mid y)\,v + \partial_t \bar u^\theta_{t,s}(x \mid y)\big]\Big)
&& \blacktriangleright\ \text{target}
\end{aligned}
$$

and the network's $\bar u^\theta_{t,s}(x \mid y)$ is regressed onto $Y$.
:::

Intuitively, each row asks the network's output to agree with a one-sample estimate of the identity's right side,
computed with the network's current self. Only the left side receives gradient.

Each chunk's **plain loss** $\ell$ is the squared error of $\bar u^\theta_{t,s}(x \mid y) - Y$, reduced over the
chunk's valid entries as in [Reduction](training.md#reduction). The objective averages it over the batch, scaled by
the adaptive weight of [Options](#options).

The stop-gradient matters because the target contains the network. With it, each step is an ordinary regression onto a
target computed with the current weights. No gradient flows through the derivative, so training needs no second
derivatives of the network. With the plain squared error ($p = 0$ in [Options](#options)), what training looks for is a
network that equals the posterior mean of its own target: a fixed point.

On the default linear path ($\alpha_t = t$, $\beta_t = 1 - t$) with the conditional tangent, the target is

$$
Y = \mathrm{sg}\Big((z - \epsilon)
+ (s - t)\big[\partial_x \bar u^\theta_{t,s}(x \mid y)\,(z - \epsilon) + \partial_t \bar u^\theta_{t,s}(x \mid y)\big]\Big),
\qquad x = t z + (1 - t)\,\epsilon .
$$

Its derivative term is the rate of change of $\bar u^\theta_{t,s}$ along the straight line from $\epsilon$ to $z$
through $x$, the conditional path of this one sample.

**Why the conditional target is right.** Fix $t < s$ and $y$. The pair of times is drawn independently of the chunk and
the noise, so given $x$ the pair $(z, \epsilon)$ has the posterior of [the marginal vector
field](index.md#the-marginal-vector-field).

:::{admonition} Proposition: the target's posterior mean
:class: important

For fixed $t < s$ and $y$, with the conditional tangent, for every $x$,

$$
\mathbb E[Y \mid x] = u_t^{\text{target}}(x)
+ (s - t)\big[\partial_x \bar u^\theta_{t,s}(x \mid y)\,u_t^{\text{target}}(x) + \partial_t \bar u^\theta_{t,s}(x \mid y)\big],
$$

the right side of the MeanFlow identity with the network in place of $\bar u_{t,s}$. With the predicted tangent the
same holds once $\bar u^\theta_{t,t}(x \mid y) = u_t^{\text{target}}(x)$.
:::

The target is affine in the conditional vector field, and that is all the proof needs:

$$
\begin{aligned}
\mathbb E[Y \mid x]
&\overset{(i)}{=} \big(I + (s - t)\,\partial_x \bar u^\theta_{t,s}(x \mid y)\big)\,
\mathbb E\big[u_t^{\text{target}}(x \mid z) \,\big|\, x\big] + (s - t)\,\partial_t \bar u^\theta_{t,s}(x \mid y) \\
&\overset{(ii)}{=} u_t^{\text{target}}(x)
+ (s - t)\big[\partial_x \bar u^\theta_{t,s}(x \mid y)\,u_t^{\text{target}}(x) + \partial_t \bar u^\theta_{t,s}(x \mid y)\big],
\end{aligned}
$$

where (i) collects the terms in $u_t^{\text{target}}(x \mid z)$ and takes the matrix and the offset out of the
expectation, and (ii) uses that the marginal vector field is the posterior mean of the conditional one
([index](index.md#the-marginal-vector-field)). Step (i) holds because the network computes the matrix and the offset
from $(x, t, s, y)$ and never sees $z$. The stop-gradient does not change values. The predicted tangent is a function
of $(x, t, y)$, so it leaves the expectation the same way and gives the same line once it equals
$u_t^{\text{target}}(x)$.

In other words, the regression's fixed point satisfies the identity. A squared-error regression learns the posterior
mean of its target ([training](training.md#regression-learns-a-posterior-mean)). A converged network that is expressive
enough therefore equals the right side of the identity evaluated with itself, and by the converse it is
$\bar u_{t,s}$. This holds exactly for the plain squared error, an adaptive weight of power $p = 0$. For $p > 0$ the
weight changes what is learned ([Options](#options)).

**Why not the conditional displacement.** A simpler target would be the displacement of the sample's conditional path,
$(\alpha_s z + \beta_s \epsilon - x)/(s - t)$, which needs no derivative. Its posterior mean is
$\big(\alpha_s D_t(x) + \beta_s\,\mathbb E[\epsilon \mid x] - x\big)/(s - t)$: the displacement of a DDIM step
from $t$ to $s$, per unit time. On the linear path the DDIM step is the Euler step
([sampling](sampling.md#one-euler-step-on-gvp)), so this posterior mean is $u_t^{\text{target}}(x)$, and a network
trained on it would take Euler's step again. The reason is the difference between the
[two random variables called $X_t$](index.md#the-marginal-vector-field). The conditional path and the ODE trajectory
have the same law at each time, but not the same joint law at two times, and $\bar u_{t,s}$ depends on where a
trajectory goes between them. The identity needs the trajectory at the single time $t$ only, through
$u_t^{\text{target}}(x)$, and at a single time the conditional field does average to the marginal one.

**The rows with $s = t$ are flow matching.** At $s = t$ the derivative term is multiplied by zero, and
$Y = u_t^{\text{target}}(x \mid z)$ whatever the tangent. The plain loss of such a row is the velocity loss of
{doc}`training`, the error inside $\mathcal L_{\text{CFM}}$, at MeanFlow's flow times. With the plain squared error
($p = 0$), these rows train $\bar u^\theta_{t,t}$ toward the posterior mean of $u_t^{\text{target}}(x \mid z)$, which
is $u_t^{\text{target}}(x)$: the boundary condition. For $p > 0$ they train it toward the weighted location of
[Options](#options).

**In the code.** {py:meth}`~flow.MeanFlow.loss` draws the pairs and the noise, builds $x$ and
$u_t^{\text{target}}(x \mid z) = \dot\alpha_t z + \dot\beta_t \epsilon$, computes the target and detaches it.
{py:class}`~flow.MeanFlow` is a {py:class}`~flow.FlowMatching` with the velocity prediction fixed: it reuses the path,
the flow-time samplers and the reductions of {doc}`training`. Its default path is the linear one
(`conf/flow/meanflow.yaml`).

## Computing the derivative

How does the code evaluate $\partial_x \bar u^\theta_{t,s}\,v + \partial_t \bar u^\theta_{t,s}$ without forming the
$2H \times 2H$ Jacobian? The bracket is a Jacobian-vector product: the derivative of
$(x, t) \mapsto \bar u^\theta_{t,s}(x \mid y)$ at $(x, t)$ in the direction $(v, 1)$, with $s$ fixed. One pass of
forward-mode automatic differentiation, which pushes a direction through the network alongside its inputs, returns the
value $\bar u^\theta_{t,s}(x \mid y)$ and this derivative together.

**The gap.** The network does not read the end time $s$. It reads the **gap** $h = s - t$, the step size of a step from
$t$ to $s$; {doc}`../model` shows how $h$ enters the DiT and the MLP. Write the network as a function of the gap,
$F^\theta(x, t, h \mid y)$, so that $\bar u^\theta_{t,s}(x \mid y) = F^\theta(x, t, s - t \mid y)$. At fixed $s$,
moving the start time also moves the gap. For any tangent $v$,

$$
\partial_x \bar u^\theta_{t,s}\,v + \partial_t \bar u^\theta_{t,s}
\overset{(i)}{=} (\partial_x F^\theta)\,v + \partial_t\big[F^\theta(x, t, s - t \mid y)\big]
\overset{(ii)}{=} (\partial_x F^\theta)\,v + \partial_t F^\theta - \partial_h F^\theta ,
$$

where (i) uses that $h$ does not depend on $x$, so $\bar u^\theta_{t,s}$ and $F^\theta$ have the same Jacobian, and
(ii) is the chain rule with $\mathrm dh/\mathrm dt = -1$ at fixed $s$. The forward-mode pass produces the
$-\partial_h F^\theta$ term only if $h$ is computed from $t$ inside the function it differentiates. Computed outside
the call and passed in as a constant, $h$ would stay fixed and the derivative would silently miss
$\partial_h F^\theta$.

**The zero gap.** Called without an end time, the network sees $h = 0$ and returns $\bar u^\theta_{t,t}(x \mid y)$, its
estimate of $u_t^{\text{target}}(x)$ by the boundary condition. This is the network's flow matching mode: the predicted
tangent evaluates it, and so do the Euler, Heun and SDE samplers ([Sampling](#sampling)). The gap's path into the
network starts at zero ({doc}`../model`). At initialization, therefore,
$F^\theta(x, t, h \mid y) = F^\theta(x, t, 0 \mid y)$ for every $h$, and training starts from
$\bar u^\theta_{t,s} = \bar u^\theta_{t,t}$ for every $s$.

**In the code.** {py:meth}`~flow.MeanFlow.average_velocity` runs the Jacobian-vector product with `torch.func.jvp`,
along the direction $(v, 1)$ in $(x, t)$. {py:meth}`~model.FlowMatchingPolicy._predict` computes $h$ from $t$ and $s$
itself, through {py:meth}`~model.FlowMatchingPolicy.gap`, so the gap moves with $t$ inside the pass. Two
implementation facts come with the forward mode:

- it exists only for the math attention kernel, so the pass runs under that kernel;
- the pass runs in float32 ([Training](#training) gives the consequences).

:::{admonition} Summary
:class: note

The average velocity $\bar u_{t,s}(x) = (\psi_{t \to s}(x) - x)/(s - t)$ is the displacement of the marginal flow over
$[t, s]$ per unit time, so the step $X_s = X_t + (s - t)\,\bar u_{t,s}(X_t)$ is exact; its limit at $s = t$ is
$u_t^{\text{target}}$. It is the only bounded solution of the MeanFlow identity
$\bar u_{t,s} = u_t^{\text{target}} + (s - t)\big[\partial_x \bar u_{t,s}\,u_t^{\text{target}} + \partial_t \bar u_{t,s}\big]$.
MeanFlow regresses $\bar u^\theta_{t,s}(x \mid y)$ onto that right side, with $u_t^{\text{target}}(x \mid z)$ in place
of the marginal field and the network's own derivative under a stop-gradient. One forward-mode pass computes the
derivative, with the gap $h = s - t$ formed inside it. With the plain squared error ($p = 0$), an expressive enough
network and pairs that cover $t < s$, the fixed point is $\bar u_{t,s}$; at $s = t$ the plain loss is flow matching's
velocity loss.
:::

## The papers' convention

How do the papers' formulas read here? MeanFlow and improved MeanFlow put noise at time 1 and data at time 0, as most
diffusion papers do ([time conventions](../notation.md#time-conventions)). We mark their symbols with a superscript MF:
their time pair $(t^{\mathrm{MF}}, r^{\mathrm{MF}})$ with $r^{\mathrm{MF}} \le t^{\mathrm{MF}}$, their velocity
$v^{\mathrm{MF}}$ and their average velocity $u^{\mathrm{MF}}$. The papers and this code draw times from a
[logit-normal law](training.md#flow-time-distributions) $\mathrm{LN}(\mu, \varsigma)$, under which the logit of a time
is $\mathcal N(\mu, \varsigma^2)$. Their clock runs backward, $t^{\mathrm{MF}} = 1 - t$, so

$$
\begin{aligned}
(t^{\mathrm{MF}},\, r^{\mathrm{MF}}) &= (1 - t,\ 1 - s), \quad r^{\mathrm{MF}} \le t^{\mathrm{MF}}
&& \blacktriangleright\ \text{a pair } t \le s \\
v^{\mathrm{MF}} &= -u_t^{\text{target}}
&& \blacktriangleright\ \text{the velocity} \\
u^{\mathrm{MF}} &= -\bar u_{t,s}
&& \blacktriangleright\ \text{the average velocity} \\
\mathrm{LN}(\mu, \varsigma) \text{ for } t^{\mathrm{MF}} &= \mathrm{LN}(-\mu, \varsigma) \text{ for } t
&& \blacktriangleright\ \text{a logit-normal time draw}
\end{aligned}
$$

The velocity flips sign because the clock does:
$v^{\mathrm{MF}} = \mathrm dX/\mathrm dt^{\mathrm{MF}} = -\mathrm dX/\mathrm dt$. For the average,

$$
u^{\mathrm{MF}}
= \frac{1}{t^{\mathrm{MF}} - r^{\mathrm{MF}}}\int_{r^{\mathrm{MF}}}^{t^{\mathrm{MF}}} v^{\mathrm{MF}}\,\mathrm d\tau^{\mathrm{MF}}
\overset{(i)}{=} \frac{1}{s - t}\int_s^t u_\tau^{\text{target}}(X_\tau)\,\mathrm d\tau
\overset{(ii)}{=} -\bar u_{t,s},
$$

where (i) substitutes $\tau^{\mathrm{MF}} = 1 - \tau$, so that $\mathrm d\tau^{\mathrm{MF}} = -\mathrm d\tau$,
$v^{\mathrm{MF}} = -u_\tau^{\text{target}}$ and $t^{\mathrm{MF}} - r^{\mathrm{MF}} = s - t$, and (ii) swaps the limits
of the integral. Their step subtracts $(t^{\mathrm{MF}} - r^{\mathrm{MF}})\,u^{\mathrm{MF}}$ from the state, which is
our $X_s = X_t + (s - t)\,\bar u_{t,s}(X_t)$. Their identity,
$u^{\mathrm{MF}} = v^{\mathrm{MF}} - (t^{\mathrm{MF}} - r^{\mathrm{MF}})\,\mathrm du^{\mathrm{MF}}/\mathrm dt^{\mathrm{MF}}$,
is ours too. With $\mathrm d/\mathrm dt^{\mathrm{MF}} = -\mathrm d/\mathrm dt$ at a fixed end, it reads
$-\bar u_{t,s} = -u_t^{\text{target}} - (s - t)\,\frac{\mathrm d}{\mathrm dt}\bar u_{t,s}$, our identity times $-1$. The
logit-normal flips its mean because $\mathrm{logit}(1 - t^{\mathrm{MF}}) = -\mathrm{logit}\,t^{\mathrm{MF}}$. On the
linear path their conditional velocity is the noise minus the data, $-(z - \epsilon)$ in our letters.

## Options

`conf/flow/meanflow.yaml` holds the objective, `conf/flow/pairs/` the time pairs. In theory, at $p = 0$, neither the
time pairs nor the tangent changes what is learned, provided the pairs cover $t < s$. This needs
`flow.flow_matching_share` below 1. They change only how training reaches $\bar u_{t,s}$. With $p > 0$ the adaptive
weight changes what is learned, and then the tangent does too: the weighted location that training finds depends on the
noise that the tangent puts into the target.

**Time pairs** (`flow/pairs`, {py:meth}`~flow.MeanFlow.sample_times`). Two independent draws, $t'$ from `flow.time`
and $t''$ from `flow.second_time` (`flow.time` again when it is null), are ordered into $t \le s$. Both orders take
$s = \max(t', t'')$. They differ in $t$: `sort` takes the smaller draw, $t = \min(t', t'')$, while `clip` keeps
$t = t'$, so every pair with $t'' \le t'$ collapses to $s = t$. Then the first $\mathrm{round}(\text{share}\cdot B)$
rows of a batch of $B$ get $s = t$, where the share is `flow.flow_matching_share`. These are the flow matching rows of
[the target](#the-target).

In theory the pair distribution is a weighting. Drawing $(t, s)$ from a density weights the loss by it
([flow-time distributions](training.md#flow-time-distributions)), and the identity holds at each pair separately. For
an expressive enough network at $p = 0$, every distribution that covers $t < s$ therefore has the same solution,
$\bar u_{t,s}$. In practice the distribution decides which intervals the network fits best, and the share decides how
many rows train $\bar u^\theta_{t,t}$ directly.

| setting | $t'$ | $t''$ | order | in the papers' convention | source |
| --- | --- | --- | --- | --- | --- |
| `imf` (default) | $\mathrm{logit}\,t' \sim \mathcal N(0.4, 1)$ | the same law | `sort` | $\mathrm{LN}(-0.4, 1.0)$ twice | MeanFlow's ImageNet pairs and improved MeanFlow's |
| `mf` | $\mathrm{logit}\,t' \sim \mathcal N(0.6, 1.6^2)$ | $\mathrm{logit}\,t'' \sim \mathcal N(4.0, 1.6^2)$ | `clip` | $\mathrm{LN}(-0.6, 1.6)$ for $t^{\mathrm{MF}}$, $\mathrm{LN}(-4.0, 1.6)$ for $r^{\mathrm{MF}}$ | MeanFlow's CIFAR-10 pairs (py-meanflow v1) |

Under `clip` the realized share of $s = t$ is $\text{share} + (1 - \text{share})\,P(t'' \le t')$. For `mf`,
$P(t'' \le t') = \Phi\big(-3.4/(1.6\sqrt 2)\big) \approx 0.066$, with $\Phi$ the standard normal CDF,
so a share of 0.25 gives 0.30 and the default 0.75 gives 0.77. Under `sort` the rows with $s = t$ draw their $t$ from
$\min(t', t'')$, which leans toward the noise.

The share is one minus the papers' ratio of $r^{\mathrm{MF}} \neq t^{\mathrm{MF}}$. The tests compare both pair
settings with ports of the papers' samplers. The number of rows with $s = t$ is fixed at
$\mathrm{round}(\text{share}\cdot B)$, so the draw runs inside the captured training step. Batches are shuffled, so the
first rows are random examples.

**Tangent** (`flow.tangent`). The two settings differ in one thing: which estimate of the unknown
$u_t^{\text{target}}(x)$ moves $x$ in the derivative. Both keep $u_t^{\text{target}}(x \mid z)$ as the first term of
the target, both keep the stop-gradient, and neither passes gradient through the derivative.

- `conditional` (MeanFlow) moves $x$ along $u_t^{\text{target}}(x \mid z)$, the velocity of the sample's own
  conditional path. Given $x$ it is random, but its posterior mean is exact.
- `predicted` (improved MeanFlow's boundary-condition variant) moves $x$ along the network's own
  $\bar u^\theta_{t,t}(x \mid y)$, evaluated without gradient. By the boundary condition,
  $\bar u^\theta_{t,t}(x \mid y)$ estimates $u_t^{\text{target}}(x)$, so it can stand in for the velocity. Given $x$ it
  is deterministic, but it is biased until learned. It costs one more forward pass and needs a share above 0, since the
  rows with $s = t$ are the only ones that train $\bar u^\theta_{t,t}$ directly.

At $p = 0$ the two targets have the same posterior mean once $\bar u^\theta_{t,t}(x \mid y)$ equals
$u_t^{\text{target}}(x)$. They differ in their noise. At the fixed point $\bar u^\theta_{t,s} = \bar u_{t,s}$, the
identity turns the two targets into

$$
Y_{\text{conditional}} = \bar u_{t,s}(x)
+ \partial_x \psi_{t \to s}(x)\,\big(u_t^{\text{target}}(x \mid z) - u_t^{\text{target}}(x)\big),
\qquad
Y_{\text{predicted}} = \bar u_{t,s}(x) + u_t^{\text{target}}(x \mid z) - u_t^{\text{target}}(x),
$$

since $I + (s - t)\,\partial_x \bar u_{t,s} = \partial_x \psi_{t \to s}$ by the definition of $\bar u_{t,s}$. The
predicted target carries flow matching's own noise. The conditional target carries the same noise through the Jacobian
of the flow map, which amplifies it where the flow spreads samples apart and damps it where the flow gathers them. On
the Gaussian example, $\partial_x \psi_{0 \to 1}$ is the data's standard deviation times $I$. In theory neither tangent
is better everywhere; in practice they tie (below). A test checks that, with the exact $\bar u_{t,s}$, the predicted
target's loss equals flow matching's loss of the exact velocity.

Under the default `data.padding=mask`, the predicted tangent falls back to $u_t^{\text{target}}(x \mid z)$ on the padded
steps, whose loss weight is 0. The reason is the off-diagonal blocks of the chunk Jacobian
$\partial_x \bar u^\theta_{t,s}$: the derivative at a valid step reads the tangent at every step of the chunk. No row
ever fits the network's prediction on a weight-0 step. A predicted tangent there would let that untrained output move
the targets of valid steps. Under `hold` the padded steps are ordinary targets and use the predicted tangent too.

**Adaptive weight** (`flow.adaptive_power` $p$, `flow.adaptive_eps` $\delta$). The axis is which function of each
chunk's squared error training minimizes. Chunk $i$ of a batch of $B$ gets the **adaptive weight**
$\gamma_i = (\ell_i + \delta)^{-p}$, computed from its plain loss $\ell_i$ without gradient. The objective and its
gradient are

$$
\frac{1}{B}\sum_{i=1}^{B} \mathrm{sg}(\gamma_i)\,\ell_i,
\qquad
\nabla_\theta = \frac{1}{B}\sum_{i=1}^{B} \gamma_i\,\nabla_\theta \ell_i ,
$$

so each chunk's gradient is scaled by its own weight. Since
$\mathrm{sg}(\gamma_i)\,\nabla_\theta \ell_i = \nabla_\theta\, g_p(\ell_i)$ with
$g_p(\ell) = (\ell + \delta)^{1-p}/(1 - p)$ and $g_1(\ell) = \log(\ell + \delta)$, the weight trains the network on a
concave function of each chunk's squared error:

| $p$ | $g_p(\ell)$ | behaves like | the network learns |
| --- | --- | --- | --- |
| 0 | $\ell + \delta$ | the squared error | the posterior mean of the target |
| 1/2 | $2\sqrt{\ell + \delta}$ | a pseudo-Huber loss in the error norm (MeanFlow's motivation) | close to a median |
| 0.75 | $4\,(\ell + \delta)^{1/4}$ | a loss that grows slower than the error norm (MeanFlow's CIFAR-10 recipe) | a more robust location |
| 1 (default) | $\log(\ell + \delta)$ | a Cauchy-type loss that ignores the scale of $\ell$ | a mode-seeking location |

In theory only $p = 0$ learns the posterior mean that [the target](#the-target) relies on. For $p > 0$ the fixed point
solves $\mathbb E\big[\gamma\,(\bar u^\theta_{t,s}(x \mid y) - Y) \,\big|\, x\big] = 0$. It is a **weighted
location**, a mean weighted toward the targets near the prediction. The rows with $s = t$ learn the weighted location
of $u_t^{\text{target}}(x \mid z)$ given $x$ instead of the vector field. With the predicted tangent, the identity then
holds with this location in place of $u_t^{\text{target}}(x)$. The conditional tangent first multiplies the target's
noise by the matrix $I + (s - t)\,\partial_x \bar u^\theta_{t,s}$. A weighted location does not commute with that
matrix, so the conditional tangent's fixed point differs again. A $\delta$ large against $\ell$ makes $\gamma$
constant and brings back the mean. In practice $p = 1$ trains best here (below).

At $p = 1$ the reduction only rescales $\delta$. Under `example_mean`, $\ell_i = \ell_i^{\mathrm{sum}}/(2M_i)$, with
$M_i = \sum_k m_k$ the chunk's valid steps ([Reduction](training.md#reduction)). Since $\log$ ignores constant factors,

$$
\nabla_\theta \log\Big(\frac{\ell_i^{\mathrm{sum}}}{2M_i} + \delta\Big) = \nabla_\theta \log\big(\ell_i^{\mathrm{sum}} + 2M_i\,\delta\big).
$$

Switching the reduction therefore only multiplies $\delta$ by $2M_i \le 2H$ (32 at $H = 16$), and a chunk cut short by
padding weighs like a full one under either reduction. The objective's value, $\frac1B\sum_i \ell_i/(\ell_i + \delta)$,
then stays just below 1 as long as $\ell \gg \delta$, so it cannot show the fit ([Training](#training)).

**Published recipes.** The published recipes map to overrides:

```bash
uv run train.py experiment=meanflow flow/pairs=mf flow.flow_matching_share=0.25 \
    flow.adaptive_power=0.75                                         # MeanFlow, CIFAR-10
uv run train.py experiment=meanflow flow.tangent=predicted flow.flow_matching_share=0.5 \
    flow.adaptive_eps=0.01 'train.betas=[0.9,0.95]'                  # improved MeanFlow, boundary condition
```

Improved MeanFlow's released models also use an auxiliary velocity head, training-time guidance and in-context
conditioning; none of these are carried over.

**What we found** (selection episodes)

- **adaptive weight $p = 1$ > $p = 0$.** The largest effect of the search; $p = 0.5$ is also below $p = 1$.
- **share 0.75 > share 0.25.** A share of 0.9 adds nothing.
- **`imf` = `mf`; `conditional` = `predicted`.** Improved MeanFlow as published is level too and the predicted
  tangent is slower per step, while uniform pairs tie on the mean with far more seed-to-seed spread and are not
  shipped.
- **one-step corner rows = none.** Rows with $(t, s) = (0, 1)$ directly gained nothing and once made the `imf` pairs
  collapse, so the option was removed.

Try: `uv run train.py experiment=meanflow flow.adaptive_power=0.0`

## Sampling

How does a MeanFlow policy sample? It steps with the average velocity over each interval of the grid
$t_0 < t_1 < \dots < t_n = 1$, with step size $h_i = t_{i+1} - t_i$:

$$
X_{t_{i+1}} = X_{t_i} + h_i\,\bar u^\theta_{t_i, t_{i+1}}(X_{t_i} \mid y),
$$

one network evaluation per step. Every ODE solver of {doc}`sampling` replaces the average velocity over a step by an
estimate; they differ in which one.

| solver | in place of $\bar u_{t_i, t_{i+1}}(X_{t_i})$ | NFE per step | for exact models |
| --- | --- | --- | --- |
| `euler` | $u^\theta_{t_i}(X_{t_i} \mid y)$, the velocity at the left end | 1 | order 1; exact when the velocity is constant along the step |
| `heun` | the mean of the velocities at $X_{t_i}$ and at Euler's guess for $X_{t_{i+1}}$ | 2 | order 2 |
| `mean` | $\bar u^\theta_{t_i, t_{i+1}}(X_{t_i} \mid y)$, the average itself | 1 | exact on any grid |

For an exact average velocity the steps compose: a step from $t_0$ to $t_1$ and then to $t_2$ lands where one step from
$t_0$ to $t_2$ lands, by the flow property. In theory one step is then as good as any number, and the tests check this
on the Gaussian example: the `mean` sampler reaches the exact endpoint in 1, 2, 4 and 7 steps, on the uniform and the
`data_dense` grids. A learned $\bar u^\theta_{t,s}$ is not exact, so in practice the number of steps matters
([below](#meanflow-against-flow-matching)).

{py:class}`~flow.MeanFlowStep` takes these steps, one {py:meth}`~flow.LearnedFlow.average` call each. `sampler=mean`
sets `sampler.solver=mean` with one step from noise; `sampler.nfe` sets the number of steps, on the uniform or the
`data_dense` grid. {py:meth}`~flow.Sampler.check` refuses the `mean` solver for a policy trained with flow matching.

MeanFlow policies also run under `euler`, `heun` and `sde`, through the zero gap
([Computing the derivative](#computing-the-derivative)): they simulate $\bar u^\theta_{t,t}(x \mid y)$. That is a
velocity prediction, defined at $t = 0$, so these grids start at $t = 0$; only a scaled score starts later
([where sampling starts](sampling.md#where-sampling-starts)). With a guidance scale $w \ne 1$, the `mean` solver mixes
two average velocities, which mixes two flow maps and agrees with guidance only to first order in the step
([guidance](sampling.md#guidance)).

## Training

- **Warm-up.** `experiment=meanflow` sets `train.warmup_steps=5000` with `train.optimizer=adamw` and
  `train.weight_decay=0.0` (Adam). Warming up the learning rate helped (below). Muon trains at a constant rate,
  and {py:func}`~train.check_config` refuses it with warm-up.
- **float32.** The forward-mode pass runs in float32, so bf16 autocast cannot speed it up;
  {py:func}`~train.check_config` refuses `train.precision=bf16` with MeanFlow.
- **Cost.** A MeanFlow step costs more than a flow matching step: it adds the forward-mode pass, and the predicted
  tangent one more forward pass.
- **Losses.** The logged `loss` is the adaptively weighted objective, which sits just below 1 at $p = 1$ and cannot
  show the fit. The plain loss $\ell$ is logged three ways, listed in {py:attr}`~flow.MeanFlow.metric_names`.
  `loss_ref` is MeanFlow's reference loss over every row, `loss_u` covers the rows with $s > t$, and `loss_v` those
  with $s = t$, where $\bar u^\theta_{t,t}(x \mid y)$ is regressed onto $u_t^{\text{target}}(x \mid z)$ as in
  $\mathcal L_{\text{CFM}}$. A batch without rows of one kind gives NaN. They appear in the `meanflow/` section of the
  dashboard ({doc}`../logging`). Compare them only within one pairs setting: `loss_ref` mixes the two kinds of rows in
  proportion to the share, `loss_u` averages over the gaps that the pairs draw, and none of them reaches zero, since
  the target is noisy given $x$ ([comparing training losses](training.md#comparing-training-losses)).

**What we found** (selection episodes)

- **5000 warm-up steps > none.** MeanFlow's recipe trains with them.
- **learning rate 3e-4 > 1e-3.** MeanFlow is sensitive to the learning rate, unlike flow matching; 1e-4 also loses
  slightly.
- **160k steps = 80k; EMA 0.9999 = 0.999.** Longer training and a slower average add nothing at the chosen setting.

Try: `uv run train.py experiment=meanflow train.lr=1e-3`

## MeanFlow against flow matching

What does MeanFlow buy, and at what price? The axis is the quality of a chunk per network evaluation at sampling time.
The two objectives share most of their machinery:

| | flow matching (`velocity_mse`) | MeanFlow |
| --- | --- | --- |
| noisy chunk | $x = \alpha_t z + \beta_t \epsilon$ | the same, on the same path |
| network output | $o_t^\theta(x \mid y) = u^\theta_t(x \mid y)$ | $\bar u^\theta_{t,s}(x \mid y)$, with the gap $h$ as one more input |
| flow times per row | $t$ from `flow.time`, uniform by default | a pair $t \le s$ from `flow/pairs`, with a share of rows at $s = t$ |
| target | $u_t^{\text{target}}(x \mid z)$ | $u_t^{\text{target}}(x \mid z)$ plus $(s - t)$ times the network's derivative along $(v, 1)$, under $\mathrm{sg}$ |
| weight on a chunk's loss | none | $\gamma_i = (\ell_i + \delta)^{-p}$ |
| cost of a training step | one forward and one backward pass | the same, with the forward pass in forward mode to carry the derivative; one more forward pass for the predicted tangent |
| sampler | `euler`, `heun` or `sde`, several steps | `mean`, one step by default; the others at $h = 0$ |

At $s = t$ and $p = 0$ a MeanFlow row is a flow matching row, with the same target and the same squared error. Only the
distribution of its flow time differs. In theory, at guidance scale $w = 1$, both samplers are exact: one step of an
exact average velocity, and the ODE of an exact vector field simulated with ever smaller steps, both return
$p_{\mathrm{data}}(\cdot \mid y)$. In practice they fail in different places. Flow matching with few steps pays the
discretization error of [the first section](#why-one-step-needs-more-than-the-vector-field), which a better network
cannot remove. MeanFlow pays training error instead: its network fits a function of two times, from a target that moves
with its own weights, under a weight that leaves the posterior mean for $p > 0$. The findings below measure the balance.

The comparison ran before Muon and the scaled score became the defaults. Flow matching was trained as it was then:
the velocity loss ($\mathcal L_{\text{CFM}}$) and Adam (AdamW at weight decay 0), without warm-up. MeanFlow used the
same optimizer with 5000 warm-up steps, and a third arm trained flow matching with the same warm-up. Every arm was
trained on fresh seeds and scored on the held-out episodes, at E = 4 with averaged weights.

**What we found**

- **one MeanFlow step >> one flow matching step.** Flow matching trained with the same warm-up gains at one step too
  but stays far behind, so the warm-up does not explain the gap.
- **two MeanFlow steps > two flow matching steps.** Flow matching ran on its best two-evaluation grid, `data_dense`.
- **flow matching at 10 NFE > one MeanFlow step (weak).** The interval is too wide to claim a difference either way.
- **two MeanFlow steps = flow matching at 10 NFE.** MeanFlow needs two network evaluations per query instead of ten;
  flow matching stays the default.
- **MeanFlow under Euler on $\bar u^\theta_{t,t}(x \mid y)$ = flow matching at 10 NFE.** The MeanFlow network also
  works as an ordinary vector field.
- **MeanFlow at E = 4 > MeanFlow at E = 1, 2.** Executing fewer actions per query does not buy the latency back;
  flow matching loses at E < 4 too.

Reproduce: `uv run train.py -m experiment=meanflow`; the flow matching controls and the evaluation commands are in
`conf/experiment/meanflow.yaml`.
