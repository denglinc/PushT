# Flow matching

The {py:mod}`flow` module turns a network into a generative model of action chunks. It has three parts. A
**probability path** interpolates from Gaussian noise to clean chunks. The **training objective** noises a chunk along
that path and regresses the network onto a target computed from the chunk and the noise. The **samplers** start from
noise and simulate the learned vector field until they reach a chunk. The path and the training recipe are Hydra
options:

```bash
uv run train.py flow/path=gvp flow.recipe=velocity_mse   # the GVP path, regressing the velocity
uv run train.py experiment=meanflow                      # MeanFlow: flow=meanflow sampler=mean, AdamW with warm-up
```

This page builds the mathematical objects the other flow pages use: the path, the vector field it defines, the four
ways a network can parameterize that field, and the two paths that ship. {doc}`training` covers the loss,
{doc}`sampling` the ODE and SDE samplers and {doc}`meanflow` the one-step objective. The notation is that of
Holderrieth and Erives' notes on flow matching; {doc}`the notation page <../notation>` lists every symbol and translates
it to the other references.

## Conventions

Time runs from noise at $t = 0$ to data at $t = 1$, and sampling simulates upward in $t$. We call $t$ the **flow time**,
to tell it from the steps of an episode. A data point is a clean, normalized **chunk** $z \sim p_{\mathrm{data}}$: the
$H$ relative action targets of {doc}`../data`, so $z \in \mathbb R^{H \times 2}$. The **noise**
$\epsilon \sim p_{\mathrm{init}} = \mathcal N(0, I)$ has the shape of a chunk and is drawn independently of $z$.

The policy samples chunks for a **guiding variable** $y$, the goal and the history frames ({doc}`../model`). Its data
distribution is therefore $p_{\mathrm{data}}(\cdot \mid y)$, and its network sees $y$. Only the data distribution
depends on $y$: the path that noises a chunk and the regression targets are the same for every $y$. Everything on this
page is therefore a statement for one fixed $y$, with $p_{\mathrm{data}}(\cdot \mid y)$ in place of $p_{\mathrm{data}}$.
The marginal quantities below drop $y$ from the notation, and the network keeps it. When we need the field for the
null label $\varnothing$ of classifier-free guidance, we write it $u_t^{\text{target}}(x \mid \varnothing)$.

## The Gaussian probability path

Our goal is a vector field whose ODE carries noise $X_0 \sim p_{\mathrm{init}}$ to chunks $X_1 \sim p_{\mathrm{data}}$.
The endpoints alone do not determine such a field: many ways lead from noise to data. Flow matching first fixes what
should happen in between. A **probability path** $(p_t)_{0 \le t \le 1}$ prescribes the distribution of $X_t$ at every
time, with $p_0 = p_{\mathrm{init}}$ and $p_1 = p_{\mathrm{data}}$. It is built from simple pieces, one per chunk.

Let $\alpha_t$ and $\beta_t$ be **noise schedulers**: continuously differentiable functions of the flow time with
$\alpha_0 = \beta_1 = 0$ and $\alpha_1 = \beta_0 = 1$, $\alpha_t$ increasing and $\beta_t$ decreasing. Their time
derivatives are $\dot\alpha_t$ and $\dot\beta_t$. The schedulers say how much signal and how much noise a noisy chunk
holds at time $t$.

:::{admonition} Definition: the Gaussian conditional probability path
:class: important

For a chunk $z$, the **conditional probability path** is

$$
p_t(x \mid z) = \mathcal N(x;\, \alpha_t z,\, \beta_t^2 I), \qquad 0 \le t \le 1 .
$$

Its endpoints are $p_0(\cdot \mid z) = \mathcal N(0, I) = p_{\mathrm{init}}$ and $p_1(\cdot \mid z) = \delta_z$, the
distribution that always returns $z$.
:::

In other words, the conditional path gradually turns the noise into the single chunk $z$. Drawing the chunk first gives
the **marginal probability path** $p_t$, which turns the noise into the whole data distribution:

$$
\begin{aligned}
x &= \alpha_t z + \beta_t \epsilon \sim p_t, \quad z \sim p_{\mathrm{data}},\ \epsilon \sim \mathcal N(0, I)
&& \blacktriangleright\ \text{sampling} \\
p_t(x) &= \int p_t(x \mid z)\, p_{\mathrm{data}}(z)\, \mathrm dz
&& \blacktriangleright\ \text{density} \\
p_0 &= p_{\mathrm{init}}, \quad p_1 = p_{\mathrm{data}}
&& \blacktriangleright\ \text{endpoints}
\end{aligned}
$$

Note that we can sample $p_t$ without ever evaluating its density: training needs only the first line.

:::{admonition} Example: the linear path
:class: tip

With $\alpha_t = t$ and $\beta_t = 1 - t$, the noisy chunk is $x = t z + (1 - t)\epsilon$, the straight-line
interpolation from the noise to the chunk. The notes call this the **CondOT path**; it is the default here,
`flow/path=linear`.
:::

## The conditional vector field

A path says where the samples should be. A vector field says how they move. We first find a field that realizes the
conditional path, one chunk at a time.

The conditional path is the law of the **conditional flow** $\psi_t(\epsilon \mid z) = \alpha_t z + \beta_t\epsilon$,
which moves each noise sample $\epsilon$ along a curve that starts at $\epsilon$ and ends at $z$. Its velocity is the
**conditional vector field**. Writing the position as $x = \psi_t(\epsilon \mid z)$,

$$
\frac{\mathrm d}{\mathrm dt}\psi_t(\epsilon \mid z)
\overset{(i)}{=} \dot\alpha_t z + \dot\beta_t \epsilon
\overset{(ii)}{=} \dot\alpha_t z + \dot\beta_t\,\frac{x - \alpha_t z}{\beta_t}
\overset{(iii)}{=} \Big(\dot\alpha_t - \frac{\dot\beta_t}{\beta_t}\,\alpha_t\Big) z + \frac{\dot\beta_t}{\beta_t}\, x
= u_t^{\text{target}}(x \mid z),
$$

where (i) differentiates the flow at fixed $(z, \epsilon)$, (ii) writes the noise through the current position,
$\epsilon = (x - \alpha_t z)/\beta_t$, and (iii) collects terms. The last form is the notes' formula, a function of $x$.
The first form, $u_t^{\text{target}}(x \mid z) = \dot\alpha_t z + \dot\beta_t \epsilon$, is the same field written
through the noise that produced $x$; the code uses it because it needs no division by $\beta_t$.

Starting from $X_0 \sim p_{\mathrm{init}}$, the ODE $\frac{\mathrm d}{\mathrm dt}X_t = u_t^{\text{target}}(X_t \mid z)$
therefore gives $X_t \sim p_t(\cdot \mid z)$ for every $t$, and every trajectory ends at $z$. Many other vector fields
produce the same conditional path; flow matching picks this one, the velocity of the conditional flow. On the linear
path the field is $u_t^{\text{target}}(x \mid z) = (z - x)/(1 - t)$. Along the trajectory that starts at $\epsilon$ it
equals the constant $z - \epsilon$: each sample moves on a straight line, at constant speed, to its chunk.

## The marginal vector field

At first sight, a conditional vector field seems useless: its ODE only regenerates the one chunk $z$. It turns out that
averaging these fields over the chunks gives a field that generates the whole data distribution.

:::{admonition} Theorem: the marginalization trick
:class: important

Define the **marginal vector field** as the conditional one averaged over the posterior of $z$ given $x$:

$$
u_t^{\text{target}}(x) = \int u_t^{\text{target}}(x \mid z)\,\frac{p_t(x \mid z)\,p_{\mathrm{data}}(z)}{p_t(x)}\,\mathrm dz .
$$

Then $X_0 \sim p_{\mathrm{init}}$ and $\frac{\mathrm d}{\mathrm dt}X_t = u_t^{\text{target}}(X_t)$ give $X_t \sim p_t$
for every $t$. In particular $X_1 \sim p_{\mathrm{data}}$: the marginal vector field converts noise into data.
:::

Intuitively, every chunk $z$ that could have produced the noisy chunk $x$ proposes the velocity that would carry $x$ to
it. The weight $p_t(x \mid z)\,p_{\mathrm{data}}(z)/p_t(x)$ is, by Bayes' rule, the posterior probability that $x$ came
from $z$. The marginal vector field averages the proposals by how much we believe in each.

:::{dropdown} Proof, through the continuity equation
A vector field $u_t$ moves $X_0 \sim p_0$ along a path $p_t$ exactly when the pair satisfies the **continuity
equation** $\partial_t p_t(x) = -\mathrm{div}(p_t u_t)(x)$: the change of probability at $x$ is the net inflow of
probability mass. The conditional pair satisfies it, because $u_t^{\text{target}}(\cdot \mid z)$ generates
$p_t(\cdot \mid z)$. Then

$$
\begin{aligned}
\partial_t p_t(x)
&\overset{(i)}{=} \int \partial_t p_t(x \mid z)\, p_{\mathrm{data}}(z)\,\mathrm dz
\overset{(ii)}{=} -\int \mathrm{div}\big(p_t(\cdot \mid z)\,u_t^{\text{target}}(\cdot \mid z)\big)(x)\, p_{\mathrm{data}}(z)\,\mathrm dz \\
&\overset{(iii)}{=} -\mathrm{div}\Big(\int p_t(\cdot \mid z)\,u_t^{\text{target}}(\cdot \mid z)\,p_{\mathrm{data}}(z)\,\mathrm dz\Big)(x)
\overset{(iv)}{=} -\mathrm{div}\big(p_t\,u_t^{\text{target}}\big)(x),
\end{aligned}
$$

where (i) differentiates the definition of $p_t$ under the integral, (ii) uses the continuity equation of each
conditional pair, (iii) swaps the integral and the divergence, and (iv) multiplies and divides by $p_t(x)$ and uses the
definition of $u_t^{\text{target}}(x)$. The marginal pair satisfies the continuity equation, so the marginal field
generates the marginal path. The FM Guide of Lipman et al. states the regularity conditions.
:::

**The field as a posterior mean.** For $t < 1$ the noise is a function of $x$ and $z$,
$\epsilon = (x - \alpha_t z)/\beta_t$. Averaging over $z$ given $x$ is therefore the same as averaging over the pair
$(z, \epsilon)$ given $x$. With the conditional field written through the noise, the marginal vector field becomes a
combination of two posterior means:

$$
u_t^{\text{target}}(x) = \mathbb E\big[\dot\alpha_t z + \dot\beta_t \epsilon \,\big|\, x\big]
= \dot\alpha_t\, D_t(x) + \dot\beta_t\, \mathbb E[\epsilon \mid x],
\qquad D_t(x) = \mathbb E[z \mid x] .
$$

Here $z \sim p_{\mathrm{data}}$ and $\epsilon \sim \mathcal N(0, I)$ are conditioned on
$\alpha_t z + \beta_t\epsilon = x$. The **denoiser** $D_t(x)$ is the expected clean chunk given the noisy one, and
$\mathbb E[\epsilon \mid x]$ is the expected noise. Taking the same posterior mean of
$x = \alpha_t z + \beta_t \epsilon$, which is known given $x$, ties the two together. We call this the **path
identity**:

$$
x = \alpha_t\, D_t(x) + \beta_t\, \mathbb E[\epsilon \mid x].
$$

In other words, a noisy chunk splits exactly into its expected signal and its expected noise. At the two ends the
posterior means are explicit. At $t = 0$ the noisy chunk is the noise, so $\mathbb E[\epsilon \mid x] = x$ and
$D_0(x) = \mathbb E[z]$, the mean chunk for the observation. At $t = 1$ it is the chunk, so $D_1(x) = x$ and
$\mathbb E[\epsilon \mid x] = 0$.

Note that a denoiser is a mean, not a chunk. $D_0 = \mathbb E[z]$ is the **regression policy**: exactly what a
behavior-cloning policy trained with a squared error outputs. When the demonstrations for one observation take two
different routes, it can lie between them. A generative policy exists to avoid this: it samples one route instead of
averaging them.

**Two random variables called $X_t$.** The first is $X_t = \alpha_t z + \beta_t\epsilon$, built from a chunk and noise
drawn independently of each other; training uses it. The second is the solution $X_t$ of the theorem's ODE, which pairs
its start $X_0$ deterministically with its end $X_1$ through the flow of the marginal vector field. At every $t$ the two
have the same distribution $p_t$, but their joint laws over time differ. On the linear path, for example, each
conditional path is a straight line run at constant speed, while the ODE's trajectories in general accelerate.
{doc}`sampling` and {doc}`meanflow` rely on this distinction.

Why can a network learn this field without knowing $z$? A least-squares regression onto a target that depends on
$(z, \epsilon)$ learns the target's posterior mean given what the network sees. Regressing onto the conditional vector
field therefore learns the marginal one. {doc}`training` states and proves this.

## Scores

The **score** of a density, $\nabla \log p(x)$, points where the log density rises fastest. We need the score of the
marginal path for two reasons. The SDE sampler adds it to the drift to inject noise without leaving the path
({doc}`sampling`), and the default network output is defined through it.

The conditional score of the Gaussian path is explicit,
$\nabla \log p_t(x \mid z) = -(x - \alpha_t z)/\beta_t^2 = -\epsilon/\beta_t$. Like the vector field, the marginal score
is its posterior mean. For $t < 1$,

$$
\nabla \log p_t(x)
\overset{(i)}{=} \frac{\int \nabla p_t(x \mid z)\, p_{\mathrm{data}}(z)\,\mathrm dz}{p_t(x)}
\overset{(ii)}{=} \int \nabla \log p_t(x \mid z)\,\frac{p_t(x \mid z)\,p_{\mathrm{data}}(z)}{p_t(x)}\,\mathrm dz
\overset{(iii)}{=} -\frac{\mathbb E[\epsilon \mid x]}{\beta_t},
$$

where (i) writes $\nabla \log p_t = \nabla p_t / p_t$ and moves the gradient inside the integral that defines $p_t$,
(ii) writes $\nabla p = p\,\nabla \log p$ for the conditional density, and (iii) inserts the conditional score. This is
**Tweedie's formula** in its noise form. For $0 < t < 1$, where $\alpha_t > 0$, the path identity also gives the data
form, $D_t(x) = \big(x + \beta_t^2\,\nabla \log p_t(x)\big)/\alpha_t$: the denoiser takes one step up the log density
and rescales.

The code works with $\beta_t$ times the score, the **scaled score**:

$$
\beta_t\, \nabla \log p_t(x) = -\mathbb E[\epsilon \mid x], \qquad
\text{with the conditional regression target}\quad \beta_t\, \nabla \log p_t(x \mid z) = -\epsilon .
$$

The notes train a score network $s_t^\theta$ for $\nabla \log p_t$, or a **noise predictor** $\epsilon_t^\theta$ for
$\mathbb E[\epsilon \mid x]$, related by $\epsilon_t^\theta = -\beta_t s_t^\theta$. A network that outputs the scaled
score outputs $\beta_t s_t^\theta = -\epsilon_t^\theta$: a noise predictor up to sign. Its target $-\epsilon$ is
bounded at every $t$, while the score's target $-\epsilon/\beta_t$ grows without bound near the data. This is the
default output (`flow.recipe=scaled_score_mse`).

## An example with Gaussian data

Every quantity above has a closed form when the data are Gaussian. The tests check the code on this example
(`tests/test_flow.py`), and the other flow pages use it to make their arguments concrete.

:::{admonition} Example: Gaussian data
:class: tip

Let each coordinate of the chunk be independent, $z \sim \mathcal N(m, s^2)$, and write
$v_t = \alpha_t^2 s^2 + \beta_t^2$ for the variance of a coordinate of $x$. Then $x \sim \mathcal N(\alpha_t m, v_t)$
per coordinate, and

$$
\begin{aligned}
D_t(x) &= m + \frac{\alpha_t s^2}{v_t}\,(x - \alpha_t m), \qquad
\mathbb E[\epsilon \mid x] = \frac{\beta_t}{v_t}\,(x - \alpha_t m), \qquad
\nabla \log p_t(x) = -\frac{x - \alpha_t m}{v_t}, \\
u_t^{\text{target}}(x) &= \dot\alpha_t\, D_t(x) + \dot\beta_t\, \mathbb E[\epsilon \mid x]
= \dot\alpha_t m + \frac{\alpha_t\dot\alpha_t s^2 + \beta_t\dot\beta_t}{v_t}\,(x - \alpha_t m) .
\end{aligned}
$$

The marginal ODE is affine in $x$, and its trajectories are $X_t = \alpha_t m + \sqrt{v_t}\,X_0$. Every start $X_0$ ends
at $X_1 = m + s X_0$.
:::

The example shows three things that hold in general. First, the two random variables called $X_t$ differ: the training
pair $(z, \epsilon)$ is independent, while the ODE maps $X_0$ to the one chunk $m + s X_0$. Second, a straight
conditional path does not make a marginal trajectory of constant velocity. On the linear path,
$X_t = t m + \sqrt{t^2 s^2 + (1 - t)^2}\,X_0$ has constant velocity only when $s = 0$, that is, when the data are a
single point. Third, $D_0(x) = m$ for every $x$: at the noise end the denoiser knows only the mean chunk.

## Four parameterizations

The network does not have to output the vector field. It can output any quantity from which the vector field can be
computed. Which quantities qualify?

The answer rests on one observation. At a fixed time, both the noisy chunk and the conditional vector field are linear
in the pair $(z, \epsilon)$:

:::{admonition} Proposition: one pair of unknowns, four names
:class: important

$$
\begin{pmatrix} x \\ u_t^{\text{target}}(x \mid z) \end{pmatrix}
= \begin{pmatrix} \alpha_t & \beta_t \\ \dot\alpha_t & \dot\beta_t \end{pmatrix}
\begin{pmatrix} z \\ \epsilon \end{pmatrix},
\qquad
\det \begin{pmatrix} \alpha_t & \beta_t \\ \dot\alpha_t & \dot\beta_t \end{pmatrix}
= -\kappa_t, \quad \kappa_t = \dot\alpha_t\beta_t - \alpha_t\dot\beta_t .
$$

Where $\kappa_t \ne 0$ the matrix is invertible. Assume $\kappa_t > 0$, a condition we impose below (**What $\kappa_t$
measures**). For $t \in (0, 1)$, where $\alpha_t$ and $\beta_t$ are positive too, any one of
$u_t^{\text{target}}(x \mid z)$, $z$ and $-\epsilon$ then determines the others once $x$ is known:

$$
\begin{aligned}
z &= \frac{\beta_t\, u_t^{\text{target}}(x \mid z) - \dot\beta_t\, x}{\kappa_t},
& -\epsilon &= \frac{\alpha_t\, u_t^{\text{target}}(x \mid z) - \dot\alpha_t\, x}{\kappa_t}, \\
u_t^{\text{target}}(x \mid z) &= \frac{\dot\beta_t\, x + \kappa_t\, z}{\beta_t},
& u_t^{\text{target}}(x \mid z) &= \frac{\dot\alpha_t\, x - \kappa_t\, \epsilon}{\alpha_t} .
\end{aligned}
$$

Each identity is affine in $(z, \epsilon)$, with coefficients that depend only on $t$ and $x$. Taking the posterior
mean given $x$ of both sides keeps it true with $z \to D_t(x)$, $\epsilon \to \mathbb E[\epsilon \mid x]$ and
$u_t^{\text{target}}(x \mid z) \to u_t^{\text{target}}(x)$.
:::

In other words, the conditional quantities are four names for the one unknown pair $(z, \epsilon)$, and their posterior
means are four names for the one unknown pair $(D_t(x), \mathbb E[\epsilon \mid x])$. The marginal form of the first
identity is the notes' **denoiser formula**,
$D_t(x) = \big(\beta_t\, u_t^{\text{target}}(x) - \dot\beta_t\, x\big)/\kappa_t$. Substituting
$-\mathbb E[\epsilon \mid x] = \beta_t \nabla \log p_t(x)$ into the last gives the notes' **conversion formula**

$$
u_t^{\text{target}}(x) = a_t\, \nabla \log p_t(x) + b_t\, x, \qquad
a_t = \beta_t^2\,\frac{\dot\alpha_t}{\alpha_t} - \dot\beta_t\beta_t = \frac{\beta_t\kappa_t}{\alpha_t}, \qquad
b_t = \frac{\dot\alpha_t}{\alpha_t} .
$$

In the Diffusion Guide's language, $a_t$ and $b_t$ come from a forward noising SDE. Written in reversed time
$\tau = 1 - t$, the SDE $\mathrm dx = -b_t\,x\,\mathrm d\tau + \sqrt{2a_t}\,\mathrm dW_\tau$ has our path as its
perturbation kernel: its drift coefficient is $f = -b_t$ and its squared diffusion coefficient is $g^2 = 2a_t$. The
conversion formula is the probability-flow ODE of this SDE.

**What $\kappa_t$ measures.** $\kappa_t$ appears in every conversion to or from the vector field. It does not appear
between the denoiser and the expected noise, which the path identity relates directly. Three readings explain it.

- It is minus the determinant above. Where it is nonzero, $x$ and the conditional vector field together determine
  $(z, \epsilon)$. Where it vanishes, the vector field is a multiple of $x$ and adds nothing that $x$ does not already
  say.
-  It is the speed at which the path trades noise for signal. Since
  $\frac{\mathrm d}{\mathrm dt}(\alpha_t/\beta_t) = \kappa_t/\beta_t^2$, a positive $\kappa_t$ makes the ratio
  $\alpha_t/\beta_t$ increase strictly, so the path never adds noise back.
- It sets the sign of $a_t = \beta_t\kappa_t/\alpha_t$, half the squared diffusion coefficient of that forward SDE.

Monotone schedulers give $\kappa_t \ge 0$, since both of its terms are nonnegative. They do not give $\kappa_t > 0$:
$\alpha_t = t^2$ and $\beta_t = 1 - t$ have $\kappa_0 = \dot\alpha_0 = 0$. The conversions from the vector field divide
by $\kappa_t$, and so does the SDE sampler's drift, so **we require $\kappa_t > 0$ on $[0, 1]$**. At the ends this reads
$\dot\alpha_0 > 0$ and $\dot\beta_1 < 0$, because $\kappa_0 = \dot\alpha_0$ and $\kappa_1 = -\dot\beta_1$. Both shipped
paths satisfy it with a constant: $\kappa_t = 1$ on the linear path and $\pi/2$ on GVP ([below](#paths)).

**The parameterizations.** Each column below is one choice of network output $o$. Every conversion to the vector field
has the affine form $u = A_t\, x + B_t\, o$, with coefficients that depend only on $t$. The network can output the
first three; the score is reached only by conversion, for the SDE sampler.

| | vector field | denoiser | scaled score (default) | score |
| --- | --- | --- | --- | --- |
| conditional target | $\dot\alpha_t z + \dot\beta_t \epsilon$ | $z$ | $-\epsilon$ | $-\epsilon/\beta_t$ |
| posterior mean | $u_t^{\text{target}}(x)$ | $D_t(x)$ | $-\mathbb E[\epsilon \mid x] = \beta_t \nabla\log p_t(x)$ | $\nabla \log p_t(x)$ |
| $A_t$, $B_t$ in $u = A_t x + B_t o$ | $0$, $1$ | $\dot\beta_t/\beta_t$, $\kappa_t/\beta_t$ | $\dot\alpha_t/\alpha_t$, $\kappa_t/\alpha_t$ | $b_t$, $a_t$ |
| from the vector field | itself | $(\beta_t u - \dot\beta_t x)/\kappa_t$ | $(\alpha_t u - \dot\alpha_t x)/\kappa_t$ | $(\alpha_t u - \dot\alpha_t x)/(\beta_t\kappa_t)$ |
| conversion to $u$ divides by zero at | nowhere | $t = 1$ | $t = 0$ | $t = 0$ |
| code | {py:class}`~flow.VelocityPrediction` | {py:class}`~flow.DataPrediction` | {py:class}`~flow.ScaledScorePrediction` | {py:meth}`~flow.Coefficients.velocity_to_score` |

**The same in theory, different in practice.** For exact posterior means the four columns define the same vector
field, so in theory the choice cannot change the samples. A trained network is never exact, and the conversion passes
its error on with the gain $B_t$. An error $\delta$ in a denoiser becomes the error $(\kappa_t/\beta_t)\,\delta$ in the
vector field, which grows near the data. An error in a scaled score becomes $(\kappa_t/\alpha_t)\,\delta$, which grows
near the noise. The vector field has gain 1 everywhere. The same factors also weight the flow times differently in
each recipe's loss ({doc}`training`), and they are why a scaled score starts sampling at $t_0 = 0.05$
({doc}`sampling`).

**The singular ends.** Each prediction fails at the end where its job is easiest. At $t = 1$ the noisy chunk is the
clean one, so the denoiser's answer is $x$ itself, and the vector field rests on the residual $D_t(x) - x$, which is of
order $\beta_t$. At $t = 0$ the noisy chunk is the noise, so the scaled score's answer is $-x$, and the vector field
rests on a residual of order $\alpha_t$. For exact posterior means the numerator vanishes together with the denominator.
Substituting the path identity into the numerators shows that each numerator is the denominator times the vector field:

$$
\dot\beta_t\, x + \kappa_t\, D_t(x) = \beta_t\big(\dot\alpha_t D_t(x) + \dot\beta_t\, \mathbb E[\epsilon \mid x]\big),
\qquad
\dot\alpha_t\, x - \kappa_t\, \mathbb E[\epsilon \mid x] = \alpha_t\big(\dot\alpha_t D_t(x) + \dot\beta_t\, \mathbb E[\epsilon \mid x]\big),
$$

so both conversions return $u_t^{\text{target}}(x)$ for every $t \in (0, 1)$. The limits at the ends exist where the
posterior means do. At $t = 0$ that is every $x$, since $p_0 = \mathcal N(0, I)$ is positive everywhere, with
$\mathbb E[\epsilon \mid x] = x$ and $D_0(x) = \mathbb E[z]$. At $t = 1$ it is only where the data have a density, which
a finite set of demonstrations does not. A network's output, in contrast, carries an error that does not vanish with the
denominator, and the conversion multiplies that error by the unbounded gain $B_t$. The code therefore never evaluates a
prediction at its singular time ([Numerics](#numerics)).

## Paths

Two paths ship, as `conf/flow/path/linear.yaml` and `conf/flow/path/gvp.yaml`. Besides the schedulers and $\kappa_t$,
the table gives the logarithm of each path's **signal-to-noise ratio** (SNR) $\alpha_t^2/\beta_t^2$ and the shape of its
conditional trajectory:

| path | $\alpha_t$ | $\beta_t$ | $\kappa_t$ | log-SNR $2\log(\alpha_t/\beta_t)$ | conditional trajectory |
| --- | --- | --- | --- | --- | --- |
| {py:class}`~flow.LinearPath` (default) | $t$ | $1 - t$ | $1$ | $2\,\mathrm{logit}\,t$ | a straight line, at the constant velocity $z - \epsilon$ |
| {py:class}`~flow.GVPPath` | $\sin\frac{\pi t}{2}$ | $\cos\frac{\pi t}{2}$ | $\pi/2$ | $2\log\tan\frac{\pi t}{2}$ | a quarter ellipse from $\epsilon$ to $z$ |

On the linear path $\dot\alpha_t = 1$ and $\dot\beta_t = -1$, so $\kappa_t = (1 - t) + t = 1$. On GVP
$\dot\alpha_t = \frac{\pi}{2}\beta_t$ and $\dot\beta_t = -\frac{\pi}{2}\alpha_t$, so
$\kappa_t = \frac{\pi}{2}(\beta_t^2 + \alpha_t^2) = \frac{\pi}{2}$. GVP is the generalized variance-preserving path of
SiT and the cosine scheduler of the FM Guide. It keeps $\alpha_t^2 + \beta_t^2 = 1$. The chunks are z-scored, so each
coordinate of $z$ has unit variance, and on GVP each coordinate of $x$ keeps unit variance at every $t$. On the linear
path that variance dips to $1/2$ at $t = 1/2$. This holds per coordinate, not for the covariance, because the steps of a
chunk are correlated.

The name CondOT says less than it seems. For a single target chunk, the straight line from $\epsilon$ to $z$ is the
optimal-transport path, and among all conditional flows the linear one minimizes an upper bound on the kinetic energy
of the marginal flow. This does not make the marginal flow an optimal-transport map. On every Gaussian path, GVP
included, the marginal vector field is a gradient field and is therefore the cheapest field that follows its own
marginal path.

How do the two paths compare? It turns out that they are the same path, run on a different clock and scale.

:::{admonition} Proposition: the two paths differ only by a clock and a scale
:class: important

Match the signal-to-noise ratios, $\tan\frac{\pi r}{2} = \frac{t}{1 - t}$. For every $r \in [0, 1]$,

$$
\sin\tfrac{\pi r}{2}\, z + \cos\tfrac{\pi r}{2}\, \epsilon = s_r\,\big(t_r\, z + (1 - t_r)\, \epsilon\big),
\qquad
s_r = \sin\tfrac{\pi r}{2} + \cos\tfrac{\pi r}{2}, \quad
t_r = \frac{\sin\frac{\pi r}{2}}{\sin\frac{\pi r}{2} + \cos\frac{\pi r}{2}} .
$$

The GVP sample at time $r$ is the linear sample at time $t_r$, scaled by $s_r \in [1, \sqrt 2]$. The exact marginal
flows obey the same relation when they start from the same noise, $X^{\text{GVP}}_r = s_r X^{\text{lin}}_{t_r}$. Since
$t_0 = 0$, $t_1 = 1$ and $s_0 = s_1 = 1$, **with exact vector fields the two paths carry the same noise to the same
chunk.**
:::

This is the FM Guide's equivalence of schedulers, made concrete for our two paths. In theory, then, the path does not
change the samples. In practice the network is trained and simulated on one clock, and three things differ.

1. **The noise levels a uniform $t$ trains on.** Equal $t$ means a different signal-to-noise ratio on the two paths
   (the log-SNR column), and each recipe's loss weight is a different function of $t$ ({doc}`training`).
2.  **The trajectories the solver follows.** The Euler method's error depends on how much the marginal trajectories
   accelerate. With one step it lands on $\dot\alpha_0\,\mathbb E[z] + (1 + \dot\beta_0)\,\epsilon$, which keeps the
   noise on GVP ({doc}`sampling`).
3.  **The scale of the velocity targets.** The conditional flow's kinetic energy
   $\mathbb E\int_0^1 \lVert u_t^{\text{target}}(x \mid z)\rVert^2\,\mathrm dt$ is $\pi^2/8 \approx 1.23$ times larger
   on GVP than on the linear path. The velocity targets are larger on GVP, and so is the velocity loss's floor, by
   $(\pi/2)^2$ at $t = 0$. Velocity losses therefore do not compare across paths ([Comparing training
   losses](training.md#comparing-training-losses)).

The findings below measure these effects; the path itself is, in theory, neutral.

**What we found**

- **GVP = linear, averaged over the recipes.** The path is a second-order choice; linear stays the default.
- **GVP > linear (weak), with the velocity loss.** GVP is slightly ahead, within the seed-to-seed noise.
- **GVP > linear, with the data loss.** This fits GVP's weight on the data loss, which falls later and more gently
  (see {doc}`training`).
- **linear > GVP, with a scaled score sampled from $t = 0.001$; GVP = linear from $t = 0.01$ or $0.05$.** The
  default start is 0.05.
- **linear >> GVP at one Euler step, for the predictions without a scaled score.** One step of the Euler method on
  GVP leaves the noise in the chunk ({doc}`sampling` derives it).

These ran on AdamW, before Muon became the default. Reproduce:
`uv run train.py -m experiment=flow_matching sampler.start_time=0.001` (the study sampled the scaled score from that
old start), then rescore the runs without retraining:
`uv run evaluation.py -m runs=runs/flow_matching split=selection 'sampler.start_time=0.01,0.05'` and
`uv run evaluation.py runs=runs/flow_matching split=selection sampler.nfe=1`.

:::{admonition} Summary
:class: note

The Gaussian path $x = \alpha_t z + \beta_t \epsilon$ makes the noisy chunk and the conditional vector field linear in
$(z, \epsilon)$. The marginal vector field, which carries noise to data, is the posterior mean
$\dot\alpha_t D_t(x) + \dot\beta_t\,\mathbb E[\epsilon \mid x]$, and the path identity ties the two posterior means
together. With $\kappa_t > 0$, the vector field, the denoiser, the scaled score and the score determine each other
through affine conversions. With exact models neither the parameterization nor the path changes the samples. What they
change is how training weights the flow times and how the conversions and the solver amplify errors, and that is what
the findings measure.
:::

## Map of `flow.py`

Each base class names one mathematical object. The abstract ones leave a few lines of closed-form math to their
subclasses; {py:class}`~flow.Recipe`, {py:class}`~flow.FlowMatching` and {py:class}`~flow.Sampler` are concrete:

| base class | defined by | implementations |
| --- | --- | --- |
| {py:class}`~flow.ProbabilityPath` | the noise schedulers $\alpha_t$, $\beta_t$ and their derivatives (and $\kappa_t$); returns {py:class}`~flow.Coefficients` | {py:class}`~flow.LinearPath`, {py:class}`~flow.GVPPath` |
| {py:class}`~flow.TimeSampler` | a CDF and its inverse, restricted to a support | {py:class}`~flow.UniformTime`, {py:class}`~flow.LogitNormalTime` |
| {py:class}`~flow.Prediction` | a regression target, the map to the vector field, a singular time | {py:class}`~flow.VelocityPrediction`, {py:class}`~flow.DataPrediction`, {py:class}`~flow.ScaledScorePrediction` |
| {py:class}`~flow.Recipe`, in `RECIPES` | a prediction and the space its loss compares in | `velocity_mse`, `data_mse`, `data_output_velocity_loss`, `scaled_score_mse` |
| {py:class}`~flow.FlowMatching` | a path, a flow-time distribution, a recipe, a reduction; the loss | itself (flow matching), {py:class}`~flow.MeanFlow` |
| {py:class}`~flow.ODE`, {py:class}`~flow.SDE` | a drift (and a diffusion coefficient $\sigma_t$) | {py:class}`~flow.LearnedFlow`, {py:class}`~flow.LangevinFlow` |
| {py:class}`~flow.Simulator` | one step and its network evaluations; {py:meth}`~flow.Simulator.simulate` runs a grid | {py:class}`~flow.Euler`, {py:class}`~flow.Heun`, {py:class}`~flow.EulerMaruyama`, {py:class}`~flow.MeanFlowStep` |
| {py:class}`~flow.Sampler` | solver, NFE, time grid, start time, SDE noise, guidance | (one dataclass) |

**Coefficients.** {py:meth}`~flow.ProbabilityPath.__call__` evaluates a path at a tensor of flow times and returns the
{py:class}`~flow.Coefficients` $(\alpha_t, \beta_t, \dot\alpha_t, \dot\beta_t, \kappa_t)$, in the fields `alpha`,
`beta`, `d_alpha`, `d_beta` and `D`. The field `D` holds $\kappa_t$, not the denoiser $D_t$; its name comes from the
denominator of the denoiser formula. {py:meth}`~flow.ProbabilityPath.at` does the same at one grid time, in float64.
{py:meth}`~flow.Coefficients.interpolate` builds $x = \alpha_t z + \beta_t\epsilon$,
{py:meth}`~flow.Coefficients.velocity` builds $u_t^{\text{target}}(x \mid z) = \dot\alpha_t z + \dot\beta_t\epsilon$,
and {py:meth}`~flow.Coefficients.data_to_velocity`, {py:meth}`~flow.Coefficients.scaled_score_to_velocity`,
{py:meth}`~flow.Coefficients.velocity_to_scaled_score` and {py:meth}`~flow.Coefficients.velocity_to_score` are the
conversions of the table above.

**Predictions.** A {py:class}`~flow.Prediction` names what the network outputs. It holds the conditional regression
target ({py:meth}`~flow.Prediction.target`), the conversion of an output to the vector field
({py:meth}`~flow.Prediction.to_velocity`), which the samplers simulate, and the flow time where that conversion
divides by zero ({py:attr}`~flow.Prediction.singular_time`).

Nothing in `flow.py` knows about Push-T. The network enters as a function `predict(x, t[, s])` that maps noisy
chunks and flow times to an output of the chunk's shape, with the guiding variable $y$ fixed; `s`, MeanFlow's end
time, is omitted for flow matching. The policy of {doc}`../model` builds that function and hands it over in two
places:

- **Training.** {py:meth}`~model.FlowMatchingPolicy.compute_loss` encodes the guiding variable $y$ once and calls
  {py:meth}`~flow.FlowMatching.loss` with `predict` bound to it. The loss draws times and noise, calls `predict` once
  and returns the scalar objective with its metrics ({doc}`training`).
- **Sampling.** {py:meth}`~model.FlowMatchingPolicy.sample_actions` wraps `predict`, the path and the prediction in a
  {py:class}`~flow.LearnedFlow`. Its {py:meth}`~flow.LearnedFlow.drift` calls `predict` and maps the output to
  $u_t^\theta(x \mid y)$ with {py:meth}`~flow.Prediction.to_velocity`. With a guidance scale $w \ne 1$
  (`sampler.cfg_scale`), `predict` is {py:meth}`~model.FlowMatchingPolicy.predictor`'s mix of the network's outputs
  with and without $y$, and the same conversion turns the mix into the classifier-free guided vector field
  $\tilde u_t^\theta(x \mid y)$ ({doc}`sampling` shows why). {py:meth}`~flow.Sampler.sample` then runs a
  {py:class}`~flow.Simulator` from noise to $t = 1$: the ODE of that vector field, the SDE of {doc}`sampling`, or
  MeanFlow's steps ({doc}`meanflow`).

## Numerics

- **Nothing is clamped**: not the flow times, not the conversions, not the intermediate samples. Denominators that
  vanish at an endpoint are kept away from it by rules.
- **The rules are checked when an object is built**, not on every step. {py:func}`~train.check_config` builds the
  objects before training starts, so an unsupported combination fails early.
  - **Exact endpoints.** {py:class}`~flow.ProbabilityPath` checks in float32, when it is built, that
    $\alpha_0 = 0$, $\alpha_1 = 1$, $\beta_0 = 1$ and $\beta_1 = 0$ exactly. The samplers rely on exact zeros, not
    approximate ones. {py:meth}`~flow.GVPPath.beta` therefore writes $\beta_t$ as $\sin\frac{\pi(1 - t)}{2}$: the
    float32 value of $\cos\frac{\pi}{2}$ is not 0. A path with a closed-form $\kappa_t$ overrides
    {py:meth}`~flow.ProbabilityPath.D`; otherwise it is computed from its definition.
  - **Time supports.** {py:class}`~flow.TimeSampler` refuses a support whose extreme draws would round to 0 or 1 in
    float32.
  - **Singular times.** {py:class}`~flow.FlowMatching` refuses, when the objective is built, a velocity-space loss
    whose time support (`flow.time.support`) contains the prediction's singular time. Draws never land exactly on 0 or
    1, but inside such a support they come arbitrarily close, where the conversion's gain is unbounded.
    {py:meth}`~flow.Sampler.times` refuses a time grid that is not increasing in float32 or that would evaluate a
    prediction at its singular time.
- **float32**: simulation, conversions and losses run in float32, also under bf16 autocast; network outputs are cast
  to float32 before use. Time quantiles and grid coefficients are computed in float64.
- **Non-finite samples raise.** {py:meth}`~flow.Sampler.sample` checks the final chunk (a non-finite value stays
  non-finite through later steps) and also raises if the number of network evaluations differs from
  {py:attr}`~flow.Sampler.nfe`.

```{toctree}
:hidden:

training
sampling
meanflow
```
