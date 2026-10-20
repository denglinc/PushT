# Training

`train.py` trains one policy, evaluates it every `eval.interval` steps and saves it. The whole optimizer step is one
CUDA graph, replayed for every step, and its loss is compiled by TorchInductor.

```bash
uv run train.py                                              # the default: Muon, compiled loss, CUDA graph
uv run train.py train.optimizer=adamw                        # AdamW on every parameter
uv run train.py train.compile=false train.cuda_graph=false   # eager, for debugging
```

## What a run does

```text
main                          Hydra entry point: run directory, arm name, Logger
└── run_training
    ├── check_config          refuses settings that do not fit together; no data, no GPU
    ├── set_seed
    ├── PushtChunkDataset     every training window, on the device
    ├── build_policy          the policy, then one copy per weight average
    ├── build_optimizers
    ├── build_loss            compute_loss, compiled or eager
    ├── step loop             a permutation per epoch, the learning rate, train_step
    │   ├── every log.train_interval steps   read the window sums back, log them
    │   └── every eval.interval steps        evaluate, log, save checkpoint.pt
    ├── Logger.log_late       the late evaluations pooled: the run's headline
    └── Logger.save_model     checkpoint.pt as a W&B model artifact
```

{py:func}`~train.main` is the Hydra entry point. Hydra composes `conf/config.yaml` with the command line and creates
the run directory; `main` names the arm (the overrides other than the seed and the logging keys, `base` without any),
opens a {py:class}`~logger.Logger` there and calls {py:func}`~train.run_training`. If training raises, the W&B run is
marked failed and the error propagates.

{py:func}`~train.run_training` first calls {py:func}`~train.check_config`, which needs neither data nor a GPU, so a bad
combination fails before any data or GPU work. It instantiates the flow objective and the sampler, so their
constructors' checks run there too. {py:func}`~train.set_seed` then seeds numpy, torch and every CUDA device: the seed
fixes the initialization, the batch order, the flow times $t$, the noise $\epsilon$ and, with `model.cond_dropout` > 0,
the condition-dropout draws, while evaluation draws from streams of its own (symbols as in {doc}`notation`).

{py:func}`~data.download_pusht` and {py:func}`~data.load_pusht_zarr` read the demonstrations, and
{py:class}`~data.PushtChunkDataset` builds every window once, on the device: K history frames and the goal, all
normalized (together the guiding variable $y$, which the policy encodes), a clean chunk $z$ of H normalized relative
action targets and its loss weight (0 on masked padding, see {doc}`data`).
{py:func}`~evaluation.build_policy` instantiates `config.model` with the flow objective (see {doc}`model`), and each
decay of `train.ema_decays` gets a deep copy of the policy that never trains.

The step loop counts optimizer updates. An epoch is one pass over a fresh `torch.randperm` drawn on the device and cut
into `len(dataset) // batch_size` batches; each step copies its slice of the permutation into `batch_index`, sets the
scheduled learning rate and runs the step (next section). The step itself gathers the batch by index.

Every `eval.interval` steps, and at the last step, {py:func}`~evaluation.evaluate_policy` plays the selection episodes
(env seeds 0 and up) with the default setting: the first entry of `eval.weights` at `eval.exec_horizon`. At the last
`eval.num_late` evaluations it also plays every entry of `eval.weights`, at `eval.exec_horizon` and at each execution
horizon of `eval.late_exec_horizons` up to H. {py:meth}`~logger.Logger.log_evaluation` logs each setting, and
{py:func}`~evaluation.save_checkpoint` writes `checkpoint.pt` (raw and averaged weights, the policy's data-dependent
constructor arguments, the normalizer, the resolved config, the step) after every evaluation. After the last step
{py:meth}`~logger.Logger.log_late` pools each setting's late evaluations under keys of its own
(`late/<metric>_E<e>_<weights>`), and the default setting's pooled success, `late/success_rate`, is the run's headline;
{py:meth}`~logger.Logger.save_model` then logs the checkpoint as a W&B model artifact. See {doc}`evaluation` and
{doc}`logging`.

## The captured step

`train_step`, a closure inside {py:func}`~train.run_training`, is the unit that becomes one CUDA graph. In order, it:

1. gathers the batch with {py:meth}`~data.PushtChunkDataset.batch` at `batch_index`, a static buffer the loop fills
   in place;
2. computes the loss with the function {py:func}`~train.build_loss` returned, under a bf16 autocast when
   `train.precision=bf16`: {py:meth}`~model.FlowMatchingPolicy.compute_loss` encodes the guiding variable $y$ and
   calls {py:meth}`~flow.FlowMatching.loss`, which draws the flow times $t$ and the noise $\epsilon$ itself;
3. backpropagates;
4. takes the global gradient norm before clipping (and clips when `train.grad_clip` > 0); neither call checks the norm
   for finite values, so nothing waits for the host;
5. steps every optimizer;
6. adds the loss, the gradient norm and the objective's {py:attr}`~flow.FlowMatching.metric_names` to on-device window
   sums;
7. moves every weight average $\bar\theta$ toward the new parameters $\theta$.

The loop runs the first `GRAPH_WARMUP_STEPS` (3) steps eagerly on a side stream, as capture requires. At the next step
it sets the gradients to None, so that they are allocated from the graph's memory pool, captures `train_step` into a
`torch.cuda.CUDAGraph` and replays it; every later step is one replay. A replay re-runs the same kernels on the same
memory: the batch changes because `batch_index` changes, and the flow times, the noise and the condition dropout (which
replaces $y$ by the null label $\varnothing$ with probability $p_{\mathrm{uncond}}$, `model.cond_dropout`) are fresh
draws of torch's generator. On the CPU, or with `train.cuda_graph=false`, every step runs eagerly.

{py:func}`~train.build_loss` compiles {py:meth}`~model.FlowMatchingPolicy.compute_loss` with `torch.compile`
(`fullgraph=True`) when `train.compile` is set and the device is a GPU. Attention runs on the math SDPA backend, faster
than the fused kernels at a chunk of 16 tokens. `fallback_random` keeps torch's own random draws, so the compiled loss
draws the flow times, noise and dropout mask that the eager one draws, and every kernel keeps Inductor's first config,
so a seed repeats bit for bit whatever the compile cache holds. The compiled and eager losses agree to rounding, not bit
for bit: `train.compile=false` reproduces runs from before the compiled loss bit for bit. The first step compiles; the
compiled loss is then captured with the rest of the step. Only the loss is compiled, and the optimizers run eagerly
inside the graph: compiling torch 2.9.1's Muon step silently broke training. Each run resets Dynamo, because the jobs of
a `-m` sweep share one process and would otherwise reach Dynamo's recompile limit.

```{admonition} Code that runs inside the captured step
:class: important

- **No host syncs and no control flow on tensor values.** No `.item()`, no printing of tensors, no Python `if` or
  `bool()` on a tensor: the step runs on the GPU stream without the host.
- **`torch.where`, not masked assignment.** Condition dropout in {py:meth}`~model.FlowMatchingPolicy.condition` picks
  the learned encoding of the null label $\varnothing$ (`null_condition`) with `torch.where` for this reason.
- **Captured tensors must outlive the graph.** A graph keeps writing into every tensor it captured; if one is freed, a
  later allocation in its memory is silently overwritten. The loop keeps its buffers (`batch_index`, the window sums,
  the parameter lists) alive for the whole run.
- **A Python number read during capture is frozen.** The learning rate is therefore a device tensor that the schedule
  fills in place. torch's Muon reads its learning rate as a number, so Muon trains at a constant rate, and
  {py:func}`~train.check_config` refuses Muon with a schedule or a warm-up.
- **The loss is read back only at log steps.** Every `log.train_interval` steps the loop reads the window sums (the
  only recurring host sync of training; one synchronize after capture starts the throughput clock), logs their means
  with {py:meth}`~logger.Logger.log_training` and zeroes them. A non-finite mean loss then stops the run with
  `FloatingPointError`, so a NaN surfaces at the next log step, not at the step that produced it.
- **No graph breaks in the loss.** The compiled loss uses `fullgraph=True`: code under
  {py:meth}`~model.FlowMatchingPolicy.compute_loss` must trace without a break, or the first step fails.
```

The logged `train/loss` is the objective $\mathcal L(\theta)$ of [One training
step](flow/training.md#one-training-step), estimated on each batch of the logging window and averaged over them. With
uniform times, the `sum` reduction and no masked step, $\mathcal L(\theta)$ is the conditional flow matching loss
$\mathcal L_{\text{CFM}}$ under `velocity_mse` and $\mathcal L_{\text{DDPM}}$ under `scaled_score_mse`. The default
`mask` padding leaves the padded steps out, so the default run logs the masked form of $\mathcal L_{\text{DDPM}}$. Its
absolute value says little: the loss has a floor that even the best network cannot remove, and the floor depends on the
objective and the data. [Comparing training losses](flow/training.md#comparing-training-losses) says when two runs'
losses compare. `train/throughput` counts optimizer steps per second of training alone: it leaves out the warm-up,
compile and capture steps and the time spent in evaluation.

## Optimizers

{py:func}`~train.build_optimizers` splits the policy's parameters with
{py:meth}`~model.FlowMatchingPolicy.parameter_groups` into the three roles of {py:class}`~model.ParameterGroups`:
the hidden weight matrices, the boundary matrices (the layers that read the inputs or write the prediction) and the
vectors (biases, positions, the learned encoding of the null label $\varnothing$, the time frequencies). See
{doc}`model` for which layers each network puts where.

- **AdamW** (`train.optimizer=adamw`) moves every parameter. Its weight decay $\mathrm{wd}$ (`train.weight_decay`) is
  decoupled: it shrinks the weights directly instead of entering the gradient, and it acts on the matrices only. Each
  step multiplies a matrix by $1 - \mathrm{lr}\cdot\mathrm{wd}$, with $\mathrm{lr}$ = `train.lr`, a decay time of
  $1/(\mathrm{lr}\cdot\mathrm{wd})$ steps; the vectors never decay. At weight decay 0 it steps bit for bit as torch's
  Adam over all parameters. It is fused on the GPU, and capturable with the graph, where its groups share one
  learning-rate tensor.
- **Muon** (`train.optimizer=muon`, the default) is torch's Muon on the hidden matrices; AdamW at `train.lr` moves
  the boundary matrices and the vectors. One Muon step on a hidden matrix $W \in \mathbb R^{A \times B}$ has three
  stages:
  1. **Momentum.** It keeps a momentum buffer, an exponential average of the gradients with factor 0.95, and mixes it
     with the current gradient by Nesterov's look-ahead (torch's defaults). Call the result $G$.
  2. **Orthogonalization.** With the singular value decomposition $G = U \Sigma V^\top$, Muon replaces $G$ by
     $U V^\top$, which keeps the singular vectors of $G$ and sets every singular value to 1. Five Newton-Schulz
     iterations approximate $U V^\top$ with matrix products alone. The approximation is rough: torch's iteration
     leaves the singular values roughly between 0.5 and 1.5.
  3. **Scaling.** For a $G$ of full rank $\min(A, B)$, $U V^\top$ has root mean square $1/\sqrt{\max(A, B)}$ over its
     entries. Muon multiplies it by $0.2\sqrt{\max(A, B)}$ (`adjust_lr_fn="match_rms_adamw"`), so every hidden matrix
     moves by a root mean square of about 0.2 times `train.muon_lr`, the size of a typical AdamW update. This is why
     `train.muon_lr` is on AdamW's scale.

  Muon's weight decay multiplies $W$ by $1 - \mathrm{lr}\cdot\mathrm{wd}$ with the unscaled rate `train.muon_lr` as
  $\mathrm{lr}$, so with `train.muon_lr` equal to `train.lr` the decay time is AdamW's.

{py:func}`~train.build_optimizers` returns AdamW, which the schedule drives, and the list of every optimizer the step
calls.

**Muon against AdamW.** The two settings of `train.optimizer` differ along one axis: how a hidden matrix moves. The
boundary matrices and the vectors follow AdamW at `train.lr` under both.

| on a hidden matrix | AdamW (`train.optimizer=adamw`) | Muon (`train.optimizer=muon`) |
| --- | --- | --- |
| update direction | entry by entry: each entry's momentum divided by the running root mean square of its gradient | the whole matrix: the momentum orthogonalized, every singular value near 1 (torch's five Newton-Schulz steps leave them roughly between 0.5 and 1.5) |
| update size | `train.lr` times a ratio of order one, per entry | a root mean square of about 0.2 times `train.muon_lr`, the same for every hidden matrix |
| weight decay per step | the factor $1 - \mathrm{lr}\cdot\mathrm{wd}$ at `train.lr` | the factor $1 - \mathrm{lr}\cdot\mathrm{wd}$ at `train.muon_lr` |
| cost per step | elementwise | five Newton-Schulz iterations of matrix products |
| learning-rate schedule | any | constant (see the box on the captured step) |

Intuitively, AdamW scales each entry's step by that entry's own gradient history, while Muon moves the matrix by about
the same amount along each pair of singular vectors of $G$, so directions with small gradients are not left behind. Both
descend the same loss. In theory, for an expressive enough network, they therefore aim at the same minimizer, the
posterior mean of [Regression learns a posterior mean](flow/training.md#regression-learns-a-posterior-mean). This sets
weight decay aside: decoupled decay is not a term of the loss, and it moves each optimizer's fixed point in its own way.
The two differ in where a run of 80k steps gets with a finite network, which is a question of practice; the findings
below answer it.

**What we found** (phase 5, with $\mathcal L_{\text{CFM}}$: weight decays against Adam, Muon against AdamW at weight
decay 0.01; phase 7 for the combination with the scaled score, `scaled_score_mse`):

- **Muon > AdamW.** Narrowly: held-out success is ahead by the Welch interval but not when paired by seed. The gain in
  mean max reward is the firmer evidence, and Muon's mean max reward is the most consistent over seeds.
- **Muon > Adam (weak).** Success is ahead within the noise; mean max reward is clearly higher.
- **`muon_lr` 3e-4 > 1e-3 (weak) >> 3e-3.** Three times AdamW's update size trains stably but ends with a higher
  training loss and no better success; ten times is unstable in every seed, and not every run recovered.
- **AdamW (weight decay 0.01) = Adam.** On held-out success, on fresh seeds; the decay raises the training loss.
- **weight decay 0 = 0.001 = 0.01 = 0.05.** Success shows no resolved change (the averaged weights' mean rises with the
  decay within the noise, and the raw and slower-averaged weights do not share the rise) while the training loss rises
  with the decay; 0.01 is kept by a tie-break fixed before the runs, the lower training loss among the cells near the
  best.
- **Muon with the scaled score > either alone.** On held-out success, on fresh seeds; each gain is about the same with
  or without the other, and no interaction is detected.

Reproduce (the preset pins AdamW and `velocity_mse`, and scores raw, 0.999 and 0.9999 weights; the confirmations
train fresh seeds and score them on the held-out episodes):

```bash
uv run train.py -m experiment=optimizer                                   # 5a: weight decay
uv run train.py -m experiment=optimizer train.optimizer=muon \
    'train.muon_lr=3e-4,1e-3,3e-3' train.weight_decay=0.01                 # 5b: Muon's learning rate
uv run train.py -m experiment=optimizer train.weight_decay=0.01 \
    name=confirm 'seed=range(600,606)'                                     # 5c: AdamW, fresh seeds
uv run train.py -m experiment=optimizer train.optimizer=muon train.muon_lr=3e-4 \
    train.weight_decay=0.01 name=confirm 'seed=range(600,606)'             # 5c: Muon, fresh seeds
uv run evaluation.py runs=runs/confirm/<run>                              # 5c: held-out confirmation
uv run train.py -m name=default_confirm 'seed=range(600,606)'             # 7: Muon with the scaled score
uv run evaluation.py runs=runs/default_confirm/<run>                      # 7: held-out
```

## Learning rate and schedule

`train.lr` sets AdamW's rate, `train.schedule` its shape after `train.warmup_steps` steps of linear warm-up. At
optimizer step $k$ (counted from 1) of $N$ = `train.steps`, with $N_{\mathrm w}$ = `train.warmup_steps`:

$$
\mathrm{lr}(k) =
\begin{cases}
\mathrm{lr}\,k/N_{\mathrm w} & k \le N_{\mathrm w}, \\
\mathrm{lr} & k > N_{\mathrm w},\ \text{constant}, \\
\tfrac12\,\mathrm{lr}\,\bigl(1 + \cos\bigl(\pi\,\tfrac{k - N_{\mathrm w}}{N - N_{\mathrm w}}\bigr)\bigr)
  & k > N_{\mathrm w},\ \text{cosine}.
\end{cases}
$$

Under the graph the loop writes this rate into AdamW's learning-rate tensor before each replay; without it, into each
group's number. With the defaults (constant, no warm-up) it never touches it. Muon's rate is fixed, so a schedule or a
warm-up needs `train.optimizer=adamw`; {py:func}`~train.check_config` refuses the rest. `experiment=meanflow` trains
MeanFlow with AdamW at weight decay 0 and warm-up, the setting its recipe was tuned with; MeanFlow's own training-loop
findings are on {doc}`flow/meanflow`.

**What we found** (phase 4, in nano-PushT, with Adam and $\mathcal L_{\text{CFM}}$):

- **lr 1e-4 = 3e-4 = 1e-3, constant.** Success is flat across a factor of ten.
- **constant > cosine at 1e-4; constant > cosine (weak) at 3e-4 and 1e-3.** Decaying to zero never helps. At 1e-4 success
  drops in the second half of training instead of holding. A plausible reading, not tested: the small late rate
  freezes the policy before it settles, while the weight average already removes the noise a decay would.

Reproduce (the preset trains AdamW, at the default weight decay 0.01, with `velocity_mse`; add
`train.weight_decay=0` for phase 4's Adam, which AdamW at 0 matches bit for bit; a sweep is a product, so the schedule
is a second launch):

```bash
uv run train.py -m experiment=training_loop
uv run train.py -m experiment=training_loop train.schedule=cosine train.warmup_steps=1000
```

## Weight averages

Each decay $\lambda$ of `train.ema_decays` keeps an exponential moving average $\bar\theta$ of the network
parameters $\theta$, updated inside the captured step after every optimizer step with `torch._foreach_lerp_`:

$$
\bar\theta \leftarrow \bar\theta + (1 - \lambda)(\theta - \bar\theta).
$$

Unrolling the update writes the average as a weighted mean of past parameters:

$$
\bar\theta_n
\overset{(i)}{=} \lambda\,\bar\theta_{n-1} + (1 - \lambda)\,\theta_n
\overset{(ii)}{=} \lambda^n\theta_0 + \sum_{j=0}^{n-1}(1 - \lambda)\lambda^j\,\theta_{n-j},
$$

where $\theta_n$ are the parameters after $n$ optimizer steps, (i) rewrites the update, and (ii) applies it $n$ times
from $\bar\theta_0 = \theta_0$, the copy made at initialization. The weights sum to 1 and shrink by a factor $\lambda$
per step back. Counting $\theta_0$ at age $n$, their mean age is $\lambda(1 - \lambda^n)/(1 - \lambda)$. Once
$n \gg 1/(1 - \lambda)$ it approaches $\lambda/(1 - \lambda)$: about $10^3$ steps for $\lambda = 0.999$ and $10^4$ for
$0.9999$. At the first evaluation, after $10^4$ steps, the 0.9999 average has not reached that limit; its mean age is
about $6.3\cdot10^3$. The initialization keeps the weight $\lambda^n$. At the first evaluation that weight is
$5\cdot10^{-5}$ for $\lambda = 0.999$ but $e^{-1} \approx 0.37$ for $\lambda = 0.9999$. The decay 0.9999 is not a
default, but `train.ema_decays` can add it, and an early 0.9999 curve still averages in the initialization. After 80k
steps the weight is negligible for both.

The averages never train: they are deep copies of the policy with gradients off. `eval.weights` names the weights to
evaluate (`raw` or `ema<decay>`, one of `train.ema_decays`; {py:func}`~train.check_config` refuses others), and its
first entry (the default, `ema0.999`) gives the learning curve and the headline. The checkpoint holds the raw weights
and every average; {py:func}`~evaluation.load_policy` rebuilds either, and `evaluation.py` scores the 0.999 average by
default.

**What we found** (the runs of phases 2 and 3, with Adam and $\mathcal L_{\text{CFM}}$; confirmed on fresh seeds in
phase 4):

- **averaged (0.999) > raw.** Paired within each run over the configurations of phases 2 and 3, and again on fresh
  seeds and held-out episodes; averaging also narrows the seed-to-seed spread slightly.
- **EMA 0.999 = 0.9999.** 0.999 is kept for its shorter memory, which stays meaningful when a larger batch shortens
  training.

Try: `uv run train.py 'train.ema_decays=[0.999,0.9999]' 'eval.weights=[ema0.999,ema0.9999,raw]'`

## Precision, batch size and length

Float32 matmuls always run in TF32 on the GPU: float32 storage and accumulation, with tensor-core products.
`train.precision=bf16` adds a bf16 autocast around the loss, which in effect runs the network in bf16; the weights, the
optimizer state and the loss stay float32, and network outputs come back as float32. MeanFlow refuses bf16: it
differentiates the network forward in float32 (see {doc}`flow/meanflow`). The default trains batches of 128 windows
for 80k steps.

**What we found** (phase 4, in nano-PushT, with Adam and $\mathcal L_{\text{CFM}}$):

- **bf16 = TF32.** Success is tied; bf16 is barely faster at H = 16 and pays off only for long chunks, so TF32 is kept.
- **batch 64 = 128 = 256 = 512, at equal samples.** The learning rate scaled with the square root of the batch. 128 is
  the efficient choice per sample for the captured step.
- **Adam = AdamW (weight decay 0.01); beta2 0.95 = 0.999; clipping at 1.0 = none.** A factorial of the three; no main
  effect is detected.
- **160k steps = 80k.** The DiT's success curves plateau well before 80k steps.
- **earlier recipe = base, at equal samples.** AdamW at 1e-4 with betas (0.9, 0.95), weight decay 0.01 and batch 32
  does as well, but needs more optimizer steps for the same samples and takes longer in wall-clock time.
- **captured loop > DataLoader loop, in speed.** Data on the GPU, the loss read only at log steps, fused optimizers and
  the CUDA graph gain the most where kernel launches bound the step (small batch, short chunk), the fused optimizer
  and the graph most of all; at batch 128 the DiT is bound by arithmetic and gains less.
- **compiled loss > eager, in speed (PushT).** Compiling the loss makes each step faster again.

Try: `uv run train.py train.precision=bf16` or `uv run train.py train.steps=160000`
