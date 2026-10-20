# Configuration and experiments

Hydra composes every run from `conf/`: each key is an option and each config group a choice of component. The command
line overrides any of them, sweeps over lists of them, or loads a preset from `conf/experiment/`.

```bash
uv run train.py flow/path=gvp model.conditioning=concat seed=1   # a config group and two keys
uv run train.py -m 'data.obs_horizon=2,4' 'seed=range(0,3)'      # a sweep: six jobs, one after another
uv run train.py -m experiment=horizons                           # a preset's grid
uv run evaluation.py runs=runs/horizons sampler.nfe=4            # score the finished runs again
```

## Hydra layout

`train.py` composes `conf/config.yaml`; `evaluation.py` composes `conf/evaluation.yaml`. A group is overridden with a
slash (`flow/path=gvp` loads `conf/flow/path/gvp.yaml`), a key with a dot (`flow.recipe=velocity_mse`, the conditional
flow matching loss $\mathcal L_{\text{CFM}}$; symbols as in {doc}`notation`).

| group | options | builds |
| --- | --- | --- |
| `model/` | `dit`, `mlp` | {py:class}`~model.DiTPolicy`, {py:class}`~model.MLPPolicy` |
| `flow/` | `default`, `meanflow` | the objective: {py:class}`~flow.FlowMatching`, {py:class}`~flow.MeanFlow` |
| `flow/path/` | `linear`, `gvp` | {py:class}`~flow.LinearPath` (the CondOT path), {py:class}`~flow.GVPPath` |
| `flow/time/` | `uniform`, `logit_normal` | {py:class}`~flow.UniformTime`, {py:class}`~flow.LogitNormalTime` |
| `flow/pairs/` | `imf`, `mf` | MeanFlow's $(t, s)$ pairs (used by `flow=meanflow` only) |
| `sampler/` | `default`, `mean` | {py:class}`~flow.Sampler`; `mean` is MeanFlow's one-step sampler |
| `experiment/` | the presets below | overrides of any key, and a sweep |

Every component node names its class in `_target_`, and `hydra.utils.instantiate` builds it, nested nodes first: the
flow's `path` and `time` come with the objective. {py:func}`~evaluation.build_policy` instantiates the model's node
with the data-dependent sizes (frame features, goal features, action dimensions, K and H) and
`flow=instantiate(config.flow)`, so the policy carries its objective. The sampler is built on its own and is not part
of the policy: {py:func}`~evaluation.load_policy` rebuilds a policy from the resolved config in its checkpoint, and
`evaluation.py` samples it with the sampler of its own command line.

{py:func}`~train.check_config` runs first and fails on combinations that cannot run, before any data or GPU work: a
history shorter than two frames, an execution horizon outside $[1, H]$, evaluation weights that are not tracked, Muon
with a schedule or warm-up, classifier-free guidance without condition dropout, MeanFlow in bf16, and a sampler that
cannot run on the objective ({py:meth}`~flow.Sampler.check`): a time grid that would evaluate the network's output
where its conversion to a vector field is undefined, or the `mean` sampler without MeanFlow. The constructors check
their own arguments as well. A test composes every job of every preset and passes it through
{py:func}`~train.check_config`.

Runs keep the repository root as their working directory (`hydra.job.chdir: false`), so every run shares the
downloaded dataset in `data_dir`.

## Runs

```text
runs/<name>/<time>_seed_<seed>/          uv run train.py ...
runs/<name>/<time>/<job>_seed_<seed>/    uv run train.py -m ...
    .hydra/                              the composed config, Hydra's config, the command line's overrides
    log.csv                              every row of scalars at its step
    config.json                          the resolved config, the arm and facts about the run
    git/<repository>.diff                commit, status and uncommitted diff of the code
    wandb/                               the W&B run's local files (W&B on)
    checkpoint.pt                        raw and averaged weights, normalizer, config, step
    evaluations/<time>[/<job>]/          the scores of evaluation.py: log.csv, config.json
```

{py:func}`~train.main` takes the directory Hydra creates and hands it to the {py:class}`~logger.Logger`.
{py:func}`~evaluation.save_checkpoint` rewrites `checkpoint.pt` at every evaluation, so an interrupted run keeps its
last evaluated weights; {py:func}`~evaluation.finished_checkpoints` treats a run as finished once its checkpoint is at
`train.steps`. {doc}`logging` describes the log files.

A run's **arm** is the set of overrides that change its experiment: Hydra's `override_dirname` (the overrides,
sorted, joined by commas) without the seed, `name`, `experiment`, `data_dir` and the logging keys, or `base` when
nothing is left. `uv run train.py flow/path=gvp seed=1` is arm `flow/path=gvp`. The W&B run is named
`<arm>/seed_<seed>` in the group `<name>`, and `config.json` stores `arm`, so grouping runs by `arm` pools an arm's
seeds. A sweep's values become the overrides of its jobs, so they name the arms; the preset itself names only the
group.

`evaluation.py runs=<dir>` scores every finished run below `<dir>`. Each scored run gets
`<run>/evaluations/<time>[/<job>]/` and a W&B run of job type `evaluate` in the training run's group, named
`<arm>/seed_<seed>/<overrides>`. Runs the sampler does not fit (classifier-free guidance without condition dropout,
Heun's method for a denoiser, the `mean` sampler for flow matching) are reported and skipped. {doc}`evaluation`
covers the splits and the scoring.

## Presets

A preset in `conf/experiment/` sets `name`, pins some settings of its study and, except `smoke`, defines a grid as a
Hydra sweep, which runs only with `-m`; without it the fixed settings train once. A sweep runs its jobs one after
another in one process; a failed job is marked failed in W&B and the next job still runs. A list on the command line
replaces the preset's, which adds seeds from a second shell:
`uv run train.py -m experiment=horizons 'seed=range(304,308)'`. Each file's comments give its follow-up commands
(further grids, rescoring with `evaluation.py`).

| preset | sweeps | pins | reproduces | page |
| --- | --- | --- | --- | --- |
| `conditioning` | `model.conditioning` × `model.cond_dropout` | Adam, $\mathcal L_{\text{CFM}}$, padding `hold`, E = 8, raw weights first and two averages | adaln > concat > cross on the raw weights; adaln = concat with averaged weights; classifier-free guidance gains nothing | {doc}`model` |
| `history` | `data.obs_horizon` × `data.action_history`; every E late | Adam, $\mathcal L_{\text{CFM}}$, padding `hold`, E = 8, raw weights first and two averages | K = 2 = K = 4 > K = 8 > K = 16; states only > with action history; E by K | {doc}`data`, {doc}`evaluation` |
| `horizons` | `data.pred_horizon`; every E <= H late | Adam, $\mathcal L_{\text{CFM}}$, raw weights first and two averages | H = 16 = H = 32 > H = 8 = H = 64; E = 4 > E = 8 | {doc}`evaluation` |
| `training_loop` | `train.lr`; a second launch adds warm-up and cosine decay | AdamW, $\mathcal L_{\text{CFM}}$, two averages | 1e-4 = 3e-4 = 1e-3; constant > cosine at 1e-4; averaged > raw weights | {doc}`training` |
| `optimizer` | `train.weight_decay`; a second launch runs Muon over `train.muon_lr` | AdamW, $\mathcal L_{\text{CFM}}$, two averages | weight decay 0 = 0.001 = 0.01 = 0.05; Muon > AdamW (weak) on the preset's seeds, Muon > AdamW on fresh seeds scored held-out (`name=confirm 'seed=range(600,606)'`, then `evaluation.py`) | {doc}`training` |
| `flow_matching` | `flow/path` × `flow.recipe`; further launches for the denoiser output, flow times and reduction | AdamW | linear = GVP; the loss's weighting matters more than the output; logit-normal(+0.5) = uniform = logit-normal(0) > logit-normal(-0.5); sum = example_mean; sampler findings by rescoring with `evaluation.py` | {doc}`flow/training`, {doc}`flow/sampling` |
| `meanflow` | seeds; the flow matching control is a second launch | `flow=meanflow`, `sampler=mean`, AdamW at weight decay 0, warm-up | one MeanFlow step >> one flow matching step; two MeanFlow steps = flow matching at 10 NFE | {doc}`flow/meanflow` |
| `smoke` | nothing | 2000 steps, short evaluations | the one-minute end-to-end check | {doc}`contributing` |

A preset pins only what its file names; every other key follows today's defaults. `conditioning`, `history` and
`horizons` pin what their studies ran with: Adam (AdamW at weight decay 0, which steps bit for bit as Adam),
$\mathcal L_{\text{CFM}}$, and the raw weights as the learning curve and headline, with the two averages scored
alongside.
`training_loop` trains AdamW at weight decay 0.01, while its study used Adam (weight decay 0; AdamW = Adam was found
later). The `flow_matching` study sampled the noise predictor (`scaled_score_mse`) from $t_0 = 0.001$, the default
start time then; its file says how to log its runs alike.

## Defaults at a glance

One row per component: the key, its default, and what it beat or tied, in the verdicts of
{doc}`overview`. Each comparison ran on the defaults of its time (the first ones on an MLP, later ones on Adam or
$\mathcal L_{\text{CFM}}$); the linked page says where that matters.

| component | config key | default | what it beat or tied | page |
| --- | --- | --- | --- | --- |
| representation, frame | `data.frame` | `block` | relative > absolute world coordinates on mean max reward, success tied (MLP; a pipeline baseline, not an option); block > agent frame on mean max reward, success tied (MLP) | {doc}`data` |
| padding | `data.padding` | `mask` | mask > hold at H = 64 (firm with averaged weights, weak with raw); at H = 16 (MLP) hold > drop and hold = mask | {doc}`data` |
| normalization | none | per history lag, chunk step and feature | a design choice, not compared | {doc}`data` |
| history | `data.obs_horizon`, `data.action_history` | 2, `false` | K = 2 = K = 4 > K = 8 > K = 16; states only > with action history | {doc}`data` |
| horizons | `data.pred_horizon`, `eval.exec_horizon` | 16, 4 | H = 16 = H = 32 > H = 8 = H = 64 (K = 2); E = 4 > E = 8 >> E = 16 | {doc}`evaluation` |
| network, conditioning | `model`, `model.conditioning` | `dit`, `adaln` | DiT >> MLP; adaln > concat > cross with raw weights (the second gap weak); adaln = concat with averaged weights; width, depth and heads not studied | {doc}`model` |
| classifier-free guidance | `model.cond_dropout`, `sampler.cfg_scale` | 0.0, 1.0 | classifier-free guidance at $w$ from 1.1 to 1.5 = none ($w = 1$); condition dropout = none at $w = 1$ | {doc}`model` |
| objective | `flow` | `default` (flow matching) | one MeanFlow step >> one flow matching step, but flow matching at 10 NFE > one MeanFlow step (weak) and = two (against flow matching with Adam and $\mathcal L_{\text{CFM}}$) | {doc}`flow/meanflow` |
| path | `flow/path` | `linear` | linear = GVP on average; GVP breaks one-step Euler | {doc}`flow/index` |
| flow times | `flow/time` | `uniform` | logit-normal(+0.5) = uniform = logit-normal(0) > logit-normal(-0.5) | {doc}`flow/training` |
| output and loss | `flow.recipe` | `scaled_score_mse` | `scaled_score_mse` sampled from $t_0 = 0.05$ > `velocity_mse` (held-out, fresh seeds); `velocity_mse` = `data_output_velocity_loss` > `data_mse` | {doc}`flow/training` |
| reduction | `flow.reduction` | `sum` | sum = example_mean | {doc}`flow/training` |
| solver, NFE | `sampler.solver`, `sampler.nfe` | `euler`, 10 | Euler NFE 50 = 20 = 10 >= 4 > 2 > 1; at 10 NFE Euler = SDE at $\eta$ <= 0.03 > SDE at $\eta$ 0.1 to 0.3 > Heun | {doc}`flow/sampling` |
| time grid | `sampler.grid` | `uniform` | at 2 NFE data_dense > uniform on GVP, data_dense > uniform (weak) on linear; tied at 4 NFE | {doc}`flow/sampling` |
| start time $t_0$ | `sampler.start_time` | `null`: 0, or 0.05 for the noise predictor | noise predictor: 0.05 = 0.01 >> 0.001; no effect on vector field and denoiser outputs | {doc}`flow/sampling` |
| optimizer | `train.optimizer`, `train.muon_lr` | `muon`, 3e-4 | Muon > AdamW (narrow); Muon > Adam (weak); AdamW = Adam; Muon at 1e-3 no better, at 3e-3 unstable | {doc}`training` |
| weight decay | `train.weight_decay` | 0.01 | 0 = 0.001 = 0.01 = 0.05 on success, with AdamW and $\mathcal L_{\text{CFM}}$; not swept under Muon (0.01 by a tie-break fixed in advance) | {doc}`training` |
| learning rate, schedule | `train.lr`, `train.schedule` | 3e-4, `constant` | 1e-4 = 3e-4 = 1e-3; constant > cosine at 1e-4, constant > cosine (weak) at 3e-4 and 1e-3 | {doc}`training` |
| betas, clipping | `train.betas`, `train.grad_clip` | [0.9, 0.999], 0.0 | beta2 0.999 = 0.95; no clipping = clipping at 1.0 | {doc}`training` |
| batch, length | `train.batch_size`, `train.steps` | 128, 80000 | 64 = 128 = 256 = 512 at equal samples; 160k steps = 80k | {doc}`training` |
| precision | `train.precision` | `tf32` | TF32 = bf16 on success | {doc}`training` |
| weights used | `train.ema_decays`, `eval.weights` | [0.999], [ema0.999] | averaged > raw; decay 0.999 = 0.9999 | {doc}`training` |

`train.cuda_graph` and `train.compile` are about speed; {doc}`training` explains them.

## Reference

The option files, as they are. Their comments document each key.

```{literalinclude} ../../conf/config.yaml
:language: yaml
:caption: conf/config.yaml
```

```{literalinclude} ../../conf/evaluation.yaml
:language: yaml
:caption: conf/evaluation.yaml
```

```{literalinclude} ../../conf/model/dit.yaml
:language: yaml
:caption: conf/model/dit.yaml
```

```{literalinclude} ../../conf/flow/default.yaml
:language: yaml
:caption: conf/flow/default.yaml
```

```{literalinclude} ../../conf/flow/meanflow.yaml
:language: yaml
:caption: conf/flow/meanflow.yaml
```

```{literalinclude} ../../conf/sampler/default.yaml
:language: yaml
:caption: conf/sampler/default.yaml
```
