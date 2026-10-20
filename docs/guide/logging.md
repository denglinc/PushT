# Logging

Every run writes its rows of scalars to `log.csv` in its run directory and, unless W&B is disabled, to a W&B run.
`evaluation.py` logs the scores of each run it evaluates the same way, in a run of its own.

```bash
uv run wandb login                                                    # once, before the first upload
uv run train.py wandb_mode=offline                                    # keep the W&B run in the run directory
uv run wandb sync $(find runs/<name> -type d -name 'offline-run-*')   # upload offline runs later, evaluations too
uv run train.py wandb_mode=disabled                                   # log.csv only
uv run logger.py --project pusht                                      # lay out your W&B workspace from the project's runs
```

## The logger

{py:func}`~train.main` creates one {py:class}`~logger.Logger` per run, in the directory Hydra made for it, with the
W&B project `wandb_project`, the group `name` and the run name `<arm>/seed_<seed>` (see {doc}`configuration`).
{py:func}`~train.run_training` then calls it in this order:

- {py:meth}`~logger.Logger.init_logging_writer` once, with the resolved config and a few facts about the run: it starts
  the W&B run and writes `config.json` and the code state;
- {py:meth}`~logger.Logger.log_training` every `log.train_interval` optimizer steps, with means over the steps since
  the last call (the sums stay on the device until then);
- {py:meth}`~logger.Logger.log_evaluation` after every evaluation, once per evaluated setting;
- {py:meth}`~logger.Logger.log_late` after the last step, and {py:meth}`~logger.Logger.save_model`;
- {py:meth}`~logger.Logger.stop_logging_writer` from {py:func}`~train.main`, which finishes the W&B run, marked failed
  when training raised.

`evaluation.py` gives each scored run a logger of job type `evaluate` in `<run>/evaluations/<time>[/<job>]/`, and
{py:func}`~evaluation.score_run` calls {py:meth}`~logger.Logger.log_scores` once.

`log_training`, `log_evaluation`, `log_late` and `log_scores` each end in {py:meth}`~logger.Logger.log`, which takes
one row at one optimizer step. The row's scalars and the step go to `log.csv`; when a row brings keys the file does
not have yet, the header widens and the file is rewritten. The same row goes to W&B in one `wandb.log` call at that
step, without a `step` key: W&B's own step is the x-axis, and `log.csv` keeps the step column. Rows of one step may
arrive in several calls (evaluation, then training, then at the last step `late/`), and a row leaves the keys it does
not bring empty. The logger claims `log.csv` when it starts, so a second logger in the same directory fails instead of
mixing rows.

Every `log.print_interval` steps the logger also prints the latest values with the elapsed time and an estimate of the
time left; after the last step it prints each late setting's pooled success and mean max reward.

### Modes

`wandb_mode` picks where rows go besides `log.csv`: `online` uploads while the run trains, `offline` keeps the W&B run
in `wandb/` until `wandb sync` uploads it, and `disabled` writes `log.csv` alone. {py:func}`~train.check_config`
refuses any other value. The default reads the environment variable `WANDB_MODE`:

```{literalinclude} ../../conf/config.yaml
:language: yaml
:start-at: wandb_project
:end-at: wandb_mode
```

`conf/evaluation.yaml` has the same two keys. The entity comes from `WANDB_ENTITY`, or the login's default.

## Keys

Keys are grouped by prefix; W&B draws a section per prefix. The module docstring of {py:mod}`logger` is the
authoritative list.

| prefix | keys | logged |
| --- | --- | --- |
| `train/` | `loss`, `lr`, `grad_norm`, `throughput`, `epoch` | every `log.train_interval` steps |
| `eval/` | `success_rate`, `mean_reward`; `<metric>_E<e>_<weights>` | at every evaluation |
| `late/` | `success_rate`, `mean_reward`; `<metric>_E<e>_<weights>` | once, at the last step |
| `meanflow/` | `loss_ref`, `loss_u`, `loss_v` | with the `train/` rows, MeanFlow runs only |
| `evaluate/` | `success_rate`, `mean_reward`; `<metric>_E<e>`; `nfe`; `latency_ms` | once per scored run, by `evaluation.py` |

- **`train/`.** `loss` is the objective's value and `grad_norm` the global gradient norm before clipping, both
  averaged over the steps since the last row. `lr` is AdamW's learning rate; with Muon it is the rate of the boundary
  matrices and the vectors, while Muon's own rate stays at `train.muon_lr`. `throughput` counts optimizer steps per
  second of training alone: the steps up to the CUDA graph's capture (the first one compiles the loss) and the time of
  evaluations are left out. It measures the code only on an otherwise idle GPU. `epoch` counts completed passes over a
  fresh permutation of the windows.
- **`eval/`.** `<metric>` is `success_rate` or `mean_reward` ({py:meth}`~evaluation.Rollouts.summary`), `<e>` the
  execution horizon E and `<weights>` `raw` or `ema<decay>`. The default setting is the first of `eval.weights` at
  `eval.exec_horizon`; `eval/success_rate` and `eval/mean_reward` are its learning curve, and the W&B summary keeps
  their maximum and last value. Every evaluation scores the default setting; the last `eval.num_late` evaluations
  also score every weight copy of `eval.weights` at `eval.exec_horizon` and at each of `eval.late_exec_horizons` up
  to H.
- **`late/`.** Each setting's late evaluations, pooled. `late/success_rate` and `late/mean_reward` are the default
  setting's: the run's headline.
- **`meanflow/`.** Plain squared errors of the network's average velocity $\bar u^\theta_{t,s}(x \mid y)$ (symbols as in
  {doc}`notation`) in the objective's reduction, without the adaptive weight: `loss_ref` over every row of the batch,
  `loss_u` over the rows with $s > t$, `loss_v` over those with $s = t$, where the network's
  $\bar u^\theta_{t,t}(x \mid y)$, its estimate of the marginal vector field, is compared with the conditional vector
  field $u_t^{\text{target}}(x \mid z)$ as in $\mathcal L_{\text{CFM}}$. A batch without rows of one kind gives NaN. A
  MeanFlow run's `train/loss` is the adaptively weighted objective, which stays near 1 at `flow.adaptive_power` 1
  whatever the fit, so these keys show the fit. See {py:class}`~flow.MeanFlow`.
- **`evaluate/`.** `evaluation.py` logs `<metric>_E<e>` for every execution horizon it scored, the first one's also as
  `success_rate` and `mean_reward`, the sampler's network evaluations as `nfe`, and with `latency=true` the median
  batch-1 query time as `latency_ms`. The row sits at the checkpoint's step.

## The W&B workspace

W&B draws a panel per key, in a section per key prefix. When an online run finishes without error, its logger lays
out its user's workspace of the project with the keys it logged ({py:func}`~logger.lay_out_workspace`): it orders,
names and folds the sections by prefix and titles the axes of the curves ({py:func}`~logger.y_title`);
{py:func}`~logger.lay_out_spec` holds the current layout. `uv run logger.py` ({py:func}`~logger.main`) does the same
from the keys of all the project's runs, for instance after a `wandb sync`. W&B creates a user's workspace on their
first visit to the project; until then there is nothing to lay out, and the logger says so. A layout that fails is
reported and the run still ends.

## What a run directory keeps

Besides `log.csv`, the logger leaves:

- `config.json`: the config the logger was started with. For training, the resolved config, the arm and `info`, a few
  facts about the run ({py:func}`~train.run_training`); for `evaluation.py`, its own config, the run's config, the
  checkpoint and the sampler's grid.
- `git/<repository>.diff`: the commit, `git status` and the uncommitted diff against `HEAD` of the code that ran.
- `wandb/` with W&B on. An offline run keeps copies, not symlinks, of `config.json`, the code state and `.hydra/`, so
  it stays complete when its directory moves before `wandb sync`. The same files are saved to the W&B run.
- The checkpoint as a W&B model artifact, `policy-<run id>` with its step (not when disabled).

## Comparing runs

`late/success_rate` is the number to compare: group the runs of a preset by `arm` in W&B, or read each run's `log.csv`.
Training losses compare only between runs that agree on everything that sets the loss's floor, which [Comparing training
losses](flow/training.md#comparing-training-losses) lists. MeanFlow's losses compare within one pairs setting. There is
no validation loss: an earlier validation set was removed because its loss did not track success.

**What we found**

- **weight decay 0 = 0.001 = 0.01 = 0.05 on success.** With AdamW and the conditional flow matching loss
  $\mathcal L_{\text{CFM}}$ (`velocity_mse`), the late training loss rises with the decay all the same: across these
  arms a lower training loss does not mean a higher success.

Reproduce: `uv run train.py -m experiment=optimizer`
