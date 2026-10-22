# Flow Matching Policy

Flow matching policies for the state-based [Push-T](https://diffusion-policy.cs.columbia.edu/) task, built for
experiments.

## Install

```bash
uv sync
```

Needs Python 3.11+ and [uv](https://docs.astral.sh/uv/). PyTorch 2.9 with CUDA 12.8 runs on GPUs up to the RTX 50
series. A JAX version lives on the [`jax` branch](https://github.com/denglinc/PushT/tree/jax); the documentation's
*PyTorch and JAX* page compares the two.

## Quick start

```bash
uv run train.py                     # train the default policy
```

The first run downloads the training set (31 MB) to `data/`. The default trains for 80,000 steps, ~10 minutes on
an RTX 5080 plus a few minutes of evaluation: every 10,000
steps it plays 200 episodes and prints the success rate. The run lands in `runs/default/<time>_seed_0/`, with
`log.csv` and `checkpoint.pt`. Its first step compiles the loss; `train.compile=false` trains eagerly (7.8 ms per
step). Runs from commit 07c0a0a or earlier drew their flow times differently and repeat bit for bit only with that code.

Then score finished runs on 1,000 held-out episodes, under any sampler, without retraining:

```bash
uv run evaluation.py runs=runs/default                  # every finished run below runs/default
uv run evaluation.py runs=runs/default sampler.nfe=4    # the same policies with 4 Euler steps instead of 10
```

## One-step generation (MeanFlow)

```bash
uv run train.py experiment=meanflow                     # MeanFlow with its tuned recipe
uv run evaluation.py runs=runs/meanflow sampler=mean    # one network evaluation per chunk
```

## Configuration

Every key of `conf/config.yaml` is an option, with a comment on what it does; override any on the command line:

```bash
uv run train.py flow/path=gvp model.conditioning=cross seed=1
uv run train.py -m experiment=horizons    # a preset grid from conf/experiment/, one run after another
```

## Results

See `docs/`.

## Logging

Every run writes its scalars to `log.csv` and to a W&B run, online by default (log in once with `uv run wandb login`):

```bash
uv run train.py wandb_mode=offline                                     # or disabled: log.csv only
uv run wandb sync $(find runs/<name> -type d -name 'offline-run-*')    # upload offline runs later
```

`uv run logger.py` lays log out without a run, from the keys of the project's runs.

## Files

| file | contents |
| --- | --- |
| `train.py` | the training loop with a compiled loss, captured as one CUDA graph |
| `evaluation.py` | closed-loop evaluation, checkpoints, scoring of finished runs |
| `flow.py` | flow matching and MeanFlow: paths, flow times, predictions, losses, samplers |
| `model.py` | the policy and its networks (DiT, MLP) |
| `data.py` | dataset, relative representation, normalizer |
| `logger.py` | `log.csv` and W&B logging, the layout of your W&B workspace |
| `conf/` | every option, as Hydra configs |

`uv run pytest` runs the tests. See `docs/` for details.

## Acknowledgements

- Task and data: [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/), [gym-pusht](https://pypi.org/project/gym-pusht/).
- Relative representation: [UMI](https://umi-gripper.github.io/). Network: [DiT](https://arxiv.org/abs/2212.09748).
- MeanFlow: [MeanFlow](https://github.com/Gsunshine/meanflow), [improved MeanFlow](https://github.com/Lyy-iiis/imeanflow/).
- Flow matching: debugging insights from [SiT](https://arxiv.org/abs/2401.08740); code written with reference to
  [facebookresearch/flow_matching](https://github.com/facebookresearch/flow_matching).

