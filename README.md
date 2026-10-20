# Flow Matching Policy (JAX)

Flow matching policies for the state-based [Push-T](https://diffusion-policy.cs.columbia.edu/) task, built for
experiments, in JAX. This is the JAX version of [PushT](https://github.com/denglinc/PushT) (PyTorch, branch `main`),
ported from its commit 9f1b3a0: the same configs, data, env seeds and evaluation noise, and the same held-out results
within seed noise. The [documentation](https://denglinc.github.io/PushT/) explains the method, which both versions
share (its API pages are PyTorch's); [PyTorch and JAX](https://denglinc.github.io/PushT/guide/frameworks.html) says
which to use and what differs.

## Install

```bash
uv sync
```

Needs Python 3.12 and [uv](https://docs.astral.sh/uv/). JAX 0.11 with CUDA 13 (NVIDIA driver 580 or newer) runs on
GPUs up to the RTX 50 series.

## Quick start

```bash
uv run train.py                         # train the default policy
uv run train.py seed=600 num_seeds=6    # six seeds at once, one run each
```

The first run downloads the training set (31 MB) to `data/`. The default trains for 80,000 steps, about 8 minutes on
an RTX 5080 plus a few minutes of evaluation: every 10,000 steps it plays 200 episodes per seed and prints the success
rate. The run lands in `runs/default/<time>_seed_0/seed_0/`, with `log.csv` and `checkpoint.pkl`. A seed reruns bit
for bit (deterministic GPU kernels; `XLA_FLAGS=--xla_gpu_deterministic_ops=false` trains 11% to 18% faster).

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
uv run train.py -m experiment=horizons    # a preset grid from conf/experiment/, one job after another
```

## Results

Held-out success agrees with PushT's within the noise between training seeds; see
[PyTorch and JAX](https://denglinc.github.io/PushT/guide/frameworks.html) in the documentation.

## Logging

Every run writes its scalars to `log.csv` and to a W&B run (project `pusht_jax`), online by default (log in once
with `uv run wandb login`):

```bash
uv run train.py wandb_mode=offline                                     # or disabled: log.csv only
uv run wandb sync $(find runs/<name> -type d -name 'offline-run-*')    # upload offline runs later
```

`uv run logger.py` lays out your W&B workspace without a run, from the keys of the project's runs.

## Files

| file | contents |
| --- | --- |
| `train.py` | the training loop: 100 steps of every seed per compiled call |
| `evaluation.py` | closed-loop evaluation, checkpoints, scoring of finished runs |
| `flow.py` | flow matching and MeanFlow: paths, flow times, predictions, losses, samplers |
| `model.py` | the policy and its networks (DiT, MLP), as Flax NNX modules |
| `data.py` | dataset, relative representation, normalizer |
| `logger.py` | `log.csv` and W&B logging, the layout of your W&B workspace |
| `conf/` | every option, as Hydra configs |

`uv run pytest` runs the tests (on the CPU).

## Acknowledgements

- Task and data: [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/), [gym-pusht](https://pypi.org/project/gym-pusht/).
- Relative representation: [UMI](https://umi-gripper.github.io/). Network: [DiT](https://arxiv.org/abs/2212.09748).
- MeanFlow: [MeanFlow](https://github.com/Gsunshine/meanflow), [improved MeanFlow](https://github.com/Lyy-iiis/imeanflow/).
- Flow matching: debugging insights from [SiT](https://arxiv.org/abs/2401.08740); code written with reference to
  [facebookresearch/flow_matching](https://github.com/facebookresearch/flow_matching).
- The JAX training state: [openpi](https://github.com/Physical-Intelligence/openpi).
