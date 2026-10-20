# PyTorch and JAX

PushT has two implementations of the same method: PyTorch on `main`, which these pages document, and JAX on the
[`jax` branch](https://github.com/denglinc/PushT/tree/jax). They share the configs, the data, the evaluation and every
finding in this guide, and their held-out success agrees within the noise between training seeds.

## Which to use

Use JAX for research on the method. MeanFlow's target is a Jacobian-vector product ({doc}`flow/meanflow`), which JAX
writes as `jax.jvp` and composes with `jit` and `vmap` like any other function.

Use PyTorch for long-term engineering. It fits the training, simulation and deployment tools of robotics, which are
built on it, and eager code is easier to debug.

## Running JAX

```bash
git clone -b jax https://github.com/denglinc/PushT.git PushT_jax && cd PushT_jax
uv sync
uv run train.py
```

The configs and commands are the same, except that `train.cuda_graph`, `train.compile` and `train.precision` do not
exist and `num_seeds` trains several seeds in one process. Runs save `checkpoint.pkl` in one `seed_<s>/` directory per
seed and log to the W&B project `pusht_jax`.

## What differs

In the JAX version, these differ from PyTorch:

- [`flow.py`](https://github.com/denglinc/PushT/blob/jax/flow.py): randomness is an argument.
  `TimeSampler.sample(key, n)` and `FlowMatching.loss(..., key=...)` take a JAX key, and
  `MeanFlow.average_velocity` calls `jax.jvp`. `Sampler.sample` runs under `jit`, so the caller checks that its
  result is finite.
- [`model.py`](https://github.com/denglinc/PushT/blob/jax/model.py): the policies are Flax NNX modules that hold only
  parameters. `parameter_roles()` labels them for the optimizers, and `condition(..., dropout_key=...)` replaces the
  training mode of `nn.Module`.
- [`train.py`](https://github.com/denglinc/PushT/blob/jax/train.py): no training step is captured.
  `make_train_window` compiles `log.train_interval` steps as one `lax.scan` over a `TrainState` of plain dicts
  (weights, optimizer state, averages), vmapped over the seeds. Optimizers are optax; `scale_by_torch_muon` carries
  over PyTorch's Muon.
- [`evaluation.py`](https://github.com/denglinc/PushT/blob/jax/evaluation.py): `make_query` compiles one query per
  run, vmapped over seeds. Every query covers all episodes, so its shape never changes. `configure_jax` sets TF32,
  deterministic GPU kernels and the compilation cache.
- [`data.py`](https://github.com/denglinc/PushT/blob/jax/data.py): the windows are one pytree on the device
  (`Windows`), which the compiled window takes as an argument.
- [`logger.py`](https://github.com/denglinc/PushT/blob/jax/logger.py): each logger owns its W&B run, since one process
  logs a run per seed.

The branch is ported from `main`, and its README names the commit. A change on `main` reaches it with the next port.
