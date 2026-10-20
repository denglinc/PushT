# PushT

Flow matching policies for the state-based [Push-T](https://diffusion-policy.cs.columbia.edu/) task, built for
experiments. A diffusion transformer reads the last two states, predicts the next 16 target positions of the agent,
executes 4 of them and plans again. Every part of it (the representation, the network, the probability path, the
flow-time distribution, the network's output, the sampler and the training loop) is a Hydra option, and almost every
default comes out of a measured comparison. This documentation explains how the parts fit together, the math behind
them, what was compared and which choice won.

```bash
uv sync                                  # Python 3.11+ and uv
uv run train.py experiment=smoke         # a one-minute end-to-end check
uv run train.py                          # train the default policy
uv run evaluation.py runs=runs/default   # score it on held-out episodes, under any sampler, without retraining
```

::::{grid} 1 2 2 3
:gutter: 2

:::{grid-item-card} Data
:link: guide/data
:link-type: doc
Relative representation, chunk windows, padding, normalization.
:::

:::{grid-item-card} Model
:link: guide/model
:link-type: doc
The policy as a guided vector field, the DiT and its conditioning, the MLP.
:::

:::{grid-item-card} Flow matching
:link: guide/flow/index
:link-type: doc
Probability paths, vector fields and their conversions, the training loss, ODE and SDE sampling, MeanFlow.
:::

:::{grid-item-card} Training
:link: guide/training
:link-type: doc
The captured training step, Muon and AdamW, weight averaging.
:::

:::{grid-item-card} Evaluation
:link: guide/evaluation
:link-type: doc
Lock-step rollouts, execution horizon, held-out scoring, latency.
:::

:::{grid-item-card} Extending
:link: guide/extending
:link-type: doc
Add a probability path, a network output, a solver, a network or an experiment.
:::
::::

Start with the {doc}`guide/overview`. The math follows the notation of Holderrieth and Erives' flow matching notes,
which {doc}`guide/notation` maps to the code. For the default configuration alone in four files, see nano-PushT. A JAX
implementation lives on the `jax` branch; {doc}`guide/frameworks` compares the two and says which to use.

```{toctree}
:hidden:
:caption: Guide

guide/overview
guide/notation
guide/data
guide/model
guide/flow/index
guide/training
guide/evaluation
guide/configuration
guide/logging
guide/frameworks
```

```{toctree}
:hidden:
:caption: Development

guide/extending
guide/contributing
```

```{toctree}
:hidden:
:caption: API reference

api/data
api/model
api/flow
api/train
api/evaluation
api/logger
```

```{toctree}
:hidden:
:caption: Project links

GitHub <https://github.com/denglinc/PushT>
```
