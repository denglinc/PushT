# Contributing

The code, its tests and these docs move together: a change lands with its test, its docstring and, when it changes
behaviour or a default, the guide page that describes it.

```bash
uv sync                               # the project and the dev group (pytest, ruff)
uv run pytest                         # every test; the CUDA tests skip without a GPU
uv run pytest -m "not cuda"           # leave the CUDA tests out explicitly
uv run ruff check .                   # lint the modules, the tests and docs/
uv run ruff format .                  # format them (Python blocks in docs/*.md too)
```

`uv sync` installs the `dev` group by default. Tests marked `cuda` need a GPU (graph capture, the compiled loss, the
training step) and skip themselves without one. The repository passes `ruff check` and is formatted
with `ruff format`, which also formats the Python blocks of the Markdown pages; keep it that way.

## What the tests guarantee

- **`tests/test_flow.py`** runs `flow.py` on a Gaussian toy where every quantity is analytic: $p_{\mathrm{data}}$ is
  Gaussian with its own mean and standard deviation per coordinate, so the denoiser $D_t(x) = \mathbb E[z \mid x]$,
  the marginal vector field $u_t^{\text{target}}(x)$, the score $\nabla\log p_t(x)$ and the flow $\psi_t$ of the
  marginal vector field, with the exact endpoint of every sample, have closed forms (symbols as in {doc}`notation`).
  On it the tests check that
  - each path has exact endpoints in float32, a $\kappa_t = \dot\alpha_t\beta_t - \alpha_t\dot\beta_t$ (`D` in
    the code) that matches its definition and derivatives that match finite differences (`test_path_coefficients`),
    and the conditional vector field $u_t^{\text{target}}(x\mid z) = \dot\alpha_t z + \dot\beta_t\epsilon$ is the
    time derivative of $x = \alpha_t z + \beta_t\epsilon$;
  - the conversions between predictions reproduce the analytic posterior quantities and commute with classifier-free
    guidance (`test_adapters_on_the_gaussian_posterior`);
  - flow times lie on the quantile lattice, truncated draws follow the truncated CDF
    (`test_truncated_draws_follow_the_conditional_cdf`), and distributions that would reach an endpoint in float32 are
    refused;
  - an oracle has zero loss under every recipe and path, both reductions match a hand computation, and a
    velocity-space loss refuses a support with the singular time;
  - every solver, grid, path and prediction reaches the exact endpoint (`test_ode_reaches_the_exact_endpoint`), the
    Euler method converges at order 1 and Heun's method at order 2 (`test_convergence_orders`), and unsupported
    sampler settings raise;
  - the SDE sampler accepts every path and prediction, keeps the marginals $p_t$ (`test_sde_keeps_the_marginals`), has
    the drift $u_t^\theta + \tfrac{\sigma_t^2}{2}s_t^\theta$ of the SDE extension trick, with the score network
    $s_t^\theta$ converted from $u_t^\theta$, and keeps its closed form on the CondOT path bit for bit;
  - MeanFlow's identity holds for the exact average velocity $\bar u_{t,s}$ (the wrong sign fails), the mean sampler
    reaches the exact endpoint in any number of steps, its loss and target match hand computations, its time pairs
    match the papers' samplers, and at $s = t$ it is flow matching;
  - on a GPU, the loss of every recipe and time sampler, and MeanFlow's, is captured in a CUDA graph without host
    syncs.
- **`tests/test_data.py`**: relative actions map back to their world targets in both frames, the current frame's
  observation has the block at the origin and recovers the goal, windows and history padding are right for each
  padding mode, set-points pair with the previous action, and the normalizer maps constant features to 0 and keeps
  masked targets out of its statistics.
- **`tests/test_train.py`**:
  - every job of every preset's sweep passes {py:func}`~train.check_config`
    (`test_presets_pass_config_checks`), and unsupported combinations fail before training;
  - the captured step trains under both optimizers and both precisions, with the noise predictor's and the denoiser's
    recipes (`scaled_score_mse`, `data_mse`), the DiT's adaln and cross conditionings and the MLP, and several MeanFlow
    variants, eager and with the compiled loss: the loss falls, every weight matrix moves and the average $\bar\theta$
    moves (`test_captured_step_trains`);
  - the compiled loss draws what the eager one draws (`test_compiled_loss_draws_like_eager`);
  - the parameter groups partition the parameters by role, weight decay acts on the matrices only, AdamW at weight
    decay 0 steps bit for bit as Adam, and under the graph AdamW's groups share one learning-rate tensor;
  - MeanFlow networks: the forward-mode derivative of the real network matches finite differences, the network sees
    the gap, starts at the instantaneous velocity and adds only the gap parameters, and its checkpoint rebuilds;
  - runs are named by their arm and land in Hydra's directories, only finished checkpoints are scored, and a lock-step
    evaluation scores every episode.
- **`tests/test_logger.py`**: every row reaches `log.csv` and W&B at its step under the keys of the three modules, and
  a finished online run lays out its user's W&B workspace.

## The docstring convention

The API pages render the docstrings through `docs/_ext/docstrings.py`, which reads plain text in this form:

```python
def interpolate(
    z: torch.Tensor, eps: torch.Tensor, t: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    Samples the linear path and its conditional velocity at flow times $t$:
        $$
        x = t z + (1 - t)\,\epsilon, \qquad u_t^{\text{target}}(x \mid z) = z - \epsilon
        $$
    Note: one flow time applies to every action of a chunk.
    Args:
        - z, eps: clean chunks and Gaussian noise, shape (bs, H, d)
        - t: flow times, shape (bs, 1, 1), which broadcast against the chunk
          (the network itself sees them as (bs, 1))
    Returns:
        - x: noisy chunks $x$, shape (bs, H, d)
        - u: the conditional velocity $u_t^{\text{target}}(x \mid z)$, shape (bs, H, d)
    """
```

- The first line is the summary.
- A `Note:` line, with the lines after it up to the next section or blank line, becomes a note box.
- `Args:` items `- name: description` become the parameter list; the types come from the annotations, so the
  docstring never repeats them. A continuation line is indented deeper than its dash. `- a, b: ...` documents two
  arguments with one description.
- `Returns:` items are `- name: description`, or `- description` for an unnamed value.
- Formulas are LaTeX in the notation of {doc}`notation`, rendered by MathJax: inline as `$...$` on one line, and a
  display formula as a block between two lines that hold only `$$`. A docstring with a backslash is a raw string,
  `r"""`, so that `\beta` or `\theta` is not read as an escape.
- Other text indented deeper than the paragraph before it (an example, a table) becomes a literal block.
- An override that changes nothing in the contract says `Args and Returns: see ODE.drift.` instead of repeating them.
- Shapes go in the descriptions, as "shape (bs, H, d)", with the symbols the module docstring defines.
- Outside formulas, write plain text: the extension escapes reST markup (asterisks, backticks, bars, stray
  underscores), so `x_t` and `*args` render as written.

## The documentation

The guide (`docs/guide`) is written by hand in MyST Markdown; the API pages (`docs/api`) are generated by autodoc from
the modules.

```bash
uv run --only-group docs sphinx-build -W --keep-going -b html docs docs/_build/html      # the CI build
uv run --only-group docs sphinx-autobuild docs docs/_build/html --watch data.py --watch flow.py \
    --watch model.py --watch train.py --watch evaluation.py --watch logger.py --watch conf  # live preview
```

`.github/workflows/docs.yml` runs the build on every push and pull request and publishes `main` to GitHub Pages.
The build is strict: `nitpicky = True` and `-W` turn every unresolved cross-reference into an error, so renaming a
function, method or class without updating the guide's references fails the build, which is how the guide stays in
sync with the code. `nitpick_ignore_regex` in `docs/conf.py` exempts only types from packages without an inventory,
a few type aliases and `Rollouts`, which `logger.py` imports only for type checking. `autodoc_mock_imports` mocks
torch, the simulator (`gym_pusht`, `gymnasium`), W&B and `zarr`, so the `docs` group needs no GPU stack; hydra,
omegaconf and numpy are imported for real.

Pages cross-reference the code with roles such as ``{py:func}`~train.check_config` `` and pages with
``{doc}`training` ``. A module-level constant appears in the API pages, and can be referenced, only with a `#:` doc
comment; with a plain comment, name it as inline code.

## Upkeep when changing code

- **New option.** A comment on its line of the YAML file: {doc}`configuration` includes the config files.
- **New public function or class.** A docstring in the convention above; the API pages pick it up.
- **Renamed or removed symbol.** Fix the guide's references; the build lists every one it can no longer resolve.
- **Changed default.** The defaults table on {doc}`configuration`, the page that describes the setting, and the
  presets whose results were measured with the old default (pin it there).
- **New comparison.** Its result as a number-free bullet under the page's **What we found**, with the preset or the
  override that reproduces it.

## Adding a finding

A finding follows the study protocol of the experiment report:

1. **Fix the decision rule before launching** (the metric, the tie-break, what counts as a win) and note any change
   made after seeing data.
2. **Choose on the selection episodes** (env seeds 0 and up, what training evaluates), with several training seeds
   per arm. Compare late success, pooled over the last `eval.num_late` evaluations; mean max reward breaks ties.
3. **Compare samplers paired**, on the same policies and the same noise: rescore finished runs with `evaluation.py`
   (`split=selection`), where each episode draws its noise from its own generators.
4. **Confirm on the held-out episodes** (env seeds 10000 and up, the default of `evaluation.py`) with fresh training
   seeds that no decision saw.
5. **Adopt a change only if its interval excludes zero in its favour.** A pure speed gain may be adopted when its
   interval stays above a non-inferiority margin fixed in advance.
6. **Record it** on its page as `A > B`, `A = B` or `A > B (weak)`, without numbers, with the preset that reproduces
   it; the numbers stay in the report. `A = B` means no difference was detected, not that the two are equal.

```bash
uv run train.py -m experiment=my_study                                   # the grid, on the selection episodes
uv run evaluation.py -m runs=runs/my_study split=selection 'sampler.nfe=4,10'   # samplers, paired
uv run train.py -m experiment=my_study name=my_study_confirm 'seed=range(<a>,<b>)'   # fresh seeds
uv run evaluation.py runs=runs/my_study_confirm                          # held-out episodes
```

Training losses compare only between runs that agree on everything that sets the loss's floor
([Comparing training losses](flow/training.md#comparing-training-losses)). The demonstrations never reach the
success threshold, so success measures how often a policy goes beyond its data (see {doc}`evaluation`).
