# Parset And Strategy Guide

Use this guide when adding, renaming, or changing user-facing options and
strategy behavior.

## Where User-Facing Options Live

- Defaults: `rapthor/settings/defaults.parset` and
  `rapthor/settings/defaults.json`.
- User docs: `docs/source/parset.rst`, `docs/source/strategy.rst`, operation
  docs, and examples where relevant.
- Test templates: `tests/resources/rapthor_minimal.parset*`,
  `tests/resources/rapthor_complete.parset*`, and integration templates.
- Examples: `examples/default_calibration_strategy.py`,
  `examples/flexible_calibration_strategy.py`, `examples/custom_*_strategy.py`,
  and example parsets.
- Runtime code: domain parsing/state in `rapthor/lib`, operation hand-off in
  `rapthor/operations`, execution payloads and commands under
  `rapthor/execution/<owner>/`.

## Option Change Checklist

When adding or changing a parset option, update these together:

1. Default parset and JSON defaults.
2. Parset parsing and domain state.
3. Operation input hand-off.
4. Execution payload dataclass or dictionary contract.
5. Payload validation.
6. Command builders or output handling, if the option affects external tools.
7. Docs and examples.
8. Minimal and complete test templates.
9. Focused tests for defaults, parsing, payloads, and emitted command tokens.

`PLAN.md` calls this the stabilization gate. For runtime, cluster, and strategy
options, specifically check `docs/source/running.rst`,
`tests/execution/test_config.py`, runtime bootstrap tests, CLI tests, and any
test templates under `tests/resources/` that snapshot parset dictionaries.

The roadmap also calls for a lightweight smoke lane that starts from
`rapthor input.parset`, not only internal bootstrap helpers. If you change CLI
startup, parset materialization, path handling, or runtime bootstrap, look for
or add coverage at that user-facing entry point.

## Calibration Strategy Tokens

`calibration_strategy` is the production interface for solve order:

```python
{"dd": ["fast_phase", "medium_phase", "slow_gains", "medium_phase"], "di": []}
```

Allowed strategy tokens are:

- `fast_phase`: fast scalar phase solve.
- `medium_phase`: medium scalar phase solve.
- `slow_gains`: slow diagonal amplitude/phase solve.
- `full_jones`: direction-independent full-Jones solve.

Do not reintroduce legacy solve toggles such as `do_fulljones_solve` or
`do_slowgain_solve`. Use `slow_gains` in strategy values; use "slow gain" or
"slow diagonal gain" in prose.

## Scientific Strategy Guidance

- Default HBA selfcal with `generate_initial_skymodel = True` runs the full DD
  solve order, including `slow_gains`, from each cycle. This relies on the
  target-field model Rapthor generated from the data and first-cycle flux-scale
  normalization.
- Built-in selfcal with `generate_initial_skymodel = False` starts with
  phase-only DD cycles before adding `slow_gains`. That is intentional because
  downloaded or user-provided starting models may not support immediate
  amplitude calibration.
- Custom strategies for weak fields can also start with phase-only DI or DD
  cycles before adding `slow_gains`.
- DI-only cycles are useful as a bootstrap, diagnostic, or narrow-field choice;
  they do not replace DD correction when residuals vary across the field.
- Keep the post-slow `medium_phase` solve explicit in DD strategies. Do not rely
  on hidden slot behavior.
- Use `target_flux`, `max_directions`, and `max_distance` to balance DD
  direction count against signal-to-noise.
- Use `facet_layout` when direction geometry must stay fixed across cycles,
  restarts, or supplied h5parm products.

## Model And Imaging Options

- Prefer a trusted `input_skymodel` and matching `apparent_skymodel` when
  available; source names must match exactly.
- If no trusted model is available, `generate_initial_skymodel = True` is the
  normal starting point.
- Keep `filter_skymodel = True` for ordinary selfcal so noisy clean components
  outside detected islands do not feed later calibration.
- Treat `average_visibilities` and `max_peak_smearing` as a science/runtime
  tradeoff. If smearing correction is enabled, align calibration and imaging
  settings.
- Prefer `final_data_fraction = 1.0` for final science images when resources
  allow.
- Use `dde_mode = faceting` unless a guarded screen or hybrid path is explicitly
  being tested and validated.
