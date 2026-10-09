# Scientific contracts

Read the relevant section when changing solve order, calibration application,
beam handling, sky models, averaging or scientific products. This reference
records distinctions that are easy to lose in code; detailed user guidance is
linked below. Current behavior is defined by the implementation and its tests.

## Vocabulary

| Term | Meaning in Rapthor |
| --- | --- |
| Visibility | Complex antenna-pair correlation; the data from which images are formed. |
| Observation | Measurement Set metadata and time/frequency sampling. |
| Sector | An image or prediction work unit. |
| Facet | A geometric sky region used for direction-dependent processing. |
| Patch | A sky-model source group used as a calibration direction. |
| DI | Direction-independent: a correction common to sky directions. |
| DD | Direction-dependent: corrections vary across the field. |
| Solve | Derive calibration terms by comparing observed and model visibilities. |
| Applycal | Apply existing calibration terms to visibilities. |
| Predict | Produce model visibilities from a sky model. |
| Apparent sky | Sky brightness attenuated by the primary beam. |
| True sky | Sky brightness without primary-beam attenuation. |

Sectors, facets and patches often correspond but are not interchangeable names.
Keep `apparent_sky` and `true_sky` qualifiers in paths, payloads and variables;
beam-corrected fluxes and flat-noise image products have different meanings.

Self-calibration iterates prediction, solving, imaging and model improvement.
An incomplete model can bias the solutions, and DI bootstrap cannot correct
spatially varying residuals. Source filtering protects later solves from noise
in deconvolved model components.

## Ordered calibration strategies

The top-level processing `strategy` chooses operations and cycle settings.
`calibration_strategy` specifies ordered solve lists under `di` and `dd`.

| Strategy token | Solve mode | Contract |
| --- | --- | --- |
| `fast_phase` | `scalarphase` | Fast scalar phase corrections. |
| `medium_phase` | `scalarphase` | Medium phase corrections; can occur before and after slow gains. |
| `slow_gains` | `diagonal` | Slow diagonal amplitude and phase corrections; DD slow gains remain diagonal. |
| `full_jones` | `fulljones` | Full 2×2 polarized gain matrix; currently supported only for DI. |

Use canonical strategy tokens. Existing filename prefixes such as `slow_gain`
and `fulljones` do not define alternative tokens and should not be renamed
incidentally.

The default DD solve order is explicit:

```python
{"dd": ["fast_phase", "medium_phase", "slow_gains", "medium_phase"], "di": []}
```

- Preserve the requested order and validation of supported combinations. Do not
  infer a hidden post-slow phase solve from a slot number.
- The built-in HBA selfcal strategy uses the full DD sequence each cycle when
  Rapthor generated the initial sky model. Supplied/downloaded initial models
  retain early phase-only DD cycles before adding amplitude solves.
- Legacy solve flags are translated with warnings for compatibility. New code
  should use `calibration_strategy`; retirement work belongs in [PLAN](../PLAN.md).

See [strategy](../docs/source/strategy.rst),
[initial-model preparation](../docs/source/preparation.rst), and
[the custom strategy example](../examples/flexible_calibration_strategy.py).

## Solutions: corrections versus optimizer seeds

Track each solution's role, calibration mode, solve type and cycle. DI, DD,
full-Jones, normalization and screen products are distinct scientific objects.

**Applied corrections** change visibilities. Preserve current-cycle guards for
`applycal_h5parm`, `fulljones_h5parm`, image `prepare_data_h5parm` and imaging-time
`h5parm`, unless the operation explicitly allows carry-forward. Supplied products
need the appropriate sky directions and time/frequency coverage. Never silently
substitute a known stale product during normal execution or restart.

**Initial solutions** seed the optimizer without pre-applying a correction.
Current-cycle or compatible earlier same-mode, same-solve products may initialize
a solve. DI seeds DI, DD seeds DD; reject known future-cycle products.

DD seeds deliberately need not match the current patch names or direction count:
DP3 selects the nearest h5parm direction (`GetNearestSource`). This requires
correct sky positions in the h5parm `source` table, maintained by
`adjust_h5parm_sources`. Do not extend this exception to applied corrections.

The implementation lives in
[calibration adapters](../rapthor/operations/calibrate/base.py); regression cases
are in [calibration tests](../tests/operations/test_calibrate.py).

## Product state and compatibility

- Normalization products change the scientific flux scale. They are distinct
  from display scaling and ordinary solve outputs, and have intentional
  carry-forward behavior separate from correction-cycle guards.
- Facet solutions and smooth spatial screen representations are different
  products. Faceting is the production path; hybrid/screens require the guarded
  capabilities and a validated external-tool environment.
- Keep `apply_amplitudes`, `apply_fulljones`, `apply_normalizations`,
  `generate_screens` and `apply_screens` independent. Generating a product does
  not imply applying it.
- Preserve distinctions among images, models, residuals, dirty images, catalogs,
  masks and diagnostics. Renaming a product requires checking its consumers,
  discovery, finalizers and restart records.

See [operation semantics](../docs/source/operations.rst),
[product names](../docs/source/products.rst), and
[screen/normalization options](../docs/source/parset.rst).

## Checks for scientific changes

- Shorter solution intervals improve time resolution only when the calibrator
  signal supports them. More directions or weaker models can reduce solve SNR.
- Averaging changes measured samples and smearing; chunking partitions execution.
  Preserve sample membership and the intended solve intervals when changing
  chunk boundaries, and keep calibration/imaging averaging settings compatible.
- Model completeness, source filtering, beam conventions and normalization can
  change fluxes or astrometry even when RMS improves.
- For changes to products feeding later cycles, check relevant flux ratios,
  astrometry, RMS, dynamic range, source counts, unflagged fraction, restoring
  beam and diagnostics. Improvement in one metric alone is insufficient.

Use [parset definitions](../docs/source/parset.rst) for averaging and interval
settings, and [scientific tips](../docs/source/tips.rst) for data fractions,
direction counts and exported calibrated visibilities. Follow
[TESTING.md](../TESTING.md) for focused and external-tool validation.
