# External Tools Map

Rapthor orchestrates external radio astronomy tools. Keep command construction
and tool-specific behavior in execution owner packages, with tests that assert
the emitted command contracts.

`PLAN.md` asks for clearer preflight/runtime UX. When changing external tool
requirements, update feature detection, error messages, runtime docs, and tests
so users can see missing tools or unsupported runtime combinations before a long
flow fails.

## DP3

Used for LOFAR data processing tasks such as prediction, applycal, averaging,
and calibration solves.

- Command logic generally belongs under `rapthor/execution/<owner>/`.
- Preserve external argument names when they make the command easier to audit.
- Check data column names carefully: `DATA`, `CORRECTED_DATA`, model columns,
  and intermediate prediction columns have different meanings.

## WSClean

Used for imaging, deconvolution, wide-field imaging, facet imaging, primary-beam
products, and model extraction.

- Imaging command changes belong under `rapthor/execution/image/`.
- Be careful with image/model/residual/dirty products and FITS sidecars.
- `skip_final_major_iteration` is an intermediate-cycle speed choice; final
  science products still need full-quality image outputs.
- `dd_psf_grid`, `dde_method`, faceting options, masks, and source filtering can
  change scientific interpretation, not just runtime.

## EveryBeam

Used through the LOFAR/SKA toolchain for station beam models in prediction,
calibration, sky-model apparent/true conversions, and primary-beam correction.

- Keep true-sky and apparent-sky products explicit.
- Beam-related changes need photometry and possibly astrometry checks.
- Do not mix primary-beam-corrected products with flat-noise products under
  ambiguous names.

## IDG And IDGCal

Used for image-domain gridding paths and planned or guarded screen generation.

- `dde_mode = faceting` is the production default.
- Treat `hybrid`, `generate_screens`, and `apply_screens` as distinct product
  states.
- Screen products are smooth DD surfaces, not ordinary facet h5parm solutions.

## PyBDSF

Used for source finding, catalogs, masks, filtering, diagnostics, and
normalization workflows.

- Source filtering affects later calibration models.
- Changes to thresholds, masks, island handling, or catalog matching need
  sky-model and image-diagnostic checks.

## Casacore And Measurement Sets

Used for Measurement Set metadata and table access. Some environments require
compiled `python-casacore` and Casacore system libraries.

- Prefer existing small fixtures in `tests/resources/`.
- Do not add large Measurement Sets or generated run roots to version control.
- If tests need Casacore, keep them focused and mark heavier cases correctly.

## Prefect And Dask

Used for production orchestration and distributed task execution.

- Worker payloads must remain serializable.
- Avoid hidden global state in task functions.
- Prefect-marked tests run serially in the tox split; mirror that behavior when
  reproducing tox manually.

## Containers, MPI, And Cluster Runtime

Rapthor can run inside containers and on cluster backends such as Slurm.

- Runtime bootstrap and command profile changes belong under
  `rapthor/execution/`.
- Do not assume local paths are shared across nodes unless the option contract
  says they are.
- Integration, equivalence, and demo outputs can be large; put run roots under
  `/tmp` or another spacious filesystem.

## Benchmarking And Profiling

- Use `prefect_command_profile = time` or the documented profiling mode when
  collecting command timings for benchmark work.
- Benchmark reports should connect external-command timings,
  `logs/commands.jsonl`, Prefect task durations, Dask scheduler gaps, and
  scientific output equivalence.
- Do not split external commands into smaller tasks without benchmark evidence
  and restart/output-record coverage.
