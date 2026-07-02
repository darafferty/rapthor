# Scientific Glossary And Self-Calibration Guide

This document is a developer-facing reference for the scientific vocabulary used
in Rapthor. It is meant to help contributors and AI agents reason about radio
interferometry, direction-dependent self-calibration, and naming in the codebase.

It is not a replacement for a radio astronomy text. Use it as a map from
scientific concepts to Rapthor objects, options, products, and solve names.

## Mental Model

Radio interferometers do not directly measure images. They measure complex
correlations between pairs of antennas. Each correlation is a visibility sample,
and each antenna pair samples one spatial frequency set by the projected
baseline in wavelengths. Imaging is therefore an inverse problem: turn sparse,
noisy, instrument-corrupted visibility samples into a sky model and image.

The central calibration idea is the measurement equation. A useful simplified
form is:

```text
observed visibility = antenna/source corruptions * model visibility + noise
```

In Jones/RIME notation, those corruptions are represented by matrices or scalar
terms per antenna, time, frequency, direction, and polarization. Calibration
solves for those terms by comparing observed visibilities to visibilities
predicted from a sky model.

Self-calibration is an iterative loop:

1. Start with an initial target-field sky model.
2. Predict model visibilities from that sky model.
3. Solve for calibration terms that align observed and model visibilities.
4. Apply or use those terms while imaging.
5. Deconvolve and source-find to improve the sky model.
6. Repeat until image diagnostics stop improving or a strategy limit is reached.

Rapthor implements a direction-dependent version of this loop for LOFAR HBA
data, with ongoing SKA-Low support. The high-level loop is:

```text
Field + strategy
  -> optional initial image and sky model
  -> calibrate DI and/or DD
  -> predict/subtract sources when needed
  -> image sectors and optionally mosaic
  -> update sky models and diagnostics
  -> final imaging cycle
```

## Calibration Generations

**First-generation calibration, or 1GC**

Calibration from a separate calibrator observation. The calibrator has a known
or trusted model, solutions are derived from it, and those solutions are
transferred to the target. This is normally upstream of Rapthor. Rapthor assumes
LOFAR data have already passed through preparation such as LINC, and that
direction-independent preparation has been applied to the input data column.

**Second-generation calibration, or 2GC**

Direction-independent self-calibration on the target field. The target itself is
used to improve its sky model and solve for target-specific gains. In many
contexts, "selfcal" without a qualifier means this DI loop.

**Third-generation calibration, or 3GC**

Direction-dependent self-calibration. It solves or applies different corrections
for different sky directions because the ionosphere, primary beam, pointing
error, wide-field geometry, and other effects do not affect all sources equally.
Rapthor's main scientific value is in this 3GC regime.

## Rapthor Science Architecture

Use these names consistently when translating science to code.

**Field**

The whole target pointing and processing state. In code, `Field` owns the parset,
input `Observation` objects, strategy values for the current cycle, sky-model
state, h5parm state, sectors, calibrator patches, and diagnostics.

**Observation**

A physical Measurement Set time/frequency range. `Observation` stores MS
metadata such as pointing, channels, time sampling, station names, antenna type,
and calibration/imaging chunk parameters.

**Sector**

A sky region used by image and predict operations. Sectors are Rapthor image
work units. A sector can be an imaging sector, outlier sector, bright-source
sector, or predict sector. Do not use "sector" as a synonym for calibration
patch or facet.

**Patch**

A sky-model group used as a calibration direction. In LSM/WSClean sky models,
patches group one or more sources. In Rapthor, `calibrator_patch_names`,
`calibrator_fluxes`, and `calibration_skymodel.txt` refer to this concept.

**Facet**

A spatial partition used to approximate direction-dependent behavior across the
field. In Rapthor, calibration patches and imaging facets often correspond,
especially when a `facet_layout` is provided or Voronoi faceting is used. Use
"facet" for geometric regions and WSClean facet imaging, and "patch" for sky
model groups/directions.

**Direction**

The line of sight for a calibration solution. A direction may be represented by
a patch name, a source group, or a facet center. When writing code, prefer a
specific name such as `calibrator_patch_names`, `solve_directions`, or
`facet_region_file` instead of a vague `directions` variable unless it mirrors a
tool API.

## Rapthor Operation Loop

**Concatenate**

Combines frequency-split MS files for an epoch when needed. This is a data
layout operation, not a calibration step.

**Calibrate**

Builds DP3 or IDGCal solve commands, runs solve chunks, collects h5parm outputs,
plots solutions, optionally processes slow gains, and combines solution products.
It supports both `di` and `dd` modes.

**Predict**

Predicts and subtracts model visibilities for outlier sources, bright sources,
sector models, or DI full-Jones preparation. Prediction is about model
visibilities, not about solving.

**Image**

Prepares imaging Measurement Sets, applies available calibration products as
appropriate, runs WSClean, discovers FITS/skymodel/catalog products, and records
diagnostics. If multiple sectors exist, a mosaic operation combines them.

**ImageNormalize**

Builds and applies flux-scale normalization products. In naming, distinguish
this from image intensity normalization or display scaling.

## Rapthor Calibration Solve Taxonomy

The strategy interface is `calibration_strategy`, a dictionary whose top-level
keys are `di` and `dd`. Values are ordered lists of solve tokens.

| Strategy token | Scientific name | DP3 mode | Typical role | Rapthor products |
| --- | --- | --- | --- | --- |
| `fast_phase` | fast scalar phase solve | `scalarphase` | Short-timescale phase correction, mostly ionospheric structure on longer baselines. DD fast solves constrain core stations together. | `fast_phases.h5parm`, `field-solutions-fast-phase.h5`, or DI equivalents |
| `medium_phase` | medium scalar phase solve | `scalarphase` | Medium-timescale phase correction, often refining shorter-baseline or residual phase errors. The default DD strategy can run it before and after slow gains. | `medium1_phases.h5parm`, `medium2_phases.h5parm`, `field-solutions-medium*-phase.h5` |
| `slow_gains` | slow diagonal gain solve | `diagonal` | Longer-timescale XX/YY amplitude and phase correction, mostly beam and flux-scale structure. | `slow_gains.h5parm`, `field-solutions-slow-gain.h5` |
| `full_jones` | full-Jones solve | `fulljones` | Direction-independent 2x2 polarized gain solve. Used when cross-hand/full-matrix behavior matters. | `fulljones_solutions.h5`, `fulljones-solutions.h5` |

Naming rule: use the token `slow_gains` in strategy and payload code. Use "slow
gain" or "slow diagonal gain" in prose. Avoid introducing `slow_gain` as a new
strategy token unless adding an explicit compatibility migration.
The singular `slow_gain` still appears in established DD per-chunk filename
prefixes and labels, such as `slow_gain_0.h5parm`; do not copy that file-prefix
form into strategy values.

The default DD self-calibration solve order is:

```python
{"dd": ["fast_phase", "medium_phase", "slow_gains", "medium_phase"], "di": []}
```

The default HBA strategy used with `generate_initial_skymodel = True` runs this
full DD solve order in every selfcal cycle, including `slow_gains`. That choice
assumes Rapthor has built a target-field model from the data and uses long
slow-gain intervals plus first-cycle flux-scale normalization to keep amplitude
corrections constrained. When `generate_initial_skymodel = False`, the built-in
selfcal strategy starts with phase-only DD cycles and adds `slow_gains` later.
A custom strategy can also omit `slow_gains` in early cycles when the starting
model, calibrator signal-to-noise, or flux-scale diagnostics do not support
amplitude calibration yet.

## Direction-Independent And Direction-Dependent Effects

**Direction-independent effect, or DIE**

A corruption that affects all sky directions the same way for a given antenna,
time, frequency, and polarization. Examples include electronic gain drift and
station clock-like residuals after preparation. In code, use `di` for solve
mode and filenames such as `di_h5parm_filename`.

**Direction-dependent effect, or DDE**

A corruption that changes across the field of view. Examples include primary
beam attenuation, ionospheric phase screens, pointing error, wide-field
projection effects, and direction-varying PSFs. In code, use `dd` for solve
mode and names such as `dd_h5parm_filename`, `dde_mode`, and `dde_method`.

**DDE mode**

`dde_mode = faceting` means Rapthor uses Voronoi faceting throughout. `dde_mode
= hybrid` is intended to use faceting during selfcal and IDGCal screens in the
final cycle. The defaults currently describe hybrid as future/guarded
functionality, so treat it as a capability that needs feature checks and tests.

**DDE method**

`dde_method = full` means WSClean applies the full direction-dependent solution
field during imaging. `dde_method = single` means Rapthor applies one solution,
chosen by sector center, as if it were constant across the sector. Do not name
`single` behavior as truly DD in new code; it is a sector-wise approximation.

## Glossary

### Data, Geometry, And Coordinates

**Antenna / station**

An individual receiving element or phased station. LOFAR station names such as
`CS...` and `RS...` matter in calibration constraints. Code often uses
`stations`, `core_stations`, and `antenna` for HBA/LBA mode.

**Baseline**

The vector separation between two antennas. A baseline samples one Fourier
component of the sky brightness at a time/frequency-dependent `u,v,w` position.

**Baseline-dependent averaging, or BDA**

Averaging in time/frequency with limits that depend on baseline length. It
reduces data volume while controlling smearing. In Rapthor, calibration and
imaging have `bda_timebase` options, and calibration also has
`bda_frequencybase`.

**Channel**

A frequency bin in a Measurement Set or image cube. Calibration has frequency
solution intervals such as `fast_freqstep_hz`, `medium_freqstep_hz`, and
`slow_freqstep_hz`.

**Epoch**

A separate observation time range, such as a different night. Rapthor groups
input MS files into `epoch_observations` and may concatenate frequency pieces
inside an epoch.

**Field of view, or FoV**

The angular sky area observed or imaged. For LOFAR, the useful FoV is strongly
affected by the primary beam and low-frequency DDEs.

**Measurement Set, or MS**

The standard radio interferometry data container. It stores visibility data,
flags, weights, antenna metadata, spectral windows, fields, and related tables.
Rapthor should not modify the original input MS files.

**Phase center**

The reference sky direction used for correlation/fringe stopping and image
projection. Farther from the phase center, wide-field and smearing effects grow.

**Pointing center**

The direction the antennas or station beam are aimed at. It is often close to
the phase center, but do not assume they are identical unless the MS metadata or
tool contract says so.

**Time slot / integration**

One sampled time interval in the MS. Calibration solution intervals are usually
integer multiples of time slots.

**uv distance**

Projected baseline length in wavelengths. Cuts such as `solve_min_uv_lambda`,
`min_uv_lambda`, and `max_uv_lambda` select which spatial scales are used in
calibration or imaging.

**uvw coordinates**

Baseline coordinates in wavelengths relative to the phase center. `u` and `v`
are the usual image-plane Fourier coordinates; `w` is the line-of-sight
component that drives wide-field non-coplanar effects.

**Visibility**

A complex correlation measured by one baseline for one time, frequency, and
polarization product. Visibility data are the primary input to calibration and
imaging.

### Imaging And Deconvolution

**Apparent sky**

The sky as attenuated and distorted by the instrument response, especially the
primary beam. In Rapthor product names, apparent-sky images are not primary-beam
corrected and tend to have flatter noise. Apparent-sky sky models contain beam
attenuated flux densities.

**True sky**

The estimated intrinsic sky after primary-beam correction. In Rapthor product
names, `*-pb.fits` products and `*.true_sky.txt` sky models are intended to
represent true-sky fluxes.

Naming rule: never drop the `apparent_sky` or `true_sky` qualifier when both
could exist. Bugs in this distinction become flux-scale bugs.

**Clean component**

A component in a deconvolution model, often point-like or scale-like. Rapthor
uses WSClean-generated models and may filter components outside PyBDSF islands
before using them in later calibration.

**Clean mask**

A mask controlling where deconvolution can place components. Good masks can
help selfcal converge because the model is less likely to absorb noise or
artifacts.

**CLEAN / deconvolution**

The iterative process of separating a sky model from the dirty image and PSF.
Over-deconvolution can add noise to the model; under-deconvolution leaves
source flux in residuals. Both can harm selfcal.

**Dirty image**

The direct image made from sampled visibilities before deconvolution. It is the
true sky convolved with the dirty beam, plus noise and calibration artifacts.

**Dirty beam / PSF**

The point spread function implied by the sampling function, weighting, and
wide-field imaging method. Direction-dependent PSFs matter for wide-field LOFAR
imaging; Rapthor exposes `dd_psf_grid` for WSClean.

**Dynamic range**

An image-quality proxy, often peak brightness divided by an RMS noise estimate.
It is useful for cycle-to-cycle comparison but can hide local artifacts.

**Gridding / degridding**

Gridding maps irregular visibility samples onto a regular Fourier grid for FFT
imaging. Degridding predicts model visibilities from an image or sky model at
the measured sample positions.

**Image cube**

A stack of channel images, often used for spectral analysis. Rapthor can save
final image cubes for selected Stokes parameters with `save_image_cube` and
`image_cube_stokes_list`.

**Major iteration**

An imaging iteration that alternates between deconvolution and visibility-domain
prediction/subtraction. Rapthor can skip the final WSClean major iteration in
intermediate selfcal cycles because the sky model remains useful while imaging
runtime drops.

**MFS, or multi-frequency synthesis**

Combines frequency coverage to improve uv sampling and image sensitivity. The
frequency dependence of source spectra and instrumental effects still matters.

**Residual image**

The image left after subtracting the current model. Residual structure around
bright sources is often a calibration, deconvolution, or DDE clue.

**Restoring beam**

The idealized beam convolved with the deconvolved model before final image
products are written. Rapthor records restoring beam diagnostics.

**Source finder**

A tool that identifies islands and sources in an image. Rapthor uses PyBDSF
style products for catalogs, masks, filtering, diagnostics, and normalization.

**Time and bandwidth smearing**

Loss of peak flux and radial/azimuthal broadening caused by averaging in time or
frequency while phase changes across the averaging bin. It grows with baseline
length and distance from the phase center. Rapthor uses `max_peak_smearing`,
`average_visibilities`, and optional smearing corrections.

**W-term**

The non-coplanar baseline term in wide-field imaging. Ignoring it makes a
direction-dependent phase error that grows away from the phase center. Faceting,
w-projection, and related methods exist to control it.

### Calibration And RIME Terms

**Antenna gain**

A complex per-antenna correction. Gains can include amplitude, phase, or a full
2x2 matrix depending on solve mode.

**Applycal**

Applying existing calibration solutions to data or model data. Do not use
`applycal` to mean solving. In Rapthor, pre-application steps include names such
as `fastphase`, `slowgain`, `fulljones`, and `normalization`.

**Calibration direction**

The direction for which a solution is solved or applied. In DD mode, this is
usually a calibrator patch/facet direction.

**Calibration patch**

A sky-model patch used as a DD solve direction. Rapthor chooses bright compact
source groups as calibrator patches, tessellates the full sky model around them,
and still uses all sources in the calibration model.

**Calibration strategy**

The ordered DI/DD solve plan for one cycle. In code, `calibration_strategy`
means a dictionary like `{"dd": [...], "di": [...]}`. It is not the same as the
top-level processing `strategy`, which is `selfcal`, `image`, or a custom
strategy file.

**Closure quantity**

A combination of visibility phases or amplitudes that cancels antenna-based
errors. Historically important for calibration and still useful diagnostically,
but Rapthor's selfcal path is least-squares/RIME based.

**Core station constraint**

A constraint that forces selected LOFAR core stations to share solutions. This
can stabilize fast phase solves where the array core samples similar ionospheric
or instrumental behavior.

**Diagonal gain**

A 2x2 gain matrix with only XX and YY diagonal terms. Rapthor's `slow_gains`
solve is diagonal, solving phase and amplitude without cross-hand terms.

**Differential gain**

A direction-specific gain term, often described as a way to extend a DI RIME to
multiple source directions. It is a conceptual bridge to DD calibration.

**Flux-scale normalization**

A correction that adjusts frequency-dependent station amplitudes so measured
fluxes match external reference catalogs or supplied normalization models. In
Rapthor, this produces a normalization h5parm and is distinct from image display
normalization.

**Full-Jones**

A full 2x2 complex gain matrix. In Rapthor, `full_jones` is a DI solve token and
uses full-Jones h5parm products. Avoid spelling drift: prose can say
"full-Jones", but strategy code uses `full_jones`; filenames often use
`fulljones`.

**h5parm**

The HDF5 solution-table format used by LOFAR tooling. Rapthor uses h5parm files
for DD scalar/diagonal solutions, DI full-Jones solutions, normalization
solutions, and future screen products. Important solsets/soltabs include
`sol000`, `phase000`, `amplitude000`, and, for screens, `coefficients000`.

**Initial solutions**

Solutions used as the starting point for a later solve. Be explicit about
cycle, solve type, and mode. Rapthor deliberately avoids silently reusing
solutions from older cycles unless the strategy and operation contract allow it.

**Jones matrix**

A matrix representation of how one propagation or instrumental effect changes
the electric field for one antenna or direction. Jones chains compose multiple
effects.

**Least-squares calibration**

Solving gains by minimizing the mismatch between observed and model
visibilities. Most Rapthor solve commands are deterministic builders around
external least-squares solvers in DP3 or IDGCal.

**Model visibility**

The visibility predicted from the current sky model, beam model, and imaging or
prediction method. Name this as `model_visibility`, `modeldatacolumn`, or
`predict_*` rather than a generic `model` when ambiguity is likely.

**Phase-only solve**

A solve where only phase is adjusted. Rapthor's `fast_phase` and
`medium_phase` solves are scalar phase solves.

**Primary beam**

The direction-dependent antenna or station response. It attenuates sources away
from the pointing center and can be polarized, time-variable, and
frequency-variable. Rapthor uses EveryBeam/DP3/WSClean beam handling and has
`onebeamperpatch` as a calibration speed/accuracy tradeoff.

**RIME, or Radio Interferometer Measurement Equation**

The formal equation relating sky coherency, geometric phase, propagation,
instrumental Jones terms, and measured visibilities. Use it as the conceptual
test for new calibration logic: what term are we solving, applying, predicting,
or ignoring?

**Scalar phase**

A single phase correction applied equally to relevant polarization products. In
Rapthor, this maps to DP3 `scalarphase`.

**Screen**

A smooth 2-D direction-dependent model, often used for ionospheric phase or
beam-like effects. Rapthor's `hybrid` mode plans to generate IDGCal screens in
the final cycle and apply them during imaging. Use "screen" only for this smooth
surface representation, not for ordinary facet h5parm solutions.

**Solution interval**

The time and/or frequency span over which one solution is fitted. Shorter
intervals capture faster variations but need more signal-to-noise; longer
intervals are more stable but can smear real changes. Rapthor strategy values
include `fast_timestep_sec`, `medium_timestep_sec`, `slow_timestep_sec`, and
`fulljones_timestep_sec`.

**Smoothness constraint**

A regularization constraint that discourages noisy or unphysical variation in
solutions across time/frequency/direction. Rapthor has separate fast, medium,
slow, and full-Jones smoothness settings.

**Sky model**

A representation of source positions, flux densities, shapes, spectra, and
patch grouping. In selfcal, the sky model is both an input to calibration and an
output of imaging/deconvolution.

**Stokes I, Q, U, V**

The standard polarization basis for total intensity and polarized emission.
Rapthor primarily images Stokes I, with optional final Q/U/V products and image
cubes.

### Rapthor Tooling Terms

**DP3**

The LOFAR data-processing tool Rapthor uses for prediction, applycal,
averaging, and DDE calibration solves.

**EveryBeam**

A beam-model library used by LOFAR/SKA tooling. In Rapthor, beam behavior
usually enters through DP3, WSClean, and sky-model conversion workflows for
prediction, calibration, apparent/true-sky handling, and primary-beam
correction.

**IDG / IDGCal**

Image Domain Gridding and its calibration path. Rapthor uses IDG-related code
for the `hybrid` screen path and WSClean IDG modes.

**LINC**

The upstream LOFAR initial calibration pipeline. Rapthor expects LINC-like
preparation before its DD selfcal work.

**LoTSS**

The LOFAR Two-metre Sky Survey. Rapthor inherits many practical conventions
from LoTSS processing, including flux/astrometry diagnostics and target-field
selfcal expectations.

**PyBDSF / PyBDSM**

Source finding and catalog tooling used by LOFAR pipelines. Rapthor uses PyBDSF
style source catalogs, island masks, and diagnostics.

**WSClean**

The imager used by Rapthor for CLEAN, wide-field imaging, facet imaging,
primary-beam products, and model extraction.

## Naming Guidance

### Prefer These Canonical Terms

| Concept | Prefer | Avoid |
| --- | --- | --- |
| Direction-independent mode | `di` in strategy/payload keys, "DI" in prose | `independent`, `dir_indep`, `global` |
| Direction-dependent mode | `dd` in strategy/payload keys, "DD" in prose | `dependent`, `dir_dep`, `local` |
| Slow diagonal solve token | `slow_gains` | `slow_gain` as a strategy token |
| Full-Jones solve token | `full_jones` | `fulljones` in strategy values |
| Full-Jones filenames/labels | `fulljones` when matching existing file names | mixed `full_jones` filenames unless changing all references |
| Calibration patch list | `calibrator_patch_names` | `calibrators` if it may mean sources, patches, or directions |
| Image work unit | `sector` | `facet` unless WSClean/facet geometry is meant |
| Sky-model group | `patch` | `sector` |
| Physical correction direction | `direction` with a qualifier | bare `dir` or ambiguous `source` |
| True flux sky model | `true_sky` | `pbcorrected`, `intrinsic` in new file names |
| Beam-attenuated sky model | `apparent_sky` | `observed_sky`, `uncorrected` |
| Solution file | `h5parm` | `calfile`, `parmdb` unless matching an external API |
| Prediction step | `predict` | `model` as a verb |
| Applying existing solutions | `applycal` or `apply_*_solutions` | `calibrate` |
| Solving for new solutions | `solve` or `calibrate` | `apply` |

### Use The Layer's Vocabulary

- Domain objects should use scientific nouns: `Field`, `Observation`, `Sector`,
  `calibration_strategy`, `calibrator_patch_names`.
- Operation adapters should use lifecycle and hand-off nouns:
  `set_input_parameters`, `finalize`, `*_h5parm_filename`, `cycle_number`.
- Payload builders should use serializable contract nouns:
  `solve_slots`, `chunks`, `collected_h5parms`, `combined_h5parms`.
- Command builders should mirror external tool arguments when necessary:
  `msin`, `msin.datacolumn`, `solve1.mode`, `applycal.parmdb`.

### Keep Cycle Locality Visible

Solutions from one selfcal cycle should not silently leak into later cycles.
When a variable holds a solution path, include enough context to know whether it
is current-cycle, previous-cycle, DI, DD, full-Jones, or normalization:

```text
dd_h5parm_filename
dd_h5parm_cycle_number
di_h5parm_filename
fulljones_h5parm_cycle_number
normalize_h5parm
```

If adding a new solution product, also add or update cycle-number tracking,
finalizer behavior, operation tests, and restart behavior.

### Distinguish Product State From Processing State

- `apply_amplitudes` means the active h5parm contains amplitude solutions that
  should affect later prediction/imaging.
- `apply_fulljones` means a full-Jones h5parm is active.
- `apply_normalizations` means flux-scale normalization is active.
- `generate_screens` means solve screen products now.
- `apply_screens` means image with screen products now.

Do not collapse these into a generic `apply_calibration` flag; they drive
different external tool arguments.

## Configuration Tips For Scientific Results

There is no single best Rapthor configuration for every field. Treat these as
science-facing heuristics for choosing parset and strategy values, then verify
the result with image diagnostics.

**Start from the strongest sky model you can justify**

Use a trusted `input_skymodel` and matching `apparent_skymodel` when available.
The source names must match exactly because DD h5parm direction names are tied
to sky-model patches. If no trusted model is available, keep
`generate_initial_skymodel = True` so Rapthor builds a target-field model from
the data. Use `download_initial_skymodel` only when generation is disabled or
inappropriate and catalog access is scientifically acceptable.

Keep `regroup_input_skymodel = True` unless the input patches encode a deliberate
calibration geometry. Regrouping lets Rapthor build calibrator patches with more
consistent signal-to-noise, while a fixed `facet_layout` is better when you need
stable directions for comparison runs, restarts, or externally supplied h5parm
solutions.

**Prefer a conservative calibration progression**

For normal LOFAR HBA wide-field work where Rapthor generates the initial sky
model, start from the default DD solve order:

```python
{"dd": ["fast_phase", "medium_phase", "slow_gains", "medium_phase"], "di": []}
```

In that default, `fast_phase` and the first `medium_phase` solve establish the
phase structure before the slower diagonal amplitude/phase solve, and the final
`medium_phase` solve cleans up residual phase structure after `slow_gains`.
The slow-gain interval is deliberately long, and flux-scale normalization is
derived in the first cycle and then propagated forward. This is a constrained
amplitude step, not a license to solve amplitudes blindly. Prefer amplitude
solves only when the starting model, calibrator signal-to-noise, solution plots,
and flux-scale diagnostics support them.

If the run starts from a downloaded or user-provided sky model
(`generate_initial_skymodel = False`), Rapthor's built-in selfcal strategy
already starts with phase-only DD cycles before adding `slow_gains`. For weak
fields, noisy early solution plots, or unstable photometry/astrometry checks,
use an even more conservative custom strategy with phase-only DI or DD cycles
first and add `slow_gains` only after the model and phase solutions are stable
enough to support amplitude calibration.

**Use DI-only cycles as a bootstrap or diagnostic, not as a DD replacement**

DI-only cycles are appropriate when the dominant residual is plausibly common
to the whole field, or when the data do not yet support splitting the solve into
many directions. In strategy terms, that means a cycle like
`{"di": ["fast_phase"], "dd": []}` or, when polarization and cross-hand behavior
matter, `{"di": ["full_jones"], "dd": []}`. This can be useful at the start of
a custom strategy to stabilize a weak target-field model, remove residual
target-wide phase structure from upstream calibration, or get a high-S/N
full-field correction before asking for DD solutions.

DI-only cycles are also useful for narrow-field or central-target science where
off-axis beam and ionospheric structure are not the limiting errors, and for
controlled comparisons that show whether DD calibration is actually improving
the image. The tradeoff is that DI solves have fewer degrees of freedom and
higher effective signal-to-noise, but they cannot correct spatially varying
DDEs. If residuals change across facets, artifacts cluster around off-axis
bright sources, or source positions/fluxes vary with direction, switch to DD
calibration rather than adding more DI cycles.

The `examples/flexible_calibration_strategy.py` pattern is the intended mental
model: optional DI-only bootstrap cycles first, then DD phase cycles, then the
default DD solve order once the model and diagnostics can support it.

**Choose solution intervals for signal-to-noise, not just resolution**

Shorter `fast_timestep_sec` and `medium_timestep_sec` values can track faster
phase variation, but they need stronger calibrators. Longer `slow_timestep_sec`
and `fulljones_timestep_sec` values are usually more stable for amplitude or
full-matrix solves. If faint directions produce noisy solutions, prefer fewer
calibrator directions, longer intervals, or stronger smoothing over simply
solving more often. Watch the solution plots and image metrics together: a
calibration table that looks detailed but worsens RMS, dynamic range, flux
ratios, or astrometry is not a scientific improvement.

**Use enough DD directions, but not more than the data support**

The `target_flux`, `max_directions`, and `max_distance` strategy values control
how many DD calibrator patches are used. Too few directions leave real DDEs
unmodelled; too many directions split the signal-to-noise and can make the
solutions unstable. Prefer bright, compact calibrator patches, and keep the
sector/facet/patch distinction clear when reviewing a run. If supplying an
`input_h5parm`, make sure its directions, time coverage, and frequency coverage
match the selected sky model or `facet_layout`.

**Protect the sky model from noise**

Keep `filter_skymodel = True` for ordinary selfcal so WSClean clean components
outside PyBDSF islands are not fed back into calibration. Multiscale cleaning is
usually helpful for extended emission, but masks and thresholds still matter:
over-cleaning can move noise into the model and make later solves chase it.
`skip_final_major_iteration = True` is a reasonable speed-up for intermediate
selfcal cycles because it does not change the sky model used for calibration;
the final cycle should still produce the full image products.

**Be deliberate about averaging and smearing**

`average_visibilities = True` is normally useful for run time, but the scientific
limit is `max_peak_smearing`. Lower `max_peak_smearing` for photometry,
astrometry, or wide fields where edge sources matter; higher values are a speed
tradeoff. If `correct_time_frequency_smearing` is enabled, keep the calibration
and imaging settings aligned so prediction and imaging make the same assumption
about averaged data.

**Keep final science products stricter than intermediate products**

It is reasonable to use `selfcal_data_fraction < 1` during calibration to reduce
cost, but prefer `final_data_fraction = 1.0` for science images when resources
allow. Use `ntimes_to_repeat_final_cycle` only when the extra final pass improves
diagnostics rather than simply making the run longer. Enable `save_image_cube`,
`image_cube_stokes_list`, `make_quv_images`, or `save_supplementary_images` when
those products are needed for downstream science or quality review, but do not
make them a substitute for checking the primary Stokes-I continuum image.

**Treat flux scale and astrometry as first-class outputs**

Amplitude solves, `do_normalize`, `input_normalization_h5parm`, and
`normalization_skymodels` all affect scientific flux densities. Use external
photometry and astrometry sky models when possible, and compare RMS, dynamic
range, source count, flux ratios, positional offsets, unflagged fraction, and
restoring beam across cycles. A run that only improves one scalar metric can
still be scientifically worse if it moves sources, changes flux scales, or
creates implausible beam or residual structure.

**Use production DD imaging paths unless testing a guarded feature**

`dde_mode = faceting` is the production default. `dde_mode = hybrid` and screen
products should be treated as guarded or future-facing unless the runtime and
scientific validation for that path are available. For final DD imaging,
`dde_method = full` applies the full direction-dependent solution; `single` is a
speed or approximation choice that applies one solution per sector.

## Key Scientific Checks For Code Changes

Use this checklist when changing calibration, prediction, imaging, or sky-model
logic.

1. What measurement-equation term is affected?

   Name whether the change touches geometry, beam, ionosphere, scalar phase,
   diagonal gain, full-Jones gain, normalization, smearing, or source model.

2. Is the change DI, DD, or both?

   Update `calibration_strategy`, feature collection, payload validation, command
   references, and finalizer state accordingly.

3. Is the product apparent-sky or true-sky?

   Preserve explicit names in files, payloads, and docs. Do not mix
   primary-beam-corrected products with flat-noise products.

4. Does the change affect solve intervals or smoothing?

   Check time/frequency chunking, `solint_*` values, BDA behavior, smoothness
   constraints, and signal-to-noise assumptions for faint calibrators.

5. Does the change affect sky-model grouping?

   Check patch names, source names, facet layout behavior, calibrator selection,
   outlier sectors, bright-source sectors, and duplicate handling between
   cycles.

6. Does the change affect model prediction?

   Check whether the model is component-based, image-based, beam-applied,
   normalized, full-Jones-preapplied, or smearing-corrected.

7. Does the change affect final image quality metrics?

   Check RMS, dynamic range, source count, flux ratios, astrometric offsets,
   unflagged fraction, restoring beam, and diagnostic JSON/artifacts.

8. Does the change require external-tool capability checks?

   Update preflight feature detection for WSClean, DP3, IDGCal, EveryBeam, MPI,
   shared facet reads/writes, or internet/catalog access.

## Common Failure Modes

**Model incompleteness**

Selfcal can converge toward the wrong answer if the sky model is incomplete or
biased. Early phase-only cycles are safer than amplitude solves when the model
is weak.

**Over-cleaning**

Deconvolution can move noise into the sky model. That noise then becomes part
of the next calibration prediction. Masking and source filtering are important
scientific controls, not cosmetic steps.

**Flux-scale drift**

Amplitude solves and normalization can change flux scales. Any change to slow
gains, normalization, or apparent/true-sky conversion needs photometry checks.

**Astrometric drift**

DD phase errors can move source positions. Rapthor image diagnostics compare
against external astrometric references.

**Direction mismatch**

DD h5parm source names and sky-model patch names must match, or be adjusted
deliberately. Mismatches can apply a valid solution to the wrong direction.

**Cycle leakage**

Reusing old h5parm files can make a run appear stable while using stale
solutions. Keep cycle-local filtering and tests intact.

**Ambiguous sector/facet/patch naming**

This is a common source of incorrect geometry. A sector is an image work unit, a
facet is a sky partition for DD imaging/calibration geometry, and a patch is a
sky-model group/direction.

**Smearing by averaging**

Averaging can reduce data volume but lower peak flux away from the phase center.
Any change to BDA or imaging averaging should be checked against
`max_peak_smearing` and smearing-correction options.

## AI Agent Guidance

When asked to modify scientific logic in Rapthor:

1. Read the domain object and operation adapter before changing execution code.
2. Identify whether the change belongs in `rapthor.lib`, `rapthor.operations`,
   or an owner package under `rapthor.execution`.
3. Preserve plain serializable payloads between Prefect/Dask tasks.
4. Prefer adding a narrow payload/command/finalizer test before broad
   integration tests.
5. Check command reference fixtures when command tokens change.
6. Update defaults, docs, examples, payload builders, validators, and tests
   together when adding a user-facing scientific option.
7. Do not rename scientific tokens casually. If a rename is needed, support a
   migration path and document old/new names.
8. Treat external catalogs and internet access as optional unless the parset
   explicitly allows them or user-provided files are available.

## Quick Term Map

```text
visibility data       -> Measurement Sets, DATA/CORRECTED_DATA columns
sky model             -> WSClean/LSM source lists, true_sky/apparent_sky files
calibration direction -> sky-model patch, calibrator patch, facet center
DI calibration        -> calibration_strategy["di"], di h5parm products
DD calibration        -> calibration_strategy["dd"], dd h5parm products
phase solve           -> fast_phase, medium_phase, scalarphase
amplitude+phase solve -> slow_gains, diagonal
full matrix solve     -> full_jones, fulljones products
apply existing terms  -> applycal, preapply steps, WSClean facet solutions
image work unit       -> Sector
geometric DD region   -> Facet
smooth DD surface     -> Screen
quality checks        -> RMS, dynamic range, source count, flux ratio, astrometry
```
