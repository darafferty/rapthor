# Rapthor Branch Equivalence

Scenario: `bda-averaging`
Run root: `/tmp/rob-bda`

## Branch Runs

| Side | Ref | Return Code | Parset | Work Dir | Log | Input Snapshot |
| --- | --- | ---: | --- | --- | --- | --- |
| base | `master` | 0 | `/app/docs/source/development/science_equivalence_runs/2026-07-06-option-matrix/inputs/base/bda_averaging.parset` | `/tmp/robb` | `/tmp/rob-bda/base/rapthor-command.log` | parset: `inputs/base/bda_averaging.parset`, strategy: `inputs/base/bda_averaging_strategy.py` |
| current | `current` | 0 | `/app/docs/source/development/science_equivalence_runs/2026-07-06-option-matrix/inputs/current/bda_averaging.parset` | `/tmp/robc` | `/tmp/rob-bda/current/rapthor-command.log` | parset: `inputs/current/bda_averaging.parset`, strategy: `inputs/current/bda_averaging_strategy.py` |

## Comparison Summary

| Result | Ops | Records | FITS | Image HDUs | Table HDUs | H5 | Text | Diagnostics | Visuals |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pass | 3 | 3 | 7 | 6 | 1 | 2 | 10 | 1 | 5 |

## FITS Residual Metrics

| Product | Max Abs Delta | P99 Abs Delta | Residual RMS | RMS / Ref RMS | RMS / Ref MAD |
| --- | ---: | ---: | ---: | ---: | ---: |
| `field-MFS-dirty.fits` | 9.537e-07 | 7.916e-08 | 2.797e-08 | 1.519e-07 | 1.611e-07 |
| `field-MFS-image-pb-ast.fits` | 9.537e-07 | 7.451e-08 | 2.750e-08 | 3.407e-07 | 5.533e-07 |
| `field-MFS-image-pb.fits` | 9.537e-07 | 7.451e-08 | 2.750e-08 | 3.407e-07 | 5.533e-07 |
| `field-MFS-image.fits` | 9.537e-07 | 7.264e-08 | 2.700e-08 | 3.392e-07 | 5.545e-07 |
| `field-MFS-residual.fits` | 1.788e-07 | 6.333e-08 | 2.405e-08 | 4.787e-07 | 4.954e-07 |
| `field-MFS-model-pb.fits` | 1.192e-07 | 0.000e+00 | 1.002e-10 | 7.187e-08 | n/a |

## Image Diagnostics

| Operation | Sector | Field | Reference | Current | Delta | Relative Delta |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `image_1` | `sector_1` | `nsources` | 1.500e+01 | 1.500e+01 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `theoretical_rms` | 9.006e-03 | 9.006e-03 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `min_rms_flat_noise` | 3.180e-02 | 3.180e-02 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `median_rms_flat_noise` | 4.735e-02 | 4.735e-02 | 3.725e-09 | 0.000% |
| `image_1` | `sector_1` | `dynamic_range_global_flat_noise` | 1.432e+02 | 1.432e+02 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `min_rms_true_sky` | 3.233e-02 | 3.233e-02 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `median_rms_true_sky` | 4.836e-02 | 4.836e-02 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `dynamic_range_global_true_sky` | 1.408e+02 | 1.408e+02 | 0.000e+00 | 0.000% |

## Visual Comparisons

Five visual comparison PNGs were generated under the ignored run root. They are
not tracked because the numeric FITS, h5parm, diagnostic, and classification
summaries are the review contract for this pass.

## Difference Classification

| Category | Disposition | Count | Examples | Recommendation |
| --- | --- | ---: | --- | --- |
| `output_record_auxiliary_artifacts` | warning | 1 | `calibrate_1` | Keep as non-blocking when only plot filenames or known local intermediate aliases differ and final scientific products pass. |

## Warnings

- output-record auxiliary artifact basenames differ for calibrate_1
