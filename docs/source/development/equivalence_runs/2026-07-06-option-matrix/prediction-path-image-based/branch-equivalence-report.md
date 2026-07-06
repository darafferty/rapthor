# Rapthor Branch Equivalence

Scenario: `prediction-path-image-based`
Run root: `/tmp/rob-run/prediction-path-image-based`

## Branch Runs

| Side | Ref | Return Code | Parset | Work Dir | Log | Input Snapshot |
| --- | --- | ---: | --- | --- | --- | --- |
| base | `master` | 0 | `/app/docs/source/development/equivalence_runs/2026-07-06-option-matrix/inputs/base/prediction_path_image_based.parset` | `/tmp/ropib` | `/tmp/rob-run/prediction-path-image-based/base/rapthor-command.log` | parset: `inputs/base/prediction_path_image_based.parset`, strategy: `inputs/base/prediction_path_image_based_strategy.py` |
| current | `current` | 0 | `/app/docs/source/development/equivalence_runs/2026-07-06-option-matrix/inputs/current/prediction_path_image_based.parset` | `/tmp/ropic` | `/tmp/rob-run/prediction-path-image-based/current/rapthor-command.log` | parset: `inputs/current/prediction_path_image_based.parset`, strategy: `inputs/current/prediction_path_image_based_strategy.py` |

## Comparison Summary

| Result | Ops | Records | FITS | Image HDUs | Table HDUs | H5 | Text | Diagnostics | Visuals |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pass | 3 | 3 | 7 | 6 | 1 | 2 | 10 | 1 | 5 |

## FITS Residual Metrics

| Product | Max Abs Delta | P99 Abs Delta | Residual RMS | RMS / Ref RMS | RMS / Ref MAD |
| --- | ---: | ---: | ---: | ---: | ---: |
| `field-MFS-dirty.fits` | 9.537e-07 | 7.451e-08 | 2.809e-08 | 1.625e-07 | 1.811e-07 |
| `field-MFS-image-pb-ast.fits` | 9.537e-07 | 7.451e-08 | 2.628e-08 | 3.519e-07 | 1.612e-06 |
| `field-MFS-image-pb.fits` | 9.537e-07 | 7.451e-08 | 2.628e-08 | 3.519e-07 | 1.612e-06 |
| `field-MFS-image.fits` | 9.537e-07 | 7.311e-08 | 2.578e-08 | 3.498e-07 | 1.615e-06 |
| `field-MFS-model-pb.fits` | 3.576e-07 | 0.000e+00 | 2.150e-10 | 1.186e-07 | n/a |
| `field-MFS-residual.fits` | 2.831e-07 | 4.889e-08 | 1.886e-08 | 1.109e-06 | 1.183e-06 |

## Image Diagnostics

| Operation | Sector | Field | Reference | Current | Delta | Relative Delta |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `image_1` | `sector_1` | `nsources` | 1.000e+01 | 1.000e+01 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `theoretical_rms` | 9.006e-03 | 9.006e-03 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `min_rms_flat_noise` | 8.515e-03 | 8.515e-03 | 1.034e-07 | 0.001% |
| `image_1` | `sector_1` | `median_rms_flat_noise` | 1.555e-02 | 1.555e-02 | -4.657e-09 | -0.000% |
| `image_1` | `sector_1` | `dynamic_range_global_flat_noise` | 5.368e+02 | 5.368e+02 | -6.518e-03 | -0.001% |
| `image_1` | `sector_1` | `min_rms_true_sky` | 8.618e-03 | 8.618e-03 | 2.282e-07 | 0.003% |
| `image_1` | `sector_1` | `median_rms_true_sky` | 1.584e-02 | 1.584e-02 | -1.863e-09 | -0.000% |
| `image_1` | `sector_1` | `dynamic_range_global_true_sky` | 5.304e+02 | 5.304e+02 | -1.404e-02 | -0.003% |

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
