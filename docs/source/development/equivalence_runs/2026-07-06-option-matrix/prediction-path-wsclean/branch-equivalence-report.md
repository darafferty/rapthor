# Rapthor Branch Equivalence

Scenario: `prediction-path-wsclean`
Run root: `/tmp/rop-run3/prediction-path-wsclean`

## Branch Runs

| Side | Ref | Return Code | Parset | Work Dir | Log | Input Snapshot |
| --- | --- | ---: | --- | --- | --- | --- |
| base | `master` | 0 | `/app/docs/source/development/equivalence_runs/2026-07-06-option-matrix/inputs/base/prediction_path_wsclean.parset` | `/tmp/ropwb` | `/tmp/rop-run3/prediction-path-wsclean/base/rapthor-command.log` | parset: `inputs/base/prediction_path_wsclean.parset`, strategy: `inputs/base/prediction_path_wsclean_strategy.py` |
| current | `current` | 0 | `/app/docs/source/development/equivalence_runs/2026-07-06-option-matrix/inputs/current/prediction_path_wsclean.parset` | `/tmp/ropwc` | `/tmp/rop-run3/prediction-path-wsclean/current/rapthor-command.log` | parset: `inputs/current/prediction_path_wsclean.parset`, strategy: `inputs/current/prediction_path_wsclean_strategy.py` |

## Comparison Summary

| Result | Ops | Records | FITS | Image HDUs | Table HDUs | H5 | Text | Diagnostics | Visuals |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pass | 3 | 3 | 7 | 6 | 1 | 2 | 10 | 1 | 5 |

## FITS Residual Metrics

| Product | Max Abs Delta | P99 Abs Delta | Residual RMS | RMS / Ref RMS | RMS / Ref MAD |
| --- | ---: | ---: | ---: | ---: | ---: |
| `field-MFS-dirty.fits` | 9.537e-07 | 7.451e-08 | 2.607e-08 | 1.507e-07 | 1.679e-07 |
| `field-MFS-image-pb-ast.fits` | 9.537e-07 | 6.519e-08 | 2.292e-08 | 3.068e-07 | 1.370e-06 |
| `field-MFS-image-pb.fits` | 9.537e-07 | 6.519e-08 | 2.292e-08 | 3.068e-07 | 1.370e-06 |
| `field-MFS-image.fits` | 7.153e-07 | 6.380e-08 | 2.243e-08 | 3.043e-07 | 1.368e-06 |
| `field-MFS-model-pb.fits` | 3.576e-07 | 0.000e+00 | 1.769e-10 | 9.611e-08 | n/a |
| `field-MFS-residual.fits` | 1.639e-07 | 4.377e-08 | 1.675e-08 | 9.652e-07 | 1.024e-06 |

## Image Diagnostics

| Operation | Sector | Field | Reference | Current | Delta | Relative Delta |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `image_1` | `sector_1` | `nsources` | 1.000e+01 | 1.000e+01 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `theoretical_rms` | 9.006e-03 | 9.006e-03 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `min_rms_flat_noise` | 8.920e-03 | 8.918e-03 | -1.607e-06 | -0.018% |
| `image_1` | `sector_1` | `median_rms_flat_noise` | 1.593e-02 | 1.593e-02 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `dynamic_range_global_flat_noise` | 5.109e+02 | 5.110e+02 | 9.204e-02 | 0.018% |
| `image_1` | `sector_1` | `min_rms_true_sky` | 9.060e-03 | 9.060e-03 | 4.657e-09 | 0.000% |
| `image_1` | `sector_1` | `median_rms_true_sky` | 1.622e-02 | 1.622e-02 | 1.863e-09 | 0.000% |
| `image_1` | `sector_1` | `dynamic_range_global_true_sky` | 5.030e+02 | 5.030e+02 | -2.585e-04 | -0.000% |

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
