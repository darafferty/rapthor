# Rapthor Branch Equivalence

Scenario: `normalization-rich-demo`
Run root: `/tmp/rom5/normalization-rich-demo`

## Branch Runs

| Side | Ref | Return Code | Parset | Work Dir | Log | Input Snapshot |
| --- | --- | ---: | --- | --- | --- | --- |
| base | `master` | 0 | `/app/docs/source/development/equivalence_runs/2026-07-06-option-matrix/inputs/base/normalization_rich_demo.parset` | `/tmp/rom-base` | `/tmp/rom5/normalization-rich-demo/base/rapthor-command.log` | parset: `inputs/base/normalization_rich_demo.parset`, strategy: `inputs/base/normalization_rich_demo_strategy.py` |
| current | `current` | 0 | `/app/docs/source/development/equivalence_runs/2026-07-06-option-matrix/inputs/current/normalization_rich_demo.parset` | `/tmp/rom-current` | `/tmp/rom5/normalization-rich-demo/current/rapthor-command.log` | parset: `inputs/current/normalization_rich_demo.parset`, strategy: `inputs/current/normalization_rich_demo_strategy.py` |

## Comparison Summary

| Result | Ops | Records | FITS | Image HDUs | Table HDUs | H5 | Text | Diagnostics | Visuals |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pass | 4 | 4 | 8 | 7 | 1 | 3 | 12 | 1 | 5 |

## FITS Residual Metrics

| Product | Max Abs Delta | P99 Abs Delta | Residual RMS | RMS / Ref RMS | RMS / Ref MAD |
| --- | ---: | ---: | ---: | ---: | ---: |
| `field-MFS-image-pb-ast.fits` | 9.537e-07 | 6.333e-08 | 2.315e-08 | 3.768e-07 | 1.523e-06 |
| `field-MFS-image-pb.fits` | 9.537e-07 | 6.333e-08 | 2.315e-08 | 3.768e-07 | 1.523e-06 |
| `sector_1_I_freq_cube.fits` | 9.537e-07 | 1.118e-07 | 3.803e-08 | 4.492e-07 | 1.332e-06 |
| `field-MFS-dirty.fits` | 7.153e-07 | 5.960e-08 | 2.164e-08 | 1.530e-07 | 1.707e-07 |
| `field-MFS-image.fits` | 7.153e-07 | 6.240e-08 | 2.259e-08 | 3.726e-07 | 1.517e-06 |
| `field-MFS-residual.fits` | 5.513e-07 | 4.936e-08 | 1.895e-08 | 1.209e-06 | 1.275e-06 |
| `field-MFS-model-pb.fits` | 1.192e-07 | 0.000e+00 | 9.161e-11 | 6.278e-08 | n/a |

## Image Diagnostics

| Operation | Sector | Field | Reference | Current | Delta | Relative Delta |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `image_1` | `sector_1` | `nsources` | 1.000e+01 | 1.000e+01 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `theoretical_rms` | 9.006e-03 | 9.006e-03 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `min_rms_flat_noise` | 7.544e-03 | 7.534e-03 | -1.095e-05 | -0.145% |
| `image_1` | `sector_1` | `median_rms_flat_noise` | 1.458e-02 | 1.458e-02 | 9.313e-10 | 0.000% |
| `image_1` | `sector_1` | `dynamic_range_global_flat_noise` | 4.953e+02 | 4.960e+02 | 7.199e-01 | 0.145% |
| `image_1` | `sector_1` | `min_rms_true_sky` | 7.631e-03 | 7.631e-03 | -8.848e-09 | -0.000% |
| `image_1` | `sector_1` | `median_rms_true_sky` | 1.485e-02 | 1.485e-02 | 6.519e-09 | 0.000% |
| `image_1` | `sector_1` | `dynamic_range_global_true_sky` | 4.896e+02 | 4.896e+02 | 6.301e-04 | 0.000% |

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
