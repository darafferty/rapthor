# Rapthor Branch Equivalence

Scenario: `normalization-rich-demo`
Run root: `/tmp/rop-run3/normalization-rich-demo`

## Branch Runs

| Side | Ref | Return Code | Parset | Work Dir | Log | Input Snapshot |
| --- | --- | ---: | --- | --- | --- | --- |
| base | `master` | 0 | `/app/docs/source/development/equivalence_runs/2026-07-06-option-matrix/inputs/base/normalization_rich_demo.parset` | `/tmp/rom-base` | `/tmp/rop-run3/normalization-rich-demo/base/rapthor-command.log` | parset: `inputs/base/normalization_rich_demo.parset`, strategy: `inputs/base/normalization_rich_demo_strategy.py` |
| current | `current` | 0 | `/app/docs/source/development/equivalence_runs/2026-07-06-option-matrix/inputs/current/normalization_rich_demo.parset` | `/tmp/rom-current` | `/tmp/rop-run3/normalization-rich-demo/current/rapthor-command.log` | parset: `inputs/current/normalization_rich_demo.parset`, strategy: `inputs/current/normalization_rich_demo_strategy.py` |

## Comparison Summary

| Result | Ops | Records | FITS | Image HDUs | Table HDUs | H5 | Text | Diagnostics | Visuals |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pass | 4 | 4 | 8 | 7 | 1 | 3 | 12 | 1 | 5 |

## FITS Residual Metrics

| Product | Max Abs Delta | P99 Abs Delta | Residual RMS | RMS / Ref RMS | RMS / Ref MAD |
| --- | ---: | ---: | ---: | ---: | ---: |
| `field-MFS-image.fits` | 9.537e-07 | 5.588e-08 | 1.956e-08 | 3.227e-07 | 1.314e-06 |
| `sector_1_I_freq_cube.fits` | 9.537e-07 | 1.118e-07 | 4.058e-08 | 4.792e-07 | 1.421e-06 |
| `field-MFS-dirty.fits` | 7.153e-07 | 5.960e-08 | 2.312e-08 | 1.635e-07 | 1.824e-07 |
| `field-MFS-image-pb-ast.fits` | 7.153e-07 | 5.774e-08 | 1.991e-08 | 3.241e-07 | 1.310e-06 |
| `field-MFS-image-pb.fits` | 7.153e-07 | 5.774e-08 | 1.991e-08 | 3.241e-07 | 1.310e-06 |
| `field-MFS-residual.fits` | 1.490e-07 | 3.912e-08 | 1.475e-08 | 9.409e-07 | 9.928e-07 |
| `field-MFS-model-pb.fits` | 1.192e-07 | 0.000e+00 | 1.446e-10 | 9.907e-08 | n/a |

## Image Diagnostics

| Operation | Sector | Field | Reference | Current | Delta | Relative Delta |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `image_1` | `sector_1` | `nsources` | 1.000e+01 | 1.000e+01 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `theoretical_rms` | 9.006e-03 | 9.006e-03 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `min_rms_flat_noise` | 7.534e-03 | 7.535e-03 | 5.914e-07 | 0.008% |
| `image_1` | `sector_1` | `median_rms_flat_noise` | 1.458e-02 | 1.458e-02 | -9.313e-10 | -0.000% |
| `image_1` | `sector_1` | `dynamic_range_global_flat_noise` | 4.959e+02 | 4.959e+02 | -3.893e-02 | -0.008% |
| `image_1` | `sector_1` | `min_rms_true_sky` | 7.632e-03 | 7.632e-03 | -9.779e-09 | -0.000% |
| `image_1` | `sector_1` | `median_rms_true_sky` | 1.485e-02 | 1.485e-02 | 9.313e-10 | 0.000% |
| `image_1` | `sector_1` | `dynamic_range_global_true_sky` | 4.896e+02 | 4.896e+02 | 6.585e-04 | 0.000% |

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
