# Rapthor Branch Equivalence

Scenario: `normalization-rich-demo`
Run root: `/tmp/rob-run/normalization-rich-demo`

## Branch Runs

| Side | Ref | Return Code | Parset | Work Dir | Log | Input Snapshot |
| --- | --- | ---: | --- | --- | --- | --- |
| base | `master` | 0 | `/app/docs/source/development/equivalence_runs/2026-07-06-option-matrix/inputs/base/normalization_rich_demo.parset` | `/tmp/rom-base` | `/tmp/rob-run/normalization-rich-demo/base/rapthor-command.log` | parset: `inputs/base/normalization_rich_demo.parset`, strategy: `inputs/base/normalization_rich_demo_strategy.py` |
| current | `current` | 0 | `/app/docs/source/development/equivalence_runs/2026-07-06-option-matrix/inputs/current/normalization_rich_demo.parset` | `/tmp/rom-current` | `/tmp/rob-run/normalization-rich-demo/current/rapthor-command.log` | parset: `inputs/current/normalization_rich_demo.parset`, strategy: `inputs/current/normalization_rich_demo_strategy.py` |

## Comparison Summary

| Result | Ops | Records | FITS | Image HDUs | Table HDUs | H5 | Text | Diagnostics | Visuals |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pass | 4 | 4 | 8 | 7 | 1 | 3 | 12 | 1 | 5 |

## FITS Residual Metrics

| Product | Max Abs Delta | P99 Abs Delta | Residual RMS | RMS / Ref RMS | RMS / Ref MAD |
| --- | ---: | ---: | ---: | ---: | ---: |
| `sector_1_I_freq_cube.fits` | 9.537e-07 | 1.192e-07 | 4.189e-08 | 4.947e-07 | 1.467e-06 |
| `field-MFS-dirty.fits` | 7.153e-07 | 5.960e-08 | 2.149e-08 | 1.519e-07 | 1.695e-07 |
| `field-MFS-image-pb-ast.fits` | 7.153e-07 | 6.496e-08 | 2.303e-08 | 3.748e-07 | 1.515e-06 |
| `field-MFS-image-pb.fits` | 7.153e-07 | 6.496e-08 | 2.303e-08 | 3.748e-07 | 1.515e-06 |
| `field-MFS-image.fits` | 7.153e-07 | 6.380e-08 | 2.287e-08 | 3.773e-07 | 1.536e-06 |
| `field-MFS-residual.fits` | 3.576e-07 | 4.470e-08 | 1.710e-08 | 1.091e-06 | 1.151e-06 |
| `field-MFS-model-pb.fits` | 2.086e-07 | 0.000e+00 | 1.568e-10 | 1.074e-07 | n/a |

## Image Diagnostics

| Operation | Sector | Field | Reference | Current | Delta | Relative Delta |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `image_1` | `sector_1` | `nsources` | 1.000e+01 | 1.000e+01 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `theoretical_rms` | 9.006e-03 | 9.006e-03 | 0.000e+00 | 0.000% |
| `image_1` | `sector_1` | `min_rms_flat_noise` | 7.544e-03 | 7.534e-03 | -1.005e-05 | -0.133% |
| `image_1` | `sector_1` | `median_rms_flat_noise` | 1.458e-02 | 1.458e-02 | 1.863e-09 | 0.000% |
| `image_1` | `sector_1` | `dynamic_range_global_flat_noise` | 4.953e+02 | 4.959e+02 | 6.603e-01 | 0.133% |
| `image_1` | `sector_1` | `min_rms_true_sky` | 7.631e-03 | 7.631e-03 | 1.029e-07 | 0.001% |
| `image_1` | `sector_1` | `median_rms_true_sky` | 1.485e-02 | 1.485e-02 | 9.313e-10 | 0.000% |
| `image_1` | `sector_1` | `dynamic_range_global_true_sky` | 4.896e+02 | 4.896e+02 | -6.540e-03 | -0.001% |

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
