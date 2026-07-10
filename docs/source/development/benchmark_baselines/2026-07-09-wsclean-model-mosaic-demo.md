# 2026-07-09 WSClean Model-Mosaic Demo Check

Source artifacts:

- `runs/prefect-demo-multisector-wsclean-model-20260709`

Raw run directories, logs, FITS products, and Dask HTML reports are intentionally
not committed. This note preserves the compact evidence from the first
multi-sector demo run using WSClean to render model mosaics from sector sky
models.

## Run Setup

The run used the generated quadrant-balanced multi-sector demo dataset and a
temporary run-local parset copy with FITS previews enabled:

```bash
scripts/dev/run-rapthor-prefect-demo.py \
  runs/prefect-demo-multisector-wsclean-model-20260709/prefect_demo_multisector_benchmark_previews.parset \
  --run-dir runs/prefect-demo-multisector-wsclean-model-20260709/run \
  --local-dask-workers 2 \
  --cpus-per-task 4 \
  --max-threads 4 \
  --filter-skymodel-ncores 4 \
  --dask-performance-report \
  --no-keep-server \
  --no-keep-server-on-failure
```

Runtime highlights:

- 4 imaging sectors (`2 x 2`)
- FITS previews enabled
- local Dask scheduler with 2 workers and 4 threads per worker
- command profiling enabled

## WSClean Model-Mosaic Evidence

The run reached and completed `mosaic_1`, `mosaic_2`, and `mosaic_3`. Each
cycle rendered `MFS-model-pb` with WSClean from 4 sector sky models.

| Operation | Command | Return Code | Duration (s) |
| --- | --- | ---: | ---: |
| `mosaic_1` | `wsclean -draw-model ... mosaic_1-MFS-model-pb.skymodel` | `0` | `4.750` |
| `mosaic_2` | `wsclean -draw-model ... mosaic_2-MFS-model-pb.skymodel` | `0` | `6.469` |
| `mosaic_3` | `wsclean -draw-model ... mosaic_3-MFS-model-pb.skymodel` | `0` | `15.406` |

Generated products:

| Product | Shape | Finite Pixels | NaN Pixels | Nonzero Pixels | Min | Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `mosaic_1-MFS-model-pb.fits` | `(1, 1, 3966, 3966)` | `15729156` | `0` | `113902` | `-0.639634` | `2.367754` |
| `mosaic_2-MFS-model-pb.fits` | `(1, 1, 3966, 3966)` | `15729156` | `0` | `117212` | `-0.584616` | `2.134284` |
| `mosaic_3-MFS-model-pb.fits` | `(1, 1, 3966, 3966)` | `15729156` | `0` | `117792` | `-0.430247` | `1.985871` |

This is the behavior we wanted from the WSClean path: model products are
rendered from component lists into the mosaic geometry, with finite image data
and sparse nonzero model support rather than interpolation artifacts in empty
background regions.

## Incomplete Full-Demo Result

The full demo did not complete. It failed later, in `image_4`, after the
successful WSClean model-mosaic checks above.

The traceback was a Prefect/Dask runtime failure while worker threads were
hydrating Prefect task settings:

```text
KeyError: 'toml_file:/usr/local/lib/python3.10/dist-packages/prefect/settings/profiles.toml:...'
```

The failure occurred inside Prefect's settings/cachetools path while collecting
image-sector task futures. It is not a WSClean model-mosaic command failure and
not a FITS product-generation error from the completed mosaic cycles.

## Decision

The WSClean model-mosaic implementation has useful positive smoke evidence from
three completed multi-sector mosaic cycles, but the full demo gate remains
open.

Before closing the mosaic queue item or removing the sparse FITS fallback, fix
or isolate the Prefect/Dask threaded settings-cache failure and rerun the full
multi-sector demo, the targeted mosaic science check, and the relevant
benchmark comparison.

## 2026-07-10 Runtime-Fix Rerun

After decoupling Dask task-engine threads from external-command thread counts,
the demo was rerun with:

- `local_dask_workers = 4`
- `cpus_per_task = 4`
- `max_threads = 4`
- `filter_skymodel_ncores = 4`
- FITS previews enabled

This confirmed the intended resource layout: four image-sector tasks were
assigned to four separate Dask worker processes, while each
`filter_skymodel` command still received `--ncores=4`.

The rerun did not reproduce the Prefect settings/cachetools `KeyError`. It
completed `image_1`, `mosaic_1`, `calibrate_2`, `predict_2`, `image_2`, and
started `mosaic_2`. WSClean rendered model mosaics successfully for the two
completed mosaic checks:

| Operation | Command | Return Code | Duration (s) |
| --- | --- | ---: | ---: |
| `mosaic_1` | `wsclean -draw-model ... mosaic_1-MFS-model-pb.skymodel` | `0` | `5.787` |
| `mosaic_2` | `wsclean -draw-model ... mosaic_2-MFS-model-pb.skymodel` | `0` | `3.973` |

The rerun then failed while writing a regular regridded mosaic FITS product:

```text
OSError: Not enough free space to write 125833248 bytes
```

The workspace inside the dev container was full (`/app` had about 161 MB free).
This is a run-environment problem, not the original Prefect settings-cache
runtime failure. The full-demo gate still needs to be rerun after clearing disk
space.

## 2026-07-10 Full Rerun After Disk Cleanup

After clearing old run products and stale `/tmp` artifacts, the full demo was
rerun with the same resource shape:

```bash
scripts/dev/run-rapthor-prefect-demo.py \
  runs/prefect-demo-multisector-wsclean-model-20260710-full-rerun/prefect_demo_multisector_benchmark_previews.parset \
  --run-dir runs/prefect-demo-multisector-wsclean-model-20260710-full-rerun/run \
  --local-dask-workers 4 \
  --cpus-per-task 4 \
  --max-threads 4 \
  --filter-skymodel-ncores 4 \
  --dask-performance-report \
  --no-keep-server \
  --no-keep-server-on-failure
```

Result: the run completed successfully.

Runtime checks:

- 4 imaging sectors (`2 x 2`) were processed through all 4 cycles.
- Four Dask worker processes ran sector WSClean tasks concurrently.
- Each external WSClean/filter command retained the intended 4-thread command
  budget.
- `rapthor.log` contained no `Traceback`, `KeyError`, `cachetools`,
  disk-space, `ERROR`, or `Failed` entries.
- The Dask performance report was written to
  `runs/prefect-demo-multisector-wsclean-model-20260710-full-rerun/run/dask-performance-report.html`.

WSClean model-mosaic commands:

| Operation | Command | Return Code | Duration (s) |
| --- | --- | ---: | ---: |
| `mosaic_1` | `wsclean -draw-model ... mosaic_1-MFS-model-pb.skymodel` | `0` | `9.306` |
| `mosaic_2` | `wsclean -draw-model ... mosaic_2-MFS-model-pb.skymodel` | `0` | `9.944` |
| `mosaic_3` | `wsclean -draw-model ... mosaic_3-MFS-model-pb.skymodel` | `0` | `12.585` |
| `mosaic_4` | `wsclean -draw-model ... mosaic_4-MFS-model-pb.skymodel` | `0` | `9.418` |

Uncompressed model-mosaic product sanity checks:

| Product | Shape | Finite Pixels | Nonzero Pixels | Min | Max | Sum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `mosaic_1-MFS-model-pb.fits` | `(3966, 3966)` | `15729156` | `113902` | `-0.639631` | `2.367754` | `16.321564` |
| `mosaic_2-MFS-model-pb.fits` | `(3966, 3966)` | `15729156` | `117212` | `-0.584616` | `2.134280` | `16.094401` |
| `mosaic_3-MFS-model-pb.fits` | `(3966, 3966)` | `15729156` | `117792` | `-0.430251` | `1.985863` | `15.706178` |
| `mosaic_4-MFS-model-pb.fits` | `(3966, 3966)` | `15729156` | `115867` | `-0.296288` | `1.852920` | `15.643409` |

The final mosaic model previews looked scientifically sensible: compact model
components were rendered in the expected four-sector geometry without the
previous continuous-interpolation artifacts.

One diagnostic follow-up was found while inspecting intermediate sector
`*-MFS-model-pb.fits.fz` products in CARTA: default `fpack` compression of
sparse floating-point model images introduced visible horizontal stripe
artifacts. The uncompressed sector model image was sparse and well behaved
(`9` nonzero pixels in the checked sector), but the default compressed version
had `12600` nonzero pixels and row-level artifacts because floating-point
images are quantized and dithered by default.

A local compression check showed that `fpack -g -q 0` preserves the sparse model
image exactly while still producing a compressed `.fits.fz` product:

| Compression | Nonzero Pixels | Max Abs Diff From Original | Notes |
| --- | ---: | ---: | --- |
| default `fpack` | `12600` | `0.144348` | visible row artifacts |
| `fpack -g -q 0` | `9` | `0.0` | lossless sparse model image |

Sector model-product compression has therefore been changed so
`*-model-pb.fits` uses lossless gzip compression, while regular image,
residual, and dirty products keep the existing default compression. The demo
products in this run were generated before that fix, so rerun the multi-sector
demo if clean sector `*-model-pb.fits.fz` products are needed for manual
inspection.
