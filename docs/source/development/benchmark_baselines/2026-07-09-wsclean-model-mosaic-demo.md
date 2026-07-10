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
