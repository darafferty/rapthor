# Rapthor Branch Equivalence

Scenario: `benchmark-default-like`
Run root: `/app/runs/rbe-default-like-master-ref-codex`

## Branch Runs

| Side | Ref | Return Code | Parset | Work Dir | Log |
| --- | --- | ---: | --- | --- | --- |
| base | `master` | 0 | `/app/runs/master-benchmark-default-like-manual/inputs/master_benchmark_default_like.parset` | `/tmp/rbe-master-work` | `/app/runs/rbe-default-like-master-ref-codex/base/rapthor-command.log` |
| current | `current` | 0 | `/app/runs/current-benchmark-default-like-manual/inputs/current_benchmark_default_like.parset` | `/tmp/rbe-current-work` | `/app/runs/rbe-default-like-master-ref-codex/current/rapthor-command.log` |

## Comparison Summary

| Result | Ops | Records | FITS | Image HDUs | Table HDUs | H5 | Text |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fail | 12 | 12 | 28 | 20 | 3 | 14 | 37 |

## FITS Residual Metrics

| Product | Max Abs Delta | P99 Abs Delta | Residual RMS | RMS / Ref RMS | RMS / Ref MAD |
| --- | ---: | ---: | ---: | ---: | ---: |
| `field-MFS-model-pb.fits` | 6.671e-01 | 0.000e+00 | 5.928e-04 | 3.133e-01 | n/a |
| `field-MFS-model-pb.fits.fz` | 4.714e-01 | 1.687e-09 | 2.548e-03 | 9.978e-01 | n/a |
| `field-MFS-dirty.fits` | 3.403e-01 | 1.596e-01 | 4.203e-02 | 2.453e-01 | 2.739e-01 |
| `field-MFS-model-pb.fits.fz` | 2.336e-01 | 0.000e+00 | 3.473e-03 | 1.148e+00 | n/a |
| `field-MFS-model-pb.fits.fz` | 1.667e-01 | 0.000e+00 | 2.697e-03 | 9.712e-01 | n/a |
| `field-MFS-dirty.fits.fz` | 1.101e-01 | 2.396e-02 | 8.862e-03 | 5.152e-02 | 5.718e-02 |
| `field-MFS-image.fits` | 1.095e-01 | 3.514e-02 | 1.119e-02 | 1.427e-01 | 3.848e-01 |
| `field-MFS-image-pb.fits` | 9.779e-02 | 3.459e-02 | 1.111e-02 | 1.398e-01 | 3.744e-01 |
| `field-MFS-image-pb.fits.fz` | 8.349e-02 | 2.436e-02 | 7.907e-03 | 9.048e-02 | 1.676e-01 |
| `field-MFS-image.fits.fz` | 8.290e-02 | 2.427e-02 | 7.852e-03 | 9.121e-02 | 1.698e-01 |
| `field-MFS-residual.fits.fz` | 8.289e-02 | 2.383e-02 | 7.808e-03 | 1.663e-01 | 1.694e-01 |
| `field-MFS-residual.fits` | 7.279e-02 | 3.489e-02 | 1.111e-02 | 3.667e-01 | 3.829e-01 |
| `field-MFS-dirty.fits.fz` | 1.462e-05 | 9.928e-06 | 4.153e-06 | 2.404e-05 | 2.682e-05 |
| `field-MFS-dirty.fits.fz` | 1.398e-05 | 9.894e-06 | 4.157e-06 | 2.419e-05 | 2.686e-05 |
| `field-MFS-image-pb.fits.fz` | 4.601e-06 | 3.219e-06 | 1.355e-06 | 1.564e-05 | 2.962e-05 |
| `field-MFS-residual.fits.fz` | 4.523e-06 | 3.137e-06 | 1.320e-06 | 2.898e-05 | 2.954e-05 |
| `field-MFS-image.fits.fz` | 4.485e-06 | 3.159e-06 | 1.328e-06 | 1.556e-05 | 2.964e-05 |
| `field-MFS-image-pb.fits.fz` | 4.284e-06 | 2.851e-06 | 1.194e-06 | 1.417e-05 | 2.862e-05 |
| `field-MFS-image.fits.fz` | 4.217e-06 | 2.794e-06 | 1.169e-06 | 1.409e-05 | 2.861e-05 |
| `field-MFS-residual.fits.fz` | 4.185e-06 | 2.772e-06 | 1.161e-06 | 2.785e-05 | 2.850e-05 |

## Warnings

- output-record summary differs for calibrate_1
- output-record summary differs for calibrate_2
- output-record summary differs for calibrate_3
- output-record summary differs for calibrate_4
- output-record summary differs for image_1
- output-record summary differs for image_2
- output-record summary differs for image_3
- output-record summary differs for image_4

## Failures

- FITS image pixels differ for field-MFS-dirty.fits.fz: max_abs_delta=1.461803913116455e-05, p99_abs_delta=9.927898645401001e-06, residual_rms=4.153114363263751e-06
- missing FITS product: /tmp/rbe-current-work/images/image_1/field-MFS-image-pb-ast.fits.fz
- FITS image pixels differ for field-MFS-image-pb.fits.fz: max_abs_delta=4.284083843231201e-06, p99_abs_delta=2.8507784008979797e-06, residual_rms=1.193581974166202e-06
- FITS image pixels differ for field-MFS-image.fits.fz: max_abs_delta=4.217028617858887e-06, p99_abs_delta=2.7939677238464355e-06, residual_rms=1.1691680257568501e-06
- FITS std differs for field-MFS-model-pb.fits.fz: 0.0030252687923785033 != 0.003041388773045781
- FITS rms differs for field-MFS-model-pb.fits.fz: 0.0030252730528795525 != 0.003041393485271446
- FITS max differs for field-MFS-model-pb.fits.fz: 1.6747643947601318 != 1.6856900453567505
- FITS image pixels differ for field-MFS-model-pb.fits.fz: max_abs_delta=0.2336164265871048, p99_abs_delta=0.0, residual_rms=0.0034727893186978125
- FITS image pixels differ for field-MFS-residual.fits.fz: max_abs_delta=4.184897989034653e-06, p99_abs_delta=2.771615982055664e-06, residual_rms=1.1614027951351234e-06
- FITS image pixels differ for field-MFS-dirty.fits.fz: max_abs_delta=1.3984739780426025e-05, p99_abs_delta=9.894371032714844e-06, residual_rms=4.1571770859844585e-06
- missing FITS product: /tmp/rbe-current-work/images/image_2/field-MFS-image-pb-ast.fits.fz
- FITS image pixels differ for field-MFS-image-pb.fits.fz: max_abs_delta=4.600733518600464e-06, p99_abs_delta=3.2186508178710938e-06, residual_rms=1.3549099459133352e-06
- FITS image pixels differ for field-MFS-image.fits.fz: max_abs_delta=4.4852495193481445e-06, p99_abs_delta=3.159046173095703e-06, residual_rms=1.3281968992140964e-06
- FITS image pixels differ for field-MFS-model-pb.fits.fz: max_abs_delta=0.1667155995965004, p99_abs_delta=0.0, residual_rms=0.0026973665808031986
- FITS image pixels differ for field-MFS-residual.fits.fz: max_abs_delta=4.522502422332764e-06, p99_abs_delta=3.1366944313049316e-06, residual_rms=1.3197429936663649e-06
- FITS mean differs for field-MFS-dirty.fits.fz: -0.0005202056840861393 != -0.0005084568185183787
- FITS std differs for field-MFS-dirty.fits.fz: 0.17202014371259003 != 0.16789766409169604
- FITS rms differs for field-MFS-dirty.fits.fz: 0.17202093028714235 != 0.16789843398848095
- FITS min differs for field-MFS-dirty.fits.fz: -0.7257910370826721 != -0.7099578380584717
- FITS max differs for field-MFS-dirty.fits.fz: 4.512298583984375 != 4.402451515197754
- FITS image pixels differ for field-MFS-dirty.fits.fz: max_abs_delta=0.1101236343383789, p99_abs_delta=0.02396302819252011, residual_rms=0.008861978428047562
- missing FITS product: /tmp/rbe-current-work/images/image_3/field-MFS-image-pb-ast.fits.fz
- FITS mean differs for field-MFS-image-pb.fits.fz: 0.002656418390959526 != 0.0026757617742633757
- FITS std differs for field-MFS-image-pb.fits.fz: 0.08735635573238212 != 0.0890870886494599
- FITS rms differs for field-MFS-image-pb.fits.fz: 0.08739673589734526 != 0.08912726330988363
- FITS min differs for field-MFS-image-pb.fits.fz: -0.22368136048316956 != -0.24955418705940247
- FITS max differs for field-MFS-image-pb.fits.fz: 4.69946813583374 != 4.733382701873779
- FITS image pixels differ for field-MFS-image-pb.fits.fz: max_abs_delta=0.08349386975169182, p99_abs_delta=0.02436171844601631, residual_rms=0.007907300319243629
- FITS mean differs for field-MFS-image.fits.fz: 0.0026164600973858443 != 0.002648755971583895
- FITS std differs for field-MFS-image.fits.fz: 0.0860471372607587 != 0.08814060672854139
- FITS rms differs for field-MFS-image.fits.fz: 0.0860869077979519 != 0.08818039726987163
- FITS min differs for field-MFS-image.fits.fz: -0.22036121785640717 != -0.244085893034935
- FITS max differs for field-MFS-image.fits.fz: 4.699453353881836 != 4.749789237976074
- FITS image pixels differ for field-MFS-image.fits.fz: max_abs_delta=0.08289575949311256, p99_abs_delta=0.02427015902474521, residual_rms=0.007851906420107139
- FITS std differs for field-MFS-model-pb.fits.fz: 0.0025538921377748643 != 0.0028083335087920385
- FITS rms differs for field-MFS-model-pb.fits.fz: 0.0025538958054191125 != 0.002808337322891752
- FITS min differs for field-MFS-model-pb.fits.fz: -0.10289786756038666 != -0.09083002060651779
- FITS max differs for field-MFS-model-pb.fits.fz: 2.853027105331421 != 2.797551393508911
- FITS image pixels differ for field-MFS-model-pb.fits.fz: max_abs_delta=0.4713519960641861, p99_abs_delta=1.6869523356412705e-09, residual_rms=0.0025483459710280393
- FITS mean differs for field-MFS-residual.fits.fz: -0.00028164066042781687 != -0.00024764968995588947
- FITS std differs for field-MFS-residual.fits.fz: 0.046950767685467854 != 0.049461582429077916
- FITS rms differs for field-MFS-residual.fits.fz: 0.046951612408056655 != 0.04946220240504262
- FITS min differs for field-MFS-residual.fits.fz: -0.22035935521125793 != -0.21967588365077972
- FITS max differs for field-MFS-residual.fits.fz: 0.23137694597244263 != 0.24514909088611603
- FITS image pixels differ for field-MFS-residual.fits.fz: max_abs_delta=0.08289386332035065, p99_abs_delta=0.023834750279784198, residual_rms=0.007808209226693329
- FITS mean differs for field-MFS-dirty.fits: -0.0004350740624319749 != -0.0006805365433907418
- FITS std differs for field-MFS-dirty.fits: 0.17133759251209538 != 0.1664614769155411
- FITS rms differs for field-MFS-dirty.fits: 0.1713381448985622 != 0.1664628680123293
- FITS min differs for field-MFS-dirty.fits: -0.7061915397644043 != -0.6950209736824036
- FITS max differs for field-MFS-dirty.fits: 4.510161876678467 != 4.401115417480469
- ... 92 more failure(s)
