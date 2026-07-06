# Rapthor Branch Option Matrix

Run root: `/tmp/rop-run3`

Risk-based branch-vs-master option equivalence checks after the 2026-07-06 core science gate.

| Scenario | Result | Command RC | Pairs | Passed Pairs | Failures | Warnings | Report | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `normalization-rich-demo` | pass | 0 | 1 | 1 | 0 | 1 | `branch-equivalence-report.json` | Flux-scale normalization enabled on the generated rich demo data with provided reference sky models. |
| `prediction-path-image-based` | pass | 0 | 1 | 1 | 0 | 1 | `branch-equivalence-report.json` | DD fast+medium phase calibration with DP3 image-based predict enabled on the generated rich demo data. |
| `prediction-path-wsclean` | pass | 0 | 1 | 1 | 0 | 1 | `branch-equivalence-report.json` | DD fast+medium phase calibration with WSClean predict enabled on the generated rich demo data. |
| `bda-averaging` | skipped | None | 0 | 0 | 0 | 0 |  | Use one BDA or averaging-related configuration against the same stable baseline. |
| `screens` | skipped | None | 0 | 0 | 0 | 0 |  | Record as skipped until the target environment can run screen workflows reproducibly. |
