# Rapthor Branch Option Matrix

Run root: `/tmp/rob-run`

Risk-based branch-vs-master option equivalence checks after the 2026-07-06 core science gate.

| Scenario | Result | Command RC | Pairs | Passed Pairs | Failures | Warnings | Report | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `normalization-rich-demo` | pass | 0 | 1 | 1 | 0 | 1 | `branch-equivalence-report.json` | Flux-scale normalization enabled on the generated rich demo data with provided reference sky models. |
| `prediction-path-image-based` | pass | 0 | 1 | 1 | 0 | 1 | `branch-equivalence-report.json` | DD fast+medium phase calibration with DP3 image-based predict enabled on the generated rich demo data. |
| `prediction-path-wsclean` | pass | 0 | 1 | 1 | 0 | 1 | `branch-equivalence-report.json` | DD fast+medium phase calibration with WSClean predict enabled on the generated rich demo data. |
| `bda-averaging` | pass | 0 | 1 | 1 | 0 | 1 | `branch-equivalence-report.json` | DD fast+medium phase calibration with calibration BDA and imaging averaging/BDA enabled on the generated rich demo data; compact report generated from focused rerun root `/tmp/rob-bda`. |
| `screens` | skipped | None | 0 | 0 | 0 | 0 |  | Record as skipped until the target environment can run screen workflows reproducibly. |
