# Rapthor Branch Option Matrix

Run root: `/tmp/rom5`

Risk-based branch-vs-master option equivalence checks after the 2026-07-06 core science gate.

| Scenario | Result | Command RC | Pairs | Passed Pairs | Failures | Warnings | Report | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `normalization-rich-demo` | pass | 0 | 1 | 1 | 0 | 1 | `branch-equivalence-report.json` | Flux-scale normalization enabled on the generated rich demo data with provided reference sky models. |
| `prediction-path-image-based` | skipped | None | 0 | 0 | 0 | 0 |  | Add after the normalization option scenario has a compact report. |
| `prediction-path-wsclean` | skipped | None | 0 | 0 | 0 | 0 |  | Pair with the image-based predict scenario so prediction-path differences remain attributable. |
| `bda-averaging` | skipped | None | 0 | 0 | 0 | 0 |  | Use one BDA or averaging-related configuration against the same stable baseline. |
| `screens` | skipped | None | 0 | 0 | 0 | 0 |  | Record as skipped until the target environment can run screen workflows reproducibly. |
