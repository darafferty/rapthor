# Rapthor Branch Option Matrix

Run root: `/app/runs/science-gate-20260716-post-master-sync-option-matrix`

Risk-based branch-vs-master option equivalence checks after the 2026-07-06 core science gate.

| Scenario | Result | Command RC | Pairs | Passed Pairs | Failures | Warnings | Report | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `normalization-rich-demo` | pass | 0 | 1 | 1 | 0 | 1 | `branch-equivalence-report.json` | Flux-scale normalization enabled on the generated rich demo data with provided reference sky models. |
| `prediction-path-wsclean` | fail | 1 | 1 | 0 | 33 | 1 | `branch-equivalence-report.json` | DD fast+medium phase calibration with WSClean predict split into multiple frequency bands on the generated rich demo data. |
