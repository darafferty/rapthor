# Post-Master-Sync Option Checks

Status: **accepted with one intentional master-reference divergence**

Both scenarios completed successfully on `master` and the current branch using
the same EveryBeam 0.8.3, WSClean 3.7, and DP3 6.6 tool stack.

| Scenario | Raw comparator | Decision |
| --- | --- | --- |
| `normalization-rich-demo` | pass | No branch divergence after the dependency update. |
| `prediction-path-wsclean` | fail | Expected consequence of fixing incomplete channel coverage in `master`. |

## WSClean Prediction Classification

The WSClean prediction scenario deliberately requests a `75 kHz` maximum
prediction bandwidth for an eight-channel Measurement Set. WSClean interprets
the upper value of `-channel-range` as exclusive.

Legacy `master` divides the data into two arrays and passes their first and
last channel indices as `0 3` and `4 7`. WSClean therefore predicts channels
`0-2` and `4-6`, leaving channels 3 and 7 without model data. Its own log shows
two three-channel prediction groups.

The current branch uses complete end-exclusive ranges:

```text
0 3
3 6
6 8
```

All eight channels are covered exactly once, each band remains within the
configured bandwidth, and every current model column has non-zero values in
all eight channels. Focused unit tests cover one-channel, exact-division, and
uneven-final-band cases; the focused integration test asserts these production
command ranges and successful downstream products.

The raw branch comparator correctly reports downstream FITS, h5parm, catalog,
and sky-model differences because calibration is no longer solving against
two unpredicted channels. Those differences are evidence of the fix, not a
reason to reproduce master's incomplete model. The raw report and failing
matrix status are preserved unchanged for auditability.
Compact channel-range and model-column evidence is retained in
`wsclean-prediction-command-evidence.json`.

## Decision

These checks support retaining the current branch's science decision. They do
not justify relaxing any comparator tolerance: normalization passes directly,
and the WSClean prediction mismatch has an explicit command-level root cause
with regression coverage.
