"""Imaging option constraints shared by parsets, strategies, and execution."""

import math
from typing import Optional

WSCLEAN_AUTO_THRESHOLD = 1.0


def validate_shared_facet_bda(frequencybase: Optional[float], shared_facet_rw: bool) -> None:
    """Reject shared facet I/O with the multi-band layout produced by frequency BDA.

    WSClean's bulk inversion/prediction requires a single spectral window.
    Time-only BDA does not introduce the variable channel layout at issue here.
    """
    if shared_facet_rw and float(frequencybase or 0.0) > 0:
        raise ValueError(
            "WSClean shared facet reads/writes cannot be combined with imaging "
            "frequency BDA (bda_frequencybase > 0). Set [imaging] "
            "shared_facet_rw = False to keep BDA, or bda_frequencybase = 0 "
            "to use shared facet I/O. Time-only BDA is allowed."
        )


def validate_dd_psf_grid(value: object, *, allow_auto: bool = False) -> list[int]:
    """Require positive grid dimensions, allowing Rapthor's startup auto sentinel."""
    if (
        isinstance(value, list)
        and len(value) == 2
        and all(isinstance(size, int) and not isinstance(size, bool) for size in value)
        and (all(size > 0 for size in value) or (allow_auto and value == [0, 0]))
    ):
        return list(value)
    auto_hint = "[0, 0] for automatic sizing or " if allow_auto else ""
    raise ValueError(
        f"dd_psf_grid must be {auto_hint}two positive integers (e.g. [1, 1]); got {value!r}"
    )


def validate_auto_mask(value: float) -> None:
    """Keep automatic masking above the fixed WSClean CLEAN stopping threshold."""
    if not math.isfinite(value) or value <= WSCLEAN_AUTO_THRESHOLD:
        raise ValueError(
            f"auto_mask must be finite and greater than {WSCLEAN_AUTO_THRESHOLD}, "
            f"Rapthor's WSClean auto-threshold; got {value!r}"
        )
