"""Image diagnostic reporting helpers used by image operation finalizers."""


def report_sector_diagnostics(sector_name, diagnostics_dict, log):
    """
    Report the sector's image diagnostics

    Parameters
    ----------
    sector_name : str
        The name of the sector.
    diagnostics_dict : dict
        The dict containing the diagnostics. A check is mode for the required keys;
        if any is not present, the report is skipped.
    log : logging.Logger object
        The logger to use for the report.

    Returns
    -------
    lofar_to_true_flux_ratio : float
        Mean ratio of the LOFAR flux densities to the "true" ones. The true flux
        densities are assumed to be from one of the TGSS, NVSS, or LoTSS surveys.
        If ratios from multiple surveys are present, the one with the lowest scatter
        is returned
    lofar_to_true_flux_std : float
        Stdev of the ratio of the LOFAR flux densities to the "true" ones
    """
    try:
        log.info("Diagnostics for %s:", sector_name)
        log.info(
            "    Min RMS noise = %.1f uJy/beam (non-PB-corrected), %.1f uJy/beam (PB-corrected), "
            "%f (expected)",
            diagnostics_dict["min_rms_flat_noise"] * 1e6,
            diagnostics_dict["min_rms_true_sky"] * 1e6,
            diagnostics_dict["theoretical_rms"] * 1e6,
        )
        if (
            diagnostics_dict["min_rms_flat_noise"] == 0.0
            or diagnostics_dict["min_rms_true_sky"] == 0.0
        ):
            log.warning("The min RMS noise is 0, likely indicating a problem with the processing.")
        log.info(
            "    Median RMS noise = %.1f uJy/beam (non-PB-corrected), %.1f uJy/beam (PB-corrected)",
            diagnostics_dict["median_rms_flat_noise"] * 1e6,
            diagnostics_dict["median_rms_true_sky"] * 1e6,
        )
        log.info(
            "    Dynamic range = %.2f (non-PB-corrected), %.2f (PB-corrected)",
            diagnostics_dict["dynamic_range_global_flat_noise"],
            diagnostics_dict["dynamic_range_global_true_sky"],
        )
        if (
            diagnostics_dict["dynamic_range_global_flat_noise"] == 0.0
            or diagnostics_dict["dynamic_range_global_true_sky"] == 0.0
        ):
            log.warning("The dynamic range is 0, likely indicating a problem with the processing.")

        log.info("    Number of sources found by PyBDSF = %s", diagnostics_dict["nsources"])
        if diagnostics_dict["nsources"] == 0:
            log.warning(
                "No sources were found by PyBDSF, possibly indicating a problem with the processing."
            )
        log.info("    Reference frequency = %.1f MHz", diagnostics_dict["freq"] / 1e6)
        log.info(
            '    Beam = %.1f" x %.1f", PA = %.1f deg',
            diagnostics_dict["beam_fwhm"][0] * 3600,
            diagnostics_dict["beam_fwhm"][1] * 3600,
            diagnostics_dict["beam_fwhm"][2],
        )
        log.info(
            "    Fraction of unflagged data = %.2f", diagnostics_dict["unflagged_data_fraction"]
        )

        # Log the estimates of the global flux ratio and astrometry offsets.
        # If the required keys are not present, then there were not enough
        # sources for a reliable estimate to be made so report 'N/A' (not
        # available)
        #
        # Note: the reported error is not allowed to fall below 10% for
        # the flux ratio and 0.5" for the astrometry, as these are the
        # realistic minimum uncertainties in these values
        lofar_to_true_flux_ratio = 1.0
        lofar_to_true_flux_std = 0.0
        missing_surveys = []
        for survey in ["TGSS", "LOTSS", "NVSS"]:
            if survey in ["TGSS", "LOTSS"] or (
                survey == "NVSS" and missing_surveys == ["TGSS", "LOTSS"]
            ):
                # Always report TGSS and LoTSS values when available, but only
                # report NVSS values if both the TGSS and LoTSS comparisons failed (the
                # NVSS ones can be highly uncertain due to the large extrapolation needed).
                # We add the warning below for NVSS
                if (
                    f"meanClippedRatio_{survey}" in diagnostics_dict
                    and f"stdClippedRatio_{survey}" in diagnostics_dict
                ):
                    log.info(
                        "    LOFAR/%s flux ratio = %.1f +/- %.1f%s",
                        survey,
                        diagnostics_dict[f"meanClippedRatio_{survey}"],
                        max(0.1, diagnostics_dict[f"stdClippedRatio_{survey}"]),
                        " (warning: may be highly uncertain due to large extrapolation)"
                        if survey == "NVSS"
                        else "",
                    )

                    if (
                        lofar_to_true_flux_std == 0.0
                        or diagnostics_dict[f"stdClippedRatio_{survey}"] < lofar_to_true_flux_std
                    ) and survey != "NVSS":
                        # Save the ratio with the lowest scatter (excluding NVSS
                        # estimate) for later use
                        lofar_to_true_flux_ratio = diagnostics_dict[f"meanClippedRatio_{survey}"]
                        lofar_to_true_flux_std = max(
                            0.1, diagnostics_dict[f"stdClippedRatio_{survey}"]
                        )
                else:
                    missing_surveys.append(survey)
                    log.info("    LOFAR/%s flux ratio = N/A", survey)

        for axis in ("RA", "DEC"):
            if (clipped_mean := f"meanClipped{axis}OffsetDeg") in diagnostics_dict and (
                clipped_std := f"stdClipped{axis}OffsetDeg"
            ) in diagnostics_dict:
                log.info(
                    "    LOFAR-PanSTARRS %s offset = %.1f +/- %.1f",
                    axis,
                    diagnostics_dict[clipped_mean] * 3600,
                    max(0.5, diagnostics_dict[clipped_std] * 3600),
                )
            else:
                log.info("    LOFAR-PanSTARRS %s offset = N/A", axis)

        return (lofar_to_true_flux_ratio, lofar_to_true_flux_std)

    except KeyError:
        log.warning(
            "One or more of the expected image diagnostics is unavailable "
            "for %s. Logging of diagnostics skipped.",
            sector_name,
        )

        req_keys = [
            "theoretical_rms",
            "min_rms_flat_noise",
            "median_rms_flat_noise",
            "dynamic_range_global_flat_noise",
            "min_rms_true_sky",
            "median_rms_true_sky",
            "dynamic_range_global_true_sky",
            "nsources",
            "freq",
            "beam_fwhm",
            "unflagged_data_fraction",
            "meanClippedRatio_TGSS",
            "stdClippedRatio_TGSS",
            "meanClippedRAOffsetDeg",
            "stdClippedRAOffsetDeg",
            "meanClippedDecOffsetDeg",
            "stdClippedDecOffsetDeg",
        ]
        missing_keys = []
        for key in req_keys:
            if key not in diagnostics_dict:
                missing_keys.append(key)
        log.debug("Keys missing from the diagnostics dict: %s.", ", ".join(missing_keys))

        return (1.0, 0.0)
