#!/usr/bin/env python3
"""CLI wrapper for calculating image diagnostics."""

from argparse import ArgumentParser, RawTextHelpFormatter

from rapthor.execution.image.diagnostic_calculation import (
    DEFAULT_PHOTOMETRY_COMPARISON_SURVEYS,
    PHOTOMETRY_BACKUP_SURVEY,
    calculate_image_diagnostics,
)


def main(
    flat_noise_image,
    flat_noise_rms_image,
    true_sky_image,
    true_sky_rms_image,
    input_catalog,
    obs_ms,
    obs_starttime,
    obs_ntimes,
    diagnostics_file,
    output_root,
    facet_region_file=None,
    allow_internet_access=True,
    photometry_comparison_skymodel=None,
    photometry_comparison_surveys=None,
    photometry_backup_survey=PHOTOMETRY_BACKUP_SURVEY,
    astrometry_comparison_skymodel=None,
    min_number=5,
):
    """Calculate image diagnostics."""
    calculate_image_diagnostics(
        flat_noise_image,
        flat_noise_rms_image,
        true_sky_image,
        true_sky_rms_image,
        input_catalog,
        obs_ms,
        obs_starttime,
        obs_ntimes,
        diagnostics_file,
        output_root,
        facet_region_file=facet_region_file,
        allow_internet_access=allow_internet_access,
        photometry_comparison_skymodel=photometry_comparison_skymodel,
        photometry_comparison_surveys=photometry_comparison_surveys,
        photometry_backup_survey=photometry_backup_survey,
        astrometry_comparison_skymodel=astrometry_comparison_skymodel,
        min_number=min_number,
    )


def parse_args():
    """Parse command-line arguments."""
    descriptiontext = "Calculate image photometry and astrometry diagnostics.\n"

    parser = ArgumentParser(description=descriptiontext, formatter_class=RawTextHelpFormatter)
    parser.add_argument("flat_noise_image", help="Filename of flat-noise FITS image")
    parser.add_argument("flat_noise_rms_image", help="Filename of flat-noise RMS FITS image")
    parser.add_argument("true_sky_image", help="Filename of true sky FITS image")
    parser.add_argument("true_sky_rms_image", help="Filename of true sky RMS FITS image")
    parser.add_argument("input_catalog", help="Filename of input PyBDSF FITS catalog")
    parser.add_argument("obs_ms", help="Filename of observation MS")
    parser.add_argument("obs_starttime", help="Start time of observation")
    parser.add_argument("obs_ntimes", help="Number of time slots of observation")
    parser.add_argument("diagnostics_file", help="Filename of diagnostics JSON file")
    parser.add_argument("output_root", help="Root of output files")
    parser.add_argument(
        "--facet_region_file",
        help="Filename of ds9 facet region file",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--allow_internet_access",
        help="Whether to allow internet access for downloading sky models when "
        "they are not available locally.",
        action="store_true",
    )
    parser.add_argument(
        "--photometry_comparison_skymodel",
        help="Filename of photometry sky model",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--photometry_comparison_surveys",
        help="List of photometry surveys to use when photometry_comparison_skymodel is not given",
        type=list,
        default=DEFAULT_PHOTOMETRY_COMPARISON_SURVEYS,
    )
    parser.add_argument(
        "--photometry_backup_survey",
        help="Name of photometry survey to use as backup "
        "if all queries to photometry_comparison_surveys fail",
        type=str,
        default=PHOTOMETRY_BACKUP_SURVEY,
    )
    parser.add_argument(
        "--astrometry_comparison_skymodel",
        help="Filename of astrometry sky model",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--min_number",
        help="Minimum number of sources for diagnostics",
        type=int,
        default=5,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        args.flat_noise_image,
        args.flat_noise_rms_image,
        args.true_sky_image,
        args.true_sky_rms_image,
        args.input_catalog,
        args.obs_ms,
        args.obs_starttime,
        args.obs_ntimes,
        args.diagnostics_file,
        args.output_root,
        facet_region_file=args.facet_region_file,
        allow_internet_access=args.allow_internet_access,
        photometry_comparison_skymodel=args.photometry_comparison_skymodel,
        photometry_comparison_surveys=args.photometry_comparison_surveys,
        photometry_backup_survey=args.photometry_backup_survey,
        astrometry_comparison_skymodel=args.astrometry_comparison_skymodel,
        min_number=args.min_number,
    )
